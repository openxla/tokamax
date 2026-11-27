# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for FlexAttention API."""

import functools
import typing

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import benchmarking
from tokamax._src import shape as shape_lib
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.experimental.flex_attention import api
from tokamax._src.ops.experimental.flex_attention import base
from tokamax._src.ops.experimental.flex_attention import example_masks
from tokamax._src.ops.experimental.flex_attention import example_score_mods

_IMPLEMENTATIONS = typing.get_args(api.Implementation.__value__)


def _make_inputs(
    dtype: jax.typing.DTypeLike,
    batch_size: int = 2,
    seq_len: int = 256,
    num_heads: int = 4,
    head_dim: int = 128,
    is_mha: bool = False,
) -> dict[str, jax.ShapeDtypeStruct]:
  kv_heads = 1 if is_mha else num_heads
  q_shape = jax.ShapeDtypeStruct(
      (batch_size, seq_len, num_heads, head_dim), dtype
  )
  kv_shape = jax.ShapeDtypeStruct(
      (batch_size, seq_len, kv_heads, head_dim), dtype
  )
  return dict(query=q_shape, key=kv_shape, value=kv_shape)


def _round_f32_to_bf16(x: jax.Array) -> jax.Array:
  if x.dtype == jnp.float32:
    return x.astype(jnp.bfloat16).astype(jnp.float32)
  return x


def _get_precision(
    dtype: jax.typing.DTypeLike, implementation: api.Implementation
) -> jax.lax.DotAlgorithmPreset | None:
  if dtype != jnp.float32:
    return None
  if implementation == 'mosaic':
    return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
  return jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6


class FlexAttentionApiTest(parameterized.TestCase):

  def _run_test(
      self,
      fn_ref,
      fn_flex,
      inputs,
      *,
      implementation,
      precision,
      fwd_tol,
      grad_atol,
      grad_rtol,
  ):
    if implementation == 'mosaic':
      mosaic_impl = api.IMPLEMENTATIONS.get(f'mosaic_{jax.default_backend()}')
      if not isinstance(
          mosaic_impl, base.FlexAttention
      ) or not mosaic_impl.supported_on(jax.devices()[0]):
        self.skipTest('Mosaic is not supported on this platform.')
      mode = 'forward'
    else:
      mode = 'forward_and_vjp'

    fn_ref, args_ref = benchmarking.standardize_function(
        fn_ref, kwargs=inputs, mode=mode
    )
    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      # Bottleneck Q/K/V precision of reference inputs to match the
      # implementation.
      args_ref = (*map(_round_f32_to_bf16, args_ref[:3]), *args_ref[3:])

    fn_flex, args_flex = benchmarking.standardize_function(
        fn_flex, kwargs=inputs, mode=mode
    )

    if mode == 'forward':
      out_ref = jax.jit(fn_ref)(args_ref)
      out_flex = jax.jit(fn_flex)(args_flex)
      chex.assert_trees_all_close(out_flex, out_ref, atol=fwd_tol, rtol=fwd_tol)
    else:
      out_ref, (grads_ref,) = jax.jit(fn_ref)(args_ref)
      out_flex, (grads_flex,) = jax.jit(fn_flex)(args_flex)
      chex.assert_trees_all_close(out_flex, out_ref, atol=fwd_tol, rtol=fwd_tol)
      grads_ref = [g for g in grads_ref if g.dtype != jax.dtypes.float0]
      grads_flex = [g for g in grads_flex if g.dtype != jax.dtypes.float0]
      chex.assert_trees_all_close(
          grads_flex, grads_ref, atol=grad_atol, rtol=grad_rtol
      )

  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      is_causal=(True, False),
      scale=(4.3, None),
      is_mha=(True, False),
      implementation=_IMPLEMENTATIONS,
  )
  def test_against_dot_product_attention(
      self, dtype, is_causal, scale, is_mha, implementation
  ):
    soft_cap = 32.0
    precision = _get_precision(dtype, implementation)
    high_prec = precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
    inputs = _make_inputs(dtype=dtype, is_mha=is_mha)

    f_ref = functools.partial(
        attention_api.dot_product_attention,
        logits_soft_cap=soft_cap,
        is_causal=is_causal,
        scale=scale,
        precision=jax.lax.Precision.HIGHEST,
        implementation='xla',
    )
    mask_mode = example_masks.causal_mask if is_causal else None
    f_flex = functools.partial(
        api.flex_attention,
        score_mod=example_score_mods.generate_tanh_softcap(soft_cap),
        mask_mod=mask_mode,
        precision=precision,
        scale=scale,
        implementation=implementation,
    )
    if dtype == jnp.float32:
      grad_atol = 1e-3
      grad_rtol = 1e-3
    else:
      grad_atol = 6.0 if scale else 1e-1
      grad_rtol = 1e-1
    self._run_test(
        f_ref,
        f_flex,
        inputs,
        implementation=implementation,
        precision=precision,
        fwd_tol=3e-4 if high_prec else 2e-2,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
    )

  @parameterized.product(
      local_window_size=((32, 16), 32),
      dtype=(jnp.float32, jnp.bfloat16),
      implementation=_IMPLEMENTATIONS,
  )
  def test_against_dot_product_attention_local_window_size(
      self, local_window_size, dtype, implementation
  ):
    precision = _get_precision(dtype, implementation)
    high_prec = precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
    inputs = _make_inputs(dtype=dtype)

    f_ref = functools.partial(
        attention_api.dot_product_attention,
        local_window_size=local_window_size,
        precision=jax.lax.Precision.HIGHEST,
        implementation='xla',
    )
    f_flex = functools.partial(
        api.flex_attention,
        mask_mod=example_masks.generate_local_window_mask(local_window_size),
        precision=precision,
        implementation=implementation,
    )
    grad_tol = 1e-4 if dtype == jnp.float32 else 5e-2
    self._run_test(
        f_ref,
        f_flex,
        inputs,
        implementation=implementation,
        precision=precision,
        fwd_tol=1e-4 if high_prec else 2e-2,
        grad_atol=grad_tol,
        grad_rtol=grad_tol,
    )

  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      implementation=_IMPLEMENTATIONS,
  )
  def test_against_dot_product_attention_alphagenome(
      self, dtype, implementation
  ):
    batch_size = 1
    seq_len = 256
    upcast_rate = 16
    small_seq = seq_len // upcast_rate
    logits_soft_cap = 5.0
    precision = _get_precision(dtype, implementation)
    high_prec = precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
    inputs = dict(
        q=jax.ShapeDtypeStruct((batch_size, seq_len, 8, 128), dtype),
        k=jax.ShapeDtypeStruct((batch_size, seq_len, 1, 128), dtype),
        v=jax.ShapeDtypeStruct((batch_size, seq_len, 1, 192), dtype),
        pair_bias=jax.ShapeDtypeStruct(
            (batch_size, 8, small_seq, small_seq), dtype
        ),
    )
    soft_cap_fn = example_score_mods.generate_tanh_softcap(logits_soft_cap)

    def upscale_bias(bias):
      with shape_lib.upcast_broadcast():
        return jnp.repeat(
            jnp.repeat(bias, upcast_rate, axis=2), upcast_rate, axis=3
        )

    def f_ref(q, k, v, pair_bias):
      return attention_api.dot_product_attention(
          q,
          k,
          v,
          bias=upscale_bias(pair_bias),
          logits_soft_cap=logits_soft_cap,
          precision=jax.lax.Precision.HIGHEST,
          implementation='xla',
      )

    def f_flex(q, k, v, pair_bias):
      return api.flex_attention(
          q,
          k,
          v,
          score_mod=lambda score: soft_cap_fn(score + upscale_bias(pair_bias)),
          precision=precision,
          implementation=implementation,
      )

    grad_tol = 1e-4 if dtype == jnp.float32 else 3e-2
    self._run_test(
        f_ref,
        f_flex,
        inputs,
        implementation=implementation,
        precision=precision,
        fwd_tol=1e-4 if high_prec else 3e-2,
        grad_atol=grad_tol,
        grad_rtol=grad_tol,
    )

  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      implementation=_IMPLEMENTATIONS,
  )
  def test_against_dot_product_attention_alphafold(self, dtype, implementation):
    n = 128
    num_heads = 4
    head_dim = 64
    precision = _get_precision(dtype, implementation)
    high_prec = precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
    inputs = dict(
        query=jax.ShapeDtypeStruct((n, n, num_heads, head_dim), dtype),
        key=jax.ShapeDtypeStruct((n, n, num_heads, head_dim), dtype),
        value=jax.ShapeDtypeStruct((n, n, num_heads, head_dim), dtype),
        bias=jax.ShapeDtypeStruct((1, num_heads, n, n), dtype),
        mask=jax.ShapeDtypeStruct((n, 1, 1, n), jnp.bool_),
        scale=1.25,
    )

    f_ref = functools.partial(
        attention_api.dot_product_attention,
        precision=jax.lax.Precision.HIGHEST,
        implementation='xla',
    )

    def f_flex(query, key, value, bias, mask, *, scale):
      return api.flex_attention(
          query,
          key,
          value,
          scale=scale,
          score_mod=lambda score: score + bias,
          mask_mod=lambda _: mask,
          precision=precision,
          implementation=implementation,
      )

    self._run_test(
        f_ref,
        f_flex,
        inputs,
        implementation=implementation,
        precision=precision,
        fwd_tol=1e-4 if high_prec else 2e-2,
        grad_atol=1e-3 if dtype == jnp.float32 else 2.0,
        grad_rtol=1e-3 if dtype == jnp.float32 else 1e-1,
    )

  def test_fallback_on_unsupported_reshape(self):
    if jax.default_backend() == 'tpu' and 'mosaic' in _IMPLEMENTATIONS:
      self.skipTest('Not supported on TPU.')

    inputs = _make_inputs(dtype=jnp.bfloat16, batch_size=1, seq_len=128)
    _, args = benchmarking.standardize_function(
        lambda query, key, value: (query, key, value),
        kwargs=inputs,
        mode='forward',
    )
    query, key, value = args

    def score_mod(scores):
      k_len = scores.shape[-1]
      k_idx = (
          jnp.arange(4)[None, None, None, :, None] * (k_len // 4)
          + jnp.arange(k_len // 4)[None, None, None, None, :]
      )
      return scores + k_idx.reshape(1, 1, 1, k_len).astype(scores.dtype)

    actual = jax.jit(
        functools.partial(
            api.flex_attention, score_mod=score_mod, implementation=None
        )
    )(query, key, value)
    expected = jax.jit(
        functools.partial(
            api.flex_attention, score_mod=score_mod, implementation='xla'
        )
    )(query, key, value)
    chex.assert_trees_all_close(actual, expected)


if __name__ == '__main__':
  absltest.main()
