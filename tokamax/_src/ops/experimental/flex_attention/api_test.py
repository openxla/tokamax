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


class FlexAttentionApiTest(parameterized.TestCase):

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
    precision = (
        jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
        if dtype == jnp.float32
        else None
    )
    inputs = _make_inputs(dtype=dtype, is_mha=is_mha)

    f_ref = functools.partial(
        attention_api.dot_product_attention,
        logits_soft_cap=soft_cap,
        is_causal=is_causal,
        scale=scale,
        precision=precision,
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
    fn_ref, args_ref = benchmarking.standardize_function(
        f_ref, kwargs=inputs, mode='forward_and_vjp'
    )
    fn_flex, args_flex = benchmarking.standardize_function(
        f_flex, kwargs=inputs, mode='forward_and_vjp'
    )
    out_ref, (grads_ref,) = jax.jit(fn_ref)(args_ref)
    out_flex, (grads_flex,) = jax.jit(fn_flex)(args_flex)
    fwd_tol = 1e-4 if dtype == jnp.float32 else 2e-2
    grad_atol = 1e-4 if dtype == jnp.float32 else (6.0 if scale else 1e-1)
    grad_rtol = 1e-4 if dtype == jnp.float32 else 1e-1
    chex.assert_trees_all_close(out_flex, out_ref, atol=fwd_tol, rtol=fwd_tol)
    chex.assert_trees_all_close(
        grads_flex, grads_ref, atol=grad_atol, rtol=grad_rtol
    )

  @parameterized.product(
      local_window_size=((32, 16), 32),
      dtype=(jnp.float32, jnp.bfloat16),
      implementation=_IMPLEMENTATIONS,
  )
  def test_against_dot_product_attention_local_window_size(
      self, local_window_size, dtype, implementation
  ):
    precision = (
        jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
        if dtype == jnp.float32
        else None
    )
    inputs = _make_inputs(dtype=dtype)

    f_ref = functools.partial(
        attention_api.dot_product_attention,
        local_window_size=local_window_size,
        precision=precision,
        implementation='xla',
    )
    f_flex = functools.partial(
        api.flex_attention,
        mask_mod=example_masks.generate_local_window_mask(local_window_size),
        precision=precision,
        implementation=implementation,
    )
    fn_ref, args_ref = benchmarking.standardize_function(
        f_ref, kwargs=inputs, mode='forward_and_vjp'
    )
    fn_flex, args_flex = benchmarking.standardize_function(
        f_flex, kwargs=inputs, mode='forward_and_vjp'
    )
    out_ref, (grads_ref,) = jax.jit(fn_ref)(args_ref)
    out_flex, (grads_flex,) = jax.jit(fn_flex)(args_flex)
    fwd_tol = 1e-4 if dtype == jnp.float32 else 2e-2
    grad_tol = 1e-4 if dtype == jnp.float32 else 5e-2
    chex.assert_trees_all_close(out_flex, out_ref, atol=fwd_tol, rtol=fwd_tol)
    chex.assert_trees_all_close(
        grads_flex, grads_ref, atol=grad_tol, rtol=grad_tol
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
    precision = (
        jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
        if dtype == jnp.float32
        else None
    )
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
          precision=precision,
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

    fn_ref, args_ref = benchmarking.standardize_function(
        f_ref, kwargs=inputs, mode='forward_and_vjp'
    )
    fn_flex, args_flex = benchmarking.standardize_function(
        f_flex, kwargs=inputs, mode='forward_and_vjp'
    )
    out_ref, (grads_ref,) = jax.jit(fn_ref)(args_ref)
    out_flex, (grads_flex,) = jax.jit(fn_flex)(args_flex)
    tol = 1e-4 if dtype == jnp.float32 else 3e-2
    chex.assert_trees_all_close(out_flex, out_ref, atol=tol, rtol=tol)
    chex.assert_trees_all_close(grads_flex, grads_ref, atol=tol, rtol=tol)

  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      implementation=_IMPLEMENTATIONS,
  )
  def test_against_dot_product_attention_alphafold(self, dtype, implementation):
    n = 128
    num_heads = 4
    head_dim = 64
    precision = (
        jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
        if dtype == jnp.float32
        else None
    )
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
        precision=precision,
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

    fn_ref, args_ref = benchmarking.standardize_function(
        f_ref, kwargs=inputs, mode='forward_and_vjp'
    )
    fn_flex, args_flex = benchmarking.standardize_function(
        f_flex, kwargs=inputs, mode='forward_and_vjp'
    )
    out_ref, (grads_ref,) = jax.jit(fn_ref)(args_ref)
    out_flex, (grads_flex,) = jax.jit(fn_flex)(args_flex)
    fwd_tol = 1e-4 if dtype == jnp.float32 else 2e-2
    grad_atol = 1e-4 if dtype == jnp.float32 else 2.0
    grad_rtol = 1e-4 if dtype == jnp.float32 else 1e-1
    chex.assert_trees_all_close(out_flex, out_ref, atol=fwd_tol, rtol=fwd_tol)
    chex.assert_trees_all_close(
        grads_flex[:4], grads_ref[:4], atol=grad_atol, rtol=grad_rtol
    )


if __name__ == '__main__':
  absltest.main()
