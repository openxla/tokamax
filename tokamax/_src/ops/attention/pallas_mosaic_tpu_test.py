# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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


import functools
import typing
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.attention import base as fa_base
from tokamax._src.ops.attention import pallas_mosaic_tpu as fa
from tokamax._src.ops.attention import pallas_mosaic_tpu_vjp as fa_vjp


class PallasMosaicTpuFlashAttentionTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != 'tpu':
      self.skipTest('Only supported on TPUs.')
    super().setUp()

  def _test_attention(
      self, q, k, v, do, mask, is_causal, logits_soft_cap, config=None, vjp_config=None
  ):
    kwargs = dict(
        mask=mask,
        is_causal=is_causal,
        logits_soft_cap=logits_soft_cap,
        logits_scale=0.1,
    )
    vjp = fa_vjp.PallasMosaicTpuFlashAttentionVjp(config=vjp_config)
    attention_impl = fa.PallasMosaicTpuFlashAttention(config=config, vjp=vjp)

    @jax.jit
    def f_base(query, key, value, do):
      primals, f_vjp = jax.vjp(
          functools.partial(fa_base.DotProductAttention(), **kwargs),
          query,
          key,
          value,
      )
      return primals, f_vjp(do)

    @jax.jit
    def f(query, key, value, do):
      primals, f_vjp = jax.vjp(
          functools.partial(attention_impl, **kwargs),
          query,
          key,
          value,
      )
      return primals, f_vjp(do)

    out_base, (dq_base, dk_base, dv_base) = f_base(q, k, v, do)
    out, (dq, dk, dv) = f(q, k, v, do)

    atol = 0.035 if logits_soft_cap else 0.15
    with self.subTest('output'):
      chex.assert_trees_all_close(out, out_base, atol=atol)

    atol = 0.15 if logits_soft_cap else 1.5
    with self.subTest('dq'):
      chex.assert_trees_all_close(dq, dq_base, atol=atol)

    atol = 0.13 if logits_soft_cap else 1.2
    with self.subTest('dk'):
      chex.assert_trees_all_close(dk, dk_base, atol=atol)

    atol = 0.03 if logits_soft_cap else 0.15
    with self.subTest('dv'):
      chex.assert_trees_all_close(dv, dv_base, atol=atol)

  @parameterized.product(
      dtype=[jnp.float32],
      num_kv_heads=[1, 2, 4],
      masking=[None, 'causal', 'bool'],
      logits_soft_cap=[None, 3.4],
  )
  def test_simple(self, dtype, num_kv_heads, masking, logits_soft_cap):
    head_dim = 32
    q_seq_len = 128
    kv_seq_len = 128
    num_q_heads = 4
    batch_size = 2

    q = jax.ShapeDtypeStruct(
        (batch_size, q_seq_len, num_q_heads, head_dim), dtype
    )
    k = jax.ShapeDtypeStruct(
        (batch_size, kv_seq_len, num_kv_heads, head_dim), dtype
    )
    v = jax.ShapeDtypeStruct(
        (batch_size, kv_seq_len, num_kv_heads, head_dim), dtype
    )

    is_causal = masking == 'causal'
    mask = (
        jax.ShapeDtypeStruct((1, 1, q_seq_len, kv_seq_len), dtype=jnp.bool_)
        if masking == 'bool'
        else None
    )
    do = jax.ShapeDtypeStruct(
        (batch_size, q_seq_len, num_q_heads, head_dim), dtype=jnp.float32
    )
    q, k, v, mask, do = numerics.random_initialize((q, k, v, mask, do))
    self._test_attention(q, k, v, do, mask, is_causal, logits_soft_cap)

  def test_autotune_configs(self):
    head_dim = 32
    q_seq_len = 128
    kv_seq_len = 128
    num_q_heads = 4
    num_kv_heads = 4
    batch_size = 2
    dtype = jnp.float32

    q_shape = (batch_size, q_seq_len, num_q_heads, head_dim)
    k_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
    v_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)

    q = jax.ShapeDtypeStruct(q_shape, dtype)
    k = jax.ShapeDtypeStruct(k_shape, dtype)
    v = jax.ShapeDtypeStruct(v_shape, dtype)
    do = jax.ShapeDtypeStruct(q_shape, dtype=jnp.float32)
    q, k, v, do = numerics.random_initialize((q, k, v, do))

    attention_fn = fa.PallasMosaicTpuFlashAttention()
    bound_args = attention_fn.bind(q, k, v)
    configs = attention_fn._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)

    for config in configs:
      with self.subTest(f'{config=}'):
        self._test_attention(
            q,
            k,
            v,
            do,
            mask=None,
            is_causal=False,
            logits_soft_cap=None,
            config=config,
        )

  def test_autotune_vjp(self):
    head_dim = 128
    q_seq_len = 512
    kv_seq_len = 512
    num_q_heads = 4
    num_kv_heads = 4
    batch_size = 2
    dtype = jnp.float32

    q_shape = (batch_size, q_seq_len, num_q_heads, head_dim)
    k_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
    v_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)

    q = jax.ShapeDtypeStruct(q_shape, dtype)
    k = jax.ShapeDtypeStruct(k_shape, dtype)
    v = jax.ShapeDtypeStruct(v_shape, dtype)
    do = jax.ShapeDtypeStruct(q_shape, dtype=jnp.float32)
    q, k, v, do = numerics.random_initialize((q, k, v, do))

    residuals = (
        jnp.zeros((batch_size, num_q_heads, q_seq_len), dtype=jnp.float32),
        jnp.zeros((batch_size, num_q_heads, q_seq_len), dtype=jnp.float32),
    )
    out = jnp.zeros(q_shape, dtype=jnp.float32)

    attention_fn = fa.PallasMosaicTpuFlashAttention()
    vjp_op = typing.cast(
        fa_vjp.PallasMosaicTpuFlashAttentionVjp, attention_fn.vjp
    )
    bound_args = vjp_op.bind(
        residuals,
        out,
        do,
        q,
        k,
        v,
        bias=None,
        logits_scale=0.1,
        mask=fa_base.Mask(),
        precision=(
            jax.lax.DotAlgorithmPreset.DEFAULT,
            jax.lax.DotAlgorithmPreset.DEFAULT,
        ),
        logits_dtype=jnp.float32,
        logits_soft_cap=None,
        dropout_mask=None,
        dropout_rate=0.0,
        paging_info=None,
        q_indices=None,
        k_indices=None,
        normalize_output=True,
        return_residuals=False,
    )
    configs = vjp_op._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)

    for config in configs:
      with self.subTest(f'{config=}'):
        self._test_attention(
            q,
            k,
            v,
            do,
            mask=None,
            is_causal=False,
            logits_soft_cap=None,
            vjp_config=config,
        )


if __name__ == '__main__':
  absltest.main()
