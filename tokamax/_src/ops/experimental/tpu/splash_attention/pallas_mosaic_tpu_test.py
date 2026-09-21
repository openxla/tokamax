# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Tests for Pallas TPU Splash Attention operator wrapper."""

import functools
import typing
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.splash_attention import pallas_mosaic_tpu_vjp as sp_vjp
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

PallasTpuSplash = pallas_mosaic_tpu.PallasMosaicTpuSplashAttention
ReferenceSplash = base.SplashAttention


class PallasMosaicTpuSplashAttentionTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != 'tpu':
      self.skipTest('Only tested on TPU.')
    super().setUp()

  def _test_attention(
      self,
      q,
      k,
      v,
      do,
      mask=base.CAUSAL_MASK,
      is_mqa=False,
      attn_logits_soft_cap=None,
      segment_ids=None,
      sinks=None,
      config=None,
      vjp_config=None,
  ):
    kwargs = dict(
        mask=mask,
        segment_ids=segment_ids,
        sinks=sinks,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    vjp = sp_vjp.PallasMosaicTpuSplashAttentionVjp(config=vjp_config)
    attention_impl = PallasTpuSplash(config=config, vjp=vjp)
    reference_impl = ReferenceSplash()

    @jax.jit
    def f_base(query, key, value, dout):
      primals, f_vjp = jax.vjp(
          functools.partial(reference_impl, **kwargs),
          query,
          key,
          value,
      )
      return primals, f_vjp(dout)

    @jax.jit
    def f(query, key, value, dout):
      primals, f_vjp = jax.vjp(
          functools.partial(attention_impl, **kwargs),
          query,
          key,
          value,
      )
      return primals, f_vjp(dout)

    out_base, (dq_base, dk_base, dv_base) = f_base(q, k, v, do)
    out, (dq, dk, dv) = f(q, k, v, do)

    atol = 0.05 if attn_logits_soft_cap else 0.1
    with self.subTest('output'):
      chex.assert_trees_all_close(out, out_base, atol=atol)

    atol = 0.35 if attn_logits_soft_cap else 1.5
    with self.subTest('dq'):
      chex.assert_trees_all_close(dq, dq_base, atol=atol)

    atol = 0.3 if attn_logits_soft_cap else 1.2
    with self.subTest('dk'):
      chex.assert_trees_all_close(dk, dk_base, atol=atol)

    atol = 0.05 if attn_logits_soft_cap else 0.15
    with self.subTest('dv'):
      chex.assert_trees_all_close(dv, dv_base, atol=atol)

  @parameterized.parameters(
      (4, 4, 256, 256, 64, False),
      (4, 2, 256, 256, 64, False),
      (8, 2, 128, 128, 64, False),
      (4, 1, 256, 256, 64, False),
      (4, 1, 256, 256, 64, True),
      (4, 1, 256, 256, 64, True, None, False),
      (2, 2, 128, 128, 64, False, 30.0),
      (4, 2, 128, 128, 64, False, 30.0),
      (4, 2, 128, 256, 64, False),
      (4, 2, 256, 128, 64, False),
  )
  def test_pallas_mosaic_tpu_vs_reference(
      self,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      kv_seq_len,
      head_dim,
      is_mqa,
      attn_logits_soft_cap=None,
      mqa_2d=True,
  ):
    q = jax.ShapeDtypeStruct(
        (num_q_heads, q_seq_len, head_dim), dtype=jnp.float32
    )

    if is_mqa and mqa_2d:
      k = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype=jnp.float32)
      v = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype=jnp.float32)
    else:
      k = jax.ShapeDtypeStruct(
          (num_kv_heads, kv_seq_len, head_dim), dtype=jnp.float32
      )
      v = jax.ShapeDtypeStruct(
          (num_kv_heads, kv_seq_len, head_dim), dtype=jnp.float32
      )

    mask = base.CAUSAL_MASK

    do = jax.ShapeDtypeStruct(
        (num_q_heads, q_seq_len, head_dim), dtype=jnp.float32
    )
    q, k, v, do = numerics.random_initialize((q, k, v, do))

    self._test_attention(
        q,
        k,
        v,
        do,
        mask=mask,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

  @parameterized.parameters(1, 2, 4)
  def test_mask(self, num_heads):
    head_dim = 32
    q_seq_len = 128
    kv_seq_len = 128

    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)

    mask = mask_lib.CausalMask(shape=(q_seq_len, kv_seq_len))

    q, k, v, do = numerics.random_initialize((q, k, v, do))

    self._test_attention(q, k, v, do, mask=mask)

  def test_custom_config(self):
    q_seq_len, kv_seq_len, head_dim = 128, 128, 64
    num_heads = 2
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)

    mask = base.CAUSAL_MASK

    q, k, v, do = numerics.random_initialize((q, k, v, do))

    config = pallas_mosaic_tpu.Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        k_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        v_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
        qk_diag_skip=True,
        qk_diag_grid=2,
        sv_diag_skip=True,
    )
    vjp_config = sp_vjp.Config(
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        use_base2_exp=True,
    )
    self._test_attention(
        q, k, v, do, mask=mask, config=config, vjp_config=vjp_config
    )

  def test_autotune_configs(self):
    head_dim = 64
    q_seq_len = 256
    kv_seq_len = 256
    num_q_heads = 4
    num_kv_heads = 4
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    q, k, v, do = numerics.random_initialize((q, k, v, do))
    mask = base.CAUSAL_MASK

    attention_fn = PallasTpuSplash()
    bound_args = attention_fn.bind(q, k, v, mask=mask)
    configs = attention_fn._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)

    for config in configs:
      with self.subTest(f'{config=}'):
        self._test_attention(
            q,
            k,
            v,
            do,
            mask=mask,
            config=config,
        )

  def test_autotune_vjp(self):
    head_dim = 64
    q_seq_len = 256
    kv_seq_len = 256
    num_q_heads = 4
    num_kv_heads = 4
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    q, k, v, do = numerics.random_initialize((q, k, v, do))
    mask = base.CAUSAL_MASK

    residuals = (
        jnp.zeros((num_q_heads, q_seq_len), dtype=jnp.float32),
        jnp.zeros((num_q_heads, q_seq_len), dtype=jnp.float32),
    )
    out = jnp.zeros((num_q_heads, q_seq_len, head_dim), dtype=jnp.float32)

    attention_fn = PallasTpuSplash()
    vjp_op = typing.cast(
        sp_vjp.PallasMosaicTpuSplashAttentionVjp, attention_fn.vjp
    )
    bound_args = vjp_op.bind(
        residuals,
        out,
        do,
        q,
        k,
        v,
        mask=mask,
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
            mask=mask,
            vjp_config=config,
        )


if __name__ == '__main__':
  absltest.main()
