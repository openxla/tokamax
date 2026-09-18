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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

PallasTpuSplash = pallas_mosaic_tpu.PallasMosaicTpuSplashAttention
ReferenceSplash = base.SplashAttention


class PallasMosaicTpuSplashAttentionTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only tested on TPU.")
    super().setUp()

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
    atol = 0.1

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

    q, k, v = numerics.random_initialize((q, k, v))

    op_pallas = PallasTpuSplash()
    op_base = ReferenceSplash()

    out_pallas = op_pallas(
        q,
        k,
        v,
        mask,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    out_base = op_base(
        q,
        k,
        v,
        mask,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    chex.assert_trees_all_close(out_pallas, out_base, atol=atol)

  @parameterized.parameters(1, 2, 4)
  def test_mask(self, num_heads):
    head_dim = 32
    q_seq_len = 128
    kv_seq_len = 128
    atol = 0.04

    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)

    mask = mask_lib.CausalMask(shape=(q_seq_len, kv_seq_len))

    q, k, v = numerics.random_initialize((q, k, v))

    op_pallas = PallasTpuSplash()
    op_base = ReferenceSplash()
    out_pallas = op_pallas(q, k, v, mask)
    out_base = op_base(q, k, v, mask)

    chex.assert_trees_all_close(out_pallas, out_base, atol=atol)

  def test_custom_config(self):
    q_seq_len, kv_seq_len, head_dim = 128, 128, 64
    num_heads = 2
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)

    mask = base.CAUSAL_MASK

    q, k, v = numerics.random_initialize((q, k, v))

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
    op = PallasTpuSplash(config=config)
    out = op(q, k, v, mask)
    self.assertEqual(out.shape, (num_heads, q_seq_len, head_dim))


if __name__ == "__main__":
  absltest.main()
