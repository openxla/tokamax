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


from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.attention import base as fa_base
from tokamax._src.ops.attention import pallas_mosaic_tpu as fa


class PallasMosaicTpuFlashAttentionTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[jnp.float32],
      is_mqa=[False, True],
      masking=[None, 'causal', 'bool'],
      logits_soft_cap=[None, 3.4],
  )
  def test_simple(self, dtype, is_mqa, masking, logits_soft_cap):

    head_dim = 32
    q_seq_len = 128
    kv_seq_len = 128
    num_q_heads = 4
    num_kv_heads = 1 if is_mqa else num_q_heads
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
        jax.ShapeDtypeStruct(
            (1, 1, q_seq_len, kv_seq_len), dtype=jnp.bool_
        )
        if masking == 'bool'
        else None
    )
    q, k, v, mask = numerics.random_initialize((q, k, v, mask))

    @jax.jit
    def f_base(query, key, value):
      return fa_base.DotProductAttention()(
          q=query,
          k=key,
          v=value,
          mask=mask,
          is_causal=is_causal,
          logits_soft_cap=logits_soft_cap,
          logits_scale=2.2,
      )

    @jax.jit
    def f(query, key, value):
      return fa.PallasMosaicTpuFlashAttention()(
          q=query,
          k=key,
          v=value,
          mask=mask,
          is_causal=is_causal,
          logits_soft_cap=logits_soft_cap,
          logits_scale=2.2,
      )

    out = f(query=q, key=k, value=v)
    out_base = f_base(query=q, key=k, value=v)

    atol = 0.035 if logits_soft_cap else 0.15
    chex.assert_trees_all_close(
        out.astype(jnp.float32), out_base.astype(jnp.float32), atol=atol
    )


if __name__ == '__main__':
  absltest.main()
