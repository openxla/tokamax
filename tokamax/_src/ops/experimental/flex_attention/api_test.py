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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.experimental.flex_attention import api
from tokamax._src.ops.experimental.flex_attention import example_masks
from tokamax._src.ops.experimental.flex_attention import example_score_mods


def _make_inputs(
    dtype: jax.typing.DTypeLike,
    batch_size: int = 2,
    seq_len: int = 256,
    num_heads: int = 4,
    head_dim: int = 128,
    is_mha: bool = False,
) -> dict[str, jax.Array]:
  kv_heads = 1 if is_mha else num_heads
  q_shape = jax.ShapeDtypeStruct(
      (batch_size, seq_len, num_heads, head_dim), dtype
  )
  kv_shape = jax.ShapeDtypeStruct(
      (batch_size, seq_len, kv_heads, head_dim), dtype
  )
  out = dict(query=q_shape, key=kv_shape, value=kv_shape)
  return numerics.random_initialize(out)


class FlexAttentionApiTest(parameterized.TestCase):

  @parameterized.product(
      is_causal=(True, False),
      scale=(4.3, None),
      is_mha=(True, False),
  )
  def test_against_dot_product_attention(self, is_causal, scale, is_mha):
    soft_cap = 32.0
    precision = jax.lax.Precision.HIGHEST
    inputs = _make_inputs(dtype=jnp.float32, is_mha=is_mha)

    f_ref = jax.jit(
        functools.partial(
            attention_api.dot_product_attention,
            logits_soft_cap=soft_cap,
            is_causal=is_causal,
            scale=scale,
            precision=precision,
            implementation='xla',
        )
    )
    mask_mode = example_masks.causal_mask if is_causal else None
    f_flex = jax.jit(
        functools.partial(
            api.flex_attention,
            score_mod=example_score_mods.generate_tanh_softcap(soft_cap),
            mask_mod=mask_mode,
            precision=precision,
            scale=scale,
            implementation='xla',
        )
    )
    out_ref = f_ref(**inputs)
    out_flex = f_flex(**inputs)
    chex.assert_trees_all_close(out_ref, out_flex, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
