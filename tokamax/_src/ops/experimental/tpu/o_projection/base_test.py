# Copyright 2026 Google LLC
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
"""Tests for the baseline JAX DeepSeek-V4 `wo_a` projection."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.o_projection import base
from tokamax._src.ops.experimental.tpu.o_projection import test_base

jax.config.parse_flags_with_absl()


def _args(
    *,
    num_tokens=8,
    num_heads=16,
    head_dim=128,
    rotary_dim=64,
    x_dtype=jnp.bfloat16,
    reduction=None,
    out_features=512,
    wo_a_dtype=jnp.float8_e4m3fn,
    scale_shape=None,
    num_positions=None,
):
  reduction = 8 * head_dim if reduction is None else reduction
  scale_shape = (out_features,) if scale_shape is None else scale_shape
  num_positions = num_tokens if num_positions is None else num_positions
  return (
      jax.ShapeDtypeStruct((num_tokens, num_heads, head_dim), x_dtype),
      jax.ShapeDtypeStruct((num_positions,), jnp.int32),
      jax.ShapeDtypeStruct((16, rotary_dim), jnp.float32),
      jax.ShapeDtypeStruct((reduction, out_features), wo_a_dtype),
      jax.ShapeDtypeStruct(scale_shape, jnp.float32),
  )


class BaseOProjectionTest(test_base.OProjectionTestBase):

  def __init__(self, *args):
    super().__init__(*args, o_projection_fn=base.OProjection())

  @parameterized.parameters(False, True)
  def test_single_lane_block_heads(self, quantize_activations):
    """`head_dim == 128`: the whole head is the roped lane block."""
    self._check(
        quantize_activations=quantize_activations,
        num_tokens=64,
        num_groups=2,
        lora_rank=256,
        head_dim=128,
        rotary_dim=128,
    )

  def test_accepts_valid_args(self):
    base.OProjection().bind(*_args())

  def test_rejects_non_bf16_x(self):
    with self.assertRaisesRegex(ValueError, "bf16"):
      base.OProjection().bind(*_args(x_dtype=jnp.float32))

  def test_rejects_ragged_head_dim(self):
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      base.OProjection().bind(*_args(head_dim=192))

  def test_rejects_bad_positions(self):
    with self.assertRaisesRegex(ValueError, "positions"):
      base.OProjection().bind(*_args(num_positions=4))

  @parameterized.parameters(0, 63, 256)
  def test_rejects_bad_rotary_dim(self, rotary_dim):
    with self.assertRaisesRegex(ValueError, "rotary_dim"):
      base.OProjection().bind(*_args(rotary_dim=rotary_dim))

  def test_rejects_non_fp8_wo_a(self):
    with self.assertRaisesRegex(ValueError, "float8_e4m3fn"):
      base.OProjection().bind(*_args(wo_a_dtype=jnp.bfloat16))

  @parameterized.parameters(4, 16)
  def test_rejects_heads_per_group_other_than_8(self, heads_per_group):
    with self.assertRaisesRegex(ValueError, "heads_per_group"):
      base.OProjection().bind(*_args(reduction=heads_per_group * 128))

  def test_rejects_ragged_num_heads(self):
    with self.assertRaisesRegex(ValueError, "num_heads"):
      base.OProjection().bind(*_args(num_heads=12))

  def test_rejects_ragged_out_features(self):
    with self.assertRaisesRegex(ValueError, "groups"):
      base.OProjection().bind(
          *_args(num_heads=24, out_features=512, scale_shape=(512,))
      )

  def test_rejects_bad_scale_shape(self):
    with self.assertRaisesRegex(ValueError, "wo_a_scale"):
      base.OProjection().bind(*_args(scale_shape=(256,)))


if __name__ == "__main__":
  absltest.main()
