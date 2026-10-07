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
"""Tests for the baseline JAX implementation of the DeepSeek-V4 RoPE."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jaxtyping
from tokamax._src.ops.experimental.tpu.rope import base
from tokamax._src.ops.experimental.tpu.rope import test_base

jax.config.parse_flags_with_absl()


def _args(x_shape, *, x_dtype=jnp.float32, rotary_dim=64, num_positions=None):
  num_positions = x_shape[0] if num_positions is None else num_positions
  return (
      jnp.zeros(x_shape, x_dtype),
      jnp.zeros((num_positions,), jnp.int32),
      jnp.zeros((16, rotary_dim), jnp.float32),
  )


class BaseRopeTest(test_base.RopeTestBase):

  def __init__(self, *args):
    super().__init__(*args, rope_fn=base.Rope())

  @parameterized.parameters((8, 512), (8, 2, 512))
  def test_rope_quant_rejects_wide_heads(self, *x_shape):
    with self.assertRaisesRegex(ValueError, "head_dim == 128"):
      base.Rope()(*_args(x_shape), mode="rope_quant")

  def test_qnorm_rope_rejects_rank_2(self):
    with self.assertRaisesRegex(ValueError, "rank 3"):
      base.Rope()(*_args((8, 512)), mode="qnorm_rope")

  @parameterized.parameters((8,), (8, 2, 2, 128))
  def test_rejects_bad_rank(self, *x_shape):
    with self.assertRaisesRegex(ValueError, "rank 2 or 3"):
      base.Rope()(*_args(x_shape))

  @parameterized.parameters(64, 192)
  def test_rejects_ragged_head_dim(self, head_dim):
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      base.Rope()(*_args((8, head_dim)))

  @parameterized.parameters(0, 63, 130, 256)
  def test_rejects_bad_rotary_dim(self, rotary_dim):
    with self.assertRaisesRegex(ValueError, "rotary_dim"):
      base.Rope()(*_args((8, 128), rotary_dim=rotary_dim))

  def test_rejects_bad_positions_shape(self):
    with self.assertRaisesRegex(ValueError, "positions"):
      base.Rope()(*_args((8, 128), num_positions=4))

  def test_rejects_bad_positions_dtype(self):
    x, positions, cos_sin_cache = _args((8, 128))
    with self.assertRaisesRegex(ValueError, "positions"):
      base.Rope()(x, positions.astype(jnp.float32), cos_sin_cache)

  def test_rejects_non_f32_cos_sin_cache(self):
    x, positions, cos_sin_cache = _args((8, 128))
    with self.assertRaisesRegex(ValueError, "cos_sin_cache"):
      base.Rope()(x, positions, cos_sin_cache.astype(jnp.bfloat16))

  def test_rejects_unknown_mode(self):
    # `jaxtyping` rejects anything but a `Mode` literal before `bind` runs.
    with self.assertRaises((ValueError, jaxtyping.TypeCheckError)):
      base.Rope()(*_args((8, 128)), mode="rope_qnorm")

  def test_rejects_integer_quant_dtype(self):
    with self.assertRaisesRegex(ValueError, "quant_dtype"):
      base.Rope()(*_args((8, 128)), mode="rope_quant", quant_dtype=jnp.int8)

  def test_quant_dtype_canonicalization(self):
    args = _args((8, 128))
    ba = base.Rope().bind(*args, mode="rope_quant")
    self.assertEqual(ba.arguments["quant_dtype"], jnp.float8_e4m3fn)
    ba = base.Rope().bind(*args, mode="rope", quant_dtype=jnp.float8_e5m2)
    self.assertIsNone(ba.arguments["quant_dtype"])


if __name__ == "__main__":
  absltest.main()
