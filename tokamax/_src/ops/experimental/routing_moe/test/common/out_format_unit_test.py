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
"""Unit tests for out_format."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec


class OutFormatTest(parameterized.TestCase):

  @parameterized.parameters(None, out_format_lib.ActivationFormat())
  def test_empty_or_none_activation_format_resolves_to_default(
      self, fmt: out_format_lib.ActivationFormat | None
  ):
    resolved = out_format_lib.ResolvedActivationFormat.create(
        fmt,
        default_dtype=jnp.bfloat16,
    )
    self.assertNotIsInstance(resolved, out_format_lib.ActivationFormat)
    self.assertEqual(
        resolved,
        out_format_lib.ResolvedActivationFormat(
            dtype=jnp.dtype(jnp.bfloat16), quant=None
        ),
    )

  def test_quant_without_dtype_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "dtype must be specified"):
      out_format_lib.ActivationFormat(quant=out_format_lib.auto_quant)
    with self.assertRaisesRegex(ValueError, "dtype must be specified"):
      out_format_lib.ActivationFormat(quant=out_format_lib.LaneStrided())

  def test_quant_without_default_quant_raises_not_implemented(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    with self.assertRaisesRegex(
        NotImplementedError, "Quantization is not supported"
    ):
      out_format_lib.ResolvedActivationFormat.create(
          fmt, default_dtype=jnp.bfloat16
      )

  @parameterized.parameters(
      out_format_lib.auto_quant,
      out_format_lib.LaneStrided(),
  )
  def test_resolved_activation_format_populates_quant_and_passes_through(
      self, quant_arg: out_format_lib.QuantFormat
  ):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=quant_arg
    )
    resolved_quant = out_format_lib.ResolvedActivationFormat.create(
        fmt,
        default_dtype=jnp.bfloat16,
        default_quant=out_format_lib.ResolvedLaneStrided(
            block_col_size=out_format_lib.tile_size
        ),
    )
    self.assertNotIsInstance(resolved_quant, out_format_lib.ActivationFormat)
    self.assertNotIsInstance(resolved_quant.quant, out_format_lib.LaneStrided)
    self.assertEqual(
        resolved_quant,
        out_format_lib.ResolvedActivationFormat(
            dtype=jnp.dtype(jnp.float8_e4m3fn),
            quant=out_format_lib.ResolvedLaneStrided(
                block_col_size=out_format_lib.tile_size
            ),
        ),
    )

    passthrough = out_format_lib.ResolvedActivationFormat.create(
        resolved_quant,
        default_dtype=jnp.float32,
    )
    self.assertIs(passthrough, resolved_quant)

  def test_cross_scheme_unset_field_raises_value_error(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.Subchannel(),
    )
    with self.assertRaisesRegex(ValueError, "Subchannel.block_size must be"):
      out_format_lib.ResolvedActivationFormat.create(
          fmt,
          default_dtype=jnp.bfloat16,
          default_quant=out_format_lib.ResolvedLaneStrided(
              block_col_size=out_format_lib.tile_size
          ),
      )

  @parameterized.parameters(
      (None, None),
      (
          out_format_lib.ResolvedSubchannel(
              block_size=out_format_lib.tile_size
          ),
          spec.SubchannelQuantSpec(block_size=1024),
      ),
      (
          out_format_lib.ResolvedSubchannel(block_size=512),
          spec.SubchannelQuantSpec(block_size=512),
      ),
      (
          out_format_lib.ResolvedLaneStrided(
              block_col_size=out_format_lib.tile_size
          ),
          spec.LaneStridedQuantSpec(block_col_size=1024),
      ),
      (
          out_format_lib.ResolvedLaneStrided(block_col_size=512),
          spec.LaneStridedQuantSpec(block_col_size=512),
      ),
  )
  def test_to_quant_spec_materializes_all_variants(
      self,
      resolved: out_format_lib.ResolvedQuantFormat | None,
      expected: spec.QuantSpec | None,
  ):
    self.assertEqual(
        out_format_lib.to_quant_spec(resolved, tile_col_size=1024), expected
    )


if __name__ == "__main__":
  absltest.main()
