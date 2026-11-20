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
"""Unit tests for RoutingGmmSpec."""

import dataclasses

from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.gmm import spec as gmm_spec_lib


def _make_spec(
    num_tokens: int,
    *,
    with_down: bool = False,
) -> gmm_spec_lib.RoutingGmmSpec:
  """Builds a RoutingGmmSpec for a 512-expert, top-10 MoE layer."""
  num_experts, ep_size, top_k, size_k = 512, 4, 10, 2560
  size_n = 1536 if with_down else 1280
  x = spec.ActivationSpec(
      row_size=num_tokens,
      col_size=size_k,
      dtype=jnp.dtype(jnp.bfloat16),
      quant=None,
  )

  def weight_spec(
      in_size: int, out_size: int, *, fused_gate_up: bool = False
  ) -> spec.WeightSpec:
    return spec.WeightSpec(
        group_size=num_experts // ep_size,
        in_size=in_size,
        out_size=out_size,
        dtype=jnp.dtype(jnp.float8_e4m3fn),
        quant=spec.SubchannelQuantSpec(block_size=in_size),
        has_bias=False,
        fused_gate_up=fused_gate_up,
    )

  w = weight_spec(size_k, size_n, fused_gate_up=with_down)
  w2 = weight_spec(size_n // 2, size_k) if with_down else None
  routed_rows = num_tokens * top_k
  return gmm_spec_lib.RoutingGmmSpec(
      x=x,
      w=w,
      w2=w2,
      gather=spec.RoutingSpec(num_indices=routed_rows, remote_axis=None),
      scatter=spec.RoutingSpec(num_indices=routed_rows, remote_axis=None),
      num_global_groups=num_experts,
      fuse_act="silu" if with_down else None,
      size_ep=1,
      size_row_group=routed_rows,
  )


class RoutingGmmSpecTest(absltest.TestCase):

  def test_derived_sizes_single_and_two_stage(self):
    single = _make_spec(8, with_down=False)
    self.assertEqual(single.size_k, 2560)
    self.assertIsNone(single.size_k2)
    self.assertEqual(single.size_out_m, 80)
    self.assertEqual(single.size_out_n, 1280)
    self.assertEqual(single.size_local_groups, 128)
    self.assertFalse(single.is_fused_act)

    fused = _make_spec(8, with_down=True)
    self.assertEqual(fused.size_k, 2560)
    self.assertEqual(fused.size_k2, 768)
    self.assertEqual(fused.size_out_m, 80)
    self.assertEqual(fused.size_out_n, 2560)
    self.assertTrue(fused.is_fused_act)

  def test_validation_rejects_mismatched_specs(self):
    base = _make_spec(8, with_down=True)
    bad_x = dataclasses.replace(base.x, col_size=512)
    with self.assertRaisesRegex(ValueError, "must match w in_size"):
      dataclasses.replace(base, x=bad_x)

    with self.assertRaisesRegex(ValueError, "requires a remote scatter"):
      dataclasses.replace(base, size_ep=2)

    bad_scatter = spec.RoutingSpec(num_indices=79, remote_axis=None)
    with self.assertRaisesRegex(ValueError, "scatter must map every computed"):
      dataclasses.replace(base, scatter=bad_scatter)

  def test_subchannel_weight_spec_scale_dtype(self):
    subchannel = _make_spec(8, with_down=False).w
    self.assertEqual(subchannel.scale_dtype, jnp.dtype(jnp.float32))


def _block_scaled_weight_spec(
    *,
    in_size: int = 512,
    scale_rows: int = 32,
    dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
) -> spec.WeightSpec:
  return spec.WeightSpec(
      group_size=4,
      in_size=in_size,
      out_size=64,
      dtype=jnp.dtype(dtype),
      quant=spec.BlockScaledQuantSpec(scale_rows=scale_rows),
      has_bias=False,
      fused_gate_up=False,
  )


def _e8m0(shape: tuple[int, ...]) -> jax.Array:
  return jax.lax.bitcast_convert_type(
      jnp.full(shape, 127, jnp.uint8), jnp.float8_e8m0fnu
  )


class BlockScaledWeightSpecTest(absltest.TestCase):

  def test_scale_shape_and_dtype(self):
    w = _block_scaled_weight_spec(scale_rows=64, dtype=jnp.float8_e5m2)
    self.assertEqual(w.scale_dtype, jnp.dtype(jnp.float8_e8m0fnu))
    self.assertEqual(w.scale_shape, (4, 64, 64))

  def test_rejects_non_positive_scale_rows(self):
    for bad_rows in (0, -4):
      with self.assertRaisesRegex(ValueError, "scale_rows must be positive"):
        spec.BlockScaledQuantSpec(scale_rows=bad_rows)

  def test_rejects_non_fp8_dtype(self):
    with self.assertRaisesRegex(ValueError, "requires dtype in"):
      _block_scaled_weight_spec(dtype=jnp.bfloat16)

  def test_wrapper_infers_spec_from_3d_scales(self):
    weight = jnp.zeros((4, 512, 64), dtype=jnp.float8_e4m3fn)
    for rows in (32, 64):
      scale = _e8m0((4, rows, 64))
      wrapped = arg.SingleWeightWrapper.create(weight, scale)
      self.assertEqual(
          wrapped.spec.quant, spec.BlockScaledQuantSpec(scale_rows=rows)
      )
      self.assertEqual(wrapped.spec.scale_shape, (4, rows, 64))

  def test_wrapper_create_with_explicit_quant_validates_shape(self):
    weight = jnp.zeros((4, 512, 64), dtype=jnp.float8_e4m3fn)
    explicit_quant = spec.BlockScaledQuantSpec(scale_rows=32)
    wrapped = arg.SingleWeightWrapper.create(
        weight, _e8m0((4, 32, 64)), quant=explicit_quant
    )
    self.assertEqual(wrapped.spec.quant, explicit_quant)
    with self.assertRaisesRegex(ValueError, "Weight scale must have shape"):
      arg.SingleWeightWrapper.create(
          weight, _e8m0((4, 64, 64)), quant=explicit_quant
      )

  def test_wrapper_rejects_non_e8m0_3d_scale(self):
    weight = jnp.zeros((4, 512, 64), dtype=jnp.float8_e4m3fn)
    with self.assertRaisesRegex(ValueError, "must have dtype float8_e8m0fnu"):
      arg.SingleWeightWrapper.create(
          weight, jnp.ones((4, 32, 64), dtype=jnp.float32)
      )

  def test_wrapper_rejects_invalid_scale_rank(self):
    weight = jnp.zeros((4, 512, 64), dtype=jnp.float8_e4m3fn)
    with self.assertRaisesRegex(ValueError, "must be 3D .* or 4D"):
      arg.SingleWeightWrapper.create(weight, _e8m0((4, 64)))


class SubchannelWeightWrapperTest(absltest.TestCase):

  def test_allows_4d_e8m0_scale(self):
    weight = jnp.zeros((4, 512, 64), dtype=jnp.float8_e4m3fn)
    wrapped = arg.SingleWeightWrapper.create(weight, _e8m0((4, 16, 1, 64)))
    self.assertEqual(
        wrapped.spec.quant, spec.SubchannelQuantSpec(block_size=32)
    )
    self.assertEqual(wrapped.spec.scale_shape, (4, 16, 1, 64))


if __name__ == "__main__":
  absltest.main()
