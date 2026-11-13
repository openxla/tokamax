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
"""Unit tests for RoutingGmmConfig."""

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.gmm import config as gmm_config
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.gmm import spec as gmm_spec_lib


def _tpu_info() -> pltpu.TpuInfo:
  return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _make_config(
    num_tokens: int,
    custom_plan: gmm_plan.RoutingGmmPlan | None = None,
    *,
    out_format: out_format_lib.ActivationFormat | None = None,
    size_ep: int = 1,
    size_n: int = 1280,
    quantize_w: bool = True,
) -> gmm_config.RoutingGmmConfig:
  """Builds a RoutingGmmConfig for a 512-expert, top-10 MoE layer."""
  num_experts, ep_size, top_k, size_k = 512, 4, 10, 2560
  x = spec.ActivationSpec(
      row_size=num_tokens,
      col_size=size_k,
      dtype=jnp.dtype(jnp.bfloat16),
      quant=None,
  )
  w = spec.WeightSpec(
      group_size=num_experts // ep_size,
      in_size=size_k,
      out_size=size_n,
      dtype=jnp.dtype(jnp.float8_e4m3fn if quantize_w else jnp.bfloat16),
      quant=spec.SubchannelQuantSpec(block_size=size_k) if quantize_w else None,
      has_bias=False,
      fused_gate_up=False,
  )
  routed_rows = num_tokens * top_k
  gather = spec.RoutingSpec(num_indices=routed_rows, remote_axis=None)
  scatter = spec.RoutingSpec(
      num_indices=routed_rows,
      remote_axis=None if size_ep == 1 else "ep",
  )
  resolved_out_format = out_format_lib.ResolvedActivationFormat.create(
      out_format,
      default_dtype=x.dtype,
      default_quant=out_format_lib.ResolvedLaneStrided(
          block_col_size=out_format_lib.tile_size
      ),
  )
  gmm_spec = gmm_spec_lib.RoutingGmmSpec(
      x=x,
      w=w,
      w2=None,
      gather=gather,
      scatter=scatter,
      num_global_groups=num_experts,
      fuse_act=None,
      size_ep=size_ep,
      size_row_group=routed_rows,
  )
  tpu_info = _tpu_info()
  resolved_plan = gmm_plan.ResolvedGmmPlan.create(
      custom_plan,
      spec=gmm_spec,
      out_format=resolved_out_format,
      tpu_info=tpu_info,
  )
  return gmm_config.RoutingGmmConfig.create(
      spec=gmm_spec,
      out_format=resolved_out_format,
      plan=resolved_plan,
      tpu_info=tpu_info,
  )


class OutSpecAndValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="unquantized",
          quantize_w=False,
          custom_plan=None,
          expected=gmm_config.MatmulMode.UNQUANTIZED,
      ),
      dict(
          testcase_name="quantized",
          quantize_w=True,
          custom_plan=None,
          expected=gmm_config.MatmulMode.QUANTIZED,
      ),
      dict(
          testcase_name="weight_only",
          quantize_w=True,
          custom_plan=gmm_plan.RoutingGmmPlan(matmul_dtype=jnp.bfloat16),
          expected=gmm_config.MatmulMode.WEIGHT_ONLY_SCALE_AFTER,
      ),
  )
  def test_matmul_mode(
      self,
      quantize_w: bool,
      custom_plan: gmm_plan.RoutingGmmPlan | None,
      expected: gmm_config.MatmulMode,
  ):
    cfg = _make_config(8, custom_plan, quantize_w=quantize_w)
    self.assertEqual(cfg.compute.matmul_mode, expected)

  def test_default_out_format_uses_input_dtype_and_unquantized(self):
    out_spec = _make_config(8).out.spec
    self.assertEqual(
        out_spec,
        spec.ActivationSpec(
            row_size=80,
            col_size=1280,
            dtype=jnp.dtype(jnp.bfloat16),
            quant=None,
        ),
    )

  def test_remote_scatter_divides_out_rows_by_size_ep(self):
    out_spec = _make_config(8, size_ep=4).out.spec
    self.assertEqual(out_spec.row_size, 20)
    self.assertEqual(out_spec.col_size, 1280)

  @parameterized.parameters(1024, 1536)
  def test_tile_size_quant_format_resolves_to_tile_n(self, size_n: int):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    out_spec = _make_config(8, out_format=fmt, size_n=size_n).out.spec
    self.assertEqual(out_spec.dtype, jnp.dtype(jnp.float8_e4m3fn))
    self.assertEqual(
        out_spec.quant,
        spec.LaneStridedQuantSpec(block_col_size=size_n),
    )

  def test_explicit_quant_block_col_size_matching_tile_n_is_preserved(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.LaneStrided(block_col_size=1024),
    )
    out_spec = _make_config(8, out_format=fmt, size_n=1024).out.spec
    self.assertEqual(
        out_spec.quant,
        spec.LaneStridedQuantSpec(block_col_size=1024),
    )

  def test_explicit_quant_block_col_size_mismatching_tile_n_raises(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.LaneStrided(block_col_size=512),
    )
    with self.assertRaisesRegex(
        ValueError, "Output quantization block_col_size"
    ):
      _make_config(8, out_format=fmt, size_n=1024)

  def test_subchannel_out_quant_on_gmm_raises_not_implemented(self):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.Subchannel(block_size=512),
    )
    with self.assertRaisesRegex(NotImplementedError, "Only LaneStrided"):
      _make_config(8, out_format=fmt, size_n=1024)

  @parameterized.named_parameters(
      dict(
          testcase_name="num_stash_buffers",
          custom_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=2),
      ),
      dict(
          testcase_name="stash_dtype",
          custom_plan=gmm_plan.RoutingGmmPlan(stash_dtype=jnp.bfloat16),
      ),
  )
  def test_stash_fields_on_unquantized_out_raise_value_error(
      self, custom_plan: gmm_plan.RoutingGmmPlan
  ):
    with self.assertRaisesRegex(
        ValueError, "has no effect when output is not quantized"
    ):
      _make_config(8, custom_plan)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_buffers",
          custom_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=0),
          expected_regex="num_stash_buffers other than 2",
      ),
      dict(
          testcase_name="one_buffer",
          custom_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=1),
          expected_regex="num_stash_buffers other than 2",
      ),
      dict(
          testcase_name="three_buffers",
          custom_plan=gmm_plan.RoutingGmmPlan(num_stash_buffers=3),
          expected_regex="num_stash_buffers other than 2",
      ),
      dict(
          testcase_name="float32_stash_dtype",
          custom_plan=gmm_plan.RoutingGmmPlan(stash_dtype=jnp.float32),
          expected_regex="stash_dtype other than bfloat16",
      ),
  )
  def test_unsupported_stash_fields_on_quantized_out_raise_not_implemented(
      self, custom_plan: gmm_plan.RoutingGmmPlan, expected_regex: str
  ):
    fmt = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=out_format_lib.auto_quant
    )
    with self.assertRaisesRegex(NotImplementedError, expected_regex):
      _make_config(8, custom_plan, out_format=fmt, size_n=1024)


if __name__ == "__main__":
  absltest.main()
