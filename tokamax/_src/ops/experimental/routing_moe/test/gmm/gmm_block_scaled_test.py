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
"""Block-scaled FP8 numerical tests for routing_gmm on TPU_8T."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import block_scaled
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import out_format
from tokamax._src.ops.experimental.routing_moe.gmm import kernel
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.test.gmm import gmm_test_lib
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_inputs
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_reference
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_shapes
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_util

jax.config.parse_flags_with_absl()

_FP8_OUT_FORMAT = out_format.ActivationFormat(
    dtype=jnp.float8_e4m3fn, quant=out_format.LaneStrided()
)


def _assert_close(actual: np.ndarray, expected: np.ndarray, tol: float) -> None:
  atol = tol * max(float(np.abs(expected).max()), 1e-6)
  np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)


def _block_scaled_weight(
    w: jax.Array,
    dtype: jax.typing.DTypeLike,
    *,
    bias: jax.Array | None = None,
    fused_gate_up: bool = False,
    prepare_scales: bool = False,
) -> arg.SingleWeightWrapper[jax.Array]:
  w_q, w_scale = block_scaled.quantize(w, dtype)
  if prepare_scales:
    w_scale = block_scaled.prepare_scales(w_scale, w.shape[1], dtype=dtype)
  return arg.SingleWeightWrapper.create(
      w_q, w_scale, bias, fused_gate_up=fused_gate_up
  )


def _make_block_scaled_inputs(
    num_tokens: int,
    *,
    hidden_size: int = 512,
    intermediate_size: int = 256,
    w_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    has_bias: bool = False,
    prepare_scales: bool = False,
) -> tuple[
    jax.Array,
    arg.SingleWeightWrapper[jax.Array],
    arg.SingleWeightWrapper[jax.Array],
    helper_inputs.Routing,
]:
  shape = helper_shapes.MoEShape(
      name="block_scaled",
      num_experts=4,
      top_k=2,
      hidden_size=hidden_size,
      intermediate_size=intermediate_size,
      has_bias=has_bias,
  )
  inputs = helper_inputs.MoEInputs.generate(
      shape, num_tokens, force_expert=0, has_bias=has_bias
  )
  _, topk_indices = helper_inputs.top_k(inputs.gating_output, shape.top_k)
  routing = helper_inputs.Routing.make_routing(topk_indices, shape.num_experts)
  w1 = _block_scaled_weight(
      inputs.w1,
      w_dtype,
      bias=inputs.w1_bias,
      fused_gate_up=True,
      prepare_scales=prepare_scales,
  )
  w2 = _block_scaled_weight(
      inputs.w2, w_dtype, bias=inputs.w2_bias, prepare_scales=prepare_scales
  )
  return inputs.hidden_states, w1, w2, routing


class RoutingGmmBlockScaledTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    helper_util.skip_unless_block_scaled_supported(self)

  @parameterized.product(
      num_tokens=[1, 16, 64],
      dtype_and_tol=[(jnp.float8_e4m3fn, 0.06), (jnp.float8_e5m2, 0.12)],
  )
  def test_gather_gmm_fused_act(
      self, num_tokens: int, dtype_and_tol: tuple[jax.typing.DTypeLike, float]
  ):
    w_dtype, tol = dtype_and_tol
    x, w1, _, routing = _make_block_scaled_inputs(num_tokens, w_dtype=w_dtype)
    out = kernel.routing_gmm(
        input_pack.input_pack(x),
        w1,
        gather=routing.gather,
        fuse_act="silu",
        **routing.kernel_kwargs,
    )
    actual = np.asarray(gmm_test_lib.to_large_layout(out)[routing.sorted_rows])
    expected = np.asarray(
        helper_reference.unfused_gather_gmm(
            x, w1, routing.group_sizes, routing.gather_indices
        )[routing.sorted_rows]
    )
    _assert_close(actual, expected, tol)

  def test_gather_gmm_fused_act_bias(self):
    x, w1, _, routing = _make_block_scaled_inputs(16, has_bias=True)
    assert w1.bias is not None
    w1 = dataclasses.replace(w1, bias=w1.bias * 10)
    out = kernel.routing_gmm(
        input_pack.input_pack(x),
        w1,
        gather=routing.gather,
        fuse_act="silu",
        **routing.kernel_kwargs,
    )
    actual = np.asarray(gmm_test_lib.to_large_layout(out)[routing.sorted_rows])
    expected = np.asarray(
        helper_reference.unfused_gather_gmm(
            x, w1, routing.group_sizes, routing.gather_indices
        )[routing.sorted_rows]
    )
    _assert_close(actual, expected, 0.06)

  def test_gather_gmm_mixed_dtypes_f32_acc_and_large_tokens(self):
    x, w1, _, routing = _make_block_scaled_inputs(160, w_dtype=jnp.float8_e5m2)
    out = kernel.routing_gmm(
        input_pack.input_pack(x),
        w1,
        gather=routing.gather,
        fuse_act="silu",
        custom_plan=gmm_plan.RoutingGmmPlan(
            matmul_dtype=jnp.float8_e4m3fn, acc_dtype=jnp.float32
        ),
        **routing.kernel_kwargs,
    )
    actual = np.asarray(gmm_test_lib.to_large_layout(out)[routing.sorted_rows])
    expected = np.asarray(
        helper_reference.unfused_gather_gmm(
            x, w1, routing.group_sizes, routing.gather_indices
        )[routing.sorted_rows]
    )
    _assert_close(actual, expected, 0.12)

  def test_gmm_scatter_bf16(self):
    x, _, w2, routing = _make_block_scaled_inputs(
        16, hidden_size=512, intermediate_size=512
    )
    x_sorted = x[routing.gather_indices]
    out = kernel.routing_gmm(
        input_pack.input_pack(x_sorted),
        w2,
        scatter=routing.scatter,
        **routing.kernel_kwargs,
    )
    actual = np.asarray(gmm_test_lib.to_large_layout(out))
    expected = np.asarray(
        helper_reference.unfused_gmm_scatter(
            x_sorted, w2, routing.group_sizes, routing.sorted_to_flat
        )
    )
    rows = routing.scattered_rows
    _assert_close(actual[rows], expected[rows], 0.08)

  def test_gmm_scatter_fp8(self):
    x, _, w2, routing = _make_block_scaled_inputs(
        16, hidden_size=512, intermediate_size=512
    )
    x_sorted = x[routing.gather_indices]
    out = kernel.routing_gmm(
        input_pack.input_pack(x_sorted),
        w2,
        scatter=routing.scatter,
        out_format=_FP8_OUT_FORMAT,
        **routing.kernel_kwargs,
    )
    self.assertEqual(out.data.dtype, jnp.float8_e4m3fn)
    actual = np.asarray(gmm_test_lib.dequantize_rows(out.data))
    expected = np.asarray(
        helper_reference.unfused_gmm_scatter(
            x_sorted, w2, routing.group_sizes, routing.sorted_to_flat
        )
    )
    rows = routing.scattered_rows
    _assert_close(actual[rows], expected[rows], 0.08)

  @parameterized.named_parameters(
      dict(testcase_name="default_512", hidden_size=512, custom_plan=None),
      dict(testcase_name="default_1024", hidden_size=1024, custom_plan=None),
      dict(
          testcase_name="fused_512",
          hidden_size=512,
          custom_plan=gmm_plan.RoutingGmmPlan(fuse_up_and_down=True),
      ),
      dict(
          testcase_name="fused_1024",
          hidden_size=1024,
          custom_plan=gmm_plan.RoutingGmmPlan(fuse_up_and_down=True),
      ),
      dict(
          testcase_name="unfused_512",
          hidden_size=512,
          custom_plan=gmm_plan.RoutingGmmPlan(fuse_up_and_down=False),
      ),
      dict(
          testcase_name="unfused_1024",
          hidden_size=1024,
          custom_plan=gmm_plan.RoutingGmmPlan(fuse_up_and_down=False),
      ),
      dict(
          testcase_name="fused_with_bias",
          hidden_size=512,
          has_bias=True,
          custom_plan=gmm_plan.RoutingGmmPlan(fuse_up_and_down=True),
      ),
      dict(
          testcase_name="unfused_multi_tile_n_with_bias",
          hidden_size=512,
          intermediate_size=512,
          has_bias=True,
          custom_plan=gmm_plan.RoutingGmmPlan(
              fuse_up_and_down=False, tile_n=256
          ),
      ),
      dict(
          testcase_name="large_hidden_and_prepared_scales",
          hidden_size=1536,
          prepare_scales=True,
          custom_plan=None,
      ),
      dict(
          testcase_name="large_hidden_compact_scales",
          hidden_size=1536,
          prepare_scales=False,
          custom_plan=None,
      ),
  )
  def test_gather_gmm_gmm_scatter(
      self,
      *,
      hidden_size: int,
      intermediate_size: int = 256,
      has_bias: bool = False,
      prepare_scales: bool = False,
      custom_plan: gmm_plan.RoutingGmmPlan | None,
  ):
    x, w1, w2, routing = _make_block_scaled_inputs(
        16,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        has_bias=has_bias,
        prepare_scales=prepare_scales,
    )
    out = kernel.routing_gmm(
        input_pack.input_pack(x),
        w1,
        w2=w2,
        gather=routing.gather,
        scatter=routing.scatter,
        fuse_act="silu",
        custom_plan=custom_plan,
        **routing.kernel_kwargs,
    )
    actual = np.asarray(gmm_test_lib.to_large_layout(out))
    expected = np.asarray(
        helper_reference.unfused_gather_gmm_gmm_scatter(
            x,
            w1,
            w2,
            routing.group_sizes,
            routing.gather_indices,
            routing.sorted_to_flat,
        )
    )
    rows = routing.scattered_rows
    _assert_close(actual[rows], expected[rows], 0.08)


if __name__ == "__main__":
  absltest.main()
