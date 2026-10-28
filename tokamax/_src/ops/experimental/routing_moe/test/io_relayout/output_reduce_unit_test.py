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
"""Unit tests for output_reduce_unpack against the XLA reference."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.io_relayout import output_reduce as kernel
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_reference
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes
from tokamax._src.ops.experimental.routing_moe.test.common import test_util
from tokamax._src.ops.experimental.routing_moe.test.io_relayout import output_reduce_test_lib

jax.config.parse_flags_with_absl()


class OutputReduceUnitTest(parameterized.TestCase):

  def setUp(self):
    test_util.skip_unless_supported(self)
    super().setUp()

  def _assert_matches_reference(
      self, case_spec: test_cases.Case, out_dtype: jax.typing.DTypeLike
  ) -> None:
    case = output_reduce_test_lib.ReduceCase.make_case(case_spec, seed=42)
    if case_spec.ep_size == 1:
      actual = kernel.output_reduce_unpack(
          case.rows_compact, case.topk_weights, out_dtype=out_dtype
      )
    else:
      actual = kernel.output_reduce_unpack(
          case.rows_compact,
          case.topk_weights,
          case.topk_indices,
          case.group_offset,
          case.num_local_experts,
          out_dtype=out_dtype,
      )
    case.assert_close(actual)

  @parameterized.product(
      case_spec=test_cases.UNIT_CASES,
      out_dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_unit_cases(
      self, case_spec: test_cases.Case, out_dtype: jax.typing.DTypeLike
  ):
    self._assert_matches_reference(case_spec, out_dtype)

  @parameterized.product(
      num_tokens=[48, 128, 300, 512],
      top_k=[1, 2, 4, 10],
      ep_size=[1, 8],
      out_dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_matches_reference(
      self,
      num_tokens: int,
      top_k: int,
      ep_size: int,
      out_dtype: jax.typing.DTypeLike,
  ):
    shape = dataclasses.replace(test_shapes.BASIC, top_k=top_k)
    case_spec = test_cases.Case("edge", shape, num_tokens, ep_size=ep_size)
    self._assert_matches_reference(case_spec, out_dtype)

  def test_ragged_sum(self):
    """Each shard reduces only its picks, so the shard outputs sum to all."""
    ep_size = 8
    shape = test_shapes.MoEShape("ep", 512, 10, 2048, 1024)
    case = output_reduce_test_lib.ReduceCase.make_case(
        test_cases.Case("ep", shape, num_tokens=256, ep_size=ep_size), seed=99
    )
    local = case.num_local_experts

    shard_sum = jnp.zeros((256, shape.hidden_size), dtype=jnp.float32)
    for shard in range(ep_size):
      w_shard = test_reference.local_pick_weights(
          case.topk_weights, case.topk_indices, shard * local, local
      )
      valid = w_shard.reshape(-1)[:, None] != 0
      rows_shard = jnp.where(valid, case.rows, jnp.nan).reshape(
          layout_util.to_compact_shape(case.rows.shape, case.rows.dtype)
      )
      out_shard = kernel.output_reduce_unpack(
          rows_shard,
          case.topk_weights,
          case.topk_indices,
          shard * local,
          local,
          out_dtype=jnp.float32,
      )
      self.assertFalse(bool(jnp.isnan(out_shard).any()))
      shard_sum = shard_sum + out_shard

    dense = test_reference.topk_reduce(
        case.rows, case.topk_weights, jnp.float32
    )
    np.testing.assert_allclose(shard_sum, dense, rtol=2e-2, atol=2e-2)

  def test_all_zero_weights(self):
    """Zero weights produce zeros without reading the (NaN) rows."""
    num_tokens, top_k, hidden_size = 256, 10, 2048
    rows_nan = jnp.full(
        layout_util.to_compact_shape(
            (num_tokens * top_k, hidden_size), jnp.bfloat16
        ),
        jnp.nan,
        dtype=jnp.bfloat16,
    )
    w_zero = jnp.zeros((num_tokens, top_k), dtype=jnp.float32)
    out = kernel.output_reduce_unpack(rows_nan, w_zero)
    np.testing.assert_array_equal(
        out, jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16)
    )


if __name__ == "__main__":
  absltest.main()
