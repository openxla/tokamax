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
from tokamax._src.ops.experimental.routing_moe.common import spec as spec_lib
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.io_relayout import output_reduce as kernel
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_cases
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_reference
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_shapes
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_util
from tokamax._src.ops.experimental.routing_moe.test.io_relayout import output_reduce_test_lib

jax.config.parse_flags_with_absl()


class OutputReduceUnitTest(parameterized.TestCase):

  def setUp(self):
    helper_util.skip_unless_supported(self)
    super().setUp()

  def _assert_matches_reference(
      self,
      case_spec: helper_cases.Case,
      out_dtype: jax.typing.DTypeLike,
      *,
      in_dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ) -> None:
    case = output_reduce_test_lib.ReduceCase.make_case(
        case_spec, in_dtype=in_dtype, seed=42
    )
    if case_spec.sharding.ep == 1:
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
      case_spec=helper_cases.UNIT_CASES,
      in_dtype=[jnp.bfloat16, jnp.float8_e4m3fn],
      out_dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_unit_cases(
      self,
      case_spec: helper_cases.Case,
      in_dtype: jax.typing.DTypeLike,
      out_dtype: jax.typing.DTypeLike,
  ):
    self._assert_matches_reference(case_spec, out_dtype, in_dtype=in_dtype)

  @parameterized.product(
      num_tokens=[48, 128, 300, 512],
      top_k=[1, 2, 4, 10],
      ep_size=[1, 8],
      in_dtype=[jnp.bfloat16, jnp.float8_e4m3fn],
      out_dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_matches_reference(
      self,
      num_tokens: int,
      top_k: int,
      ep_size: int,
      in_dtype: jax.typing.DTypeLike,
      out_dtype: jax.typing.DTypeLike,
  ):
    shape = dataclasses.replace(helper_shapes.BASIC, top_k=top_k)
    case_spec = helper_cases.Case(
        "edge", shape, num_tokens, helper_cases.MoESharding(ep=ep_size)
    )
    self._assert_matches_reference(case_spec, out_dtype, in_dtype=in_dtype)

  @parameterized.product(
      num_tokens=[512, 16384],
      top_k=[11, 16],
      ep_size=[1, 8],
  )
  def test_stress_high_topk_and_large_tokens(
      self, num_tokens: int, top_k: int, ep_size: int
  ):
    """Top-k > 10 and many tokens: the SMEM ring must not overflow or spill."""
    shape = helper_shapes.MoEShape("stress", 512, top_k, 1024, 1024)
    case_spec = helper_cases.Case(
        "stress", shape, num_tokens, helper_cases.MoESharding(ep=ep_size)
    )
    self._assert_matches_reference(case_spec, jnp.bfloat16)

  @parameterized.parameters(jnp.bfloat16, jnp.float8_e4m3fn)
  def test_ragged_sum(self, in_dtype: jax.typing.DTypeLike):
    """Each shard reduces only its picks, so the shard outputs sum to all."""
    in_dtype = jnp.dtype(in_dtype)
    ep_size = 8
    shape = helper_shapes.MoEShape("ep", 512, 10, 2048, 1024)
    case = output_reduce_test_lib.ReduceCase.make_case(
        helper_cases.Case(
            "ep",
            shape,
            num_tokens=256,
            sharding=helper_cases.MoESharding(ep=ep_size),
        ),
        in_dtype=in_dtype,
        seed=99,
    )
    local = case.num_local_experts

    shard_sum = jnp.zeros((256, shape.hidden_size), dtype=jnp.float32)
    for shard in range(ep_size):
      w_shard = helper_reference.local_pick_weights(
          case.topk_weights, case.topk_indices, shard * local, local
      )
      valid = w_shard.reshape(-1) != 0
      if in_dtype.itemsize == 1:
        rows_shard = output_reduce_test_lib.quantize_to_compact(
            case.rows, in_dtype, valid=valid
        )
      else:
        rows_shard = input_pack.input_pack(
            jnp.where(valid[:, None], case.rows, jnp.nan)
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

    dense = helper_reference.topk_reduce(
        case.dequantized_rows, case.topk_weights, jnp.float32
    )
    np.testing.assert_allclose(shard_sum, dense, rtol=2e-2, atol=2e-2)

  @parameterized.parameters(jnp.bfloat16, jnp.float8_e4m3fn)
  def test_all_zero_weights(self, in_dtype: jax.typing.DTypeLike):
    """Zero weights produce zeros without reading the (NaN) rows."""
    num_tokens, top_k, hidden_size = 256, 10, 2048
    if jnp.dtype(in_dtype) == jnp.bfloat16:
      rows_nan = input_pack.input_pack(
          jnp.full(
              (num_tokens * top_k, hidden_size), jnp.nan, dtype=jnp.bfloat16
          )
      )
    else:
      rows_nan = output_reduce_test_lib.quantize_to_compact(
          jnp.zeros((num_tokens * top_k, hidden_size), dtype=jnp.bfloat16),
          in_dtype,
          valid=jnp.zeros((num_tokens * top_k,), dtype=bool),
      )
    w_zero = jnp.zeros((num_tokens, top_k), dtype=jnp.float32)
    out = kernel.output_reduce_unpack(rows_nan, w_zero)
    np.testing.assert_array_equal(
        out, jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16)
    )

  def test_unsupported_rows_dtype_raises(self):
    num_tokens, top_k, hidden_size = 16, 4, 512
    rows_i8 = input_pack.input_pack(
        jnp.zeros((num_tokens * top_k, hidden_size), dtype=jnp.int8)
    )
    weights = jnp.ones((num_tokens, top_k), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "rows dtype must be"):
      kernel.output_reduce_unpack(rows_i8, weights)

  def test_non_strided_activation_quant_raises(self):
    spec = spec_lib.ActivationSpec(
        row_size=16,
        col_size=512,
        dtype=jnp.dtype(jnp.float8_e4m3fn),
        quant=spec_lib.SubchannelQuantSpec(block_size=512),
    )
    with self.assertRaisesRegex(NotImplementedError, "LaneStridedQuantSpec"):
      _ = spec.quant_col_size

  @parameterized.parameters(128, 384, 640)
  def test_partial_chunk_hidden_size(self, hidden_size: int):
    shape = helper_shapes.MoEShape("partial_chunk", 512, 4, hidden_size, 256)
    case_spec = helper_cases.Case(
        "partial_chunk",
        shape,
        num_tokens=64,
        sharding=helper_cases.MoESharding(),
    )
    self._assert_matches_reference(case_spec, jnp.bfloat16)

  @parameterized.parameters(128, 384, 640)
  def test_partial_chunk_unquantized_fp8_hidden_size(self, hidden_size: int):
    num_tokens, top_k = 64, 4
    key = jax.random.key(7)
    rows_bf16 = jax.random.normal(
        key, (num_tokens * top_k, hidden_size), dtype=jnp.bfloat16
    )
    rows_fp8 = rows_bf16.astype(jnp.float8_e4m3fn)
    weights = jax.random.uniform(key, (num_tokens, top_k), dtype=jnp.float32)
    actual = kernel.output_reduce_unpack(
        input_pack.input_pack(rows_fp8), weights, out_dtype=jnp.bfloat16
    )
    expected = helper_reference.topk_reduce(
        rows_fp8.astype(jnp.float32), weights, jnp.bfloat16
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)

  def test_partial_chunk_quantized_raises(self):
    num_tokens, top_k, hidden_size = 16, 4, 640
    rows_packed = input_pack.input_pack(
        jnp.zeros((num_tokens * top_k, hidden_size), dtype=jnp.float8_e4m3fn)
    )
    q_spec = dataclasses.replace(
        rows_packed.spec,
        quant=spec_lib.LaneStridedQuantSpec(block_col_size=hidden_size),
    )
    data = jnp.zeros(q_spec.physical_compact_shape, dtype=jnp.float8_e4m3fn)
    rows_q = dataclasses.replace(rows_packed, data=data, spec=q_spec)
    weights = jnp.ones((num_tokens, top_k), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "must fill whole"):
      kernel.output_reduce_unpack(rows_q, weights)


if __name__ == "__main__":
  absltest.main()
