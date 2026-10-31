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
"""Input preparation for output_reduce tests."""

import dataclasses
import functools
from typing import Self

import jax
import jax.numpy as jnp
from tokamax._src import test_utils
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_reference


@functools.partial(jax.jit, static_argnames=("dtype",))
def quantize_to_compact(
    rows: jax.Array,
    dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    *,
    valid: jax.Array | None = None,
) -> arg.ActivationWrapper[jax.Array]:
  """Quantizes `[m, hidden]` rows to `[m, chunks + 1, 4, 128]` with scales."""
  dtype = jnp.dtype(dtype)
  lanes = 128
  m, cols = rows.shape
  rows_3d = rows.astype(jnp.float32).reshape(m, -1, lanes)
  absmax = jnp.max(jnp.abs(rows_3d), axis=1)
  dtype_max = float(jnp.finfo(dtype).max)
  scales = jnp.where(absmax == 0, 1.0, absmax / dtype_max)
  q_rows = (rows_3d / scales[:, None, :]).astype(dtype).reshape(rows.shape)
  payload = q_rows.reshape(layout_util.to_compact_shape(q_rows.shape, dtype))
  scale_bytes = jnp.swapaxes(
      jax.lax.bitcast_convert_type(scales, jnp.uint8), 1, 2
  )
  scale_chunk = jax.lax.bitcast_convert_type(scale_bytes[:, None], dtype)
  compact = jnp.concatenate([payload, scale_chunk], axis=1)
  if valid is not None:
    nan_bytes = jnp.full(compact.shape, 0xFF, dtype=jnp.uint8)
    nan_compact = jax.lax.bitcast_convert_type(nan_bytes, dtype)
    compact = jnp.where(valid[:, None, None, None], compact, nan_compact)
  spec = arg.ActivationSpec(
      row_size=m,
      col_size=cols,
      dtype=dtype,
      quant=arg.QuantSpec(block_size=cols // lanes, stride=lanes),
  )
  return arg.ActivationWrapper(data=compact, scale=None, spec=spec)


@jax.jit
def dequantize_compact(x: jax.Array) -> jax.Array:
  """Dequantizes `[m, chunks + 1, 4, 128]` rows to `f32[m, hidden]`."""
  scale_bytes = jax.lax.bitcast_convert_type(x, jnp.uint8)[:, -1]
  scales = jax.lax.bitcast_convert_type(
      jnp.swapaxes(scale_bytes, 1, 2), jnp.float32
  )
  payload = x[:, :-1]
  m, lanes = scales.shape
  chunks = payload.astype(jnp.float32).reshape(m, -1, lanes)
  return (chunks * scales[:, None, :]).reshape(m, -1)


@dataclasses.dataclass(frozen=True)
class ReduceCase:
  """Inputs of one `output_reduce_unpack` call on the first EP shard.

  Attributes:
    rows: `[num_tokens * top_k, hidden]` gmm2 output rows in `(token, pick)`
      order.
    topk_weights: `[num_tokens, top_k]` routing weights of all shards.
    topk_indices: `[num_tokens, top_k]` expert of each pick.
    num_local_experts: Experts per shard.
    in_dtype: Element dtype of the compact input rows (`bfloat16` or
      `float8_e4m3fn`).
  """

  rows: jax.Array
  topk_weights: jax.Array
  topk_indices: jax.Array
  num_local_experts: int
  in_dtype: jnp.dtype = jnp.bfloat16.dtype

  @property
  def group_offset(self) -> jax.Array:
    """Starting expert index of the first EP shard."""
    return jnp.array([0], dtype=jnp.int32)

  @property
  def local_weights(self) -> jax.Array:
    """`topk_weights` restricted to this shard's picks."""
    return test_reference.local_pick_weights(
        self.topk_weights, self.topk_indices, 0, self.num_local_experts
    )

  @property
  def rows_compact(self) -> arg.ActivationWrapper[jax.Array]:
    """`rows` in the compact layout, NaN where this shard holds no row."""
    valid = self.local_weights.reshape(-1) != 0
    if self.in_dtype.itemsize == 1:
      return quantize_to_compact(self.rows, self.in_dtype, valid=valid)
    rows = jnp.where(valid[:, None], self.rows, jnp.nan)
    return input_pack.input_pack(rows)

  @property
  def rows_clean_compact(self) -> jax.Array:
    """`rows` in the compact layout with no NaN, for the XLA baseline."""
    if self.in_dtype.itemsize == 1:
      return quantize_to_compact(self.rows, self.in_dtype).data
    return input_pack.input_pack(self.rows).data

  @property
  def dequantized_rows(self) -> jax.Array:
    """`rows` after compact quantization/dequantization when `in_dtype` is fp8."""
    if self.in_dtype.itemsize == 1:
      return dequantize_compact(self.rows_clean_compact)
    return self.rows

  def expected(self, out_dtype: jax.typing.DTypeLike) -> jax.Array:
    """Computes the reference weighted top-k sum in `out_dtype`."""
    return test_reference.topk_reduce(
        self.dequantized_rows, self.local_weights, out_dtype
    )

  def assert_close(self, actual: jax.Array) -> None:
    """Asserts that `actual` has no NaNs and matches `expected(actual.dtype)`."""
    assert not bool(jnp.isnan(actual).any())
    expected = self.expected(actual.dtype)
    rtol = 2e-2
    atol = rtol * max(float(jnp.abs(expected).max()), 1e-3)
    test_utils.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  @classmethod
  def make_case(
      cls,
      case: test_cases.Case,
      *,
      in_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      seed: int = 1234,
  ) -> Self:
    """Generates random gmm2 rows and the top-k routing of `case`."""
    shape = case.shape
    k_gate, k_rows = jax.random.split(jax.random.key(seed))
    gating_output = test_inputs.random_gating(
        k_gate,
        case.num_tokens,
        shape.num_experts,
        max_to_mean_load=case.max_to_mean_load,
        dtype=jnp.float32,
    )
    topk_weights, topk_indices = test_inputs.top_k(
        gating_output, shape.top_k, renormalize=shape.renormalize
    )
    rows = jax.random.normal(
        k_rows, (case.num_rows, shape.hidden_size), jnp.bfloat16
    )
    return cls(
        rows=rows / 10,
        topk_weights=topk_weights,
        topk_indices=topk_indices,
        num_local_experts=case.num_local_experts,
        in_dtype=jnp.dtype(in_dtype),
    )
