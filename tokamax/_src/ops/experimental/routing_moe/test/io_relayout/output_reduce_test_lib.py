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
from typing import Self

import jax
import jax.numpy as jnp
from tokamax._src import test_utils
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_reference
from tokamax._src.ops.experimental.routing_moe.test.common import test_util


@dataclasses.dataclass(frozen=True)
class ReduceCase:
  """Inputs of one `output_reduce_unpack` call on the first EP shard.

  Attributes:
    rows: `[num_tokens * top_k, hidden]` gmm2 output rows in `(token, pick)`
      order.
    topk_weights: `[num_tokens, top_k]` routing weights of all shards.
    topk_indices: `[num_tokens, top_k]` expert of each pick.
    num_local_experts: Experts per shard.
  """

  rows: jax.Array
  topk_weights: jax.Array
  topk_indices: jax.Array
  num_local_experts: int

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
  def rows_compact(self) -> jax.Array:
    """`rows` in the compact layout, NaN where this shard holds no row."""
    valid = self.local_weights.reshape(-1)[:, None] != 0
    rows = jnp.where(valid, self.rows, jnp.nan)
    return rows.reshape(layout_util.to_compact_shape(rows.shape, rows.dtype))

  @property
  def rows_clean_compact(self) -> jax.Array:
    """`rows` in the compact layout with no NaN, for the XLA baseline."""
    return self.rows.reshape(
        layout_util.to_compact_shape(self.rows.shape, self.rows.dtype)
    )

  def expected(self, out_dtype: jax.typing.DTypeLike) -> jax.Array:
    """Computes the reference weighted top-k sum in `out_dtype`."""
    return test_reference.topk_reduce(self.rows, self.local_weights, out_dtype)

  def assert_close(self, actual: jax.Array) -> None:
    """Asserts that `actual` has no NaNs and matches `expected(actual.dtype)`."""
    expected = self.expected(actual.dtype)
    assert not bool(jnp.isnan(actual).any())
    rtol = 2e-2
    atol = rtol * max(float(jnp.abs(expected).max()), 1e-3)
    test_utils.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  @classmethod
  def make_case(cls, case: test_cases.Case, *, seed: int = 1234) -> Self:
    """Generates random gmm2 rows and the top-k routing of `case`."""
    test_util.check_output_reduce_supported(case)
    shape = case.shape
    k_gate, k_rows = jax.random.split(jax.random.key(seed))
    gating_output = test_inputs.random_gating(
        k_gate,
        case.num_tokens,
        shape.num_experts,
        active_expert_ratio=case.active_expert_ratio,
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
    )
