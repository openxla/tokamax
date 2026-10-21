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
"""Fused MoE Decode API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe_decode import base

type Implementation = Literal["xla", "mosaic_tpu"]

_IMPLEMENTATIONS: dict[str, base.FusedMoeDecode] = dict(
    xla=base.FusedMoeDecode()
)
_DEFAULT_IMPLEMENTATIONS = ("xla",)  # pylint: disable=invalid-name

try:
  from tokamax._src.ops.experimental.fused_moe_decode import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasMosaicTpuFusedMoeDecode()
  )
  _DEFAULT_IMPLEMENTATIONS = ("mosaic_tpu",) + _DEFAULT_IMPLEMENTATIONS  # pylint: disable=invalid-name
except ImportError:
  pallas_mosaic_tpu = None

IMPLEMENTATIONS: Final[
    immutabledict.immutabledict[str, base.FusedMoeDecode]
] = immutabledict.immutabledict(_IMPLEMENTATIONS)
del _IMPLEMENTATIONS


def fused_moe_decode(
    rank: jax.Array,
    hidden: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    shared_w1: jax.Array,
    shared_w1_scale: jax.Array,
    shared_w2: jax.Array,
    shared_w2_scale: jax.Array,
    shared_gate_weight: jax.Array | None,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    *,
    n_ranks: int = 1,
    expert_start: jax.Array | int | None = None,
    scatter_rows: bool = False,
    routed_scale: float = 1.0,
    axis_name: str | None = None,
    i_block: int | None = None,
    implementation: (
        Implementation | Sequence[Implementation | Callable[..., Any]] | None
    ) = None,
) -> jax.Array:
  """Runs the fused MoE decode operator (routed + shared expert + all-reduce).

  Args:
    rank: `[1]` int32 array holding this rank's mesh index.
    hidden: `[tokens, H]` BF16 hidden states.
    w1: `[E, H, 2 * i]` gate and up projections for this rank's `E` experts.
    w2: `[E, i, H]` down projections for this rank's `E` experts.
    w1_scale: `[E, 1, 1, 2 * i]` FP32 per-output-channel scales of `w1`.
    w2_scale: `[E, 1, 1, H]` FP32 per-output-channel scales of `w2`.
    shared_w1: `[H, 2 * i_s]` shared expert gate and up slice.
    shared_w1_scale: `[1, 2 * i_s]` FP32 scales of `shared_w1`.
    shared_w2: `[i_s, H]` shared expert down slice.
    shared_w2_scale: `[1, H]` FP32 scales of `shared_w2`.
    shared_gate_weight: Optional `[H, 1]` replicated `shared_expert_gate` weight
      (scales shared expert by `sigmoid(hidden @ shared_gate_weight)`); `None`
      for an ungated shared expert.
    topk_weights: `[tokens, k]` router weights.
    topk_ids: `[tokens, k]` int32 routed expert indices.
    n_ranks: Number of ranks in the mesh.
    expert_start: Global id of this rank's first expert under EP; `None` for TP.
    scatter_rows: If `True` (DP attention), performs all-to-all + local sum over
      token rows and returns `[tokens // n_ranks, H]`.
    routed_scale: Multiplicative scale applied to routed experts only.
    axis_name: Optional mesh axis name for cross-rank reduction (`"d"` by
      default).
    i_block: Optional intermediate block width forcing blocked streaming mode on
      the Mosaic TPU backend.
    implementation: The implementation to use (`"mosaic_tpu"`, `"xla"`, a
      sequence of implementations, or `None` to select the best available).

  Returns:
    The all-reduced `[tokens, H]` (or `[tokens // n_ranks, H]` when
    `scatter_rows=True`) BF16 MoE output.

  Raises:
    ExceptionGroup: If multiple implementations are tried and all raise
      `NotImplementedError`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  if isinstance(expert_start, int):
    expert_start = jnp.asarray(expert_start, dtype=jnp.int32)

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      impl = IMPLEMENTATIONS[impl]

    if (
        i_block is not None
        and pallas_mosaic_tpu is not None
        and isinstance(impl, pallas_mosaic_tpu.PallasMosaicTpuFusedMoeDecode)
    ):
      impl = impl.replace(config=pallas_mosaic_tpu.Config(i_block=i_block))

    try:
      return impl(
          rank,
          hidden,
          w1,
          w2,
          w1_scale,
          w2_scale,
          shared_w1,
          shared_w1_scale,
          shared_w2,
          shared_w2_scale,
          shared_gate_weight,
          topk_weights,
          topk_ids,
          n_ranks=n_ranks,
          expert_start=expert_start,
          scatter_rows=scatter_rows,
          routed_scale=routed_scale,
          axis_name=axis_name,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
