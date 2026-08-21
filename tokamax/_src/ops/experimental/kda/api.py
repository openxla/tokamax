# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Experimental Kimi Delta Attention API."""

from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import base
from tokamax._src.ops.experimental.kda.cp_utils import ContextParallelMetadata


Implementation: TypeAlias = Literal["xla", "mosaic"]

IMPLEMENTATIONS = dict(xla=base.KimiDeltaAttention())

try:
  from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention()
  )
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = (
    ("mosaic", "xla") if "mosaic_tpu" in IMPLEMENTATIONS else ("xla",)
)


@jaxtyping.jaxtyped
def kimi_delta_attention(
    query: Float[Array, "H B T K"],
    key: Float[Array, "H B T K"],
    value: Float[Array, "H B T V"],
    gate: Float[Array, "H B T K"],
    beta: Float[Array, "H B T"],
    *,
    a_log: Float[Array, "H"] | None = None,
    delta_time_bias: Float[Array, "H*K"] | None = None,
    scale: float | None = None,
    initial_state: Float[Array, "B N H K V"] | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids: Int[Array, "B T"] | None = None,
    lower_bound: float | None = None,
    context_parallel_metadata: ContextParallelMetadata | None = None,
    max_num_segments: int | None = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> tuple[Float[Array, "H B T V"], Float[Array, "B N H K V"] | None]:
  """Kimi Delta Attention.

  Kimi Delta Attention is a recurrent linear attention module. The `"xla"`
  implementation evaluates the dense recurrence and serves as the reference
  contract for chunk-wise implementations.

  Args:
    query: Query tensor with shape `[H, B, T, K]`.
    key: Key tensor with shape `[H, B, T, K]`.
    value: Value tensor with shape `[H, B, T, V]`.
    gate: Per-channel gate tensor with shape `[H, B, T, K]`. When
      `use_gate_in_kernel=True`, this is the raw delta-time input; otherwise,
      it is already the log-space decay.
    beta: Post-activation per-token delta-rule learning-rate tensor with values
      in `[0, 1]`, shape `[H, B, T]`.
    a_log: Per-head log decay-rate parameter, shape `[H]`. Required when
      `use_gate_in_kernel=True`.
    delta_time_bias: Optional per-head, per-key-channel bias added to the raw
      gate before the delta-time activation, shape `[H * K]`. In the standard
      activation path, KDA computes
      `delta_time = softplus(gate + delta_time_bias)` and
      `log_decay = -exp(a_log) * delta_time`. This argument is used only when
      `use_gate_in_kernel=True`.
    scale: Query scale. Defaults to `K ** -0.5`.
    initial_state: Optional initial recurrent state, shape `[B, N, H, K, V]`.
      Its segment dimension `N` determines `max_num_segments` when the latter
      is omitted.
    output_final_state: Whether to return the final recurrent state.
    use_qk_l2norm: Whether to normalize query/key on the last dimension before
      running KDA.
    use_gate_in_kernel: Whether `gate` is raw input that should be activated
      with `a_log` and `delta_time_bias`. When false, `gate` is already in log
      space.
    segment_ids: Optional 1-indexed varlen segment IDs, shape `[B, T]`.
      Padding is represented by 0.
    lower_bound: Optional sigmoid-gate lower bound.
    context_parallel_metadata: Optional context-parallel metadata. Construct
      it with `kda.ContextParallelMetadata(mesh, axis_name)`.
    max_num_segments: Static upper bound for the number of varlen segments.
      Required when `segment_ids` is provided without `initial_state`;
      otherwise inferred from the initial state's segment dimension.
    implementation: The implementation to use. By default, the Mosaic TPU
      implementation is attempted first when available, with XLA as a fallback.
      `"xla"` evaluates the recurrent reference implementation. `"mosaic"`
      uses the experimental Pallas/Mosaic TPU forward and custom VJP
      implementation. A sequence tries implementations in order, falling back
      when an implementation raises `NotImplementedError`.

  Returns:
    A pair `(output, final_state)`. The output has shape `[H, B, T, V]`.
    The final state has shape `[B, N, H, K, V]` when requested, otherwise
    `None`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  errors = []
  for impl in implementation:
    if impl == "mosaic":
      impl = "mosaic_tpu"
    if impl not in IMPLEMENTATIONS:
      raise ValueError(f"Unknown implementation: {impl}")

    try:
      return IMPLEMENTATIONS[impl](
          query=query,
          key=key,
          value=value,
          gate=gate,
          beta=beta,
          a_log=a_log,
          delta_time_bias=delta_time_bias,
          scale=scale,
          initial_state=initial_state,
          output_final_state=output_final_state,
          use_qk_l2norm=use_qk_l2norm,
          use_gate_in_kernel=use_gate_in_kernel,
          segment_ids=segment_ids,
          lower_bound=lower_bound,
          context_parallel_metadata=context_parallel_metadata,
          max_num_segments=max_num_segments,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
