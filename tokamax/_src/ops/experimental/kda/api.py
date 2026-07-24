# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
from tokamax._src.ops.experimental.kda.cp_utils import CPContext


Implementation: TypeAlias = Literal["xla", "pallas_tpu"]

IMPLEMENTATIONS = dict(xla=base.KimiDeltaAttention())

try:
  from tokamax._src.ops.experimental.kda import pallas_tpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["pallas_tpu"] = pallas_tpu.PallasTpuKimiDeltaAttention()
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: Final[Sequence[Implementation]] = (
    ("pallas_tpu", "xla")
    if "pallas_tpu" in IMPLEMENTATIONS
    else ("xla",)
)


@jaxtyping.jaxtyped
def kimi_delta_attention(
    q: Float[Array, "H B T K"],
    k: Float[Array, "H B T K"],
    v: Float[Array, "H B T V"],
    g: Float[Array, "H B T K"],
    beta: Float[Array, "H B T"],
    *,
    A_log: Float[Array, "H"] | None = None,
    dt_bias: Float[Array, "H*K"] | None = None,
    scale: float | None = None,
    initial_state: Float[Array, "B N H K V"] | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids: Int[Array, "B T"] | None = None,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    disable_recompute: bool = True,
    cp_context: CPContext | None = None,
    chunk_size: int = 64,
    N_max: int | None = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> tuple[Float[Array, "H B T V"], Float[Array, "B N H K V"] | None]:
  """Kimi Delta Attention.

  Kimi Delta Attention is a recurrent linear attention module. The `"xla"`
  implementation evaluates the dense recurrence and serves as the reference
  contract for chunk-wise implementations.

  Args:
    q: Query tensor with shape `[H, B, T, K]`.
    k: Key tensor with shape `[H, B, T, K]`.
    v: Value tensor with shape `[H, B, T, V]`.
    g: Per-channel gate tensor in log space, shape `[H, B, T, K]`.
    beta: Per-token delta-rule learning-rate tensor, shape `[H, B, T]`.
    A_log: Gate parameter, shape `[H]`. Required when
      `use_gate_in_kernel=True`.
    dt_bias: Optional gate bias, shape `[H * K]`.
    scale: Query scale. Defaults to `K ** -0.5`.
    initial_state: Optional initial recurrent state, shape `[B, N, H, K, V]`.
      Its segment dimension `N` determines `N_max` when the latter is omitted.
    output_final_state: Whether to return the final recurrent state.
    use_qk_l2norm_in_kernel: Whether to normalize q/k on the last dimension
      before running KDA.
    use_gate_in_kernel: Whether `g` is raw gate input that should be activated
      with `A_log` and `dt_bias`. When false, `g` is already in log space.
    segment_ids: Optional 1-indexed varlen segment IDs, shape `[B, T]`.
      Padding is represented by 0.
    safe_gate: Match pallas-kernel gate validation.
    lower_bound: Optional sigmoid-gate lower bound.
    disable_recompute: Pallas custom-VJP recompute policy. XLA reference
      implementations accept it but the mathematical result is unchanged.
    cp_context: Optional context-parallel metadata. Construct it with
      `kda.CPContext(mesh, axis_name)`.
    chunk_size: Chunk size used by Pallas.
    N_max: Static upper bound for the number of varlen segments. Required when
      `segment_ids` is provided without `initial_state`; otherwise inferred
      from the initial state's segment dimension.
    implementation: The implementation to use. By default, the Pallas TPU
      implementation is attempted first when available, with XLA as a fallback.
      `"xla"` evaluates the recurrent reference implementation. `"pallas_tpu"`
      uses the experimental Pallas TPU forward and custom VJP implementation
      from pallas-kernel. A sequence tries implementations in order, falling
      back when an implementation raises `NotImplementedError`.

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
    if impl not in IMPLEMENTATIONS:
      raise ValueError(f"Unknown implementation: {impl}")

    try:
      return IMPLEMENTATIONS[impl](
          q=q,
          k=k,
          v=v,
          g=g,
          beta=beta,
          A_log=A_log,
          dt_bias=dt_bias,
          scale=scale,
          initial_state=initial_state,
          output_final_state=output_final_state,
          use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
          use_gate_in_kernel=use_gate_in_kernel,
          segment_ids=segment_ids,
          safe_gate=safe_gate,
          lower_bound=lower_bound,
          disable_recompute=disable_recompute,
          cp_context=cp_context,
          chunk_size=chunk_size,
          N_max=N_max,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
