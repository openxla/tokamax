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
"""Experimental Pallas/Mosaic TPU implementation of Kimi Delta Attention."""

import dataclasses
from typing import Annotated, Any, ClassVar

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.kda import base
from tokamax._src.ops.experimental.kda.cp_utils import (
    ContextParallelMetadata,
    ContextParallelMetadataArg,
)
from tokamax._src.ops.experimental.kda.pallas_mosaic_tpu_kernel import (
    chunk_kda_bwd_custom,
    chunk_kda_fwd_custom,
)
from tokamax._src.ops.experimental.kda.pallas_mosaic_tpu_types import (
    KdaResiduals,
)
from tokamax._src.ops.experimental.kda.utils import (
    _align_seqs,
    align_segment_ids,
    derive_context_parallel_metadata,
    l2norm_fwd,
    prepare_chunk_indices,
    segment_ids_to_cu_seqlens,
)
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Mosaic TPU KDA.

  `safe_gate=None` selects the exponent-stabilization strategy from the gate
  activation mode. `rematerialize_for_backward=True` omits chunk hidden states
  from forward residuals and manually rebuilds them in the custom backward.
  """

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 64
  safe_gate: bool | None = None
  rematerialize_for_backward: bool = False


def _resolve_safe_gate(
    config: Config,
    *,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
) -> bool:
  """Selects the internal exponent-stabilization strategy."""
  if config.safe_gate is not None:
    return config.safe_gate
  return not use_gate_in_kernel or lower_bound is not None


@dataclasses.dataclass(frozen=True)
class _PreparedKdaInputs:
  q: jax.Array
  k: jax.Array
  v: jax.Array
  g: jax.Array
  beta: jax.Array
  initial_state: jax.Array | None
  context_parallel_metadata: ContextParallelMetadata | None
  cu_seqlens: jax.Array | None
  aligned_cu_seqlens: jax.Array | None
  chunk_indices: jax.Array | None
  aligned_segment_ids: jax.Array | None
  q_rstd: jax.Array | None
  k_rstd: jax.Array | None


def check_inputs_support(
    q: jax.Array,
    v: jax.Array,
    *,
    initial_state: jax.Array | None,
    output_final_state: bool,
    segment_ids: jax.Array | None,
    context_parallel_metadata: ContextParallelMetadata | None,
    chunk_size: int,
    max_num_segments: int | None,
) -> None:
  """Checks whether the Pallas/Mosaic TPU backend supports static inputs."""
  if q.dtype not in (jnp.bfloat16, jnp.float32):
    raise NotImplementedError(
        "`mosaic` currently supports bfloat16 and float32 inputs only."
    )
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  if heads < 1 or batch < 1 or seq_len < 1:
    raise NotImplementedError(
        "`mosaic` requires positive head, batch, and sequence "
        f"dimensions; got H={heads}, B={batch}, T={seq_len}."
    )
  if key_dim < 1 or value_dim < 1:
    raise NotImplementedError(
        "`mosaic` requires positive key and value dimensions; got "
        f"K={key_dim}, V={value_dim}."
    )
  if key_dim > 256:
    raise NotImplementedError(
        "`mosaic` currently supports key dimensions up to 256; got "
        f"K={key_dim}."
    )
  cp_enabled = (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  )
  if cp_enabled:
    if initial_state is not None:
      raise NotImplementedError(
          "`mosaic` context-parallel execution does not support "
          "`initial_state`."
      )
    if output_final_state:
      raise NotImplementedError(
          "`mosaic` context-parallel execution does not support "
          "`output_final_state=True`."
      )
    if segment_ids is None:
      raise NotImplementedError(
          "`mosaic` context-parallel execution requires rank-local "
          "`segment_ids`."
      )
    if max_num_segments is None:
      raise NotImplementedError(
          "`mosaic` context-parallel execution requires `max_num_segments`."
      )
    if key_dim % 128 != 0 or value_dim % 128 != 0:
      raise NotImplementedError(
          "`mosaic` context-parallel execution requires key and value "
          "dimensions to be multiples of 128; got "
          f"K={key_dim}, V={value_dim}."
      )
  if initial_state is not None and segment_ids is None:
    if initial_state.shape[1] != 1:
      raise NotImplementedError(
          "`mosaic` fixed-length execution requires exactly one "
          "recurrent state per batch item; got "
          f"N={initial_state.shape[1]}."
      )
  if chunk_size != 64:
    raise NotImplementedError("`mosaic` currently supports chunk_size=64.")
  if segment_ids is None and seq_len % chunk_size != 0:
    raise NotImplementedError(
        "`mosaic` requires the sequence length to be divisible by "
        f"`chunk_size`; got T={seq_len}, chunk_size={chunk_size}."
    )


@dataclasses.dataclass(frozen=True)
class PallasMosaicTpuKimiDeltaAttention(
    base.KimiDeltaAttention[Config, Any]
):
  """Pallas/Mosaic TPU KDA backend.

  This adapter preserves Tokamax's experimental head-first KDA contract:
  inputs are `[H, B, T, D]` and recurrent states are `[B, N, H, K, V]`.
  """

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", PallasMosaicTpuKimiDeltaAttentionVjp())

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def _get_autotuning_configs(
      self, ba: op.BoundArguments
  ) -> set[Config]:
    del ba
    return {Config(chunk_size=64)}

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6

  @staticmethod
  def _preprocess_inputs(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      beta: jax.Array,
      *,
      initial_state: jax.Array | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      segment_ids: jax.Array | None,
      context_parallel_metadata: ContextParallelMetadata | None,
      chunk_size: int,
      max_num_segments: int | None,
  ) -> _PreparedKdaInputs:
    """Canonicalizes inputs shared by the forward and backward kernels."""
    context_parallel_metadata, cu_seqlens = derive_context_parallel_metadata(
        segment_ids=segment_ids,
        initial_state=initial_state,
        output_final_state=output_final_state,
        context_parallel_metadata=context_parallel_metadata,
        max_num_segments=max_num_segments,
    )
    if cu_seqlens is None:
      cu_seqlens, max_num_segments = segment_ids_to_cu_seqlens(
          segment_ids,
          initial_state=initial_state,
          max_num_segments=max_num_segments,
      )

    aligned_cu_seqlens = None
    chunk_indices = None
    if cu_seqlens is None:
      q_aligned, k_aligned, v_aligned = q, k, v
      g_aligned, beta_aligned = g, beta
    else:
      (
          [q_aligned, k_aligned, v_aligned, g_aligned],
          [beta_aligned],
          aligned_cu_seqlens,
          _,
      ) = _align_seqs(
          [q, k, v, g],
          [beta],
          cu_seqlens,
          align=chunk_size,
      )
      chunk_indices = prepare_chunk_indices(
          aligned_cu_seqlens,
          chunk_size,
          max_T=q_aligned.shape[2],
      )

      if use_gate_in_kernel:
        aligned_seq_len = g_aligned.shape[2]
        original_lengths = jnp.diff(cu_seqlens, axis=-1)
        aligned_starts = aligned_cu_seqlens[..., :-1]
        positions = jnp.arange(aligned_seq_len)
        for batch_index in range(cu_seqlens.shape[0]):
          in_range = (
              positions[None, :]
              >= aligned_starts[batch_index, :, None]
          ) & (
              positions[None, :]
              < (
                  aligned_starts[batch_index]
                  + original_lengths[batch_index]
              )[:, None]
          )
          valid_mask = in_range.any(axis=0)
          g_aligned = g_aligned.at[:, batch_index].set(
              jnp.where(
                  valid_mask[None, :, None],
                  g_aligned[:, batch_index],
                  -1e4,
              )
          )

    initial_state_prepared = initial_state
    if initial_state is not None:
      if aligned_cu_seqlens is None:
        # Fixed-length execution has one recurrent state per batch item.
        if initial_state.ndim == 5:
          initial_state_prepared = initial_state[:, 0]
      else:
        state_count = aligned_cu_seqlens.shape[-1] - 1
        if initial_state.shape[1] < state_count:
          initial_state_prepared = jnp.pad(
              initial_state,
              (
                  (0, 0),
                  (0, state_count - initial_state.shape[1]),
                  (0, 0),
                  (0, 0),
                  (0, 0),
              ),
          )
        assert initial_state_prepared is not None
        if initial_state_prepared.shape[1] != state_count:
          raise ValueError(
              "`initial_state` state count must match aligned segment "
              f"count {state_count}; got {initial_state.shape[1]}."
          )

    aligned_segment_ids = None
    if aligned_cu_seqlens is not None and segment_ids is not None:
      effective_max_num_segments = (
          max_num_segments
          if max_num_segments is not None
          else aligned_cu_seqlens.shape[-1] - 1
      )
      aligned_segment_ids = jnp.stack(
          [
              align_segment_ids(
                  segment_ids[batch_index],
                  effective_max_num_segments,
                  chunk_size,
              )
              for batch_index in range(segment_ids.shape[0])
          ]
      )

    if use_qk_l2norm:
      q_prepared, q_rstd = l2norm_fwd(q_aligned)
      k_prepared, k_rstd = l2norm_fwd(k_aligned)
    else:
      q_prepared, k_prepared = q_aligned, k_aligned
      q_rstd = k_rstd = None

    return _PreparedKdaInputs(
        q=q_prepared,
        k=k_prepared,
        v=v_aligned,
        g=g_aligned,
        beta=beta_aligned,
        initial_state=initial_state_prepared,
        context_parallel_metadata=context_parallel_metadata,
        cu_seqlens=cu_seqlens,
        aligned_cu_seqlens=aligned_cu_seqlens,
        chunk_indices=chunk_indices,
        aligned_segment_ids=aligned_segment_ids,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      query: Float[Array, "H B T K"],
      key: Float[Array, "H B T K"],
      value: Float[Array, "H B T V"],
      gate: Float[Array, "H B T K"],
      beta: Float[Array, "H B T"],
      *,
      a_log: Float[Array, "H"] | None,
      delta_time_bias: Float[Array, "H*K"] | None,
      scale: float,
      initial_state: Float[Array, "B N H K V"] | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      segment_ids: Int[Array, "B T"] | None,
      lower_bound: float | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      max_num_segments: int | None,
      return_residuals: bool,
      config: Config,
  ) -> tuple[base.Output, base.Residuals]:
    chunk_size = config.chunk_size
    safe_gate = _resolve_safe_gate(
        config,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )
    save_intermediates_for_backward = not config.rematerialize_for_backward

    # Reject unsupported calls before preprocessing or tracing a Pallas kernel,
    # so API dispatch can fall through to the next implementation.
    check_inputs_support(
        query,
        value,
        initial_state=initial_state,
        output_final_state=output_final_state,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
        chunk_size=chunk_size,
        max_num_segments=max_num_segments,
    )

    prepared = self._preprocess_inputs(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
        chunk_size=chunk_size,
        max_num_segments=max_num_segments,
    )

    output, residuals = chunk_kda_fwd_custom(
        prepared.q,
        prepared.k,
        prepared.v,
        prepared.g,
        prepared.beta,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        scale=scale,
        initial_state=prepared.initial_state,
        output_final_state=output_final_state,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=save_intermediates_for_backward,
        context_parallel_metadata=prepared.context_parallel_metadata,
        chunk_size=chunk_size,
        return_residuals=return_residuals,
        cu_seqlens=prepared.cu_seqlens,
        aligned_cu_seqlens=prepared.aligned_cu_seqlens,
        chunk_indices=prepared.chunk_indices,
        aligned_segment_ids=prepared.aligned_segment_ids,
        q_rstd=prepared.q_rstd,
        k_rstd=prepared.k_rstd,
    )
    value, final_state = output
    if final_state is not None and final_state.ndim == 4:
      final_state = final_state[:, None]
    return (value, final_state), residuals

@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuKimiDeltaAttentionVjp(
    op.Op[Any, dict[str, Any], None, Config, Any]
):
  """Tokamax Op VJP wrapper for the Pallas/Mosaic TPU KDA backward path."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def _get_autotuning_configs(
      self, ba: op.BoundArguments
  ) -> set[Config]:
    del ba
    return {Config(chunk_size=64)}

  def _fwd(
      self,
      residuals: KdaResiduals,
      out: base.Output,
      dout: base.Output,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      gate: jax.Array,
      beta: jax.Array,
      *,
      a_log: jax.Array | None,
      delta_time_bias: jax.Array | None,
      scale: float,
      initial_state: jax.Array | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      segment_ids: jax.Array | None,
      lower_bound: float | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      max_num_segments: int | None,
      return_residuals: bool,
      config: Config,
  ) -> tuple[dict[str, jax.Array], None]:
    # Tokamax's VJP contract replays the original inputs here, but the backward
    # kernel consumes the aligned and optionally L2-normalized copies retained
    # in `residuals`. Reusing these arguments would skip that preprocessing.
    del out, query, key, value, gate, beta, output_final_state, return_residuals
    chunk_size = config.chunk_size
    # The forward residual set records the selected policy: a retained hidden
    # state means backward can use the saved-state path; otherwise it must
    # rematerialize the forward state recurrence.
    use_saved_state = residuals.h is not None

    (
        dq,
        dk,
        dv,
        dg,
        db,
        dA,
        dbias,
        dh0,
        dsegment_ids,
    ) = chunk_kda_bwd_custom(
        scale,
        use_qk_l2norm,
        use_gate_in_kernel,
        lower_bound,
        use_saved_state,
        context_parallel_metadata,
        chunk_size,
        max_num_segments,
        initial_state is not None,
        residuals,
        dout,
    )

    grads = {
        "query": dq,
        "key": dk,
        "value": dv,
        "gate": dg,
        "beta": db,
    }
    if a_log is not None:
      grads["a_log"] = dA if dA is not None else jnp.zeros_like(a_log)
    if delta_time_bias is not None:
      grads["delta_time_bias"] = (
          dbias if dbias is not None else jnp.zeros_like(delta_time_bias)
      )
    if initial_state is not None:
      grads["initial_state"] = (
          dh0 if dh0 is not None else jnp.zeros_like(initial_state)
      )
    if segment_ids is not None:
      grads["segment_ids"] = (
          dsegment_ids
          if dsegment_ids is not None
          else jnp.zeros_like(segment_ids)
      )
    return grads, None
