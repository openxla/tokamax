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
"""Experimental Pallas TPU implementation of Kimi Delta Attention."""

import dataclasses
from typing import Any

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.kda import base
from tokamax._src.ops.experimental.kda.cp_utils import (
    CPContext,
    CPContextArg,
)
from tokamax._src.ops.experimental.kda.pallas_tpu_bwd import (
    chunk_kda_bwd_custom,
)
from tokamax._src.ops.experimental.kda.pallas_tpu_fwd import (
    chunk_kda_fwd_custom,
)
from tokamax._src.ops.experimental.kda.pallas_tpu_types import (
    CpMetadata,
    KdaResiduals,
)
from tokamax._src.ops.experimental.kda.utils import (
    _align_seqs,
    align_segment_ids,
    derive_cp_context,
    l2norm_fwd,
    prepare_chunk_indices,
    segment_ids_to_cu_seqlens,
)
from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class _PreparedKdaInputs:
  q: jax.Array
  k: jax.Array
  v: jax.Array
  g: jax.Array
  beta: jax.Array
  initial_state: jax.Array | None
  cp_context: CPContext | None
  cu_seqlens: jax.Array | None
  aligned_cu_seqlens: jax.Array | None
  chunk_indices: jax.Array | None
  aligned_segment_ids: jax.Array | None
  q_rstd: jax.Array | None
  k_rstd: jax.Array | None
  cp_metadata: CpMetadata


def check_inputs_support(
    q: jax.Array,
    v: jax.Array,
    *,
    initial_state: jax.Array | None,
    output_final_state: bool,
    segment_ids: jax.Array | None,
    cp_context: CPContext | None,
    chunk_size: int,
    N_max: int | None,
) -> None:
  """Checks whether the Pallas TPU backend supports the static inputs."""
  if q.dtype not in (jnp.bfloat16, jnp.float32):
    raise NotImplementedError(
        "`pallas_tpu` currently supports bfloat16 and float32 inputs only."
    )
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  if heads < 1 or batch < 1 or seq_len < 1:
    raise NotImplementedError(
        "`pallas_tpu` requires positive head, batch, and sequence "
        f"dimensions; got H={heads}, B={batch}, T={seq_len}."
    )
  if key_dim < 1 or value_dim < 1:
    raise NotImplementedError(
        "`pallas_tpu` requires positive key and value dimensions; got "
        f"K={key_dim}, V={value_dim}."
    )
  if key_dim > 256:
    raise NotImplementedError(
        "`pallas_tpu` currently supports key dimensions up to 256; got "
        f"K={key_dim}."
    )
  cp_enabled = cp_context is not None and cp_context.is_cp_enabled
  if cp_enabled:
    if initial_state is not None:
      raise NotImplementedError(
          "`pallas_tpu` context-parallel execution does not support "
          "`initial_state`."
      )
    if output_final_state:
      raise NotImplementedError(
          "`pallas_tpu` context-parallel execution does not support "
          "`output_final_state=True`."
      )
    if segment_ids is None:
      raise NotImplementedError(
          "`pallas_tpu` context-parallel execution requires rank-local "
          "`segment_ids`."
      )
    if N_max is None:
      raise NotImplementedError(
          "`pallas_tpu` context-parallel execution requires `N_max`."
      )
    if key_dim % 128 != 0 or value_dim % 128 != 0:
      raise NotImplementedError(
          "`pallas_tpu` context-parallel execution requires key and value "
          "dimensions to be multiples of 128; got "
          f"K={key_dim}, V={value_dim}."
      )
  if initial_state is not None and segment_ids is None:
    if initial_state.shape[1] != 1:
      raise NotImplementedError(
          "`pallas_tpu` fixed-length execution requires exactly one "
          "recurrent state per batch item; got "
          f"N={initial_state.shape[1]}."
      )
  if chunk_size != 64:
    raise NotImplementedError("`pallas_tpu` currently supports chunk_size=64.")
  if segment_ids is None and seq_len % chunk_size != 0:
    raise NotImplementedError(
        "`pallas_tpu` requires the sequence length to be divisible by "
        f"`chunk_size`; got T={seq_len}, chunk_size={chunk_size}."
    )


@dataclasses.dataclass(frozen=True)
class PallasTpuKimiDeltaAttention(base.KimiDeltaAttention):
  """Pallas TPU KDA backend.

  This adapter preserves Tokamax's experimental head-first KDA contract:
  inputs are `[H, B, T, D]` and recurrent states are `[B, N, H, K, V]`.
  """

  chunk_size: int = 64

  def __post_init__(self):
    if self.chunk_size != 64:
      raise ValueError("`pallas_tpu` only supports chunk_size=64.")
    if self.vjp is None:
      object.__setattr__(self, "vjp", PallasTpuKimiDeltaAttentionVjp())

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
      use_qk_l2norm_in_kernel: bool,
      use_gate_in_kernel: bool,
      segment_ids: jax.Array | None,
      cp_context: CPContext | None,
      chunk_size: int,
      N_max: int | None,
  ) -> _PreparedKdaInputs:
    """Canonicalizes inputs shared by the forward and backward kernels."""
    cp_context, cu_seqlens = derive_cp_context(
        segment_ids=segment_ids,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cp_context=cp_context,
        N_max=N_max,
    )
    if cu_seqlens is None:
      cu_seqlens, N_max = segment_ids_to_cu_seqlens(
          segment_ids,
          initial_state=initial_state,
          N_max=N_max,
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
        if initial_state_prepared.shape[1] != state_count:
          raise ValueError(
              "`initial_state` state count must match aligned segment "
              f"count {state_count}; got {initial_state.shape[1]}."
          )

    aligned_segment_ids = None
    if aligned_cu_seqlens is not None and segment_ids is not None:
      effective_n_max = (
          N_max
          if N_max is not None
          else aligned_cu_seqlens.shape[-1] - 1
      )
      aligned_segment_ids = jnp.stack(
          [
              align_segment_ids(
                  segment_ids[batch_index], effective_n_max, chunk_size
              )
              for batch_index in range(segment_ids.shape[0])
          ]
      )

    if use_qk_l2norm_in_kernel:
      q_prepared, q_rstd = l2norm_fwd(q_aligned)
      k_prepared, k_rstd = l2norm_fwd(k_aligned)
    else:
      q_prepared, k_prepared = q_aligned, k_aligned
      q_rstd = k_rstd = None

    cp_metadata = None
    if cp_context is not None and cp_context.is_cp_enabled:
      if any(
          value is None
          for value in (
              cp_context.is_first_rank,
              cp_context.is_last_rank,
              cp_context.pre_num_ranks,
              cp_context.post_num_ranks,
          )
      ):
        raise ValueError("Enabled CP context is missing derived rank metadata.")
      cp_metadata = (
          cp_context.is_first_rank,
          cp_context.is_last_rank,
          cp_context.pre_num_ranks,
          cp_context.post_num_ranks,
      )

    return _PreparedKdaInputs(
        q=q_prepared,
        k=k_prepared,
        v=v_aligned,
        g=g_aligned,
        beta=beta_aligned,
        initial_state=initial_state_prepared,
        cp_context=cp_context,
        cu_seqlens=cu_seqlens,
        aligned_cu_seqlens=aligned_cu_seqlens,
        chunk_indices=chunk_indices,
        aligned_segment_ids=aligned_segment_ids,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        cp_metadata=cp_metadata,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "H B T K"],
      k: Float[Array, "H B T K"],
      v: Float[Array, "H B T V"],
      g: Float[Array, "H B T K"],
      beta: Float[Array, "H B T"],
      *,
      A_log: Float[Array, "H"] | None,
      dt_bias: Float[Array, "H*K"] | None,
      scale: float,
      initial_state: Float[Array, "B N H K V"] | None,
      output_final_state: bool,
      use_qk_l2norm_in_kernel: bool,
      use_gate_in_kernel: bool,
      segment_ids: Int[Array, "B T"] | None,
      safe_gate: bool,
      lower_bound: float | None,
      disable_recompute: bool,
      cp_context: CPContextArg,
      chunk_size: int,
      N_max: int | None,
      return_residuals: bool,
      config: Any,
  ) -> tuple[base.Output, base.Residuals]:
    del config

    # Reject unsupported calls before preprocessing or tracing a Pallas kernel,
    # so API dispatch can fall through to the next implementation.
    check_inputs_support(
        q,
        v,
        initial_state=initial_state,
        output_final_state=output_final_state,
        segment_ids=segment_ids,
        cp_context=cp_context,
        chunk_size=chunk_size,
        N_max=N_max,
    )

    prepared = self._preprocess_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        cp_context=cp_context,
        chunk_size=chunk_size,
        N_max=N_max,
    )

    output, residuals = chunk_kda_fwd_custom(
        prepared.q,
        prepared.k,
        prepared.v,
        prepared.g,
        prepared.beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=prepared.initial_state,
        output_final_state=output_final_state,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=disable_recompute,
        cp_context=prepared.cp_context,
        chunk_size=chunk_size,
        return_residuals=return_residuals,
        cu_seqlens=prepared.cu_seqlens,
        aligned_cu_seqlens=prepared.aligned_cu_seqlens,
        chunk_indices=prepared.chunk_indices,
        aligned_segment_ids=prepared.aligned_segment_ids,
        q_rstd=prepared.q_rstd,
        k_rstd=prepared.k_rstd,
        cp_metadata=prepared.cp_metadata,
    )
    value, final_state = output
    if final_state is not None and final_state.ndim == 4:
      final_state = final_state[:, None]
    return (value, final_state), residuals

@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuKimiDeltaAttentionVjp(
    op.Op[Any, dict[str, Any], None, Any, Any]
):
  """Tokamax Op VJP wrapper for the Pallas TPU KDA backward path."""

  def _fwd(
      self,
      residuals: KdaResiduals,
      out: base.Output,
      dout: base.Output,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      beta: jax.Array,
      *,
      A_log: jax.Array | None,
      dt_bias: jax.Array | None,
      scale: float,
      initial_state: jax.Array | None,
      output_final_state: bool,
      use_qk_l2norm_in_kernel: bool,
      use_gate_in_kernel: bool,
      segment_ids: jax.Array | None,
      safe_gate: bool,
      lower_bound: float | None,
      disable_recompute: bool,
      cp_context: CPContextArg,
      chunk_size: int,
      N_max: int | None,
      return_residuals: bool,
      config: Any,
  ) -> tuple[dict[str, jax.Array], None]:
    # Tokamax's VJP contract replays the original inputs here, but the backward
    # kernel consumes the aligned and optionally L2-normalized copies retained
    # in `residuals`. Reusing these arguments would skip that preprocessing.
    del (
        out,
        q,
        k,
        v,
        g,
        beta,
        output_final_state,
        return_residuals,
        safe_gate,
        config,
    )

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
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        lower_bound,
        disable_recompute,
        cp_context,
        chunk_size,
        N_max,
        initial_state is not None,
        residuals,
        dout,
    )

    grads = {
        "q": dq,
        "k": dk,
        "v": dv,
        "g": dg,
        "beta": db,
    }
    if A_log is not None:
      grads["A_log"] = dA if dA is not None else jnp.zeros_like(A_log)
    if dt_bias is not None:
      grads["dt_bias"] = (
          dbias if dbias is not None else jnp.zeros_like(dt_bias)
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
