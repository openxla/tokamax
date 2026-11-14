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
from tokamax._src.ops.kda import base
from tokamax._src.ops.kda.cp_utils import (
    ContextParallelMetadata,
    ContextParallelMetadataArg,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_kernel import (
    chunk_kda_bwd_custom,
    chunk_kda_fwd_custom,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_types import (
    KdaResiduals,
)
from tokamax._src.ops.kda.utils import (
    _align_seqs,
    align_segment_ids,
    derive_context_parallel_metadata,
    l2norm_fwd,
    prepare_chunk_indices,
    segment_ids_to_cu_seqlens,
)
from typing_extensions import override

# Dimensions.
#   B: batch size
#   H: number of heads (H_QK for query/key heads, H_V for value heads)
#   T: sequence length
#   NT: number of chunks (T // BT)
#   BT: chunk size (64 or 128)
#   N: max number of segments per batch element
#   K: key/query head dimension
#   V: value head dimension
#   GW: gate channel width (1 for scalar gate, K for per-channel gate)
#   W: depthwise conv1d kernel width


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Mosaic TPU KDA.

  `safe_gate=None` selects the exponent-stabilization strategy from the gate
  activation mode. Fused gate activation requires a lower bound and
  `safe_gate=True`; pre-activated gates retain the existing caller-provided
  log-space path without an inferred numerical bound.
  `rematerialize_for_backward=True` omits chunk hidden states from forward
  residuals and manually rebuilds them in the custom backward.
  `fuse_forward=True` keeps the non-CP forward bridge tensors in VMEM for
  128-aligned K/V. Other shapes and CP retain the staged implementation.
  `packed_forward=True` reads non-CP packed input windows directly while
  retaining aligned outputs and backward residuals. It requires fused forward.
  `packed_output=True` uses Pallas output compaction when its full head-group
  buffers fit VMEM; other shapes retain gather. It is independently opt-in.
  `packed_backward=True` reads original packed Q/K/V/beta in the non-CP
  fused reverse pass; aligned gradient outputs and residuals are retained.
  `packed_gradients=True` compacts feature gradients after gate and
  normalization backward; beta retains its existing gather.
  `fuse_backward=True` also fuses the non-CP saved-state backward. Manual
  state rematerialization is fused separately with `fuse_rematerialization=True`;
  `fuse_cp_backward=True` fuses the saved-state local reverse pass after
  CP communication. With `fuse_rematerialization=True`, this also applies
  to CP rematerialization, preserving its staged state reconstruction.
  `cp_megakernel=True` selects the full CP core (including remote DMA)
  instead, for 128-aligned K/V. It is opt-in pending TPU validation.
  """

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 64
  mini_batch: Annotated[int, pydantic.Field(gt=0)] | None = None
  safe_gate: bool | None = None
  rematerialize_for_backward: bool = False
  fuse_forward: bool = False
  # Opt-in until TPU lowering, allocation and device-time validation completes.
  packed_forward: bool = False
  packed_output: bool = False
  packed_backward: bool = False
  packed_gradients: bool = False
  fuse_backward: bool = False
  fuse_cp_backward: bool = False
  cp_megakernel: bool = False
  # Opt-in with the stateless CP workspace adapter. Cross-rank labels remain
  # unchanged; only aligned backward state-slot indices become local ordinals.
  cp_local_state_indices: bool = False
  # Opt-in until TPU compilation and device-time validation completes.
  fuse_rematerialization: bool = False


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


def _check_gate_support(
    *,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    safe_gate: bool,
) -> None:
  """Checks the numerically bounded fused gate configuration."""
  if not use_gate_in_kernel:
    # The caller supplies log-space gates; lower_bound does not constrain them.
    return
  if lower_bound is None:
    raise NotImplementedError("`mosaic` requires `lower_bound` to be set.")
  if not safe_gate:
    raise NotImplementedError("`mosaic` requires `safe_gate=True`.")


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


def _requested_chunk_size(ba) -> int:
  """Returns the requested chunk_size, defaulting to 64 (mandatory for per-channel gates)."""
  args = getattr(ba, "arguments", None) or {}
  requested = args.get("chunk_size")
  if requested is None or args.get("per_channel_gate", True):
    return 64
  return int(requested)


# Maximum fused conv1d width: the forward kernel keeps an 8-row halo of the
# previous chunk (`_CONV_HALO` in the forward kernel).
_MAX_CONV_WIDTH = 8


def _check_chunk_size_support(chunk_size: int, *, per_channel_gate: bool):
  """Checks the chunk sizes supported by the Mosaic TPU kernels."""
  # Per-channel gates require chunk_size=64; scalar gates support powers of 2
  # in [16, 512].
  if per_channel_gate:
    if chunk_size != 64:
      raise NotImplementedError(
          "`mosaic` supports chunk_size=64 with a per-channel gate; got"
          f" {chunk_size}."
      )
  elif chunk_size not in (16, 32, 64, 128, 256, 512):
    raise NotImplementedError(
        "`mosaic` scalar-gate path needs a power-of-two chunk_size in"
        f" [16, 512]; got {chunk_size}."
    )


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
    per_channel_gate: bool = True,
    conv_weights: tuple[jax.Array | None, ...] = (),
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
  _check_chunk_size_support(chunk_size, per_channel_gate=per_channel_gate)
  for conv_weight in conv_weights:
    if conv_weight is None:
      continue
    _, conv_width, _ = conv_weight.shape
    if not 1 <= conv_width <= _MAX_CONV_WIDTH:
      raise NotImplementedError(
          f"`mosaic` supports conv1d widths in [1, {_MAX_CONV_WIDTH}]; got"
          f" W={conv_width}."
      )
  if segment_ids is None and seq_len % chunk_size != 0:
    raise NotImplementedError(
        "`mosaic` requires the sequence length to be divisible by "
        f"`chunk_size`; got T={seq_len}, chunk_size={chunk_size}."
    )


@dataclasses.dataclass(frozen=True)
class PallasMosaicTpuKimiDeltaAttention(base.KimiDeltaAttention[Config, Any]):
  """Pallas/Mosaic TPU KDA backend.

  This adapter preserves Tokamax's experimental head-first KDA contract:
  inputs are `[H, B, T, D]` and recurrent states are `[B, N, H, K, V]`.
  """

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self, "vjp", PallasMosaicTpuKimiDeltaAttentionVjp(config=self.config)
      )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(chunk_size=_requested_chunk_size(ba))

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    args = getattr(ba, "arguments", None)
    if not args:
      return {Config(chunk_size=_requested_chunk_size(ba))}
    q = args.get("query")
    heads = q.shape[0] if q is not None and hasattr(q, "shape") else 32
    per_channel_gate = args.get("per_channel_gate", True)
    segment_ids = args.get("segment_ids")
    chunk_sizes = (
        (64,)
        if (per_channel_gate or segment_ids is not None)
        else (32, 64, 128)
    )
    mini_batches = tuple(
        mb for mb in (8, 16, 32) if mb <= heads and heads % mb == 0
    ) or (None,)
    configs = set()
    for cs in chunk_sizes:
      for mb in mini_batches:
        configs.add(Config(chunk_size=cs, mini_batch=mb))
    return configs

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
      local_state_indices: bool = False,
      use_conv1d_in_kernel: bool = False,
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
      _, _, aligned_seq_len, _ = q_aligned.shape
      chunk_indices = prepare_chunk_indices(
          aligned_cu_seqlens,
          chunk_size,
          max_T=aligned_seq_len,
      )

      if use_gate_in_kernel:
        original_lengths = jnp.diff(cu_seqlens, axis=-1)
        aligned_starts = aligned_cu_seqlens[..., :-1]
        positions = jnp.arange(aligned_seq_len)
        for batch_index in range(cu_seqlens.shape[0]):
          if cu_seqlens.shape[-1] > 8193:
            boundaries = aligned_cu_seqlens[batch_index]
            seq = jnp.minimum(
                jnp.searchsorted(boundaries[1:], positions, side="right"),
                original_lengths.shape[-1] - 1,
            )
            valid_mask = (positions < boundaries[-1]) & (
                positions - boundaries[seq] < original_lengths[batch_index, seq]
            )
          else:
            in_range = (
                positions[None, :] >= aligned_starts[batch_index, :, None]
            ) & (
                positions[None, :]
                < (aligned_starts[batch_index] + original_lengths[batch_index])[
                    :, None
                ]
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

    if local_state_indices and aligned_segment_ids is not None:
      chunk_ids = aligned_segment_ids[:, ::chunk_size]
      starts = jnp.concatenate(
          (
              jnp.ones((chunk_ids.shape[0], 1), jnp.bool_),
              chunk_ids[:, 1:] != chunk_ids[:, :-1],
          ),
          axis=1,
      )
      ordinal = jnp.cumsum(starts & (chunk_ids != 0), axis=1, dtype=jnp.int32)
      aligned_segment_ids = jnp.repeat(
          jnp.where(chunk_ids != 0, ordinal, 0), chunk_size, axis=1
      )

    if use_qk_l2norm and not use_conv1d_in_kernel:
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
      gate: Float[Array, "H B T GW"],
      beta: Float[Array, "H B T"],
      *,
      a_log: Float[Array, "H"] | None,
      delta_time_bias: Float[Array, "H*K"] | None,
      scale: float,
      initial_state: Float[Array, "B N H K V"] | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      per_channel_gate: bool,
      chunk_size: int | None,
      segment_ids: Int[Array, "B T"] | None,
      lower_bound: float | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      max_num_segments: int | None,
      return_residuals: bool,
      conv_weight_q: Float[Array, "H_Q W K"] | None = None,
      conv_weight_k: Float[Array, "H_K W K"] | None = None,
      conv_weight_v: Float[Array, "H W V"] | None = None,
      conv_bias_q: Float[Array, "H_Q K"] | None = None,
      conv_bias_k: Float[Array, "H_K K"] | None = None,
      conv_bias_v: Float[Array, "H V"] | None = None,
      use_conv1d_in_kernel: bool = False,
      config: Config,
  ) -> tuple[base.Output, base.Residuals]:
    if chunk_size is not None:
      # `config.chunk_size` is pinned to 64 for per-channel gates (see
      # `_requested_chunk_size`), so the requested value is checked as well.
      _check_chunk_size_support(chunk_size, per_channel_gate=per_channel_gate)
    chunk_size = config.chunk_size
    safe_gate = _resolve_safe_gate(
        config,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )
    _check_gate_support(
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        safe_gate=safe_gate,
    )
    save_intermediates_for_backward = not config.rematerialize_for_backward

    if config.cp_local_state_indices and not (
        context_parallel_metadata is not None
        and context_parallel_metadata.is_cp_enabled
    ):
      raise NotImplementedError(
          "Local state indices require stateless CP training"
      )

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
        per_channel_gate=per_channel_gate,
        conv_weights=(
            (conv_weight_q, conv_weight_k, conv_weight_v)
            if use_conv1d_in_kernel
            else ()
        ),
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
        local_state_indices=config.cp_local_state_indices,
        use_conv1d_in_kernel=use_conv1d_in_kernel,
    )

    packed_inputs = None
    if (
        config.packed_forward
        and config.fuse_forward
        and prepared.cu_seqlens is not None
        and not (
            prepared.context_parallel_metadata is not None
            and prepared.context_parallel_metadata.is_cp_enabled
        )
        and query.shape[-1] % 128 == 0
        and value.shape[-1] % 128 == 0
    ):
      packed_q = l2norm_fwd(query)[0] if use_qk_l2norm else query
      packed_k = l2norm_fwd(key)[0] if use_qk_l2norm else key
      packed_inputs = (packed_q, packed_k, value, gate, beta)

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
        per_channel_gate=per_channel_gate,
        segment_ids=segment_ids,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=save_intermediates_for_backward,
        fuse_forward=config.fuse_forward,
        packed_inputs=packed_inputs,
        packed_output=config.packed_output,
        context_parallel_metadata=prepared.context_parallel_metadata,
        chunk_size=chunk_size,
        return_residuals=return_residuals,
        cu_seqlens=prepared.cu_seqlens,
        aligned_cu_seqlens=prepared.aligned_cu_seqlens,
        chunk_indices=prepared.chunk_indices,
        aligned_segment_ids=prepared.aligned_segment_ids,
        q_rstd=prepared.q_rstd,
        k_rstd=prepared.k_rstd,
        conv_weight_q=conv_weight_q,
        conv_weight_k=conv_weight_k,
        conv_weight_v=conv_weight_v,
        conv_bias_q=conv_bias_q,
        conv_bias_k=conv_bias_k,
        conv_bias_v=conv_bias_v,
        use_conv1d_in_kernel=use_conv1d_in_kernel,
        use_qk_l2norm=use_qk_l2norm,
        mini_batch=config.mini_batch,
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
    return Config(chunk_size=_requested_chunk_size(ba))

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    args = getattr(ba, "arguments", None)
    if not args:
      return {Config(chunk_size=_requested_chunk_size(ba))}
    q = args.get("query")
    heads = q.shape[0] if q is not None and hasattr(q, "shape") else 32
    per_channel_gate = args.get("per_channel_gate", True)
    segment_ids = args.get("segment_ids")
    chunk_sizes = (
        (64,)
        if (per_channel_gate or segment_ids is not None)
        else (32, 64, 128)
    )
    mini_batches = tuple(
        mb for mb in (8, 16, 32) if mb <= heads and heads % mb == 0
    ) or (None,)
    configs = set()
    for cs in chunk_sizes:
      for mb in mini_batches:
        configs.add(Config(chunk_size=cs, mini_batch=mb))
    return configs

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
      per_channel_gate: bool,
      chunk_size: int | None,
      segment_ids: jax.Array | None,
      lower_bound: float | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      max_num_segments: int | None,
      return_residuals: bool,
      conv_weight_q: jax.Array | None = None,
      conv_weight_k: jax.Array | None = None,
      conv_weight_v: jax.Array | None = None,
      conv_bias_q: jax.Array | None = None,
      conv_bias_k: jax.Array | None = None,
      conv_bias_v: jax.Array | None = None,
      use_conv1d_in_kernel: bool = False,
      config: Config,
  ) -> tuple[dict[str, jax.Array], None]:
    # Packed loading normalizes the replayed original Q/K explicitly. The
    # aligned residuals still supply gate prefixes and normalization backward.
    del out, gate, output_final_state, return_residuals
    del use_conv1d_in_kernel
    chunk_size = config.chunk_size
    safe_gate = _resolve_safe_gate(
        config,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )
    _check_gate_support(
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        safe_gate=safe_gate,
    )
    # The forward residual set records the selected policy: a retained hidden
    # state means backward can use the saved-state path; otherwise it must
    # rematerialize the forward state recurrence.
    use_saved_state = residuals.h is not None

    packed_inputs = None
    if (
        config.packed_backward
        and config.fuse_backward
        and (use_saved_state or config.fuse_rematerialization)
        and residuals.cu_seqlens is not None
        and not (
            context_parallel_metadata is not None
            and context_parallel_metadata.is_cp_enabled
        )
        and query.shape[-1] % 128 == 0
        and value.shape[-1] % 128 == 0
    ):
      packed_q = l2norm_fwd(query)[0] if use_qk_l2norm else query
      packed_k = l2norm_fwd(key)[0] if use_qk_l2norm else key
      packed_inputs = (packed_q, packed_k, value, beta)

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
        dw_q,
        dw_k,
        dw_v,
        db_q,
        db_k,
        db_v,
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
        fuse_backward=config.fuse_backward,
        fuse_rematerialization=config.fuse_rematerialization,
        packed_inputs=packed_inputs,
        packed_gradients=config.packed_gradients,
        fuse_cp_backward=config.fuse_cp_backward,
        cp_megakernel=config.cp_megakernel,
        per_channel_gate=per_channel_gate,
        mini_batch=config.mini_batch,
    )

    grads = {
        "query": dq,
        "key": dk,
        "value": dv,
        "gate": dg,
        "beta": db,
    }
    if conv_weight_q is not None:
      grads["conv_weight_q"] = (
          dw_q if dw_q is not None else jnp.zeros_like(conv_weight_q)
      )
    if conv_weight_k is not None:
      grads["conv_weight_k"] = (
          dw_k if dw_k is not None else jnp.zeros_like(conv_weight_k)
      )
    if conv_weight_v is not None:
      grads["conv_weight_v"] = (
          dw_v if dw_v is not None else jnp.zeros_like(conv_weight_v)
      )
    if conv_bias_q is not None:
      grads["conv_bias_q"] = (
          db_q if db_q is not None else jnp.zeros_like(conv_bias_q)
      )
    if conv_bias_k is not None:
      grads["conv_bias_k"] = (
          db_k if db_k is not None else jnp.zeros_like(conv_bias_k)
      )
    if conv_bias_v is not None:
      grads["conv_bias_v"] = (
          db_v if db_v is not None else jnp.zeros_like(conv_bias_v)
      )
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
