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
from jax import ad_checkpoint
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import shape
from tokamax._src.ops import op
from tokamax._src.ops.kda import base
from tokamax._src.ops.kda import common
from tokamax._src.ops.kda import packing
from tokamax._src.ops.kda import compact
from tokamax._src.ops.kda.cp_utils import (
    ContextParallelMetadata,
    ContextParallelMetadataArg,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_kernel import (
    chunk_gdn_bwd_custom,
    chunk_gdn_fwd_custom,
    chunk_kda_bwd_custom,
    chunk_kda_fwd_custom,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_types import (
    GdnResiduals,
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
  `two_pass_bf16=True` refines the BF16 triangular inverse with one
  Newton-Schulz residual step and keeps `(u, w, kg, Aqk, Akk)` and saved
  per-chunk `h` (plus KDA `v_new`) in `float32` (doubling saved `h` residual
  memory unless `rematerialize_for_backward=True`), while returning outputs in
  the input activation dtype.
  `fuse_forward=True` keeps the non-CP forward bridge tensors in VMEM for
  128-aligned K/V. Other shapes and CP retain the staged implementation.
  `packed_forward=True` reads non-CP packed input windows directly for
  forward-only calls. Training reuses the aligned inputs retained for backward
  instead of preparing both layouts. It requires fused forward.
  `packed_output=True` uses Pallas output compaction when its full head-group
  buffers fit VMEM; other shapes retain gather. It is independently opt-in.
  `compact_forward=True` selects the original-token grid for supported non-CP
  BF16 saved-state profiles. The paired compact backward consumes its tape.
  `compact_sorted_segments=True` declares grouped labels and trailing padding,
  omitting the general input permutation and its worst-case temporary buffers.
  """

  # Opt into packing correction before aligned Mosaic execution. The inner
  # call disables this after preparing the operands.
  canonicalize_packed_inputs: bool = False
  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 64
  mini_batch: Annotated[int, pydantic.Field(gt=0)] | None = None
  safe_gate: bool | None = None
  rematerialize_for_backward: bool = False
  residual_checkpoint_name: str | None = None
  two_pass_bf16: bool = False
  fuse_forward: bool = False
  # Opt-in until TPU lowering, allocation and device-time validation completes.
  packed_forward: bool = False
  packed_output: bool = False
  # Original-token grid; the dependent backward increment consumes its tape.
  compact_forward: bool = False
  # Static packed-input contract avoids allocating an interleaved-label fallback.
  compact_sorted_segments: bool = False


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
    per_channel_gate: bool = True,
) -> None:
  """Checks the numerically bounded fused gate configuration."""
  if not use_gate_in_kernel or not per_channel_gate:
    # Pre-activated log-space gates and scalar gates (`-exp(a_log) * softplus <= 0`)
    # do not require a sigmoid lower_bound.
    return
  if lower_bound is None:
    raise NotImplementedError("`mosaic` requires `lower_bound` to be set.")
  if not safe_gate:
    raise NotImplementedError("`mosaic` requires `safe_gate=True`.")


def _tag_array(arr: jax.Array | None, name: str) -> jax.Array | None:
  return ad_checkpoint.checkpoint_name(arr, name=name) if arr is not None else None


def _tag_kda_residuals(
    output: base.Output,
    residuals: KdaResiduals | None,
    ckpt_name: str,
    *,
    tag_qkv: bool,
    tag_initial_state: bool = False,
) -> tuple[base.Output, KdaResiduals | None]:
  """Tags KDA outputs and computed forward residuals with `checkpoint_name`."""
  val, final_state = output
  output = (
      ad_checkpoint.checkpoint_name(val, name=ckpt_name),
      _tag_array(final_state, ckpt_name),
  )
  if residuals is None:
    return output, None
  residuals = dataclasses.replace(
      residuals,
      q=(
          ad_checkpoint.checkpoint_name(residuals.q, name=ckpt_name)
          if tag_qkv
          else residuals.q
      ),
      k=(
          ad_checkpoint.checkpoint_name(residuals.k, name=ckpt_name)
          if tag_qkv
          else residuals.k
      ),
      v=(
          ad_checkpoint.checkpoint_name(residuals.v, name=ckpt_name)
          if tag_qkv
          else residuals.v
      ),
      g_cumsum=_tag_array(residuals.g_cumsum, ckpt_name),
      aqk=ad_checkpoint.checkpoint_name(residuals.aqk, name=ckpt_name),
      akk=_tag_array(residuals.akk, ckpt_name),
      h=_tag_array(residuals.h, ckpt_name),
      v_new=_tag_array(residuals.v_new, ckpt_name),
      q_rstd=_tag_array(residuals.q_rstd, ckpt_name),
      k_rstd=_tag_array(residuals.k_rstd, ckpt_name),
      initial_state=(
          _tag_array(residuals.initial_state, ckpt_name)
          if tag_initial_state
          else residuals.initial_state
      ),
  )
  return output, residuals


def _tag_gdn_residuals(
    output: base.Output,
    residuals: GdnResiduals | None,
    ckpt_name: str,
    *,
    tag_qkv_conv: bool,
    tag_beta: bool,
) -> tuple[base.Output, GdnResiduals | None]:
  """Tags GDN outputs and computed forward residuals with `checkpoint_name`."""
  val, final_state = output
  output = (
      ad_checkpoint.checkpoint_name(val, name=ckpt_name),
      _tag_array(final_state, ckpt_name),
  )
  if residuals is None:
    return output, None
  residuals = dataclasses.replace(
      residuals,
      qkv_conv=(
          _tag_array(residuals.qkv_conv, ckpt_name)
          if tag_qkv_conv
          else residuals.qkv_conv
      ),
      beta=(
          ad_checkpoint.checkpoint_name(residuals.beta, name=ckpt_name)
          if tag_beta
          else residuals.beta
      ),
      g_cumsum=_tag_array(residuals.g_cumsum, ckpt_name),
      akk=_tag_array(residuals.akk, ckpt_name),
      h=_tag_array(residuals.h, ckpt_name),
  )
  return output, residuals


def _call_with_selective_checkpoint_vjp(
    op_self: Any, *args: Any, **kwargs: Any
) -> base.Output:
  """Calls `op_self` with a direct `custom_vjp` visible to `jax.checkpoint`."""
  if not op_self.supports_symbolic_shapes and shape.contains_symbolic_shape(
      (args, kwargs)
  ):
    raise NotImplementedError("This op does not support symbolic shapes.")
  ba = op_self.bind(*args, return_residuals=False, **kwargs)
  if not (op_self.bypass_device_check or op.config_lib.cross_compile.value):
    devices = op.infer_devices(ba) or {op.backend.get_default_device()}
    for device in devices:
      if not op_self.supported_on(device):
        raise NotImplementedError(f"Not supported on {device.device_kind}.")
  config = ba.default_config
  args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs))
  is_array = lambda x: isinstance(x, (jax.Array, np.ndarray))
  array_idxs = {i for i, x in enumerate(args_flat) if is_array(x)}
  arrays = [x for i, x in enumerate(args_flat) if i in array_idxs]
  other = [x for i, x in enumerate(args_flat) if i not in array_idxs]

  def unflatten_inputs(arrs: tuple[jax.Array, ...]):
    arr_it = iter(arrs)
    oth_it = iter(other)
    merged = [
        next(arr_it) if i in array_idxs else next(oth_it)
        for i in range(len(args_flat))
    ]
    return args_tree.unflatten(merged)

  def _metadata_json(call_args: Any, call_kwargs: dict[str, Any]) -> str:
    ba_meta = op_self.bind(*call_args, **call_kwargs)
    json_op = op_self.replace(config=config, vjp=None)
    json_ba = op.BoundArguments(
        json_op,
        op._abstractify(dict(ba_meta.arguments)),  # pylint: disable=protected-access
    )
    return str(op.BOUND_ARGS_ADAPTER.dump_json(json_ba), "utf-8")

  def f(*arrs: jax.Array) -> base.Output:
    call_args, call_kwargs = unflatten_inputs(arrs)
    call_kwargs["return_residuals"] = False
    with op._tokamax_metadata(_metadata_json(call_args, call_kwargs)):  # pylint: disable=protected-access
      return op_self._fwd(*call_args, config=config, **call_kwargs)[0]  # pylint: disable=protected-access

  def fwd(*arrs: jax.Array):
    call_args, call_kwargs = unflatten_inputs(arrs)
    call_kwargs["return_residuals"] = True
    with op._tokamax_metadata(_metadata_json(call_args, call_kwargs)):  # pylint: disable=protected-access
      out, res = op_self._fwd(*call_args, config=config, **call_kwargs)  # pylint: disable=protected-access
    return out, op.Residuals(call_args, call_kwargs, out, res)

  def bwd(full_res: op.Residuals, dout: base.Output):
    call_args = full_res.args
    call_kwargs = dict(full_res.kwargs)
    call_kwargs.pop("return_residuals", None)
    grads = op_self.vjp(
        full_res.residuals, full_res.out, dout, *call_args, **call_kwargs
    )
    grads_ba = ba.signature.bind_partial(**grads)
    in_args = ba.signature.bind_partial(*call_args, **call_kwargs).arguments
    for k_name, v_val in in_args.items():
      if (dv := grads_ba.arguments.get(k_name)) is None:
        zeros_like = lambda x: jnp.zeros_like(x) if is_array(x) else x
        grads_ba.arguments[k_name] = jax.tree.map(zeros_like, v_val)
      else:
        grads_ba.arguments[k_name] = jax.tree.map(lambda _, g: g, v_val, dv)
    dargs = jax.tree.map(lambda _, g: g, call_args, grads_ba.args)
    return tuple(
        x for x in jax.tree.leaves((dargs, grads_ba.kwargs)) if is_array(x)
    )

  f_vjp = jax.custom_vjp(f)
  f_vjp.defvjp(fwd, bwd, optimize_remat=False)
  return f_vjp(*arrays)


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
  segment_ids: jax.Array | None = None
  packed_q: jax.Array | None = None
  packed_k: jax.Array | None = None


def _requested_chunk_size(ba) -> int:
  """Returns the requested chunk_size, defaulting to 64."""
  args = getattr(ba, "arguments", None) or {}
  requested = args.get("chunk_size")
  return 64 if requested is None else int(requested)


def _vjp_chunk_size(ba: op.BoundArguments | None, fallback_fn) -> int:
  args = getattr(ba, "arguments", None) or {}
  residuals = args.get("residuals")
  if residuals is not None and hasattr(residuals, "chunk_size"):
    return int(residuals.chunk_size)
  return fallback_fn(ba)


def _vjp_two_pass_bf16(ba: op.BoundArguments | None) -> bool:
  args = getattr(ba, "arguments", None) or {}
  residuals = args.get("residuals")
  if residuals is not None and hasattr(residuals, "two_pass_bf16"):
    return bool(residuals.two_pass_bf16)
  return False


def _estimate_per_head_vmem_bytes(
    *,
    act_dtype: Any,
    chunk_size: int,
    head_k_dim: int,
    head_v_dim: int,
    per_channel_gate: bool = False,
    is_kda: bool = False,
    two_pass_bf16: bool = False,
) -> int:
  """Estimates max(forward intra, backward M4) per-head VMEM footprint."""
  elem_size = jnp.dtype(act_dtype).itemsize
  wy_elem = 4 if (two_pass_bf16 and elem_size == 2) else elem_size
  bt, k, v = chunk_size, head_k_dim, head_v_dim
  gw = k if per_channel_gate else 1
  store_akk = is_kda or bt < 128
  if bt <= 64:
    per_head_intra = (bt * k + bt * v + 2 * bt * bt) * wy_elem
  else:
    akk_elems = bt * bt if store_akk else 0
    io_intra = 2 * (
        (2 * bt * k + bt * v) * elem_size
        + (2 * bt * k + bt * v + bt * bt + akk_elems) * wy_elem
        + bt * gw * 4
    )
    scratch_bt2 = 7 if per_channel_gate else 5
    scratch_intra = (
        3 * bt * k + 2 * bt * (v + k) + bt * gw + scratch_bt2 * bt * bt
    ) * 4
    per_head_intra = io_intra + scratch_intra

  has_a = store_akk
  a_elem = wy_elem if has_a else 0
  da_elem = 4 if is_kda else 0
  vn_dv0_bytes = bt * v * (wy_elem + 4) if is_kda else 0
  offdiag_pairs = (
      (bt // 16) * (bt // 16 - 1) // 2 if (per_channel_gate and bt > 64) else 0
  )
  dqkv_elem = 4 if is_kda else elem_size
  io_bwd = (
      (2 * bt * k + 2 * bt * v + bt * gw + bt) * elem_size
      + (3 * bt * k * wy_elem if is_kda else 0)
      + vn_dv0_bytes
      + k * v * wy_elem
      + bt * bt * (a_elem + da_elem)
      + (2 * bt * k + bt * v) * dqkv_elem
      + (
          k * v
          + 5 * bt * k
          + bt * v
          + bt * gw
          + bt
          + (5 if not has_a else 2) * bt * bt
          + 3 * offdiag_pairs * 16 * k
      )
      * 4
  )
  per_head_bwd = io_bwd + io_bwd * 3 // 2
  return max(per_head_intra, per_head_bwd)


def _valid_autotune_mini_batches(
    mini_batches: tuple[int | None, ...],
    *,
    chunk_size: int,
    act_dtype: Any,
    head_k_dim: int,
    head_v_dim: int,
    per_channel_gate: bool,
    is_kda: bool,
    two_pass_bf16: bool = False,
) -> tuple[int | None, ...]:
  vmem_limit = common.get_vmem_limit_bytes()
  max_mb = common.get_max_mini_batch(
      chunk_size=chunk_size if chunk_size > 128 else None,
      vmem_limit_bytes=vmem_limit,
  )
  if per_channel_gate and chunk_size > 64:
    max_mb = max_mb // 2
  per_head = _estimate_per_head_vmem_bytes(
      act_dtype=act_dtype,
      chunk_size=chunk_size,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      per_channel_gate=per_channel_gate,
      is_kda=is_kda,
      two_pass_bf16=two_pass_bf16,
  )
  return tuple(
      mb
      for mb in mini_batches
      if mb is not None
      and (chunk_size <= 64 or mb <= max_mb)
      and mb * per_head <= vmem_limit
  ) or (None,)


# Maximum fused conv1d width: the forward kernel keeps an 8-row halo of the
# previous chunk (`_CONV_HALO` in the forward kernel).
_MAX_CONV_WIDTH = 8


def _check_chunk_size_support(chunk_size: int, *, per_channel_gate: bool):
  """Checks the chunk sizes supported by the Mosaic TPU kernels."""
  # Per-channel gates support chunk_size in (64, 128); scalar gates support
  # powers of 2 in [16, 512].
  if per_channel_gate:
    if chunk_size not in (64, 128):
      raise NotImplementedError(
          "`mosaic` supports chunk_size in (64, 128) with a per-channel gate;"
          f" got {chunk_size}."
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
    k: jax.Array | None = None,
    initial_state: jax.Array | None,
    output_final_state: bool,
    segment_ids: jax.Array | None,
    context_parallel_metadata: ContextParallelMetadata | None,
    chunk_size: int,
    max_num_segments: int | None,
    per_channel_gate: bool = True,
    conv_weights: tuple[jax.Array | None, ...] = (),
    mini_batch: int | None = None,
) -> None:
  """Checks whether the Pallas/Mosaic TPU backend supports static inputs."""
  if q.dtype not in (jnp.bfloat16, jnp.float32):
    raise NotImplementedError(
        "`mosaic` currently supports bfloat16 and float32 inputs only."
    )
  if (k is not None and k.dtype != q.dtype) or v.dtype != q.dtype:
    raise NotImplementedError(
        "`mosaic` requires `query`, `key`, and `value` to share the same"
        f" dtype; got q={q.dtype},"
        f" k={k.dtype if k is not None else None}, v={v.dtype}."
    )
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  if heads < 1 or batch < 1 or seq_len < 1:
    raise NotImplementedError(
        "`mosaic` requires positive head, batch, and sequence "
        f"dimensions; got H={heads}, B={batch}, T={seq_len}."
    )
  if mini_batch is not None and (mini_batch < 1 or heads % mini_batch != 0):
    raise NotImplementedError(
        "`mosaic` requires `mini_batch` to be a positive divisor of"
        f" H={heads}; got mini_batch={mini_batch}."
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
    if segment_ids is not None and max_num_segments is None:
      raise NotImplementedError(
          "`mosaic` context-parallel execution requires `max_num_segments` "
          "when `segment_ids` is provided."
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
  use_inplace_packed = (
      segment_ids is not None
      and not cp_enabled
      and initial_state is None
      and not output_final_state
  )
  if use_inplace_packed and chunk_size not in (16, 32, 64, 128):
    raise NotImplementedError(
        "`mosaic` in-place packed KDA supports chunk_size in"
        f" (16, 32, 64, 128); got {chunk_size}."
    )
  for conv_weight in conv_weights:
    if conv_weight is None:
      continue
    _, conv_width, _ = conv_weight.shape
    if not 1 <= conv_width <= _MAX_CONV_WIDTH:
      raise NotImplementedError(
          f"`mosaic` supports conv1d widths in [1, {_MAX_CONV_WIDTH}]; got"
          f" W={conv_width}."
      )
  if (segment_ids is None or cp_enabled) and seq_len % chunk_size != 0:
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
  def replace(self, **kwargs):
    if (
        "config" in kwargs
        and "vjp" not in kwargs
        and isinstance(self.vjp, PallasMosaicTpuKimiDeltaAttentionVjp)
        and self.vjp.config == self.config
    ):
      kwargs = kwargs | {
          "vjp": PallasMosaicTpuKimiDeltaAttentionVjp(config=kwargs["config"])
      }
    return super().replace(**kwargs)

  def bind(self, *args, _packing_order=None, **kwargs):
    bound = super().bind(*args, **kwargs)
    if _packing_order is None:
      return bound
    return op.BoundArguments(
        self, {**bound.arguments, "_packing_order": _packing_order}
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
    v = args.get("value")
    q_shaped = q is not None and hasattr(q, "shape")
    heads = int(q.shape[0]) if q_shaped else 32
    seq_len = q.shape[2] if q_shaped else None
    key_dim = int(q.shape[3]) if q_shaped else 128
    val_dim = int(v.shape[3]) if v is not None and hasattr(v, "shape") else key_dim
    act_dtype = q.dtype if q is not None and hasattr(q, "dtype") else jnp.bfloat16
    per_channel_gate = args.get("per_channel_gate", True)
    segment_ids = args.get("segment_ids")
    cp_meta = args.get("context_parallel_metadata")
    cp_enabled = cp_meta is not None and getattr(
        cp_meta, "is_cp_enabled", False
    )
    req_cs = args.get("chunk_size")
    stateless = args.get("initial_state") is None and not args.get(
        "output_final_state", False
    )
    if req_cs is not None:
      chunk_sizes = (int(req_cs),)
    elif segment_ids is not None:
      if not stateless:
        chunk_sizes = (64,)
      elif cp_enabled:
        if isinstance(seq_len, int):
          chunk_sizes = (
              tuple(cs for cs in (64, 128) if seq_len % cs == 0) or (64,)
          )
        else:
          chunk_sizes = (64, 128)
      elif (
          isinstance(seq_len, int)
          and not per_channel_gate
          and seq_len % 64 != 0
      ):
        chunk_sizes = tuple(
            cs for cs in (32, 16) if seq_len % cs == 0
        ) or (64,)
      elif isinstance(seq_len, int) and seq_len <= 64:
        chunk_sizes = (64,)
      else:
        chunk_sizes = (64, 128)
    else:
      base_cs = (
          (64, 128) if (per_channel_gate or cp_enabled) else (32, 64, 128)
      )
      if isinstance(seq_len, int):
        chunk_sizes = tuple(cs for cs in base_cs if seq_len % cs == 0) or (64,)
      else:
        chunk_sizes = base_cs
    mini_batches = tuple(
        mb for mb in (8, 16, 32) if mb <= heads and heads % mb == 0
    ) or (None,)
    configs = set()
    for cs in chunk_sizes:
      cs_mbs = _valid_autotune_mini_batches(
          mini_batches,
          chunk_size=cs,
          act_dtype=act_dtype,
          head_k_dim=key_dim,
          head_v_dim=val_dim,
          per_channel_gate=per_channel_gate,
          is_kda=True,
      )
      for mb in cs_mbs:
        configs.add(Config(chunk_size=cs, mini_batch=mb))
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6

  def __call__(  # pyrefly: ignore[bad-override]
      self, *args: Any, return_residuals: bool = False, **kwargs: Any
  ) -> Any:
    ckpt_name = kwargs.get("residual_checkpoint_name") or (
        self.config.residual_checkpoint_name
        if self.config is not None
        else None
    )
    cp_meta = kwargs.get("context_parallel_metadata")
    has_cp_ckpt = (
        cp_meta is not None
        and cp_meta.is_cp_enabled
        and cp_meta.cp_residual_checkpoint_name is not None
    )
    if (
        (ckpt_name is None and not has_cp_ckpt)
        or return_residuals
        or self.vjp is None
    ):
      return super().__call__(*args, return_residuals=return_residuals, **kwargs)
    return _call_with_selective_checkpoint_vjp(self, *args, **kwargs)

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
      token_order=None,
      use_conv1d_in_kernel: bool = False,
      normalize_before_alignment: bool = False,
      require_alignment: bool = False,
  ) -> _PreparedKdaInputs:
    """Canonicalizes inputs shared by the forward and backward kernels."""
    cp_enabled = (
        context_parallel_metadata is not None
        and context_parallel_metadata.is_cp_enabled
    )
    use_inplace_packed = (
        not require_alignment
        and segment_ids is not None
        and initial_state is None
        and not output_final_state
        and (not cp_enabled or q.shape[2] % chunk_size == 0)
    )
    use_inplace_unpacked_cp = (
        cp_enabled
        and segment_ids is None
        and initial_state is None
        and not output_final_state
    )
    seg_prepared = segment_ids
    if use_inplace_packed:
      assert segment_ids is not None
      if segment_ids.ndim != 2 or segment_ids.shape != (q.shape[1], q.shape[2]):
        raise ValueError(
            f"`segment_ids` must have shape {(q.shape[1], q.shape[2])}; got"
            f" {segment_ids.shape}."
        )
      if max_num_segments is None:
        raise ValueError(
            "`max_num_segments` is required when `segment_ids` is provided "
            "without `initial_state`."
        )
      if max_num_segments <= 0:
        raise ValueError("`max_num_segments` must be positive.")
      seg_prepared = jnp.where(
          (segment_ids > 0) & (segment_ids <= max_num_segments),
          segment_ids,
          0,
      )
      pad_t = (-q.shape[2]) % chunk_size
      if pad_t:
        pad_4d = ((0, 0), (0, 0), (0, pad_t), (0, 0))
        q = jnp.pad(q, pad_4d)
        k = jnp.pad(k, pad_4d)
        v = jnp.pad(v, pad_4d)
        g = jnp.pad(g, pad_4d)
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_t)))
        seg_prepared = jnp.pad(seg_prepared, ((0, 0), (0, pad_t)))
      cu_seqlens = None
    elif use_inplace_unpacked_cp:
      cu_seqlens = None
    else:
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

    packed_q = packed_k = q_rstd = k_rstd = None
    if normalize_before_alignment and use_qk_l2norm:
      q, q_rstd = l2norm_fwd(q)
      k, k_rstd = l2norm_fwd(k)
      packed_q, packed_k = q, k

    aligned_cu_seqlens = None
    chunk_indices = None
    if cu_seqlens is None:
      q_aligned, k_aligned, v_aligned = q, k, v
      g_aligned, beta_aligned = g, beta
    else:
      (
          [q_aligned, k_aligned, v_aligned, g_aligned],
          aligned_scalars,
          aligned_cu_seqlens,
          _,
      ) = _align_seqs(
          [q, k, v, g],
          [beta] if q_rstd is None else [beta, q_rstd, k_rstd],
          cu_seqlens,
          align=chunk_size,
          token_order=token_order,
      )
      beta_aligned = aligned_scalars[0]
      if q_rstd is not None:
        q_rstd, k_rstd = aligned_scalars[1:]
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

    if use_qk_l2norm and not use_conv1d_in_kernel and not normalize_before_alignment:
      if use_inplace_packed and seg_prepared is not None:
        valid_4d = (seg_prepared > 0)[None, :, :, None]
        q_aligned = jnp.where(valid_4d, q_aligned, 0.0)
        k_aligned = jnp.where(valid_4d, k_aligned, 0.0)
      q_prepared, q_rstd = l2norm_fwd(q_aligned)
      k_prepared, k_rstd = l2norm_fwd(k_aligned)
    else:
      q_prepared, k_prepared = q_aligned, k_aligned
      if not normalize_before_alignment:
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
        segment_ids=seg_prepared,
        packed_q=packed_q,
        packed_k=packed_k,
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
      delta_time_bias: Float[Array, "H*GW"] | None,
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
      residual_checkpoint_name: str | None = None,
      _packing_order: jax.Array | None = None,
      config: Config,
  ) -> tuple[base.Output, base.Residuals]:
    if config.canonicalize_packed_inputs and segment_ids is not None:
      arguments = locals().copy()
      arguments.pop("self")
      arguments.pop("config")
    if chunk_size is not None:
      _check_chunk_size_support(chunk_size, per_channel_gate=per_channel_gate)
    effective_chunk_size = (
        chunk_size if chunk_size is not None else config.chunk_size
    )
    safe_gate = _resolve_safe_gate(
        config,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )
    _check_gate_support(
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        safe_gate=safe_gate,
        per_channel_gate=per_channel_gate,
    )
    save_intermediates_for_backward = not config.rematerialize_for_backward

    # Reject unsupported calls before preprocessing or tracing a Pallas kernel,
    # so API dispatch can fall through to the next implementation.
    check_inputs_support(
        query,
        value,
        k=key,
        initial_state=initial_state,
        output_final_state=output_final_state,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
        chunk_size=effective_chunk_size,
        max_num_segments=max_num_segments,
        per_channel_gate=per_channel_gate,
        conv_weights=(
            (conv_weight_q, conv_weight_k, conv_weight_v)
            if use_conv1d_in_kernel
            else ()
        ),
        mini_batch=config.mini_batch,
    )

    if (
        per_channel_gate
        and not config.two_pass_bf16
        and not use_conv1d_in_kernel
        and config.mini_batch is None
        and compact.supported(
            query,
            value,
            segment_ids=segment_ids,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm=use_qk_l2norm,
            context_parallel_metadata=context_parallel_metadata,
            config=config,
        )
    ):
      return compact.forward(
          query,
          key,
          value,
          gate,
          beta,
          a_log=a_log,
          delta_time_bias=delta_time_bias,
          scale=scale,
          segment_ids=segment_ids,
          lower_bound=lower_bound,
          use_gate_in_kernel=use_gate_in_kernel,
          return_residuals=return_residuals,
          sorted_segments=config.compact_sorted_segments,
          max_num_segments=max_num_segments,
      )

    if config.canonicalize_packed_inputs and segment_ids is not None:
      return packing.invoke(self, config, arguments)

    use_packed_inputs = (
        config.packed_forward
        and not config.two_pass_bf16
        and config.fuse_forward
        and not return_residuals
        and segment_ids is not None
        and not use_conv1d_in_kernel
        and not (
            context_parallel_metadata is not None
            and context_parallel_metadata.is_cp_enabled
        )
        and query.shape[-1] % 128 == 0
        and value.shape[-1] % 128 == 0
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
        chunk_size=effective_chunk_size,
        max_num_segments=max_num_segments,
        token_order=None if _packing_order is None else _packing_order[0],
        use_conv1d_in_kernel=use_conv1d_in_kernel,
        normalize_before_alignment=use_packed_inputs,
        require_alignment=(
            _packing_order is not None
            or (
                config.fuse_forward
                and not config.two_pass_bf16
                and not (
                    context_parallel_metadata is not None
                    and context_parallel_metadata.is_cp_enabled
                )
            )
        ),
    )

    ckpt_name = residual_checkpoint_name or config.residual_checkpoint_name
    cp_enabled = (
        prepared.context_parallel_metadata is not None
        and prepared.context_parallel_metadata.is_cp_enabled
    )
    cp_ckpt_name = (
        (
            prepared.context_parallel_metadata.cp_residual_checkpoint_name
            or ckpt_name
        )
        if cp_enabled and prepared.context_parallel_metadata is not None
        else None
    )

    packed_inputs = None
    if use_packed_inputs and prepared.cu_seqlens is not None:
      packed_q = prepared.packed_q if use_qk_l2norm else query
      packed_k = prepared.packed_k if use_qk_l2norm else key
      packed_inputs = (packed_q, packed_k, value, gate, beta)

    inverse_order = None if _packing_order is None else _packing_order[1]
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
        segment_ids=prepared.segment_ids,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=save_intermediates_for_backward,
        fuse_forward=config.fuse_forward,
        packed_inputs=packed_inputs,
        token_unpermute=inverse_order,
        padding_guard=prepared.cu_seqlens is not None and effective_chunk_size > 1,
        packed_output=config.packed_output,
        context_parallel_metadata=prepared.context_parallel_metadata,
        chunk_size=effective_chunk_size,
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
        two_pass_bf16=config.two_pass_bf16,
        cp_checkpoint_name=cp_ckpt_name,
    )
    if residuals is not None and _packing_order is not None:
      residuals = dataclasses.replace(
          residuals,
          token_order=_packing_order[0],
          token_unpermute=inverse_order,
      )
    out_val, final_state = output
    if out_val.shape[2] != query.shape[2]:
      out_val = out_val[:, :, : query.shape[2], :]
    if final_state is not None and final_state.ndim == 4:
      final_state = final_state[:, None]
    output = (out_val, final_state)
    if ckpt_name is not None:
      output, residuals = _tag_kda_residuals(
          output,
          residuals,
          ckpt_name,
          tag_qkv=use_conv1d_in_kernel or use_qk_l2norm,
          tag_initial_state=(cp_enabled and prepared.cu_seqlens is not None),
      )
    return output, residuals


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuKimiDeltaAttentionVjp(
    op.Op[Any, dict[str, Any], None, Config, Any]
):
  """Tokamax Op VJP wrapper for the Pallas/Mosaic TPU KDA backward path."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        chunk_size=_vjp_chunk_size(ba, _requested_chunk_size),
        two_pass_bf16=_vjp_two_pass_bf16(ba),
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    args = getattr(ba, "arguments", None)
    if not args:
      return {Config(chunk_size=_requested_chunk_size(ba))}
    q = args.get("query")
    v = args.get("value")
    q_shaped = q is not None and hasattr(q, "shape")
    heads = int(q.shape[0]) if q_shaped else 32
    key_dim = int(q.shape[3]) if q_shaped else 128
    val_dim = int(v.shape[3]) if v is not None and hasattr(v, "shape") else key_dim
    act_dtype = q.dtype if q is not None and hasattr(q, "dtype") else jnp.bfloat16
    per_channel_gate = args.get("per_channel_gate", True)
    cs = _vjp_chunk_size(ba, _requested_chunk_size)
    two_pass = _vjp_two_pass_bf16(ba)
    mini_batches = tuple(
        mb for mb in (8, 16, 32) if mb <= heads and heads % mb == 0
    ) or (None,)
    cs_mbs = _valid_autotune_mini_batches(
        mini_batches,
        chunk_size=cs,
        act_dtype=act_dtype,
        head_k_dim=key_dim,
        head_v_dim=val_dim,
        per_channel_gate=per_channel_gate,
        is_kda=True,
        two_pass_bf16=two_pass,
    )
    return {
        Config(chunk_size=cs, mini_batch=mb, two_pass_bf16=two_pass)
        for mb in cs_mbs
    }

  def _fwd(
      self,
      residuals: KdaResiduals | compact.CompactResiduals | packing.PackingResiduals,
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
      residual_checkpoint_name: str | None = None,
      _packing_order: jax.Array | None = None,
      config: Config,
  ) -> tuple[dict[str, jax.Array], None]:
    # Tokamax's VJP contract replays the original inputs here, but the backward
    # kernel consumes the aligned and optionally L2-normalized copies retained
    # in `residuals`. Reusing these arguments would skip that preprocessing.
    if isinstance(residuals, packing.PackingResiduals):
      return packing.backward(residuals, dout, segment_ids)
    primal_dtypes = {name: value.dtype for name, value in locals().items()
                     if name in packing.ARRAY_NAMES and value is not None}
    if isinstance(residuals, compact.CompactResiduals):
      gradients = compact.backward(
          residuals,
          dout,
          query=query,
          value=value,
          gate=gate,
          scale=scale,
          lower_bound=lower_bound,
          use_gate_in_kernel=use_gate_in_kernel,
          a_log=a_log,
          delta_time_bias=delta_time_bias,
      )
      if segment_ids is not None:
        gradients["segment_ids"] = jnp.zeros_like(segment_ids)
      return gradients, None
    del out, query, key, value, gate, beta
    del output_final_state, return_residuals, use_conv1d_in_kernel
    del residual_checkpoint_name, chunk_size
    effective_chunk_size = residuals.chunk_size
    effective_two_pass = (
        config.two_pass_bf16
        if self.config is not None
        else residuals.two_pass_bf16
    )
    safe_gate = _resolve_safe_gate(
        config,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )
    _check_gate_support(
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        safe_gate=safe_gate,
        per_channel_gate=per_channel_gate,
    )
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
        effective_chunk_size,
        max_num_segments,
        initial_state is not None,
        residuals,
        dout,
        per_channel_gate=per_channel_gate,
        mini_batch=config.mini_batch,
        two_pass_bf16=effective_two_pass,
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
    for name in packing.ARRAY_NAMES:
      if name in grads:
        grads[name] = grads[name].astype(primal_dtypes[name])
    return grads, None


def _requested_gdn_chunk_size(ba: op.BoundArguments) -> int:
  args = getattr(ba, "arguments", None) or {}
  requested = args.get("chunk_size")
  return 64 if requested is None else int(requested)


def check_gdn_inputs_support(
    *,
    act_dtype: jnp.dtype,
    k_dtype: Any = None,
    v_dtype: Any = None,
    batch: int,
    seq_len: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int,
    conv_weight: jax.Array | None,
    segment_ids: jax.Array | None,
    context_parallel_metadata: ContextParallelMetadata | None,
    mini_batch: int | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    two_pass_bf16: bool = False,
) -> None:
  """Checks whether the Pallas/Mosaic TPU GDN backend supports static inputs."""
  if act_dtype not in (jnp.bfloat16, jnp.float32):
    raise NotImplementedError(
        "`mosaic` currently supports bfloat16 and float32 inputs only."
    )
  if (k_dtype is not None and k_dtype != act_dtype) or (
      v_dtype is not None and v_dtype != act_dtype
  ):
    raise NotImplementedError(
        "`mosaic` requires `query`, `key`, and `value` to share the same"
        f" dtype; got q={act_dtype}, k={k_dtype}, v={v_dtype}."
    )
  if batch < 1 or seq_len < 1 or num_k_heads < 1 or num_v_heads < 1:
    raise NotImplementedError(
        "`mosaic` requires positive batch, sequence, and head dimensions."
    )
  if head_k_dim < 1 or head_v_dim < 1 or head_k_dim > 256:
    raise NotImplementedError(
        "`mosaic` supports key dimensions in [1, 256] and positive value"
        f" dimensions; got K={head_k_dim}, V={head_v_dim}."
    )
  _check_chunk_size_support(chunk_size, per_channel_gate=False)
  if seq_len % chunk_size != 0:
    raise NotImplementedError(
        "`mosaic` requires sequence length divisible by `chunk_size`; got"
        f" T={seq_len}, chunk_size={chunk_size}."
    )
  if conv_weight is not None:
    conv_width = conv_weight.shape[0]
    if not 1 <= conv_width <= _MAX_CONV_WIDTH:
      raise NotImplementedError(
          f"`mosaic` supports conv1d widths in [1, {_MAX_CONV_WIDTH}]; got"
          f" W={conv_width}."
      )
  if segment_ids is not None and chunk_size not in (16, 32, 64, 128):
    raise NotImplementedError(
        "`mosaic` packed GDN supports chunk_size in (16, 32, 64, 128); got"
        f" {chunk_size}."
    )
  if (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  ):
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
    if head_k_dim % 128 != 0 or head_v_dim % 128 != 0:
      raise NotImplementedError(
          "`mosaic` context-parallel execution requires key and value "
          "dimensions to be multiples of 128; got "
          f"K={head_k_dim}, V={head_v_dim}."
      )
  gqa_repeats = num_v_heads // num_k_heads
  if mini_batch is not None and (
      mini_batch < 1
      or num_v_heads % mini_batch != 0
      or mini_batch % gqa_repeats != 0
  ):
    raise NotImplementedError(
        "`mosaic` GDN requires `mini_batch` to be a positive divisor of"
        f" H_V={num_v_heads} and a multiple of gqa_repeats={gqa_repeats};"
        f" got mini_batch={mini_batch}."
    )
  vmem_limit_bytes = common.get_vmem_limit_bytes()
  per_head_bytes = _estimate_per_head_vmem_bytes(
      act_dtype=act_dtype,
      chunk_size=chunk_size,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      per_channel_gate=False,
      is_kda=False,
      two_pass_bf16=two_pass_bf16,
  )
  if gqa_repeats * per_head_bytes > vmem_limit_bytes:
    raise NotImplementedError(
        f"`mosaic` GDN GQA group size ({gqa_repeats}) exceeds VMEM budget"
        f" ({gqa_repeats * per_head_bytes} > {vmem_limit_bytes} bytes)."
    )


@dataclasses.dataclass(frozen=True)
class PallasMosaicTpuGatedDeltaNet(base.GatedDeltaNet[Config, Any]):
  """Pallas/Mosaic TPU Gated Delta Net backend."""

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self, "vjp", PallasMosaicTpuGatedDeltaNetVjp(config=self.config)
      )

  @override
  def replace(self, **kwargs):
    if (
        "config" in kwargs
        and "vjp" not in kwargs
        and isinstance(self.vjp, PallasMosaicTpuGatedDeltaNetVjp)
        and self.vjp.config == self.config
    ):
      kwargs = kwargs | {
          "vjp": PallasMosaicTpuGatedDeltaNetVjp(config=kwargs["config"])
      }
    return super().replace(**kwargs)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(chunk_size=_requested_gdn_chunk_size(ba))

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    args = getattr(ba, "arguments", None)
    if not args:
      return {Config(chunk_size=_requested_gdn_chunk_size(ba))}
    heads = int(args.get("num_v_heads") or 32)
    gqa_repeats = heads // int(args.get("num_k_heads") or heads)
    act_src = (
        args.get("qkv") if args.get("qkv") is not None else args.get("query")
    )
    src_shaped = act_src is not None and hasattr(act_src, "shape")
    seq_len = act_src.shape[1] if src_shaped else None
    act_dtype = (
        act_src.dtype
        if act_src is not None and hasattr(act_src, "dtype")
        else jnp.bfloat16
    )
    key_dim = int(args.get("head_k_dim") or 128)
    val_dim = int(args.get("head_v_dim") or 128)
    segment_ids = args.get("segment_ids")
    cp_meta = args.get("context_parallel_metadata")
    cp_enabled = cp_meta is not None and getattr(
        cp_meta, "is_cp_enabled", False
    )
    req_cs = args.get("chunk_size")
    if req_cs is not None:
      chunk_sizes = (int(req_cs),)
    elif segment_ids is not None:
      if isinstance(seq_len, int):
        if cp_enabled:
          chunk_sizes = (
              tuple(cs for cs in (64, 128) if seq_len % cs == 0) or (64,)
          )
        else:
          chunk_sizes = (
              tuple(cs for cs in (64, 128) if seq_len % cs == 0)
              or tuple(cs for cs in (32, 16) if seq_len % cs == 0)
              or (64,)
          )
      else:
        chunk_sizes = (64, 128)
    else:
      base_cs = (64, 128) if cp_enabled else (32, 64, 128)
      if isinstance(seq_len, int):
        chunk_sizes = tuple(cs for cs in base_cs if seq_len % cs == 0) or (64,)
      else:
        chunk_sizes = base_cs
    mini_batches = tuple(
        mb
        for mb in (8, 16, 32)
        if mb <= heads and heads % mb == 0 and mb % gqa_repeats == 0
    ) or (None,)
    configs = set()
    for cs in chunk_sizes:
      cs_mbs = _valid_autotune_mini_batches(
          mini_batches,
          chunk_size=cs,
          act_dtype=act_dtype,
          head_k_dim=key_dim,
          head_v_dim=val_dim,
          per_channel_gate=False,
          is_kda=False,
      )
      for mb in cs_mbs:
        configs.add(Config(chunk_size=cs, mini_batch=mb))
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6

  def __call__(  # pyrefly: ignore[bad-override]
      self, *args: Any, return_residuals: bool = False, **kwargs: Any
  ) -> Any:
    ckpt_name = kwargs.get("residual_checkpoint_name") or (
        self.config.residual_checkpoint_name
        if self.config is not None
        else None
    )
    cp_meta = kwargs.get("context_parallel_metadata")
    has_cp_ckpt = (
        cp_meta is not None
        and cp_meta.is_cp_enabled
        and cp_meta.cp_residual_checkpoint_name is not None
    )
    if (
        (ckpt_name is None and not has_cp_ckpt)
        or return_residuals
        or self.vjp is None
    ):
      return super().__call__(*args, return_residuals=return_residuals, **kwargs)
    return _call_with_selective_checkpoint_vjp(self, *args, **kwargs)

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      query: Float[Array, "B T H_QK K"] | None,
      key: Float[Array, "B T H_QK K"] | None,
      value: Float[Array, "B T H_V V"] | None,
      gate: Float[Array, "B T H_V"],
      beta: Float[Array, "B T H_V"],
      *,
      qkv: Float[Array, "B T C_QKV"] | None,
      num_k_heads: int,
      num_v_heads: int,
      head_k_dim: int,
      head_v_dim: int,
      conv_weight: Float[Array, "W C_QKV"] | None,
      conv_bias: Float[Array, "C_QKV"] | None,
      a_log: Float[Array, "H_V"] | None,
      delta_time_bias: Float[Array, "H_V"] | None,
      scale: float,
      initial_state: Float[Array, "B H_V K V"] | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      use_beta_sigmoid_in_kernel: bool,
      chunk_size: int | None,
      segment_ids: Int[Array, "B T"] | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      residual_checkpoint_name: str | None = None,
      return_residuals: bool,
      config: Config,
  ) -> tuple[base.Output, base.Residuals]:
    if chunk_size is not None:
      _check_chunk_size_support(chunk_size, per_channel_gate=False)
    effective_chunk_size = (
        chunk_size if chunk_size is not None else config.chunk_size
    )
    act_source = qkv if qkv is not None else query
    assert act_source is not None
    check_gdn_inputs_support(
        act_dtype=act_source.dtype,
        k_dtype=key.dtype if qkv is None and key is not None else None,
        v_dtype=value.dtype if qkv is None and value is not None else None,
        batch=act_source.shape[0],
        seq_len=act_source.shape[1],
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        chunk_size=effective_chunk_size,
        conv_weight=conv_weight,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
        mini_batch=config.mini_batch,
        initial_state=initial_state,
        output_final_state=output_final_state,
        two_pass_bf16=config.two_pass_bf16,
    )
    safe_gate = _resolve_safe_gate(
        config, use_gate_in_kernel=use_gate_in_kernel, lower_bound=None
    )
    save_intermediates = not config.rematerialize_for_backward
    ckpt_name = residual_checkpoint_name or config.residual_checkpoint_name
    cp_ckpt_name = (
        (context_parallel_metadata.cp_residual_checkpoint_name or ckpt_name)
        if context_parallel_metadata is not None
        and context_parallel_metadata.is_cp_enabled
        else None
    )
    output, residuals = chunk_gdn_fwd_custom(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        qkv=qkv,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        safe_gate=safe_gate,
        disable_recompute=save_intermediates,
        chunk_size=effective_chunk_size,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
        return_residuals=return_residuals,
        mini_batch=config.mini_batch,
        two_pass_bf16=config.two_pass_bf16,
        cp_checkpoint_name=cp_ckpt_name,
    )
    if ckpt_name is not None:
      output, residuals = _tag_gdn_residuals(
          output,
          residuals,
          ckpt_name,
          tag_qkv_conv=(
              save_intermediates and (conv_weight is not None or qkv is None)
          ),
          tag_beta=save_intermediates and use_beta_sigmoid_in_kernel,
      )
    return output, residuals


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuGatedDeltaNetVjp(
    op.Op[Any, dict[str, Any], None, Config, Any]
):
  """Tokamax Op VJP wrapper for Pallas/Mosaic TPU Gated Delta Net."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        chunk_size=_vjp_chunk_size(ba, _requested_gdn_chunk_size),
        two_pass_bf16=_vjp_two_pass_bf16(ba),
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    args = getattr(ba, "arguments", None)
    if not args:
      return {Config(chunk_size=_requested_gdn_chunk_size(ba))}
    heads = int(args.get("num_v_heads") or 32)
    gqa_repeats = heads // int(args.get("num_k_heads") or heads)
    act_src = (
        args.get("qkv") if args.get("qkv") is not None else args.get("query")
    )
    act_dtype = (
        act_src.dtype
        if act_src is not None and hasattr(act_src, "dtype")
        else jnp.bfloat16
    )
    key_dim = int(args.get("head_k_dim") or 128)
    val_dim = int(args.get("head_v_dim") or 128)
    cs = _vjp_chunk_size(ba, _requested_gdn_chunk_size)
    two_pass = _vjp_two_pass_bf16(ba)
    mini_batches = tuple(
        mb
        for mb in (8, 16, 32)
        if mb <= heads and heads % mb == 0 and mb % gqa_repeats == 0
    ) or (None,)
    cs_mbs = _valid_autotune_mini_batches(
        mini_batches,
        chunk_size=cs,
        act_dtype=act_dtype,
        head_k_dim=key_dim,
        head_v_dim=val_dim,
        per_channel_gate=False,
        is_kda=False,
        two_pass_bf16=two_pass,
    )
    return {
        Config(chunk_size=cs, mini_batch=mb, two_pass_bf16=two_pass)
        for mb in cs_mbs
    }

  def _fwd(
      self,
      residuals: GdnResiduals,
      out: base.Output,
      dout: base.Output,
      query: jax.Array | None,
      key: jax.Array | None,
      value: jax.Array | None,
      gate: jax.Array,
      beta: jax.Array,
      *,
      qkv: jax.Array | None,
      num_k_heads: int,
      num_v_heads: int,
      head_k_dim: int,
      head_v_dim: int,
      conv_weight: jax.Array | None,
      conv_bias: jax.Array | None,
      a_log: jax.Array | None,
      delta_time_bias: jax.Array | None,
      scale: float,
      initial_state: jax.Array | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      use_beta_sigmoid_in_kernel: bool,
      chunk_size: int | None,
      segment_ids: jax.Array | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      residual_checkpoint_name: str | None = None,
      return_residuals: bool,
      config: Config,
  ) -> tuple[dict[str, jax.Array], None]:
    del out, output_final_state, return_residuals, residual_checkpoint_name
    del chunk_size
    effective_chunk_size = residuals.chunk_size
    effective_two_pass = (
        config.two_pass_bf16
        if self.config is not None
        else residuals.two_pass_bf16
    )
    has_fused_qkv = qkv is not None
    if residuals.h is None:
      safe_gate = _resolve_safe_gate(
          config, use_gate_in_kernel=use_gate_in_kernel, lower_bound=None
      )
      _, recomputed = chunk_gdn_fwd_custom(
          query=query,
          key=key,
          value=value,
          gate=gate,
          beta=beta,
          qkv=qkv,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_weight=conv_weight,
          conv_bias=conv_bias,
          a_log=a_log,
          delta_time_bias=delta_time_bias,
          scale=scale,
          initial_state=initial_state,
          output_final_state=False,
          use_qk_l2norm=use_qk_l2norm,
          use_gate_in_kernel=use_gate_in_kernel,
          use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
          safe_gate=safe_gate,
          disable_recompute=True,
          chunk_size=effective_chunk_size,
          segment_ids=segment_ids,
          context_parallel_metadata=context_parallel_metadata,
          return_residuals=True,
          mini_batch=config.mini_batch,
          two_pass_bf16=effective_two_pass,
          precomputed_initial_state=residuals.initial_state,
          precomputed_qkv_conv=residuals.qkv_conv,
          precomputed_qkv_raw=residuals.qkv_raw,
          precomputed_cp_seg_halo=residuals.cp_seg_halo,
      )
      assert recomputed is not None
      residuals = recomputed
    del query, key, value, gate, beta, qkv

    (
        dqkv,
        dq,
        dk,
        dv,
        dg,
        db,
        da_log,
        ddt_bias,
        dh0,
        dconv_w,
        dconv_b,
    ) = chunk_gdn_bwd_custom(
        scale,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        use_qk_l2norm,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        effective_chunk_size,
        initial_state is not None,
        has_fused_qkv,
        residuals,
        dout,
        context_parallel_metadata=context_parallel_metadata,
        mini_batch=config.mini_batch,
        two_pass_bf16=effective_two_pass,
    )

    grads: dict[str, jax.Array] = {
        "gate": dg,
        "beta": db,
    }
    if has_fused_qkv:
      assert dqkv is not None
      grads["qkv"] = dqkv
    else:
      assert dq is not None and dk is not None and dv is not None
      grads["query"] = dq
      grads["key"] = dk
      grads["value"] = dv
    if conv_weight is not None:
      grads["conv_weight"] = (
          dconv_w if dconv_w is not None else jnp.zeros_like(conv_weight)
      )
    if conv_bias is not None:
      grads["conv_bias"] = (
          dconv_b if dconv_b is not None else jnp.zeros_like(conv_bias)
      )
    if a_log is not None:
      grads["a_log"] = da_log if da_log is not None else jnp.zeros_like(a_log)
    if delta_time_bias is not None:
      grads["delta_time_bias"] = (
          ddt_bias
          if ddt_bias is not None
          else jnp.zeros_like(delta_time_bias)
      )
    if initial_state is not None:
      grads["initial_state"] = (
          dh0 if dh0 is not None else jnp.zeros_like(initial_state)
      )
    if segment_ids is not None:
      grads["segment_ids"] = jnp.zeros_like(segment_ids)
    return grads, None
