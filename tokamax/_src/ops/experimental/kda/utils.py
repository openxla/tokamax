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
"""Utilities used by the experimental KDA implementation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import os

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.kda.cp_utils import (
    CPContext,
    _derive_cp_metadata_from_segment_ids,
)


def exp(x):
  return jnp.exp(x.astype(jnp.float32))


def exp2(x):
  """Base-2 exponential, matching Triton's tl.exp2."""
  return jnp.exp2(x.astype(jnp.float32))


def get_interpret() -> bool:
  env = os.environ.get("PALLAS_INTERPRET", "")
  return env.strip().lower() in ("1", "true")


def cdiv(x, y: int):
  return (x + y - 1) // y


def l2norm_fwd(x: jax.Array, eps: float = 1e-6):
  x_f = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + eps)
  return (x_f * rstd[..., None]).astype(x.dtype), rstd.astype(jnp.float32)


def derive_cp_context(
    *,
    segment_ids: jax.Array | None,
    initial_state: jax.Array | None,
    output_final_state: bool,
    cp_context: CPContext | None,
    N_max: int | None,
) -> tuple[CPContext | None, jax.Array | None]:
  cu_seqlens = None
  if cp_context is None or not cp_context.is_cp_enabled:
    return cp_context, cu_seqlens

  if initial_state is not None:
    raise ValueError("`initial_state` is not supported when CP is enabled.")
  if output_final_state:
    raise ValueError("`output_final_state` is not supported when CP is enabled.")
  if segment_ids is None:
    raise ValueError("CP requires rank-local `segment_ids` with shape [B, T].")

  if N_max is None:
    raise ValueError("`N_max` is required when CP uses `segment_ids`.")
  n_max = N_max
  cu_locals, chain_metas = [], []
  for b in range(segment_ids.shape[0]):
    cu_b, meta_b = _derive_cp_metadata_from_segment_ids(
        segment_ids[b],
        cp_context.axis_name,
        n_max=n_max,
    )
    cu_locals.append(cu_b)
    chain_metas.append(meta_b)
  cu_seqlens = jnp.stack(cu_locals, axis=0)
  chain_meta = {k: jnp.stack([m[k] for m in chain_metas]) for k in chain_metas[0]}
  cp_context = dataclasses.replace(
      cp_context,
      is_first_rank=chain_meta["is_first_rank"],
      is_last_rank=chain_meta["is_last_rank"],
      pre_num_ranks=chain_meta["pre_num_ranks"],
      post_num_ranks=chain_meta["post_num_ranks"],
  )
  return cp_context, cu_seqlens


def segment_ids_to_cu_seqlens(
    segment_ids: jax.Array | None,
    *,
    initial_state: jax.Array | None,
    N_max: int | None,
) -> tuple[jax.Array | None, int | None]:
  if segment_ids is None:
    return None, N_max
  if N_max is None:
    if initial_state is None:
      raise ValueError(
          "`N_max` is required when `segment_ids` is provided without "
          "`initial_state`."
      )
    N_max = initial_state.shape[1]
  return segment_ids_to_seqlens(segment_ids, max_segs=N_max), N_max


def align_up(x, align: int):
  return cdiv(x, align) * align


def pad_to_multiple(x: jax.Array, multiple: int | list[int], axis: int | list[int], val):
  if isinstance(multiple, int):
    multiple = [multiple]
  if isinstance(axis, int):
    axis = [axis]
  if len(multiple) != len(axis):
    raise ValueError(
        f"Length of multiple {len(multiple)} must match axis {len(axis)}."
    )

  shape = list(x.shape)
  pad_width = [(0, 0)] * len(shape)
  for ax, mu in zip(axis, multiple):
    remainder = shape[ax] % mu
    if remainder:
      pad_width[ax] = (0, mu - remainder)
  return jnp.pad(x, pad_width, constant_values=val)


def prepare_lens(cu_seqlens: jax.Array) -> jax.Array:
  return cu_seqlens[1:] - cu_seqlens[:-1]


def _align_seqs(
    tensors_4d,
    tensors_3d,
    cu_seqlens,
    align,
    aligned_cu_seqlens=None,
):
  """Align (pad) each variable-length sequence to a multiple of ``align``.

  Supports both single-batch (cu_seqlens [N+1]) and batched
  (cu_seqlens [B, N+1]) modes.  In batched mode, each batch element is
  aligned independently and all results are padded to the maximum
  aligned T across batches.
  """
  if cu_seqlens.ndim == 2:
    # Batched: loop over B (values are concrete at trace time).
    B = cu_seqlens.shape[0]
    per_batch_4d = [[] for _ in tensors_4d]
    per_batch_3d = [[] for _ in tensors_3d]
    padded_cus = []
    t_aligned_sizes = []
    for b in range(B):
      t4 = [t[:, b:b+1, :, :] for t in tensors_4d]
      t3 = [t[:, b:b+1, :] for t in tensors_3d]
      aligned_cu_b = (
          None
          if aligned_cu_seqlens is None
          else aligned_cu_seqlens[b]
      )
      aligned_4d, aligned_3d, padded_cu_b, _ = _align_seqs(
        t4,
        t3,
        cu_seqlens[b],
        align,
        aligned_cu_seqlens=aligned_cu_b,
      )
      for idx, a in enumerate(aligned_4d):
        per_batch_4d[idx].append(a)
      for idx, a in enumerate(aligned_3d):
        per_batch_3d[idx].append(a)
      padded_cus.append(padded_cu_b)
      t_aligned_sizes.append(aligned_4d[0].shape[2])

    T_max = max(t_aligned_sizes)
    # Pad each batch element to T_max and concatenate along B.
    def _pad_and_cat_4d(tensors_per_batch):
      padded = []
      for t in tensors_per_batch:
        pad_len = T_max - t.shape[2]
        if pad_len > 0:
          t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        padded.append(t)
      return jnp.concatenate(padded, axis=1)

    def _pad_and_cat_3d(tensors_per_batch):
      padded = []
      for t in tensors_per_batch:
        pad_len = T_max - t.shape[2]
        if pad_len > 0:
          t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len)))
        padded.append(t)
      return jnp.concatenate(padded, axis=1)

    out_4d = [_pad_and_cat_4d(per_batch_4d[i]) for i in range(len(tensors_4d))]
    out_3d = [_pad_and_cat_3d(per_batch_3d[i]) for i in range(len(tensors_3d))]
    stacked_cu = jnp.stack(padded_cus, axis=0)
    return out_4d, out_3d, stacked_cu, cu_seqlens

  # --- Single-batch path (original) ---
  N = cu_seqlens.shape[0] - 1
  T_old = tensors_4d[0].shape[2]

  seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
  if aligned_cu_seqlens is None:
    padded_lens = ((seg_lens + align - 1) // align) * align
    padded_cu = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(padded_lens)]
    )
  else:
    padded_cu = aligned_cu_seqlens
  T_new = ((T_old + N * (align - 1) + align - 1) // align) * align

  def _build_gather(i, gather_idx):
    old_start = cu_seqlens[i]
    new_start = padded_cu[i]
    sl = seg_lens[i]
    j = jnp.arange(T_new)
    in_seg = (j >= new_start) & (j < new_start + sl)
    src = old_start + (j - new_start)
    return jnp.where(in_seg, src, gather_idx)

  gather_idx = jnp.full(T_new, T_old, dtype=jnp.int32)
  gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)

  def repack_4d(t):
    # t: [H, B, T, K] — gather along axis 2 (T dimension)
    return jnp.pad(t, ((0, 0), (0, 0), (0, T_new - T_old), (0, 0)))[:, :, gather_idx]

  def repack_3d(t):
    # t: [H, B, T] — gather along axis 2 (T dimension)
    return jnp.pad(t, ((0, 0), (0, 0), (0, T_new - T_old)))[:, :, gather_idx]

  return (
    [repack_4d(t) for t in tensors_4d],
    [repack_3d(t) for t in tensors_3d],
    padded_cu,
    cu_seqlens,
  )



def _unalign_output(o, orig_cu_seqlens, aligned_cu_seqlens, T_out):
  """Reverse _align_seqs: scatter aligned output back to original positions.

  Supports batched cu_seqlens [B, N+1] — processes each batch element
  independently.
  """
  if orig_cu_seqlens.ndim == 2:
    B = orig_cu_seqlens.shape[0]
    per_batch = []
    for b in range(B):
      # Use slicing that works for both 3D [H,B,T] and 4D [H,B,T,X]
      ob_slice = jax.lax.dynamic_slice_in_dim(o, b, 1, axis=1)
      ob = _unalign_output(
        ob_slice,
        orig_cu_seqlens[b],
        aligned_cu_seqlens[b],
        T_out,
      )
      per_batch.append(ob)
    return jnp.concatenate(per_batch, axis=1)

  # --- Single-batch path (original) ---
  N = orig_cu_seqlens.shape[0] - 1
  orig_seg_lens = orig_cu_seqlens[1:] - orig_cu_seqlens[:-1]

  def _build_gather(i, gather_idx):
    orig_start = orig_cu_seqlens[i]
    aligned_start = aligned_cu_seqlens[i]
    sl = orig_seg_lens[i]
    j = jnp.arange(T_out)
    in_seg = (j >= orig_start) & (j < orig_start + sl)
    src = aligned_start + (j - orig_start)
    return jnp.where(in_seg, src, gather_idx)

  # Default to aligned_cu_seqlens[-1] — a known-zero padding position.
  # After the _align_seqs fix above, T_aligned > padded_cu[-1], so this
  # index is always valid and always reads padding (zero).
  safe_default = aligned_cu_seqlens[-1]
  gather_idx = jnp.full(T_out, safe_default, dtype=jnp.int32)
  gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)
  return o[:, :, gather_idx]



def align_segment_ids(
    segment_ids: jax.Array,
    N_max: int,
    chunk_size: int,
) -> jax.Array:
  """Align 1D segment IDs to chunk boundaries, matching `_align_seqs`."""
  if segment_ids.ndim != 1:
    raise ValueError(f"`segment_ids` must be 1D, got {segment_ids.shape}.")
  T = segment_ids.shape[0]

  is_boundary = jnp.concatenate([
      jnp.ones(1, dtype=jnp.bool_),
      segment_ids[1:] != segment_ids[:-1],
  ])
  seg_idx = jnp.cumsum(is_boundary.astype(jnp.int32)) - 1
  positions = jnp.arange(T, dtype=jnp.int32)

  seg_starts = jnp.full(N_max, T, dtype=jnp.int32).at[seg_idx].min(positions)
  seg_ends = jnp.zeros(N_max, dtype=jnp.int32).at[seg_idx].max(positions + 1)
  seg_lens = jnp.maximum(seg_ends - seg_starts, 0)
  seg_labels = segment_ids[jnp.minimum(seg_starts, T - 1)]

  is_real = (seg_labels > 0) & (seg_lens > 0)
  aligned_lens = jnp.where(is_real, align_up(seg_lens, chunk_size), 0)
  aligned_starts = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(aligned_lens)[:-1],
  ])

  T_aligned = ((T + N_max * (chunk_size - 1) + chunk_size - 1) // chunk_size) * chunk_size
  out = jnp.zeros((T_aligned,), dtype=jnp.int32)
  apos = jnp.arange(T_aligned, dtype=jnp.int32)

  def body(i, ids):
    start = aligned_starts[i]
    end = start + aligned_lens[i]
    mask = is_real[i] & (apos >= start) & (apos < end)
    return jnp.where(mask, seg_labels[i], ids)

  return jax.lax.fori_loop(0, N_max, body, out)


def segment_ids_to_seqlens(
    segment_ids: jax.Array,
    max_segs: int,
    chunk_size: int = 1,
) -> jax.Array:
  """Convert 1-indexed segment IDs with 0 padding to FLA-style cu_seqlens."""
  if segment_ids.ndim == 2:
    rows = [
        segment_ids_to_seqlens(segment_ids[b], max_segs, chunk_size)
        for b in range(segment_ids.shape[0])
    ]
    return jnp.stack(rows, axis=0)
  if segment_ids.ndim != 1:
    raise ValueError(f"`segment_ids` must be [T] or [B, T], got {segment_ids.shape}.")

  seg = segment_ids.reshape(-1)
  valid = seg != 0
  is_boundary = jnp.concatenate([
      jnp.ones(1, dtype=jnp.bool_),
      seg[1:] != seg[:-1],
  ]) & valid
  seg_idx = jnp.where(valid, jnp.cumsum(is_boundary.astype(jnp.int32)), 0)
  n_segs = seg_idx.max()
  n_real = jnp.sum(valid).astype(jnp.int32)

  is_end = jnp.concatenate([
      seg[1:] != seg[:-1],
      jnp.ones(1, dtype=jnp.bool_),
  ]) & valid
  prefix_len = jnp.cumsum(valid.astype(jnp.int32))

  drop_idx = jnp.asarray(max_segs + 1, dtype=jnp.int32)
  scatter_idx = jnp.where(is_end, seg_idx, drop_idx)
  scatter_val = jnp.where(is_end, prefix_len, 0)

  cu_seqlens = jnp.zeros((max_segs + 1,), dtype=jnp.int32)
  cu_seqlens = cu_seqlens.at[scatter_idx].max(scatter_val, mode="drop")
  out_idx = jnp.arange(max_segs + 1, dtype=jnp.int32)
  return jnp.where(out_idx > n_segs, n_real, cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
    max_T: int | None = None,
) -> jax.Array:
  """Compute per-chunk `(seq_id, block_id)` mapping from cu_seqlens."""
  if cu_seqlens.ndim == 2:
    rows = [
        prepare_chunk_indices(cu_seqlens[b], chunk_size, max_T=max_T)
        for b in range(cu_seqlens.shape[0])
    ]
    return jnp.stack(rows, axis=0)
  lens = prepare_lens(cu_seqlens)
  n_chunks = cdiv(lens, chunk_size)
  num_seqs = len(lens)
  if max_T is None:
    max_T = cu_seqlens[-1]
  total_nt = max_T // chunk_size
  seq_ids = jnp.repeat(
      jnp.arange(num_seqs, dtype=jnp.int32),
      n_chunks,
      total_repeat_length=total_nt,
  )
  prefix_chunks = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(n_chunks),
  ])
  seq_offsets = jnp.repeat(
      prefix_chunks[:-1],
      n_chunks,
      total_repeat_length=total_nt,
  )
  block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets
  return jnp.stack([seq_ids, block_ids], axis=1)


@dataclass(frozen=True)
class TpuConfig:
  generation: str
  vmem_per_core_bytes: int
  smem_per_core_bytes: int
  tflops_bf16_2d: float
  tflops_fp8_2d: float
  tflops_fp32_2d: float
  block_align_minor: int = 8
  block_align_major: int = 128
  hbm_bandwidth_gbps: float = 0.0
  frequency_ghz: float = 0.0
  tflops_fp32_1d: float = 0.0
  description: str = ""
  num_lanes: int = 128
  num_sublanes: int = 8
  mxu_column_size: int = 128
  cmem_per_core_bytes: int = 0
  hbm_per_core_bytes: int = 0
  mem_bw_bytes_per_second: int = 0
  tflops_int8_2d: float = 0.0
  tflops_int4_2d: float = 0.0

  @property
  def vmem_limit_bytes(self) -> int:
    return int(self.vmem_per_core_bytes * 0.9)

  @property
  def vmem_hw_limit_bytes(self) -> int:
    return int(self.vmem_per_core_bytes * 0.9)


TPU_V6E = TpuConfig(
    generation="v6e",
    vmem_per_core_bytes=128 * 1024 * 1024,
    smem_per_core_bytes=1024 * 1024,
    tflops_fp8_2d=920.0,
    tflops_bf16_2d=920.0,
    tflops_fp32_2d=460.0,
    hbm_bandwidth_gbps=1640.0,
    frequency_ghz=1.75,
    tflops_fp32_1d=7.168,
    description="TPU v6e (Trillium)",
    mxu_column_size=256,
    hbm_per_core_bytes=34_400_000_000,
    mem_bw_bytes_per_second=int(1.64e12),
    tflops_int8_2d=1840.0,
    tflops_int4_2d=3680.0,
)

TPU_V7 = TpuConfig(
    generation="v7",
    vmem_per_core_bytes=64 * 1024 * 1024,
    smem_per_core_bytes=1024 * 1024,
    tflops_fp8_2d=2300.0,
    tflops_bf16_2d=1155.0,
    tflops_fp32_2d=577.5,
    hbm_bandwidth_gbps=3700.0,
    frequency_ghz=2.2,
    tflops_fp32_1d=9.0112,
    description="TPU v7 (Ironwood) - 2 devices per chip, values are per-device",
    mxu_column_size=256,
    hbm_per_core_bytes=103_000_000_000,
    mem_bw_bytes_per_second=int(3.70e12),
)

_PRESETS: dict[str, TpuConfig] = {
    "v6e": TPU_V6E,
    "v7": TPU_V7,
}

_DEVICE_KIND_MAP: list[tuple[str, str]] = [
    ("v6 lite", "v6e"),
    ("v7 lite", "v7"),
    ("v7e", "v7"),
    ("v6e", "v6e"),
    ("v7", "v7"),
    ("v6", "v6e"),
]

_current_config: TpuConfig | None = None


def _detect_tpu_config() -> TpuConfig:
  try:
    devices = jax.devices()
  except RuntimeError:
    devices = []

  for device in devices:
    device_kind = device.device_kind.lower()
    for needle, key in _DEVICE_KIND_MAP:
      if needle in device_kind:
        return _PRESETS[key]
  return TPU_V6E


def get_tpu_config() -> TpuConfig:
  global _current_config
  if _current_config is None:
    _current_config = _detect_tpu_config()
  return _current_config
