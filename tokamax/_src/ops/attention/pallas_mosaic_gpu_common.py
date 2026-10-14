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
"""Common utilities for Mosaic GPU attention implementations."""

import functools
import math
from typing import Annotated, Any

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import vector
import numpy as np
import pydantic
import qwix
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.attention import base


CanonicalPrecision = precision_lib.CanonicalPrecision
QArray = qwix.QArray
_FORBID_EXTRA = pydantic.ConfigDict(extra="forbid")


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=_FORBID_EXTRA
)
class ConfigBase:
  """Common configuration parameters for Pallas-Mosaic-GPU kernels.

  Attributes:
    block_q: Block size along Q sequence length.
    block_kv: Block size along K/V sequence length.
    num_stages: Number of tma stages for loading KV.
    fold_q_sequence_heads: Whether to fold seq_q into num_q_heads.
    split_k: Number of chunks to split seq_len_k into to improve parallelism.
  """

  # TODO: Relax block size constraints to multiple of 32.
  block_q: Annotated[int, pydantic.Field(multiple_of=64, gt=0)] = 64
  block_kv: Annotated[int, pydantic.Field(multiple_of=64, gt=0)] = 64
  num_stages: pydantic.PositiveInt = 2
  fold_q_sequence_heads: pydantic.StrictBool = False
  split_k: pydantic.PositiveInt = 1

  def __post_init__(self):
    if type(self) is ConfigBase:  # pylint: disable=unidiomatic-typecheck
      raise ValueError("Cannot use ConfigBase directly. Use a subclass.")


MIN_SWIZZLE = 32
_RECIP_DIV_SHIFT = 16
_RECIP_DIV_SCALE = 1 << _RECIP_DIV_SHIFT
_SENTINEL_COORD = 1 << 20


# The contracting dimension for `wgmma` / `tcgen05.mma` must be a multiple of
# the minimum swizzle size (in number of elements).
def pad_head_dim_to_next_multiple_of_min_swizzle(x):
  m = 8 * MIN_SWIZZLE // mgpu_lib.num_bits(x.dtype)
  return shape_lib.pad_to_next_multiple_of(x, m, -1)


def decompose_mask(mask, q, k, q_indices, k_indices):
  """Decomposes `mask` into a mask array, `is_causal`, `k_start` and `k_end`."""
  if mask is None:
    return None, False, None, None

  is_causal = False
  k_start = None
  k_end = None

  if k_indices is None:
    mask, is_causal, k_start, k_end = mask.take("is_causal", "k_start", "k_end")

    # Fold `is_causal` into `k_end`. If `q_indices` is not `None`, then this is
    # necessary for correctness. Otherwise, it is a performance optimization.
    if is_causal and (q_indices is not None or k_end is not None):
      if q_indices is None:
        q_indices = jnp.arange(q.shape[-3])
      k_end_ = q_indices + 1
      k_end = k_end_ if k_end is None else jnp.minimum(k_end, k_end_)
      is_causal = False

    if k_start is not None:
      k_start = jax.lax.broadcast_to_rank(k_start, 2)
    if k_end is not None:
      k_end = jax.lax.broadcast_to_rank(k_end, 2)

  q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
  k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
  mask = mask.as_array(q_len_or_indices, k_len_or_indices)
  return mask, is_causal, k_start, k_end


def cast_qkv[
    Q: (jax.Array, QArray), K: (jax.Array, QArray), V: (jax.Array, QArray)
](
    q: Q, k: K, v: V, precision: tuple[CanonicalPrecision, CanonicalPrecision]
) -> tuple[Q, K, V]:
  """Casts Q, K, and V to the given precision."""

  def cast(x, precision):
    # Quantized arrays inherently define their precision (e.g., int8) and
    # cannot contain infinities, so we bypass sanitization and downcasting.
    if isinstance(x, QArray):
      return x
    assert precision != jax.lax.DotAlgorithmPreset.DEFAULT
    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      return safe_downcast(x, jnp.bfloat16)
    if precision == jax.lax.DotAlgorithmPreset.F16_F16_F32:
      return safe_downcast(x, jnp.float16)
    raise NotImplementedError(f"Unsupported precision: {precision}")

  q_k_dot_precision, p_v_dot_precision = precision
  q_k_dot_precision = precision_lib.to_dot_algorithm_preset(
      q.dtype, k.dtype, q_k_dot_precision
  )
  p_v_dot_precision = precision_lib.to_dot_algorithm_preset(
      v.dtype, v.dtype, p_v_dot_precision
  )
  q = cast(q, q_k_dot_precision)
  k = cast(k, q_k_dot_precision)
  v = cast(v, p_v_dot_precision)
  return q, k, v


def _broadcast_to_rank(x, rank):
  return None if x is None else jax.lax.broadcast_to_rank(x, rank)


def prepare_inputs(
    q,
    k,
    v,
    *,
    bias,
    mask,
    q_indices,
    k_indices,
    precision,
    fold_q_sequence_heads,
    split_k: int = 1,
    supports_local_3d: bool = True,
):
  """Prepares inputs for the standard attention kernel."""
  # TODO: Support in-kernel dequantization.
  q, k, v = map(quantization.as_array, (q, k, v))
  q, k, v = cast_qkv(q, k, v, precision)

  if fold_q_sequence_heads:
    q, bias, mask, _, q_indices = base.fold_q_sequence_heads(
        q, bias, mask, None, q_indices, k.shape[-3], k.shape[-2]
    )

  if (
      mask is not None
      and supports_local_3d
      and split_k == 1
      and q_indices is None
      and k_indices is None
      and not fold_q_sequence_heads
  ):
    mask, _ = mask.take("local_3d_config")

  mask, is_causal, k_start, k_end = decompose_mask(
      mask, q, k, q_indices, k_indices
  )

  bias = _broadcast_to_rank(bias, q.ndim)
  mask = _broadcast_to_rank(mask, q.ndim)
  k_start = _broadcast_to_rank(k_start, q.ndim - 1)
  k_end = _broadcast_to_rank(k_end, q.ndim - 1)
  return q, k, v, bias, mask, is_causal, k_start, k_end


@functools.lru_cache(maxsize=128)
def _build_3d_kv_tile_schedule_np(
    local_3d_config: base.Local3DConfig,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    q_tile_size: int,
    block_kv: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Precomputes the active KV tile indices per Q tile in NumPy."""
  latent_shape = local_3d_config.latent_shape
  window_sizes = local_3d_config.window_sizes
  sfc_block_sizes = local_3d_config.sfc_block_sizes
  orig_q_len = local_3d_config.q_seq_len
  orig_kv_len = local_3d_config.kv_seq_len
  flat_latent_len = math.prod(latent_shape)
  ndim = len(latent_shape)

  if sfc_block_sizes is None:
    radices = tuple(latent_shape)
  else:
    blocks = tuple(
        dim // block for dim, block in zip(latent_shape, sfc_block_sizes)
    )
    radices = blocks + tuple(sfc_block_sizes)

  def _unravel_idx(idx: np.ndarray) -> np.ndarray:
    unraveled = np.unravel_index(idx, radices)
    if sfc_block_sizes is None:
      return np.stack(unraveled, axis=-1).astype(np.int32)
    return np.stack(
        [
            unraveled[d] * sfc_block_sizes[d] + unraveled[ndim + d]
            for d in range(ndim)
        ],
        axis=-1,
    ).astype(np.int32)

  def _tile_coord_boxes(num_tiles: int, tile_size: int):
    start_idx = np.minimum(
        flat_latent_len, np.arange(num_tiles, dtype=np.int64) * tile_size
    )
    end_idx = np.minimum(
        flat_latent_len, (np.arange(num_tiles, dtype=np.int64) + 1) * tile_size
    )
    has_video = end_idx > start_idx
    block_starts = []
    block_ends = []
    stride = 1
    for radix in reversed(radices):
      if radix > 1:
        need_left = (start_idx < end_idx) & ((start_idx % radix) != 0)
        left_end = np.where(
            need_left,
            np.minimum(end_idx, (start_idx // radix + 1) * radix),
            start_idx,
        )
        if np.any(left_end > start_idx):
          block_starts.append(start_idx * stride)
          block_ends.append(left_end * stride)
        start_idx = left_end

        need_right = (start_idx < end_idx) & ((end_idx % radix) != 0)
        right_start = np.where(
            need_right,
            np.maximum(start_idx, (end_idx // radix) * radix),
            end_idx,
        )
        if np.any(end_idx > right_start):
          block_starts.append(right_start * stride)
          block_ends.append(end_idx * stride)
        end_idx = right_start

        start_idx = start_idx // radix
        end_idx = end_idx // radix
        stride *= radix
    if np.any(end_idx > start_idx) or not block_starts:
      block_starts.append(start_idx * stride)
      block_ends.append(end_idx * stride)

    b_start = np.stack(block_starts, axis=1)
    b_end = np.stack(block_ends, axis=1)
    valid = b_end > b_start
    safe_start = np.where(valid, b_start, 0)
    safe_last = np.where(valid, b_end - 1, 0)
    box_min = np.where(
        valid[..., None], _unravel_idx(safe_start), _SENTINEL_COORD
    )
    box_max = np.where(
        valid[..., None], _unravel_idx(safe_last), -_SENTINEL_COORD
    )
    tile_min = box_min.min(axis=1)
    tile_max = box_max.max(axis=1)
    is_single_box = valid.sum(axis=1) == 1
    return box_min, box_max, tile_min, tile_max, has_video, is_single_box

  num_q_tiles = pl.cdiv(q_seq_len, q_tile_size)
  num_kv_blocks = pl.cdiv(kv_seq_len, block_kv)

  q_box_min, q_box_max, q_vid_min, q_vid_max, q_has_video, q_single_box = (
      _tile_coord_boxes(num_q_tiles, q_tile_size)
  )
  (
      kv_box_min,
      kv_box_max,
      kv_vid_min,
      kv_vid_max,
      kv_has_video,
      kv_single_box,
  ) = _tile_coord_boxes(num_kv_blocks, block_kv)

  active = q_has_video[:, None] & kv_has_video[None, :]
  for d, (left_w, right_w) in enumerate(window_sizes):
    active &= (kv_vid_max[None, :, d] >= q_vid_min[:, None, d] - left_w) & (
        kv_vid_min[None, :, d] <= q_vid_max[:, None, d] + right_w
    )

  need_refine = active & ~(q_single_box[:, None] & kv_single_box[None, :])
  qi_idx, ki_idx = np.nonzero(need_refine)
  if len(qi_idx) > 0:
    q_bmin = q_box_min[qi_idx]
    q_bmax = q_box_max[qi_idx]
    k_bmin = kv_box_min[ki_idx]
    k_bmax = kv_box_max[ki_idx]
    pair_ok = np.ones(
        (len(qi_idx), q_bmin.shape[1], k_bmin.shape[1]), dtype=bool
    )
    for d, (left_w, right_w) in enumerate(window_sizes):
      pair_ok &= (k_bmax[:, None, :, d] >= q_bmin[:, :, None, d] - left_w) & (
          k_bmin[:, None, :, d] <= q_bmax[:, :, None, d] + right_w
      )
    active[qi_idx, ki_idx] = pair_ok.any(axis=(1, 2))

  if orig_q_len > flat_latent_len or orig_kv_len > flat_latent_len:
    q_tile_starts = np.arange(num_q_tiles) * q_tile_size
    q_tile_ends = np.minimum(q_tile_starts + q_tile_size, orig_q_len)
    q_has_valid = q_tile_starts < orig_q_len
    q_has_cond = (q_tile_ends > flat_latent_len) & q_has_valid

    kv_tile_starts = np.arange(num_kv_blocks) * block_kv
    kv_tile_ends = np.minimum(kv_tile_starts + block_kv, orig_kv_len)
    kv_has_valid = kv_tile_starts < orig_kv_len
    kv_has_cond = (kv_tile_ends > flat_latent_len) & kv_has_valid

    if local_3d_config.attend_over_conditioning:
      active |= (q_has_cond[:, None] & kv_has_valid[None, :]) | (
          q_has_valid[:, None] & kv_has_cond[None, :]
      )
    else:
      active |= q_has_cond[:, None] & kv_has_cond[None, :]

  num_kv_tiles = active.sum(axis=-1).astype(np.int32)
  max_kv_tiles = max(1, int(num_kv_tiles.max()))
  kv_tiles = np.argsort(~active, axis=-1, kind="stable")[
      :, :max_kv_tiles
  ].astype(np.int32)
  return kv_tiles, num_kv_tiles


def build_3d_kv_tile_schedule(
    local_3d_config: base.Local3DConfig,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    q_tile_size: int,
    block_kv: int,
) -> tuple[jax.Array, jax.Array]:
  """Precomputes the active KV tile indices per Q tile for 3D local attention."""
  kv_tiles, num_kv_tiles = _build_3d_kv_tile_schedule_np(
      local_3d_config,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      q_tile_size=q_tile_size,
      block_kv=block_kv,
  )
  return jnp.asarray(kv_tiles, dtype=jnp.int32), jnp.asarray(
      num_kv_tiles, dtype=jnp.int32
  )


def make_3d_mask_fn(
    q_base: jax.Array,
    q_iota_1d_fn: Any,
    iota_fn: Any,
    acc_shape: tuple[int, int],
    local_3d_config: base.Local3DConfig,
):
  """Precomputes 1D Q-tile 3D window bounds and returns a division-free per-KV-tile mask fn."""
  q_idx_1d = q_base + q_iota_1d_fn()
  q_coords_1d = base.unravel_3d_coords(
      q_idx_1d, local_3d_config.latent_shape, local_3d_config.sfc_block_sizes
  )
  q_bounds_1d = [
      (qc - left_w, left_w + right_w)
      for qc, (left_w, right_w) in zip(
          q_coords_1d, local_3d_config.window_sizes
      )
  ]
  flat_latent_len = math.prod(local_3d_config.latent_shape)
  has_conditioning = (
      local_3d_config.q_seq_len > flat_latent_len
      or local_3d_config.kv_seq_len > flat_latent_len
  )
  is_q_cond_1d = (q_idx_1d >= flat_latent_len) if has_conditioning else None
  _, block_kv = acc_shape
  latent_shape = local_3d_config.latent_shape
  sfc_block_sizes = local_3d_config.sfc_block_sizes

  raster_perm = None
  if len(latent_shape) == 3:
    if sfc_block_sizes is None or sfc_block_sizes == (1, 1, 1):
      raster_perm = (0, 1, 2)
    elif sfc_block_sizes == (1, latent_shape[1], 1):
      raster_perm = (0, 2, 1)
    elif sfc_block_sizes == (latent_shape[0], 1, 1):
      raster_perm = (1, 2, 0)

  if raster_perm is not None:
    eff_latent_shape = tuple(latent_shape[i] for i in raster_perm)
    eff_q_bounds_1d = [q_bounds_1d[i] for i in raster_perm]
  else:
    eff_latent_shape = latent_shape
    eff_q_bounds_1d = q_bounds_1d

  use_fast_3d = (
      raster_perm is not None
      and block_kv <= eff_latent_shape[1] * eff_latent_shape[2]
      and eff_latent_shape[2] + block_kv <= 512
  )
  use_fast_sfc_t_bh_1 = (
      not use_fast_3d
      and is_q_cond_1d is None
      and len(latent_shape) == 3
      and sfc_block_sizes is not None
      and len(sfc_block_sizes) == 3
      and sfc_block_sizes[0] == latent_shape[0]
      and sfc_block_sizes[2] == 1
      and sfc_block_sizes[1] > 1
      and latent_shape[1] % sfc_block_sizes[1] == 0
      and latent_shape[0] * sfc_block_sizes[1] <= 256
      and block_kv <= latent_shape[2] * latent_shape[0] * sfc_block_sizes[1]
  )

  def apply_3d_mask(scores: jax.Array, k_base: jax.Array) -> jax.Array:
    bcast_row = lambda x: lax.broadcast_in_dim(x, acc_shape, (0,))
    if use_fast_sfc_t_bh_1:
      assert sfc_block_sizes is not None
      t_size, h_size, w_size = latent_shape
      bh_size = sfc_block_sizes[1]
      hb_size = h_size // bh_size
      rem0 = lax.div(k_base, bh_size)
      hi0 = lax.rem(k_base, bh_size)
      rem1 = lax.div(rem0, t_size)
      t0 = lax.rem(rem0, t_size)
      hb0 = lax.div(rem1, w_size)
      w0 = lax.rem(rem1, w_size)

      col_offset = iota_fn(1)
      u_hi = hi0 + col_offset
      magic_bh = (_RECIP_DIV_SCALE + bh_size - 1) // bh_size
      dt_step = lax.shift_right_logical(u_hi * magic_bh, _RECIP_DIV_SHIFT)
      khi = u_hi - dt_step * bh_size

      u_t = t0 + dt_step
      magic_t = (_RECIP_DIV_SCALE + t_size - 1) // t_size
      dw_step = lax.shift_right_logical(u_t * magic_t, _RECIP_DIV_SHIFT)
      kt = u_t - dw_step * t_size

      u_w = w0 + dw_step
      w_wrap = u_w >= w_size
      kw = jnp.where(w_wrap, u_w - w_size, u_w)
      khb = jnp.where(w_wrap, hb0 + 1, hb0)
      kh = khb * bh_size + khi

      (q_min_t, span_t), (q_min_h, span_h), (q_min_w, span_w) = q_bounds_1d
      dt = kt - bcast_row(q_min_t)
      dh = kh - bcast_row(q_min_h)
      dw = kw - bcast_row(q_min_w)
      mask = (
          (dt >= 0)
          & (dt <= span_t)
          & (dh >= 0)
          & (dh <= span_h)
          & (dw >= 0)
          & (dw <= span_w)
          & (khb < hb_size)
      )
    elif use_fast_3d:
      col_offset = iota_fn(1)
      t_size, h_size, w_size = eff_latent_shape
      rem0 = lax.div(k_base, w_size)
      w0 = lax.rem(k_base, w_size)
      t0 = lax.div(rem0, h_size)
      h0 = lax.rem(rem0, h_size)
      w_unwrapped = w0 + col_offset
      if w_size >= block_kv:
        w_wrap = w_unwrapped >= w_size
        kw = jnp.where(w_wrap, w_unwrapped - w_size, w_unwrapped)
        dh_step = jnp.where(w_wrap, 1, 0)
      else:
        magic_w = (_RECIP_DIV_SCALE + w_size - 1) // w_size
        dh_step = lax.shift_right_logical(
            w_unwrapped * magic_w, _RECIP_DIV_SHIFT
        )
        kw = w_unwrapped - dh_step * w_size
      q_min_w, span_w = eff_q_bounds_1d[2]
      dw = kw - bcast_row(q_min_w)
      mask = (dw >= 0) & (dw <= span_w)

      h_sum = h0 + dh_step
      h_wrap = h_sum >= h_size
      kh = jnp.where(h_wrap, h_sum - h_size, h_sum)
      q_min_h, span_h = eff_q_bounds_1d[1]
      dh = kh - bcast_row(q_min_h)
      mask &= (dh >= 0) & (dh <= span_h)

      q_min_t, span_t = eff_q_bounds_1d[0]
      dt0 = t0 - q_min_t
      t_ok_0 = bcast_row((t0 < t_size) & (dt0 >= 0) & (dt0 <= span_t))
      t_ok_1 = bcast_row(
          (t0 + 1 < t_size) & (dt0 + 1 >= 0) & (dt0 + 1 <= span_t)
      )
      mask &= jnp.where(h_wrap, t_ok_1, t_ok_0)
    else:
      col_offset = iota_fn(1)
      k_idx = k_base + col_offset
      k_coords = base.unravel_3d_coords(
          k_idx, latent_shape, local_3d_config.sfc_block_sizes
      )
      mask = None
      for kc, (q_min, span) in zip(k_coords, q_bounds_1d):
        diff = kc - bcast_row(q_min)
        in_dim = (diff >= 0) & (diff <= span)
        mask = in_dim if mask is None else (mask & in_dim)
      assert mask is not None
      if is_q_cond_1d is None:
        mask &= k_idx < flat_latent_len

    if is_q_cond_1d is not None:
      col_offset = iota_fn(1)
      is_q_cond = bcast_row(is_q_cond_1d)
      is_k_cond = (k_base + col_offset) >= flat_latent_len
      if local_3d_config.attend_over_conditioning:
        mask = mask | is_q_cond | is_k_cond
      else:
        mask = (mask & ~is_q_cond & ~is_k_cond) | (is_q_cond & is_k_cond)
    return jnp.where(mask, scores, -jnp.inf)

  return apply_3d_mask


def eval_input_shapes(
    ba: op_lib.BoundArguments, *, fold_q_sequence_heads: bool, split_k: int
):
  """Evaluates the shape of the inputs after `prepare_inputs`."""
  q, k, v, bias, mask, _, _, _ = jax.eval_shape(
      functools.partial(
          prepare_inputs,
          precision=ba.kwargs["precision"],
          fold_q_sequence_heads=fold_q_sequence_heads,
          split_k=split_k,
      ),
      *ba.args,
      bias=ba.kwargs["bias"],
      mask=ba.kwargs["mask"],
      q_indices=ba.kwargs["q_indices"],
      k_indices=ba.kwargs["k_indices"],
  )

  if split_k > 1:
    k_seq_len = pl.cdiv(k.shape[-3], split_k)
    k_shape = (split_k, *k.shape[:-3], k_seq_len, *k.shape[-2:])
    k = jax.ShapeDtypeStruct(k_shape, k.dtype)
    v = jax.ShapeDtypeStruct(k_shape[:-1] + (v.shape[-1],), v.dtype)
    if bias is not None:
      bias_shape = (split_k, *bias.shape[:-1], k_seq_len)
      bias = jax.ShapeDtypeStruct(bias_shape, bias.dtype)
    if mask is not None:
      mask_shape = (split_k, *mask.shape[:-1], k_seq_len)
      mask = jax.ShapeDtypeStruct(mask_shape, mask.dtype)

  return q, k, v, bias, mask


def load_bcast(
    ref: Any,
    idx: tuple[int | jax.Array | pl.Slice, ...],
    *,
    layout: Any,
    optimized: bool = False,
) -> jax.Array:
  """Loads from a reference, with given index, broadcasting if needed."""
  new_idx = []
  shape = []
  bcast_dims = []
  ld_layout = layout
  # NOTE: We could add support for `idx` shorter than `ref.ndim`.
  for d, ix in zip(ref.shape, idx, strict=True):
    new_idx.append(0 if d == 1 else ix)

    if isinstance(ix, pl.Slice):
      if d == 1:
        ld_layout = ld_layout.reduce(len(shape))
      else:
        bcast_dims.append(len(shape))
      shape.append(ix.size)

  new_idx = tuple(new_idx)
  if bcast_dims:
    value = plgpu.load(ref.at[new_idx], layout=ld_layout, optimized=optimized)
  else:  # Scalar value.
    value = ref[new_idx]
  value = jax.lax.broadcast_in_dim(value, shape, bcast_dims)
  return value if layout is None else plgpu.layout_cast(value, layout)


def unpack_bool_bits_tmem_native(a):
  """Unpacks boolean bits from an int packed array in TMEM_NATIVE layout."""
  packed_bits = mgpu_lib.num_bits(a.dtype)
  if packed_bits not in {4, 8, 16}:
    raise ValueError("Only 4, 8, 16 boolean packing is supported")
  target_cols = a.shape[1] * packed_bits
  out_layout = plgpu.Layout.TCGEN05_TMEM_NATIVE
  out_mgpu_layout = out_layout.to_mgpu()

  @plgpu.inline_mgpu(
      arg_types=(plgpu.Layout.TCGEN05_TMEM_NATIVE(32 // packed_bits),),
      return_type=plgpu.ShapeDtypeStruct(
          (128, target_cols),
          jnp.bool_,  # pyrefly: ignore[bad-argument-type]
          out_layout,
      ),
  )
  def unpack_booleans(_, fa: mgpu.FragmentedArray):
    out_registers = np.empty(
        out_mgpu_layout.registers_shape((128, target_cols)),
        dtype=object,
    )
    i32_type = ir.IntegerType.get_signless(32)
    out_i32_ty = ir.VectorType.get([2], i32_type)
    zero_i32_vec = vector.broadcast(out_i32_ty, mgpu.c(0, i32_type))

    for (row_idx, col_idx, *_), reg_a in np.ndenumerate(fa.registers):
      vec1_i32_ty = ir.VectorType.get([1], i32_type)
      reg_v1_i32 = vector.bitcast(vec1_i32_ty, reg_a)
      reg_i32_scalar = vector.extract(
          reg_v1_i32,
          dynamic_position=[],
          static_position=ir.DenseI64ArrayAttr.get([0]),
      )
      reg_v2_i32 = vector.broadcast(out_i32_ty, reg_i32_scalar)

      for i in range(16):
        bit_masks = np.array([1 << (i * 2), 1 << (i * 2 + 1)], dtype=np.uint32)
        bit_mask_attr = ir.DenseIntElementsAttr.get(bit_masks, type=out_i32_ty)
        bit_mask = arith.constant(out_i32_ty, bit_mask_attr)
        and_res = arith.andi(reg_v2_i32, bit_mask)
        cmp_res = arith.cmpi(arith.CmpIPredicate.ne, and_res, zero_i32_vec)
        logical_col = col_idx * 16 + i
        out_registers[row_idx, logical_col] = cmp_res

    return mgpu.FragmentedArray(
        _registers=out_registers,
        _layout=out_mgpu_layout,
        _is_signed=False,
    )

  return unpack_booleans(a)

def safe_downcast(
    arr: jax.Array, target_dtype: jax.typing.DTypeLike
) -> jax.Array:
  """Clips the array to the target dtype's range before casting to prevent infinities."""
  if arr.dtype == target_dtype:
    return arr
  finfo = jnp.finfo(target_dtype)
  return jnp.clip(arr, finfo.min, finfo.max).astype(target_dtype)
