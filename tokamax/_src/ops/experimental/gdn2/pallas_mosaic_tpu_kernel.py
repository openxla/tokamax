# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Pallas TPU compute kernel for Gated Delta Net 2 (GDN-2)."""

# Dimensions.
#   B   batch size
#   T   sequence length (nt * tt)
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension
#   BH  flattened batch-head count (B * H)
#   tb  tile extent of BH (micro-batch size dividing BH)
#   tt  tile extent of T (chunk_size)
#   nb  grid extent along BH (BH // tb)
#   nt  grid extent along T (T // tt, number of chunks)

import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_CHUNK_SIZE = 64
_MAX_TB = 16


def _select_tb(BH: int, max_tb: int = _MAX_TB) -> int:
  """Selects largest divisor of `BH` up to `max_tb` for stream batching."""
  for tb in range(min(BH, max_tb), 0, -1):
    if BH % tb == 0:
      return tb
  return 1


def _chunkify(x: jax.Array, B: int, nt: int, tt: int) -> jax.Array:
  # [B, T, H, ...] -> [BH, nt, tt, ...]
  _, _, H, *rest = x.shape
  rest_shape = tuple(rest)
  x = x.reshape((B, nt, tt, H) + rest_shape)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  x_t = jnp.transpose(x, perm)
  return x_t.reshape((B * H, nt, tt) + rest_shape)


def _unchunk(
    x_c: jax.Array,
    D: int,
    B: int,
    nt: int,
    tt: int,
    H: int,
) -> jax.Array:
  x_ = x_c.reshape((B, H, nt, tt, D))
  x_ = jnp.transpose(x_, (0, 2, 3, 1, 4))
  return x_.reshape(B, nt * tt, H, D)


def _decayed_gram(
    lhs_in: jax.Array,
    rhs_in: jax.Array,
    g_lhs: jax.Array,
    g_rhs: jax.Array,
) -> jax.Array:
  """Computes decayed Gram matrix `(lhs_in @ rhs_in^T) * exp(g_lhs - g_rhs)`."""
  # Shift log-decays by the chunk's first token `g0` and clamp at 80.0 so the
  # factored `exp(g_lhs - g0)` and `exp(g0 - g_rhs)` terms never overflow f32
  # before contracting on the MXU.
  g0 = g_rhs[..., :1, :]
  lhs = (lhs_in * jnp.exp(jnp.minimum(g_lhs - g0, 80.0))).astype(jnp.float32)
  rhs = (rhs_in * jnp.exp(jnp.minimum(g0 - g_rhs, 80.0))).astype(jnp.float32)
  out = jnp.matmul(
      lhs,
      jnp.swapaxes(rhs, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return out.astype(lhs_in.dtype)


def _unit_lower_inverse(a0: jax.Array) -> jax.Array:
  """Computes `(I - a0)^{-1}` for strictly lower-triangular `a0` via Schulz doubling."""
  *_, tt, _ = a0.shape
  power = a0.astype(jnp.float32)
  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  eye = (row_idx == col_idx).astype(jnp.float32)
  inv = eye + power
  # Because `a0` is strictly lower-triangular of size `tt`, `a0^tt == 0`, so
  # `ceil(log2(tt)) - 1` squaring steps compute the exact inverse on the MXU.
  num_doubling_steps = max(0, math.ceil(math.log2(tt)) - 1)
  for _ in range(num_doubling_steps):
    power = jnp.matmul(
        power,
        power,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    inv = inv + jnp.matmul(
        inv,
        power,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
  return inv.astype(a0.dtype)


def _gdn2_fwd_kernel(
    # Inputs.
    q_ref,  # [tb, 1, tt, K]
    k_ref,  # [tb, 1, tt, K]
    v_ref,  # [tb, 1, tt, V]
    g_ref,  # [tb, 1, tt, K] f32
    b_ref,  # [tb, 1, tt, K]
    w_ref,  # [tb, 1, tt, V]
    s0_ref,  # [tb, K, V] f32
    # Outputs.
    out_ref,  # [tb, 1, tt, V]
    s_final_ref,  # [tb, K, V] f32
    a_mat_ref,  # [tb, 1, tt, tt]
    merged_out_ref,  # [tb, 1, tt, K + V] f32
    s_in_ref,  # [tb, 1, K, V] f32
    # Scratch.
    s_scratch_ref,  # [tb, K, V] f32   persists across the nt chunk grid axis
    *,
    nt: int,
) -> None:
  """Single-pass `tb`-batched intra-chunk WY + in-VMEM state recurrence kernel."""
  t_i = pl.program_id(1)
  _, _, tt, K = q_ref.shape

  # Seed the persistent VMEM state buffer from `s0_ref` on the first chunk;
  # subsequent chunks along axis 1 reuse `s_scratch_ref` in place without HBM round-trips.
  @pl.when(t_i == 0)
  def _():
    s_scratch_ref[...] = s0_ref[...].astype(jnp.float32)

  s_prev = s_scratch_ref[...]
  s_in_ref[:, 0] = s_prev.astype(s_in_ref.dtype)

  q_tile = q_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  k_tile = k_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  v_tile = v_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]
  g_tile = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  b_tile = b_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  w_tile = w_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)

  bk = b_tile * k_tile
  t_lower = -_decayed_gram(bk, k_tile, g_tile, g_tile) * strictly_lower
  a_mat = _unit_lower_inverse(t_lower)

  exp_g = jnp.exp(g_tile)
  k_g_b = exp_g * bk
  wv = w_tile * v_tile
  # Concatenate `k_g_b` and `wv` along the feature axis so `(I + T)^{-1}` is
  # applied to both WY factors in a single `[tb, tt, tt] x [tb, tt, K + V]` MXU matmul.
  merged_rhs = jnp.concatenate([k_g_b, wv], axis=-1)
  merged_out = jnp.matmul(
      a_mat,
      merged_rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  aqk_tile = _decayed_gram(q_tile, k_tile, g_tile, g_tile) * causal_keep

  a_mat_ref[:, 0] = a_mat.astype(a_mat_ref.dtype)
  merged_out_ref[:, 0] = merged_out.astype(merged_out_ref.dtype)

  wwy_tile = merged_out[:, :, :K]
  u_tile = merged_out[:, :, K:]

  v_new = u_tile - jnp.matmul(
      wwy_tile,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  out_tile = jnp.matmul(
      q_tile * exp_g,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) + jnp.matmul(
      aqk_tile,
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_tile[:, -1, :]  # [tb, K]
  decayed_k = jnp.exp(g_last[:, None, :] - g_tile) * k_tile
  s_update = jnp.matmul(
      jnp.swapaxes(decayed_k, -1, -2),
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s_new = s_prev * jnp.exp(g_last)[:, :, None] + s_update

  s_scratch_ref[...] = s_new
  out_ref[:, 0] = out_tile.astype(out_ref.dtype)

  @pl.when(t_i == nt - 1)
  def _():
    s_final_ref[...] = s_new.astype(s_final_ref.dtype)


def _gdn2_fwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    g_c: jax.Array,
    b_c: jax.Array,
    w_c: jax.Array,
    s0: jax.Array,
    out_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs fused single-pass `tb`-batched intra-chunk WY + state recurrence."""
  BH, nt, tt, K = q_c.shape
  _, _, _, V = v_c.shape
  tb = _select_tb(BH)
  nb = BH // tb
  grid = (nb, nt)
  idx = lambda b_i, t_i: (b_i, t_i, 0, 0)  # noqa: E731
  bspec_k = pl.BlockSpec((tb, 1, tt, K), idx)
  bspec_v = pl.BlockSpec((tb, 1, tt, V), idx)
  bspec_tt = pl.BlockSpec((tb, 1, tt, tt), idx)
  bspec_merged = pl.BlockSpec((tb, 1, tt, K + V), idx)
  bspec_s = pl.BlockSpec((tb, K, V), lambda b_i, t_i: (b_i, 0, 0))
  bspec_s_in = pl.BlockSpec((tb, 1, K, V), idx)

  interpret = jax.default_backend() != "tpu"
  return pl.pallas_call(
      functools.partial(_gdn2_fwd_kernel, nt=nt),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_k, bspec_v, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_tt, bspec_merged, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((tb, K, V), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, V), out_dtype),
          jax.ShapeDtypeStruct((BH, K, V), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, tt, tt), q_c.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, K + V), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, K, V), jnp.float32),
      ],
      interpret=interpret,
  )(q_c, k_c, v_c, g_c, b_c, w_c, s0)


def gdn2_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> tuple[Any, ...]:
  """Computes GDN-2 forward pass using fused `tb`-batched Pallas kernel."""
  B, T, H, K = q.shape
  _, _, _, V = v.shape
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )

  BH = B * H
  tt = chunk_size
  nt = T // tt
  if scale is None:
    scale = float(K**-0.5)

  q_c = _chunkify(q.astype(jnp.float32) * scale, B, nt, tt).astype(q.dtype)
  k_c = _chunkify(k, B, nt, tt)
  v_c = _chunkify(v, B, nt, tt)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  )
  b_c = _chunkify(b, B, nt, tt)
  w_c = _chunkify(w, B, nt, tt)

  if initial_state is None:
    s0 = jnp.zeros((BH, K, V), dtype=jnp.float32)
  else:
    s0 = initial_state.reshape((BH, K, V)).astype(jnp.float32)

  out_c, s_final, a_mat_c, merged_out_c, s_in_c = _gdn2_fwd_pallas(
      q_c, k_c, v_c, g_c, b_c, w_c, s0, v_c.dtype
  )

  out = _unchunk(out_c, V, B, nt, tt, H)
  final_state = (
      s_final.reshape((B, H, K, V)).astype(q.dtype)
      if output_final_state
      else None
  )
  if return_residuals:
    return (out, final_state), (a_mat_c, merged_out_c, s_in_c)
  return out, final_state
