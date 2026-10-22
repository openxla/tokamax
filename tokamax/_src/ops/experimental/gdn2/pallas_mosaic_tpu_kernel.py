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

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_CHUNK_SIZE = 64
_MAX_MB = 16


def _select_mb(bh: int, max_mb: int = _MAX_MB) -> int:
  """Selects largest divisor of `bh` up to `max_mb` for stream batching."""
  for mb in range(min(bh, max_mb), 0, -1):
    if bh % mb == 0:
      return mb
  return 1


def _chunkify(
    x: jax.Array, batch_size: int, num_chunks: int, chunk_size: int
) -> jax.Array:
  # (batch_size, seq_len, num_heads, ...) -> (batch_size * num_heads, num_chunks, chunk_size, ...)
  num_heads = x.shape[2]
  rest = x.shape[3:]
  x = x.reshape((batch_size, num_chunks, chunk_size, num_heads) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  x_t = jnp.transpose(x, perm)
  return x_t.reshape((batch_size * num_heads, num_chunks, chunk_size) + rest)


def _unchunk(
    x_c: jax.Array,
    rest_dim: int,
    batch_size: int,
    num_chunks: int,
    chunk_size: int,
    num_heads: int,
) -> jax.Array:
  x_ = x_c.reshape((batch_size, num_heads, num_chunks, chunk_size, rest_dim))
  x_ = jnp.transpose(x_, (0, 2, 3, 1, 4))
  return x_.reshape(batch_size, num_chunks * chunk_size, num_heads, rest_dim)


def _decayed_gram(
    a: jax.Array,
    b: jax.Array,
    g_a: jax.Array,
    g_b: jax.Array,
) -> jax.Array:
  """Computes decayed Gram matrix `(a @ b^T) * exp(g_a - g_b)` over (..., BT, K)."""
  g0 = g_b[..., :1, :]
  lhs = (a * jnp.exp(jnp.minimum(g_a - g0, 80.0))).astype(jnp.float32)
  rhs = (b * jnp.exp(jnp.minimum(g0 - g_b, 80.0))).astype(jnp.float32)
  out = jnp.matmul(
      lhs,
      jnp.swapaxes(rhs, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return out.astype(a.dtype)


def _unit_lower_inverse(a0: jax.Array) -> jax.Array:
  """Computes `(I - a0)^{-1}` for strictly lower-triangular `a0` via Schulz doubling."""
  bt = a0.shape[-1]
  p = a0.astype(jnp.float32)
  row = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  eye = (row == col).astype(jnp.float32)
  inv = eye + p
  for _ in range(bt.bit_length() - 2):
    p = jnp.matmul(
        p,
        p,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    inv = inv + jnp.matmul(
        inv,
        p,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
  return inv.astype(a0.dtype)


def _gdn2_fwd_kernel(
    # Inputs.
    q_ref,  # [MB, 1, BT, K] dtype
    k_ref,  # [MB, 1, BT, K] dtype
    v_ref,  # [MB, 1, BT, V] dtype
    g_ref,  # [MB, 1, BT, K] float32
    b_ref,  # [MB, 1, BT, K] dtype
    w_ref,  # [MB, 1, BT, V] dtype
    s0_ref,  # [MB, K, V] float32
    # Outputs.
    o_ref,  # [MB, 1, BT, V] dtype
    s_final_ref,  # [MB, K, V] float32
    a_mat_ref,  # [MB, 1, BT, BT] dtype
    merged_out_ref,  # [MB, 1, BT, K + V] float32
    s_in_ref,  # [MB, 1, K, V] float32
    # Scratch.
    scratch_ref,  # [MB, K, V] float32
    *,
    num_chunks: int,
) -> None:
  """Single-pass MB-batched intra-chunk WY + in-VMEM state recurrence kernel."""
  c = pl.program_id(1)
  bt = q_ref.shape[2]
  k_dim = q_ref.shape[3]

  @pl.when(c == 0)
  def _():
    scratch_ref[...] = s0_ref[...].astype(jnp.float32)

  s_prev = scratch_ref[...]
  s_in_ref[:, 0] = s_prev.astype(s_in_ref.dtype)

  q_i = q_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  k_i = k_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  v_i = v_ref[:, 0].astype(jnp.float32)  # [MB, BT, V]
  g_i = g_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  b_i = b_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  w_i = w_ref[:, 0].astype(jnp.float32)  # [MB, BT, V]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)

  bk = b_i * k_i
  t_lower = -_decayed_gram(bk, k_i, g_i, g_i) * strictly_lower
  a_mat = _unit_lower_inverse(t_lower)

  exp_g = jnp.exp(g_i)
  k_g_b = exp_g * bk
  wv = w_i * v_i
  merged_rhs = jnp.concatenate([k_g_b, wv], axis=-1)
  merged_out = jnp.matmul(
      a_mat,
      merged_rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  aqk_i = _decayed_gram(q_i, k_i, g_i, g_i) * causal_keep

  a_mat_ref[:, 0] = a_mat.astype(a_mat_ref.dtype)
  merged_out_ref[:, 0] = merged_out.astype(merged_out_ref.dtype)

  wwy_i = merged_out[:, :, :k_dim]
  u_i = merged_out[:, :, k_dim:]

  v_new = u_i - jnp.matmul(
      wwy_i,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  o_i = jnp.matmul(
      q_i * exp_g,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) + jnp.matmul(
      aqk_i,
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]  # [MB, K]
  decayed_k = jnp.exp(g_last[:, None, :] - g_i) * k_i
  s_update = jnp.matmul(
      jnp.swapaxes(decayed_k, -1, -2),
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s_new = s_prev * jnp.exp(g_last)[:, :, None] + s_update

  scratch_ref[...] = s_new
  o_ref[:, 0] = o_i.astype(o_ref.dtype)

  @pl.when(c == num_chunks - 1)
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
  """Runs fused single-pass MB-batched intra-chunk WY + state recurrence."""
  bh, num_chunks, chunk_size, key_dim = q_c.shape
  val_dim = v_c.shape[-1]
  mb = _select_mb(bh)
  grid = (bh // mb, num_chunks)
  idx = lambda i, n: (i, n, 0, 0)  # noqa: E731
  bspec_k = pl.BlockSpec((mb, 1, chunk_size, key_dim), idx)
  bspec_v = pl.BlockSpec((mb, 1, chunk_size, val_dim), idx)
  bspec_bt = pl.BlockSpec((mb, 1, chunk_size, chunk_size), idx)
  bspec_merged = pl.BlockSpec((mb, 1, chunk_size, key_dim + val_dim), idx)
  bspec_s = pl.BlockSpec((mb, key_dim, val_dim), lambda i, n: (i, 0, 0))
  bspec_s_in = pl.BlockSpec((mb, 1, key_dim, val_dim), idx)

  interpret = jax.default_backend() == "cpu"
  return pl.pallas_call(
      functools.partial(_gdn2_fwd_kernel, num_chunks=num_chunks),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_k, bspec_v, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_bt, bspec_merged, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((mb, key_dim, val_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((bh, num_chunks, chunk_size, val_dim), out_dtype),
          jax.ShapeDtypeStruct((bh, key_dim, val_dim), jnp.float32),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, chunk_size), q_c.dtype
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, key_dim + val_dim), jnp.float32
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, key_dim, val_dim), jnp.float32
          ),
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
) -> (
    tuple[jax.Array, jax.Array | None]
    | tuple[
        tuple[jax.Array, jax.Array | None],
        tuple[jax.Array, jax.Array, jax.Array],
    ]
):
  """Computes GDN-2 forward pass using fused MB-batched Pallas kernel."""
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )

  batch_size, seq_len, num_heads, key_dim = q.shape
  val_dim = v.shape[-1]
  bh = batch_size * num_heads
  num_chunks = seq_len // chunk_size
  if scale is None:
    scale = key_dim**-0.5

  q_c = _chunkify(
      q.astype(jnp.float32) * scale, batch_size, num_chunks, chunk_size
  ).astype(q.dtype)
  k_c = _chunkify(k, batch_size, num_chunks, chunk_size)
  v_c = _chunkify(v, batch_size, num_chunks, chunk_size)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), batch_size, num_chunks, chunk_size),
      axis=-2,
  )
  b_c = _chunkify(b, batch_size, num_chunks, chunk_size)
  w_c = _chunkify(w, batch_size, num_chunks, chunk_size)

  if initial_state is None:
    s0 = jnp.zeros((bh, key_dim, val_dim), dtype=jnp.float32)
  else:
    s0 = initial_state.reshape((bh, key_dim, val_dim)).astype(jnp.float32)

  o_c, s_final, a_mat_c, merged_out_c, s_in_c = _gdn2_fwd_pallas(
      q_c, k_c, v_c, g_c, b_c, w_c, s0, v_c.dtype
  )

  o = _unchunk(o_c, val_dim, batch_size, num_chunks, chunk_size, num_heads)
  final_state = (
      s_final.reshape((batch_size, num_heads, key_dim, val_dim)).astype(q.dtype)
      if output_final_state
      else None
  )
  if return_residuals:
    return (o, final_state), (a_mat_c, merged_out_c, s_in_c)
  return o, final_state
