# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Kimi Delta Attention (KDA) custom TPU Pallas kernels for PyTorch TPU."""

import functools
import logging
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch

_TORCH_TPU_PALLAS_AVAILABLE = False
try:
  from torch_tpu._internal import pallas as _pallas  # pylint: disable=g-import-not-at-top

  _TORCH_TPU_PALLAS_AVAILABLE = True
except ImportError:
  _pallas = None

_CHUNK_SIZE = 64


def _select_mb(bh: int, max_mb: int = 16) -> int:
  """Selects the largest batch-head tile `mb <= max_mb` dividing `bh`."""
  for mb in range(min(bh, max_mb), 0, -1):
    if bh % mb == 0:
      return mb
  return 1


def _decayed_gram(
    a: jax.Array, b: jax.Array, g_a: jax.Array, g_b: jax.Array
) -> jax.Array:
  """Computes decayed Gram matrix via factored 2D MXU matmul.

  Computes `out[..., r, c] = sum_d a[..., r, d] * b[..., c, d] *
  exp(g_a[..., r, d] - g_b[..., c, d])` as a single `(..., BT, K) x (..., K, BT)`
  matmul instead of materializing a `(..., BT, BT, K)` elementwise tensor, via
  `exp(g_a - g0) * exp(g0 - g_b)` with `g0` the chunk's first cumulative-gate
  row. Supports both 2D `(BT, K)` and batched 3D `(MB, BT, K)` inputs.
  """
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


@jax.custom_vjp
def _unit_lower_inverse(a0: jax.Array) -> jax.Array:
  """Computes `(I - a0)^{-1}` for strictly lower-triangular `a0` via Schulz doubling.

  Evaluates `prod_j (I + a0^(2^j))` for `j < log2(BT)`: exact because `a0` is
  nilpotent (`a0^BT = 0`), so the Neumann series `sum_n a0^n` terminates at
  `n = BT - 1`. Supports both 2D `(BT, BT)` and batched 3D `(MB, BT, BT)`
  inputs.
  """
  bt = a0.shape[-1]
  if bt & (bt - 1) != 0:
    raise ValueError(f"chunk size {bt} must be a power of two")
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


def _unit_lower_inverse_fwd(a0: jax.Array) -> tuple[jax.Array, jax.Array]:
  inv = _unit_lower_inverse(a0)
  return inv, inv


def _unit_lower_inverse_bwd(
    inv: jax.Array, d_inv: jax.Array
) -> tuple[jax.Array]:
  """Computes `da0 = T^T dT T^T` restricted to strictly lower triangle."""
  bt = inv.shape[-1]
  t_t = jnp.swapaxes(inv.astype(jnp.float32), -1, -2)
  d_t_t = jnp.matmul(
      d_inv.astype(jnp.float32),
      t_t,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  d_a0 = jnp.matmul(
      t_t,
      d_t_t,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  lower = jnp.tril(jnp.ones((bt, bt), dtype=jnp.float32), k=-1)
  return ((d_a0 * lower).astype(inv.dtype),)


_unit_lower_inverse.defvjp(_unit_lower_inverse_fwd, _unit_lower_inverse_bwd)


def _chunkify(x: jax.Array, b: int, nt: int, bt: int) -> jax.Array:
  """Reshapes `(B, T, H, ...)` into `(B, H, NT, BT, ...)`."""
  rest = x.shape[3:]
  x = x.reshape((b, nt, bt, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _unchunk(
    x_c: jax.Array, rest_dim: int, b: int, nt: int, bt: int, h: int
) -> jax.Array:
  """Reshapes `(B, H, NT, BT, D)` back into `(B, T, H, D)`."""
  x_ = jnp.transpose(x_c, (0, 2, 3, 1, 4))
  return x_.reshape(b, nt * bt, h, rest_dim)


def _decayed_gram_bwd(
    d_out: jax.Array,
    a: jax.Array,
    b: jax.Array,
    g_a: jax.Array,
    g_b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Analytical backward of `_decayed_gram` using factored 2D MXU matmuls.

  Avoids materializing any `(..., BT, BT, K)` intermediate tensors. Supports
  both 2D `(BT, K)` and batched 3D `(MB, BT, K)` inputs.
  """
  g0 = g_b[..., :1, :]
  exp_a = jnp.exp(jnp.minimum(g_a - g0, 80.0)).astype(jnp.float32)
  exp_b = jnp.exp(jnp.minimum(g0 - g_b, 80.0)).astype(jnp.float32)
  lhs = a.astype(jnp.float32) * exp_a
  rhs = b.astype(jnp.float32) * exp_b

  d_lhs = jnp.matmul(
      d_out.astype(jnp.float32),
      rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  d_rhs = jnp.matmul(
      jnp.swapaxes(d_out.astype(jnp.float32), -1, -2),
      lhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  da = d_lhs * exp_a
  db = d_rhs * exp_b
  dg_a = d_lhs * lhs
  dg_b = -d_rhs * rhs
  return da, db, dg_a, dg_b


def _kda_fwd_kernel(
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    g_ref: Any,
    beta_ref: Any,
    s0_ref: Any,
    o_ref: Any,
    s_final_ref: Any,
    a_mat_ref: Any,
    merged_out_ref: Any,
    s_in_ref: Any,
    scratch_ref: Any,
    *,
    num_chunks: int,
) -> None:
  """Single-pass `MB`-batched Pallas forward kernel over `grid=(bh // mb, nt)`."""
  c = pl.program_id(1)
  bt = q_ref.shape[2]
  k_dim = q_ref.shape[3]

  @pl.when(c == 0)
  def _() -> None:
    scratch_ref[...] = s0_ref[:].astype(jnp.float32)

  s_prev = scratch_ref[...]
  s_in_ref[:, 0] = s_prev.astype(s_in_ref.dtype)

  q_i = q_ref[:, 0].astype(jnp.float32)
  k_i = k_ref[:, 0].astype(jnp.float32)
  v_i = v_ref[:, 0].astype(jnp.float32)
  g_i = g_ref[:, 0].astype(jnp.float32)
  beta_i = beta_ref[:, 0].astype(jnp.float32)

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)

  bk = beta_i * k_i
  t_lower = -_decayed_gram(bk, k_i, g_i, g_i) * strictly_lower
  a_mat = _unit_lower_inverse(t_lower)

  exp_g = jnp.exp(g_i)
  k_g_b = exp_g * bk
  wv = beta_i * v_i
  merged_rhs = jnp.concatenate([k_g_b, wv], axis=-1)
  merged_out = jnp.matmul(
      a_mat,
      merged_rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  wwy_i = merged_out[..., :k_dim]
  u_i = merged_out[..., k_dim:]

  aqk = _decayed_gram(q_i, k_i, g_i, g_i) * causal_keep

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
      aqk,
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]
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
  a_mat_ref[:, 0] = a_mat.astype(a_mat_ref.dtype)
  merged_out_ref[:, 0] = merged_out.astype(merged_out_ref.dtype)

  @pl.when(c == num_chunks - 1)
  def _() -> None:
    s_final_ref[:] = s_new.astype(s_final_ref.dtype)


def _kda_fwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    g_c: jax.Array,
    beta_c: jax.Array,
    s0: jax.Array,
    out_dtype: Any,
    state_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Launches `_kda_fwd_kernel` on `(B, H, NT, BT, ...)` chunked inputs."""
  b, h, nt, bt, k_dim = q_c.shape
  v_dim = v_c.shape[-1]
  bh = b * h
  mb = _select_mb(bh)

  q_r = q_c.reshape(bh, nt, bt, k_dim)
  k_r = k_c.reshape(bh, nt, bt, k_dim)
  v_r = v_c.reshape(bh, nt, bt, v_dim)
  g_r = g_c.reshape(bh, nt, bt, k_dim)
  beta_r = beta_c.astype(jnp.float32).reshape(bh, nt, bt, 1)
  s0_r = s0.reshape(bh, k_dim, v_dim)

  grid = (bh // mb, nt)
  idx = lambda i, n: (i, n, 0, 0)
  bspec_k = pl.BlockSpec((mb, 1, bt, k_dim), idx)
  bspec_v = pl.BlockSpec((mb, 1, bt, v_dim), idx)
  bspec_beta = pl.BlockSpec((mb, 1, bt, 1), idx)
  bspec_bt = pl.BlockSpec((mb, 1, bt, bt), idx)
  bspec_merged = pl.BlockSpec((mb, 1, bt, k_dim + v_dim), idx)
  bspec_s_in = pl.BlockSpec((mb, 1, k_dim, v_dim), idx)
  bspec_s = pl.BlockSpec((mb, k_dim, v_dim), lambda i, n: (i, 0, 0))

  interpret = jax.default_backend() == "cpu"
  o_r, s_final_r, a_mat_r, merged_out_r, s_in_r = pl.pallas_call(
      functools.partial(_kda_fwd_kernel, num_chunks=nt),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_beta, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_bt, bspec_merged, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((mb, k_dim, v_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((bh, nt, bt, v_dim), out_dtype),
          jax.ShapeDtypeStruct((bh, k_dim, v_dim), state_dtype),
          jax.ShapeDtypeStruct((bh, nt, bt, bt), q_c.dtype),
          jax.ShapeDtypeStruct((bh, nt, bt, k_dim + v_dim), jnp.float32),
          jax.ShapeDtypeStruct((bh, nt, k_dim, v_dim), jnp.float32),
      ],
      interpret=interpret,
  )(q_r, k_r, v_r, g_r, beta_r, s0_r)

  return (
      o_r.reshape(b, h, nt, bt, v_dim),
      s_final_r.reshape(b, h, k_dim, v_dim),
      a_mat_r.reshape(b, h, nt, bt, bt),
      merged_out_r.reshape(b, h, nt, bt, k_dim + v_dim),
      s_in_r.reshape(b, h, nt, k_dim, v_dim),
  )


def _kda_ring_phase1_fwd_kernel(
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    g_ref: Any,
    beta_ref: Any,
    s0_ref: Any,
    o_ref: Any,
    s_final_ref: Any,
    a_rank_ref: Any,
    a_mat_ref: Any,
    merged_out_ref: Any,
    s_in_ref: Any,
    s_scratch_ref: Any,
    a_scratch_ref: Any,
    *,
    num_chunks: int,
) -> None:
  """Ring Phase 1 Pallas forward kernel fusing `S` and `A_rank` in VMEM."""
  c = pl.program_id(1)
  mb = q_ref.shape[0]
  bt = q_ref.shape[2]
  k_dim = q_ref.shape[3]

  @pl.when(c == 0)
  def _() -> None:
    s_scratch_ref[...] = s0_ref[:].astype(jnp.float32)
    row = jax.lax.broadcasted_iota(jnp.int32, (k_dim, k_dim), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (k_dim, k_dim), 1)
    eye = (row == col).astype(jnp.float32)
    a_scratch_ref[...] = jnp.broadcast_to(eye[None, :, :], (mb, k_dim, k_dim))

  s_prev = s_scratch_ref[...]
  a_prev = a_scratch_ref[...]
  s_in_ref[:, 0] = s_prev.astype(s_in_ref.dtype)

  q_i = q_ref[:, 0].astype(jnp.float32)
  k_i = k_ref[:, 0].astype(jnp.float32)
  v_i = v_ref[:, 0].astype(jnp.float32)
  g_i = g_ref[:, 0].astype(jnp.float32)
  beta_i = beta_ref[:, 0].astype(jnp.float32)

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)

  bk = beta_i * k_i
  t_lower = -_decayed_gram(bk, k_i, g_i, g_i) * strictly_lower
  a_mat = _unit_lower_inverse(t_lower)

  exp_g = jnp.exp(g_i)
  k_g_b = exp_g * bk
  wv = beta_i * v_i
  merged_rhs = jnp.concatenate([k_g_b, wv], axis=-1)
  merged_out = jnp.matmul(
      a_mat,
      merged_rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  wwy_i = merged_out[..., :k_dim]
  u_i = merged_out[..., k_dim:]

  aqk = _decayed_gram(q_i, k_i, g_i, g_i) * causal_keep

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
      aqk,
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]
  exp_g_last = jnp.exp(g_last)[:, :, None]
  decayed_k = jnp.exp(g_last[:, None, :] - g_i) * k_i
  decayed_k_t = jnp.swapaxes(decayed_k, -1, -2)
  s_update = jnp.matmul(
      decayed_k_t,
      v_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s_new = s_prev * exp_g_last + s_update

  wwy_a = jnp.matmul(
      wwy_i,
      a_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  a_new = a_prev * exp_g_last - jnp.matmul(
      decayed_k_t,
      wwy_a,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  s_scratch_ref[...] = s_new
  a_scratch_ref[...] = a_new
  o_ref[:, 0] = o_i.astype(o_ref.dtype)
  a_mat_ref[:, 0] = a_mat.astype(a_mat_ref.dtype)
  merged_out_ref[:, 0] = merged_out.astype(merged_out_ref.dtype)

  @pl.when(c == num_chunks - 1)
  def _() -> None:
    s_final_ref[:] = s_new.astype(s_final_ref.dtype)
    a_rank_ref[:] = a_new.astype(a_rank_ref.dtype)


def _kda_ring_phase1_fwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    g_c: jax.Array,
    beta_c: jax.Array,
    s0: jax.Array,
    out_dtype: Any,
    state_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Launches `_kda_ring_phase1_fwd_kernel` on `(B, H, NT, BT, ...)` inputs."""
  b, h, nt, bt, k_dim = q_c.shape
  v_dim = v_c.shape[-1]
  bh = b * h
  mb = _select_mb(bh)

  q_r = q_c.reshape(bh, nt, bt, k_dim)
  k_r = k_c.reshape(bh, nt, bt, k_dim)
  v_r = v_c.reshape(bh, nt, bt, v_dim)
  g_r = g_c.reshape(bh, nt, bt, k_dim)
  beta_r = beta_c.astype(jnp.float32).reshape(bh, nt, bt, 1)
  s0_r = s0.reshape(bh, k_dim, v_dim)

  grid = (bh // mb, nt)
  idx = lambda i, n: (i, n, 0, 0)
  bspec_k = pl.BlockSpec((mb, 1, bt, k_dim), idx)
  bspec_v = pl.BlockSpec((mb, 1, bt, v_dim), idx)
  bspec_beta = pl.BlockSpec((mb, 1, bt, 1), idx)
  bspec_bt = pl.BlockSpec((mb, 1, bt, bt), idx)
  bspec_merged = pl.BlockSpec((mb, 1, bt, k_dim + v_dim), idx)
  bspec_s_in = pl.BlockSpec((mb, 1, k_dim, v_dim), idx)
  bspec_s = pl.BlockSpec((mb, k_dim, v_dim), lambda i, n: (i, 0, 0))
  bspec_a = pl.BlockSpec((mb, k_dim, k_dim), lambda i, n: (i, 0, 0))

  interpret = jax.default_backend() == "cpu"
  o_r, s_final_r, a_rank_r, a_mat_r, merged_out_r, s_in_r = pl.pallas_call(
      functools.partial(_kda_ring_phase1_fwd_kernel, num_chunks=nt),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_beta, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_a, bspec_bt, bspec_merged, bspec_s_in],
      scratch_shapes=[
          pltpu.VMEM((mb, k_dim, v_dim), jnp.float32),
          pltpu.VMEM((mb, k_dim, k_dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((bh, nt, bt, v_dim), out_dtype),
          jax.ShapeDtypeStruct((bh, k_dim, v_dim), state_dtype),
          jax.ShapeDtypeStruct((bh, k_dim, k_dim), jnp.float32),
          jax.ShapeDtypeStruct((bh, nt, bt, bt), q_c.dtype),
          jax.ShapeDtypeStruct((bh, nt, bt, k_dim + v_dim), jnp.float32),
          jax.ShapeDtypeStruct((bh, nt, k_dim, v_dim), jnp.float32),
      ],
      interpret=interpret,
  )(q_r, k_r, v_r, g_r, beta_r, s0_r)

  return (
      o_r.reshape(b, h, nt, bt, v_dim),
      s_final_r.reshape(b, h, k_dim, v_dim),
      a_rank_r.reshape(b, h, k_dim, k_dim),
      a_mat_r.reshape(b, h, nt, bt, bt),
      merged_out_r.reshape(b, h, nt, bt, k_dim + v_dim),
      s_in_r.reshape(b, h, nt, k_dim, v_dim),
  )


def _kda_state_fwd_kernel(
    q_ref: Any,
    k_ref: Any,
    g_ref: Any,
    merged_out_ref: Any,
    s0_ref: Any,
    o_ref: Any,
    s_final_ref: Any,
    s_in_ref: Any,
    scratch_ref: Any,
    *,
    num_chunks: int,
) -> None:
  """Ring Phase 3 state-propagation Pallas kernel reusing `merged_out_c`."""
  c = pl.program_id(1)
  bt = q_ref.shape[2]
  k_dim = q_ref.shape[3]

  @pl.when(c == 0)
  def _() -> None:
    scratch_ref[...] = s0_ref[:].astype(jnp.float32)

  s_prev = scratch_ref[...]
  s_in_ref[:, 0] = s_prev.astype(s_in_ref.dtype)

  q_i = q_ref[:, 0].astype(jnp.float32)
  k_i = k_ref[:, 0].astype(jnp.float32)
  g_i = g_ref[:, 0].astype(jnp.float32)
  merged_i = merged_out_ref[:, 0].astype(jnp.float32)

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)

  wwy_i = merged_i[..., :k_dim]
  u_i = merged_i[..., k_dim:]
  exp_g = jnp.exp(g_i)
  aqk_i = _decayed_gram(q_i, k_i, g_i, g_i) * causal_keep

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

  g_last = g_i[:, -1, :]
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
  def _() -> None:
    s_final_ref[:] = s_new.astype(s_final_ref.dtype)


def _kda_state_fwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    g_c: jax.Array,
    merged_out_c: jax.Array,
    s0: jax.Array,
    out_dtype: Any,
    state_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Launches `_kda_state_fwd_kernel` for ring ranks `r > 0`."""
  b, h, nt, bt, k_dim = q_c.shape
  v_dim = merged_out_c.shape[-1] - k_dim
  bh = b * h
  mb = _select_mb(bh)

  q_r = q_c.reshape(bh, nt, bt, k_dim)
  k_r = k_c.reshape(bh, nt, bt, k_dim)
  g_r = g_c.reshape(bh, nt, bt, k_dim)
  merged_r = merged_out_c.reshape(bh, nt, bt, k_dim + v_dim)
  s0_r = s0.reshape(bh, k_dim, v_dim)

  grid = (bh // mb, nt)
  idx = lambda i, n: (i, n, 0, 0)
  bspec_k = pl.BlockSpec((mb, 1, bt, k_dim), idx)
  bspec_v = pl.BlockSpec((mb, 1, bt, v_dim), idx)
  bspec_merged = pl.BlockSpec((mb, 1, bt, k_dim + v_dim), idx)
  bspec_s_in = pl.BlockSpec((mb, 1, k_dim, v_dim), idx)
  bspec_s = pl.BlockSpec((mb, k_dim, v_dim), lambda i, n: (i, 0, 0))

  interpret = jax.default_backend() == "cpu"
  o_r, s_final_r, s_in_r = pl.pallas_call(
      functools.partial(_kda_state_fwd_kernel, num_chunks=nt),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_k, bspec_merged, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((mb, k_dim, v_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((bh, nt, bt, v_dim), out_dtype),
          jax.ShapeDtypeStruct((bh, k_dim, v_dim), state_dtype),
          jax.ShapeDtypeStruct((bh, nt, k_dim, v_dim), jnp.float32),
      ],
      interpret=interpret,
  )(q_r, k_r, g_r, merged_r, s0_r)

  return (
      o_r.reshape(b, h, nt, bt, v_dim),
      s_final_r.reshape(b, h, k_dim, v_dim),
      s_in_r.reshape(b, h, nt, k_dim, v_dim),
  )


def _fwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Single-device JAX/Pallas forward step returning outputs and residuals."""
  b, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = _CHUNK_SIZE
  nt = t // bt
  scale = k_dim**-0.5

  q_c = _chunkify(q.astype(jnp.float32) * scale, b, nt, bt).astype(q.dtype)
  k_c = _chunkify(k, b, nt, bt)
  v_c = _chunkify(v, b, nt, bt)
  g_c = jnp.cumsum(_chunkify(g.astype(jnp.float32), b, nt, bt), axis=-2)
  beta_c = _chunkify(beta[..., None].astype(jnp.float32), b, nt, bt)
  s0 = initial_state.astype(jnp.float32)

  o_c, s_final, a_mat_c, merged_out_c, s_in_c = _kda_fwd_pallas(
      q_c, k_c, v_c, g_c, beta_c, s0, v_c.dtype, initial_state.dtype
  )

  o = _unchunk(o_c, v_dim, b, nt, bt, h)
  return o, s_final, a_mat_c, merged_out_c, s_in_c


def _kda_bwd_kernel(
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    g_ref: Any,
    beta_ref: Any,
    a_mat_ref: Any,
    merged_out_ref: Any,
    s_in_ref: Any,
    do_ref: Any,
    d_final_state_ref: Any,
    dq_ref: Any,
    dk_ref: Any,
    dv_ref: Any,
    dg_ref: Any,
    dbeta_ref: Any,
    ds0_ref: Any,
    scratch_ref: Any,
    *,
    num_chunks: int,
) -> None:
  """Single-pass `MB`-batched reverse-chunk Pallas analytical backward kernel."""
  rev_c = pl.program_id(1)
  bt = q_ref.shape[2]
  k_dim = q_ref.shape[3]

  @pl.when(rev_c == 0)
  def _() -> None:
    scratch_ref[...] = d_final_state_ref[:].astype(jnp.float32)

  ds_out = scratch_ref[...]
  s_prev = s_in_ref[:, 0].astype(jnp.float32)

  q_i = q_ref[:, 0].astype(jnp.float32)
  k_i = k_ref[:, 0].astype(jnp.float32)
  v_i = v_ref[:, 0].astype(jnp.float32)
  g_i = g_ref[:, 0].astype(jnp.float32)
  beta_i = beta_ref[:, 0].astype(jnp.float32)
  a_mat = a_mat_ref[:, 0].astype(jnp.float32)
  merged_out = merged_out_ref[:, 0].astype(jnp.float32)
  do_i = do_ref[:, 0].astype(jnp.float32)

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)
  upper_tri = (row_idx <= col_idx).astype(jnp.float32)

  bk = beta_i * k_i
  exp_g = jnp.exp(g_i)
  k_g_b = exp_g * bk

  wwy_i = merged_out[..., :k_dim]
  u_i = merged_out[..., k_dim:]
  aqk = _decayed_gram(q_i, k_i, g_i, g_i) * causal_keep
  v_new = u_i - jnp.matmul(
      wwy_i,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]
  exp_tail = jnp.exp(g_last[:, None, :] - g_i)
  decayed_k = exp_tail * k_i

  q_exp_g = q_i * exp_g
  s_prev_t = jnp.swapaxes(s_prev, -1, -2)
  d_q_exp_g = jnp.matmul(
      do_i,
      s_prev_t,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dq_off = d_q_exp_g * exp_g
  dg_off = d_q_exp_g * q_exp_g
  d_s_from_o = jnp.matmul(
      jnp.swapaxes(q_exp_g, -1, -2),
      do_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  d_aqk = (
      jnp.matmul(
          do_i,
          jnp.swapaxes(v_new, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * causal_keep
  )
  dv_from_o = jnp.matmul(
      jnp.swapaxes(aqk, -1, -2),
      do_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dv_new = dv_from_o + jnp.matmul(
      decayed_k,
      ds_out,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  ds_prev = (
      ds_out * jnp.exp(g_last)[:, :, None]
      + d_s_from_o
      - jnp.matmul(
          jnp.swapaxes(wwy_i, -1, -2),
          dv_new,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
  )
  scratch_ref[...] = ds_prev

  @pl.when(rev_c == num_chunks - 1)
  def _() -> None:
    ds0_ref[:] = ds_prev.astype(ds0_ref.dtype)

  d_decayed_k = jnp.matmul(
      v_new,
      jnp.swapaxes(ds_out, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  dk_tail = d_decayed_k * exp_tail
  d_tail_exp = d_decayed_k * decayed_k
  mask_strictly_prior = (
      jax.lax.broadcasted_iota(jnp.int32, (bt, 1), 0) < (bt - 1)
  ).astype(jnp.float32)
  dg_tail = -d_tail_exp * mask_strictly_prior
  dg_last_tail = jnp.sum(d_tail_exp * mask_strictly_prior, axis=-2)
  dg_last_state = jnp.sum(
      ds_out * s_prev * jnp.exp(g_last)[:, :, None], axis=-1
  )
  dg_last = dg_last_tail + dg_last_state

  du_i = dv_new
  dwwy_i = -jnp.matmul(
      dv_new,
      s_prev_t,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  d_merged_out = jnp.concatenate([dwwy_i, du_i], axis=-1)
  d_merged_rhs = jnp.matmul(
      jnp.swapaxes(a_mat, -1, -2),
      d_merged_out,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dk_g_b = d_merged_rhs[..., :k_dim]
  dwv = d_merged_rhs[..., k_dim:]

  d_tlower = (
      jnp.matmul(
          d_merged_rhs,
          jnp.swapaxes(merged_out, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * strictly_lower
  )

  dv = dwv * beta_i
  dbeta_wv = jnp.sum(dwv * v_i, axis=-1, keepdims=True)

  db_kgb = dk_g_b * exp_g * k_i
  dk_kgb = dk_g_b * exp_g * beta_i
  dg_kgb = dk_g_b * k_g_b

  dq_intra, dk_intra, dg_aqk_a, dg_aqk_b = _decayed_gram_bwd(
      d_aqk, q_i, k_i, g_i, g_i
  )
  dg_decay_aqk = dg_aqk_a + dg_aqk_b

  dbk, dk_tlower, dg_tl_a, dg_tl_b = _decayed_gram_bwd(
      -d_tlower, bk, k_i, g_i, g_i
  )
  db_tlower = dbk * k_i
  dk_tlower2 = dbk * beta_i
  dg_decay_tlower = dg_tl_a + dg_tl_b

  mask_last = (
      jax.lax.broadcasted_iota(jnp.int32, (bt, 1), 0) == (bt - 1)
  ).astype(jnp.float32)

  dq = dq_off + dq_intra
  dk = dk_kgb + dk_tail + dk_intra + dk_tlower + dk_tlower2
  dbeta = jnp.sum(db_kgb + db_tlower, axis=-1, keepdims=True) + dbeta_wv
  dg = (
      dg_off
      + dg_kgb
      + dg_tail
      + dg_decay_aqk
      + dg_decay_tlower
      + mask_last * dg_last[:, None, :]
  )

  dg_raw = jnp.matmul(
      upper_tri,
      dg,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  dq_ref[:, 0] = dq.astype(dq_ref.dtype)
  dk_ref[:, 0] = dk.astype(dk_ref.dtype)
  dv_ref[:, 0] = dv.astype(dv_ref.dtype)
  dg_ref[:, 0] = dg_raw.astype(dg_ref.dtype)
  dbeta_ref[:, 0] = dbeta.astype(dbeta_ref.dtype)


def _kda_bwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    g_c: jax.Array,
    beta_c: jax.Array,
    a_mat_c: jax.Array,
    merged_out_c: jax.Array,
    s_in_c: jax.Array,
    do_c: jax.Array,
    d_final_state: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Launches `_kda_bwd_kernel` in reverse chunk order (`nt - 1 - n`)."""
  b, h, nt, bt, k_dim = q_c.shape
  v_dim = v_c.shape[-1]
  bh = b * h
  mb = _select_mb(bh)

  q_r = q_c.reshape(bh, nt, bt, k_dim)
  k_r = k_c.reshape(bh, nt, bt, k_dim)
  v_r = v_c.reshape(bh, nt, bt, v_dim)
  g_r = g_c.reshape(bh, nt, bt, k_dim)
  beta_r = beta_c.astype(jnp.float32).reshape(bh, nt, bt, 1)
  a_mat_r = a_mat_c.reshape(bh, nt, bt, bt)
  merged_r = merged_out_c.reshape(bh, nt, bt, k_dim + v_dim)
  s_in_r = s_in_c.reshape(bh, nt, k_dim, v_dim)
  do_r = do_c.reshape(bh, nt, bt, v_dim)
  ds_final_r = d_final_state.astype(jnp.float32).reshape(bh, k_dim, v_dim)

  grid = (bh // mb, nt)
  rev_idx = lambda i, n: (i, nt - 1 - n, 0, 0)
  bspec_k = pl.BlockSpec((mb, 1, bt, k_dim), rev_idx)
  bspec_v = pl.BlockSpec((mb, 1, bt, v_dim), rev_idx)
  bspec_beta = pl.BlockSpec((mb, 1, bt, 1), rev_idx)
  bspec_bt = pl.BlockSpec((mb, 1, bt, bt), rev_idx)
  bspec_merged = pl.BlockSpec((mb, 1, bt, k_dim + v_dim), rev_idx)
  bspec_s_in = pl.BlockSpec((mb, 1, k_dim, v_dim), rev_idx)
  bspec_s = pl.BlockSpec((mb, k_dim, v_dim), lambda i, n: (i, 0, 0))

  interpret = jax.default_backend() == "cpu"
  dq_r, dk_r, dv_r, dg_raw_r, dbeta_r, ds0_r = pl.pallas_call(
      functools.partial(_kda_bwd_kernel, num_chunks=nt),
      grid=grid,
      in_specs=[
          bspec_k,
          bspec_k,
          bspec_v,
          bspec_k,
          bspec_beta,
          bspec_bt,
          bspec_merged,
          bspec_s_in,
          bspec_v,
          bspec_s,
      ],
      out_specs=[
          bspec_k,
          bspec_k,
          bspec_v,
          bspec_k,
          bspec_beta,
          bspec_s,
      ],
      scratch_shapes=[pltpu.VMEM((mb, k_dim, v_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((bh, nt, bt, k_dim), jnp.float32),
          jax.ShapeDtypeStruct((bh, nt, bt, k_dim), k_c.dtype),
          jax.ShapeDtypeStruct((bh, nt, bt, v_dim), v_c.dtype),
          jax.ShapeDtypeStruct((bh, nt, bt, k_dim), g_c.dtype),
          jax.ShapeDtypeStruct((bh, nt, bt, 1), jnp.float32),
          jax.ShapeDtypeStruct((bh, k_dim, v_dim), d_final_state.dtype),
      ],
      interpret=interpret,
  )(
      q_r,
      k_r,
      v_r,
      g_r,
      beta_r,
      a_mat_r,
      merged_r,
      s_in_r,
      do_r,
      ds_final_r,
  )

  return (
      dq_r.reshape(b, h, nt, bt, k_dim),
      dk_r.reshape(b, h, nt, bt, k_dim),
      dv_r.reshape(b, h, nt, bt, v_dim),
      dg_raw_r.reshape(b, h, nt, bt, k_dim),
      dbeta_r.reshape(b, h, nt, bt, 1),
      ds0_r.reshape(b, h, k_dim, v_dim).astype(d_final_state.dtype),
  )


def _kda_state_bwd_ds0_kernel(
    q_ref: Any,
    k_ref: Any,
    g_ref: Any,
    merged_out_ref: Any,
    do_ref: Any,
    d_final_state_ref: Any,
    ds0_ref: Any,
    scratch_ref: Any,
    *,
    num_chunks: int,
) -> None:
  """Ring Phase 1 reverse scan Pallas kernel computing `ds0`."""
  rev_c = pl.program_id(1)
  bt = q_ref.shape[2]
  k_dim = q_ref.shape[3]

  @pl.when(rev_c == 0)
  def _() -> None:
    scratch_ref[...] = d_final_state_ref[:].astype(jnp.float32)

  ds_out = scratch_ref[...]

  q_i = q_ref[:, 0].astype(jnp.float32)
  k_i = k_ref[:, 0].astype(jnp.float32)
  g_i = g_ref[:, 0].astype(jnp.float32)
  merged_i = merged_out_ref[:, 0].astype(jnp.float32)
  do_i = do_ref[:, 0].astype(jnp.float32)

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)

  wwy_i = merged_i[..., :k_dim]
  exp_g = jnp.exp(g_i)
  aqk_i = _decayed_gram(q_i, k_i, g_i, g_i) * causal_keep
  g_last = g_i[:, -1, :]
  exp_tail = jnp.exp(g_last[:, None, :] - g_i)
  decayed_k = exp_tail * k_i

  q_exp_g = q_i * exp_g
  d_s_from_o = jnp.matmul(
      jnp.swapaxes(q_exp_g, -1, -2),
      do_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  dv_from_o = jnp.matmul(
      jnp.swapaxes(aqk_i, -1, -2),
      do_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dv_new = dv_from_o + jnp.matmul(
      decayed_k,
      ds_out,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  ds_prev = (
      ds_out * jnp.exp(g_last)[:, :, None]
      + d_s_from_o
      - jnp.matmul(
          jnp.swapaxes(wwy_i, -1, -2),
          dv_new,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
  )
  scratch_ref[...] = ds_prev

  @pl.when(rev_c == num_chunks - 1)
  def _() -> None:
    ds0_ref[:] = ds_prev.astype(ds0_ref.dtype)


def _kda_state_bwd_ds0_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    g_c: jax.Array,
    merged_out_c: jax.Array,
    do_c: jax.Array,
    d_final_state: jax.Array,
) -> jax.Array:
  """Launches `_kda_state_bwd_ds0_kernel` in reverse chunk order."""
  b, h, nt, bt, k_dim = q_c.shape
  v_dim = do_c.shape[-1]
  bh = b * h
  mb = _select_mb(bh)

  q_r = q_c.reshape(bh, nt, bt, k_dim)
  k_r = k_c.reshape(bh, nt, bt, k_dim)
  g_r = g_c.reshape(bh, nt, bt, k_dim)
  merged_r = merged_out_c.reshape(bh, nt, bt, k_dim + v_dim)
  do_r = do_c.reshape(bh, nt, bt, v_dim)
  ds_final_r = d_final_state.astype(jnp.float32).reshape(bh, k_dim, v_dim)

  grid = (bh // mb, nt)
  rev_idx = lambda i, n: (i, nt - 1 - n, 0, 0)
  bspec_k = pl.BlockSpec((mb, 1, bt, k_dim), rev_idx)
  bspec_v = pl.BlockSpec((mb, 1, bt, v_dim), rev_idx)
  bspec_merged = pl.BlockSpec((mb, 1, bt, k_dim + v_dim), rev_idx)
  bspec_s = pl.BlockSpec((mb, k_dim, v_dim), lambda i, n: (i, 0, 0))

  interpret = jax.default_backend() == "cpu"
  ds0_r = pl.pallas_call(
      functools.partial(_kda_state_bwd_ds0_kernel, num_chunks=nt),
      grid=grid,
      in_specs=[
          bspec_k,
          bspec_k,
          bspec_k,
          bspec_merged,
          bspec_v,
          bspec_s,
      ],
      out_specs=bspec_s,
      scratch_shapes=[pltpu.VMEM((mb, k_dim, v_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=jax.ShapeDtypeStruct((bh, k_dim, v_dim), d_final_state.dtype),
      interpret=interpret,
  )(q_r, k_r, g_r, merged_r, do_r, ds_final_r)

  return ds0_r.reshape(b, h, k_dim, v_dim).astype(d_final_state.dtype)


def _bwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    a_mat_c: jax.Array,
    merged_out_c: jax.Array,
    s_in_c: jax.Array,
    do: jax.Array,
    d_final_state: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Single-device JAX/Pallas analytical backward step."""
  b, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = _CHUNK_SIZE
  nt = t // bt
  scale = k_dim**-0.5

  q_c = _chunkify(q.astype(jnp.float32) * scale, b, nt, bt).astype(q.dtype)
  k_c = _chunkify(k, b, nt, bt)
  v_c = _chunkify(v, b, nt, bt)
  g_c = jnp.cumsum(_chunkify(g.astype(jnp.float32), b, nt, bt), axis=-2)
  beta_c = _chunkify(beta[..., None].astype(jnp.float32), b, nt, bt)
  do_c = _chunkify(do, b, nt, bt)

  dq_c, dk_c, dv_c, dg_raw_c, dbeta_c, ds0 = _kda_bwd_pallas(
      q_c,
      k_c,
      v_c,
      g_c,
      beta_c,
      a_mat_c,
      merged_out_c,
      s_in_c,
      do_c,
      d_final_state,
  )

  dq = (_unchunk(dq_c, k_dim, b, nt, bt, h) * scale).astype(q.dtype)
  dk = _unchunk(dk_c, k_dim, b, nt, bt, h).astype(k.dtype)
  dv = _unchunk(dv_c, v_dim, b, nt, bt, h).astype(v.dtype)
  dg = _unchunk(dg_raw_c, k_dim, b, nt, bt, h).astype(g.dtype)
  dbeta = _unchunk(dbeta_c, 1, b, nt, bt, h)[..., 0].astype(beta.dtype)

  return dq, dk, dv, dg, dbeta, ds0


def _bwd_ds0_jax(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    merged_out_c: jax.Array,
    do: jax.Array,
    d_final_state: jax.Array,
) -> jax.Array:
  """Ring Phase 1 reverse scan entry point returning `ds0`."""
  b, t, _h, k_dim = q.shape
  bt = _CHUNK_SIZE
  nt = t // bt
  scale = k_dim**-0.5

  q_c = _chunkify(q.astype(jnp.float32) * scale, b, nt, bt).astype(q.dtype)
  k_c = _chunkify(k, b, nt, bt)
  g_c = jnp.cumsum(_chunkify(g.astype(jnp.float32), b, nt, bt), axis=-2)
  do_c = _chunkify(do, b, nt, bt)

  return _kda_state_bwd_ds0_pallas(
      q_c, k_c, g_c, merged_out_c, do_c, d_final_state
  )


def _ring_phase1_fwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Ring Phase 1 forward entry point returning `(o, s_final, a_rank, ...)`."""
  b, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = _CHUNK_SIZE
  nt = t // bt
  scale = k_dim**-0.5

  q_c = _chunkify(q.astype(jnp.float32) * scale, b, nt, bt).astype(q.dtype)
  k_c = _chunkify(k, b, nt, bt)
  v_c = _chunkify(v, b, nt, bt)
  g_c = jnp.cumsum(_chunkify(g.astype(jnp.float32), b, nt, bt), axis=-2)
  beta_c = _chunkify(beta[..., None].astype(jnp.float32), b, nt, bt)
  s0 = initial_state.astype(jnp.float32)

  o_c, s_final, a_rank, a_mat_c, merged_out_c, s_in_c = (
      _kda_ring_phase1_fwd_pallas(
          q_c, k_c, v_c, g_c, beta_c, s0, v_c.dtype, initial_state.dtype
      )
  )
  o = _unchunk(o_c, v_dim, b, nt, bt, h)
  return o, s_final, a_rank, a_mat_c, merged_out_c, s_in_c


def _state_fwd_jax(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    merged_out_c: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Ring Phase 3 forward entry point for ranks `r > 0`."""
  b, t, h, k_dim = q.shape
  v_dim = merged_out_c.shape[-1] - k_dim
  bt = _CHUNK_SIZE
  nt = t // bt
  scale = k_dim**-0.5

  q_c = _chunkify(q.astype(jnp.float32) * scale, b, nt, bt).astype(q.dtype)
  k_c = _chunkify(k, b, nt, bt)
  g_c = jnp.cumsum(_chunkify(g.astype(jnp.float32), b, nt, bt), axis=-2)
  s0 = initial_state.astype(jnp.float32)

  o_c, s_final, s_in_c = _kda_state_fwd_pallas(
      q_c, k_c, g_c, merged_out_c, s0, q.dtype, initial_state.dtype
  )
  o = _unchunk(o_c, v_dim, b, nt, bt, h)
  return o, s_final, s_in_c


def kda_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> Any:
  """Executes KDA forward via `_kda_fwd_pallas` and optionally returns residuals."""
  b, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  if chunk_size != _CHUNK_SIZE:
    raise NotImplementedError(
        f"Only chunk_size={_CHUNK_SIZE} is supported; got {chunk_size}."
    )
  if t % chunk_size != 0:
    raise ValueError(
        f"Sequence length {t} must be divisible by chunk_size {chunk_size}."
    )
  nt = t // chunk_size
  if scale is None:
    scale = k_dim**-0.5

  if beta.ndim == 4 and beta.shape[-1] == 1:
    beta = beta[..., 0]

  if initial_state is None:
    s0 = jnp.zeros((b, h, k_dim, v_dim), dtype=jnp.float32)
    state_dtype = jnp.float32
  else:
    s0 = initial_state.astype(jnp.float32)
    state_dtype = initial_state.dtype

  q_c = _chunkify(q.astype(jnp.float32) * scale, b, nt, chunk_size).astype(
      q.dtype
  )
  k_c = _chunkify(k, b, nt, chunk_size)
  v_c = _chunkify(v, b, nt, chunk_size)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), b, nt, chunk_size), axis=-2
  )
  beta_c = _chunkify(beta[..., None].astype(jnp.float32), b, nt, chunk_size)

  o_c, s_final, a_mat_c, merged_out_c, s_in_c = _kda_fwd_pallas(
      q_c, k_c, v_c, g_c, beta_c, s0, v_c.dtype, state_dtype
  )
  o = _unchunk(o_c, v_dim, b, nt, chunk_size, h)
  primary = (o, s_final) if output_final_state else o
  if return_residuals:
    return primary, (a_mat_c, merged_out_c, s_in_c)
  return primary


def kda_pallas_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    do: jax.Array,
    grad_final_state: jax.Array | None = None,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    a_mat_c: jax.Array | None = None,
    merged_out_c: jax.Array | None = None,
    s_in_c: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
]:
  """Executes KDA analytical backward via `_kda_bwd_pallas`."""
  b, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  if chunk_size != _CHUNK_SIZE:
    raise NotImplementedError(
        f"Only chunk_size={_CHUNK_SIZE} is supported; got {chunk_size}."
    )
  nt = t // chunk_size
  if scale is None:
    scale = k_dim**-0.5

  beta_was_4d = beta.ndim == 4 and beta.shape[-1] == 1
  beta_3d = beta[..., 0] if beta_was_4d else beta

  if a_mat_c is None or merged_out_c is None or s_in_c is None:
    _, (a_mat_c, merged_out_c, s_in_c) = kda_pallas_fwd(
        q,
        k,
        v,
        g,
        beta_3d,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        return_residuals=True,
    )

  if grad_final_state is None:
    d_final_state = jnp.zeros((b, h, k_dim, v_dim), dtype=jnp.float32)
  else:
    d_final_state = grad_final_state

  q_c = _chunkify(q.astype(jnp.float32) * scale, b, nt, chunk_size).astype(
      q.dtype
  )
  k_c = _chunkify(k, b, nt, chunk_size)
  v_c = _chunkify(v, b, nt, chunk_size)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), b, nt, chunk_size), axis=-2
  )
  beta_c = _chunkify(beta_3d[..., None].astype(jnp.float32), b, nt, chunk_size)
  do_c = _chunkify(do, b, nt, chunk_size)

  dq_c, dk_c, dv_c, dg_raw_c, dbeta_c, ds0 = _kda_bwd_pallas(
      q_c,
      k_c,
      v_c,
      g_c,
      beta_c,
      a_mat_c,
      merged_out_c,
      s_in_c,
      do_c,
      d_final_state,
  )

  dq = (_unchunk(dq_c, k_dim, b, nt, chunk_size, h) * scale).astype(q.dtype)
  dk = _unchunk(dk_c, k_dim, b, nt, chunk_size, h).astype(k.dtype)
  dv = _unchunk(dv_c, v_dim, b, nt, chunk_size, h).astype(v.dtype)
  dg = _unchunk(dg_raw_c, k_dim, b, nt, chunk_size, h).astype(g.dtype)
  dbeta_un = _unchunk(dbeta_c, 1, b, nt, chunk_size, h)
  dbeta = (dbeta_un if beta_was_4d else dbeta_un[..., 0]).astype(beta.dtype)
  ds0_out = (
      ds0.astype(initial_state.dtype) if initial_state is not None else None
  )

  return dq, dk, dv, dg, dbeta, ds0_out


def kda_scan_pure_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> Any:
  """Executes the KDA forward pass via `kda_pallas_fwd`."""
  return kda_pallas_fwd(
      q,
      k,
      v,
      g,
      beta,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      return_residuals=False,
  )


def kda_bwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    do: jax.Array,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Executes KDA forward + analytical Pallas backward pass."""
  dq, dk, dv, dg, dbeta, _ = kda_pallas_bwd(
      q, k, v, g, beta, do, chunk_size=chunk_size, scale=1.0
  )
  return dq, dk, dv, dg, dbeta


def _execute_cpu_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = _CHUNK_SIZE,
) -> torch.Tensor:
  """Reference token-by-token KDA delta-rule recurrence on CPU."""
  batch_size, seq_len, num_heads, key_dim = q.shape
  val_dim = v.shape[-1]
  if chunk_size != _CHUNK_SIZE:
    raise NotImplementedError(
        f"Only chunk_size={_CHUNK_SIZE} is supported; got {chunk_size}."
    )
  if seq_len % chunk_size != 0:
    raise ValueError(
        f"Sequence length ({seq_len}) must be divisible by chunk_size"
        f" ({chunk_size})."
    )

  q_f = q.float()
  k_f = k.float()
  v_f = v.float()
  exp_g = torch.exp(g.float()).unsqueeze(-1)
  beta_f = beta.float()
  if beta_f.ndim == 3:
    beta_f = beta_f.unsqueeze(-1)

  state = torch.zeros(
      (batch_size, num_heads, key_dim, val_dim),
      dtype=torch.float32,
      device=q.device,
  )
  outputs = []
  for t in range(seq_len):
    k_t = k_f[:, t]
    v_t = v_f[:, t]
    q_t = q_f[:, t]
    beta_t = beta_f[:, t]

    state = state * exp_g[:, t]
    v_pred = torch.einsum("bhk,bhkv->bhv", k_t, state)
    v_delta = beta_t * (v_t - v_pred)
    state = state + torch.einsum("bhk,bhv->bhkv", k_t, v_delta)
    o_t = torch.einsum("bhk,bhkv->bhv", q_t, state)
    outputs.append(o_t)

  return torch.stack(outputs, dim=1).to(v.dtype)


if _TORCH_TPU_PALLAS_AVAILABLE and _pallas is not None:
  try:
    _pallas_kda_fwd_op = _pallas.jax_op("tokamax_tpu::_pallas_kda_fwd", _fwd_jax)
    _pallas_kda_bwd_op = _pallas.jax_op("tokamax_tpu::_pallas_kda_bwd", _bwd_jax)
    _pallas_kda_bwd_ds0_op = _pallas.jax_op(
        "tokamax_tpu::_pallas_kda_bwd_ds0", _bwd_ds0_jax
    )
    _pallas_kda_ring_phase1_fwd_op = _pallas.jax_op(
        "tokamax_tpu::_pallas_kda_ring_phase1_fwd", _ring_phase1_fwd_jax
    )
    _pallas_kda_state_fwd_op = _pallas.jax_op(
        "tokamax_tpu::_pallas_kda_state_fwd", _state_fwd_jax
    )

    def _setup_context(ctx: Any, inputs: Any, output: Any) -> None:
      q, k, v, g, beta, _initial_state = inputs
      _, _, a_mat_c, merged_out_c, s_in_c = output
      ctx.save_for_backward(q, k, v, g, beta, a_mat_c, merged_out_c, s_in_c)

    def _backward_impl(
        ctx: Any,
        grad_out: torch.Tensor,
        grad_final_state: torch.Tensor,
        _grad_a_mat_c: torch.Tensor,
        _grad_merged_out_c: torch.Tensor,
        _grad_s_in_c: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
      q, k, v, g, beta, a_mat_c, merged_out_c, s_in_c = ctx.saved_tensors
      return _pallas_kda_bwd_op(
          q,
          k,
          v,
          g,
          beta,
          a_mat_c,
          merged_out_c,
          s_in_c,
          grad_out,
          grad_final_state,
      )

    _pallas_kda_fwd_op.register_autograd(
        _backward_impl, setup_context=_setup_context
    )

    def _pallas_kda_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      o, s_final, _a_mat_c, _merged_out_c, _s_in_c = _pallas_kda_fwd_op(
          q, k, v, g, beta, initial_state
      )
      return o, s_final

  except (AttributeError, RuntimeError, TypeError) as e:
    logging.warning("Failed to register KDA Pallas TPU ops: %s", e)
    _TORCH_TPU_PALLAS_AVAILABLE = False


def kda_pallas_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = _CHUNK_SIZE,
) -> torch.Tensor:
  """Executes Kimi Delta Attention (KDA) via TPU Pallas or CPU fallback."""
  batch_size, seq_len, num_heads, key_dim = q.shape
  val_dim = v.shape[-1]
  if chunk_size != _CHUNK_SIZE:
    raise NotImplementedError(
        f"Only chunk_size={_CHUNK_SIZE} is supported; got {chunk_size}."
    )
  if seq_len % chunk_size != 0:
    raise ValueError(
        f"Sequence length ({seq_len}) must be divisible by chunk_size"
        f" ({chunk_size})."
    )
  if q.device.type == "tpu" and _TORCH_TPU_PALLAS_AVAILABLE:
    beta_3d = beta[..., 0] if beta.ndim == 4 and beta.shape[-1] == 1 else beta
    q_unscaled = q * (key_dim**0.5)
    initial_state = torch.zeros(
        (batch_size, num_heads, key_dim, val_dim),
        dtype=torch.float32,
        device=q.device,
    )
    out, _ = _pallas_kda_fwd(q_unscaled, k, v, g, beta_3d, initial_state)
    return out
  return _execute_cpu_fallback(q, k, v, g, beta, chunk_size=chunk_size)
