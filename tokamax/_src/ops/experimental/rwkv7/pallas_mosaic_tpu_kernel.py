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

"""Pallas TPU compute kernel for RWKV-7 (DPLR Delta Rule)."""

import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Dimensions:
#   B: batch size
#   T: sequence length (NT * BT)
#   H: number of attention heads
#   BH: flattened batch-head count (B * H)
#   MB: stream micro-batch size dividing BH
#   NB: number of micro-batches (BH // MB)
#   NT: number of chunks (T // BT)
#   BT: chunk size (chunk_size)
#   K: key head dimension (key_dim)
#   V: value head dimension (val_dim)

_CHUNK_SIZE = 64


def _pick_mb(bh: int) -> int:
  for candidate in (4, 2, 1):
    if bh % candidate == 0:
      return candidate
  return 1


def _chunkify(
    x: jax.Array, batch_size: int, num_chunks: int, chunk_size: int
) -> jax.Array:
  rest = x.shape[3:]
  x = x.reshape((batch_size, num_chunks, chunk_size, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _mm(x: jax.Array, y: jax.Array) -> jax.Array:
  return jnp.matmul(
      x,
      y,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _rwkv7_fwd_kernel(
    # Inputs.
    q_vmem_ref,  # [1, 1, MB, BT, K] dtype
    k_vmem_ref,  # [1, 1, MB, BT, K] dtype
    v_vmem_ref,  # [1, 1, MB, BT, V] dtype
    alpha_vmem_ref,  # [1, 1, MB, BT, K] dtype
    beta_vmem_ref,  # [1, 1, MB, BT, K] dtype
    gk_vmem_ref,  # [1, 1, MB, BT, K] dtype
    gk_cs_vmem_ref,  # [1, 1, MB, BT, K] float32
    s0_vmem_ref,  # [1, MB, K, V] float32
    # Outputs.
    o_vmem_ref,  # [1, 1, MB, BT, V] dtype
    a_ab_inv_vmem_ref,  # [1, 1, MB, BT, BT] dtype
    merged_out_vmem_ref,  # [1, 1, MB, BT, K + V] float32
    s_in_vmem_ref,  # [1, 1, MB, K, V] float32
    s_final_vmem_ref,  # [1, MB, K, V] float32
    # Scratch.
    s_scratch_vmem_ref,  # [MB, K, V] float32
    *,
    num_chunks: int,
) -> None:
  """Single-pass fused RWKV-7 forward Pallas kernel."""
  c = pl.program_id(1)
  bt = q_vmem_ref.shape[3]
  k_dim = q_vmem_ref.shape[4]

  @pl.when(c == 0)
  def _init_state() -> None:
    s_scratch_vmem_ref[...] = s0_vmem_ref[0].astype(jnp.float32)

  q_i = q_vmem_ref[0, 0].astype(jnp.float32)
  k_i = k_vmem_ref[0, 0].astype(jnp.float32)
  v_i = v_vmem_ref[0, 0].astype(jnp.float32)
  alpha_i = alpha_vmem_ref[0, 0].astype(jnp.float32)
  beta_i = beta_vmem_ref[0, 0].astype(jnp.float32)
  gk_i = gk_vmem_ref[0, 0].astype(jnp.float32)
  gk_cs = gk_cs_vmem_ref[0, 0].astype(jnp.float32)

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (1, bt, bt), 1)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (1, bt, bt), 2)
  incl_causal = (col_idx <= row_idx).astype(jnp.float32)
  strict_causal = (col_idx < row_idx).astype(jnp.float32)
  eye = (row_idx == col_idx).astype(jnp.float32)

  gk_excl = gk_cs - gk_i
  g0 = gk_cs[:, bt // 2 : bt // 2 + 1, :]
  exp_pos_incl = jnp.exp(jnp.clip(gk_cs - g0, -80.0, 80.0))
  exp_pos_excl = jnp.exp(jnp.clip(gk_excl - g0, -80.0, 80.0))
  exp_neg = jnp.exp(jnp.clip(g0 - gk_cs, -80.0, 80.0))

  q_hat = q_i * exp_pos_incl
  alpha_hat = alpha_i * exp_pos_excl
  k_hat = k_i * exp_neg
  beta_hat = beta_i * exp_neg

  a_qk = _mm(q_hat, k_hat.swapaxes(-1, -2)) * incl_causal
  a_qb = _mm(q_hat, beta_hat.swapaxes(-1, -2)) * incl_causal
  a_ab = _mm(alpha_hat, beta_hat.swapaxes(-1, -2)) * strict_causal
  a_ak = _mm(alpha_hat, k_hat.swapaxes(-1, -2)) * strict_causal

  p_mat = eye + a_ab
  curr = a_ab
  num_doubling_steps = max(0, math.ceil(math.log2(bt)) - 1)
  for _ in range(num_doubling_steps):
    curr = _mm(curr, curr)
    p_mat = _mm(p_mat, eye + curr)
  a_ab_inv = p_mat
  a_ab_inv_vmem_ref[0, 0] = a_ab_inv.astype(a_ab_inv_vmem_ref.dtype)

  ak_v = _mm(a_ak, v_i)
  decayed_alpha = jnp.exp(jnp.minimum(gk_excl, 0.0)) * alpha_i
  merged_rhs = jnp.concatenate([decayed_alpha, ak_v], axis=-1)
  merged_uw = _mm(a_ab_inv, merged_rhs)
  merged_out_vmem_ref[0, 0] = merged_uw.astype(merged_out_vmem_ref.dtype)
  w_i = merged_uw[:, :, :k_dim]
  u_i = merged_uw[:, :, k_dim:]

  s_prev = s_scratch_vmem_ref[...]
  s_in_vmem_ref[0, 0] = s_prev.astype(s_in_vmem_ref.dtype)

  v2_i = u_i + _mm(w_i, s_prev)
  q_exp = q_i * jnp.exp(jnp.minimum(gk_cs, 0.0))
  o_i = _mm(q_exp, s_prev) + _mm(a_qk, v_i) + _mm(a_qb, v2_i)
  o_vmem_ref[0, 0] = o_i.astype(o_vmem_ref.dtype)

  g_last = gk_cs[:, -1:, :]
  exp_g_last = jnp.exp(jnp.minimum(g_last, 0.0)).swapaxes(-1, -2)
  decay_last = jnp.exp(jnp.minimum(g_last - gk_cs, 0.0))
  dk_i = k_i * decay_last
  db_i = beta_i * decay_last
  s_new = (
      s_prev * exp_g_last
      + _mm(dk_i.swapaxes(-1, -2), v_i)
      + _mm(db_i.swapaxes(-1, -2), v2_i)
  )
  s_scratch_vmem_ref[...] = s_new

  @pl.when(c == num_chunks - 1)
  def _store_final() -> None:
    s_final_vmem_ref[0] = s_new.astype(s_final_vmem_ref.dtype)


def _rwkv7_fwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    alpha_c: jax.Array,
    beta_c: jax.Array,
    gk_c: jax.Array,
    gk_cs_c: jax.Array,
    s0: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Executes single-pass fused RWKV-7 forward kernel."""
  batch_size, num_heads, num_chunks, chunk_size, key_dim = q_c.shape
  val_dim = v_c.shape[-1]
  bh = batch_size * num_heads
  mb = _pick_mb(bh)
  nb = bh // mb

  def _prep_in(x: jax.Array, d: int) -> jax.Array:
    return jnp.swapaxes(x.reshape(nb, mb, num_chunks, chunk_size, d), 1, 2)

  q_in = _prep_in(q_c, key_dim)
  k_in = _prep_in(k_c, key_dim)
  v_in = _prep_in(v_c, val_dim)
  alpha_in = _prep_in(alpha_c, key_dim)
  beta_in = _prep_in(beta_c, key_dim)
  gk_in = _prep_in(gk_c, key_dim)
  gk_cs_in = _prep_in(gk_cs_c, key_dim)
  s0_in = s0.reshape(nb, mb, key_dim, val_dim).astype(jnp.float32)

  grid = (nb, num_chunks)
  bspec_k = pl.BlockSpec(
      (1, 1, mb, chunk_size, key_dim), lambda i_b, c: (i_b, c, 0, 0, 0)
  )
  bspec_v = pl.BlockSpec(
      (1, 1, mb, chunk_size, val_dim), lambda i_b, c: (i_b, c, 0, 0, 0)
  )
  bspec_bt = pl.BlockSpec(
      (1, 1, mb, chunk_size, chunk_size), lambda i_b, c: (i_b, c, 0, 0, 0)
  )
  bspec_kv = pl.BlockSpec(
      (1, 1, mb, chunk_size, key_dim + val_dim),
      lambda i_b, c: (i_b, c, 0, 0, 0),
  )
  bspec_s_in = pl.BlockSpec(
      (1, 1, mb, key_dim, val_dim), lambda i_b, c: (i_b, c, 0, 0, 0)
  )
  bspec_s = pl.BlockSpec(
      (1, mb, key_dim, val_dim), lambda i_b, c: (i_b, 0, 0, 0)
  )

  interpret = jax.default_backend() == "cpu"
  o_out, a_ab_inv_out, merged_out, s_in_out, s_final_out = pl.pallas_call(
      functools.partial(_rwkv7_fwd_kernel, num_chunks=num_chunks),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              bspec_k,
              bspec_k,
              bspec_v,
              bspec_k,
              bspec_k,
              bspec_k,
              bspec_k,
              bspec_s,
          ],
          out_specs=[bspec_v, bspec_bt, bspec_kv, bspec_s_in, bspec_s],
          scratch_shapes=[pltpu.VMEM((mb, key_dim, val_dim), jnp.float32)],
      ),
      out_shape=[
          jax.ShapeDtypeStruct(
              (nb, num_chunks, mb, chunk_size, val_dim), v_c.dtype
          ),
          jax.ShapeDtypeStruct(
              (nb, num_chunks, mb, chunk_size, chunk_size), q_c.dtype
          ),
          jax.ShapeDtypeStruct(
              (nb, num_chunks, mb, chunk_size, key_dim + val_dim), jnp.float32
          ),
          jax.ShapeDtypeStruct(
              (nb, num_chunks, mb, key_dim, val_dim), jnp.float32
          ),
          jax.ShapeDtypeStruct((nb, mb, key_dim, val_dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary")
      ),
      interpret=interpret,
  )(q_in, k_in, v_in, alpha_in, beta_in, gk_in, gk_cs_in, s0_in)

  o_c = jnp.swapaxes(o_out, 1, 2).reshape(
      batch_size, num_heads, num_chunks, chunk_size, val_dim
  )
  a_ab_inv_c = jnp.swapaxes(a_ab_inv_out, 1, 2).reshape(
      batch_size, num_heads, num_chunks, chunk_size, chunk_size
  )
  merged_out_c = jnp.swapaxes(merged_out, 1, 2).reshape(
      batch_size, num_heads, num_chunks, chunk_size, key_dim + val_dim
  )
  s_in = jnp.swapaxes(s_in_out, 1, 2).reshape(
      batch_size, num_heads, num_chunks, key_dim, val_dim
  )
  s_final = s_final_out.reshape(batch_size, num_heads, key_dim, val_dim)
  return o_c, a_ab_inv_c, merged_out_c, s_in, s_final


def rwkv7_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    gk: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> (
    tuple[jax.Array, jax.Array | None]
    | tuple[jax.Array, jax.Array | None, jax.Array, jax.Array, jax.Array]
):
  """Computes RWKV-7 forward pass using single-pass fused Pallas TPU kernel."""
  batch_size, seq_len, num_heads, key_dim = q.shape
  if chunk_size <= 0 or seq_len % chunk_size != 0:
    raise ValueError(
        f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})."
    )
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )
  val_dim = v.shape[-1]
  num_chunks = seq_len // chunk_size
  if scale is None:
    scale = key_dim**-0.5

  q_c = _chunkify(q, batch_size, num_chunks, chunk_size) * scale
  k_c = _chunkify(k, batch_size, num_chunks, chunk_size)
  v_c = _chunkify(v, batch_size, num_chunks, chunk_size)
  alpha_c = _chunkify(alpha, batch_size, num_chunks, chunk_size)
  beta_c = _chunkify(beta, batch_size, num_chunks, chunk_size)
  gk_c = _chunkify(gk, batch_size, num_chunks, chunk_size)
  gk_cs_c = jnp.cumsum(gk_c.astype(jnp.float32), axis=-2)

  if initial_state is None:
    s0 = jnp.zeros(
        (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
    )
  else:
    s0 = initial_state.astype(jnp.float32)

  o_c, a_ab_inv_c, merged_out_c, s_in, s_final = _rwkv7_fwd_pallas(
      q_c, k_c, v_c, alpha_c, beta_c, gk_c, gk_cs_c, s0
  )

  o = (
      jnp.transpose(o_c, (0, 2, 3, 1, 4))
      .reshape(batch_size, seq_len, num_heads, val_dim)
      .astype(q.dtype)
  )
  final_state = s_final.astype(q.dtype) if output_final_state else None
  if return_residuals:
    return o, final_state, a_ab_inv_c, merged_out_c, s_in
  return o, final_state
