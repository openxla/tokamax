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

"""Pallas TPU analytical backward kernel for RWKV-7 (DPLR Delta Rule)."""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu_kernel

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

_CHUNK_SIZE = pallas_mosaic_tpu_kernel._CHUNK_SIZE
_pick_mb = pallas_mosaic_tpu_kernel._pick_mb
_chunkify = pallas_mosaic_tpu_kernel._chunkify
_mm = pallas_mosaic_tpu_kernel._mm


def _unchunkify(
    x: jax.Array, batch_size: int, seq_len: int, num_heads: int, dim: int
) -> jax.Array:
  return jnp.transpose(x, (0, 2, 3, 1, 4)).reshape(
      batch_size, seq_len, num_heads, dim
  )


def _rwkv7_bwd_kernel(
    # Inputs.
    q_vmem_ref,  # [1, 1, MB, BT, K] dtype
    k_vmem_ref,  # [1, 1, MB, BT, K] dtype
    v_vmem_ref,  # [1, 1, MB, BT, V] dtype
    alpha_vmem_ref,  # [1, 1, MB, BT, K] dtype
    beta_vmem_ref,  # [1, 1, MB, BT, K] dtype
    gk_vmem_ref,  # [1, 1, MB, BT, K] dtype
    gk_cs_vmem_ref,  # [1, 1, MB, BT, K] float32
    a_ab_inv_vmem_ref,  # [1, 1, MB, BT, BT] dtype
    merged_out_vmem_ref,  # [1, 1, MB, BT, K + V] float32
    s_in_vmem_ref,  # [1, 1, MB, K, V] float32
    do_vmem_ref,  # [1, 1, MB, BT, V] dtype
    d_s_final_vmem_ref,  # [1, MB, K, V] float32
    # Outputs.
    dq_vmem_ref,  # [1, 1, MB, BT, K] float32
    dk_vmem_ref,  # [1, 1, MB, BT, K] float32
    dv_vmem_ref,  # [1, 1, MB, BT, V] float32
    dalpha_vmem_ref,  # [1, 1, MB, BT, K] float32
    dbeta_vmem_ref,  # [1, 1, MB, BT, K] float32
    dgk_vmem_ref,  # [1, 1, MB, BT, K] float32
    d_s0_vmem_ref,  # [1, MB, K, V] float32
    # Scratch.
    ds_scratch_vmem_ref,  # [MB, K, V] float32
    *,
    num_chunks: int,
) -> None:
  """Single-pass fused analytical backward kernel for RWKV-7."""
  c = pl.program_id(1)
  bt = q_vmem_ref.shape[3]
  k_dim = q_vmem_ref.shape[4]

  @pl.when(c == 0)
  def _init_ds() -> None:
    ds_scratch_vmem_ref[...] = d_s_final_vmem_ref[0].astype(jnp.float32)

  q_i = q_vmem_ref[0, 0].astype(jnp.float32)
  k_i = k_vmem_ref[0, 0].astype(jnp.float32)
  v_i = v_vmem_ref[0, 0].astype(jnp.float32)
  alpha_i = alpha_vmem_ref[0, 0].astype(jnp.float32)
  beta_i = beta_vmem_ref[0, 0].astype(jnp.float32)
  gk_i = gk_vmem_ref[0, 0].astype(jnp.float32)
  gk_cs = gk_cs_vmem_ref[0, 0].astype(jnp.float32)
  a_ab_inv = a_ab_inv_vmem_ref[0, 0].astype(jnp.float32)
  merged_uw = merged_out_vmem_ref[0, 0].astype(jnp.float32)
  s_prev = s_in_vmem_ref[0, 0].astype(jnp.float32)
  do_i = do_vmem_ref[0, 0].astype(jnp.float32)
  ds_new = ds_scratch_vmem_ref[...]

  w_i = merged_uw[:, :, :k_dim]
  u_i = merged_uw[:, :, k_dim:]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (1, bt, bt), 1)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (1, bt, bt), 2)
  incl_causal = (col_idx <= row_idx).astype(jnp.float32)
  strict_causal = (col_idx < row_idx).astype(jnp.float32)

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
  a_ak = _mm(alpha_hat, k_hat.swapaxes(-1, -2)) * strict_causal

  exp_gk_excl = jnp.exp(jnp.minimum(gk_excl, 0.0))
  v2_i = u_i + _mm(w_i, s_prev)

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

  dv2 = _mm(a_qb.swapaxes(-1, -2), do_i) + _mm(db_i, ds_new)
  dz = _mm(a_ab_inv.swapaxes(-1, -2), dv2)
  dv_i = (
      _mm(a_qk.swapaxes(-1, -2), do_i)
      + _mm(dk_i, ds_new)
      + _mm(a_ak.swapaxes(-1, -2), dz)
  )

  exp_gk_cs = jnp.exp(jnp.minimum(gk_cs, 0.0))
  q_exp = q_i * exp_gk_cs
  ds_prev = (
      ds_new * exp_g_last
      + _mm(q_exp.swapaxes(-1, -2), do_i)
      + _mm(w_i.swapaxes(-1, -2), dv2)
  )
  ds_scratch_vmem_ref[...] = ds_prev

  d_aqk = _mm(do_i, v_i.swapaxes(-1, -2)) * incl_causal
  d_aqb = _mm(do_i, v2_i.swapaxes(-1, -2)) * incl_causal
  d_aab = _mm(dz, v2_i.swapaxes(-1, -2)) * strict_causal
  d_aak = _mm(dz, v_i.swapaxes(-1, -2)) * strict_causal

  dq_o3 = _mm(do_i, s_prev.swapaxes(-1, -2)) * exp_gk_cs
  dq_gram = exp_pos_incl * (_mm(d_aqk, k_hat) + _mm(d_aqb, beta_hat))
  dq_i = dq_o3 + dq_gram

  dalpha_w = _mm(dz, s_prev.swapaxes(-1, -2)) * exp_gk_excl
  dalpha_gram = exp_pos_excl * (_mm(d_aab, beta_hat) + _mm(d_aak, k_hat))
  dalpha_i = dalpha_w + dalpha_gram

  dk_s = _mm(v_i, ds_new.swapaxes(-1, -2)) * decay_last
  dk_gram = exp_neg * (
      _mm(d_aqk.swapaxes(-1, -2), q_hat)
      + _mm(d_aak.swapaxes(-1, -2), alpha_hat)
  )
  dk_out = dk_s + dk_gram

  dbeta_s = _mm(v2_i, ds_new.swapaxes(-1, -2)) * decay_last
  dbeta_gram = exp_neg * (
      _mm(d_aqb.swapaxes(-1, -2), q_hat)
      + _mm(d_aab.swapaxes(-1, -2), alpha_hat)
  )
  dbeta_out = dbeta_s + dbeta_gram

  dg_cs = q_i * dq_i - k_i * dk_out - beta_i * dbeta_out
  dg_last = jnp.sum(s_new * ds_new, axis=-1)
  last_mask = (
      jax.lax.broadcasted_iota(jnp.int32, (1, bt, 1), 1) == (bt - 1)
  ).astype(jnp.float32)
  dg_cs = dg_cs + dg_last[:, None, :] * last_mask
  dg_excl = alpha_i * dalpha_i

  dg_total = dg_cs + dg_excl
  dgk_out = _mm(incl_causal.swapaxes(-1, -2), dg_total) - dg_excl

  dq_vmem_ref[0, 0] = dq_i.astype(dq_vmem_ref.dtype)
  dk_vmem_ref[0, 0] = dk_out.astype(dk_vmem_ref.dtype)
  dv_vmem_ref[0, 0] = dv_i.astype(dv_vmem_ref.dtype)
  dalpha_vmem_ref[0, 0] = dalpha_i.astype(dalpha_vmem_ref.dtype)
  dbeta_vmem_ref[0, 0] = dbeta_out.astype(dbeta_vmem_ref.dtype)
  dgk_vmem_ref[0, 0] = dgk_out.astype(dgk_vmem_ref.dtype)

  @pl.when(c == num_chunks - 1)
  def _store_ds0() -> None:
    d_s0_vmem_ref[0] = ds_prev.astype(d_s0_vmem_ref.dtype)


def _rwkv7_bwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    alpha_c: jax.Array,
    beta_c: jax.Array,
    gk_c: jax.Array,
    gk_cs_c: jax.Array,
    a_ab_inv_c: jax.Array,
    merged_out_c: jax.Array,
    s_in_c: jax.Array,
    do_c: jax.Array,
    d_s_final: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Executes single-pass fused analytical backward kernel for RWKV-7."""
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
  a_ab_inv_in = _prep_in(a_ab_inv_c, chunk_size)
  merged_in = _prep_in(merged_out_c, key_dim + val_dim)
  do_in = _prep_in(do_c, val_dim)
  s_in = jnp.swapaxes(
      s_in_c.reshape(nb, mb, num_chunks, key_dim, val_dim), 1, 2
  )
  d_s_final_in = d_s_final.reshape(nb, mb, key_dim, val_dim).astype(
      jnp.float32
  )

  grid = (nb, num_chunks)
  bspec_k_rev = pl.BlockSpec(
      (1, 1, mb, chunk_size, key_dim),
      lambda i_b, c: (i_b, num_chunks - 1 - c, 0, 0, 0),
  )
  bspec_v_rev = pl.BlockSpec(
      (1, 1, mb, chunk_size, val_dim),
      lambda i_b, c: (i_b, num_chunks - 1 - c, 0, 0, 0),
  )
  bspec_bt_rev = pl.BlockSpec(
      (1, 1, mb, chunk_size, chunk_size),
      lambda i_b, c: (i_b, num_chunks - 1 - c, 0, 0, 0),
  )
  bspec_kv_rev = pl.BlockSpec(
      (1, 1, mb, chunk_size, key_dim + val_dim),
      lambda i_b, c: (i_b, num_chunks - 1 - c, 0, 0, 0),
  )
  bspec_s_rev = pl.BlockSpec(
      (1, 1, mb, key_dim, val_dim),
      lambda i_b, c: (i_b, num_chunks - 1 - c, 0, 0, 0),
  )
  bspec_ds = pl.BlockSpec(
      (1, mb, key_dim, val_dim), lambda i_b, c: (i_b, 0, 0, 0)
  )

  interpret = jax.default_backend() == "cpu"
  dq_out, dk_out, dv_out, dalpha_out, dbeta_out, dgk_out, d_s0_out = (
      pl.pallas_call(
          functools.partial(_rwkv7_bwd_kernel, num_chunks=num_chunks),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=grid,
              in_specs=[
                  bspec_k_rev,
                  bspec_k_rev,
                  bspec_v_rev,
                  bspec_k_rev,
                  bspec_k_rev,
                  bspec_k_rev,
                  bspec_k_rev,
                  bspec_bt_rev,
                  bspec_kv_rev,
                  bspec_s_rev,
                  bspec_v_rev,
                  bspec_ds,
              ],
              out_specs=[
                  bspec_k_rev,
                  bspec_k_rev,
                  bspec_v_rev,
                  bspec_k_rev,
                  bspec_k_rev,
                  bspec_k_rev,
                  bspec_ds,
              ],
              scratch_shapes=[
                  pltpu.VMEM((mb, key_dim, val_dim), jnp.float32)
              ],
          ),
          out_shape=[
              jax.ShapeDtypeStruct(
                  (nb, num_chunks, mb, chunk_size, key_dim), jnp.float32
              ),
              jax.ShapeDtypeStruct(
                  (nb, num_chunks, mb, chunk_size, key_dim), jnp.float32
              ),
              jax.ShapeDtypeStruct(
                  (nb, num_chunks, mb, chunk_size, val_dim), jnp.float32
              ),
              jax.ShapeDtypeStruct(
                  (nb, num_chunks, mb, chunk_size, key_dim), jnp.float32
              ),
              jax.ShapeDtypeStruct(
                  (nb, num_chunks, mb, chunk_size, key_dim), jnp.float32
              ),
              jax.ShapeDtypeStruct(
                  (nb, num_chunks, mb, chunk_size, key_dim), jnp.float32
              ),
              jax.ShapeDtypeStruct((nb, mb, key_dim, val_dim), jnp.float32),
          ],
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "arbitrary")
          ),
          interpret=interpret,
      )(
          q_in,
          k_in,
          v_in,
          alpha_in,
          beta_in,
          gk_in,
          gk_cs_in,
          a_ab_inv_in,
          merged_in,
          s_in,
          do_in,
          d_s_final_in,
      )
  )

  def _unprep(x: jax.Array, d: int) -> jax.Array:
    return jnp.swapaxes(x, 1, 2).reshape(
        batch_size, num_heads, num_chunks, chunk_size, d
    )

  return (
      _unprep(dq_out, key_dim),
      _unprep(dk_out, key_dim),
      _unprep(dv_out, val_dim),
      _unprep(dalpha_out, key_dim),
      _unprep(dbeta_out, key_dim),
      _unprep(dgk_out, key_dim),
      d_s0_out.reshape(batch_size, num_heads, key_dim, val_dim),
  )


def rwkv7_pallas_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    gk: jax.Array,
    do: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    a_ab_inv: jax.Array | None = None,
    merged_out: jax.Array | None = None,
    s_in: jax.Array | None = None,
    d_final_state: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
]:
  """Computes RWKV-7 analytical backward pass using Pallas TPU kernel."""
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

  if a_ab_inv is None or merged_out is None or s_in is None:
    _, _, a_ab_inv, merged_out, s_in = (
        pallas_mosaic_tpu_kernel.rwkv7_pallas_fwd(
            q,
            k,
            v,
            alpha,
            beta,
            gk,
            chunk_size=chunk_size,
            scale=scale,
            initial_state=initial_state,
            return_residuals=True,
        )
    )

  q_c = _chunkify(q, batch_size, num_chunks, chunk_size) * scale
  k_c = _chunkify(k, batch_size, num_chunks, chunk_size)
  v_c = _chunkify(v, batch_size, num_chunks, chunk_size)
  alpha_c = _chunkify(alpha, batch_size, num_chunks, chunk_size)
  beta_c = _chunkify(beta, batch_size, num_chunks, chunk_size)
  gk_c = _chunkify(gk, batch_size, num_chunks, chunk_size)
  gk_cs_c = jnp.cumsum(gk_c.astype(jnp.float32), axis=-2)
  do_c = _chunkify(do, batch_size, num_chunks, chunk_size)

  if d_final_state is None:
    d_s_final = jnp.zeros(
        (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
    )
  else:
    d_s_final = d_final_state.astype(jnp.float32)

  dq_c, dk_c, dv_c, dalpha_c, dbeta_c, dgk_c, d_s0 = _rwkv7_bwd_pallas(
      q_c,
      k_c,
      v_c,
      alpha_c,
      beta_c,
      gk_c,
      gk_cs_c,
      a_ab_inv,
      merged_out,
      s_in,
      do_c,
      d_s_final,
  )

  dq = (
      _unchunkify(dq_c, batch_size, seq_len, num_heads, key_dim) * scale
  ).astype(q.dtype)
  dk = _unchunkify(dk_c, batch_size, seq_len, num_heads, key_dim).astype(
      k.dtype
  )
  dv = _unchunkify(dv_c, batch_size, seq_len, num_heads, val_dim).astype(
      v.dtype
  )
  dalpha = _unchunkify(
      dalpha_c, batch_size, seq_len, num_heads, key_dim
  ).astype(alpha.dtype)
  dbeta = _unchunkify(
      dbeta_c, batch_size, seq_len, num_heads, key_dim
  ).astype(beta.dtype)
  dgk = _unchunkify(dgk_c, batch_size, seq_len, num_heads, key_dim).astype(
      gk.dtype
  )
  d_s0_out = (
      d_s0.astype(initial_state.dtype) if initial_state is not None else None
  )
  return dq, dk, dv, dalpha, dbeta, dgk, d_s0_out
