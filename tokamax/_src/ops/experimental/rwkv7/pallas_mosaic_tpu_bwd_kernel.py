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

# Dimensions.
#   B      batch size
#   T      sequence length (nt * tt)
#   H      number of attention heads
#   BH     flattened batch-head count (B * H)
#   K      key head dimension
#   V      value head dimension
#   tb     stream micro-batch tile extent dividing BH
#   tt     sequence chunk tile extent (chunk_size)
#   nb     grid extent over BH: nb = BH // tb
#   nt     grid extent over T: nt = T // tt

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu_kernel

# pylint: disable=invalid-name

_CHUNK_SIZE = pallas_mosaic_tpu_kernel._CHUNK_SIZE
_pick_tb = pallas_mosaic_tpu_kernel._pick_tb
_chunkify = pallas_mosaic_tpu_kernel._chunkify
_mm = pallas_mosaic_tpu_kernel._mm


def _unchunkify(x: jax.Array, B: int, T: int, H: int, D: int) -> jax.Array:
  return jnp.transpose(x, (0, 2, 3, 1, 4)).reshape(B, T, H, D)


def _rwkv7_bwd_kernel(
    # Inputs.
    q_ref,  # [1, 1, tb, tt, K] dtype  reversed along nt by index_map
    k_ref,  # [1, 1, tb, tt, K] dtype  reversed along nt by index_map
    v_ref,  # [1, 1, tb, tt, V] dtype  reversed along nt by index_map
    alpha_ref,  # [1, 1, tb, tt, K] dtype  reversed along nt by index_map
    beta_ref,  # [1, 1, tb, tt, K] dtype  reversed along nt by index_map
    gk_ref,  # [1, 1, tb, tt, K] dtype  reversed along nt by index_map
    gk_cs_ref,  # [1, 1, tb, tt, K] f32    reversed along nt by index_map
    a_ab_inv_ref,  # [1, 1, tb, tt, tt] dtype reversed along nt by index_map
    merged_out_ref,  # [1, 1, tb, tt, K + V] f32 reversed along nt by index_map
    s_in_ref,  # [1, 1, tb, K, V] f32     reversed along nt by index_map
    do_ref,  # [1, 1, tb, tt, V] dtype  reversed along nt by index_map
    d_s_final_ref,  # [1, tb, K, V] f32
    # Outputs.
    dq_ref,  # [1, 1, tb, tt, K] f32    reversed along nt by index_map
    dk_ref,  # [1, 1, tb, tt, K] f32    reversed along nt by index_map
    dv_ref,  # [1, 1, tb, tt, V] f32    reversed along nt by index_map
    dalpha_ref,  # [1, 1, tb, tt, K] f32    reversed along nt by index_map
    dbeta_ref,  # [1, 1, tb, tt, K] f32    reversed along nt by index_map
    dgk_ref,  # [1, 1, tb, tt, K] f32    reversed along nt by index_map
    d_s0_ref,  # [1, tb, K, V] f32
    # Scratch.
    ds_scratch_ref,  # [tb, K, V] f32   persists across reversed nt grid axis
    *,
    nt: int,
):
  """Single-pass fused analytical backward kernel for RWKV-7."""
  rev_t_i = pl.program_id(1)
  _, _, _, tt, K = q_ref.shape

  @pl.when(rev_t_i == 0)
  def _():
    ds_scratch_ref[...] = d_s_final_ref[0].astype(jnp.float32)

  q_i = q_ref[0, 0].astype(jnp.float32)
  k_i = k_ref[0, 0].astype(jnp.float32)
  v_i = v_ref[0, 0].astype(jnp.float32)
  alpha_i = alpha_ref[0, 0].astype(jnp.float32)
  beta_i = beta_ref[0, 0].astype(jnp.float32)
  gk_i = gk_ref[0, 0].astype(jnp.float32)
  gk_cs = gk_cs_ref[0, 0].astype(jnp.float32)
  a_ab_inv = a_ab_inv_ref[0, 0].astype(jnp.float32)
  merged_uw = merged_out_ref[0, 0].astype(jnp.float32)
  s_prev = s_in_ref[0, 0].astype(jnp.float32)
  do_i = do_ref[0, 0].astype(jnp.float32)
  ds_new = ds_scratch_ref[...]

  w_i = merged_uw[:, :, :K]
  u_i = merged_uw[:, :, K:]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (1, tt, tt), 1)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (1, tt, tt), 2)
  incl_causal = (col_idx <= row_idx).astype(jnp.float32)
  strict_causal = (col_idx < row_idx).astype(jnp.float32)

  # Re-center intra-chunk decay differences around midpoint g0 to prevent f32
  # overflow/underflow during Gram matrix reconstruction.
  gk_excl = gk_cs - gk_i
  g0 = gk_cs[:, tt // 2 : tt // 2 + 1, :]
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
  ds_scratch_ref[...] = ds_prev

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

  # Reverse cumulative sum of log-decay gradients via upper-triangular MXU
  # matmul with incl_causal.T.
  dg_cs = q_i * dq_i - k_i * dk_out - beta_i * dbeta_out
  dg_last = jnp.sum(s_new * ds_new, axis=-1)
  last_mask = (
      jax.lax.broadcasted_iota(jnp.int32, (1, tt, 1), 1) == (tt - 1)
  ).astype(jnp.float32)
  dg_cs = dg_cs + dg_last[:, None, :] * last_mask
  dg_excl = alpha_i * dalpha_i

  dg_total = dg_cs + dg_excl
  dgk_out = _mm(incl_causal.swapaxes(-1, -2), dg_total) - dg_excl

  dq_ref[0, 0] = dq_i.astype(dq_ref.dtype)
  dk_ref[0, 0] = dk_out.astype(dk_ref.dtype)
  dv_ref[0, 0] = dv_i.astype(dv_ref.dtype)
  dalpha_ref[0, 0] = dalpha_i.astype(dalpha_ref.dtype)
  dbeta_ref[0, 0] = dbeta_out.astype(dbeta_ref.dtype)
  dgk_ref[0, 0] = dgk_out.astype(dgk_ref.dtype)

  @pl.when(rev_t_i == nt - 1)
  def _():
    d_s0_ref[0] = ds_prev.astype(d_s0_ref.dtype)


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
  B, H, nt, tt, K = q_c.shape
  _, _, _, _, V = v_c.shape
  BH = B * H
  tb = _pick_tb(BH)
  nb = BH // tb

  def _prep_in(x: jax.Array, D: int) -> jax.Array:
    return jnp.swapaxes(x.reshape(nb, tb, nt, tt, D), 1, 2)

  q_in = _prep_in(q_c, K)
  k_in = _prep_in(k_c, K)
  v_in = _prep_in(v_c, V)
  alpha_in = _prep_in(alpha_c, K)
  beta_in = _prep_in(beta_c, K)
  gk_in = _prep_in(gk_c, K)
  gk_cs_in = _prep_in(gk_cs_c, K)
  a_ab_inv_in = _prep_in(a_ab_inv_c, tt)
  merged_in = _prep_in(merged_out_c, K + V)
  do_in = _prep_in(do_c, V)
  s_in = jnp.swapaxes(s_in_c.reshape(nb, tb, nt, K, V), 1, 2)
  d_s_final_in = d_s_final.reshape(nb, tb, K, V).astype(jnp.float32)

  grid = (nb, nt)
  bspec_k_rev = pl.BlockSpec(
      (1, 1, tb, tt, K),
      lambda b_i, rev_t_i: (b_i, nt - 1 - rev_t_i, 0, 0, 0),
  )
  bspec_v_rev = pl.BlockSpec(
      (1, 1, tb, tt, V),
      lambda b_i, rev_t_i: (b_i, nt - 1 - rev_t_i, 0, 0, 0),
  )
  bspec_tt_rev = pl.BlockSpec(
      (1, 1, tb, tt, tt),
      lambda b_i, rev_t_i: (b_i, nt - 1 - rev_t_i, 0, 0, 0),
  )
  bspec_kv_rev = pl.BlockSpec(
      (1, 1, tb, tt, K + V),
      lambda b_i, rev_t_i: (b_i, nt - 1 - rev_t_i, 0, 0, 0),
  )
  bspec_s_rev = pl.BlockSpec(
      (1, 1, tb, K, V),
      lambda b_i, rev_t_i: (b_i, nt - 1 - rev_t_i, 0, 0, 0),
  )
  bspec_ds = pl.BlockSpec(
      (1, tb, K, V), lambda b_i, rev_t_i: (b_i, 0, 0, 0)
  )

  interpret = jax.default_backend() != "tpu"
  dq_out, dk_out, dv_out, dalpha_out, dbeta_out, dgk_out, d_s0_out = (
      pl.pallas_call(
          functools.partial(_rwkv7_bwd_kernel, nt=nt),
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
                  bspec_tt_rev,
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
              scratch_shapes=[pltpu.VMEM((tb, K, V), jnp.float32)],
          ),
          out_shape=[
              jax.ShapeDtypeStruct((nb, nt, tb, tt, K), jnp.float32),
              jax.ShapeDtypeStruct((nb, nt, tb, tt, K), jnp.float32),
              jax.ShapeDtypeStruct((nb, nt, tb, tt, V), jnp.float32),
              jax.ShapeDtypeStruct((nb, nt, tb, tt, K), jnp.float32),
              jax.ShapeDtypeStruct((nb, nt, tb, tt, K), jnp.float32),
              jax.ShapeDtypeStruct((nb, nt, tb, tt, K), jnp.float32),
              jax.ShapeDtypeStruct((nb, tb, K, V), jnp.float32),
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

  def _unprep(x: jax.Array, D: int) -> jax.Array:
    return jnp.swapaxes(x, 1, 2).reshape(B, H, nt, tt, D)

  return (
      _unprep(dq_out, K),
      _unprep(dk_out, K),
      _unprep(dv_out, V),
      _unprep(dalpha_out, K),
      _unprep(dbeta_out, K),
      _unprep(dgk_out, K),
      d_s0_out.reshape(B, H, K, V),
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
  B, T, H, K = q.shape
  _, _, _, V = v.shape
  tt = chunk_size
  if tt <= 0 or T % tt != 0:
    raise ValueError(
        f"seq_len ({T}) must be divisible by chunk_size ({tt})."
    )
  if tt > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {tt}."
    )
  nt = T // tt
  if scale is None:
    scale = K**-0.5

  if a_ab_inv is None or merged_out is None or s_in is None:
    _, _, a_ab_inv, merged_out, s_in = (
        pallas_mosaic_tpu_kernel.rwkv7_pallas_fwd(
            q,
            k,
            v,
            alpha,
            beta,
            gk,
            chunk_size=tt,
            scale=scale,
            initial_state=initial_state,
            return_residuals=True,
        )
    )
  assert a_ab_inv is not None
  assert merged_out is not None
  assert s_in is not None

  q_c = _chunkify(q, B, nt, tt) * scale
  k_c = _chunkify(k, B, nt, tt)
  v_c = _chunkify(v, B, nt, tt)
  alpha_c = _chunkify(alpha, B, nt, tt)
  beta_c = _chunkify(beta, B, nt, tt)
  gk_c = _chunkify(gk, B, nt, tt)
  gk_cs_c = jnp.cumsum(gk_c.astype(jnp.float32), axis=-2)
  do_c = _chunkify(do, B, nt, tt)

  if d_final_state is None:
    d_s_final = jnp.zeros((B, H, K, V), dtype=jnp.float32)
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

  dq = (_unchunkify(dq_c, B, T, H, K) * scale).astype(q.dtype)
  dk = _unchunkify(dk_c, B, T, H, K).astype(k.dtype)
  dv = _unchunkify(dv_c, B, T, H, V).astype(v.dtype)
  dalpha = _unchunkify(dalpha_c, B, T, H, K).astype(alpha.dtype)
  dbeta = _unchunkify(dbeta_c, B, T, H, K).astype(beta.dtype)
  dgk = _unchunkify(dgk_c, B, T, H, K).astype(gk.dtype)
  d_s0_out = (
      d_s0.astype(initial_state.dtype) if initial_state is not None else None
  )
  return dq, dk, dv, dalpha, dbeta, dgk, d_s0_out
