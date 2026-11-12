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
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# pylint: disable=invalid-name

_CHUNK_SIZE = 64


def _pick_tb(BH: int) -> int:
  for candidate in (4, 2, 1):
    if BH % candidate == 0:
      return candidate
  return 1


def _chunkify(x: jax.Array, B: int, nt: int, tt: int) -> jax.Array:
  # [B, T, H, ...] -> [B, H, nt, tt, ...]
  _, _, H, *rest = x.shape
  x = x.reshape((B, nt, tt, H, *rest))
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _mm(x: jax.Array, y: jax.Array) -> jax.Array:
  # Pin HIGHEST precision on f32 contractions so TPU MXU does not downcast
  # recurrence accumulators to bf16.
  return jnp.matmul(
      x,
      y,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _rwkv7_fwd_kernel(
    # Inputs.
    q_ref,  # [1, 1, tb, tt, K] dtype
    k_ref,  # [1, 1, tb, tt, K] dtype
    v_ref,  # [1, 1, tb, tt, V] dtype
    alpha_ref,  # [1, 1, tb, tt, K] dtype
    beta_ref,  # [1, 1, tb, tt, K] dtype
    gk_ref,  # [1, 1, tb, tt, K] dtype
    gk_cs_ref,  # [1, 1, tb, tt, K] f32
    s0_ref,  # [1, tb, K, V] f32
    # Outputs.
    o_ref,  # [1, 1, tb, tt, V] dtype
    a_ab_inv_ref,  # [1, 1, tb, tt, tt] dtype
    merged_out_ref,  # [1, 1, tb, tt, K + V] f32
    s_in_ref,  # [1, 1, tb, K, V] f32
    s_final_ref,  # [1, tb, K, V] f32
    # Scratch.
    s_scratch_ref,  # [tb, K, V] f32  persists across the nt grid axis
    *,
    nt: int,
):
  """Single-pass fused RWKV-7 forward Pallas kernel."""
  t_i = pl.program_id(1)
  _, _, _, tt, K = q_ref.shape

  @pl.when(t_i == 0)
  def _():
    s_scratch_ref[...] = s0_ref[0].astype(jnp.float32)

  q_i = q_ref[0, 0].astype(jnp.float32)
  k_i = k_ref[0, 0].astype(jnp.float32)
  v_i = v_ref[0, 0].astype(jnp.float32)
  alpha_i = alpha_ref[0, 0].astype(jnp.float32)
  beta_i = beta_ref[0, 0].astype(jnp.float32)
  gk_i = gk_ref[0, 0].astype(jnp.float32)
  gk_cs = gk_cs_ref[0, 0].astype(jnp.float32)

  # Broadcasted iota avoids host-to-device constant materialization for masks.
  row_idx = jax.lax.broadcasted_iota(jnp.int32, (1, tt, tt), 1)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (1, tt, tt), 2)
  incl_causal = (col_idx <= row_idx).astype(jnp.float32)
  strict_causal = (col_idx < row_idx).astype(jnp.float32)
  eye = (row_idx == col_idx).astype(jnp.float32)

  # Center cumulative log-decay around the chunk midpoint g0 so both positive
  # and negative exponentials stay within f32 dynamic range before MXU matmul.
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
  a_ab = _mm(alpha_hat, beta_hat.swapaxes(-1, -2)) * strict_causal
  a_ak = _mm(alpha_hat, k_hat.swapaxes(-1, -2)) * strict_causal

  # Invert (I - A_ab) via Neumann series repeated squaring on the MXU in
  # log2(tt) steps rather than sequential triangular substitution on the VPU.
  p_mat = eye + a_ab
  curr = a_ab
  num_doubling_steps = max(0, math.ceil(math.log2(tt)) - 1)
  for _ in range(num_doubling_steps):
    curr = _mm(curr, curr)
    p_mat = _mm(p_mat, eye + curr)
  a_ab_inv = p_mat
  a_ab_inv_ref[0, 0] = a_ab_inv.astype(a_ab_inv_ref.dtype)

  # Fuse w_i and u_i solves into a single [tt, tt] @ [tt, K + V] matmul to
  # maximize MXU utilization.
  ak_v = _mm(a_ak, v_i)
  decayed_alpha = jnp.exp(jnp.minimum(gk_excl, 0.0)) * alpha_i
  merged_rhs = jnp.concatenate([decayed_alpha, ak_v], axis=-1)
  merged_uw = _mm(a_ab_inv, merged_rhs)
  merged_out_ref[0, 0] = merged_uw.astype(merged_out_ref.dtype)
  w_i = merged_uw[:, :, :K]
  u_i = merged_uw[:, :, K:]

  s_prev = s_scratch_ref[...]
  s_in_ref[0, 0] = s_prev.astype(s_in_ref.dtype)

  v2_i = u_i + _mm(w_i, s_prev)
  q_exp = q_i * jnp.exp(jnp.minimum(gk_cs, 0.0))
  o_i = _mm(q_exp, s_prev) + _mm(a_qk, v_i) + _mm(a_qb, v2_i)
  o_ref[0, 0] = o_i.astype(o_ref.dtype)

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
  s_scratch_ref[...] = s_new

  @pl.when(t_i == nt - 1)
  def _():
    s_final_ref[0] = s_new.astype(s_final_ref.dtype)


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
  s0_in = s0.reshape(nb, tb, K, V).astype(jnp.float32)

  grid = (nb, nt)
  bspec_k = pl.BlockSpec(
      (1, 1, tb, tt, K), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_v = pl.BlockSpec(
      (1, 1, tb, tt, V), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_tt = pl.BlockSpec(
      (1, 1, tb, tt, tt), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_kv = pl.BlockSpec(
      (1, 1, tb, tt, K + V), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_s_in = pl.BlockSpec(
      (1, 1, tb, K, V), lambda b_i, t_i: (b_i, t_i, 0, 0, 0)
  )
  bspec_s = pl.BlockSpec((1, tb, K, V), lambda b_i, t_i: (b_i, 0, 0, 0))

  interpret = jax.default_backend() != "tpu"
  o_out, a_ab_inv_out, merged_out, s_in_out, s_final_out = pl.pallas_call(
      functools.partial(_rwkv7_fwd_kernel, nt=nt),
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
          out_specs=[bspec_v, bspec_tt, bspec_kv, bspec_s_in, bspec_s],
          scratch_shapes=[pltpu.VMEM((tb, K, V), jnp.float32)],
      ),
      out_shape=[
          jax.ShapeDtypeStruct((nb, nt, tb, tt, V), v_c.dtype),
          jax.ShapeDtypeStruct((nb, nt, tb, tt, tt), q_c.dtype),
          jax.ShapeDtypeStruct((nb, nt, tb, tt, K + V), jnp.float32),
          jax.ShapeDtypeStruct((nb, nt, tb, K, V), jnp.float32),
          jax.ShapeDtypeStruct((nb, tb, K, V), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary")
      ),
      interpret=interpret,
  )(q_in, k_in, v_in, alpha_in, beta_in, gk_in, gk_cs_in, s0_in)

  o_c = jnp.swapaxes(o_out, 1, 2).reshape(B, H, nt, tt, V)
  a_ab_inv_c = jnp.swapaxes(a_ab_inv_out, 1, 2).reshape(B, H, nt, tt, tt)
  merged_out_c = jnp.swapaxes(merged_out, 1, 2).reshape(B, H, nt, tt, K + V)
  s_in = jnp.swapaxes(s_in_out, 1, 2).reshape(B, H, nt, K, V)
  s_final = s_final_out.reshape(B, H, K, V)
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
) -> tuple[Any, ...]:
  """Computes RWKV-7 forward pass using single-pass fused Pallas TPU kernel."""
  B, T, H, K = q.shape
  _, _, _, V = v.shape
  tt = chunk_size
  if tt > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {tt}."
    )
  nt = T // tt
  if scale is None:
    scale = K**-0.5

  q_c = _chunkify(q, B, nt, tt) * scale
  k_c = _chunkify(k, B, nt, tt)
  v_c = _chunkify(v, B, nt, tt)
  alpha_c = _chunkify(alpha, B, nt, tt)
  beta_c = _chunkify(beta, B, nt, tt)
  gk_c = _chunkify(gk, B, nt, tt)
  gk_cs_c = jnp.cumsum(gk_c.astype(jnp.float32), axis=-2)

  if initial_state is None:
    s0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32)

  o_c, a_ab_inv_c, merged_out_c, s_in, s_final = _rwkv7_fwd_pallas(
      q_c, k_c, v_c, alpha_c, beta_c, gk_c, gk_cs_c, s0
  )

  o = jnp.transpose(o_c, (0, 2, 3, 1, 4)).reshape(B, T, H, V).astype(q.dtype)
  final_state = s_final.astype(q.dtype) if output_final_state else None
  if return_residuals:
    return o, final_state, a_ab_inv_c, merged_out_c, s_in
  return o, final_state
