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

"""Pallas TPU compute kernel for Raven Gated Slot Attention (GSA)."""

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   BH     flattened batch * heads (B * H)
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension
#   tb     tile extent of BH (stream-batching factor per Pallas program)
#   tt     tile extent of T (chunk_size)
#   nb     grid extent along BH: nb = BH // tb
#   nt     grid extent along T: nt = T // tt

import functools
import os
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_CHUNK_SIZE = 64


def _resolve_interpret(interpret: bool | None = None) -> bool:
  """Resolves whether ``pallas_call`` should run in CPU interpret mode."""
  if interpret is not None:
    return interpret
  env = os.environ.get("PALLAS_INTERPRET", "").strip().lower()
  if env in ("0", "false"):
    return False
  if env in ("1", "true"):
    return True
  return jax.default_backend() != "tpu"


def _validate_chunk_size(T: int, tt: int) -> None:
  """Validates that ``chunk_size`` fits TPU VMEM layout."""
  del T
  if tt > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {tt}."
    )


def _select_tb(BH: int) -> int:
  """Selects the largest stream-batching factor ``tb <= 16`` dividing ``BH``."""
  for d in (16, 8, 4, 2):
    if BH % d == 0:
      return d
  return 1


def _chunkify(x: jax.Array, B: int, nt: int, tt: int) -> jax.Array:
  # [B, T, H, ...] -> [B, H, nt, tt, ...]
  _, _, H, *rest = x.shape
  x = x.reshape((B, nt, tt, H, *rest))
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _unchunk(
    x_c: jax.Array,
    rest_dim: int,
    B: int,
    nt: int,
    tt: int,
    H: int,
) -> jax.Array:
  x_ = jnp.transpose(x_c, (0, 2, 3, 1, 4))
  return x_.reshape(B, nt * tt, H, rest_dim)


def _decayed_gram(
    a: jax.Array, b: jax.Array, g_a: jax.Array, g_b: jax.Array
) -> jax.Array:
  """Computes decayed Gram matrix ``(a @ b^T) * exp(g_a[:, None] - g_b[None, :])``."""
  # Factor exp(g_a - g_b) through the chunk-initial baseline g0 so that both
  # factors can be absorbed into the operands before a single MXU matmul, while
  # clamping exponents to avoid f32 overflow.
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


def _decayed_apply(qk: jax.Array, s: jax.Array, g: jax.Array) -> jax.Array:
  """Computes ``out[r, m] = sum_c qk[r, c] * exp(g[r, m] - g[c, m]) * s[c, m]``."""
  # Factor the slot-dependent decay exp(g[r, m] - g[c, m]) around g0 = g[0, m]
  # so the c-dependent term scales s before the MXU contraction and the
  # r-dependent term scales the output afterward.
  g0 = g[..., :1, :]
  rhs = (s * jnp.exp(jnp.minimum(g0 - g, 80.0))).astype(jnp.float32)
  out = jnp.matmul(
      qk.astype(jnp.float32),
      rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return (out * jnp.exp(jnp.minimum(g - g0, 80.0))).astype(s.dtype)


def _stage1_fwd_kernel(
    # Inputs.
    q_ref,  # [tb, 1, tt, K] bf16/f32
    k_ref,  # [tb, 1, tt, K] bf16/f32
    s_ref,  # [tb, 1, tt, M] bf16/f32
    g_ref,  # [tb, 1, tt, M] f32
    s0_ref,  # [tb, K, M] f32
    # Outputs.
    ok_ref,  # [tb, 1, tt, M] bf16/f32
    s1_final_ref,  # [tb, K, M] f32
    s1_in_ref,  # [tb, 1, K, M] f32
    # Scratch.
    s1_scratch_ref,  # [tb, K, M] f32   persists across the nt grid axis
    *,
    nt: int,
) -> None:
  """Fused single-pass tb-batched compute kernel for Raven Stage 1."""
  t_i = pl.program_id(1)
  _, _, tt, _ = q_ref.shape

  @pl.when(t_i == 0)
  def _():
    s1_scratch_ref[...] = s0_ref[...].astype(jnp.float32)

  # Keep the recurrent state in f32 in VMEM across all nt chunks to prevent
  # precision drift over long sequences.
  s1 = s1_scratch_ref[...]  # [tb, K, M]
  s1_in_ref[:, 0] = s1.astype(s1_in_ref.dtype)

  q_i = q_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  k_i = k_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  s_i = s_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  g_i = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)

  qk = (
      jnp.matmul(
          q_i,
          jnp.swapaxes(k_i, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * causal
  )
  y_diag = _decayed_apply(qk, s_i, g_i)
  y_off = jnp.matmul(
      q_i,
      s1,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) * jnp.exp(g_i)
  ok_i = y_diag + y_off

  g_last = g_i[:, -1, :]  # [tb, M]
  decay_tail = jnp.exp(jnp.minimum(g_last[:, None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      jnp.swapaxes(k_i, -1, -2),
      s_i * decay_tail,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s1_new = s1 * jnp.exp(g_last)[:, None, :] + state_contrib

  s1_scratch_ref[...] = s1_new
  ok_ref[:, 0] = ok_i.astype(ok_ref.dtype)

  @pl.when(t_i == nt - 1)
  def _():
    s1_final_ref[...] = s1_new.astype(s1_final_ref.dtype)


def raven_pallas_stage1_fwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> (
    tuple[jax.Array, jax.Array | None]
    | tuple[tuple[jax.Array, jax.Array | None], jax.Array]
):
  """Computes Raven Stage 1 forward pass using a single-pass tb-batched Pallas kernel."""
  B, T, H, K = q.shape
  _, _, _, M = s.shape
  tt = chunk_size
  _validate_chunk_size(T, tt)
  nt = T // tt
  BH = B * H
  tb = _select_tb(BH)
  nb = BH // tb

  if scale is None:
    scale = float(K**-0.5)
  q_scaled = (q.astype(jnp.float32) * scale).astype(q.dtype)

  q_c = _chunkify(q_scaled, B, nt, tt).reshape(BH, nt, tt, K)
  k_c = _chunkify(k, B, nt, tt).reshape(BH, nt, tt, K)
  s_c = _chunkify(s, B, nt, tt).reshape(BH, nt, tt, M)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  ).reshape(BH, nt, tt, M)

  if initial_state is None:
    s0 = jnp.zeros((BH, K, M), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32).reshape(BH, K, M)

  grid = (nb, nt)
  bspec_k = pl.BlockSpec((tb, 1, tt, K), lambda b_i, t_i: (b_i, t_i, 0, 0))
  bspec_m = pl.BlockSpec((tb, 1, tt, M), lambda b_i, t_i: (b_i, t_i, 0, 0))
  bspec_s = pl.BlockSpec((tb, K, M), lambda b_i, t_i: (b_i, 0, 0))
  bspec_s_in = pl.BlockSpec((tb, 1, K, M), lambda b_i, t_i: (b_i, t_i, 0, 0))

  interpret = _resolve_interpret()
  ok_c, s1_final, s1_in_c = pl.pallas_call(
      functools.partial(_stage1_fwd_kernel, nt=nt),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_m, bspec_m, bspec_s],
      out_specs=[bspec_m, bspec_s, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((tb, K, M), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, M), s.dtype),
          jax.ShapeDtypeStruct((BH, K, M), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, K, M), jnp.float32),
      ],
      interpret=interpret,
  )(q_c, k_c, s_c, g_c, s0)

  ok = _unchunk(ok_c.reshape(B, H, nt, tt, M), M, B, nt, tt, H)
  s1_final_4d = s1_final.reshape(B, H, K, M)
  s1_in_5d = s1_in_c.reshape(B, H, nt, K, M)
  final_state = s1_final_4d.astype(q.dtype) if output_final_state else None
  if return_residuals:
    return (ok, final_state), s1_in_5d
  return ok, final_state


def _stage2_fwd_kernel(
    # Inputs.
    qv_ref,  # [tb, 1, tt, M] bf16/f32
    s_ref,  # [tb, 1, tt, M] bf16/f32
    v_ref,  # [tb, 1, tt, V] bf16/f32
    g_ref,  # [tb, 1, tt, M] f32
    s0_ref,  # [tb, M, V] f32
    # Outputs.
    out_ref,  # [tb, 1, tt, V] bf16/f32
    s2_final_ref,  # [tb, M, V] f32
    s2_in_ref,  # [tb, 1, M, V] f32
    # Scratch.
    s2_scratch_ref,  # [tb, M, V] f32   persists across the nt grid axis
    *,
    nt: int,
) -> None:
  """Fused single-pass tb-batched compute kernel for Raven Stage 2."""
  t_i = pl.program_id(1)
  _, _, tt, _ = qv_ref.shape

  @pl.when(t_i == 0)
  def _():
    s2_scratch_ref[...] = s0_ref[...].astype(jnp.float32)

  s2 = s2_scratch_ref[...]  # [tb, M, V]
  s2_in_ref[:, 0] = s2.astype(s2_in_ref.dtype)

  qv_i = qv_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  s_i = s_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  v_i = v_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]
  g_i = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)

  aqk = _decayed_gram(qv_i, s_i, g_i, g_i) * causal
  y_diag = jnp.matmul(
      aqk,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  y_off = jnp.matmul(
      qv_i * jnp.exp(g_i),
      s2,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  out_i = y_diag + y_off

  g_last = g_i[:, -1, :]  # [tb, M]
  decay_tail = jnp.exp(jnp.minimum(g_last[:, None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      jnp.swapaxes(s_i * decay_tail, -1, -2),
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s2_new = s2 * jnp.exp(g_last)[:, :, None] + state_contrib

  s2_scratch_ref[...] = s2_new
  out_ref[:, 0] = out_i.astype(out_ref.dtype)

  @pl.when(t_i == nt - 1)
  def _():
    s2_final_ref[...] = s2_new.astype(s2_final_ref.dtype)


def raven_pallas_stage2_fwd(
    qv: jax.Array,
    s: jax.Array,
    v: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> (
    tuple[jax.Array, jax.Array | None]
    | tuple[tuple[jax.Array, jax.Array | None], jax.Array]
):
  """Computes Raven Stage 2 forward pass using a single-pass tb-batched Pallas kernel."""
  B, T, H, M = qv.shape
  _, _, _, V = v.shape
  tt = chunk_size
  _validate_chunk_size(T, tt)
  nt = T // tt
  BH = B * H
  tb = _select_tb(BH)
  nb = BH // tb

  qv_c = _chunkify(qv, B, nt, tt).reshape(BH, nt, tt, M)
  s_c = _chunkify(s, B, nt, tt).reshape(BH, nt, tt, M)
  v_c = _chunkify(v, B, nt, tt).reshape(BH, nt, tt, V)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  ).reshape(BH, nt, tt, M)

  if initial_state is None:
    s0 = jnp.zeros((BH, M, V), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32).reshape(BH, M, V)

  grid = (nb, nt)
  bspec_m = pl.BlockSpec((tb, 1, tt, M), lambda b_i, t_i: (b_i, t_i, 0, 0))
  bspec_v = pl.BlockSpec((tb, 1, tt, V), lambda b_i, t_i: (b_i, t_i, 0, 0))
  bspec_s = pl.BlockSpec((tb, M, V), lambda b_i, t_i: (b_i, 0, 0))
  bspec_s_in = pl.BlockSpec((tb, 1, M, V), lambda b_i, t_i: (b_i, t_i, 0, 0))

  interpret = _resolve_interpret()
  out_c, s2_final, s2_in_c = pl.pallas_call(
      functools.partial(_stage2_fwd_kernel, nt=nt),
      grid=grid,
      in_specs=[bspec_m, bspec_m, bspec_v, bspec_m, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((tb, M, V), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, V), v.dtype),
          jax.ShapeDtypeStruct((BH, M, V), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, M, V), jnp.float32),
      ],
      interpret=interpret,
  )(qv_c, s_c, v_c, g_c, s0)

  out = _unchunk(out_c.reshape(B, H, nt, tt, V), V, B, nt, tt, H)
  s2_final_4d = s2_final.reshape(B, H, M, V)
  s2_in_5d = s2_in_c.reshape(B, H, nt, M, V)
  final_state = s2_final_4d.astype(v.dtype) if output_final_state else None
  if return_residuals:
    return (out, final_state), s2_in_5d
  return out, final_state


def raven_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    v: jax.Array | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: (
        jax.Array | tuple[jax.Array | None, jax.Array | None] | None
    ) = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> Any:
  """Computes Raven GSA forward pass (Stage 1 when ``v is None``, or full 2-stage)."""
  if v is None:
    if isinstance(initial_state, tuple):
      s1_init = initial_state[0]
    else:
      s1_init = initial_state
    if return_residuals:
      (ok, s1_final), s1_in_c = raven_pallas_stage1_fwd(
          q,
          k,
          s,
          g,
          chunk_size=chunk_size,
          scale=scale,
          initial_state=s1_init,
          output_final_state=output_final_state,
          return_residuals=True,
      )
      return (ok, s1_final), (s1_in_c, None, None)
    return raven_pallas_stage1_fwd(
        q,
        k,
        s,
        g,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=s1_init,
        output_final_state=output_final_state,
        return_residuals=False,
    )

  if isinstance(initial_state, tuple):
    s1_init, s2_init = initial_state
  else:
    s1_init, s2_init = initial_state, None

  (ok, s1_final), s1_in_c = raven_pallas_stage1_fwd(
      q,
      k,
      s,
      g,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=s1_init,
      output_final_state=output_final_state,
      return_residuals=True,
  )
  qv = jax.nn.softmax(ok.astype(jnp.float32), axis=-1).astype(v.dtype)
  (out, s2_final), s2_in_c = raven_pallas_stage2_fwd(
      qv,
      s,
      v,
      g,
      chunk_size=chunk_size,
      initial_state=s2_init,
      output_final_state=output_final_state,
      return_residuals=True,
  )
  final_state = (s1_final, s2_final) if output_final_state else None
  if return_residuals:
    return (out, final_state), (s1_in_c, qv, s2_in_c)
  return out, final_state
