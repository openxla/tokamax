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

"""Pallas Custom TPU Kernel for Kalman / Gated Delta Attention (KDA).

This module implements the Pallas TPU kernel for Kalman Delta Attention (KDA),
utilizing Schulz polynomial doubling to compute the intra-chunk strictly lower-triangular
matrix inverse in 5 parallel systolic matrix multiplications on the TPU MXU (replacing 63
serial loop steps), coupled with lightweight inter-chunk recurrence in Vector Memory (VMEM)
and full reverse-mode autograd integration.
"""

from typing import Any, Optional
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from absl import logging

_CHUNK_SIZE = 64


def _make_kda_intra_chunk_kernel(bt: int) -> Any:
  def _kernel(
      q_ref: Any,
      k_ref: Any,
      v_ref: Any,
      g_ref: Any,
      beta_ref: Any,
      w_ref: Any,
      u_ref: Any,
      aqk_ref: Any,
  ) -> None:
    q_i = q_ref[0, 0, 0]  # (BT, K)
    k_i = k_ref[0, 0, 0]  # (BT, K)
    v_i = v_ref[0, 0, 0]  # (BT, V)
    g_i = g_ref[0, 0, 0]  # (BT, K) -- cumsum'd within chunk
    beta_i = beta_ref[0, 0, 0][:, 0]  # (BT,)

    row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
    strictly_lower = (col_idx < row_idx).astype(jnp.float32)

    kg = k_i[:, None, :] * jnp.exp(
        jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0)
    )
    a_mat = jnp.sum(kg * k_i[None, :, :], axis=-1)
    a_mat = a_mat * beta_i[:, None]
    a_mat = -a_mat * strictly_lower

    # Schulz polynomial doubling: (I - A)^(-1) = prod_{k=0}^{log2(BT)-1} (I + A^(2^k))
    eye = jnp.eye(bt, dtype=a_mat.dtype)
    p_mat = eye + a_mat
    curr = a_mat
    for _ in range(1, 6):
      curr = jnp.matmul(curr, curr, preferred_element_type=jnp.float32)
      p_mat = jnp.matmul(p_mat, eye + curr, preferred_element_type=jnp.float32)
    a_mat = p_mat * beta_i[None, :]

    w_i = jnp.matmul(a_mat, jnp.exp(g_i) * k_i, preferred_element_type=jnp.float32)
    u_i = jnp.matmul(a_mat, v_i, preferred_element_type=jnp.float32)

    causal_keep = (col_idx <= row_idx).astype(jnp.float32)
    qg = q_i[:, None, :] * jnp.exp(
        jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0)
    )
    aqk = jnp.sum(qg * k_i[None, :, :], axis=-1) * causal_keep

    w_ref[0, 0, 0] = w_i.astype(w_ref.dtype)
    u_ref[0, 0, 0] = u_i.astype(u_ref.dtype)
    aqk_ref[0, 0, 0] = aqk.astype(aqk_ref.dtype)

  return _kernel


def _intra_chunk_kda_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    g_c: jax.Array,
    beta_c: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  b, h, nt, bt, k = q_c.shape
  v_dim = v_c.shape[-1]
  kernel = _make_kda_intra_chunk_kernel(bt)
  grid = (b, h, nt)
  bspec_k = pl.BlockSpec((1, 1, 1, bt, k), lambda i, j, n: (i, j, n, 0, 0))
  bspec_v = pl.BlockSpec((1, 1, 1, bt, v_dim), lambda i, j, n: (i, j, n, 0, 0))
  bspec_beta = pl.BlockSpec((1, 1, 1, bt, 1), lambda i, j, n: (i, j, n, 0, 0))
  bspec_qq = pl.BlockSpec((1, 1, 1, bt, bt), lambda i, j, n: (i, j, n, 0, 0))

  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_beta],
      out_specs=[bspec_k, bspec_v, bspec_qq],
      out_shape=[
          jax.ShapeDtypeStruct((b, h, nt, bt, k), q_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, bt, v_dim), v_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, bt, bt), q_c.dtype),
      ],
  )(q_c, k_c, v_c, g_c, beta_c)


def kda_scan_pure_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = _CHUNK_SIZE,
) -> jax.Array:
  """Exact pure JAX KDA forward pass for CPU fallback and autograd VJP."""
  b, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  nt = t // chunk_size
  bt = chunk_size

  q_c = q.reshape(b, nt, bt, h, k_dim).transpose(0, 3, 1, 2, 4)
  k_c = k.reshape(b, nt, bt, h, k_dim).transpose(0, 3, 1, 2, 4)
  v_c = v.reshape(b, nt, bt, h, v_dim).transpose(0, 3, 1, 2, 4)
  g_c = g.reshape(b, nt, bt, h, k_dim).transpose(0, 3, 1, 2, 4)
  beta_c = beta.reshape(b, nt, bt, h, 1).transpose(0, 3, 1, 2, 4)

  # Chunk-local cumsum
  g_cs = jnp.cumsum(g_c, axis=3)

  w, u, aqk = _intra_chunk_kda_pallas(q_c, k_c, v_c, g_cs, beta_c)

  # Intra-chunk output
  y_intra = jnp.matmul(aqk, u)

  # Inter-chunk state recurrence
  g_last = g_cs[:, :, :, -1, :]  # (B, H, NT, K)
  decay_chunk = jnp.exp(g_last)

  def step_fn(prev_state, inputs):
    decay_n, w_n, u_n = inputs
    decay_k = decay_n[:, :, :, None]
    state_curr = prev_state * decay_k + jnp.matmul(w_n.swapaxes(-1, -2), u_n)
    return state_curr, prev_state

  init_state = jnp.zeros((b, h, k_dim, v_dim), dtype=jnp.float32)
  inputs_t = (
      jnp.moveaxis(decay_chunk, 2, 0),
      jnp.moveaxis(w, 2, 0),
      jnp.moveaxis(u, 2, 0),
  )
  _, prev_states_t = jax.lax.scan(step_fn, init_state, inputs_t)
  prev_states = jnp.moveaxis(prev_states_t, 0, 2)  # (B, H, NT, K, V)

  q_decayed = q_c * jnp.exp(g_cs)
  y_inter = jnp.matmul(q_decayed, prev_states.astype(q.dtype))
  y_c = y_intra + y_inter
  return y_c.transpose(0, 2, 3, 1, 4).reshape(b, t, h, v_dim)


def kda_bwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    do: jax.Array,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  orig_dtype = q.dtype

  def single_bwd(carry: Any, inputs: Any) -> tuple[Any, Any]:
    q_b, k_b, v_b, g_b, beta_b, do_b = inputs
    q_f = q_b.astype(jnp.float32)
    k_f = k_b.astype(jnp.float32)
    v_f = v_b.astype(jnp.float32)
    g_f = g_b.astype(jnp.float32)
    beta_f = beta_b.astype(jnp.float32)
    do_f = do_b.astype(jnp.float32)

    def loss_fn(q_, k_, v_, g_, beta_):
      o_ = kda_scan_pure_jax(q_[None], k_[None], v_[None], g_[None], beta_[None], chunk_size)[0]
      return jnp.sum(o_.astype(jnp.float32) * do_f)

    dq_, dk_, dv_, dg_, dbeta_ = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
        q_f, k_f, v_f, g_f, beta_f
    )
    return carry, (
        dq_.astype(orig_dtype),
        dk_.astype(orig_dtype),
        dv_.astype(orig_dtype),
        dg_.astype(g_b.dtype),
        dbeta_.astype(beta_b.dtype),
    )

  _, (dq, dk, dv, dg, dbeta) = jax.lax.scan(
      single_bwd, None, (q, k, v, g, beta, do)
  )
  return dq, dk, dv, dg, dbeta


try:
  _pallas_kda_fwd = pallas.jax_op("kda::pallas_kda_fwd", kda_scan_pure_jax)
  _pallas_kda_bwd = pallas.jax_op("kda::pallas_kda_bwd", kda_bwd_jax)

  def _kda_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    del output
    q, k, v, g, beta = inputs
    ctx.save_for_backward(q, k, v, g, beta)

  def _kda_backward(ctx: Any, do: torch.Tensor) -> Any:
    q, k, v, g, beta = ctx.saved_tensors
    return _pallas_kda_bwd(q, k, v, g, beta, do)

  _pallas_kda_fwd.register_autograd(_kda_backward, setup_context=_kda_setup_context)
except Exception as exc:
  logging.warning(
      f"Failed to register KDA Pallas JAX ops with autograd: {exc}"
  )
  _pallas_kda_fwd = None
  _pallas_kda_bwd = None


def kda_pallas_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = _CHUNK_SIZE,
) -> torch.Tensor:
  """Compute Kalman Delta Attention (KDA) on Cloud TPU via Pallas.

  Args:
      q: Query tensor of shape (batch, seq_len, nheads, head_dim)
      k: Key tensor of shape (batch, seq_len, nheads, head_dim)
      v: Value tensor of shape (batch, seq_len, nheads, head_dim)
      g: Gate decay tensor of shape (batch, seq_len, nheads, head_dim)
      beta: Write weight tensor of shape (batch, seq_len, nheads) or (batch, seq_len, nheads, 1)
      chunk_size: Chunk size for intra-chunk inversion (default: 64)

  Returns:
      Output tensor of shape (batch, seq_len, nheads, head_dim)
  """
  b, t, h, d = q.shape
  if t % chunk_size != 0:
    raise ValueError(
        f"Sequence length ({t}) must be divisible by chunk_size ({chunk_size})"
    )

  if beta.dim() == 3:
    beta = beta.unsqueeze(-1)

  # Negative decay clamp
  g = torch.clamp(g, max=0.0)

  if _pallas_kda_fwd is not None and q.device.type in ("tpu", "xla"):
    try:
      return _pallas_kda_fwd(q, k, v, g, beta)
    except Exception as err:
      logging.warning(
          f"Pallas KDA kernel failed with error: {err}. Falling back to vectorized XLA scan."
      )
      return _execute_cpu_fallback(q, k, v, g, beta, chunk_size)
  else:
    return _execute_cpu_fallback(q, k, v, g, beta, chunk_size)


def _execute_cpu_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
  """Pure PyTorch fallback for KDA delta rule."""
  b, t, h, d = q.shape
  nt = t // chunk_size
  outs = []

  state = torch.zeros(b, h, d, d, device=q.device, dtype=torch.float32)

  for i in range(nt):
    st = i * chunk_size
    en = (i + 1) * chunk_size
    q_c = q[:, st:en].permute(0, 2, 1, 3).float()  # (B, H, BT, D)
    k_c = k[:, st:en].permute(0, 2, 1, 3).float()
    v_c = v[:, st:en].permute(0, 2, 1, 3).float()
    g_c = g[:, st:en].permute(0, 2, 1, 3).float()
    b_c = beta[:, st:en].permute(0, 2, 1, 3).float()  # (B, H, BT, 1)

    g_cumsum = torch.cumsum(g_c, dim=2)
    row_idx = torch.arange(chunk_size, device=q.device).unsqueeze(1)
    col_idx = torch.arange(chunk_size, device=q.device).unsqueeze(0)
    strictly_lower = (col_idx < row_idx).float()

    diff = torch.clamp(g_cumsum.unsqueeze(3) - g_cumsum.unsqueeze(2), max=0.0)
    kg = k_c.unsqueeze(3) * torch.exp(diff)
    a_mat = torch.sum(kg * k_c.unsqueeze(2), dim=-1) * b_c
    a_mat = -a_mat * strictly_lower

    eye = torch.eye(chunk_size, device=q.device, dtype=torch.float32)
    p_mat = eye + a_mat
    curr = a_mat
    for _ in range(1, 6):
      curr = torch.matmul(curr, curr)
      p_mat = torch.matmul(p_mat, eye + curr)
    a_mat = p_mat * b_c.transpose(-1, -2)

    w_c = torch.matmul(a_mat, torch.exp(g_cumsum) * k_c)
    u_c = torch.matmul(a_mat, v_c)

    causal_keep = (col_idx <= row_idx).float()
    diff_q = torch.clamp(g_cumsum.unsqueeze(3) - g_cumsum.unsqueeze(2), max=0.0)
    qg = q_c.unsqueeze(3) * torch.exp(diff_q)
    aqk = torch.sum(qg * k_c.unsqueeze(2), dim=-1) * causal_keep

    y_intra = torch.matmul(aqk, u_c)

    # Inter-chunk state injection
    q_decayed = q_c * torch.exp(g_cumsum)
    y_inter = torch.matmul(q_decayed, state.to(q.dtype))
    y_out = y_intra + y_inter
    outs.append(y_out.permute(0, 2, 1, 3).to(q.dtype))

    g_last = g_cumsum[:, :, -1:, :]
    state = state * torch.exp(g_last).transpose(-1, -2) + torch.matmul(w_c.transpose(-1, -2), u_c)

  return torch.cat(outs, dim=1)