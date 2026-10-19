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
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_CHUNK_SIZE = 64


def _chunkify(
    x: jax.Array, batch_size: int, num_chunks: int, chunk_size: int
) -> jax.Array:
  # (batch_size, seq_len, num_heads, ...) -> (batch_size, num_heads, num_chunks, chunk_size, ...)
  rest = x.shape[3:]
  x = x.reshape((batch_size, num_chunks, chunk_size, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _gdn2_fwd_kernel(
    # Inputs.
    q_ref,  # [1, 1, 1, BT, K] dtype
    k_ref,  # [1, 1, 1, BT, K] dtype
    v_ref,  # [1, 1, 1, BT, V] dtype
    g_ref,  # [1, 1, 1, BT, K] dtype
    b_ref,  # [1, 1, 1, BT, K] dtype
    w_ref,  # [1, 1, 1, BT, V] dtype
    s0_ref,  # [1, 1, K, V] dtype
    # Outputs.
    o_ref,  # [1, 1, 1, BT, V] dtype
    s_final_ref,  # [1, 1, K, V] dtype
    # Scratch.
    scratch_ref,  # [K, V] float32
    *,
    num_chunks: int,
) -> None:
  """Fused intra-chunk and in-VMEM inter-chunk recurrence kernel for GDN-2."""
  c = pl.program_id(2)
  bt = q_ref.shape[3]
  k_dim = q_ref.shape[4]

  @pl.when(c == 0)
  def _():
    scratch_ref[...] = s0_ref[0, 0].astype(jnp.float32)

  q_i = q_ref[0, 0, 0]  # [BT, K]
  k_i = k_ref[0, 0, 0]  # [BT, K]
  v_i = v_ref[0, 0, 0]  # [BT, V]
  g_i = g_ref[0, 0, 0]  # [BT, K]
  b_i = b_ref[0, 0, 0]  # [BT, K]
  w_i = w_ref[0, 0, 0]  # [BT, V]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)

  bk = b_i * k_i
  decay = jnp.exp(jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0))
  t_lower = -jnp.sum(bk[:, None, :] * k_i[None, :, :] * decay, axis=-1) * strictly_lower

  # Schulz polynomial doubling
  eye = jnp.eye(bt, dtype=t_lower.dtype)
  p_mat = eye + t_lower
  curr = t_lower
  num_doubling_steps = max(0, math.ceil(math.log2(bt)) - 1)
  for _ in range(num_doubling_steps):
    curr = jnp.matmul(curr, curr, preferred_element_type=jnp.float32)
    p_mat = jnp.matmul(p_mat, eye + curr, preferred_element_type=jnp.float32)
  a_mat = p_mat

  # Concatenated projection fusion (weight-stationary on TPU MXU)
  k_g_b = jnp.exp(g_i) * k_i * b_i
  wv = w_i * v_i
  merged_rhs = jnp.concat([k_g_b, wv], axis=-1)
  merged_out = jnp.matmul(a_mat, merged_rhs, preferred_element_type=jnp.float32)
  wwy_i = merged_out[:, :k_dim]
  u_i = merged_out[:, k_dim:]

  qg = q_i[:, None, :] * jnp.exp(
      jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0)
  )
  aqk = jnp.sum(qg * k_i[None, :, :], axis=-1) * causal_keep

  # Inter-chunk state update in VMEM
  s = scratch_ref[...]
  v_new = u_i - jnp.matmul(wwy_i, s, preferred_element_type=jnp.float32)
  o_i = jnp.matmul(
      q_i * jnp.exp(g_i), s, preferred_element_type=jnp.float32
  ) + jnp.matmul(aqk, v_new, preferred_element_type=jnp.float32)

  g_last = g_i[-1]
  decayed_k = jnp.exp(g_last[None, :] - g_i) * k_i
  s_update = jnp.matmul(decayed_k.T, v_new, preferred_element_type=jnp.float32)
  s_new = s * jnp.exp(g_last)[:, None] + s_update

  # Carry state in VMEM scratchpad across chunks
  scratch_ref[...] = s_new
  o_ref[0, 0, 0] = o_i.astype(o_ref.dtype)

  @pl.when(c == num_chunks - 1)
  def _():
    s_final_ref[0, 0] = s_new.astype(s_final_ref.dtype)


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
) -> tuple[jax.Array, jax.Array | None]:
  """Computes GDN-2 forward pass using Pallas TPU in-VMEM recurrence."""
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )

  batch_size, seq_len, num_heads, key_dim = q.shape
  val_dim = v.shape[-1]
  num_chunks = seq_len // chunk_size
  if scale is None:
    scale = key_dim**-0.5

  q_c = _chunkify(q, batch_size, num_chunks, chunk_size) * scale
  k_c = _chunkify(k, batch_size, num_chunks, chunk_size)
  v_c = _chunkify(v, batch_size, num_chunks, chunk_size)
  g_c = jnp.cumsum(_chunkify(g, batch_size, num_chunks, chunk_size), axis=-2)
  b_c = _chunkify(b, batch_size, num_chunks, chunk_size)
  w_c = _chunkify(w, batch_size, num_chunks, chunk_size)

  if initial_state is None:
    s0 = jnp.zeros((batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32)

  grid = (batch_size, num_heads, num_chunks)
  bspec_k = pl.BlockSpec(
      (1, 1, 1, chunk_size, key_dim), lambda i, j, n: (i, j, n, 0, 0)
  )
  bspec_v = pl.BlockSpec(
      (1, 1, 1, chunk_size, val_dim), lambda i, j, n: (i, j, n, 0, 0)
  )
  bspec_s = pl.BlockSpec(
      (1, 1, key_dim, val_dim), lambda i, j, n: (i, j, 0, 0)
  )

  interpret = jax.default_backend() == "cpu"
  o_c, s_final = pl.pallas_call(
      functools.partial(_gdn2_fwd_kernel, num_chunks=num_chunks),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_k, bspec_v, bspec_s],
      out_specs=[bspec_v, bspec_s],
      scratch_shapes=[pltpu.VMEM((key_dim, val_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct(
              (batch_size, num_heads, num_chunks, chunk_size, val_dim),
              v_c.dtype,
          ),
          jax.ShapeDtypeStruct(
              (batch_size, num_heads, key_dim, val_dim), jnp.float32
          ),
      ],
      interpret=interpret,
  )(q_c, k_c, v_c, g_c, b_c, w_c, s0)

  o = jnp.transpose(o_c, (0, 2, 3, 1, 4)).reshape(
      batch_size, seq_len, num_heads, val_dim
  )
  final_state = s_final.astype(q.dtype) if output_final_state else None
  return o, final_state
