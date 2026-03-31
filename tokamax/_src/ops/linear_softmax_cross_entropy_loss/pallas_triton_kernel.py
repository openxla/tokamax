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
"""Pallas-Triton forward kernel for Linear Softmax Cross-Entropy Loss."""

from functools import partial
from typing import Literal

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real, Scalar

from tokamax._src.pallas import block


def _validate_inputs(
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
) -> None:
  """Validates inputs and block-size constraints."""
  b_dim, h_dim = x.shape
  v_dim = w.shape[1]
  if b_dim % b_block_size != 0:
    raise ValueError(
        f"Batch dimension B={b_dim} must be divisible by"
        f" b_block_size={b_block_size}."
    )
  if w.shape[0] != h_dim:
    raise ValueError(
        f"w hidden dim {w.shape[0]} must match x hidden dim {h_dim}."
    )
  if h_dim % h_block_size != 0:
    raise ValueError(
        f"Hidden dimension H={h_dim} must be divisible by"
        f" h_block_size={h_block_size}."
    )
  if labels.shape[0] != b_dim:
    raise ValueError(
        f"labels batch size {labels.shape[0]} must match x batch size {b_dim}."
    )


def _lce_fwd_kernel(
    x_ref,
    labels_ref,
    w_ref,
    tile_lse_ref,
    correct_logit_ref,
    *,
    b_block_size: int,
    h_block_size: int,
    num_h_blocks: int,
    v_block_size: int,
    v_dim: int,
):
  """Per-(b_block, v_block) tile: fused matmul + logsumexp + correct-logit.

  Each program computes one tile of the logit matrix x[b_block, :] @ w[:, v_block]
  entirely in registers, never writing logits to HBM. It outputs:
    - tile_lse: per-token logsumexp over this V chunk (B, num_v_blocks)
    - correct_logit: per-token correct-class logit from this V chunk (B, num_v_blocks)

  These are combined outside the kernel: lse = logsumexp(tile_lse, axis=-1) and
  correct_logit = sum(correct_logit, axis=-1), giving the final per-token loss.

  w may be zero-padded to the next multiple of v_block_size. Padded columns are
  masked to -inf before the logsumexp so they contribute nothing. correct_logit
  uses the unmasked xw_tile; one_hot is 0 for padded columns (labels < v_dim).
  """
  v_idx = pl.program_id(1)
  v_start = v_idx * v_block_size

  # Accumulate x[b_block, :] @ w[:, v_block] across H blocks in float32.
  def h_body(h_idx, acc):
    x_tile = x_ref.at[:, block.ds(h_idx, h_block_size)].load(
        bounds_check=(False, True)
    )
    w_tile = w_ref.at[block.ds(h_idx, h_block_size), :].load(
        bounds_check=(True, False)
    )
    return acc + pl.dot(
        x_tile.astype(jnp.float32), w_tile.astype(jnp.float32)
    )

  xw_tile = jax.lax.fori_loop(
      0,
      num_h_blocks,
      h_body,
      jnp.zeros((b_block_size, v_block_size), dtype=jnp.float32),
  )

  # Mask zero-padded columns to -inf so they don't inflate the logsumexp.
  # For non-padded chunks this is a no-op (all col_idx < v_dim).
  col_idx = jnp.arange(v_block_size) + v_start  # (v_block_size,)
  xw_masked = jnp.where(col_idx[None, :] < v_dim, xw_tile, -jnp.inf)

  # Per-token logsumexp over this V chunk. Combined across V outside the kernel
  # via logsumexp(tile_lse, axis=-1) to get the global per-token LSE.
  tile_lse = jax.nn.logsumexp(xw_masked, axis=-1)  # (b_block_size,)
  tile_lse_ref.store(tile_lse[:, None])

  # Correct-class logit for tokens whose label falls in this V chunk.
  # Uses unmasked xw_tile (not xw_masked) to avoid 0 * -inf = NaN.
  # one_hot returns 0 for labels outside [0, v_block_size), so tokens
  # whose label is in a different V chunk (or in the padded region) contribute 0.
  labels_local = labels_ref.load().astype(jnp.int32) - v_start
  one_hot = jax.nn.one_hot(
      labels_local, num_classes=v_block_size, dtype=jnp.float32
  )
  correct_logit = jnp.sum(one_hot * xw_tile, axis=-1)  # (b_block_size,)
  correct_logit_ref.store(correct_logit[:, None])


@partial(
    jax.jit,
    static_argnames=[
        "b_block_size",
        "h_block_size",
        "v_block_size",
        "reduction",
        "num_warps",
    ],
)
def linear_softmax_cross_entropy_loss_fwd_pallas_triton(
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    b_block_size: int = 32,
    h_block_size: int = 64,
    v_block_size: int = 128,
    reduction: Literal["sum", "mean"] = "sum",
    num_warps: int = 4,
) -> tuple[Real[Scalar, ""], Real[Array, "B"]]:
  """Fused matmul + cross-entropy loss forward pass on GPU via Pallas/Triton.

  Tiles over (B, V) with an inner H loop, so the (b_block, v_block) logit tile
  lives only in registers -- no (B, V) materialisation in HBM.

  Args:
    x: Hidden states, shape (B, H).
    labels: Integer token indices, shape (B,).
    w: LM head weight matrix, shape (H, V).
    b_block_size: Tile size over the B (batch/token) dimension. B must be
      divisible by b_block_size.
    h_block_size: Tile size for the inner H accumulation loop.
    v_block_size: Tile size over the V (vocab) dimension. V is zero-padded
      to the next multiple of v_block_size inside this function; V need not
      be divisible by v_block_size.
    reduction: "sum" or "mean" over tokens.
    num_warps: Triton warp count (tunable).

  Returns:
    (loss, lse) where lse is the per-token log-sum-exp, saved as a residual
    for the backward pass.
  """
  _validate_inputs(x, labels, w, b_block_size, h_block_size, v_block_size)

  # bfloat16 is fine; float16 needs upcast to avoid precision loss.
  if x.dtype == jnp.float16:
    x = x.astype(jnp.float32)
  if w.dtype == jnp.float16:
    w = w.astype(jnp.float32)

  b_dim, h_dim = x.shape
  v_dim = w.shape[1]
  num_b_blocks = pl.cdiv(b_dim, b_block_size)
  num_h_blocks = pl.cdiv(h_dim, h_block_size)
  num_v_blocks = pl.cdiv(v_dim, v_block_size)
  v_padded = num_v_blocks * v_block_size

  # Pad w so its V dimension is an exact multiple of v_block_size.
  # Padded columns are zero; the kernel masks them to -inf before logsumexp.
  if v_padded != v_dim:
    w = jnp.pad(w, ((0, 0), (0, v_padded - v_dim)))

  kernel = partial(
      _lce_fwd_kernel,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      num_h_blocks=num_h_blocks,
      v_block_size=v_block_size,
      v_dim=v_dim,
  )

  # Outputs are (B, num_v_blocks): one value per token per V chunk.
  # Combining across V happens outside the kernel in plain JAX.
  tile_lse, correct_logit_contrib = block.pallas_call(
      kernel,
      name="pallas_triton_lce_fwd",
      grid=(num_b_blocks, num_v_blocks),
      out_shape=(
          jax.ShapeDtypeStruct((b_dim, num_v_blocks), jnp.float32),
          jax.ShapeDtypeStruct((b_dim, num_v_blocks), jnp.float32),
      ),
      in_specs=(
          pl.BlockSpec((b_block_size, h_dim), lambda b, v: (b, 0)),  # x
          pl.BlockSpec((b_block_size,), lambda b, v: (b,)),           # labels
          pl.BlockSpec((h_dim, v_block_size), lambda b, v: (0, v)),  # w (padded)
      ),
      out_specs=(
          pl.BlockSpec((b_block_size, 1), lambda b, v: (b, v)),  # tile_lse
          pl.BlockSpec((b_block_size, 1), lambda b, v: (b, v)),  # correct_logit
      ),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps),
  )(x, labels, w)

  # tile_lse[b, v] = logsumexp(x[b,:] @ w[:, v*vb:(v+1)*vb])
  # Global per-token LSE: logsumexp over V chunks (numerically stable).
  lse = jax.nn.logsumexp(tile_lse, axis=-1)  # (B,)

  # correct_logit_contrib[b, v] = xw[b, labels[b]] if labels[b] in v-chunk, else 0.
  # Exactly one V chunk is non-zero per token.
  correct_logit = jnp.sum(correct_logit_contrib, axis=-1)  # (B,)

  per_token_loss = -correct_logit + lse  # (B,) NLL per token

  if reduction == "sum":
    loss = jnp.sum(per_token_loss)
  else:  # mean
    loss = jnp.mean(per_token_loss)

  return loss.astype(jnp.float32), lse
