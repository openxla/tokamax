# Copyright 2026 Google LLC
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
"""Helpers shared by the mHC Pallas TPU kernels."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

# bf16 tiles are (16, 128); keep token blocks sublane-aligned.
SUBLANE = 16

# Explicit scoped-VMEM budget. Never rely on the backend default — it varies
# by Mosaic version.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def round_up(x: int, multiple: int) -> int:
  """Rounds `x` up to a multiple of `multiple`."""
  return (x + multiple - 1) // multiple * multiple


def select_token_block(
    num_tokens: int,
    token_block_size: int,
    *,
    vmem_need: Callable[[int], int] | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[int, int]:
  """Picks the grid token block and pads the token count to a multiple of it.

  When `vmem_need` is supplied, the block halves until it fits within
  `vmem_limit_bytes`, degrading to a smaller block instead of a compile-time
  VMEM OOM, with `SUBLANE` as the floor.

  Args:
    num_tokens: Number of tokens.
    token_block_size: Requested token block size.
    vmem_need: Estimates a kernel's VMEM footprint for a block of `tb` tokens.
    vmem_limit_bytes: Scoped-VMEM budget the block must fit within.

  Returns:
    `(token_block, padded_tokens)`.
  """
  tb = min(token_block_size, round_up(num_tokens, SUBLANE))
  if vmem_need is not None:
    while tb > SUBLANE and vmem_need(tb) > vmem_limit_bytes:
      tb //= 2
  return tb, round_up(num_tokens, tb)


def pad_to(padded_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
  """Zero-pads the leading (token) axis of each array to `padded_tokens`."""
  pad = padded_tokens - arrays[0].shape[0]
  if pad == 0:
    return arrays
  return tuple(jnp.pad(a, ((0, pad), (0, 0))) for a in arrays)


def trim_to(num_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
  """Inverse of `pad_to`: drops the rows the padding added."""
  if arrays[0].shape[0] == num_tokens:
    return arrays
  return tuple(a[:num_tokens] for a in arrays)


def split_fn3(fn: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Splits f32 `fn` into three bf16 chunks (8 mantissa bits each = f32's 24).

  Computed in XLA outside the kernel; the chunks are what stays VMEM-resident.
  Chunks MUST be built with reduce_precision, not dtype round-trips: XLA's
  excess-precision simplification folds f32->bf16->f32 into the identity, which
  silently zeroes the mid/lo chunks.

  Args:
    fn: The f32 gate projection.

  Returns:
    `(fn_hi, fn_mid, fn_lo)` bf16 chunks that sum to `fn`.
  """
  fn_hi = jax.lax.reduce_precision(fn, 8, 7)
  rem = fn - fn_hi
  fn_mid = jax.lax.reduce_precision(rem, 8, 7)
  fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
  return tuple(t.astype(jnp.bfloat16) for t in (fn_hi, fn_mid, fn_lo))  # pyrefly: ignore[bad-return]


def mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Gating / softmax / Sinkhorn on the tiny `(T, hc_mult3)` mix logits.

  Shared by the Pallas `pre_kernel` and the fused seam op.

  Args:
    mixes: `(T, hc_mult3)` f32 mix logits, before the RMS normalization.
    sqrsum: `(T, 1)` f32 per-token sum of squares of the residual streams.
    hc_mult: Number of residual streams `M`.
    hidden_size: Hidden size `H`.
    hc_scale: `(3,)` f32 pre/post/comb logit scales.
    hc_base: `(hc_mult3,)` f32 logit biases, in pre/post/comb order.
    rms_eps: RMS-norm epsilon.
    hc_pre_eps: Additive floor on the sigmoid pre gates.
    hc_sinkhorn_eps: Sinkhorn epsilon.
    hc_post_mult_value: Scale of the sigmoid post gates.
    sinkhorn_repeat: Number of Sinkhorn normalization rounds.

  Returns:
    `(pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M))`.
  """
  num_tokens = mixes.shape[0]

  mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

  pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
  pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

  post_logits = (
      mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
      + hc_base[hc_mult : 2 * hc_mult]
  )
  post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

  comb_logits = mixes[:, 2 * hc_mult :].reshape(
      num_tokens, hc_mult, hc_mult
  ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
  comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
  comb_mix = comb_mix / (
      jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
  )
  for _ in range(sinkhorn_repeat - 1):
    comb_mix = comb_mix / (
        jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
    )
    comb_mix = comb_mix / (
        jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
    )
  return pre_mix, post_mix, comb_mix
