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
"""Pure JAX reference implementation for TopK operator."""

import functools
import jax
import jax.numpy as jnp


@functools.partial(
    jax.jit,
    static_argnames=("k", "return_scores"),
)
def topk(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array | None = None,
    *,
    return_scores: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Pure JAX reference implementation for TopK.

  Args:
    scores: Input score matrix of shape (b, n), with float32 dtype or int32
      holding raw float32 bits.
    k: Number of top elements to select per row.
    row_lengths: Optional int32 array of shape (b,) specifying the effective
      length of each row. Defaults to n for all rows. Positions at or beyond
      row_lengths and -inf scores are never selected.
    return_scores: If True, also returns the selected scores as int32 raw
      float32 bits (-inf bits in padded slots).

  Returns:
    Top-k column indices of shape (b, k) with int32 dtype, -1 suffix-padded
    when fewer than k eligible elements exist. If return_scores is True,
    returns (indices, scores_bits) where scores_bits has shape (b, k) and
    int32 dtype.
  """
  if scores.ndim != 2:
    raise ValueError(f"scores must be 2D, got {scores.shape}")
  if scores.dtype not in (jnp.float32, jnp.int32):
    raise ValueError(f"scores must be f32 or i32, got {scores.dtype}")

  b, n = scores.shape
  scores_f32 = (
      jax.lax.bitcast_convert_type(scores, jnp.float32)
      if scores.dtype == jnp.int32
      else scores
  )

  if row_lengths is None:
    eff_lengths = jnp.full((b,), n, dtype=jnp.int32)
  else:
    eff_lengths = jnp.clip(row_lengths.astype(jnp.int32), 0, n)

  cols = jnp.arange(n, dtype=jnp.int32)[None, :]
  valid = (cols < eff_lengths[:, None]) & (scores_f32 > -jnp.inf)
  masked_scores = jnp.where(valid, scores_f32, -jnp.inf)

  top_vals, top_idx = jax.lax.top_k(masked_scores, k)
  indices = jnp.where(
      top_vals > -jnp.inf, top_idx.astype(jnp.int32), jnp.int32(-1)
  )
  if return_scores:
    scores_bits = jax.lax.bitcast_convert_type(top_vals, jnp.int32)
    return indices, scores_bits
  return indices
