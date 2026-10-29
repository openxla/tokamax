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
"""Sort-free top-k expert selection for the MoE router (Pallas).

`jax.lax.top_k` and `torch.topk` lower to a full sort of every expert plus a
slice, so at 512 experts they run a 45-stage bitonic network over three arrays
to keep 10 values. This kernel selects the same 10 with `k` max-and-mask passes
over one VMEM-resident block: `O(k*N)` instead of `O(N*log^2 N)`.
"""

import functools

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

# Clamp floor. Below any real routing score (softmax, sigmoid and sqrtsoftplus
# are all >= 0) and strictly above the dtype minimum used to mask a selected
# column, so a row's maximum is always above the mask.
NEG = -3.0e38

# Rows per grid step, measured fastest at 512 experts. The grid rounds up, so
# this does not have to divide the row count.
MAX_BLOCK_ROWS = 512


def _select_kernel(x_ref, w_ref, i_ref, *, topk: int, n: int):
  """One block of rows: top `topk` by repeated max-and-mask.

  Scores at or below `NEG` are clamped up to it first, so every row has a
  maximum some column equals and every index written is a real expert. Ties go
  to the lowest id.

  The clamp tests `x > NEG` rather than `isnan(x)` to catch all three spellings
  of the hazard -- NaN, `-inf`, `-FLT_MAX` -- any of which a whole row can be
  made of. Without it such a row would stay tied with the `finfo.min` mask and
  every pass would return column 0.
  """
  x = jnp.where(x_ref[...] > NEG, x_ref[...], NEG)
  masked = jnp.finfo(x.dtype).min
  iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
  for j in range(topk):
    m = jnp.max(x, axis=1, keepdims=True)
    # Lowest column attaining the max; the rest fill with the last expert so
    # the min never selects a phantom.
    idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
    w_ref[:, j] = m[:, 0]
    i_ref[:, j] = idx.astype(jnp.int32)
    x = jnp.where(iota == idx[:, None], masked, x)
  # No score above the sentinel means no real maximum: keep the row poisonous
  # rather than letting renormalization make it plausible.
  w_ref[...] = jnp.where(w_ref[:, :1] <= NEG, jnp.nan, w_ref[...])


def select(
    scores: jax.Array,
    topk: int,
    interpret: bool = False,
    block_rows: int = MAX_BLOCK_ROWS,
) -> tuple[jax.Array, jax.Array]:
  """Top-k over `[rows, experts]` f32 scores, values descending.

  Returns `(weights f32, indices int32)` of shape `[rows, topk]`, values
  descending. `interpret` runs the kernel on the host, for tests without a TPU.

  The last grid step reads a partial block when `rows` is not a multiple of the
  block; Mosaic discards the stores past `rows`, and every reduction is along
  the expert axis, so surplus rows cannot perturb a real one.

  Args:
    scores: `[rows, experts]` float32 scores.
    topk: The number of experts to select per row.
    interpret: Whether to run the kernel on the host.
    block_rows: The maximum rows per grid step. The block is `min(block_rows,
      rows)`, so it must be a multiple of 8 (the float32 sublane count) or at
      least `rows`. Defaults to `MAX_BLOCK_ROWS`.

  Returns:
    `(weights, indices)`, both `[rows, topk]`.
  """
  rows, n = scores.shape
  block = min(block_rows, rows)
  weights, indices = pl.pallas_call(
      functools.partial(_select_kernel, topk=topk, n=n),
      grid=(pl.cdiv(rows, block),),
      in_specs=[pl.BlockSpec((block, n), lambda i: (i, 0))],
      out_specs=[
          pl.BlockSpec((block, topk), lambda i: (i, 0)),
          pl.BlockSpec((block, topk), lambda i: (i, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((rows, topk), jnp.float32),
          jax.ShapeDtypeStruct((rows, topk), jnp.int32),
      ],
      interpret=interpret,
  )(scores)
  return weights, indices
