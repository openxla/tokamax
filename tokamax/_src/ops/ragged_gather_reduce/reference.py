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
"""Pure JAX reference implementation of Ragged Gather Reduce.

Ragged gather reduce is the output combine of a Mixture-of-Experts layer. The
expert outputs `x` are stored in expert-major order; route `r` of the
`input_size = num_tokens * reduce_group_size` routes reads row `indices[r]`,
scales it by `topk_weights[r]`, and adds it into destination token
`r // reduce_group_size`. Routes whose `valid_rows_mask` entry is false add
nothing, so a token with no valid route is all zeros.
"""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=("reduce_group_size",))
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
  """Pure JAX reference implementation of Ragged Gather Reduce.

  Args:
    x: `(num_rows, hidden_size)` expert outputs.
    indices: `(input_size,)` int32 row of `x` read by each route.
    topk_weights: `(input_size,)` weight of each route.
    valid_rows_mask: `(input_size,)` bool. Routes with a false entry are
      skipped.
    reduce_group_size: Number of consecutive routes summed into one output
      token, i.e. the MoE top-k. Must divide `input_size`.

  Returns:
    `(input_size // reduce_group_size, hidden_size)` array of `x.dtype`. The
    weighted sum is accumulated in float32.
  """
  out = x[indices] * topk_weights[:, None].astype(jnp.float32)
  out = jnp.where(valid_rows_mask[:, None], out, 0)
  out = out.reshape(-1, reduce_group_size, out.shape[-1])
  return jnp.sum(out, axis=1).astype(x.dtype)
