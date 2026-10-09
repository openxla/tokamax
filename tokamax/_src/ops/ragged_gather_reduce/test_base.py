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
"""Shared correctness tests for Ragged Gather Reduce implementations."""

import functools

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather_reduce import reference

# bf16 outputs of a float32 sum whose order differs between implementations.
ATOL = RTOL = 1e-2

VALID_MODES = ("all", "none", "random", "two_per_token", "uneven_pairs")


def make_valid_mask(
    key: jax.Array, num_tokens: int, reduce_group_size: int, valid_mode: str
) -> jax.Array:
  """Returns a `(num_tokens * reduce_group_size,)` bool route validity mask.

  `all` and `none` make every / no route valid, and `random` each route with
  probability 0.7. `two_per_token` makes exactly two random routes per token
  valid, the kernel's fixed two-route fast path (needs `reduce_group_size >=
  2`). `uneven_pairs` gives even tokens three valid routes and odd tokens one,
  so a 64-token block has 128 valid routes that are not two per token (needs
  `reduce_group_size >= 3`).

  Args:
    key: PRNG key.
    num_tokens: Number of destination tokens.
    reduce_group_size: Routes per token.
    valid_mode: One of `VALID_MODES`.
  """
  shape = (num_tokens, reduce_group_size)
  match valid_mode:
    case "all":
      valid = jnp.ones(shape, jnp.bool_)
    case "none":
      valid = jnp.zeros(shape, jnp.bool_)
    case "random":
      valid = jax.random.bernoulli(key, 0.7, shape)
    case "two_per_token" | "uneven_pairs":
      # Rank of each route within its token under a random order.
      ranks = jnp.argsort(jax.random.uniform(key, shape), axis=1).argsort(1)
      if valid_mode == "two_per_token":
        num_valid = 2
      else:
        num_valid = jnp.where(jnp.arange(num_tokens) % 2 == 0, 3, 1)[:, None]
      valid = ranks < num_valid
    case _:
      raise ValueError(f"Unknown valid_mode: {valid_mode}")
  return valid.reshape(-1)


@functools.cache
def make_inputs(
    num_tokens: int,
    reduce_group_size: int,
    hidden_size: int,
    valid_mode: str = "random",
    num_rows: int | None = None,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns random bf16 `(x, indices, topk_weights, valid_rows_mask)`.

  Args:
    num_tokens: Number of destination tokens.
    reduce_group_size: Routes per token.
    hidden_size: Width of `x`.
    valid_mode: See `make_valid_mask`.
    num_rows: Rows of `x`. Defaults to the number of routes.
    seed: PRNG seed.
  """
  input_size = num_tokens * reduce_group_size
  if num_rows is None:
    num_rows = input_size
  x_key, idx_key, w_key, valid_key = jax.random.split(jax.random.key(seed), 4)
  x = jax.random.normal(x_key, (num_rows, hidden_size), jnp.bfloat16)
  indices = jax.random.randint(idx_key, (input_size,), 0, num_rows, jnp.int32)
  topk_weights = jax.random.uniform(w_key, (input_size,), jnp.bfloat16)
  valid_rows_mask = make_valid_mask(
      valid_key, num_tokens, reduce_group_size, valid_mode
  )
  return x, indices, topk_weights, valid_rows_mask


class RaggedGatherReduceTestBase(parameterized.TestCase):
  """Correctness suite shared by all Ragged Gather Reduce implementations.

  Subclasses pass the implementation under test as `gather_fn`. Results must
  match `reference.ragged_gather_reduce` within `ATOL` / `RTOL`.

  `x` is at least 64 MiB in every case, so the SparseCore op runs its kernel
  rather than its small-input XLA fallback.
  """

  def __init__(self, *args, gather_fn):
    super().__init__(*args)
    self._gather_fn = gather_fn

  @parameterized.parameters(
      # Every block has two adjacent routes per token: the fixed fast path.
      (8192, 2, 2048, "all"),
      (1024, 8, 4096, "all"),
      # DeepSeek's hidden size; not a power-of-two column split.
      (1024, 8, 7168, "random"),
      (2048, 8, 2048, "two_per_token"),
      # 128 routes per block but not two per token: the generic path.
      (4096, 4, 2048, "uneven_pairs"),
      (2048, 8, 2048, "none"),
      # Not a whole number of token blocks: exercises the token padding.
      (2100, 8, 2048, "random"),
      # An odd number of rows of `x`: exercises the row padding.
      (2048, 8, 2048, "random", 16383),
  )
  def test_correctness(
      self,
      num_tokens,
      reduce_group_size,
      hidden_size,
      valid_mode,
      num_rows=None,
  ):
    """Checks the output against the reference."""
    x, indices, topk_weights, valid_rows_mask = make_inputs(
        num_tokens, reduce_group_size, hidden_size, valid_mode, num_rows
    )

    expected = reference.ragged_gather_reduce(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )
    out = self._gather_fn(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )

    self.assertEqual(out.shape, (num_tokens, hidden_size))
    self.assertEqual(out.dtype, x.dtype)
    np.testing.assert_allclose(
        np.asarray(out, np.float32),
        np.asarray(expected, np.float32),
        atol=ATOL,
        rtol=RTOL,
    )
