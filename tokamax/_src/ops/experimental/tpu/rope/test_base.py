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
"""Shared correctness tests for DeepSeek-V4 RoPE implementations.

The cases and tolerances are ported from upstream vllm-torchtpu
`tests/kernels/deepseek_v4/rope_test.py`.
"""

from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.rope import reference

# DeepSeek-V4-Flash: head_dim 512, qk_rope_head_dim 64, 64 attention heads.
HEAD_DIM = 512
ROTARY_DIM = 64
NUM_HEADS = 64
MAX_POSITION = 4096


def assert_bits_equal(actual, expected):
  """Byte-for-byte equality, for dtypes NumPy will not compare directly."""
  np.testing.assert_array_equal(
      np.asarray(actual).view(np.uint8), np.asarray(expected).view(np.uint8)
  )


def make_cos_sin_cache(
    max_position: int, rotary_dim: int, seed: int = 0
) -> np.ndarray:
  """A cos/sin cache with the layout DeepSeek-V4 builds: `[cos | sin]`."""
  rng = np.random.default_rng(seed)
  freqs = rng.uniform(0.0, 2.0 * np.pi, (max_position, rotary_dim // 2)).astype(
      np.float32
  )
  return np.concatenate([np.cos(freqs), np.sin(freqs)], axis=-1).astype(
      np.float32
  )


def make_inputs(
    shape: tuple[int, ...],
    dtype,
    *,
    seed: int = 0,
    max_position: int = MAX_POSITION,
    rotary_dim: int = ROTARY_DIM,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Returns random `(x, positions, cos_sin_cache)`, as upstream builds them.

  Args:
    shape: The shape of `x`.
    dtype: The dtype of `x`.
    seed: The random seed.
    max_position: The number of rows of `cos_sin_cache`. Positions are drawn in
      `[0, max_position)`.
    rotary_dim: The width of `cos_sin_cache`.

  Returns:
    `(x, positions, cos_sin_cache)` as JAX arrays.
  """
  rng = np.random.default_rng(seed)
  x = rng.normal(size=shape).astype(np.float32)
  positions = rng.integers(0, max_position, size=(shape[0],)).astype(np.int32)
  cos_sin_cache = make_cos_sin_cache(max_position, rotary_dim, seed)
  return (
      jnp.asarray(x, dtype=dtype),
      jnp.asarray(positions),
      jnp.asarray(cos_sin_cache),
  )


class RopeTestBase(parameterized.TestCase):
  """Correctness suite shared by all DeepSeek-V4 RoPE implementations.

  Subclasses pass the implementation under test as `rope_fn`, called as
  `rope_fn(x, positions, cos_sin_cache, mode=..., ...)`. `rope_fn` may donate
  `x`, so the tests compute the reference first and do not reuse `x`.
  """

  def __init__(self, *args, rope_fn):
    super().__init__(*args)
    self._rope_fn = rope_fn

  def _check_rope(self, shape, inverse):
    x, positions, cos_sin_cache = make_inputs(shape, jnp.float32)
    expected = reference.rope(x, positions, cos_sin_cache, inverse=inverse)
    actual = self._rope_fn(
        x, positions, cos_sin_cache, mode="rope", inverse=inverse
    )
    self.assertEqual(actual.shape, x.shape)
    self.assertEqual(actual.dtype, x.dtype)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @parameterized.product(num_tokens=(8, 64, 256), inverse=(False, True))
  def test_matches_reference_multi_head(self, num_tokens, inverse):
    """`qnorm_rope` / `_o_proj` shape: [tokens, heads, head_dim]."""
    self._check_rope((num_tokens, NUM_HEADS, HEAD_DIM), inverse)

  @parameterized.product(num_tokens=(8, 128), inverse=(False, True))
  def test_matches_reference_single_head(self, num_tokens, inverse):
    """`kv_rope` shape: [tokens, head_dim]."""
    self._check_rope((num_tokens, HEAD_DIM), inverse)

  def test_nope_channels_untouched(self):
    """Everything before the trailing `rotary_dim` must pass through."""
    x, positions, cos_sin_cache = make_inputs((64, 8, HEAD_DIM), jnp.float32)
    x_np = np.asarray(x)  # Read before `x` may be donated.
    actual = self._rope_fn(x, positions, cos_sin_cache, mode="rope")
    np.testing.assert_array_equal(
        actual[..., :-ROTARY_DIM], x_np[..., :-ROTARY_DIM]
    )

  @parameterized.product(num_tokens=(8, 64, 256), inverse=(False, True))
  def test_rope_quant_matches_reference(self, num_tokens, inverse):
    """Indexer query shape: [tokens, index_n_heads, index_head_dim]."""
    indexer_head_dim = 128
    indexer_num_heads = 64
    shape = (num_tokens, indexer_num_heads, indexer_head_dim)
    x, positions, cos_sin_cache = make_inputs(shape, jnp.bfloat16)

    q_expected, scales_expected = reference.rope_quant(
        x,
        positions,
        cos_sin_cache,
        inverse=inverse,
        quant_dtype=jnp.float8_e4m3fn,
    )
    q, scales = self._rope_fn(
        x,
        positions,
        cos_sin_cache,
        mode="rope_quant",
        inverse=inverse,
        quant_dtype=jnp.float8_e4m3fn,
    )
    self.assertEqual(q.dtype, jnp.float8_e4m3fn)
    self.assertEqual(scales.shape, shape[:-1])
    np.testing.assert_allclose(scales, scales_expected, rtol=1e-6, atol=1e-6)
    assert_bits_equal(q, q_expected)

  @parameterized.product(num_tokens=(8, 64, 256), inverse=(False, True))
  def test_qnorm_rope_matches_reference_multi_head(self, num_tokens, inverse):
    """`qnorm_rope` shape: [tokens, heads, head_dim]."""
    num_heads = 128
    head_dim = 512
    eps = 1e-6
    x, positions, cos_sin_cache = make_inputs(
        (num_tokens, num_heads, head_dim), jnp.bfloat16
    )

    expected = reference.qnorm_rope(
        x, positions, cos_sin_cache, eps, inverse=inverse
    )
    actual = self._rope_fn(
        x, positions, cos_sin_cache, mode="qnorm_rope", eps=eps, inverse=inverse
    )
    self.assertEqual(actual.dtype, x.dtype)
    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        expected.astype(jnp.float32),
        rtol=1e-2,
        atol=1e-2,
    )
