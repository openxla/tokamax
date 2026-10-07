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
"""Hypothesis tests for the DeepSeek-V4 RoPE Pallas kernels on TPU.

These sweep token counts, `tile_n`, head counts, head and rotary widths, rank 2
and rank 3 inputs, `inverse` and dtypes through the raw kernel entry points
(`rope`, `qnorm_rope` and `rope_quant`) and compare them with `reference`.
"""

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.rope import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.tpu.rope import reference
from tokamax._src.ops.experimental.tpu.rope import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=10,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile(name="deterministic")

_LANE = pallas_mosaic_tpu_kernel.LANE


def _draw_tile_n(
    data, num_tokens: int, *, multiple_of: int = 1, cap: int = 128
) -> int | None:
  """Draws `None` (the kernel's default) or a `tile_n` dividing `num_tokens`."""
  divisors = [
      n
      for n in range(multiple_of, min(num_tokens, cap) + 1, multiple_of)
      if num_tokens % n == 0
  ]
  return data.draw(hps.sampled_from([None, *divisors]), label="tile_n")


def _draw_inputs(data, shape, dtype):
  """Draws `rotary_dim`, `max_position` and a seed; returns the inputs."""
  rotary_dim = 2 * data.draw(hps.integers(1, _LANE // 2), label="half_rotary")
  max_position = data.draw(hps.sampled_from([1, 64, 4096]), label="max_pos")
  seed = data.draw(hps.integers(0, 2**31 - 1), label="seed")
  return test_base.make_inputs(
      shape, dtype, seed=seed, max_position=max_position, rotary_dim=rotary_dim
  )


def _assert_close(actual, expected):
  tol = 1e-6 if actual.dtype == jnp.float32 else 1e-2
  np.testing.assert_allclose(
      actual.astype(jnp.float32),
      expected.astype(jnp.float32),
      rtol=tol,
      atol=tol,
  )


class PallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @hp.given(hps.data())
  def test_rope(self, data):
    """Checks the raw `rope` kernel against the reference."""
    head_dim = _LANE * data.draw(hps.integers(1, 4), label="lane_blocks")
    if data.draw(hps.booleans(), label="rank_3"):
      num_tokens = data.draw(hps.integers(1, 300), label="num_tokens")
      num_heads = data.draw(hps.sampled_from([1, 3, 8, 64]), label="num_heads")
      shape = (num_tokens, num_heads, head_dim)
      # `tile_n` is the leading block dim, so any divisor will do.
      tile_n = _draw_tile_n(data, num_tokens)
    else:
      # `tile_n` is the second-minor dim of the `(tile_n, LANE)` block, so it
      # must be a multiple of 8 (16 for bf16) or the whole `num_tokens`.
      num_tokens = 16 * data.draw(hps.integers(1, 64), label="num_tokens_16")
      shape = (num_tokens, head_dim)
      tile_n = _draw_tile_n(data, num_tokens, multiple_of=16)
    inverse = data.draw(hps.booleans(), label="inverse")
    dtype = data.draw(hps.sampled_from([jnp.float32, jnp.bfloat16]))
    x, positions, cos_sin_cache = _draw_inputs(data, shape, dtype)
    rotary_dim = cos_sin_cache.shape[1]
    x_np = np.asarray(x)  # Read before `x` is donated.

    expected = reference.rope(x, positions, cos_sin_cache, inverse=inverse)
    actual = pallas_mosaic_tpu_kernel.rope(
        x, positions, cos_sin_cache, inverse=inverse, tile_n=tile_n
    )
    self.assertEqual(actual.shape, shape)
    self.assertEqual(actual.dtype, dtype)
    _assert_close(actual, expected)
    # The NoPE channels pass through bit for bit.
    np.testing.assert_array_equal(
        np.asarray(actual)[..., :-rotary_dim], x_np[..., :-rotary_dim]
    )

  @hp.given(hps.data())
  def test_qnorm_rope(self, data):
    """Checks the raw `qnorm_rope` kernel against the reference."""
    num_tokens = data.draw(hps.integers(1, 300), label="num_tokens")
    # `qnorm_rope` buffers whole heads, so stay within upstream's cap of 64.
    tile_n = _draw_tile_n(data, num_tokens, cap=64)
    num_heads = data.draw(hps.sampled_from([1, 3, 8, 64]), label="num_heads")
    # `qnorm_rope` fails to lower for `head_dim == 128`: it concatenates a
    # zero-width NoPE part, which Mosaic rejects.
    head_dim = _LANE * data.draw(hps.integers(2, 4), label="lane_blocks")
    eps = data.draw(hps.sampled_from([1e-6, 1e-5]), label="eps")
    inverse = data.draw(hps.booleans(), label="inverse")
    dtype = data.draw(hps.sampled_from([jnp.float32, jnp.bfloat16]))
    shape = (num_tokens, num_heads, head_dim)
    x, positions, cos_sin_cache = _draw_inputs(data, shape, dtype)

    expected = reference.qnorm_rope(
        x, positions, cos_sin_cache, eps, inverse=inverse
    )
    actual = pallas_mosaic_tpu_kernel.qnorm_rope(
        x, positions, cos_sin_cache, eps=eps, inverse=inverse, tile_n=tile_n
    )
    self.assertEqual(actual.shape, shape)
    self.assertEqual(actual.dtype, dtype)
    # The kernel and the reference reduce the RMS in different orders.
    tol = 1e-5 if dtype == jnp.float32 else 1e-2
    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        expected.astype(jnp.float32),
        rtol=tol,
        atol=tol,
    )

  @hp.given(hps.data())
  def test_rope_quant(self, data):
    """Checks the raw `rope_quant` kernel against the reference."""
    # The scale reduces over the whole row, but only the last lane block is
    # pipelined in, so the kernel requires `head_dim == LANE`.
    if data.draw(hps.booleans(), label="rank_3"):
      # Ragged counts up to 128 (one whole-array tile), or multiples of 8.
      num_tokens = data.draw(
          hps.one_of(
              hps.integers(1, 128), hps.integers(17, 40).map(lambda k: 8 * k)
          ),
          label="num_tokens",
      )
      num_heads = data.draw(hps.sampled_from([1, 3, 8, 64]), label="num_heads")
      shape = (num_tokens, num_heads, _LANE)
      # `tile_n` is the second-minor dim of the `(tile_n, num_heads)` scales
      # block, so it must be a multiple of 8 or the whole `num_tokens`.
      tile_n = data.draw(
          hps.sampled_from([
              n
              for n in (num_tokens, 8, 16, 32, 64, 128)
              if num_tokens % n == 0 and n <= 128
          ]),
          label="tile_n",
      )
    else:
      # `tile_n` is the minor dim of the `(tile_n,)` scales block, so it must
      # be a multiple of 128 (see `pallas_mosaic_tpu.PallasTpuRope`).
      num_tokens = _LANE * data.draw(hps.integers(1, 4), label="lane_tiles")
      shape = (num_tokens, _LANE)
      tile_n = data.draw(hps.sampled_from([None, _LANE]), label="tile_n")
    inverse = data.draw(hps.booleans(), label="inverse")
    dtype = data.draw(hps.sampled_from([jnp.float32, jnp.bfloat16]))
    quant_dtype = data.draw(
        hps.sampled_from([jnp.float8_e4m3fn, jnp.float8_e5m2]),
        label="quant_dtype",
    )
    x, positions, cos_sin_cache = _draw_inputs(data, shape, dtype)

    q_expected, scales_expected = reference.rope_quant(
        x, positions, cos_sin_cache, inverse=inverse, quant_dtype=quant_dtype
    )
    q, scales = pallas_mosaic_tpu_kernel.rope_quant(
        x,
        positions,
        cos_sin_cache,
        inverse=inverse,
        quant_dtype=quant_dtype,
        tile_n=tile_n,
    )
    self.assertEqual(q.shape, shape)
    self.assertEqual(q.dtype, quant_dtype)
    self.assertEqual(scales.shape, shape[:-1])
    self.assertEqual(scales.dtype, jnp.float32)
    np.testing.assert_allclose(scales, scales_expected, rtol=1e-6, atol=1e-6)
    test_base.assert_bits_equal(q, q_expected)

  def test_rope_quant_zero_rows(self):
    """All-zero (padding) rows quantize to zeros with a zero scale."""
    x, positions, cos_sin_cache = test_base.make_inputs(
        (64, 8, _LANE), jnp.bfloat16
    )
    x = x.at[32:].set(0)
    q, scales = pallas_mosaic_tpu_kernel.rope_quant(x, positions, cos_sin_cache)
    q_expected, scales_expected = reference.rope_quant(
        x, positions, cos_sin_cache
    )
    np.testing.assert_array_equal(scales[32:], 0.0)
    np.testing.assert_allclose(scales, scales_expected, rtol=1e-6, atol=1e-6)
    test_base.assert_bits_equal(q, q_expected)

  @parameterized.parameters("rope", "qnorm_rope")
  def test_donates_x(self, mode):
    """As upstream, the in-place kernels donate `x`."""
    x, positions, cos_sin_cache = test_base.make_inputs(
        (64, 8, 512), jnp.bfloat16
    )
    kernel = getattr(pallas_mosaic_tpu_kernel, mode)
    out = kernel(x, positions, cos_sin_cache)
    self.assertTrue(x.is_deleted())
    self.assertEqual(out.shape, (64, 8, 512))

  def test_rope_quant_does_not_donate_x(self):
    x, positions, cos_sin_cache = test_base.make_inputs(
        (64, 8, _LANE), jnp.bfloat16
    )
    pallas_mosaic_tpu_kernel.rope_quant(x, positions, cos_sin_cache)
    self.assertFalse(x.is_deleted())

  def test_rope_quant_rejects_wide_heads(self):
    x, positions, cos_sin_cache = test_base.make_inputs(
        (64, 8, 2 * _LANE), jnp.bfloat16
    )
    with self.assertRaisesRegex(ValueError, "head_dim"):
      pallas_mosaic_tpu_kernel.rope_quant(x, positions, cos_sin_cache)


if __name__ == "__main__":
  absltest.main()
