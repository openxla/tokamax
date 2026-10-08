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
"""Hypothesis tests for the DeepSeek-V4 `wo_a` projection Pallas kernels on TPU.

These sweep token counts, group counts, LoRA ranks, head and rotary widths, the
tile knobs (`tile_t`, `tile_r`, `sub_t`), `inverse` and `quantize_activations`
through the raw kernel entry points (`gather_cos_sin` and `wo_a_projection`)
and compare them with `reference`.
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
from tokamax._src.ops.experimental.tpu.o_projection import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.tpu.o_projection import test_base

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
_SUBLANE = pallas_mosaic_tpu_kernel.SUBLANE
# `gather_cos_sin` tiles the tokens by their largest divisor up to this.
_GATHER_TILE_N_CAP = 128


def _largest_divisor(x: int, cap: int) -> int:
  return max(d for d in range(1, min(x, cap) + 1) if x % d == 0)


def _aligned(tile: int, total: int) -> bool:
  """Whether `tile` may be the second-minor dim of a block over `total`."""
  return tile % _SUBLANE == 0 or tile == total


def _draw_num_tokens(data) -> int:
  """Draws a token count `gather_cos_sin` supports.

  Its `(tile_n, 2 * LANE)` output block needs `tile_n` (the largest divisor of
  the token count up to 128) to be a multiple of 8 or all the tokens.
  """
  return data.draw(
      hps.one_of(
          hps.integers(1, _GATHER_TILE_N_CAP),
          hps.integers(17, 64)
          .map(lambda k: _SUBLANE * k)
          .filter(
              lambda n: _aligned(_largest_divisor(n, _GATHER_TILE_N_CAP), n)
          ),
      ),
      label="num_tokens",
  )


def _expected_cos_sin(cos_sin_cache, positions, *, inverse):
  """`[cos | sin]` per token, widened to `LANE` lanes each.

  The NoPE lanes get the identity rotation (`cos = 1`, `sin = 0`); the roped
  lanes repeat each frequency for both channels of a pair.
  """
  cache = np.asarray(cos_sin_cache)[np.asarray(positions)]
  cos, sin = np.split(cache, 2, axis=-1)
  cos = np.repeat(cos, 2, axis=-1)
  sin = np.repeat(sin, 2, axis=-1)
  if inverse:
    sin = -sin
  nope = _LANE - cache.shape[-1]
  num_tokens = cache.shape[0]
  cos = np.concatenate([np.ones((num_tokens, nope), np.float32), cos], -1)
  sin = np.concatenate([np.zeros((num_tokens, nope), np.float32), sin], -1)
  return np.concatenate([cos, sin], axis=-1)


class PallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @hp.given(hps.data())
  def test_gather_cos_sin(self, data):
    """Checks the raw `gather_cos_sin` kernel against the expanded cache."""
    num_tokens = _draw_num_tokens(data)
    rotary_dim = 2 * data.draw(hps.integers(1, _LANE // 2), label="half_rot")
    max_position = data.draw(hps.sampled_from([1, 64, 4096]), label="max_pos")
    inverse = data.draw(hps.booleans(), label="inverse")
    out_dtype = data.draw(hps.sampled_from([jnp.float32, jnp.bfloat16]))
    seed = data.draw(hps.integers(0, 2**31 - 1), label="seed")
    _, positions, cos_sin_cache, _, _ = test_base.make_inputs(
        num_tokens=num_tokens,
        num_groups=1,
        lora_rank=_LANE,
        rotary_dim=rotary_dim,
        max_position=max_position,
        seed=seed,
    )

    cos_sin = pallas_mosaic_tpu_kernel.gather_cos_sin(
        positions, cos_sin_cache, inverse=inverse, out_dtype=out_dtype
    )
    self.assertEqual(cos_sin.shape, (num_tokens, 2 * _LANE))
    self.assertEqual(cos_sin.dtype, out_dtype)
    expected = _expected_cos_sin(cos_sin_cache, positions, inverse=inverse)
    # A pure gather: exact, up to the final cast.
    np.testing.assert_array_equal(
        np.asarray(cos_sin.astype(jnp.float32)),
        np.asarray(jnp.asarray(expected).astype(out_dtype).astype(jnp.float32)),
    )

  @hp.given(hps.data())
  def test_wo_a_projection(self, data):
    """Checks `gather_cos_sin` + `wo_a_projection` against the reference."""
    num_tokens = _draw_num_tokens(data)
    # `tile_t` is the second-minor dim of the `(tile_t, tile_r)` output and
    # `(tile_t, 2 * LANE)` cos/sin blocks, so it must be a multiple of 8 or all
    # the tokens. `None` is upstream's default, the largest divisor up to 1024.
    tile_ts: list[int | None] = [
        t
        for t in range(1, num_tokens + 1)
        if num_tokens % t == 0 and _aligned(t, num_tokens)
    ]
    if _aligned(_largest_divisor(num_tokens, 1024), num_tokens):
      tile_ts.append(None)
    tile_t = data.draw(hps.sampled_from(tile_ts), label="tile_t")
    # `sub_t` slices `tile_t` into sublane-aligned row chunks. `None` is
    # upstream's default, the largest divisor of `tile_t` up to 128.
    actual_tile_t = (
        _largest_divisor(num_tokens, 1024) if tile_t is None else tile_t
    )
    sub_ts: list[int | None] = [
        s
        for s in range(1, actual_tile_t + 1)
        if actual_tile_t % s == 0 and _aligned(s, actual_tile_t)
    ]
    if _aligned(_largest_divisor(actual_tile_t, 128), actual_tile_t):
      sub_ts.append(None)
    sub_t = data.draw(hps.sampled_from(sub_ts), label="sub_t")
    num_groups = data.draw(hps.sampled_from([1, 2, 4]), label="num_groups")
    lora_rank = data.draw(hps.sampled_from([128, 256, 512]), label="lora_rank")
    # `tile_r` is the minor dim of the output block, so a multiple of 128.
    # `None` is upstream's default, the whole LoRA rank.
    tile_r = data.draw(
        hps.sampled_from(
            [None, *(r for r in (128, 256, 512) if lora_rank % r == 0)]
        ),
        label="tile_r",
    )
    # The kernel fails to lower for `head_dim == 128`: it concatenates a
    # zero-width NoPE part, which Mosaic rejects.
    head_dim = _LANE * data.draw(hps.integers(2, 4), label="lane_blocks")
    rotary_dim = 2 * data.draw(hps.integers(1, _LANE // 2), label="half_rot")
    max_position = data.draw(hps.sampled_from([1, 64, 4096]), label="max_pos")
    inverse = data.draw(hps.booleans(), label="inverse")
    quantize_activations = data.draw(hps.booleans(), label="quantize")
    seed = data.draw(hps.integers(0, 2**31 - 1), label="seed")

    args = test_base.make_inputs(
        num_tokens=num_tokens,
        num_groups=num_groups,
        lora_rank=lora_rank,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        seed=seed,
    )
    x, positions, cos_sin_cache, wo_a, wo_a_scale = args
    cos_sin = pallas_mosaic_tpu_kernel.gather_cos_sin(
        positions, cos_sin_cache, inverse=inverse
    )
    out = pallas_mosaic_tpu_kernel.wo_a_projection(
        x,
        wo_a,
        wo_a_scale,
        cos_sin,
        tile_t=tile_t,
        tile_r=tile_r,
        sub_t=sub_t,
        quantize_activations=quantize_activations,
    )
    self.assertEqual(out.shape, (num_tokens, num_groups * lora_rank))
    self.assertEqual(out.dtype, jnp.bfloat16)
    test_base.check_output(
        out,
        args,
        inverse=inverse,
        quantize_activations=quantize_activations,
    )

  @parameterized.parameters(False, True)
  def test_fused_matches_two_step(self, quantize_activations):
    """`fused_reverse_rope_wo_a_projection` is the two kernels back to back."""
    args = test_base.make_inputs(num_tokens=96, num_groups=2, lora_rank=256)
    x, positions, cos_sin_cache, wo_a, wo_a_scale = args
    fused = pallas_mosaic_tpu_kernel.fused_reverse_rope_wo_a_projection(
        *args,
        head_dim=x.shape[-1],
        quantize_activations=quantize_activations,
    )
    cos_sin = pallas_mosaic_tpu_kernel.gather_cos_sin(
        positions, cos_sin_cache, inverse=True
    )
    two_step = pallas_mosaic_tpu_kernel.wo_a_projection(
        x,
        wo_a,
        wo_a_scale,
        cos_sin,
        quantize_activations=quantize_activations,
    )
    np.testing.assert_array_equal(
        np.asarray(fused, np.float32), np.asarray(two_step, np.float32)
    )
    test_base.check_output(
        fused, args, quantize_activations=quantize_activations
    )


if __name__ == "__main__":
  absltest.main()
