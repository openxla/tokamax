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
"""Tests for the Pallas/Mosaic DeepSeek-V4 RoPE operator on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.rope import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.rope import reference
from tokamax._src.ops.experimental.tpu.rope import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


def _check_output(mode, out, expected):
  """Compares `out` with `expected` at the suite's tolerances."""
  if mode == "rope_quant":
    q, scales = out
    q_expected, scales_expected = expected
    np.testing.assert_allclose(scales, scales_expected, rtol=1e-6, atol=1e-6)
    test_base.assert_bits_equal(q, q_expected)
    return
  tol = 1e-6 if out.dtype == jnp.float32 else 1e-2
  np.testing.assert_allclose(
      out.astype(jnp.float32),
      expected.astype(jnp.float32),
      rtol=tol,
      atol=tol,
  )


class PallasTpuRopeTest(test_base.RopeTestBase):

  def __init__(self, *args):
    super().__init__(*args, rope_fn=pallas_mosaic_tpu.PallasTpuRope())

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @parameterized.parameters(
      ("rope", (256, 64, 512), jnp.float32),
      ("rope", (1024, 512), jnp.float32),
      ("rope", (96, 8, 256), jnp.bfloat16),
      ("qnorm_rope", (256, 128, 512), jnp.bfloat16),
      ("qnorm_rope", (96, 64, 512), jnp.bfloat16),
      ("rope_quant", (256, 64, 128), jnp.bfloat16),
      ("rope_quant", (1024, 128), jnp.bfloat16),
  )
  def test_autotuning_configs(self, mode, shape, dtype):
    """Checks every autotuning config against the reference."""
    x, positions, cos_sin_cache = test_base.make_inputs(shape, dtype)
    op = pallas_mosaic_tpu.PallasTpuRope()
    ba = op.bind(x, positions, cos_sin_cache, mode=mode)
    configs = ba.autotuning_configs
    self.assertIn(op._get_heuristics_config(ba), configs)  # pylint: disable=protected-access
    self.assertGreater(len(configs), 1)

    expected = base_fn(mode)(x, positions, cos_sin_cache)
    for config in configs:
      with self.subTest(str(config)):
        # `rope` and `qnorm_rope` donate `x`; pass a copy so `x` stays valid.
        out = op.replace(config=config)(
            jnp.copy(x), positions, cos_sin_cache, mode=mode
        )
        _check_output(mode, out, expected)

  @parameterized.parameters(
      ("rope", 256, 128),
      ("rope", 100, 100),
      ("rope", 7, 7),
      ("qnorm_rope", 256, 64),
      ("qnorm_rope", 96, 48),
      ("rope_quant", 4096, 128),
  )
  def test_heuristics_config(self, mode, num_tokens, tile_n):
    """Checks the heuristic reproduces upstream's `tile_n`."""
    head_dim = 128 if mode == "rope_quant" else 512
    x, positions, cos_sin_cache = test_base.make_inputs(
        (num_tokens, 4, head_dim), jnp.bfloat16
    )
    op = pallas_mosaic_tpu.PallasTpuRope()
    ba = op.bind(x, positions, cos_sin_cache, mode=mode)
    self.assertEqual(
        op._get_heuristics_config(ba),  # pylint: disable=protected-access
        pallas_mosaic_tpu.Config(tile_n=tile_n),
    )

  @parameterized.parameters("rope", "qnorm_rope")
  def test_donates_x(self, mode):
    x, positions, cos_sin_cache = test_base.make_inputs(
        (64, 8, 512), jnp.bfloat16
    )
    pallas_mosaic_tpu.PallasTpuRope()(x, positions, cos_sin_cache, mode=mode)
    self.assertTrue(x.is_deleted())

  def test_rope_quant_does_not_donate_x(self):
    x, positions, cos_sin_cache = test_base.make_inputs(
        (64, 8, 128), jnp.bfloat16
    )
    pallas_mosaic_tpu.PallasTpuRope()(
        x, positions, cos_sin_cache, mode="rope_quant"
    )
    self.assertFalse(x.is_deleted())

  def test_qnorm_rope_head_dim_128_not_implemented(self):
    x, positions, cos_sin_cache = test_base.make_inputs(
        (64, 8, 128), jnp.bfloat16
    )
    with self.assertRaisesRegex(NotImplementedError, "head_dim"):
      pallas_mosaic_tpu.PallasTpuRope()(
          x, positions, cos_sin_cache, mode="qnorm_rope"
      )

  @parameterized.parameters(
      dict(num_tokens=96, tile_n=None), dict(num_tokens=256, tile_n=64)
  )
  def test_unaligned_rank_2_quant_not_implemented(self, num_tokens, tile_n):
    x, positions, cos_sin_cache = test_base.make_inputs(
        (num_tokens, 128), jnp.bfloat16
    )
    op = pallas_mosaic_tpu.PallasTpuRope()
    if tile_n is not None:
      op = op.replace(config=pallas_mosaic_tpu.Config(tile_n=tile_n))
    with self.assertRaisesRegex(NotImplementedError, "multiple of 128"):
      op(x, positions, cos_sin_cache, mode="rope_quant")


def base_fn(mode):
  """Returns the reference function for `mode`."""
  match mode:
    case "rope":
      return reference.rope
    case "qnorm_rope":
      return lambda x, p, c: reference.qnorm_rope(x, p, c, 1e-6)
    case _:
      return reference.rope_quant


if __name__ == "__main__":
  absltest.main()
