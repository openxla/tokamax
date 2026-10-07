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
"""Tests for the DeepSeek-V4 RoPE API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.tpu.rope import api
from tokamax._src.ops.experimental.tpu.rope import reference
from tokamax._src.ops.experimental.tpu.rope import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      mode_shape_dtype=[
          ("rope", (64, 64, 512), jnp.float32),
          ("rope", (128, 512), jnp.float32),
          ("qnorm_rope", (64, 128, 512), jnp.bfloat16),
          ("rope_quant", (64, 64, 128), jnp.bfloat16),
      ],
      inverse=[False, True],
      impl=["xla", "mosaic_tpu"],
  )
  def test_basic_api(self, mode_shape_dtype, inverse, impl):
    if "mosaic" in impl and _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

    mode, shape, dtype = mode_shape_dtype
    x, positions, cos_sin_cache = test_base.make_inputs(shape, dtype)

    @jax.jit
    def f(x, positions, cos_sin_cache):
      return api.rope(
          x,
          positions,
          cos_sin_cache,
          mode=mode,
          inverse=inverse,
          implementation=impl,
      )

    out = f(x, positions, cos_sin_cache)

    with self.subTest("value"):
      match mode:
        case "rope":
          expected = reference.rope(
              x, positions, cos_sin_cache, inverse=inverse
          )
          np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
        case "qnorm_rope":
          expected = reference.qnorm_rope(
              x, positions, cos_sin_cache, 1e-6, inverse=inverse
          )
          np.testing.assert_allclose(
              out.astype(jnp.float32),
              expected.astype(jnp.float32),
              rtol=1e-2,
              atol=1e-2,
          )
        case _:
          q, scales = out
          q_expected, scales_expected = reference.rope_quant(
              x, positions, cos_sin_cache, inverse=inverse
          )
          np.testing.assert_allclose(
              scales, scales_expected, rtol=1e-6, atol=1e-6
          )
          test_base.assert_bits_equal(q, q_expected)

    with self.subTest("correct_implementation_used"):
      # Check the lowered HLO for the kernel that was really used.
      opspecs = hlo_utils.get_opspecs(
          f.lower(x, positions, cos_sin_cache), include_xla_kernels=False
      )
      if impl == "xla":
        self.assertEmpty(opspecs)
      else:
        self.assertNotEmpty(opspecs)
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )

  def test_falls_back_to_xla_for_unaligned_rank_2_quant(self):
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")
    # The Pallas kernel needs a multiple of 128 tokens for a rank 2 `x`.
    x, positions, cos_sin_cache = test_base.make_inputs((96, 128), jnp.bfloat16)
    q, scales = api.rope(x, positions, cos_sin_cache, mode="rope_quant")
    q_expected, scales_expected = reference.rope_quant(
        x, positions, cos_sin_cache
    )
    np.testing.assert_allclose(scales, scales_expected, rtol=1e-6, atol=1e-6)
    test_base.assert_bits_equal(q, q_expected)

  def test_falls_back_to_xla_for_qnorm_rope_head_dim_128(self):
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")
    x, positions, cos_sin_cache = test_base.make_inputs(
        (64, 8, 128), jnp.bfloat16
    )
    expected = reference.qnorm_rope(x, positions, cos_sin_cache, 1e-6)
    out = api.rope(x, positions, cos_sin_cache, mode="qnorm_rope")
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
  absltest.main()
