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
"""Shared correctness tests for DeepSeek-V4 `wo_a` projection implementations.

The cases and tolerances are ported from upstream vllm-torchtpu
`tests/kernels/deepseek_v4/o_projection_test.py`, which only covers
`quantize_activations=False`; the `True` cases are new.
"""

from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.o_projection import reference
from tokamax._src.ops.experimental.tpu.rope import test_base as rope_test_base

# DeepSeek-V4: head_dim 512, qk_rope_head_dim 64, 8 heads per group.
HEAD_DIM = 512
ROTARY_DIM = 64
MAX_POSITION = 4096
HEADS_PER_GROUP = 8


def make_inputs(
    *,
    num_tokens: int,
    num_groups: int,
    lora_rank: int,
    head_dim: int = HEAD_DIM,
    rotary_dim: int = ROTARY_DIM,
    max_position: int = MAX_POSITION,
    heads_per_group: int = HEADS_PER_GROUP,
    seed: int = 0,
) -> tuple[jnp.ndarray, ...]:
  """Returns random `(x, positions, cos_sin_cache, wo_a, wo_a_scale)`.

  Built as upstream builds them: `x` in the `(T, G * H, head_dim)` view the
  kernel takes.

  Args:
    num_tokens: `T`.
    num_groups: `G`.
    lora_rank: `R`, the output columns of one group.
    head_dim: The head dimension.
    rotary_dim: The width of `cos_sin_cache`.
    max_position: The number of rows of `cos_sin_cache`. Positions are drawn in
      `[0, max_position)`.
    heads_per_group: `H`.
    seed: The random seed.

  Returns:
    `(x, positions, cos_sin_cache, wo_a, wo_a_scale)` as JAX arrays.
  """
  rng = np.random.default_rng(seed)
  reduction = heads_per_group * head_dim
  num_heads = num_groups * heads_per_group
  x = jnp.asarray(
      rng.standard_normal((num_tokens, num_heads, head_dim)),
      dtype=jnp.bfloat16,
  )
  wo_a = jnp.asarray(
      rng.standard_normal((reduction, num_groups * lora_rank)),
      dtype=jnp.float8_e4m3fn,
  )
  wo_a_scale = jnp.asarray(
      rng.uniform(0.5, 1.5, size=num_groups * lora_rank), dtype=jnp.float32
  )
  positions = jnp.asarray(
      rng.integers(0, max_position, size=(num_tokens,)), dtype=jnp.int32
  )
  cos_sin_cache = jnp.asarray(
      rope_test_base.make_cos_sin_cache(max_position, rotary_dim, seed),
      dtype=jnp.float32,
  )
  return x, positions, cos_sin_cache, wo_a, wo_a_scale


def assert_close(actual, expected):
  """Upstream's tolerance for the bf16 projection."""
  np.testing.assert_allclose(
      np.asarray(actual, dtype=np.float32),
      np.asarray(expected, dtype=np.float32),
      rtol=1e-2,
      atol=1e-2,
  )


# Relative L2 error allowed for `quantize_activations=True` against the
# unquantized reference. Rounding the activations to fp8 e4m3 (3 mantissa
# bits) has an RMS relative error of about 2-3%, which carries over to the
# projection.
QUANTIZED_RTOL = 5e-2


def assert_close_quantized(actual, unquantized_expected):
  """Checks a `quantize_activations=True` result.

  The kernel's per-row fp8 scale `448 / amax` is computed in bf16, and Mosaic
  and XLA round that division differently (e.g. 37.75 vs 38.0), so the
  quantized activations, and hence the output, are not bit-reproducible by a
  pure JAX emulation. Instead, compare against the unquantized projection with
  a bound on the quantization error.

  Args:
    actual: The `quantize_activations=True` output.
    unquantized_expected: The `quantize_activations=False` reference output.
  """
  actual = np.asarray(actual, dtype=np.float32)
  expected = np.asarray(unquantized_expected, dtype=np.float32)
  rel_err = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
  if not rel_err <= QUANTIZED_RTOL:
    raise AssertionError(
        f"Relative L2 error {rel_err:.4f} exceeds {QUANTIZED_RTOL}."
    )


def check_output(out, args, *, inverse=True, quantize_activations):
  """Checks `out` against the reference at the suite's tolerances."""
  expected = reference.o_projection(
      *args, inverse=inverse, quantize_activations=False
  )
  if quantize_activations:
    assert_close_quantized(out, expected)
  else:
    assert_close(out, expected)


class OProjectionTestBase(parameterized.TestCase):
  """Correctness suite shared by all `wo_a` projection implementations.

  Subclasses pass the implementation under test as `o_projection_fn`, called
  as `o_projection_fn(x, positions, cos_sin_cache, wo_a, wo_a_scale, ...)`.
  """

  def __init__(self, *args, o_projection_fn):
    super().__init__(*args)
    self._o_projection_fn = o_projection_fn

  def _check(self, *, quantize_activations, inverse=True, **kwargs):
    args = make_inputs(**kwargs)
    out = self._o_projection_fn(
        *args, inverse=inverse, quantize_activations=quantize_activations
    )
    num_tokens = args[0].shape[0]
    self.assertEqual(out.shape, (num_tokens, args[3].shape[1]))
    self.assertEqual(out.dtype, jnp.bfloat16)
    check_output(
        out,
        args,
        inverse=inverse,
        quantize_activations=quantize_activations,
    )

  @parameterized.product(
      shape=(
          # DeepSeek-V4-Flash, unsharded.
          dict(num_tokens=256, num_groups=8, lora_rank=1024),
          # The benchmarked shape: activations [1024, 128, 512].
          dict(num_tokens=1024, num_groups=16, lora_rank=1024),
      ),
      quantize_activations=(False, True),
  )
  def test_matches_reference(self, shape, quantize_activations):
    """The fused inverse-RoPE path against `rope` + the `wo_a` einsum."""
    self._check(quantize_activations=quantize_activations, **shape)

  @parameterized.parameters(False, True)
  def test_forward_rotation(self, quantize_activations):
    self._check(
        quantize_activations=quantize_activations,
        inverse=False,
        num_tokens=64,
        num_groups=2,
        lora_rank=512,
    )
