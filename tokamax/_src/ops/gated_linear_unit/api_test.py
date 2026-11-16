# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
import functools
import re
import typing
from typing import Final
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src import hlo_utils_common
from tokamax._src import jaxtyping
from tokamax._src.ops.gated_linear_unit import api
from tokamax._src.ops.gated_linear_unit import test_base

_IMPLEMENTATIONS: Final[tuple[str | None, ...]] = typing.get_args(
    api.Implementation.__value__
) + (None,)

_DISALLOWED_STABLEHLO_OPS_RE: Final[re.Pattern[str]] = re.compile(
    r"\bstablehlo\.(?:slice|dynamic_slice|concatenate|pad)\b"
)
_DISALLOWED_COMPILED_HLO_OPS_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:slice|dynamic-slice|concatenate|pad)\(%"
)


def _get_input_data(m, k, n, dtype=jnp.bfloat16, batch=()):
  rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
  lhs = jax.random.normal(rng0, (*batch, m, k), dtype=dtype)
  rhs = jax.random.normal(rng1, (k, 2, n), dtype=dtype)
  return (lhs, rhs)


def _assert_no_xla_slices_or_joins(
    testcase: absltest.TestCase,
    lowered: jax.stages.Lowered,
    *,
    use_tuple_weights: bool,
    implementation: str | None,
) -> None:
  stablehlo_text = lowered.as_text("stablehlo")
  testcase.assertIsNone(
      _DISALLOWED_STABLEHLO_OPS_RE.search(stablehlo_text),
      msg=f"Unexpected slice/join op in StableHLO:\n{stablehlo_text}",
  )

  compiled_hlo_text = lowered.compile().as_text()
  assert compiled_hlo_text is not None
  testcase.assertIsNone(
      _DISALLOWED_COMPILED_HLO_OPS_RE.search(compiled_hlo_text),
      msg=f"Unexpected slice/join op in HLO:\n{compiled_hlo_text}",
  )

  kernels = hlo_utils.get_kernel_info(lowered, include_xla_kernels=True)
  testcase.assertLen(kernels, 1)
  if implementation == "mosaic":
    testcase.assertIsInstance(kernels[0], hlo_utils_common.MosaicGpuKernelInfo)
  elif implementation == "triton":
    testcase.assertIsInstance(kernels[0], hlo_utils_common.TritonKernelInfo)
  else:
    testcase.assertIsInstance(
        kernels[0],
        (
            hlo_utils_common.MosaicGpuKernelInfo,
            hlo_utils_common.TritonKernelInfo,
        ),
    )
  expected_num_inputs = 3 if use_tuple_weights else 2
  testcase.assertLen(kernels[0].inputs, expected_num_inputs)


class GatedLinearUnitTest(parameterized.TestCase):

  @parameterized.product(
      implementation=_IMPLEMENTATIONS,
      use_tuple_weights=[True, False],
  )
  def test_basic_api(self, implementation, use_tuple_weights):
    if implementation == "triton" and not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")

    if not gpu_utils.has_mosaic_gpu_support() and implementation is not None:
      if "mosaic" in implementation:
        self.skipTest("Mosaic not supported on this platform.")

    if implementation == "cutedsl":
      if not gpu_utils.is_sm100():
        self.skipTest("CuteDSL requires SM100+ GPU.")
      if "cutedsl" not in api.IMPLEMENTATIONS:
        self.skipTest("CuteDSL requires nvidia-cudnn-frontend (cudnn).")

    lhs, rhs = _get_input_data(m=128, k=64, n=128)
    activation = jax.nn.silu

    @jax.jit
    def f(x, weights):
      return api.gated_linear_unit(
          x, weights, activation=activation, implementation=implementation
      )

    @jax.jit
    def f_xla(x, weights):
      return api.gated_linear_unit(
          x, weights, activation=activation, implementation="xla"
      )

    if use_tuple_weights:
      rhs = jnp.unstack(rhs, axis=1)

    out = f(lhs, rhs)
    out_golden = f_xla(lhs, rhs)

    with self.subTest("value"):
      chex.assert_trees_all_close(out, out_golden, atol=0.1, rtol=0.02)

    lowered = f.lower(lhs, rhs)
    args = hlo_utils.get_bound_args(lowered)
    self.assertLen(args, 1)

    self.assertEqual(lhs.dtype, jnp.bfloat16)
    self.assertEqual(
        args[0].arguments["precision"],
        (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
    )

    with self.subTest("correct_implementation_used"):
      op = args[0].op
      if implementation is None:
        if jax.default_backend() == "gpu":
          # Ensure either a Triton or Mosaic kernel is used.
          self.assertTrue(
              isinstance(op, api.IMPLEMENTATIONS["triton"].__class__)
              or isinstance(op, api.IMPLEMENTATIONS["mosaic"].__class__)
          )
      else:
        self.assertIsInstance(op, api.IMPLEMENTATIONS[implementation].__class__)

    if implementation in ("mosaic", "triton") or (
        implementation is None and jax.default_backend() == "gpu"
    ):
      with self.subTest("no_xla_slices_or_joins"):
        _assert_no_xla_slices_or_joins(
            self,
            lowered,
            use_tuple_weights=use_tuple_weights,
            implementation=implementation,
        )

  @parameterized.product(
      implementation=("mosaic", "triton", None),
      use_tuple_weights=(False, True),
      use_vmap=(False, True),
  )
  def test_batched_and_vmap_no_xla_slices_or_joins(
      self, implementation, use_tuple_weights, use_vmap
  ):
    if jax.default_backend() != "gpu":
      self.skipTest("GPU kernel HLO test requires GPU backend.")
    if implementation == "triton" and not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")
    if implementation == "mosaic" and not gpu_utils.has_mosaic_gpu_support():
      self.skipTest("Mosaic not supported on this platform.")

    lhs, rhs = _get_input_data(m=128, k=64, n=128, batch=(2,))
    if use_tuple_weights:
      rhs = jnp.unstack(rhs, axis=1)

    def single_fn(x, weights, impl):
      return api.gated_linear_unit(
          x, weights, activation=jax.nn.silu, implementation=impl
      )

    if use_vmap:
      f = jax.jit(
          jax.vmap(
              functools.partial(single_fn, impl=implementation),
              in_axes=(0, None),
          )
      )
      f_xla = jax.jit(
          jax.vmap(
              functools.partial(single_fn, impl="xla"),
              in_axes=(0, None),
          )
      )
    else:
      f = jax.jit(functools.partial(single_fn, impl=implementation))
      f_xla = jax.jit(functools.partial(single_fn, impl="xla"))

    out = f(lhs, rhs)
    out_golden = f_xla(lhs, rhs)
    chex.assert_trees_all_close(out, out_golden, atol=0.1, rtol=0.02)

    _assert_no_xla_slices_or_joins(
        self,
        f.lower(lhs, rhs),
        use_tuple_weights=use_tuple_weights,
        implementation=implementation,
    )

  @parameterized.product(
      implementation=("mosaic", "triton", None),
      use_tuple_weights=(False, True),
      vmap_mode=("both", "weights_only", "partial_tuple"),
  )
  def test_vmap_weights_no_xla_slices_or_joins(
      self, implementation, use_tuple_weights, vmap_mode
  ):
    if jax.default_backend() != "gpu":
      self.skipTest("GPU kernel HLO test requires GPU backend.")
    if implementation == "triton" and not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")
    if implementation == "mosaic" and not gpu_utils.has_mosaic_gpu_support():
      self.skipTest("Mosaic not supported on this platform.")
    if vmap_mode == "partial_tuple" and not use_tuple_weights:
      self.skipTest("partial_tuple only applies to tuple weights.")

    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(3), 3)
    b, m, k, n = 2, 128, 64, 128
    x_shape = (m, k) if vmap_mode == "weights_only" else (b, m, k)
    lhs = jax.random.normal(rng0, x_shape, dtype=jnp.bfloat16)

    if use_tuple_weights:
      w0 = jax.random.normal(rng1, (b, k, n), dtype=jnp.bfloat16)
      w1_shape = (k, n) if vmap_mode == "partial_tuple" else (b, k, n)
      w1 = jax.random.normal(rng2, w1_shape, dtype=jnp.bfloat16)
      rhs = (w0, w1)
    else:
      rhs = jax.random.normal(rng1, (b, k, 2, n), dtype=jnp.bfloat16)

    if vmap_mode == "both":
      in_axes = (0, 0)
    elif vmap_mode == "weights_only":
      in_axes = (None, 0)
    else:
      in_axes = (0, (0, None))

    def single_fn(x, weights, impl):
      return api.gated_linear_unit(
          x, weights, activation=jax.nn.silu, implementation=impl
      )

    f = jax.jit(
        jax.vmap(functools.partial(single_fn, impl=implementation), in_axes)
    )
    f_xla = jax.jit(jax.vmap(functools.partial(single_fn, impl="xla"), in_axes))

    chex.assert_trees_all_close(
        f(lhs, rhs), f_xla(lhs, rhs), atol=0.1, rtol=0.02
    )
    _assert_no_xla_slices_or_joins(
        self,
        f.lower(lhs, rhs),
        use_tuple_weights=use_tuple_weights,
        implementation=implementation,
    )

  @parameterized.product(
      shape=(
          (1, 4096, 33),
          (17, 33, 70),
          (0, 64, 128),
          (128, 0, 128),
          (128, 64, 0),
      ),
      use_tuple_weights=(False, True),
  )
  def test_unaligned_and_empty_shapes(self, shape, use_tuple_weights):
    m, k, n = shape
    lhs, rhs = _get_input_data(m=m, k=k, n=n)
    if k > 0:
      lhs = (lhs / (k**0.5)).astype(lhs.dtype)
    if use_tuple_weights:
      rhs = (rhs[:, 0, :], rhs[:, 1, :])

    f = jax.jit(
        lambda x, w: api.gated_linear_unit(
            x, w, activation=jax.nn.silu, implementation=None
        )
    )
    f_xla = jax.jit(
        lambda x, w: api.gated_linear_unit(
            x, w, activation=jax.nn.silu, implementation="xla"
        )
    )
    chex.assert_trees_all_close(
        f(lhs, rhs), f_xla(lhs, rhs), atol=0.1, rtol=0.02
    )

    if jax.default_backend() == "gpu" and m > 0 and n > 0 and k > 0:
      _assert_no_xla_slices_or_joins(
          self,
          f.lower(lhs, rhs),
          use_tuple_weights=use_tuple_weights,
          implementation="triton",
      )

  @parameterized.product(
      implementation=("mosaic", "triton"),
      disable_jaxtyping_check=(False, True),
  )
  def test_mismatched_unfused_weights_raises(
      self, implementation, disable_jaxtyping_check
  ):
    if implementation == "triton" and not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")
    if implementation == "mosaic" and not gpu_utils.has_mosaic_gpu_support():
      self.skipTest("Mosaic not supported on this platform.")

    x = jnp.zeros((128, 64), dtype=jnp.bfloat16)
    w_bad = (
        jnp.zeros((64, 128), dtype=jnp.bfloat16),
        jnp.zeros((64, 256), dtype=jnp.bfloat16),
    )
    with jaxtyping.disable_jaxtyping(disable_jaxtyping_check):
      with self.assertRaises(Exception):
        api.gated_linear_unit(x, w_bad, implementation=implementation)


class GatedLinearUnitTritonTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    fn = functools.partial(api.gated_linear_unit, implementation="triton")
    super().__init__(*args, glu_fn=fn)


class GatedLinearUnitXlaTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    fn = functools.partial(api.gated_linear_unit, implementation="xla")
    super().__init__(*args, glu_fn=fn)


class GatedLinearUnitMosaicGpuTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    fn = functools.partial(api.gated_linear_unit, implementation="mosaic")
    super().__init__(*args, glu_fn=fn)


if __name__ == "__main__":
  absltest.main()
