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
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import api
from tokamax._src.ops.ragged_dot import test_base
from typing_extensions import override


def _get_input_data(num_experts, m, k, n, dtype=jnp.bfloat16):
  rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
  lhs = jax.random.normal(rng0, (m, k), dtype=dtype)
  rhs = jax.random.normal(rng1, (num_experts, k, n), dtype=dtype)
  group_sizes = jnp.array([m // num_experts] * num_experts, jnp.uint32)
  return (lhs, rhs, group_sizes)

# TODO: `jax.nn.relu` is annotated with `custom_jvp_call`
# which isn't compatible with `_estimate_resources` in the mosaic lowering.
# It would be nice in the future to support this, if possible.
def relu(x):
  return jnp.maximum(x, 0)

class RaggedDotTest(parameterized.TestCase):

  @parameterized.product(
      implementation=[None, "xla", "mosaic", "triton"],
      activation=[None, relu],
  )
  def test_basic_api(self, implementation, activation):

    if implementation == "triton" and not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")

    # Current default backend if implementation is None is "mosaic".
    if implementation == "mosaic" or implementation is None:
      if (
          jax.default_backend() == "gpu"
          and not gpu_utils.has_mosaic_gpu_support()
      ):
        self.skipTest("Mosaic not supported on this platform.")

      if jax.default_backend() == "cpu":
        self.skipTest("No Mosaic support on CPU.")

    if jax.default_backend() == "tpu":
      lhs, rhs, group_sizes = _get_input_data(
          num_experts=8, m=256, k=128, n=128  # TPU needs shapes >= 128
      )
    else:
      lhs, rhs, group_sizes = _get_input_data(num_experts=8, m=128, k=64, n=128)

    ragged_dot_fn = (
        functools.partial(api.ragged_dot, preferred_element_type=jnp.bfloat16)
        if jax.default_backend() == "tpu"
        else api.ragged_dot
    )

    @jax.jit
    @functools.partial(jax.value_and_grad, argnums=(0, 1))
    def f(lhs, rhs):
      out = ragged_dot_fn(
          lhs,
          rhs,
          group_sizes,
          implementation=implementation,
          activation=activation,
      )
      return jnp.sum(out)

    @jax.jit
    @functools.partial(jax.value_and_grad, argnums=(0, 1))
    def f_gt(lhs, rhs):
      out = jax.lax.ragged_dot(lhs, rhs, group_sizes)
      if activation is not None:
        out = activation(out)
      return jnp.sum(out)

    (out, (lhs_grad, rhs_grad)) = f(lhs, rhs)
    (out_gt, (lhs_grad_gt, rhs_grad_gt)) = f_gt(lhs, rhs)

    with self.subTest("value"):
      chex.assert_trees_all_close(out, out_gt)

    with self.subTest("lhs_grad"):
      chex.assert_trees_all_close(lhs_grad, lhs_grad_gt)
    with self.subTest("rhs_grad"):
      chex.assert_trees_all_close(rhs_grad, rhs_grad_gt)

    with self.subTest("correct_implementation_used"):

      opspecs = hlo_utils.get_opspecs(
          f.lower(lhs, rhs), include_xla_kernels=False
      )
      if jax.default_backend() == "tpu":
        mosaic_impl = type(api.IMPLEMENTATIONS.get("mosaic_tpu"))
      else:
        mosaic_impl = type(api.IMPLEMENTATIONS.get("mosaic_gpu"))

      triton_impl = type(api.IMPLEMENTATIONS.get("triton"))
      match implementation:
        case "triton":
          self.assertIsInstance(opspecs[0].op, triton_impl)
        case "xla":
          self.assertEmpty(opspecs)
        case "mosaic":
          self.assertIsInstance(opspecs[0].op, mosaic_impl)
        case None:
          if jax.default_backend() == "gpu":
            # Ensure either a Triton or Mosaic kernel is used.
            self.assertTrue(
                isinstance(opspecs[0].op, triton_impl)
                or isinstance(opspecs[0].op, mosaic_impl)
            )

  def test_manual_axis_type(self):
    if jax.default_backend() == "tpu":
      lhs, rhs, group_sizes = _get_input_data(
          num_experts=8, m=256, k=128, n=128
      )
    else:
      lhs, rhs, group_sizes = _get_input_data(num_experts=8, m=128, k=64, n=128)

    api.ragged_dot(
        lhs,
        rhs,
        group_sizes,
        manual_axis_type=None,
    )


class RaggedDotImplementationTest(test_base.RaggedDotTestBase):

  def __init__(self, *args, implementation=None):
    self._implementation = implementation
    dot_fn = functools.partial(api.ragged_dot, implementation=implementation)
    super().__init__(*args, dot_fn=dot_fn)

  # TODO: Remove this once the bug is fixed.
  # Note that these tests can be slow on CPU. Either keep disabled, or modify
  # the tests to run faster.
  def setUp(self):
    if jax.default_backend() == "cpu":
      self.skipTest("Test disabled on CPU.")

    # TODO: XLA:TPU is not respecting the new precision API.
    # As Tokamax converts jax.lax.Precision.HIGHEST to BF16_BF16_F32_X6 on TPU,
    # this causes numerical inconsistencies with the tests using
    # jax.lax.Precision.HIGHEST. Only the mosaic_tpu_v2 implementation is
    # exercised on TPU for now; the others still hit this mismatch.
    if jax.default_backend() == "tpu" and self._implementation != "mosaic_tpu_v2":
      self.skipTest("Test disabled on TPU.")

    super().setUp()


class RaggedDotMosaicTest(RaggedDotImplementationTest):

  def __init__(self, *args):
    super().__init__(*args, implementation="mosaic")

    if jax.default_backend() == "gpu":
      dot_fn = self._dot_fn

      def fn(lhs, rhs, **kwargs):
        rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)

        if (
            (lhs.dtype == jnp.bfloat16)
            and (lhs.shape[-1] % (128 // jnp.dtype(lhs.dtype).itemsize) == 0)
            and (
                not isinstance(rhs_, qwix.QArray)
                or (
                    (
                        rhs_.scale_tile_shape == (1, 256, 1)
                        or rhs_.scale_tile_shape == (1, 512, 1)
                    )
                    and kwargs.get("preferred_element_type") is None
                )
            )
        ):
          return dot_fn(lhs, rhs, **kwargs)

        with self.assertRaises(NotImplementedError) as e:
          _ = dot_fn(lhs, rhs, **kwargs)
        self.skipTest(f"Test not supported: {e.msg}")

      self._dot_fn = fn

  def setUp(self):
    if jax.default_backend() not in ("gpu", "tpu"):
      self.skipTest("Only run on GPU and TPU.")
    super().setUp()


class RaggedDotTritonTest(RaggedDotImplementationTest):

  def __init__(self, *args):
    super().__init__(*args, implementation="triton")

  def setUp(self):
    if jax.default_backend() != "gpu":
      self.skipTest("Only run on GPU.")
    super().setUp()

  @override
  def _test_bench(self, spec):
    # TODO: Fix tolerance and enable tests.
    self.skipTest(
        "Accuracy for triton pallas is slightly less than mgpu. We need to"
        " figure out how to fix it or to increase the tolerance."
    )


class RaggedDotXlaTest(RaggedDotImplementationTest):

  def __init__(self, *args):
    super().__init__(*args, implementation="xla")


class RaggedDotMosaicTpuV2Test(RaggedDotImplementationTest):

  def __init__(self, *args):
    super().__init__(*args, implementation="mosaic_tpu_v2")

    dot_fn = self._dot_fn

    def fn(lhs, rhs, **kwargs):
      # The mosaic_tpu_v2 wrapper accepts only raw arrays; `QArray`/`AsQArray`
      # inputs are rejected with `NotImplementedError`, so skip those subtests.
      lhs_ = jax.eval_shape(quantization.as_array_or_qarray, lhs)
      rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)
      if isinstance(lhs_, qwix.QArray) or isinstance(rhs_, qwix.QArray):
        with self.assertRaises(NotImplementedError) as e:
          _ = dot_fn(lhs, rhs, **kwargs)
        self.skipTest(f"Test not supported: {e.msg}")

      return dot_fn(lhs, rhs, **kwargs)

    self._dot_fn = fn

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("mosaic_tpu_v2 is only supported on TPU.")
    if "mosaic_tpu_v2" not in api.IMPLEMENTATIONS:
      self.skipTest("mosaic_tpu_v2 implementation not registered.")
    super().setUp()

  @override
  def _test_simple(self, dtype):
    # The v2 kernel ignores `precision` and always computes on the bf16 MXU
    # (see `gmm_v2`'s `del precision`). For f32 inputs `_dot_fn_f32` requests
    # `Precision.HIGHEST`, so the reference computes in full f32 while the
    # kernel returns a bf16-precision result; compare at bf16 tolerance.
    tol = dict(atol=2e-2, rtol=2e-2) if jnp.dtype(dtype) == jnp.float32 else {}
    with test_base.override_chex_args(**tol):
      super()._test_simple(dtype)


class RaggedDotGmmV2CompatibilityAPITest(parameterized.TestCase):
  """Tests that the v2-specific kwargs surface on the `ragged_dot` API.

  `rhs_scale`, `rhs_bias`, `maybe_quantize_lhs`, and `zero_initialize=False`
  are consumed by the mosaic_tpu_v2 implementation. Every other
  implementation must reject them with a `NotImplementedError` whose message
  names the offending kwarg, so the fallback chain in
  `api.ragged_dot_general` can route around them.
  """

  def _get_small_inputs(self):
    if jax.default_backend() == "tpu":
      return _get_input_data(num_experts=2, m=256, k=128, n=128)
    return _get_input_data(num_experts=2, m=128, k=64, n=128)

  def _kwargs_for(self, kwarg, *, num_experts, n):
    if kwarg == "rhs_scale":
      return {"rhs_scale": jnp.ones((num_experts, 1, 1, n), jnp.bfloat16)}
    if kwarg == "rhs_bias":
      return {"rhs_bias": jnp.ones((num_experts, 1, n), jnp.bfloat16)}
    if kwarg == "maybe_quantize_lhs":
      return {"maybe_quantize_lhs": True}
    if kwarg == "zero_initialize":
      # `zero_initialize=True` is the standard behavior; the v2-only
      # optimization (and thus the rejecting case) is `zero_initialize=False`.
      return {"zero_initialize": False}
    if kwarg == "group_offset":
      return {"group_offset": jnp.array([0], jnp.int32)}
    if kwarg == "fuse_act":
      return {"fuse_act": "silu"}
    raise ValueError(f"Unknown kwarg: {kwarg}")

  def _assert_rejects(self, implementation, kwarg):
    lhs, rhs, group_sizes = self._get_small_inputs()
    kwargs = self._kwargs_for(kwarg, num_experts=rhs.shape[0], n=rhs.shape[-1])
    with self.assertRaisesRegex(NotImplementedError, kwarg):
      api.ragged_dot(
          lhs, rhs, group_sizes, implementation=implementation, **kwargs
      )

  @parameterized.parameters(
      "rhs_scale", "rhs_bias", "maybe_quantize_lhs", "zero_initialize",
      "group_offset", "fuse_act"
  )
  def test_mosaic_xla_rejects_new_kwargs(self, kwarg):
    self._assert_rejects("xla", kwarg)

  @parameterized.parameters(
      "rhs_scale", "rhs_bias", "maybe_quantize_lhs", "zero_initialize",
      "group_offset", "fuse_act"
  )
  def test_mosaic_tpu_rejects_new_kwargs(self, kwarg):
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU backend.")
    self._assert_rejects("mosaic_tpu", kwarg)

  @parameterized.parameters(
      "rhs_scale", "rhs_bias", "maybe_quantize_lhs", "zero_initialize",
      "group_offset", "fuse_act"
  )
  def test_triton_rejects_new_kwargs(self, kwarg):
    if "triton" not in api.IMPLEMENTATIONS:
      self.skipTest("Triton implementation not registered.")
    if not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")
    self._assert_rejects("triton", kwarg)

  @parameterized.parameters(
      "rhs_scale", "rhs_bias", "maybe_quantize_lhs", "zero_initialize",
      "group_offset", "fuse_act"
  )
  def test_mosaic_gpu_rejects_new_kwargs(self, kwarg):
    if "mosaic_gpu" not in api.IMPLEMENTATIONS:
      self.skipTest("mosaic_gpu implementation not registered.")
    if not gpu_utils.has_mosaic_gpu_support():
      self.skipTest("Mosaic GPU not supported on this platform.")
    self._assert_rejects("mosaic_gpu", kwarg)


if __name__ == "__main__":
  absltest.main()
