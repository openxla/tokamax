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
from collections.abc import Callable
import functools
from typing import Any, override
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import export
import jax.numpy as jnp
import qwix
from tokamax._src import config as config_lib
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import api
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import test_base


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


class _MockDeviceRestrictedOp(base.RaggedDot):

  def __call__(self, *args, **kwargs):
    del args, kwargs
    if not self.bypass_device_check:
      raise NotImplementedError("device check failed")
    return jnp.zeros((128, 128))


class RaggedDotTest(parameterized.TestCase):

  @parameterized.product(
      implementation=[None, "xla", "mosaic", "triton", "mosaic_tpu_v2"],
      activation=[None, relu],
  )
  def test_basic_api(self, implementation, activation):

    if implementation == "triton" and not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")

    if implementation == "triton" and activation is not None and gpu_utils.is_sm90():
      self.skipTest("Triton ragged_dot with activation VJP crashes on SM90.")

    if implementation == "mosaic_tpu_v2":
      if (
          jax.default_backend() != "tpu"
          or "mosaic_tpu_v2" not in api.IMPLEMENTATIONS
      ):
        self.skipTest("mosaic_tpu_v2 is only supported on TPU.")
      if activation is not None:
        self.skipTest("mosaic_tpu_v2 does not support `activation`.")

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

    out, (lhs_grad, rhs_grad) = f(lhs, rhs)
    out_gt, (lhs_grad_gt, rhs_grad_gt) = f_gt(lhs, rhs)

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
        case "mosaic_tpu_v2":
          self.assertIsInstance(
              opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu_v2"])
          )
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

  def test_device_check_bypass_auto_vs_explicit(self):
    if jax.default_backend() != "cpu":
      self.skipTest("Device check bypass test specifically targets CPU.")

    lhs, rhs, group_sizes = _get_input_data(num_experts=8, m=128, k=64, n=128)

    # Case 1: implementation is None (auto-selection).
    # On CPU, device restriction is enforced on hardware-specific backends,
    # so auto-selection safely falls back to xla and succeeds.
    out_auto = api.ragged_dot(lhs, rhs, group_sizes, implementation=None)
    self.assertEqual(out_auto.shape, (128, 128))

    # When explicitly forcing bypass_device_check=False on a TPU op on CPU,
    # device validation raises NotImplementedError.
    if "mosaic_tpu_v2" in api.IMPLEMENTATIONS:
      with self.assertRaisesRegex(
          NotImplementedError, "Not supported on cpu"
      ):
        api.ragged_dot(
            lhs,
            rhs,
            group_sizes,
            implementation="mosaic_tpu_v2",
            bypass_device_check=False,
        )

    # Case 2: implementation is explicitly passed in.
    # Device validation is automatically bypassed when implementation is
    # manually specified.
    mock_impl = _MockDeviceRestrictedOp()

    # Explicit implementation bypasses device validation.
    out_manual = api.ragged_dot(
        lhs, rhs, group_sizes, implementation=[mock_impl]
    )
    self.assertEqual(out_manual.shape, (128, 128))

    # Explicitly disabling bypass enforces device check and raises.
    with self.assertRaisesRegex(NotImplementedError, "device check failed"):
      api.ragged_dot(
          lhs,
          rhs,
          group_sizes,
          implementation=[mock_impl],
          bypass_device_check=False,
      )

  def test_device_check_bypass_sequence_fallback(self):
    if jax.default_backend() != "cpu":
      self.skipTest("Device check bypass test specifically targets CPU.")

    lhs, rhs, group_sizes = _get_input_data(num_experts=8, m=128, k=64, n=128)

    mock_fail = _MockDeviceRestrictedOp()

    # When passing a sequence of multiple candidates, device validation is
    # enforced so failing candidate falls back to the next candidate ("xla").
    out_fallback = api.ragged_dot(
        lhs, rhs, group_sizes, implementation=[mock_fail, "xla"]
    )
    self.assertEqual(out_fallback.shape, (128, 128))

  def test_cross_compile_export_on_cpu(self):
    if jax.default_backend() != "cpu":
      self.skipTest("Cross-compile export test specifically targets CPU.")

    lhs, rhs, group_sizes = _get_input_data(num_experts=8, m=128, k=64, n=128)

    if "mosaic_tpu_v2" in api.IMPLEMENTATIONS:
      # Explicit implementation automatically bypasses device checks on CPU.
      def dot_fn(lhs, rhs, group_sizes):
        return api.ragged_dot(
            lhs,
            rhs,
            group_sizes,
            implementation="mosaic_tpu_v2",
        )

      # Lowering on CPU emits TPU custom calls.
      lowered = jax.jit(dot_fn).lower(lhs, rhs, group_sizes)
      self.assertIn("custom_call", lowered.as_text("stablehlo"))

      # StableHLO export on CPU succeeds with custom call check disabled.
      exported = export.export(
          dot_fn,  # pyrefly: ignore[bad-argument-type]
          disabled_checks=[
              export.DisabledSafetyCheck.custom_call("tpu_custom_call")
          ],
      )(
          jax.ShapeDtypeStruct(lhs.shape, lhs.dtype),
          jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
          jax.ShapeDtypeStruct(group_sizes.shape, group_sizes.dtype),
      )
      self.assertIsNotNone(exported)

      # Cross-compile config globally bypasses device checks.
      with config_lib.cross_compile(True):
        lowered_cfg = jax.jit(dot_fn).lower(lhs, rhs, group_sizes)
        self.assertIn("custom_call", lowered_cfg.as_text("stablehlo"))


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
    # jax.lax.Precision.HIGHEST.
    if (
        jax.default_backend() == "tpu"
        and self._implementation != "mosaic_tpu_v2"
    ):
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
    dot_fn = self._dot_fn

    def fn(lhs, rhs, *, activation=None, **kwargs):
      # Triton kernels only support known `jax.nn` activations.
      if activation is test_base.relu:
        activation = jax.nn.relu
      return dot_fn(lhs, rhs, activation=activation, **kwargs)

    self._dot_fn = fn

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

  def _test_quantized(self, *args, **kwargs):
    if jax.default_backend() == "gpu":
      self.skipTest("Quantized ragged_dot not supported on GPU for XLA.")
    super()._test_quantized(*args, **kwargs)

  def _test_bench(self, spec):
    if jax.default_backend() == "gpu" and (
        isinstance(spec.get("lhs"), qwix.QArray)
        or isinstance(spec.get("rhs"), qwix.QArray)
    ):
      self.skipTest("Quantized ragged_dot not supported on GPU for XLA.")
    super()._test_bench(spec)


class RaggedDotMosaicTpuV2Test(RaggedDotImplementationTest):

  def __init__(self, *args):
    super().__init__(*args, implementation="mosaic_tpu_v2")
    dot_fn = self._dot_fn

    def fn(lhs, rhs, **kwargs):
      # v2 does not support the element-wise `activation`.
      if kwargs.get("activation") is not None:
        self.skipTest("v2 does not support `activation`.")
      # v2 accepts only raw arrays; `QArray`/`AsQArray` inputs are rejected.
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
    # v2 ignores `precision` and computes on the bf16 MXU, so f32 inputs
    # requested at `Precision.HIGHEST` are compared at bf16 tolerance.
    tol = dict(atol=2e-2, rtol=2e-2) if jnp.dtype(dtype) == jnp.float32 else {}
    with test_base.override_chex_args(**tol):
      super()._test_simple(dtype)

  @override
  def _test_vjp(self, num_groups, m, k, n, activation=None):
    # There is a Mosaic compile error most likely comes from the f32 dout /
    # f32 output combination. Since we mainly use bf16 for inputs and outputs,
    # this test is skipped. The bf16 path is tested in `RaggedDotTest`.
    self.skipTest("gmm_v2 dlhs kernel fails to compile in Mosaic.")

  @override
  def _test_bench(self, spec):
    device_kind = jax.devices()[0].device_kind
    if self._testMethodName.endswith("mixtral_8x7b") and device_kind in (
        "TPU v5",
        "TPU v5p",
    ):
      self.skipTest("gmm_v2 tiling exceeds scoped VMEM on TPU v5.")
    super()._test_bench(spec)


class RaggedDotMosaicTpuV2KwargsTest(parameterized.TestCase):
  """Tests that GMM v2 kwargs are forwarded by `api.ragged_dot` to v2."""

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("mosaic_tpu_v2 is only supported on TPU.")
    if "mosaic_tpu_v2" not in api.IMPLEMENTATIONS:
      self.skipTest("mosaic_tpu_v2 implementation not registered.")
    super().setUp()

  @parameterized.named_parameters(
      ("rhs_bias", "rhs_bias"),
      ("rhs_scale", "rhs_scale"),
      ("fuse_gateup_activation", "fuse_gateup_activation"),
      ("group_offset", "group_offset"),
      ("zero_initialize", "zero_initialize"),
  )
  def test_kwarg_forwarded(self, kwarg):
    num_groups, m, k, n = 4, 256, 256, 256
    lhs, rhs, group_sizes = _get_input_data(num_groups, m, k, n)
    kwargs = {}
    if kwarg == "rhs_bias":
      kwargs["rhs_bias"] = jnp.ones((num_groups, 1, n), jnp.bfloat16)
    elif kwarg == "rhs_scale":
      rhs = rhs.astype(jnp.float8_e4m3fn)
      kwargs["rhs_scale"] = jnp.full((num_groups, 1, 1, n), 0.5, jnp.float32)
    elif kwarg == "fuse_gateup_activation":
      kwargs["fuse_gateup_activation"] = "silu"
    elif kwarg == "group_offset":
      group_offset = 1
      # `rhs` holds only the local groups; `group_sizes` stays global.
      rhs = rhs[group_offset:]
      kwargs["group_offset"] = jnp.array([group_offset], jnp.int32)
    elif kwarg == "zero_initialize":
      kwargs["zero_initialize"] = False

    actual = api.ragged_dot(
        lhs, rhs, group_sizes, implementation="mosaic_tpu_v2", **kwargs
    )
    expected = api.IMPLEMENTATIONS["mosaic_tpu_v2"](
        lhs, rhs, group_sizes=group_sizes, **kwargs
    )
    chex.assert_trees_all_equal(actual, expected)


_V2_KWARGS: dict[str, Callable[[int, int], Any]] = {
    "group_offset": lambda g, n: jnp.array([0], jnp.int32),
    "rhs_scale": lambda g, n: jnp.ones((g, 1, 1, n), jnp.float32),
    "rhs_bias": lambda g, n: jnp.ones((g, 1, n), jnp.bfloat16),
    "maybe_quantize_lhs": lambda g, n: True,
    "lhs_scale": lambda g, n: jnp.ones((1, 1), jnp.float32),
    "zero_initialize": lambda g, n: False,
    "fuse_gateup_activation": lambda g, n: "silu",
    "lhs_quantization_dtype": lambda g, n: jnp.float8_e4m3fn,
    "rhs_quantization_dtype": lambda g, n: jnp.float8_e4m3fn,
}


class RaggedDotGmmV2CompatibilityAPITest(parameterized.TestCase):
  """Tests that non-v2 implementations reject the GMM v2 kwargs."""

  def _assert_rejects(self, implementation, kwarg):
    lhs, rhs, group_sizes = _get_input_data(num_experts=2, m=256, k=128, n=128)
    value = _V2_KWARGS[kwarg](rhs.shape[0], rhs.shape[-1])
    with self.assertRaisesRegex(NotImplementedError, kwarg):
      api.ragged_dot(
          lhs,
          rhs,
          group_sizes,
          implementation=implementation,
          **{kwarg: value},
      )

  def test_xla_rejects_group_offset(self):
    self._assert_rejects("xla", "group_offset")

  @parameterized.parameters(*_V2_KWARGS)
  def test_mosaic_tpu_rejects_v2_kwargs(self, kwarg):
    if (
        jax.default_backend() != "tpu"
        or "mosaic_tpu" not in api.IMPLEMENTATIONS
    ):
      self.skipTest("Requires TPU and mosaic_tpu.")
    self._assert_rejects("mosaic_tpu", kwarg)

  @parameterized.parameters(*_V2_KWARGS)
  def test_triton_rejects_v2_kwargs(self, kwarg):
    if (
        "triton" not in api.IMPLEMENTATIONS
        or not gpu_utils.has_triton_support()
    ):
      self.skipTest("Triton not supported on this platform.")
    self._assert_rejects("triton", kwarg)

  @parameterized.parameters(*_V2_KWARGS)
  def test_mosaic_gpu_rejects_v2_kwargs(self, kwarg):
    if (
        "mosaic_gpu" not in api.IMPLEMENTATIONS
        or not gpu_utils.has_mosaic_gpu_support()
    ):
      self.skipTest("Mosaic GPU not supported on this platform.")
    self._assert_rejects("mosaic_gpu", kwarg)


if __name__ == "__main__":
  absltest.main()
