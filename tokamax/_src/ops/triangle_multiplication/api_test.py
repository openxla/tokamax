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
from collections.abc import Sequence
import typing
from typing import Final
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src import jaxtyping
from tokamax._src import numerics
from tokamax._src.ops.triangle_multiplication import api

_IMPLEMENTATIONS: Final[tuple[str | None, ...]] = typing.get_args(
    api.Implementation.__value__
) + (None,)


def _get_params(n, c, h, d, dtype):
  params = dict(
      x=jax.ShapeDtypeStruct((n, n, c), dtype=dtype),
      mask=jax.ShapeDtypeStruct((n, n), dtype=jnp.bool),
      projection_in_weights=jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      gate_in_weights=jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      projection_out_weights=jax.ShapeDtypeStruct((h, d), dtype=dtype),
      gate_out_weights=jax.ShapeDtypeStruct((c, d), dtype=dtype),
      layernorm_in_scale=jax.ShapeDtypeStruct((c,), dtype=dtype),
      layernorm_in_offset=jax.ShapeDtypeStruct((c,), dtype=dtype),
      layernorm_out_scale=jax.ShapeDtypeStruct((h,), dtype=dtype),
      layernorm_out_offset=jax.ShapeDtypeStruct((h,), dtype=dtype),
  )
  return numerics.random_initialize(params)


class TriangleMultiplicationTest(parameterized.TestCase):

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      implementation=_IMPLEMENTATIONS,
  )
  def test_triangle_multiplication(self, triangle_type, implementation):
    if implementation == "triton":
      if (
          "triton" not in api.IMPLEMENTATIONS
          or not gpu_utils.has_triton_support()
      ):
        self.skipTest("Triton not supported on this platform.")
    mosaic_key = f"mosaic_{jax.default_backend()}"
    if implementation == "mosaic":
      if (
          mosaic_key not in api.IMPLEMENTATIONS
          or not api.IMPLEMENTATIONS[mosaic_key].supported_on(jax.devices()[0])  # pyrefly: ignore[missing-attribute]
      ):
        self.skipTest("Mosaic not supported on this platform.")

    n = 128
    d = 64
    params = _get_params(n=n, c=32, h=32, d=d, dtype=jnp.bfloat16)
    mask = params.pop("mask")
    dout = (
        jax.random.normal(jax.random.key(0), (n, n, d), dtype=jnp.bfloat16) / n
    )

    @jax.jit
    def f_ref(params):
      return api.triangle_multiplication(
          **params,
          mask=mask,
          triangle_type=triangle_type,
          implementation="xla",
      )

    @jax.jit
    def f(params):
      return api.triangle_multiplication(
          **params,
          mask=mask,
          triangle_type=triangle_type,
          implementation=implementation,
      )

    out_ref, vjp_ref = jax.vjp(f_ref, params)
    out, vjp_fn = jax.vjp(f, params)
    chex.assert_trees_all_close(out, out_ref, atol=0.15, rtol=0.05)

    with self.subTest("grad"):
      (grads_ref,) = jax.jit(vjp_ref)(dout)
      (grads,) = jax.jit(vjp_fn)(dout)
      chex.assert_trees_all_close(grads, grads_ref, atol=0.5, rtol=0.05)

    with self.subTest("correct_implementation_used"):
      args = hlo_utils.get_bound_args(f, params)
      self.assertLen(args, 1)
      if implementation is not None:
        impl_key = mosaic_key if implementation == "mosaic" else implementation
        impl = api.IMPLEMENTATIONS[impl_key]
      elif (
          mosaic_key in api.IMPLEMENTATIONS
          and api.IMPLEMENTATIONS[mosaic_key].supported_on(jax.devices()[0])  # pyrefly: ignore[missing-attribute]
      ):
        impl = api.IMPLEMENTATIONS[mosaic_key]
      elif "triton" in api.IMPLEMENTATIONS and gpu_utils.has_triton_support():
        impl = api.IMPLEMENTATIONS["triton"]
      else:
        impl = api.IMPLEMENTATIONS["xla"]
      self.assertIsInstance(args[0].op, impl.__class__)
      if getattr(impl, "vjp", None) is not None:
        vjp_args = hlo_utils.get_bound_args(vjp_fn, dout)
        self.assertLen(vjp_args, 1)
        self.assertIsInstance(vjp_args[0].op, impl.vjp.__class__)  # pyrefly: ignore[missing-attribute]

  def test_unsupported_implementation(self):
    params = _get_params(n=8, c=16, h=32, d=64, dtype=jnp.float32)
    with self.assertRaises(NotImplementedError):
      with jaxtyping.disable_jaxtyping():
        api.triangle_multiplication(
            **params,
            triangle_type="incoming",
            implementation="unsupported",  # pyrefly: ignore[bad-argument-type]
        )

  def test_default_fallback_on_cpu_device(self):
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
      self.skipTest("No CPU device available.")
    with jax.default_device(cpu_devices[0]):
      params = jax.device_put(
          _get_params(n=8, c=16, h=32, d=64, dtype=jnp.bfloat16), cpu_devices[0]
      )
      out_jit = jax.jit(
          lambda p: api.triangle_multiplication(
              **p, triangle_type="outgoing", implementation=None
          )
      )(params)
      expected_jit = jax.jit(
          lambda p: api.triangle_multiplication(
              **p, triangle_type="outgoing", implementation="xla"
          )
      )(params)
      chex.assert_trees_all_equal(out_jit, expected_jit)

    out_eager = api.triangle_multiplication(
        **params, triangle_type="outgoing", implementation=None
    )
    chex.assert_trees_all_close(out_eager, expected_jit)

  def test_empty_implementation_raises_value_error(self):
    params = _get_params(n=8, c=16, h=32, d=64, dtype=jnp.float32)
    with self.assertRaises(ValueError):
      api.triangle_multiplication(
          **params, triangle_type="incoming", implementation=()
      )

  def test_explicit_mosaic_on_cpu_device_raises(self):
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
      self.skipTest("No CPU device available.")
    with jax.default_device(cpu_devices[0]):
      params = jax.device_put(
          _get_params(n=8, c=16, h=32, d=64, dtype=jnp.bfloat16), cpu_devices[0]
      )
      expected_exc = (
          NotImplementedError
          if "mosaic_tpu" in api.IMPLEMENTATIONS
          else ValueError
      )

      def _assert_raises(
          impl: api.Implementation | Sequence[api.Implementation],
      ) -> None:
        with self.assertRaises(expected_exc):
          api.triangle_multiplication(
              **params, triangle_type="outgoing", implementation=impl
          )
        with self.assertRaises(expected_exc):
          jax.jit(
              lambda p: api.triangle_multiplication(
                  **p, triangle_type="outgoing", implementation=impl
              )
          )(params)

      _assert_raises("mosaic")
      _assert_raises(("mosaic",))

  def test_sequence_fallback_and_exception_group_on_cpu_device(self):
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
      self.skipTest("No CPU device available.")
    with jax.default_device(cpu_devices[0]):
      params = jax.device_put(
          _get_params(n=8, c=16, h=32, d=64, dtype=jnp.bfloat16), cpu_devices[0]
      )
      expected = api.triangle_multiplication(
          **params, triangle_type="outgoing", implementation="xla"
      )
      if "mosaic_tpu" in api.IMPLEMENTATIONS:
        out_eager = api.triangle_multiplication(
            **params, triangle_type="outgoing", implementation=("mosaic", "xla")
        )
        out_jit = jax.jit(
            lambda p: api.triangle_multiplication(
                **p, triangle_type="outgoing", implementation=("mosaic", "xla")
            )
        )(params)
        chex.assert_trees_all_equal(out_eager, expected)
        chex.assert_trees_all_equal(out_jit, expected)
        with self.assertRaises(ExceptionGroup):
          api.triangle_multiplication(
              **params,
              triangle_type="outgoing",
              implementation=("mosaic", "mosaic"),
          )
      else:
        with self.assertRaises(ValueError):
          api.triangle_multiplication(
              **params,
              triangle_type="outgoing",
              implementation=("mosaic", "xla"),
          )
        with self.assertRaises(ValueError):
          jax.jit(
              lambda p: api.triangle_multiplication(
                  **p,
                  triangle_type="outgoing",
                  implementation=("mosaic", "xla"),
              )
          )(params)

      def _failing_impl(**kwargs):
        del kwargs
        raise NotImplementedError("Unsupported on CPU.")

      with jaxtyping.disable_jaxtyping():
        with self.assertRaises(NotImplementedError):
          api.triangle_multiplication(
              **params,
              triangle_type="outgoing",
              implementation=(_failing_impl,),  # pyrefly: ignore[bad-argument-type]
          )
        out_custom = api.triangle_multiplication(
            **params,
            triangle_type="outgoing",
            implementation=(_failing_impl, "xla"),  # pyrefly: ignore[bad-argument-type]
        )
        chex.assert_trees_all_equal(out_custom, expected)
        with self.assertRaises(ExceptionGroup):
          api.triangle_multiplication(
              **params,
              triangle_type="outgoing",
              implementation=(_failing_impl, _failing_impl),  # pyrefly: ignore[bad-argument-type]
          )

    if (
        jax.default_backend() == "gpu"
        and "NVIDIA" in jax.devices()[0].device_kind
        and "mosaic_gpu" in api.IMPLEMENTATIONS
    ):
      with self.assertRaises(NotImplementedError):
        api.triangle_multiplication(
            **params, triangle_type="outgoing", implementation=("mosaic",)
        )
      out_gpu_fallback = api.triangle_multiplication(
          **params, triangle_type="outgoing", implementation=("mosaic", "xla")
      )
      chex.assert_trees_all_equal(out_gpu_fallback, expected)
      with self.assertRaises(ExceptionGroup):
        api.triangle_multiplication(
            **params,
            triangle_type="outgoing",
            implementation=("mosaic", "mosaic"),
        )


if __name__ == "__main__":
  absltest.main()
