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
    if implementation == "mosaic":
      if (
          "mosaic" not in api.IMPLEMENTATIONS
          or not gpu_utils.has_mosaic_gpu_support()
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
        impl = api.IMPLEMENTATIONS[implementation]
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


if __name__ == "__main__":
  absltest.main()
