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
"""Unit tests for mHC-lite Pallas TPU forward and backward kernels."""

import dataclasses
import functools
import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mhc_lite import api as mhc_api
from tokamax._src.ops.experimental.mhc_lite import common as mhc_common


def _get_permutation_matrices(streams: int) -> jax.Array:
  """Generates all permutation matrices for `streams` streams."""
  permutations = jnp.array(list(itertools.permutations(range(streams))))
  return jnp.eye(streams, dtype=jnp.float32)[permutations]


def _make_kernel_inputs(
    batch: int = 2,
    sequence: int = 256,
    streams: int = 4,
    embedding: int = 256,
    seed: int = 0,
):
  """Generates synthetic inputs and parameters for testing mHC kernels."""
  key = jax.random.key(seed)
  keys = jax.random.split(key, 12)
  x = jax.random.normal(
      keys[0], (batch, sequence, streams, embedding), dtype=jnp.bfloat16
  )
  norm_scale = jax.random.normal(
      keys[1], (streams * embedding,), dtype=jnp.bfloat16
  )
  w_pre = (
      jax.random.normal(
          keys[2], (streams * embedding, streams), dtype=jnp.bfloat16
      )
      * 0.1
  )
  b_pre = jax.random.normal(keys[3], (streams,), dtype=jnp.bfloat16) * 0.1
  alpha_pre = jnp.array([1.0], dtype=jnp.bfloat16)
  w_post = (
      jax.random.normal(
          keys[5], (streams * embedding, streams), dtype=jnp.bfloat16
      )
      * 0.1
  )
  b_post = jax.random.normal(keys[6], (streams,), dtype=jnp.bfloat16) * 0.1
  alpha_post = jnp.array([1.0], dtype=jnp.bfloat16)
  num_permutations = math.factorial(streams)
  w_res = (
      jax.random.normal(
          keys[8], (streams * embedding, num_permutations), dtype=jnp.bfloat16
      )
      * 0.1
  )
  b_res = (
      jax.random.normal(keys[9], (num_permutations,), dtype=jnp.bfloat16) * 0.1
  )
  alpha_res = jnp.array([1.0], dtype=jnp.bfloat16)
  permutations = _get_permutation_matrices(streams)
  cotangent = (
      jax.random.normal(
          keys[11], (batch, sequence, streams, embedding), dtype=jnp.bfloat16
      )
      * 0.01
  )
  weights = mhc_common.MhcWeights(
      norm_scale=norm_scale,
      w_pre=w_pre,
      b_pre=b_pre,
      alpha_pre=alpha_pre,
      w_post=w_post,
      b_post=b_post,
      alpha_post=alpha_post,
      w_res=w_res,
      b_res=b_res,
      alpha_res=alpha_res,
  )
  return x, weights, permutations, cotangent


def _run_pipeline_api(
    x: jax.Array,
    weights: mhc_common.MhcWeights,
    permutations: jax.Array,
    implementation: mhc_api.Implementation | None = None,
    config: mhc_common.MhcKernelConfig | None = None,
    interpret: bool = True,
) -> jax.Array:
  """Runs the mHC API pipeline (`pre` followed by `post`)."""
  if config is None:
    config = mhc_common.MhcKernelConfig(interpret=interpret)
  else:
    config = dataclasses.replace(config, interpret=interpret)
  layer_input, context = mhc_api.mhc_lite_pre(
      x,
      weights,
      permutations,
      config=config,
      implementation=implementation,
  )
  return mhc_api.mhc_lite_post(layer_input, context, config=config)


def _pipeline_vjp_fn(
    permutations: jax.Array,
    implementation: mhc_api.Implementation | None,
    config: mhc_common.MhcKernelConfig | None,
    interpret: bool,
    x_in: jax.Array,
    weights_in: mhc_common.MhcWeights,
) -> jax.Array:
  """Positional wrapper around `_run_pipeline_api` for `jax.vjp`."""
  return _run_pipeline_api(
      x_in,
      weights_in,
      permutations,
      implementation=implementation,
      config=config,
      interpret=interpret,
  )


class TestMhcKernelsFwd(parameterized.TestCase):
  """Unit tests for mHC-lite forward pass (Pallas kernel vs. JAX reference)."""

  def test_doubly_stochastic(self):
    x, weights, permutations, _ = _make_kernel_inputs(
        batch=1, sequence=256, streams=4, embedding=128
    )
    config = mhc_common.MhcKernelConfig(interpret=True)
    _, context = mhc_api.mhc_lite_pre(
        x,
        weights,
        permutations,
        config=config,
    )
    residual = context.residual.reshape(-1, 4, 4).astype(jnp.float32)
    row_sums = jnp.sum(residual, axis=-1)
    col_sums = jnp.sum(residual, axis=-2)
    np.testing.assert_allclose(
        row_sums, np.ones_like(row_sums), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        col_sums, np.ones_like(col_sums), rtol=1e-2, atol=1e-2
    )

  @parameterized.parameters("mosaic_tpu", "xla")
  def test_forward_parity(self, implementation):
    x, weights, permutations, _ = _make_kernel_inputs(
        batch=2, sequence=256, streams=4, embedding=256
    )
    expected = _run_pipeline_api(
        x,
        weights,
        permutations,
        implementation="xla",
        interpret=True,
    )
    actual = _run_pipeline_api(
        x,
        weights,
        permutations,
        implementation=implementation,
        interpret=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-2)

  def test_unsupported_shape_raises_error(self):
    x, weights, permutations, _ = _make_kernel_inputs(
        batch=1, sequence=16, streams=2, embedding=128
    )
    config = mhc_common.MhcKernelConfig(interpret=True)
    with self.assertRaises(mhc_common.UnsupportedInputError):
      mhc_api.mhc_lite_pre(
          x,
          weights,
          permutations,
          config=config,
      )


class TestMhcKernelsBwd(parameterized.TestCase):
  """Unit tests for mHC-lite backward pass (Pallas kernel vs. JAX reference)."""

  def test_forward_and_backward_vjp_parity(self):
    x, weights, permutations, cotangent = _make_kernel_inputs(
        batch=2, sequence=256, streams=4, embedding=256
    )
    expected_out, expected_vjp_fn = jax.vjp(
        functools.partial(_pipeline_vjp_fn, permutations, "xla", None, True),
        x,
        weights,
    )
    expected_dx, expected_dw = expected_vjp_fn(cotangent)

    actual_out, actual_vjp_fn = jax.vjp(
        functools.partial(
            _pipeline_vjp_fn, permutations, "mosaic_tpu", None, True
        ),
        x,
        weights,
    )
    actual_dx, actual_dw = actual_vjp_fn(cotangent)

    np.testing.assert_allclose(actual_out, expected_out, rtol=5e-2, atol=5e-2)

    actual_grads = (actual_dx,) + tuple(jax.tree.leaves(actual_dw))
    expected_grads = (expected_dx,) + tuple(jax.tree.leaves(expected_dw))
    self.assertEqual(len(actual_grads), len(expected_grads))
    for i, (actual_grad, expected_grad) in enumerate(
        zip(actual_grads, expected_grads, strict=True)
    ):
      np.testing.assert_allclose(
          np.asarray(actual_grad, np.float32),
          np.asarray(expected_grad, np.float32),
          rtol=5e-2,
          atol=1e-1,
          err_msg=f"Gradient leaf {i} mismatch",
      )

  def test_forward_and_backward_vjp_parity_feature_tiled(self):
    x, weights, permutations, cotangent = _make_kernel_inputs(
        batch=2, sequence=256, streams=4, embedding=256
    )
    config = mhc_common.MhcKernelConfig(bwd_feature_block_size=128)
    expected_out, expected_vjp_fn = jax.vjp(
        functools.partial(_pipeline_vjp_fn, permutations, "xla", config, True),
        x,
        weights,
    )
    expected_dx, expected_dw = expected_vjp_fn(cotangent)

    actual_out, actual_vjp_fn = jax.vjp(
        functools.partial(
            _pipeline_vjp_fn, permutations, "mosaic_tpu", config, True
        ),
        x,
        weights,
    )
    actual_dx, actual_dw = actual_vjp_fn(cotangent)

    np.testing.assert_allclose(actual_out, expected_out, rtol=5e-2, atol=5e-2)

    actual_grads = (actual_dx,) + tuple(jax.tree.leaves(actual_dw))
    expected_grads = (expected_dx,) + tuple(jax.tree.leaves(expected_dw))
    self.assertEqual(len(actual_grads), len(expected_grads))
    for i, (actual_grad, expected_grad) in enumerate(
        zip(actual_grads, expected_grads, strict=True)
    ):
      np.testing.assert_allclose(
          np.asarray(actual_grad, np.float32),
          np.asarray(expected_grad, np.float32),
          rtol=5e-2,
          atol=1e-1,
          err_msg=f"Feature-tiled gradient leaf {i} mismatch",
      )


class TestMhcCostEstimates(absltest.TestCase):
  """Unit tests for analytical CostEstimate computations on MhcDims."""

  def setUp(self):
    super().setUp()
    self.dims = mhc_common.MhcDims(
        tokens=256, streams=4, embedding=512, num_permutations=24
    )

  def test_dims_properties(self):
    self.assertEqual(self.dims.flattened_size, 4 * 512)
    self.assertEqual(self.dims.phi_cols, 2 * 4 + 24)
    self.assertEqual(self.dims.pre_slice, slice(0, 4))
    self.assertEqual(self.dims.post_slice, slice(4, 8))
    self.assertEqual(self.dims.res_slice, slice(8, 32))

  def test_cost_estimates_non_zero_and_valid(self):
    costs = [
        ("pre_fwd", self.dims.pre_fwd_cost()),
        ("post_apply_fwd", self.dims.post_apply_fwd_cost()),
        ("pre_bwd", self.dims.pre_bwd_cost()),
        ("post_apply_bwd", self.dims.post_apply_bwd_cost()),
    ]
    for name, cost in costs:
      self.assertIsInstance(cost.flops, int, msg=f"{name} flops must be int")
      self.assertGreater(cost.flops, 0, msg=f"{name} flops must be positive")
      self.assertIsInstance(
          cost.bytes_accessed, int, msg=f"{name} bytes_accessed must be int"
      )
      self.assertGreater(
          cost.bytes_accessed,
          0,
          msg=f"{name} bytes_accessed must be positive",
      )
      self.assertIsInstance(
          cost.transcendentals, int, msg=f"{name} transcendentals must be int"
      )
      self.assertGreaterEqual(
          cost.transcendentals,
          0,
          msg=f"{name} transcendentals must be non-negative",
      )

    self.assertGreater(self.dims.pre_fwd_cost().transcendentals, 0)
    self.assertGreater(self.dims.pre_bwd_cost().transcendentals, 0)
    self.assertEqual(self.dims.post_apply_fwd_cost().transcendentals, 0)
    self.assertEqual(self.dims.post_apply_bwd_cost().transcendentals, 0)


if __name__ == "__main__":
  absltest.main()
