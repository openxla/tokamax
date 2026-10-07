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
"""API and value_and_grad tests for mHC-lite."""

import functools
import itertools
import math
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mhc_lite import api
from tokamax._src.ops.experimental.mhc_lite import common
from tokamax._src.ops.experimental.mhc_lite import mhc_xla


def _make_inputs(
    batch: int = 2,
    sequence: int = 128,
    streams: int = 4,
    embedding: int = 256,
    seed: int = 0,
):
  keys = jax.random.split(jax.random.key(seed), 12)
  x = jax.random.normal(
      keys[0], (batch, sequence, streams, embedding), dtype=jnp.bfloat16
  )
  flattened = streams * embedding
  num_perms = math.factorial(streams)
  weights = common.MhcWeights(
      norm_scale=jax.random.normal(keys[1], (flattened,), dtype=jnp.bfloat16),
      w_pre=jax.random.normal(keys[2], (flattened, streams), dtype=jnp.bfloat16)
      * 0.1,
      b_pre=jax.random.normal(keys[3], (streams,), dtype=jnp.bfloat16) * 0.1,
      alpha_pre=jnp.array([1.0], dtype=jnp.bfloat16),
      w_post=jax.random.normal(
          keys[4], (flattened, streams), dtype=jnp.bfloat16
      )
      * 0.1,
      b_post=jax.random.normal(keys[5], (streams,), dtype=jnp.bfloat16) * 0.1,
      alpha_post=jnp.array([1.0], dtype=jnp.bfloat16),
      w_res=jax.random.normal(
          keys[6], (flattened, num_perms), dtype=jnp.bfloat16
      )
      * 0.1,
      b_res=jax.random.normal(keys[7], (num_perms,), dtype=jnp.bfloat16) * 0.1,
      alpha_res=jnp.array([1.0], dtype=jnp.bfloat16),
  )
  w_branch = (
      jax.random.normal(keys[8], (embedding, embedding), dtype=jnp.bfloat16)
      * 0.1
  )
  cotangent = (
      jax.random.normal(
          keys[9], (batch, sequence, streams, embedding), dtype=jnp.bfloat16
      )
      * 0.01
  )
  perms = jnp.eye(streams, dtype=jnp.float32)[
      jnp.array(list(itertools.permutations(range(streams))))
  ]
  return x, weights, w_branch, perms, cotangent


def _assert_grads_close(test_case, actual_tree, expected_tree):
  actual_leaves = jax.tree.leaves(actual_tree)
  expected_leaves = jax.tree.leaves(expected_tree)
  test_case.assertLen(actual_leaves, len(expected_leaves))
  for i, (actual, expected) in enumerate(
      zip(actual_leaves, expected_leaves, strict=True)
  ):
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        rtol=5e-2,
        atol=1.5e-1,
        err_msg=f"Gradient leaf {i} mismatch",
    )


def _end_to_end_loss(
    perms: jax.Array,
    cotangent: jax.Array,
    config: common.MhcKernelConfig,
    implementation: Any,
    x_in: jax.Array,
    weights_in: common.MhcWeights,
    w_branch_in: jax.Array,
) -> jax.Array:
  layer_input, ctx = api.mhc_lite_pre(
      x_in,
      weights_in,
      perms,
      config=config,
      implementation=implementation,
  )
  layer_output = jnp.tanh(
      jnp.dot(layer_input, w_branch_in, preferred_element_type=jnp.float32)
  ).astype(x_in.dtype)
  out = api.mhc_lite_post(layer_output, ctx, config=config)
  return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))


def _hc_loss(
    cotangent: jax.Array,
    x_in: jax.Array,
    weights_in: common.HcWeights,
    w_branch_in: jax.Array,
) -> jax.Array:
  layer_input, ctx = api.hc_pre(x_in, weights_in)
  layer_output = jnp.tanh(
      jnp.dot(layer_input, w_branch_in, preferred_element_type=jnp.float32)
  ).astype(x_in.dtype)
  out = api.hc_post(layer_output, ctx)
  return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))


def _post_standalone_loss(
    cotangent: jax.Array,
    config: common.MhcKernelConfig,
    implementation: Any,
    l_out: jax.Array,
    x_in: jax.Array,
    h_post_in: jax.Array,
    res_in: jax.Array,
) -> jax.Array:
  ctx = common.MHCContext(
      x=x_in,
      h_post=h_post_in,
      residual=res_in,
      implementation=implementation,
  )
  out = api.mhc_lite_post(l_out, ctx, config=config)
  return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))


def _pre_standalone_loss(
    perms: jax.Array,
    cotangent: jax.Array,
    w_li: jax.Array,
    w_hp: jax.Array,
    w_res: jax.Array,
    tokens: int,
    config: common.MhcKernelConfig,
    implementation: Any,
    x_in: jax.Array,
    weights_in: common.MhcWeights,
) -> jax.Array:
  layer_input, ctx = api.mhc_lite_pre(
      x_in, weights_in, perms, config=config, implementation=implementation
  )
  return (
      jnp.sum(layer_input.astype(jnp.float32) * w_li)
      + jnp.sum(
          ctx.x.reshape(x_in.shape).astype(jnp.float32)
          * cotangent.astype(jnp.float32)
      )
      + jnp.sum(ctx.h_post.reshape(tokens, 4).astype(jnp.float32) * w_hp)
      + jnp.sum(ctx.residual.reshape(tokens, 16).astype(jnp.float32) * w_res)
  )


def _perm_stop_grad_loss(
    cotangent: jax.Array,
    config: common.MhcKernelConfig,
    x_in: jax.Array,
    weights_in: common.MhcWeights,
    perms_in: jax.Array,
) -> jax.Array:
  layer_input, ctx = api.mhc_lite_pre(x_in, weights_in, perms_in, config=config)
  out = api.mhc_lite_post(layer_input, ctx, config=config)
  return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))


class MhcLiteApiTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("emb256", 256),
      ("emb512", 512),
  )
  def test_value_and_grad_end_to_end(self, embedding):
    x, weights, w_branch, perms, cotangent = _make_inputs(embedding=embedding)
    config = common.MhcKernelConfig(interpret=True)

    xla_grad_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(
                _end_to_end_loss, perms, cotangent, config, "xla"
            ),
            argnums=(0, 1, 2),
        )
    )
    impl_grad_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(_end_to_end_loss, perms, cotangent, config, None),
            argnums=(0, 1, 2),
        )
    )

    expected_loss, expected_grads = xla_grad_fn(x, weights, w_branch)
    actual_loss, actual_grads = impl_grad_fn(x, weights, w_branch)

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=2e-2, atol=2e-2)
    _assert_grads_close(self, actual_grads, expected_grads)

  @parameterized.named_parameters(
      ("1d_pre_1d_post", (4,), (4,)),
      ("1d_pre_2d_post", (4,), (4, 1)),
      ("2d_pre_2d_post", (4, 256), (4, 1)),
  )
  def test_hc_value_and_grad_end_to_end(self, pre_shape, post_shape):
    x, _, w_branch, _, cotangent = _make_inputs(embedding=256)
    keys = jax.random.split(jax.random.key(42), 3)
    hc_weights = common.HcWeights(
        w_pre=jax.random.normal(keys[0], pre_shape, dtype=jnp.bfloat16) * 0.25,
        w_post=jax.random.normal(keys[1], post_shape, dtype=jnp.bfloat16)
        * 0.25,
        w_res=jnp.eye(4, dtype=jnp.bfloat16),
    )

    loss_and_grad = jax.jit(
        jax.value_and_grad(
            functools.partial(_hc_loss, cotangent), argnums=(0, 1, 2)
        )
    )
    loss, (dx, dweights, dw_branch) = loss_and_grad(x, hc_weights, w_branch)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertEqual(dx.shape, x.shape)
    self.assertEqual(dweights.w_pre.shape, pre_shape)
    self.assertEqual(dweights.w_post.shape, post_shape)
    self.assertEqual(dweights.w_res.shape, (4, 4))
    self.assertEqual(dw_branch.shape, w_branch.shape)

  @parameterized.named_parameters(
      ("fast_2d_mxu_emb256", 256, 1024),
      ("fast_2d_mxu_emb512", 512, 1024),
      ("feature_tiled_emb256", 256, 128),
  )
  def test_post_value_and_grad_standalone(
      self, embedding, bwd_feature_block_size
  ):
    x, weights, _, perms, cotangent = _make_inputs(embedding=embedding)
    config = common.MhcKernelConfig(
        bwd_feature_block_size=bwd_feature_block_size, interpret=True
    )
    layer_output, ctx_xla = mhc_xla.pre_xla(x, weights, perms, config=config)

    post_loss_xla = jax.jit(
        jax.value_and_grad(
            functools.partial(_post_standalone_loss, cotangent, config, "xla"),
            argnums=(0, 1, 2, 3),
        )
    )
    post_loss_mosaic = jax.jit(
        jax.value_and_grad(
            functools.partial(
                _post_standalone_loss, cotangent, config, "mosaic_tpu"
            ),
            argnums=(0, 1, 2, 3),
        )
    )

    res_bf16 = ctx_xla.residual.astype(jnp.bfloat16)
    expected_val, expected_grads = post_loss_xla(
        layer_output, x, ctx_xla.h_post, res_bf16
    )
    actual_val, actual_grads = post_loss_mosaic(
        layer_output, x, ctx_xla.h_post, res_bf16
    )

    np.testing.assert_allclose(actual_val, expected_val, rtol=1e-2, atol=1e-2)
    _assert_grads_close(self, actual_grads, expected_grads)

  @parameterized.named_parameters(
      ("single_block", 1),
      ("software_pipelined", 2),
  )
  def test_pre_value_and_grad_standalone(self, fwd_pipeline_blocks):
    x, weights, _, perms, cotangent = _make_inputs(embedding=256)
    config = common.MhcKernelConfig(
        fwd_pipeline_blocks=fwd_pipeline_blocks, interpret=True
    )
    tokens = x.shape[0] * x.shape[1]
    keys = jax.random.split(jax.random.key(99), 3)
    w_li = (
        jax.random.normal(keys[0], x.shape[:2] + (256,), dtype=jnp.float32)
        * 0.01
    )
    w_hp = jax.random.normal(keys[1], (tokens, 4), dtype=jnp.float32) * 0.01
    w_res = jax.random.normal(keys[2], (tokens, 16), dtype=jnp.float32) * 0.01

    pre_xla_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(
                _pre_standalone_loss,
                perms,
                cotangent,
                w_li,
                w_hp,
                w_res,
                tokens,
                config,
                "xla",
            ),
            argnums=(0, 1),
        )
    )
    pre_mosaic_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(
                _pre_standalone_loss,
                perms,
                cotangent,
                w_li,
                w_hp,
                w_res,
                tokens,
                config,
                "mosaic_tpu",
            ),
            argnums=(0, 1),
        )
    )

    expected_val, expected_grads = pre_xla_fn(x, weights)
    actual_val, actual_grads = pre_mosaic_fn(x, weights)
    np.testing.assert_allclose(actual_val, expected_val, rtol=2e-2, atol=2e-2)
    _assert_grads_close(self, actual_grads, expected_grads)

  def test_permutations_stop_gradient(self):
    x, weights, _, perms, cotangent = _make_inputs(embedding=256)
    config = common.MhcKernelConfig(interpret=True)

    loss_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(_perm_stop_grad_loss, cotangent, config),
            argnums=(0, 1, 2),
        )
    )

    _, (_, _, d_perms) = loss_fn(x, weights, perms)
    np.testing.assert_array_equal(d_perms, np.zeros_like(perms))

  def test_api_validation_errors(self):
    x, weights, _, perms, _ = _make_inputs(embedding=256)
    config = common.MhcKernelConfig(interpret=True)
    invalid_impl: Any = "invalid"
    invalid_tuple_impl: Any = ("invalid",)

    with self.assertRaises(ValueError):
      api.mhc_lite_pre(
          x, weights, perms, config=config, implementation=invalid_impl
      )

    with self.assertRaises(ValueError):
      api.mhc_lite_pre(
          x, weights, perms, config=config, implementation=invalid_tuple_impl
      )

    layer_input, ctx = api.mhc_lite_pre(
        x, weights, perms, config=config, implementation="xla"
    )
    bad_ctx = common.MHCContext(
        x=ctx.x,
        h_post=ctx.h_post,
        residual=ctx.residual,
        implementation="invalid",
    )
    with self.assertRaises(ValueError):
      api.mhc_lite_post(layer_input, bad_ctx, config=config)
    with self.assertRaises(ValueError):
      api.hc_post(layer_input, bad_ctx)

    with self.assertRaises(common.UnsupportedInputError):
      api.mhc_lite_pre(x.astype(jnp.float32), weights, perms, config=config)

    with self.assertRaises(common.UnsupportedInputError):
      api.mhc_lite_pre(
          x,
          weights,
          perms,
          config=common.MhcKernelConfig(block_size=10, interpret=True),
      )


if __name__ == "__main__":
  absltest.main()
