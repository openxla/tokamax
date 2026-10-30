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


def _interpret() -> bool:
  return jax.default_backend() != "tpu"


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


def _post_and_pre_standalone_loss(
    perms: jax.Array,
    cotangent: jax.Array,
    w_li: jax.Array,
    w_hp: jax.Array,
    w_res: jax.Array,
    tokens: int,
    config: common.MhcKernelConfig,
    implementation: Any,
    l_out: jax.Array,
    x_in: jax.Array,
    h_post_in: jax.Array,
    res_in: jax.Array,
    next_weights_in: common.MhcWeights,
) -> jax.Array:
  ctx = common.MHCContext(
      x=x_in,
      h_post=h_post_in,
      residual=res_in,
      implementation=implementation,
  )
  next_li, next_ctx = api.mhc_lite_post_and_pre(
      l_out,
      ctx,
      next_weights_in,
      perms,
      config=config,
      implementation=implementation,
  )
  return (
      jnp.sum(next_li.astype(jnp.float32) * w_li)
      + jnp.sum(
          next_ctx.x.reshape(x_in.shape).astype(jnp.float32)
          * cotangent.astype(jnp.float32)
      )
      + jnp.sum(next_ctx.h_post.reshape(tokens, 4).astype(jnp.float32) * w_hp)
      + jnp.sum(
          next_ctx.residual.reshape(tokens, 16).astype(jnp.float32) * w_res
      )
  )


def _two_stage_chain_loss(
    perms: jax.Array,
    cotangent: jax.Array,
    config: common.MhcKernelConfig,
    implementation: Any,
    use_fused_boundary: bool,
    x_in: jax.Array,
    weights_1: common.MhcWeights,
    w_branch_1: jax.Array,
    weights_2: common.MhcWeights,
    w_branch_2: jax.Array,
) -> jax.Array:
  layer_input_1, ctx_1 = api.mhc_lite_pre(
      x_in, weights_1, perms, config=config, implementation=implementation
  )
  layer_output_1 = jnp.tanh(
      jnp.dot(layer_input_1, w_branch_1, preferred_element_type=jnp.float32)
  ).astype(x_in.dtype)
  if use_fused_boundary:
    layer_input_2, ctx_2 = api.mhc_lite_post_and_pre(
        layer_output_1,
        ctx_1,
        weights_2,
        perms,
        config=config,
        implementation=implementation,
    )
  else:
    x_mid = api.mhc_lite_post(layer_output_1, ctx_1, config=config)
    layer_input_2, ctx_2 = api.mhc_lite_pre(
        x_mid, weights_2, perms, config=config, implementation=implementation
    )
  layer_output_2 = jnp.tanh(
      jnp.dot(layer_input_2, w_branch_2, preferred_element_type=jnp.float32)
  ).astype(x_in.dtype)
  out = api.mhc_lite_post(layer_output_2, ctx_2, config=config)
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
    config = common.MhcKernelConfig(interpret=_interpret())

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
      ("2d_pre_full_2d_post", (4, 256), (4, 256)),
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

    tokens = x.shape[0] * x.shape[1]
    x_2d = x.reshape(tokens, 4 * 256)
    cotangent_2d = cotangent.reshape(tokens, 4 * 256)
    loss_and_grad_2d = jax.jit(
        jax.value_and_grad(
            functools.partial(_hc_loss, cotangent_2d), argnums=(0, 1, 2)
        )
    )
    loss_2d, (dx_2d, dweights_2d, dw_branch_2d) = loss_and_grad_2d(
        x_2d, hc_weights, w_branch
    )
    self.assertEqual(dx_2d.shape, x_2d.shape)
    np.testing.assert_allclose(loss_2d, loss, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        dx_2d, dx.reshape(x_2d.shape), rtol=1e-3, atol=1e-3
    )
    _assert_grads_close(
        self, (dweights_2d, dw_branch_2d), (dweights, dw_branch)
    )

    li_4d, ctx_4d = api.hc_pre(x, hc_weights)
    next_li_4d, next_ctx_4d = api.hc_post_and_pre(li_4d, ctx_4d, hc_weights)
    li_2d, ctx_2d = api.hc_pre(x_2d, hc_weights)
    next_li_2d, next_ctx_2d = api.hc_post_and_pre(li_2d, ctx_2d, hc_weights)
    self.assertEqual(next_li_2d.shape, (tokens, 256))
    self.assertEqual(next_ctx_2d.x.shape, (tokens, 4 * 256))
    np.testing.assert_allclose(
        next_li_2d, next_li_4d.reshape(tokens, 256), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        next_ctx_2d.x,
        next_ctx_4d.x.reshape(tokens, 4 * 256),
        rtol=1e-3,
        atol=1e-3,
    )

  @parameterized.named_parameters(
      ("bf16_x_f32_branch_and_weights", jnp.bfloat16, jnp.float32),
      ("f16_x_f16_branch_and_weights", jnp.float16, jnp.float16),
      ("f16_x_f32_branch_and_weights", jnp.float16, jnp.float32),
  )
  def test_hc_value_and_grad_mixed_and_float16_dtypes(
      self, x_dtype, branch_and_weight_dtype
  ):
    keys = jax.random.split(jax.random.key(7), 6)
    x = jax.random.normal(keys[0], (2, 16, 4, 64), dtype=x_dtype)
    cotangent = jax.random.normal(keys[1], (2, 16, 4, 64), dtype=x_dtype) * 0.1
    hc_weights = common.HcWeights(
        w_pre=jax.random.normal(keys[2], (4,), dtype=branch_and_weight_dtype)
        * 0.25,
        w_post=jax.random.normal(keys[3], (4,), dtype=branch_and_weight_dtype)
        * 0.25,
        w_res=jnp.eye(4, dtype=branch_and_weight_dtype),
    )
    next_weights = common.HcWeights(
        w_pre=jax.random.normal(keys[4], (4, 64), dtype=branch_and_weight_dtype)
        * 0.25,
        w_post=jax.random.normal(keys[5], (4, 1), dtype=branch_and_weight_dtype)
        * 0.25,
        w_res=jnp.eye(4, dtype=branch_and_weight_dtype),
    )

    def loss_fn(x_in, w_in, next_w_in):
      li, ctx = api.hc_pre(x_in, w_in)
      layer_out_mixed = jnp.tanh(li.astype(branch_and_weight_dtype))
      next_li, next_ctx = api.hc_post_and_pre(layer_out_mixed, ctx, next_w_in)
      out = api.hc_post(next_li.astype(branch_and_weight_dtype), next_ctx)
      return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    loss, (dx, dw, dnext_w) = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 1, 2))
    )(x, hc_weights, next_weights)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertEqual(dx.dtype, x_dtype)
    self.assertEqual(dw.w_post.dtype, branch_and_weight_dtype)
    self.assertEqual(dnext_w.w_post.dtype, branch_and_weight_dtype)

  @parameterized.named_parameters(
      ("emb128", 128),
      ("emb256", 256),
      ("emb512", 512),
      ("emb1536", 1536),
  )
  def test_post_value_and_grad_standalone(self, embedding):
    x, weights, _, perms, cotangent = _make_inputs(embedding=embedding)
    config = common.MhcKernelConfig(interpret=_interpret())
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

    np.testing.assert_allclose(actual_val, expected_val, rtol=5e-2, atol=1e-1)
    _assert_grads_close(self, actual_grads, expected_grads)

  def test_bwd_block_size_non_power_of_two(self):
    x, weights, w_branch, perms, cotangent = _make_inputs(
        batch=1, sequence=96, embedding=256
    )
    config = common.MhcKernelConfig(
        block_size=48, bwd_block_size=48, interpret=_interpret()
    )
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
            functools.partial(
                _end_to_end_loss, perms, cotangent, config, "mosaic_tpu"
            ),
            argnums=(0, 1, 2),
        )
    )
    expected_loss, expected_grads = xla_grad_fn(x, weights, w_branch)
    actual_loss, actual_grads = impl_grad_fn(x, weights, w_branch)
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=2e-2, atol=2e-2)
    _assert_grads_close(self, actual_grads, expected_grads)

  @parameterized.named_parameters(
      ("single_block", 1),
      ("software_pipelined", 2),
  )
  def test_pre_value_and_grad_standalone(self, fwd_pipeline_blocks):
    x, weights, _, perms, cotangent = _make_inputs(embedding=256)
    config = common.MhcKernelConfig(
        fwd_pipeline_blocks=fwd_pipeline_blocks, interpret=_interpret()
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

  @parameterized.named_parameters(
      ("single_block_emb256", 256, 1),
      ("pipelined_emb256", 256, 2),
      ("pipelined_emb512", 512, 2),
  )
  def test_post_and_pre_value_and_grad_standalone(
      self, embedding, fwd_pipeline_blocks
  ):
    x, weights_1, _, perms, cotangent = _make_inputs(
        embedding=embedding, seed=0
    )
    _, weights_2, _, _, _ = _make_inputs(embedding=embedding, seed=1)
    config = common.MhcKernelConfig(
        fwd_pipeline_blocks=fwd_pipeline_blocks,
        interpret=_interpret(),
    )
    layer_output, ctx_xla = mhc_xla.pre_xla(x, weights_1, perms, config=config)
    res_bf16 = ctx_xla.residual.astype(jnp.bfloat16)

    tokens = x.shape[0] * x.shape[1]
    keys = jax.random.split(jax.random.key(99), 3)
    w_li = (
        jax.random.normal(
            keys[0], x.shape[:2] + (embedding,), dtype=jnp.float32
        )
        * 0.01
    )
    w_hp = jax.random.normal(keys[1], (tokens, 4), dtype=jnp.float32) * 0.01
    w_res = jax.random.normal(keys[2], (tokens, 16), dtype=jnp.float32) * 0.01

    xla_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(
                _post_and_pre_standalone_loss,
                perms,
                cotangent,
                w_li,
                w_hp,
                w_res,
                tokens,
                config,
                "xla",
            ),
            argnums=(0, 1, 2, 3, 4),
        )
    )
    mosaic_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(
                _post_and_pre_standalone_loss,
                perms,
                cotangent,
                w_li,
                w_hp,
                w_res,
                tokens,
                config,
                "mosaic_tpu",
            ),
            argnums=(0, 1, 2, 3, 4),
        )
    )

    expected_val, expected_grads = xla_fn(
        layer_output, x, ctx_xla.h_post, res_bf16, weights_2
    )
    actual_val, actual_grads = mosaic_fn(
        layer_output, x, ctx_xla.h_post, res_bf16, weights_2
    )
    np.testing.assert_allclose(actual_val, expected_val, rtol=2e-2, atol=2e-2)
    _assert_grads_close(self, actual_grads, expected_grads)

  @parameterized.product(
      embedding=[256, 512],
      implementation=["xla", "mosaic_tpu"],
  )
  def test_two_stage_chain_fused_vs_unfused(self, embedding, implementation):
    x, weights_1, w_branch_1, perms, cotangent = _make_inputs(
        embedding=embedding, seed=0
    )
    _, weights_2, w_branch_2, _, _ = _make_inputs(embedding=embedding, seed=1)
    config = common.MhcKernelConfig(interpret=_interpret())

    unfused_grad_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(
                _two_stage_chain_loss,
                perms,
                cotangent,
                config,
                implementation,
                False,
            ),
            argnums=(0, 1, 2, 3, 4),
        )
    )
    fused_grad_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(
                _two_stage_chain_loss,
                perms,
                cotangent,
                config,
                implementation,
                True,
            ),
            argnums=(0, 1, 2, 3, 4),
        )
    )

    expected_loss, expected_grads = unfused_grad_fn(
        x, weights_1, w_branch_1, weights_2, w_branch_2
    )
    actual_loss, actual_grads = fused_grad_fn(
        x, weights_1, w_branch_1, weights_2, w_branch_2
    )

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-3, atol=1e-3)
    _assert_grads_close(self, actual_grads, expected_grads)

  def test_permutations_stop_gradient(self):
    x, weights, _, perms, cotangent = _make_inputs(embedding=256)
    config = common.MhcKernelConfig(interpret=_interpret())

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
    config = common.MhcKernelConfig(interpret=_interpret())
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
          config=common.MhcKernelConfig(block_size=10, interpret=_interpret()),
      )

    bad_bwd_config = common.MhcKernelConfig(
        bwd_block_size=10, interpret=_interpret()
    )
    layer_input_bwd_test, _ = api.mhc_lite_pre(
        x, weights, perms, config=bad_bwd_config
    )
    self.assertEqual(layer_input_bwd_test.shape, x.shape[:2] + (256,))
    with self.assertRaises(common.UnsupportedInputError):
      jax.grad(
          lambda x_in: jnp.sum(
              api.mhc_lite_pre(x_in, weights, perms, config=bad_bwd_config)[0]
          )
      )(x)


if __name__ == "__main__":
  absltest.main()
