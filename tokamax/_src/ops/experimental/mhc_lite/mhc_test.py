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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pydantic
from tokamax._src import hlo_utils
from tokamax._src.autotuning import cache as autotuning_cache
from tokamax._src.ops.experimental.mhc_lite import api as mhc_api
from tokamax._src.ops.experimental.mhc_lite import arg_specs as mhc_arg_specs
from tokamax._src.ops.experimental.mhc_lite import common as mhc_common
from tokamax._src.ops.experimental.mhc_lite import pallas_mosaic_tpu


def _interpret() -> bool:
  return jax.default_backend() != "tpu"


def _mock_tpu_info(**overrides) -> pltpu.TpuInfo:
  """Constructs a hermetic `TpuInfo` object with overrides."""
  base_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
  overrides.setdefault("vmem_capacity_bytes", 128 * 1024 * 1024)
  return dataclasses.replace(base_info, **overrides)


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
    interpret: bool | None = None,
) -> jax.Array:
  """Runs the mHC API pipeline (`pre` followed by `post`)."""
  if interpret is None:
    interpret = _interpret()
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
    interpret: bool | None,
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
    config = mhc_common.MhcKernelConfig(interpret=_interpret())
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
    )
    actual = _run_pipeline_api(
        x,
        weights,
        permutations,
        implementation=implementation,
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-2)

  def test_unsupported_shape_raises_error(self):
    x, weights, permutations, _ = _make_kernel_inputs(
        batch=1, sequence=16, streams=2, embedding=128
    )
    config = mhc_common.MhcKernelConfig(interpret=_interpret())
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
        functools.partial(_pipeline_vjp_fn, permutations, "xla", None, None),
        x,
        weights,
    )
    expected_dx, expected_dw = expected_vjp_fn(cotangent)

    actual_out, actual_vjp_fn = jax.vjp(
        functools.partial(
            _pipeline_vjp_fn, permutations, "mosaic_tpu", None, None
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

  def test_forward_and_backward_vjp_parity_2d_input(self):
    x, weights, permutations, cotangent = _make_kernel_inputs(
        batch=2, sequence=256, streams=4, embedding=256
    )
    x_2d = x.reshape(2 * 256, 4 * 256)
    cotangent_2d = cotangent.reshape(2 * 256, 4 * 256)
    expected_out, expected_vjp_fn = jax.vjp(
        functools.partial(_pipeline_vjp_fn, permutations, "xla", None, None),
        x_2d,
        weights,
    )
    expected_dx, expected_dw = expected_vjp_fn(cotangent_2d)

    actual_out, actual_vjp_fn = jax.vjp(
        functools.partial(
            _pipeline_vjp_fn, permutations, "mosaic_tpu", None, None
        ),
        x_2d,
        weights,
    )
    actual_dx, actual_dw = actual_vjp_fn(cotangent_2d)

    self.assertEqual(actual_out.shape, (512, 1024))
    self.assertEqual(actual_dx.shape, (512, 1024))
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
          err_msg=f"2D input gradient leaf {i} mismatch",
      )


class TestMhcCostEstimates(absltest.TestCase):
  """Unit tests for analytical CostEstimate helper functions in `common`."""

  def test_cost_estimates_non_zero_and_valid(self):
    tokens, streams, embedding = 256, 4, 512
    pre_fwd = mhc_common.pre_cost(tokens, streams, embedding)
    post_fwd = mhc_common.post_cost(tokens, streams, embedding)
    pre_bwd = mhc_common.pre_cost(tokens, streams, embedding, is_bwd=True)
    post_bwd = mhc_common.post_cost(tokens, streams, embedding, is_bwd=True)
    post_and_pre_fwd = mhc_common.post_and_pre_cost(tokens, streams, embedding)
    post_and_pre_bwd = mhc_common.post_and_pre_cost(
        tokens, streams, embedding, is_bwd=True
    )
    costs = [
        ("pre_fwd", pre_fwd),
        ("post_fwd", post_fwd),
        ("pre_bwd", pre_bwd),
        ("post_bwd", post_bwd),
        ("post_and_pre_fwd", post_and_pre_fwd),
        ("post_and_pre_bwd", post_and_pre_bwd),
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

    self.assertGreater(pre_fwd.transcendentals, 0)
    self.assertGreater(pre_bwd.transcendentals, 0)
    self.assertEqual(post_fwd.transcendentals, 0)
    self.assertEqual(post_bwd.transcendentals, 0)
    self.assertEqual(pre_bwd.flops, 2 * pre_fwd.flops)
    self.assertEqual(post_bwd.flops, 2 * post_fwd.flops)


class TestTpuInfoConfigAndValidation(parameterized.TestCase):
  """Unit tests for hardware-dependent `get_tpu_info()` resolution and validation."""

  def test_resolve_config_uses_tpu_info_defaults(self):
    mock_info = _mock_tpu_info(
        num_lanes=256,
        num_sublanes=16,
        vmem_capacity_bytes=64 * 1024 * 1024,
    )
    with mock.patch.object(
        mhc_common.pltpu, "get_tpu_info", return_value=mock_info
    ):
      resolved = mhc_common.resolve_config(mhc_common.MhcKernelConfig())
      self.assertEqual(resolved.block_size, 256)
      self.assertEqual(resolved.bwd_block_size, 512)
      self.assertEqual(resolved.vmem_limit_bytes, 64 * 1024 * 1024)

      explicit = mhc_common.MhcKernelConfig(
          block_size=128,
          bwd_block_size=256,
          vmem_limit_bytes=32 * 1024 * 1024,
      )
      resolved_explicit = mhc_common.resolve_config(explicit)
      self.assertEqual(resolved_explicit.block_size, 128)
      self.assertEqual(resolved_explicit.bwd_block_size, 256)
      self.assertEqual(resolved_explicit.vmem_limit_bytes, 32 * 1024 * 1024)

  def test_validate_inputs_and_block_size_with_mock_tpu_info(self):
    mock_info = _mock_tpu_info(num_lanes=256, num_sublanes=16)
    with mock.patch.object(
        mhc_common.pltpu, "get_tpu_info", return_value=mock_info
    ):
      # Embedding 128 is not a multiple of num_lanes=256.
      x_bad_emb = jnp.zeros((1, 256, 4, 128), dtype=jnp.bfloat16)
      with self.assertRaisesRegex(
          mhc_common.UnsupportedInputError,
          "Expected embedding dim to be a multiple of 256",
      ):
        mhc_common.validate_inputs(x_bad_emb)

      # Embedding 256 is valid, but block_size=24 is not a multiple of
      # num_sublanes=16.
      x_good_emb = jnp.zeros((1, 256, 4, 256), dtype=jnp.bfloat16)
      with self.assertRaisesRegex(
          mhc_common.UnsupportedInputError,
          "Expected block_size to be a positive multiple of 16",
      ):
        mhc_common.validate_inputs(x_good_emb, block_size=24)

      # Valid with default block_size (256) and explicit multiple of 16 (32).
      mhc_common.validate_inputs(x_good_emb)
      mhc_common.validate_inputs(x_good_emb, block_size=32)


class TestMhcAutotuningAndHeuristics(parameterized.TestCase):
  """Unit tests for TPU hardware heuristics and autotuning configurations."""

  def test_tpu_info_and_heuristic_config(self):
    mock_info = _mock_tpu_info(
        num_lanes=128, num_sublanes=8, vmem_capacity_bytes=64 * 1024 * 1024
    )
    with mock.patch.object(
        mhc_common.pltpu, "get_tpu_info", return_value=mock_info
    ):
      cfg_512 = mhc_common.resolve_config(
          mhc_common.MhcKernelConfig(interpret=False),
          tokens=196608,
          streams=4,
          embedding=512,
      )
      self.assertEqual(
          (
              cfg_512.block_size,
              cfg_512.fwd_pipeline_blocks,
              cfg_512.fwd_sub_tokens,
              cfg_512.fwd_strip_tokens,
              cfg_512.fwd_band_width,
              cfg_512.bwd_block_size,
              cfg_512.bwd_num_chunks,
              cfg_512.bwd_sub_tokens,
          ),
          (256, 2, 16, 512, 256, 256, 4, 16),
      )

      cfg_3200 = mhc_common.resolve_config(
          mhc_common.MhcKernelConfig(interpret=False),
          tokens=3200,
          streams=4,
          embedding=512,
      )
      # 3200 = 25 * 128 is not divisible by 256, so heuristics pick 128.
      self.assertEqual(
          (cfg_3200.block_size, cfg_3200.bwd_block_size), (128, 128)
      )

      cfg_tiny_vmem = mhc_common.resolve_config(
          mhc_common.MhcKernelConfig(vmem_limit_bytes=1), tokens=1024
      )
      self.assertEqual(cfg_tiny_vmem.vmem_limit_bytes, 1)

      cfg_wide = mhc_common.resolve_config(
          mhc_common.MhcKernelConfig(interpret=False),
          tokens=196608,
          streams=4,
          embedding=1536,
      )
      self.assertNotIn(None, dataclasses.astuple(cfg_wide))
      self.assertEqual(
          (cfg_wide.block_size, cfg_wide.fwd_pipeline_blocks), (128, 1)
      )
      self.assertEqual(cfg_wide.fwd_strip_tokens, 128)
      self.assertEqual(cfg_wide.bwd_block_size, 128)
      assert cfg_wide.fwd_strip_tokens is not None
      self.assertLessEqual(cfg_wide.fwd_strip_tokens * 1536, 12 * 128 * 512)

    mock_256 = _mock_tpu_info(
        num_lanes=256, num_sublanes=16, vmem_capacity_bytes=64 * 1024 * 1024
    )
    with mock.patch.object(
        mhc_common.pltpu, "get_tpu_info", return_value=mock_256
    ):
      spec0 = mhc_arg_specs.PRE_ARG_SPECS[0].args
      ba_256 = pallas_mosaic_tpu.PallasTpuMhcLitePre().bind(**spec0)
      self.assertEqual(ba_256.heuristics_config.block_size, 512)
      self.assertEqual(ba_256.heuristics_config.fwd_band_width, 256)

  def test_sublane_only_tokens_and_autotuning_candidates(self):
    x, weights, permutations, cotangent = _make_kernel_inputs(
        batch=1, sequence=96, streams=4, embedding=256
    )
    pre_op = pallas_mosaic_tpu.PallasTpuMhcLitePre(interpret=_interpret())
    ba = pre_op.bind(x.reshape(96, 1024), weights, permutations)
    configs = ba.autotuning_configs
    self.assertNotEmpty(configs)
    self.assertIn(ba.heuristics_config, configs)
    for cfg in configs:
      self.assertNotIn(None, dataclasses.astuple(cfg))
      self.assertEqual(96 % cfg.block_size, 0)

    vjp_op = pallas_mosaic_tpu.PallasTpuMhcLitePreVjp(interpret=_interpret())
    vjp_ba = vjp_op.bind(
        (
            jax.ShapeDtypeStruct((1024, 32), jnp.float32),
            jax.ShapeDtypeStruct((96, 4), jnp.float32),
        ),
        (
            jax.ShapeDtypeStruct((96, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((96, 1024), jnp.bfloat16),
            jax.ShapeDtypeStruct((96, 4), jnp.float32),
            jax.ShapeDtypeStruct((96, 16), jnp.bfloat16),
        ),
        (
            jax.ShapeDtypeStruct((96, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((96, 1024), jnp.bfloat16),
            jax.ShapeDtypeStruct((96, 4), jnp.float32),
            jax.ShapeDtypeStruct((96, 16), jnp.bfloat16),
        ),
        x.reshape(96, 1024),
        weights,
        permutations,
    )
    vjp_configs = vjp_ba.autotuning_configs
    self.assertNotEmpty(vjp_configs)
    self.assertIn(vjp_ba.heuristics_config, vjp_configs)
    for cfg in vjp_configs:
      self.assertNotIn(None, dataclasses.astuple(cfg))
      self.assertEqual(96 % cfg.bwd_block_size, 0)

    def _fused_chain(impl, x_in, w_in):
      li, ctx = mhc_api.mhc_lite_pre(
          x_in,
          w_in,
          permutations,
          config=mhc_common.MhcKernelConfig(interpret=_interpret()),
          implementation=impl,
      )
      li2, ctx2 = mhc_api.mhc_lite_post_and_pre(
          li,
          ctx,
          w_in,
          permutations,
          config=mhc_common.MhcKernelConfig(interpret=_interpret()),
          implementation=impl,
      )
      return mhc_api.mhc_lite_post(
          li2,
          ctx2,
          config=mhc_common.MhcKernelConfig(interpret=_interpret()),
      )

    ref_out, ref_vjp = jax.vjp(
        functools.partial(_fused_chain, "xla"), x, weights
    )
    act_out, act_vjp = jax.vjp(
        functools.partial(_fused_chain, "mosaic_tpu"), x, weights
    )
    np.testing.assert_allclose(act_out, ref_out, rtol=5e-2, atol=1.5e-1)
    np.testing.assert_allclose(
        np.asarray(act_vjp(cotangent)[0], np.float32),
        np.asarray(ref_vjp(cotangent)[0], np.float32),
        rtol=5e-2,
        atol=1.5e-1,
    )

  def test_tokamax_op_autotuning_and_cache_integration(self):
    pre_op = pallas_mosaic_tpu.PallasTpuMhcLitePre(interpret=_interpret())
    post_op = pallas_mosaic_tpu.PallasTpuMhcLitePost(interpret=_interpret())
    post_pre_op = pallas_mosaic_tpu.PallasTpuMhcLitePostAndPre(
        interpret=_interpret()
    )
    fwd_spec_pairs = (
        (pre_op, mhc_arg_specs.PRE_ARG_SPECS),
        (post_op, mhc_arg_specs.POST_ARG_SPECS),
        (post_pre_op, mhc_arg_specs.POST_AND_PRE_ARG_SPECS),
    )
    for op_inst, specs in fwd_spec_pairs:
      self.assertLen(specs, 3)
      expected_fields = {f.name for f in dataclasses.fields(op_inst.config_cls)}
      op_cache = autotuning_cache.AutotuningCache(op_inst)
      for dev_kind in ("TPU7x", "TPU v5"):
        dev_cache = op_cache._load_cache(dev_kind)
        self.assertLen(dev_cache, 3)
        for entry in dev_cache.values():
          best_cfg = entry.fastest_config
          cfg_dict = dataclasses.asdict(best_cfg)
          self.assertEqual(set(cfg_dict), expected_fields)
          pydantic.TypeAdapter(op_inst.config_cls).validate_python(cfg_dict)
      for spec in specs:
        ba = op_inst.bind(**spec.args)
        self.assertIn(ba.heuristics_config, ba.autotuning_configs)
        self.assertNotIn(None, dataclasses.astuple(ba.heuristics_config))
        key = ba.autotuning_cache_key
        for dev_kind in ("TPU7x", "TPU v5"):
          dev_cache = op_cache._load_cache(dev_kind)
          self.assertIn(key, dev_cache)
          self.assertIn(dev_cache[key].fastest_config, ba.autotuning_configs)

    vjp_ops = (
        pallas_mosaic_tpu.PallasTpuMhcLitePreVjp(interpret=_interpret()),
        pallas_mosaic_tpu.PallasTpuMhcLitePostVjp(interpret=_interpret()),
        pallas_mosaic_tpu.PallasTpuMhcLitePostAndPreVjp(interpret=_interpret()),
    )
    for vjp_op in vjp_ops:
      expected_fields = {f.name for f in dataclasses.fields(vjp_op.config_cls)}
      vjp_cache = autotuning_cache.AutotuningCache(vjp_op)
      for dev_kind in ("TPU7x", "TPU v5"):
        dev_cache = vjp_cache._load_cache(dev_kind)
        self.assertLen(dev_cache, 3)
        for entry in dev_cache.values():
          cfg_dict = dataclasses.asdict(entry.fastest_config)
          self.assertEqual(set(cfg_dict), expected_fields)
          pydantic.TypeAdapter(vjp_op.config_cls).validate_python(cfg_dict)

    spec0 = mhc_arg_specs.PRE_ARG_SPECS[0].args
    ba_default = pre_op.bind(**spec0)
    ba_res = pre_op.bind(
        **(spec0 | {"return_residuals": True, "rms_epsilon": 1e-5})
    )
    self.assertEqual(
        ba_default.autotuning_cache_key, ba_res.autotuning_cache_key
    )

    x, weights, permutations, _ = _make_kernel_inputs(
        batch=1, sequence=256, streams=4, embedding=256
    )

    def _loss(x_in, w_in, cfg):
      li, ctx = mhc_api.mhc_lite_pre(x_in, w_in, permutations, config=cfg)
      li2, ctx2 = mhc_api.mhc_lite_post_and_pre(
          li, ctx, w_in, permutations, config=cfg
      )
      out = mhc_api.mhc_lite_post(li2, ctx2, config=cfg)
      return jnp.sum(out)

    default_cfg = mhc_common.MhcKernelConfig(interpret=_interpret())
    bound_args = hlo_utils.get_bound_args(
        jax.grad(functools.partial(_loss, cfg=default_cfg)), x, weights
    )
    captured_ops = {type(ba.op).__name__ for ba in bound_args}
    self.assertEqual(
        captured_ops,
        {
            "PallasTpuMhcLitePre",
            "PallasTpuMhcLitePreVjp",
            "PallasTpuMhcLitePostAndPre",
            "PallasTpuMhcLitePostAndPreVjp",
            "PallasTpuMhcLitePost",
            "PallasTpuMhcLitePostVjp",
        },
    )

    # Forward-only override must keep VJP config as None so VJP uses cache,
    # and backward-only override must keep forward config as None.
    partial_fwd_cfg = mhc_common.MhcKernelConfig(
        block_size=128, interpret=_interpret()
    )
    configured_fwd = pallas_mosaic_tpu.configure_op(
        pre_op, partial_fwd_cfg, tokens=256, embedding=256
    )
    self.assertIsNotNone(configured_fwd.config)
    self.assertEqual(configured_fwd.config.block_size, 128)
    self.assertIsNone(configured_fwd.vjp.config)

    partial_bwd_cfg = mhc_common.MhcKernelConfig(
        bwd_block_size=128, interpret=_interpret()
    )
    configured_bwd = pallas_mosaic_tpu.configure_op(
        pre_op, partial_bwd_cfg, tokens=256, embedding=256
    )
    self.assertIsNone(configured_bwd.config)
    self.assertIsNotNone(configured_bwd.vjp.config)
    self.assertEqual(configured_bwd.vjp.config.bwd_block_size, 128)

    # Verify the lowered HLO reflects the overridden forward config.
    lowered = jax.jit(
        jax.grad(functools.partial(_loss, cfg=partial_fwd_cfg))
    ).lower(x, weights)
    by_name = {
        type(ba.op).__name__: ba.op for ba in hlo_utils.get_opspecs(lowered)
    }
    pre_cfg = by_name["PallasTpuMhcLitePre"].config
    if not isinstance(pre_cfg, pallas_mosaic_tpu.PreConfig):
      self.fail(f"Expected PreConfig; got {pre_cfg}")
    self.assertEqual(pre_cfg.block_size, 128)

    if jax.default_backend() == "cpu":
      _, cpu_ctx = mhc_api.mhc_lite_pre(x, weights, permutations)
      self.assertEqual(cpu_ctx.implementation, "xla")

  def test_invalid_config_fields_and_zero_tokens_raise(self):
    x, weights, permutations, _ = _make_kernel_inputs(
        batch=1, sequence=128, streams=4, embedding=256
    )
    for bad_kwargs in (
        {"block_size": 0},
        {"fwd_pipeline_blocks": 0},
        {"fwd_sub_tokens": 0},
        {"fwd_strip_tokens": 7},
        {"fwd_band_width": 64},
        {"bwd_num_chunks": 0},
        {"bwd_sub_tokens": -8},
        {"vmem_limit_bytes": 0},
    ):
      bad_cfg = mhc_common.MhcKernelConfig(interpret=_interpret(), **bad_kwargs)
      with self.assertRaises(mhc_common.UnsupportedInputError):
        mhc_api.mhc_lite_pre(x, weights, permutations, config=bad_cfg)

    x_zero = jnp.zeros((0, 128, 4, 256), dtype=jnp.bfloat16)
    with self.assertRaises(mhc_common.UnsupportedInputError):
      mhc_common.validate_inputs(x_zero)
    with self.assertRaises(mhc_common.UnsupportedInputError):
      mhc_api.mhc_lite_pre(
          x_zero,
          weights,
          permutations,
          config=mhc_common.MhcKernelConfig(interpret=_interpret()),
      )

  def test_custom_subtile_and_single_strip_parity(self):
    x, weights, permutations, cotangent = _make_kernel_inputs(
        batch=2, sequence=256, streams=4, embedding=256
    )
    custom_cfg = mhc_common.MhcKernelConfig(
        block_size=256,
        fwd_pipeline_blocks=2,
        fwd_sub_tokens=16,
        fwd_strip_tokens=512,
        fwd_band_width=256,
        bwd_block_size=256,
        bwd_num_chunks=2,
        bwd_sub_tokens=16,
        interpret=_interpret(),
    )
    expected_out, expected_vjp_fn = jax.vjp(
        functools.partial(_pipeline_vjp_fn, permutations, "xla", None, None),
        x,
        weights,
    )
    expected_dx, expected_dw = expected_vjp_fn(cotangent)

    actual_out, actual_vjp_fn = jax.vjp(
        functools.partial(
            _pipeline_vjp_fn, permutations, "mosaic_tpu", custom_cfg, None
        ),
        x,
        weights,
    )
    actual_dx, actual_dw = actual_vjp_fn(cotangent)
    np.testing.assert_allclose(actual_out, expected_out, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(
        np.asarray(actual_dx, np.float32),
        np.asarray(expected_dx, np.float32),
        rtol=5e-2,
        atol=1e-1,
    )
    for i, (actual_grad, expected_grad) in enumerate(
        zip(
            jax.tree.leaves(actual_dw),
            jax.tree.leaves(expected_dw),
            strict=True,
        )
    ):
      np.testing.assert_allclose(
          np.asarray(actual_grad, np.float32),
          np.asarray(expected_grad, np.float32),
          rtol=5e-2,
          atol=1e-1,
          err_msg=f"Weight gradient leaf {i} mismatch",
      )


if __name__ == "__main__":
  absltest.main()
