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

import dataclasses
from typing import Any, ClassVar
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src import batching
from tokamax._src import config as config_lib
from tokamax._src import hlo_utils
from tokamax._src import utils
from tokamax._src.ops import fuser_adapter
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.attention import arg_specs as attn_arg_specs
from tokamax._src.ops.attention import base as attn_base
from tokamax._src.ops.gated_linear_unit import arg_specs as glu_arg_specs
from tokamax._src.ops.gated_linear_unit import base as glu_base
from tokamax._src.ops.normalization import arg_specs as norm_arg_specs
from tokamax._src.ops.normalization import base as norm_base
from tokamax._src.ops.ragged_dot import api
from tokamax._src.ops.ragged_dot import arg_specs as ragged_dot_arg_specs
from tokamax._src.ops.ragged_dot import base as ragged_dot_base
from tokamax.ops import op_fusion

try:
  from tokamax._src.ops.attention import triton as triton_attn  # pylint: disable=g-import-not-at-top
except ImportError:
  triton_attn = None

try:
  from tokamax._src.ops.ragged_dot import triton as triton_ragged_dot  # pylint: disable=g-import-not-at-top
except ImportError:
  triton_ragged_dot = None

ragged_dot = api.ragged_dot

_ATTN_ARG_SPECS = attn_arg_specs.ARG_SPECS
_GLU_ARG_SPECS = glu_arg_specs.ARG_SPECS
_NORM_ARG_SPECS = norm_arg_specs.ARG_SPECS
_RAGGED_DOT_ARG_SPECS = ragged_dot_arg_specs.ARG_SPECS


def _bsd(shape, dtype, vmap_axes):
  shape = list(shape)
  vmap_axes = tuple(None if a is None else (a, shape.pop(a)) for a in vmap_axes)
  return batching.BatchedShapeDtype(tuple(shape), dtype, vmap_axes)


def _eval_shape(spec):
  if not callable(spec):
    return spec

  other: list[Any] = [None]
  merge: list[Any] = [None]
  out_tree: list[Any] = [None]

  def f():
    out = spec()
    out_flat, out_tree[0] = jax.tree.flatten(out)
    is_array = lambda x: isinstance(x, jax.Array)
    arrays, other[0], merge[0] = utils.split_merge(is_array, out_flat)
    return arrays

  shapes = jax.eval_shape(f)
  assert out_tree[0] is not None and merge[0] is not None
  return out_tree[0].unflatten(merge[0](shapes, other[0]))


@dataclasses.dataclass(frozen=True)
class _FakeOpConfig:
  blah: int


_HEURISTICS_CONFIG = _FakeOpConfig(1)
_AUTOTUNE_CONFIG = _FakeOpConfig(2)


class _FakeOp(op_lib.Op[Any, jax.Array, None, _FakeOpConfig, Any]):
  config_cls: ClassVar[type[_FakeOpConfig]] = _FakeOpConfig

  def _fwd(self, x: jax.Array, y: jax.Array, *, return_residuals: bool, config):
    assert not return_residuals
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    return x + y, None

  def _get_heuristics_config(self, ba: op_lib.BoundArguments):
    del ba  # Unused.
    return _HEURISTICS_CONFIG

  def _get_autotuning_configs(self, ba: op_lib.BoundArguments):
    del ba  # Unused.
    return {_AUTOTUNE_CONFIG}


class _DeviceRestrictedOp(_FakeOp):

  def supported_on(self, device: jax.Device) -> bool:
    del device
    return False


class NonFusibleOp(op_lib.Op[Any, jax.Array, None, _FakeOpConfig, Any]):
  config_cls: ClassVar[type[_FakeOpConfig]] = _FakeOpConfig

  def _fwd(self, x: jax.Array, y: jax.Array, *, return_residuals: bool, config):
    del config
    assert not return_residuals
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    return x + y, None

  def _get_heuristics_config(self, ba: op_lib.BoundArguments):
    del ba  # Unused.
    return _HEURISTICS_CONFIG


class FusibleOp(op_lib.Op[Any, jax.Array, None, _FakeOpConfig, Any]):
  config_cls: ClassVar[type[_FakeOpConfig]] = _FakeOpConfig
  fusible_argnames: ClassVar[tuple[str, ...]] = ("x",)

  def _fwd(
      self,
      x: jax.Array,
      y: jax.Array | ragged_dot_base.GroupSizes,
      *,
      return_residuals: bool,
      config,
  ):
    del config
    assert not return_residuals
    y_arr = jnp.asarray(y)
    assert x.shape == y_arr.shape, f"{x.shape} != {y_arr.shape}"
    return x + y_arr, None

  def _get_heuristics_config(self, ba: op_lib.BoundArguments):
    del ba  # Unused.
    return _HEURISTICS_CONFIG

  def _fusible_fwd(self, x_f, y, *, out_fusions=None, config):
    assert isinstance(x_f, fuser.Fusion)
    assert isinstance(y, jax.Array)
    assert config == _HEURISTICS_CONFIG
    out = x_f() + y
    return out if out_fusions is None else out_fusions(out)


class OpTest(parameterized.TestCase):

  def test_bind(self):
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    self.assertEqual(_FakeOp().bind(x, y).args, (x, y))

  def test_snake_case_name(self):
    self.assertEqual(
        op_lib.snake_case_name(norm_base.Normalization()), "normalization"
    )
    self.assertEqual(
        op_lib.snake_case_name(ragged_dot_base.RaggedDot()), "ragged_dot"
    )
    if triton_ragged_dot is not None:
      self.assertEqual(
          op_lib.snake_case_name(triton_ragged_dot.TritonRaggedDot()),
          "triton_ragged_dot",
      )
    self.assertEqual(op_lib.snake_case_name(_FakeOp()), "__fake_op")

  def test_device_restriction_raises_on_unsupported_device(self):
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    with self.assertRaisesRegex(NotImplementedError, "Not supported on"):
      _DeviceRestrictedOp()(x, y)

  def test_bypass_device_check_bypasses_device_restriction(self):
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    out = _DeviceRestrictedOp().replace(bypass_device_check=True)(x, y)
    self.assertTrue(jnp.array_equal(out, x + y))

  def test_cross_compile_config_bypasses_device_restriction(self):
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    with config_lib.cross_compile(True):
      out = _DeviceRestrictedOp()(x, y)
    self.assertTrue(jnp.array_equal(out, x + y))

  def test_fuser_unsupported_op_raises(self):
    x = jnp.ones((2, 2))
    y = jnp.ones((2, 2))
    op = NonFusibleOp()

    @fuser_adapter.fuse
    def fused(x, y):
      return op(x * 2, y)

    with self.assertRaisesRegex(
        NotImplementedError, "does not support kernel fusion"
    ):
      fused(x, y)

  def test_fuser_fusible_op(self):
    self.assertIs(op_fusion.fuse, fuser_adapter.fuse)
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    y = jnp.ones((2, 2), dtype=jnp.float32)
    op = FusibleOp()
    self.assertTrue(jnp.array_equal(op(x, y), x + y))

    @op_fusion.fuse
    def fused(x, y):
      return op(x * 2.0, y) - 1.0

    self.assertTrue(jnp.array_equal(fused(x, y), (x * 2.0 + y) - 1.0))

  def test_fuser_fusible_argnames_materializes_non_fusible_args(self):
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    y = jnp.ones((2, 2), dtype=jnp.float32)
    op = FusibleOp()

    @fuser_adapter.fuse
    def fused(x, y):
      return op(x * 2.0, y + 3.0) - 1.0

    self.assertTrue(jnp.array_equal(fused(x, y), (x * 2.0 + (y + 3.0)) - 1.0))

  def test_fuser_fusible_op_unwraps_group_sizes(self):
    x = jnp.array([1, 2], dtype=jnp.int32)
    y = jnp.array([10, 20], dtype=jnp.int32)
    op = FusibleOp()

    @fuser_adapter.fuse
    def fused(x, y):
      gs = ragged_dot_base.GroupSizes(y, (1, 1))
      return op(x * 2, gs)

    self.assertTrue(jnp.array_equal(fused(x, y), x * 2 + y))

  def test_fuser_adapter_materialize(self):
    arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    fusion_leaf = fuser.Fusion(
        func=lambda: arr * 3.0,
        in_type=((), {}),
        out_type=jax.ShapeDtypeStruct(arr.shape, arr.dtype),
    )
    out = fuser_adapter.materialize({"f": fusion_leaf, "a": arr, "n": None})
    self.assertTrue(jnp.array_equal(out["f"], arr * 3.0))
    self.assertTrue(jnp.array_equal(out["a"], arr))
    self.assertIsNone(out["n"])

  def test_fuser_adapter_push_pull_block_spec(self):
    raw_type = jax.ShapeDtypeStruct((32, 32), jnp.float32)

    def capture_out_fusion(epilogue):
      captured = {}

      def body(x_f, out_fusions=None):
        captured["out"] = fuser_adapter.unpack_output_fusion(
            out_fusions, raw_type
        )
        out = x_f()
        return out if out_fusions is None else out_fusions(out)

      fusible_fn = fuser.fusible(body, output_fusion_prefix=True)
      fuser.fuse(lambda x: epilogue(fusible_fn(x)))(
          jnp.ones((32, 32), jnp.float32)
      )
      return captured["out"]

    # Layout-changing epilogue pushes transposed block_shape.
    blocked_spec = pl.BlockSpec((16, 32), lambda i, j: (i, j))
    out_fn, _, out_vals_type, _ = capture_out_fusion(lambda out: out.mT)
    assert out_fn is not None and out_vals_type is not None
    _, val_specs, final_spec = fuser_adapter.push_pull_block_spec(
        out_fn, blocked_spec, out_vals_type, raw_type, grid_len=2
    )
    self.assertEqual(val_specs, ())
    self.assertEqual(final_spec.block_shape, (32, 16))

    # Reconvergent RMSNorm epilogue with BoundedSlice succeeds and matches ref.
    bounded_spec = pl.BlockSpec(
        (pl.BoundedSlice(16), 32), lambda i, j: (pl.ds(i * 16, 16), j)
    )
    w = jnp.arange(32, dtype=jnp.float32).reshape((1, 32))
    rms_fn = lambda out: (
        out
        * jax.lax.rsqrt(jnp.mean(jnp.square(out), axis=-1, keepdims=True) + 1.0)
        * w
    )
    out_fn, out_vals, out_vals_type, _ = capture_out_fusion(rms_fn)
    assert out_fn is not None
    assert out_vals is not None and out_vals_type is not None
    pulled_fn, val_specs, final_spec = fuser_adapter.push_pull_block_spec(
        out_fn, bounded_spec, out_vals_type, raw_type, grid_len=2
    )
    self.assertLen(val_specs, 1)
    self.assertEqual(final_spec.block_shape, (pl.BoundedSlice(16), 32))
    tile = jnp.full((16, 32), 2.0, dtype=jnp.float32)
    self.assertTrue(
        jnp.allclose(pulled_fn((0, 0), (), out_vals, tile), rms_fn(tile))
    )

    # Simulate pre-0.12.1 JAX rejecting unreduced full-axis strict DAG pulls.
    orig_pull = fuser.pull_block_spec

    def legacy_pull(fn, out_block_specs, *, strict_mode=True, **kwargs):
      first_spec = (
          out_block_specs[0]
          if isinstance(out_block_specs, tuple)
          else out_block_specs
      )
      if strict_mode and first_spec in (bounded_spec, spec_3d):
        raise ValueError("Fusion contains a DAG, cannot uniquely pull")
      return orig_pull(fn, out_block_specs, strict_mode=strict_mode, **kwargs)

    raw_3d = jax.ShapeDtypeStruct((16, 32, 32), jnp.float32)
    spec_3d = pl.BlockSpec((16, 16, 16), lambda i, j, k: (i, j, k))
    captured_3d = {}

    def body_3d(x_f, out_fusions=None):
      captured_3d["out"] = fuser_adapter.unpack_output_fusion(
          out_fusions, raw_3d
      )
      out = x_f()
      return out if out_fusions is None else out_fusions(out)

    fusible_3d = fuser.fusible(body_3d, output_fusion_prefix=True)
    fuser.fuse(lambda x: (lambda y: y + jnp.swapaxes(y, 1, 2))(fusible_3d(x)))(
        jnp.ones((16, 32, 32), jnp.float32)
    )
    swap_out_fn, _, swap_vals_type, _ = captured_3d["out"]
    assert swap_out_fn is not None and swap_vals_type is not None

    with mock.patch.object(fuser, "pull_block_spec", side_effect=legacy_pull):
      fb_fn, fb_val_specs, fb_spec = fuser_adapter.push_pull_block_spec(
          out_fn, bounded_spec, out_vals_type, raw_type, grid_len=2
      )
      self.assertLen(fb_val_specs, 1)
      self.assertEqual(fb_spec.block_shape, (pl.BoundedSlice(16), 32))
      self.assertTrue(
          jnp.allclose(fb_fn((0, 0), (), out_vals, tile), rms_fn(tile))
      )

      with self.assertRaises(ValueError):
        fuser_adapter.push_pull_block_spec(
            swap_out_fn, spec_3d, swap_vals_type, raw_3d, grid_len=3
        )

    # Unsound same-shape epilogues raise ValueError rather than falling back.
    v_type = (jax.ShapeDtypeStruct((16, 16), jnp.float32),)
    sub_spec = pl.BlockSpec((16, 16), lambda i, j: (i, j))
    dus_out_fn = lambda vals, out: (
        jax.lax.dynamic_update_slice(out, vals[0], (0, 0)),
    )
    with self.assertRaisesRegex(ValueError, "Inconsistent block specs"):
      fuser_adapter.push_pull_block_spec(
          dus_out_fn, sub_spec, v_type, raw_type, grid_len=2
      )

    bad_out_fn, _, bad_vals_type, _ = capture_out_fusion(
        lambda out: out + out.T
    )
    assert bad_out_fn is not None and bad_vals_type is not None
    with self.assertRaises(ValueError):
      fuser_adapter.push_pull_block_spec(
          bad_out_fn, sub_spec, bad_vals_type, raw_type, grid_len=2
      )

  def test_op_fusion_context_nested_and_inner_unfused_op(self):
    x = jnp.arange(4, dtype=jnp.float32).reshape((2, 2))
    y = jnp.ones((2, 2), dtype=jnp.float32)

    class CompositeFusibleOp(
        op_lib.Op[Any, jax.Array, None, _FakeOpConfig, Any]
    ):
      config_cls: ClassVar[type[_FakeOpConfig]] = _FakeOpConfig

      def _fwd(
          self, x: jax.Array, y: jax.Array, *, return_residuals: bool, config
      ):
        del config
        assert not return_residuals
        assert x.shape == y.shape, f"{x.shape} != {y.shape}"
        return x + y, None

      def _get_heuristics_config(self, ba: op_lib.BoundArguments):
        del ba  # Unused.
        return _HEURISTICS_CONFIG

      def _fusible_fwd(self, x_f, y_f, *, out_fusions=None, config):
        del config
        assert not fuser_adapter.in_op_fusion_context()
        assert isinstance(x_f, fuser.Fusion)
        assert isinstance(y_f, fuser.Fusion)
        # Calling an unfused helper `Op` inside `_fusible_fwd` must run its
        # normal `_fwd` path rather than treating it as a nested fusible.
        out = NonFusibleOp()(x_f(), y_f())
        return out if out_fusions is None else out_fusions(out)

    op = CompositeFusibleOp()
    self.assertFalse(fuser_adapter.in_op_fusion_context())

    @fuser_adapter.fuse(static_argnames=("scale",))
    def inner_fused(x, y, *, scale: float):
      self.assertTrue(fuser_adapter.in_op_fusion_context())
      return op(x * scale, y)

    @fuser_adapter.fuse
    def outer_fused(x, y):
      self.assertTrue(fuser_adapter.in_op_fusion_context())
      out = inner_fused(x, y, scale=2.0) - 1.0
      self.assertTrue(fuser_adapter.in_op_fusion_context())
      return out

    self.assertTrue(
        jnp.array_equal(outer_fused(x, y), (x * 2.0 + y) - 1.0)
    )
    self.assertTrue(
        jnp.array_equal(jax.jit(outer_fused)(x, y), (x * 2.0 + y) - 1.0)
    )
    self.assertFalse(fuser_adapter.in_op_fusion_context())


class BoundArgumentsTest(parameterized.TestCase):

  def test_get_config(self):
    cache = _FakeOp().get_autotuning_cache()
    cache.clear()
    ba = _FakeOp().bind(jnp.zeros((1, 2)), jnp.ones((1, 2)))
    self.assertIs(ba.get_config(), _HEURISTICS_CONFIG)
    self.assertEmpty(cache)
    config = ba.get_config(
        autotune_configs=op_lib.AUTO, cache_autotuning_results=False
    )
    self.assertIn(config, ba.autotuning_configs)
    self.assertEmpty(cache)
    tune_config = _FakeOpConfig(3)
    config = ba.get_config(
        autotune_configs={tune_config}, cache_autotuning_results=False
    )
    self.assertIs(config, tune_config)
    self.assertEmpty(cache)
    self.assertIs(ba.get_config(autotune_configs={tune_config}), tune_config)
    self.assertNotEmpty(cache)
    config = ba.get_config(check_autotuning_cache=False)
    self.assertIs(config, _HEURISTICS_CONFIG)
    self.assertIs(ba.get_config(), tune_config)
    self.assertTrue(_FakeOp().supported_on(jax.devices()[0]))

  def test_default_config(self):
    cache = _FakeOp().get_autotuning_cache()
    cache.clear()
    ba = _FakeOp().bind(jnp.zeros((1, 2)), jnp.ones((1, 2)))
    with config_lib.autotuning_cache_miss_fallback("autotune"):
      self.assertIn(ba.default_config, ba.autotuning_configs)
    cache.clear()
    with config_lib.autotuning_cache_miss_fallback("heuristics"):
      self.assertIs(ba.default_config, _HEURISTICS_CONFIG)
    with config_lib.autotuning_cache_miss_fallback("error"):
      with self.assertRaisesRegex(ValueError, "No config found"):
        _ = ba.default_config

  def test_cached_autotuning_data_cross_compile(self):
    op = _FakeOp()
    data = op_lib.AutotuningData({})
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    ba = op.bind(x, y)
    target_device = "mock_tpu"
    op.get_autotuning_cache(target_device)[ba.autotuning_cache_key] = data

    abstract_mesh = jax.sharding.AbstractMesh(
        (),
        (),
        (),
        abstract_device=jax.sharding.AbstractDevice(target_device, 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):

      def traced_fn(x, y):
        ba_traced = op.bind(x, y)
        self.assertEqual(op_lib.infer_device_kind(ba_traced), target_device)
        self.assertIs(ba_traced.cached_autotuning_data, data)
        return x + y

      jax.jit(traced_fn)(x, y)

  def test_heuristics_config(self):
    ba = _FakeOp().bind(jnp.zeros((1, 2)), jnp.ones((1, 2)))
    self.assertIs(ba.heuristics_config, _HEURISTICS_CONFIG)

  def test_autotuning_configs(self):
    ba = _FakeOp().bind(jnp.zeros((1, 2)), jnp.ones((1, 2)))
    expected = {_AUTOTUNE_CONFIG, _HEURISTICS_CONFIG}
    self.assertEqual(ba.autotuning_configs, expected)

  def test_autotune(self):
    cache = _FakeOp().get_autotuning_cache()
    cache.clear()
    config = _FakeOpConfig(3)
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    results = _FakeOp().bind(x, y).autotune({config}, cache_results=False)
    self.assertIs(results.fastest_config, config)
    self.assertEmpty(cache)
    results = _FakeOp().bind(x, y).autotune({config})
    self.assertIs(results.fastest_config, config)
    self.assertNotEmpty(cache)

  def test_autotune_with_timeout(self):
    config = _FakeOpConfig(3)
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    results = (
        _FakeOp()
        .bind(x, y)
        .autotune({config}, cache_results=False, timeout=120.0)
    )
    self.assertIs(results.fastest_config, config)

  @parameterized.parameters(
      ((1,), (None,)), ((0, 0), (0, None)), ((1, 0), (None, 0))
  )
  def test_autotune_vmap(self, x_vmap_axes, y_vmap_axes):
    config = _FakeOpConfig(4)
    x = _bsd((1, 3, 2), jnp.int8, vmap_axes=x_vmap_axes)
    y = _bsd((1, 2), jnp.int8, vmap_axes=y_vmap_axes)
    results = _FakeOp().bind(x, y).autotune({config})
    self.assertIs(results.fastest_config, config)

  def test_equals(self):
    x = _bsd((1, 3, 2), jnp.int8, vmap_axes=(0, 1))
    y = _bsd((1, 2, 3), jnp.int8, vmap_axes=(0, 1))
    y2 = _bsd((1, 2, 3), jnp.int8, vmap_axes=(1, 0))
    self.assertEqual(_FakeOp().bind(x, y), _FakeOp().bind(x, y))
    self.assertNotEqual(_FakeOp().bind(x, y), _FakeOp().bind(x, y2))

  def test_hash(self):
    x = _bsd((1, 3, 2), jnp.int8, vmap_axes=(0, 1))
    y = _bsd((1, 2, 3), jnp.int8, vmap_axes=(0, 1))
    self.assertEqual(hash(_FakeOp().bind(x, y)), hash(_FakeOp().bind(x, y)))

  @parameterized.named_parameters(
      ("attention", attn_base.DotProductAttention(), _ATTN_ARG_SPECS),
      *(
          ()
          if triton_attn is None
          else (
              (
                  "triton_attn",
                  triton_attn.TritonFlashAttention(),
                  _ATTN_ARG_SPECS,
              ),
              (
                  "triton_attn_stable_softmax",
                  triton_attn.TritonFlashAttention(use_stable_softmax=True),
                  _ATTN_ARG_SPECS,
              ),
          )
      ),
      ("glu", glu_base.GatedLinearUnit(), _GLU_ARG_SPECS),
      ("normalization", norm_base.Normalization(), _NORM_ARG_SPECS),
      ("ragged_dot", ragged_dot_base.RaggedDot(), _RAGGED_DOT_ARG_SPECS),
      *(
          ()
          if triton_ragged_dot is None
          else (
              (
                  "triton_ragged_dot",
                  triton_ragged_dot.TritonRaggedDot(),
                  _RAGGED_DOT_ARG_SPECS,
              ),
              (
                  "triton_ragged_dot_split_k_intermediate_dtype",
                  triton_ragged_dot.TritonRaggedDot(
                      split_k_intermediate_dtype=jnp.float32
                  ),
                  _RAGGED_DOT_ARG_SPECS,
              ),
          )
      ),
  )
  def test_roundtrip(self, op, arg_specs):
    op = op.replace(vjp=None)
    adapter = op_lib.BOUND_ARGS_ADAPTER
    for arg_spec in arg_specs:
      spec = arg_spec.args
      with self.subTest(arg_spec.full_name):
        ba = op.bind(**_eval_shape(spec))
        abstract_args = op_lib._abstractify(dict(ba.arguments))
        ba = op_lib.BoundArguments(op, abstract_args)
        ba2 = adapter.validate_python(adapter.dump_python(ba))
        self.assertEqual(ba, ba2.replace(op=ba2.op.replace(vjp=None)))
        ba3 = adapter.validate_json(adapter.dump_json(ba))
        self.assertEqual(ba, ba3.replace(op=ba3.op.replace(vjp=None)))

  def test_ignore_cache_overlay(self):
    if "H100" not in backend.get_default_device().device_kind:
      self.skipTest("Only test on H100 GPU.")
    # Use a real op so that we have a real autotuning cache.
    # Read in the autotuning cache and then with the overlay it should be None.
    ba = norm_base.Normalization().bind(
        x=jax.ShapeDtypeStruct((2, 2), jnp.bfloat16),
        scale=None,
        offset=None,
        axis=-1,
        epsilon=1e-6,
        scale_offset=0.0,
        subtract_mean=False,
        return_residuals=False,
    )
    self.assertIsNotNone(ba.cached_autotuning_data)

    with config_lib.ignore_autotuning_cache(True):
      self.assertIsNone(ba.cached_autotuning_data)

    with config_lib.ignore_autotuning_cache(False):
      self.assertIsNotNone(ba.cached_autotuning_data)


if __name__ == "__main__":
  absltest.main()
