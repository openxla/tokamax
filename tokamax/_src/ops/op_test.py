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
import types
from typing import Any, ClassVar

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src import batching
from tokamax._src import config as config_lib
from tokamax._src import utils
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.attention import base as attn_base
from tokamax._src.ops.attention import pallas_triton as pl_attn
from tokamax._src.ops.normalization import base as norm_base
from tokamax._src.ops.ragged_dot import base as ragged_dot_base
from tokamax._src.ops.ragged_dot import pallas_triton as pl_ragged_dot
from tokamax._src.ops.attention import arg_specs as attn_arg_specs
from tokamax._src.ops.normalization import arg_specs as norm_arg_specs
from tokamax._src.ops.ragged_dot import arg_specs as ragged_dot_arg_specs


def _bsd(shape, dtype, vmap_axes):
  shape = list(shape)
  vmap_axes = tuple(None if a is None else (a, shape.pop(a)) for a in vmap_axes)
  return batching.BatchedShapeDtype(tuple(shape), dtype, vmap_axes)


def _eval_shape(spec):
  if not callable(spec):
    return spec

  other = [None]
  merge = [None]
  out_tree = [None]

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


class _FakeOp(op_lib.Op[Any, jax.Array, types.NoneType, _FakeOpConfig, Any]):
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


class OpTest(parameterized.TestCase):

  def test_bind(self):
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    self.assertEqual(_FakeOp().bind(x, y).args, (x, y))


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
      ("attention", attn_base.DotProductAttention(), attn_arg_specs),
      ("pl_attn", pl_attn.PallasTritonFlashAttention(), attn_arg_specs),
      (
          "pl_attn_stable_softmax",
          pl_attn.PallasTritonFlashAttention(use_stable_softmax=True),
          attn_arg_specs,
      ),
      ("normalization", norm_base.Normalization(), norm_arg_specs),
      ("ragged_dot", ragged_dot_base.RaggedDot(), ragged_dot_arg_specs),
      (
          "pl_ragged_dot",
          pl_ragged_dot.PallasTritonRaggedDot(),
          ragged_dot_arg_specs,
      ),
      (
          "pl_ragged_dot_split_k_intermediate_dtype",
          pl_ragged_dot.PallasTritonRaggedDot(
              split_k_intermediate_dtype=jnp.float32
          ),
          ragged_dot_arg_specs,
      ),
  )
  def test_roundtrip(self, op, arg_specs):
    op = op.replace(vjp=None)
    adapter = op_lib.BOUND_ARGS_ADAPTER
    arg_specs = arg_specs.ARG_SPECS
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


if __name__ == "__main__":
  absltest.main()
