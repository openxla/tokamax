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

import types
from typing import Any, ClassVar

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import pydantic
from tokamax._src import batching
from tokamax._src import config as config_lib
from tokamax._src.ops import op


_HEURISTICS_CONFIG = object()
_AUTOTUNE_CONFIG = object()


class _FakeOp(op.Op[Any, jax.Array, types.NoneType, object, Any]):
  config_cls: ClassVar[type[object]] = object

  def _fwd(self, x: jax.Array, y: jax.Array, *, return_residuals: bool, config):
    assert not return_residuals
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    return x + y, None

  def _get_heuristics_config(self, ba: op.BoundArguments):
    del ba  # Unused.
    return _HEURISTICS_CONFIG

  def _get_autotuning_configs(self, ba: op.BoundArguments):
    del ba  # Unused.
    return {_AUTOTUNE_CONFIG}


class OpTest(parameterized.TestCase):

  def test_bind(self):
    x = jnp.zeros((1, 2))
    y = jnp.ones((1, 2))
    self.assertEqual(_FakeOp().bind(x, y).args, (x, y))

  def test_get_config(self):
    cache = _FakeOp().get_autotuning_cache()
    cache.clear()
    ba = _FakeOp().bind(jnp.zeros((1, 2)), jnp.ones((1, 2)))
    self.assertIs(ba.get_config(), _HEURISTICS_CONFIG)
    self.assertEmpty(cache)
    config = ba.get_config(
        autotune_configs=op.AUTO, cache_autotuning_results=False
    )
    self.assertIs(config, _AUTOTUNE_CONFIG)
    self.assertEmpty(cache)
    tune_config = object()
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

  def test_default_config(self):
    cache = _FakeOp().get_autotuning_cache()
    cache.clear()
    ba = _FakeOp().bind(jnp.zeros((1, 2)), jnp.ones((1, 2)))
    with config_lib.autotuning_cache_miss_fallback("autotune"):
      self.assertIs(ba.default_config, _AUTOTUNE_CONFIG)
    cache.clear()
    with config_lib.autotuning_cache_miss_fallback("heuristics"):
      self.assertIs(ba.default_config, _HEURISTICS_CONFIG)
    with config_lib.autotuning_cache_miss_fallback("error"):
      with self.assertRaisesRegex(ValueError, "No config found"):
        _ = ba.default_config

  def test_autotune(self):
    cache = _FakeOp().get_autotuning_cache()
    cache.clear()
    config = object()
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
    config = object()
    x = batching.BatchedShapeDtype((1, 3, 2), jnp.int8, vmap_axes=x_vmap_axes)
    y = batching.BatchedShapeDtype((1, 2), jnp.int8, vmap_axes=y_vmap_axes)
    results = _FakeOp().bind(x, y).autotune({config})
    self.assertIs(results.fastest_config, config)


class BoundArgumentsTest(absltest.TestCase):

  def test_equals(self):
    x = batching.BatchedShapeDtype((1, 3, 2), jnp.int8, vmap_axes=(0, 1))
    y = batching.BatchedShapeDtype((1, 2), jnp.int8, vmap_axes=(0, 1))
    y2 = batching.BatchedShapeDtype((1, 2), jnp.int8, vmap_axes=(1, 0))
    self.assertEqual(_FakeOp().bind(x, y), _FakeOp().bind(x, y))
    self.assertNotEqual(_FakeOp().bind(x, y), _FakeOp().bind(x, y2))

  def test_equals_array(self):
    x = jnp.zeros((1, 3, 2))
    x1 = jnp.zeros((1, 3, 2))
    x2 = jnp.ones((1, 3, 2))
    y = jnp.ones((1, 2))
    y1 = jnp.ones((1, 2))
    self.assertEqual(_FakeOp().bind(x, y), _FakeOp().bind(x1, y1))
    self.assertNotEqual(_FakeOp().bind(x, y), _FakeOp().bind(x2, y1))

  def test_hash(self):
    x = batching.BatchedShapeDtype((1, 3, 2), jnp.int8, vmap_axes=(0, 1))
    y = batching.BatchedShapeDtype((1, 2), jnp.int8, vmap_axes=(0, 1))
    self.assertEqual(hash(_FakeOp().bind(x, y)), hash(_FakeOp().bind(x, y)))

  def test_hash_array(self):
    x = jnp.zeros((1, 3, 2))
    x1 = jnp.zeros((1, 3, 2))
    y = jnp.ones((1, 2))
    y1 = jnp.ones((1, 2))
    self.assertEqual(hash(_FakeOp().bind(x, y)), hash(_FakeOp().bind(x1, y1)))

  def test_json_roundtrip(self):
    x = batching.BatchedShapeDtype((1, 3, 2), jnp.int8, vmap_axes=(0, 1))
    y = batching.BatchedShapeDtype((1, 2), jnp.int8, vmap_axes=(0, 1))
    ba = _FakeOp().bind(x, y)
    adapter = pydantic.TypeAdapter(op.PydanticBoundArguments)
    self.assertEqual(ba, adapter.validate_json(adapter.dump_json(ba)))


if __name__ == "__main__":
  absltest.main()
