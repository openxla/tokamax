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
import enum
import json

from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
import jax
import jax.numpy as jnp
import pydantic
from tokamax._src import batching
from tokamax._src import quantization
from tokamax._src import serialization


class _MyEnum(enum.Enum):
  A = enum.auto()
  B = enum.auto()


@dataclasses.dataclass(frozen=True)
class _MyData:
  a: int
  b: str


class _MyMapping(immutabledict.immutabledict):
  pass


class _MyPydanticModel(pydantic.BaseModel):
  a: int
  b: str


class SerializationTest(parameterized.TestCase):

  @parameterized.parameters(
      jnp.int32,
      jnp.dtype(jnp.int8),
      jax.ShapeDtypeStruct((1, 2, 3), jnp.bfloat16),
      batching.BatchedShapeDtype((1, 2, 3), jnp.float8_e5m2, (0, None)),
      jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3,  # pytype: disable=attribute-error
      jnp.tanh,
      jax.nn.relu,
      jax.nn.sigmoid,
      jax.nn.swish,
      (immutabledict.immutabledict({"a": 1, 42: (2, 3), jnp.int8: jnp.tanh}),),
      _MyEnum.A,
      _MyData(a=1, b="foo"),
      (_MyMapping({"a": 1, 42: (2, 3), jnp.int8: jnp.tanh}),),
      (
          quantization.QuantizedArray(  # pytype: disable=wrong-arg-types
              values=jax.ShapeDtypeStruct((1, 2, 3), jnp.int4),
              scales=jax.ShapeDtypeStruct((1, 2, 3), jnp.float32),
          ),
      ),
      (_MyPydanticModel(a=1, b="foo"),),
      ({"a": ("b", {"c": (42, ((1, (2, (3, 4)))))})},),
  )
  def test_round_trip(self, o):
    json_str = json.dumps(o, cls=serialization.JsonEncoder)
    self.assertEqual(o, json.loads(json_str, cls=serialization.JsonDecoder))


if __name__ == "__main__":
  absltest.main()
