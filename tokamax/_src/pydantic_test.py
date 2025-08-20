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
from collections.abc import Callable
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import pydantic
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import utils
from tokamax._src.ops import op as op_base
from tokamax._src.ops.attention import base as attn_base
from tokamax._src.ops.ragged_dot import base as ragged_dot_base
from tokamax._src.ops.attention import bench_arg_specs as attn_arg_specs
from tokamax._src.ops.ragged_dot import bench_arg_specs as ragged_dot_arg_specs


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
class _MyDataclass:
  array: jax.Array
  metadata: int


class _Foo:
  pass


class PydanticTest(parameterized.TestCase):

  def test_power_of_two(self):
    pow2 = pydantic.TypeAdapter(pydantic_lib.PowerOfTwo)
    pow2.validate_python(1)
    pow2.validate_python(2)
    pow2.validate_python(64)

    with self.assertRaises(pydantic.ValidationError):
      pow2.validate_python(0)

    with self.assertRaises(pydantic.ValidationError):
      pow2.validate_python(3)

    with self.assertRaises(pydantic.ValidationError):
      pow2.validate_python(-1)

  @parameterized.parameters(
      (type[_Foo], _Foo),
      (jax.lax.PrecisionLike, jax.lax.Precision.DEFAULT),
      (jax.lax.PrecisionLike, jax.lax.DotAlgorithmPreset.BF16_BF16_F32),
      (jax.lax.PrecisionLike, "highest"),
      (Callable[[jax.Array], jax.Array], jax.nn.swish),
      (Callable[[jax.Array], jax.Array], jax.nn.sigmoid),
      (Callable[[jax.Array], jax.Array], jnp.tanh),
  )
  def test_annotated_roundtrip(self, typ, data):
    config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    adapter = pydantic.TypeAdapter(pydantic_lib.annotate(typ), config=config)
    self.assertEqual(data, adapter.validate_json(adapter.dump_json(data)))

  def test_abstract_dataclass_roundtrip(self):
    shape = jax.ShapeDtypeStruct((1, 2), dtype=jnp.float32)
    data = _MyDataclass(array=shape, metadata=42)  # pytype: disable=wrong-arg-types
    adapter = pydantic.TypeAdapter(pydantic_lib.abstractify(_MyDataclass))
    self.assertEqual(data, adapter.validate_json(adapter.dump_json(data)))

  def test_abstract_tuple_roundtrip(self):
    shape = jax.ShapeDtypeStruct((1, 2), dtype=jnp.float32)
    data = (shape, 42)
    config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    adapter = pydantic.TypeAdapter(
        pydantic_lib.abstractify(tuple[jax.Array, int]), config=config
    )
    self.assertEqual(data, adapter.validate_json(adapter.dump_json(data)))

  @parameterized.named_parameters(
      ("attention", attn_base.DotProductAttention, attn_arg_specs),
      ("ragged_dot", ragged_dot_base.RaggedDot, ragged_dot_arg_specs),
  )
  def test_arg_specs_roundtrip(self, op_cls, arg_specs):
    model = pydantic_lib.get_arg_spec_model("ArgSpec", op_cls().signature)
    for name, spec in arg_specs.ARG_SPECS.items():
      with self.subTest(name):
        spec = model(**op_base._abstractify(_eval_shape(spec)))
        spec_roundtrip = model.model_validate_json(spec.model_dump_json())
        # Convert lists to tuples.
        spec_roundtrip = spec_roundtrip.model_copy(
            update={
                k: tuple(v)
                for k, v in spec_roundtrip.model_dump().items()
                if isinstance(v, list)
            }
        )
        self.assertEqual(spec, spec_roundtrip)


if __name__ == "__main__":
  absltest.main()
