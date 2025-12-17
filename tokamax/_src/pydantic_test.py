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
from typing import Annotated

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import pydantic
from tokamax._src import batching
from tokamax._src import pydantic as pydantic_lib
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

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)


class _Foo:
  pass


_PL_ATTN_CFG = pl_attn.Config(block_q=64, block_k=64, num_stages=2, num_warps=4)
_PL_DOT_CFG = pl_ragged_dot.Config(
    block_m=128, block_n=128, block_k=32, num_stages=2
)
_OPS = (
    attn_base.DotProductAttention(),
    pl_attn.PallasTritonFlashAttention(),
    pl_attn.PallasTritonFlashAttention(use_stable_softmax=True),
    pl_attn.PallasTritonFlashAttention(config=_PL_ATTN_CFG),
    ragged_dot_base.RaggedDot(),
    pl_ragged_dot.PallasTritonRaggedDot(),
    pl_ragged_dot.PallasTritonRaggedDot(
        config=_PL_DOT_CFG, split_k_intermediate_dtype=jnp.float32
    ),
)


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
      (jnp.dtype, jnp.dtype("int8")),
      (jax.typing.DTypeLike, "int32"),
      (jax.typing.DTypeLike, jnp.float32),
      (jax.typing.DTypeLike, jnp.dtype("bfloat16")),
      (jax.typing.DTypeLike, float),
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
    self.assertEqual(data, adapter.validate_python(adapter.dump_python(data)))
    self.assertEqual(data, adapter.validate_json(adapter.dump_json(data)))

  @parameterized.parameters(
      (jax.ShapeDtypeStruct((1, 2), jnp.float32)),
      (jax.ShapeDtypeStruct((3, 4), jnp.int4),),
      (batching.BatchedShapeDtype((6,), jnp.int8, vmap_axes=((0, 5), (1, 7))),),
      (batching.BatchedShapeDtype((8, 9), jnp.int8, vmap_axes=(None,)),),
      (batching.BatchedShapeDtype((10, 11), jnp.int8, vmap_axes=()),),
  )
  def test_shape_dtype_roundtrip(self, shape):
    ty = Annotated[jax.Array, pydantic_lib.ShapeDtype]
    adapter = pydantic.TypeAdapter(ty)
    self.assertEqual(shape, adapter.validate_python(adapter.dump_python(shape)))
    self.assertEqual(shape, adapter.validate_json(adapter.dump_json(shape)))

  def test_abstract_dataclass_roundtrip(self):
    shape = jax.ShapeDtypeStruct((1, 2), dtype=jnp.float32)
    data = _MyDataclass(array=shape, metadata=42)  # pytype: disable=wrong-arg-types
    adapter = pydantic.TypeAdapter(pydantic_lib.annotate(_MyDataclass))
    self.assertEqual(data, adapter.validate_python(adapter.dump_python(data)))
    self.assertEqual(data, adapter.validate_json(adapter.dump_json(data)))

  def test_abstract_tuple_roundtrip(self):
    shape = jax.ShapeDtypeStruct((1, 2), dtype=jnp.float32)
    data = (shape, 42)
    config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    adapter = pydantic.TypeAdapter(
        pydantic_lib.annotate(tuple[jax.Array, int]), config=config
    )
    self.assertEqual(data, adapter.validate_json(adapter.dump_json(data)))

  @parameterized.parameters(*_OPS)
  def test_op_roundtrip(self, op):
    adapter = pydantic.TypeAdapter(pydantic_lib.annotate(type(op)))
    object.__setattr__(op, "vjp", None)
    op_roundtrip = adapter.validate_python(adapter.dump_python(op))
    object.__setattr__(op_roundtrip, "vjp", None)
    self.assertEqual(op, op_roundtrip)
    op_roundtrip = adapter.validate_json(adapter.dump_json(op))
    object.__setattr__(op_roundtrip, "vjp", None)
    self.assertEqual(op, op_roundtrip)

  @parameterized.parameters(*_OPS)
  def test_any_instance_of_op_roundtrip(self, op):
    adapter = pydantic.TypeAdapter(pydantic_lib.AnyInstanceOf[op_lib.Op])
    object.__setattr__(op, "vjp", None)
    op_roundtrip = adapter.validate_python(adapter.dump_python(op))
    object.__setattr__(op_roundtrip, "vjp", None)
    self.assertEqual(op, op_roundtrip)
    op_roundtrip = adapter.validate_json(adapter.dump_json(op))
    object.__setattr__(op_roundtrip, "vjp", None)
    self.assertEqual(op, op_roundtrip)

  @parameterized.named_parameters(
      ("attention", attn_base.DotProductAttention, attn_arg_specs),
      ("normalization", norm_base.Normalization, norm_arg_specs),
      ("ragged_dot", ragged_dot_base.RaggedDot, ragged_dot_arg_specs),
  )
  def test_arg_specs_roundtrip(self, op_cls, arg_specs):
    spec = pydantic_lib.get_arg_spec_model("ArgSpec", op_cls().signature)
    adapter = pydantic.TypeAdapter(spec)
    for arg_spec in arg_specs.ARG_SPECS:
      spec = arg_spec.args
      with self.subTest(arg_spec.full_name):
        spec = op_lib._abstractify(_eval_shape(spec))
        spec_roundtrip = adapter.validate_python(adapter.dump_python(spec))
        self.assertEqual(spec, spec_roundtrip)
        spec_roundtrip = adapter.validate_json(adapter.dump_json(spec))
        if op_cls is ragged_dot_base.RaggedDot:
          spec["group_sizes"] = spec["group_sizes"].representative_value
        self.assertEqual(spec, spec_roundtrip)


if __name__ == "__main__":
  absltest.main()
