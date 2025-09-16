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
"""Pydantic types and utilities."""
from collections.abc import Callable, Sequence
import dataclasses
import enum
import functools
import inspect
import re
import types
import typing
from typing import Annotated, Any, Generic, TypeAlias, TypeVar, Union

import immutabledict
import jax
from jax.experimental.pallas import fuser
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pydantic
import pydantic_core
from pydantic_core import core_schema as cs
from tokamax._src import batching
from typing_extensions import TypedDict  # Required for Python <3.12.


def _int_power_of_two(n: int) -> int:
  if (n & (n - 1)) != 0:
    raise ValueError(f'Integer is not a power of two: {n}')
  return n


PowerOfTwo: TypeAlias = Annotated[
    pydantic.PositiveInt, pydantic.AfterValidator(_int_power_of_two)
]


def _validate_np_dtype(x) -> np.dtype:
  return x if isinstance(x, np.dtype) else np.dtype(x)


NumpyDtype: TypeAlias = Annotated[
    np.dtype,
    pydantic.PlainValidator(_validate_np_dtype),
    pydantic.PlainSerializer(lambda dtype: dtype.name),
]


if not typing.TYPE_CHECKING:
  # `ImportString._serialize` has a bug where it returns `None` for
  # types that have a `.name` attribute, so we patch it here
  # (https://github.com/pydantic/pydantic/issues/12218).
  _ORIG_IMPORT_STRING_SERIALIZE = pydantic.ImportString._serialize  # pylint: disable=protected-access

  def _serialize(v: Any) -> str:
    return v if (data := _ORIG_IMPORT_STRING_SERIALIZE(v)) is None else data

  pydantic.ImportString._serialize = _serialize  # pylint: disable=protected-access


# pytype: disable=invalid-annotation
def annotate(ty: Any) -> Any:
  """Annotates types with serializers and validators, as necessary."""
  # Move `str` to the end of the union.
  if ty == jax.typing.DTypeLike:
    ty = type[Any] | np.dtype | str
  elif ty == jax.typing.DTypeLike | None:
    ty = type[Any] | np.dtype | str | None

  if isinstance(ty, type):
    if issubclass(ty, jaxtyping.AbstractArray):
      ty = ty.array_type
    if ty is jax.Array:
      return Annotated[ty, ShapeDtype]
    if issubclass(ty, enum.Enum):
      return Annotated[ty, EnumByName]
    if issubclass(ty, np.dtype):
      return NumpyDtype

  origin = typing.get_origin(ty) or ty
  if hasattr(origin, '__get_pydantic_core_schema__'):
    return ty
  if origin is Annotated:
    return Annotated[annotate(ty.__origin__), *ty.__metadata__]
  if origin is Union or isinstance(ty, types.UnionType):
    return Union[tuple(map(annotate, typing.get_args(ty)))]
  if origin is tuple:
    return tuple[tuple(map(annotate, typing.get_args(ty)))]
  if origin in (type, Callable):
    return pydantic.ImportString[ty]
  if origin is Sequence:
    return Annotated[ty, pydantic.AfterValidator(tuple)]
  if origin is fuser.Fusion:
    # TODO: Add support for serializing `Fusion`s.
    return Annotated[ty, pydantic.PlainSerializer(str, return_type=str)]
  if dataclasses.is_dataclass(origin):
    return Annotated[ty, Dataclass]
  return ty

# pytype: enable=invalid-annotation


_T = TypeVar('_T')


class TypeAdapter(pydantic.TypeAdapter[_T], Generic[_T]):
  """`TypeAdapter` where serialization info is be passed to `dump_python`.

  The `mode` and `round_trip` attributes from the `SerializationInfo` are
  forwarded to the `dump_python` method of the underlying `TypeAdapter`.
  """

  def dump_python(
      self, instance: Any, info: pydantic.SerializationInfo, **kwargs
  ) -> Any:
    kwargs.setdefault('mode', info.mode)
    kwargs.setdefault('round_trip', info.round_trip)
    return super().dump_python(instance, **kwargs)


get_adapter = functools.lru_cache(TypeAdapter)


class AnyInstanceOf(Generic[_T]):  # `Generic` makes pytype happy.
  """Annotates a type, allowing serialization of any instance of the given type.

  The value is serialized with the type name, allowing it to be deserialized
  as the corresponding type.
  """

  @classmethod
  def __class_getitem__(cls, item: _T) -> _T:
    return Annotated[item, cls()]

  @classmethod
  def __get_pydantic_core_schema__(cls, source, handler):
    assert source is not cls
    type_schema = handler.generate_schema(pydantic.ImportString[type])
    dict_schema = cs.typed_dict_schema(
        dict(__type=cs.typed_dict_field(type_schema)), extra_behavior='allow'
    )

    def serialize(value, handler, info) -> dict[str, Any]:
      data = get_adapter(annotate(type(value))).dump_python(value, info)
      return dict(__type=handler(type(value))) | data

    def validate(value: dict[str, Any]) -> Any:
      return get_adapter(annotate(value.pop('__type'))).validate_python(value)

    to_cls_schema = cs.no_info_plain_validator_function(validate)
    from_dict_schema = cs.chain_schema([dict_schema, to_cls_schema])
    return cs.union_schema(
        [cs.is_instance_schema(source), from_dict_schema],
        serialization=cs.wrap_serializer_function_ser_schema(
            serialize, info_arg=True, schema=type_schema
        ),
    )


class Dataclass:
  """Annotates the fields of a dataclass."""

  @classmethod
  def __get_pydantic_core_schema__(cls, source, handler):
    schema = handler.resolve_ref_schema(handler(source))
    assert schema['type'] == 'dataclass'
    args_schema = schema['schema']
    assert args_schema['type'] == 'dataclass-args'

    origin = typing.get_origin(source) or source
    hints = typing.get_type_hints(origin, include_extras=True)
    for field in args_schema['fields']:
      assert field['type'] == 'dataclass-field'
      ty = hints[field['name']]
      if (annotated_ty := annotate(ty)) != ty:
        if field['schema']['type'] == 'default':  # Retain default values.
          field = field['schema']
        field['schema'] = handler.generate_schema(annotated_ty)
    return schema


# Use the enum name, rather than the value, for serialization. This improves the
# readability of the JSON, and disambiguates enums within a union.
EnumByName = pydantic.GetPydanticSchema(
    lambda ty, _handler: cs.no_info_wrap_validator_function(
        lambda v, handler: (v if isinstance(v, ty) else ty[handler(v)]),
        cs.literal_schema(list(ty.__members__)),
        serialization=cs.plain_serializer_function_ser_schema(
            lambda e: e.name, when_used='json'
        ),
    )
)


class ShapeDtype:
  """Serializes `jax.Array`-like types to an abstract representation."""

  PATTERN = re.compile(r'(.*?)(\[.*?\])(\{vmap_axes=(\[.*\])\})?')
  SHORT_DTYPE_NAMES_MAP = immutabledict.immutabledict(
      bool=bool,
      i4=jnp.int4,
      i8=np.int8,
      i16=np.int16,
      i32=np.int32,
      i64=np.int64,
      u8=np.uint8,
      u16=np.uint16,
      u32=np.uint32,
      u64=np.uint64,
      f16=np.float16,
      f32=np.float32,
      f64=np.float64,
      bf16=jnp.bfloat16,
  )

  @classmethod
  def __get_pydantic_core_schema__(cls, source, handler):
    del handler  # Unused.
    assert source in (jax.Array, jax.ShapeDtypeStruct)
    tuple_schema = lambda s: cs.tuple_schema([s], variadic_item_index=0)
    vmap_axes_schema = tuple_schema(cs.nullable_schema(cs.int_schema()))
    vmap_axes_serializer = pydantic_core.SchemaSerializer(vmap_axes_schema)

    def serialize(x, info) -> Any:
      # We don't want match other types with `shape` and `dtype` attributes.
      if not isinstance(x, (jax.Array, jax.ShapeDtypeStruct)):
        raise ValueError(f'Invalid ShapeDtype: {type(x)}')
      if info.mode == 'python':
        return x
      s = jax.core.ShapedArray(x.shape, x.dtype).str_short(short_dtypes=True)
      if isinstance(x, batching.BatchedShapeDtype) and x.vmap_axes:
        vmap_axes_str = str(vmap_axes_serializer.to_json(x.vmap_axes), 'utf-8')
        return f'{s}{{vmap_axes={vmap_axes_str}}}'
      return s

    def to_shape_dtype(value: str, handler) -> jax.ShapeDtypeStruct:
      if (match := cls.PATTERN.match(value)) is None:
        raise ValueError(f'Invalid ShapeDtype: {value}')
      dtype, shape, _, vmap_axes = handler(match.groups())
      dtype = cls.SHORT_DTYPE_NAMES_MAP[dtype]
      if vmap_axes is None:
        return jax.ShapeDtypeStruct(shape, dtype)
      return batching.BatchedShapeDtype(shape, dtype, vmap_axes=vmap_axes)

    groups_schema = cs.tuple_schema([
        cs.any_schema(),
        cs.json_schema(tuple_schema(cs.int_schema())),
        cs.any_schema(),
        cs.nullable_schema(cs.json_schema(vmap_axes_schema)),
    ])
    from_str_schema = cs.chain_schema([
        cs.str_schema(),
        cs.no_info_wrap_validator_function(to_shape_dtype, groups_schema),
    ])
    return cs.json_or_python_schema(
        json_schema=from_str_schema,
        python_schema=cs.union_schema([
            cs.is_instance_schema(source),
            cs.is_instance_schema(jax.ShapeDtypeStruct),
            from_str_schema,
        ]),
        serialization=cs.plain_serializer_function_ser_schema(
            serialize, info_arg=True
        ),
    )


def get_arg_spec_model(
    name: str, signature: inspect.Signature
) -> type[dict[str, Any]]:
  """Returns a new `TypedDict` type for the given `inspect.Signature`."""
  fields = {}
  for param_name, p in signature.parameters.items():
    if p.annotation is inspect.Parameter.empty:
      annotation = Any
    else:
      annotation = annotate(p.annotation)
    fields[param_name] = annotation
  ty = TypedDict(name, fields, total=False)  # pytype: disable=wrong-arg-types
  ty.__pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)
  return ty
