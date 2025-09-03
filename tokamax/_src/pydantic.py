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
import builtins
from collections.abc import Callable, Sequence
import dataclasses
import enum
import functools
import importlib
import inspect
import re
import types
import typing
from typing import Annotated, Any, Final, Generic, TypeAlias, TypeVar, Union, cast

import immutabledict
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pydantic
from tokamax._src import batching


def _int_power_of_two(n: int) -> int:
  if (n & (n - 1)) != 0:
    raise ValueError(f'Integer is not a power of two: {n}')
  return n


PowerOfTwo: TypeAlias = Annotated[
    pydantic.PositiveInt, pydantic.AfterValidator(_int_power_of_two)
]


def _serialize_named_object(x) -> str:
  module_name = inspect.getmodule(x).__name__
  name = getattr(x, '__name__', str(x))
  return name if module_name == 'builtins' else f'{module_name}.{name}'


def _validate_named_object(x) -> Any:
  if not isinstance(x, str):
    return x
  parts = x.rsplit('.', 1)
  if len(parts) == 1:
    try:
      return getattr(builtins, x)
    except AttributeError as e:
      raise ValueError(f'Invalid `builtins` member: {x}') from e
  module_name, name = parts
  # TODO: Create allowlist of modules.
  return getattr(importlib.import_module(module_name), name)


_EnumSerializer = pydantic.PlainSerializer(lambda e: e.name)
_NamedObjectSerializer = pydantic.PlainSerializer(_serialize_named_object)
_NamedObjectValidator = pydantic.PlainValidator(_validate_named_object)


def _validate_np_dtype(x) -> np.dtype:
  return x if isinstance(x, np.dtype) else np.dtype(x)


NumpyDtype: TypeAlias = Annotated[
    np.dtype,
    pydantic.PlainSerializer(lambda dtype: dtype.name),
    pydantic.PlainValidator(_validate_np_dtype),
]


# pytype: disable=invalid-annotation
def annotate(typ) -> Any:
  """Annotates types with serializers and validators, as necessary."""
  if typing.get_origin(typ) is Union or isinstance(typ, types.UnionType):
    return Union[tuple(map(annotate, typing.get_args(typ)))]
  if typing.get_origin(typ) in (type, Callable):
    return Annotated[typ, _NamedObjectSerializer, _NamedObjectValidator]
  if typing.get_origin(typ) is Sequence:
    return Annotated[typ, pydantic.AfterValidator(tuple)]
  # The default enum serialization, using the value, often leads to an ambiguous
  # serialization within unions, so use the name instead (also more readable).
  if issubclass(typ, enum.Enum):
    def validate_enum(x):
      if isinstance(x, typ):
        return x
      if not isinstance(x, str):
        raise ValueError(f'Invalid enum name: {x}')
      try:
        return typ[x]
      except KeyError as e:
        raise ValueError('Invalid enum name') from e

    validator = pydantic.PlainValidator(validate_enum)
    return Annotated[typ, _EnumSerializer, validator]
  if issubclass(typ, np.dtype):
    return NumpyDtype
  return typ


_SHORT_DTYPE_NAMES_MAP: Final[
    immutabledict.immutabledict[str, jax.typing.DTypeLike]
] = immutabledict.immutabledict(
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


def _serialize_shape_dtype(x) -> str:
  if not isinstance(x, jax.ShapeDtypeStruct):
    raise ValueError(f'Invalid ShapeDtype: {x}')
  s = jax.core.ShapedArray(x.shape, x.dtype).str_short(short_dtypes=True)
  if not (isinstance(x, batching.BatchedShapeDtype) and x.vmap_axes):
    return s
  axes_str = str(Axes.dump_json(x.vmap_axes), 'utf-8')
  return ''.join([s, '{vmap_axes=', axes_str, '}'])


_SHAPE_DTYPE_PATTERN = re.compile(r'(.*?)(\[.*?\])(\{vmap_axes=(\[.*\])\})?')


def _validate_shape_dtype(x) -> jax.ShapeDtypeStruct:
  if isinstance(x, jax.ShapeDtypeStruct):
    return x
  if not isinstance(x, str) or (match := _SHAPE_DTYPE_PATTERN.match(x)) is None:
    raise ValueError(f'Invalid ShapeDtype: {x}')
  dtype_str, shape_str, _, vmap_axes_str = match.groups()
  shape = Shape.validate_json(shape_str)
  dtype = _SHORT_DTYPE_NAMES_MAP[dtype_str]
  if vmap_axes_str is None:
    return jax.ShapeDtypeStruct(shape, dtype)
  vmap_axes = Axes.validate_json(vmap_axes_str)
  return batching.BatchedShapeDtype(shape, dtype, vmap_axes=vmap_axes)


Shape = pydantic.TypeAdapter(tuple[int, ...])
Axes = pydantic.TypeAdapter(tuple[int, ...])
ShapeDtype = Annotated[
    jax.ShapeDtypeStruct,
    pydantic.PlainSerializer(_serialize_shape_dtype),
    pydantic.PlainValidator(_validate_shape_dtype),
]


def _abstractify_dataclass(cls):
  """Converts `jax.Array` fields to `ShapeDtype`."""
  fields = dataclasses.fields(cls)
  config = pydantic.ConfigDict(arbitrary_types_allowed=True)
  # `Field.type` may be a string, rather than a resolved type, so we need to
  # use `typing.get_type_hints` to get the actual type.
  hints = typing.get_type_hints(cls)
  new_fields = tuple((f.name, abstractify(hints[f.name]), f) for f in fields)
  new_cls = dataclasses.make_dataclass(cls.__name__, new_fields)
  new_cls.__pydantic_config__ = config
  adapter = pydantic.TypeAdapter(new_cls)

  def serialize(x):
    if not isinstance(x, cls):
      raise ValueError(f'Invalid {cls.__name__}: {x}')
    return adapter.dump_python(x)

  def validate(x):
    if isinstance(x, cls):
      return x
    return cls(**dataclasses.asdict(adapter.validate_python(x)))

  return Annotated[
      cls,
      pydantic.PlainSerializer(serialize),
      pydantic.PlainValidator(validate),
  ]


def abstractify(typ):
  """Converts `jax.Array` types to `ShapeDtype`."""
  if typing.get_origin(typ) is Annotated:
    return Annotated[abstractify(typ.__origin__), *typ.__metadata__]
  if typing.get_origin(typ) is Union or isinstance(typ, types.UnionType):
    return Union[tuple(map(abstractify, typing.get_args(typ)))]
  if typing.get_origin(typ) is tuple:
    return tuple[tuple(map(abstractify, typing.get_args(typ)))]
  if isinstance(typ, type) and issubclass(typ, jaxtyping.AbstractArray):
    typ = typ.array_type
  if typ is jax.Array:
    return ShapeDtype
  if dataclasses.is_dataclass(typ):
    return _abstractify_dataclass(typ)
  return typ


# pytype: enable=invalid-annotation


_T = TypeVar('_T')
get_adapter = functools.lru_cache(pydantic.TypeAdapter)


class AnyInstanceOf(Generic[_T]):  # `Generic` makes pytype happy.
  """Annotates a type, allowing serialization of any instance of the given type.

  The value is serialized with the type name, allowing it to be deserialized
  as the corresponding type.
  """

  @classmethod
  def __class_getitem__(cls, base_type: type[_T]) -> type[_T]:  # pylint: disable=arguments-renamed
    def serialize(value: _T, handler) -> dict[str, Any]:
      return dict(__type=_serialize_named_object(type(value))) | handler(value)

    def validate(data: Any) -> _T:
      if isinstance(data, base_type):
        return data
      ty = _validate_named_object(cast(dict[str, Any], data).pop('__type'))
      return get_adapter(ty).validate_python(data)

    return Annotated[
        pydantic.InstanceOf[pydantic.SerializeAsAny[base_type]],  # pytype: disable=unsupported-operands
        pydantic.WrapSerializer(serialize),
        pydantic.PlainValidator(validate),
    ]


def get_arg_spec_model(
    name: str, signature: inspect.Signature
) -> type[pydantic.BaseModel]:
  """Returns a Pydantic model for the given `inspect.Signature`."""
  params = {}
  for param_name, p in signature.parameters.items():
    if p.annotation is inspect.Parameter.empty:
      annotation = Any
    else:
      annotation = abstractify(annotate(p.annotation))
    default = ... if p.default is inspect.Parameter.empty else p.default
    params[param_name] = (annotation, default)

  config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
  return pydantic.create_model(name, __config__=config, **params)
