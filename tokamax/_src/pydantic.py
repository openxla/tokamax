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
from typing import Annotated, Any, Final, Generic, TypeAlias, TypeVar, Union, cast

import immutabledict
import jax
from jax.experimental.pallas import fuser
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pydantic
from pydantic_core import core_schema as cs
from tokamax._src import batching


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
    pydantic.PlainSerializer(lambda dtype: dtype.name),
    pydantic.PlainValidator(_validate_np_dtype),
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
      return ShapeDtype
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
    return _annotate_dataclass(origin)
  return ty


def _annotate_dataclass(cls):
  """Annotates dataclass fields."""
  fields = dataclasses.fields(cls)
  config = pydantic.ConfigDict(arbitrary_types_allowed=True)
  # `Field.type` may be a string, rather than a resolved type, so we need to
  # use `typing.get_type_hints` to get the actual type.
  hints = typing.get_type_hints(cls)
  new_fields = tuple((f.name, annotate(hints[f.name]), f) for f in fields)
  new_cls = dataclasses.make_dataclass(cls.__name__, new_fields)
  new_cls.__pydantic_config__ = config
  adapter = pydantic.TypeAdapter(new_cls)

  def serialize(x, info) -> dict[str, Any]:
    if not isinstance(x, cls):
      raise ValueError(f'Invalid {cls.__name__}: {x}')
    return adapter.dump_python(x, mode=info.mode)

  def validate(x):
    if isinstance(x, cls):
      return x
    return cls(**dataclasses.asdict(adapter.validate_python(x)))

  return Annotated[
      cls,
      pydantic.PlainSerializer(serialize),
      pydantic.PlainValidator(validate),
  ]


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
# pytype: enable=invalid-annotation


_T = TypeVar('_T')
get_adapter = functools.lru_cache(pydantic.TypeAdapter)
_TYPE_ADAPTER = get_adapter(pydantic.ImportString[type])


class AnyInstanceOf(Generic[_T]):  # `Generic` makes pytype happy.
  """Annotates a type, allowing serialization of any instance of the given type.

  The value is serialized with the type name, allowing it to be deserialized
  as the corresponding type.
  """

  @classmethod
  def __class_getitem__(cls, base_type: type[_T]) -> type[_T]:  # pylint: disable=arguments-renamed

    def serialize(value: _T, info) -> dict[str, Any]:
      ty = _TYPE_ADAPTER.dump_python(type(value), mode=info.mode)
      data_adapter = get_adapter(annotate(type(value)))
      return dict(__type=ty) | data_adapter.dump_python(value, mode=info.mode)

    def validate(data: Any) -> _T:
      if isinstance(data, base_type):
        return data
      data = cast(dict[str, Any], data)
      ty = _TYPE_ADAPTER.validate_python(data.pop('__type'))
      return get_adapter(annotate(ty)).validate_python(data)

    return Annotated[
        base_type,
        pydantic.PlainSerializer(serialize),
        pydantic.PlainValidator(validate),
    ]


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


def get_arg_spec_model(
    name: str, signature: inspect.Signature
) -> type[pydantic.BaseModel]:
  """Returns a Pydantic model for the given `inspect.Signature`."""
  params = {}
  for param_name, p in signature.parameters.items():
    if p.annotation is inspect.Parameter.empty:
      annotation = Any
    else:
      annotation = annotate(p.annotation)
    default = ... if p.default is inspect.Parameter.empty else p.default
    params[param_name] = (annotation, default)

  config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
  return pydantic.create_model(name, __config__=config, **params)
