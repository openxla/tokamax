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
from collections.abc import Callable
import dataclasses
import enum
import importlib
import inspect
import types
import typing
from typing import Annotated, Any, Final, TypeAlias, Union

import immutabledict
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pydantic


def _int_power_of_two(n: int) -> int:
  if (n & (n - 1)) != 0:
    raise ValueError(f'Integer is not a power of two: {n}')
  return n


PowerOfTwo: TypeAlias = Annotated[
    pydantic.PositiveInt, pydantic.AfterValidator(_int_power_of_two)
]


def _serialize_module_member(x) -> str:
  module_name = inspect.getmodule(x).__name__
  name = getattr(x, '__name__', str(x))
  return name if module_name == 'builtins' else f'{module_name}.{name}'


def _validate_module_member(x) -> Any:
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


def _validate_np_dtype(x) -> np.dtype:
  return x if isinstance(x, np.dtype) else np.dtype(x)


NumpyDtype = Annotated[
    np.dtype,
    pydantic.PlainSerializer(lambda dtype: dtype.name),
    pydantic.PlainValidator(_validate_np_dtype),
]


# pytype: disable=invalid-annotation
def annotate(typ) -> Any:
  """Annotates types with serializers and validators, as necessary."""
  if typing.get_origin(typ) is Union or isinstance(typ, types.UnionType):
    return Union[tuple(map(annotate, typing.get_args(typ)))]
  if typing.get_origin(typ) is type:
    return Annotated[
        typ,
        pydantic.PlainSerializer(_serialize_module_member),
        pydantic.PlainValidator(_validate_module_member),
    ]
  if typing.get_origin(typ) is Callable:
    return Annotated[
        typ,
        pydantic.PlainSerializer(_serialize_module_member),
        pydantic.PlainValidator(_validate_module_member),
    ]
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

    return Annotated[
        typ,
        pydantic.PlainSerializer(lambda e: e.name),
        pydantic.PlainValidator(validate_enum),
    ]
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


# TODO: Support `BatchedShapeDtype`.
def _serialize_shape_dtype(x) -> str:
  if not isinstance(x, jax.ShapeDtypeStruct):
    raise ValueError(f'Invalid ShapeDtype: {x}')
  return jax.core.ShapedArray(x.shape, x.dtype).str_short(short_dtypes=True)


def _validate_shape_dtype(x) -> jax.ShapeDtypeStruct:
  if isinstance(x, jax.ShapeDtypeStruct):
    return x
  elif not isinstance(x, str) or (idx := x.find('[')) == -1:
    raise ValueError(f'Invalid ShapeDtype: {x}')
  shape = Shape.validate_json(x[idx:])
  return jax.ShapeDtypeStruct(shape, _SHORT_DTYPE_NAMES_MAP[x[:idx]])


Shape = pydantic.TypeAdapter(tuple[int, ...])
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
