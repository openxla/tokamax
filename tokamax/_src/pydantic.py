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


_SHORT_DTYPE_NAMES_MAP: Final[
    immutabledict.immutabledict[str, jax.typing.DTypeLike]
] = immutabledict.immutabledict(
    bool=bool,
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


# TODO(cjfj): Support `BatchedShapeDtype`.
def _serialize_shape_dtype(x: jax.ShapeDtypeStruct) -> str:
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


def _abstractify(typ):
  """Converts `jax.Array` types to `ShapeDtype`."""
  if typing.get_origin(typ) is Annotated:
    typ = typ.__origin__
  if typing.get_origin(typ) is Union:
    return Union[tuple(map(_abstractify, typing.get_args(typ)))]
  if isinstance(typ, types.UnionType):
    return Union[tuple(map(_abstractify, typ.__args__))]
  if typ is jax.Array or issubclass(typ, jaxtyping.AbstractArray):
    return ShapeDtype
  return typ


def get_arg_spec_model(
    name: str, signature: inspect.Signature
) -> type[pydantic.BaseModel]:
  """Returns a Pydantic model for the given `inspect.Signature`."""
  params = {}
  for param_name, p in signature.parameters.items():
    if p.annotation is inspect.Parameter.empty:
      annotation = Any
    else:
      annotation = _abstractify(p.annotation)
    default = ... if p.default is inspect.Parameter.empty else p.default
    params[param_name] = (annotation, default)

  config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
  return pydantic.create_model(name, __config__=config, **params)
