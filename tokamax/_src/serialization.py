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
"""Utilities for serializing and deserializing Tokamax types."""

from collections.abc import Mapping
import dataclasses
import enum
import functools
import importlib
import inspect
import json
from typing import Any
import jax
import jax.numpy as jnp
import pydantic
from tokamax._src import batching


_CLASS_FIELD_NAMES = {
    jax.ShapeDtypeStruct: ("shape", "dtype"),
    batching.BatchedShapeDtype: ("shape", "dtype", "vmap_axes"),
}


class JsonEncoder(json.JSONEncoder):
  """JSON encoder for Tokamax types."""

  def default(self, o: Any) -> Any:
    if isinstance(o, enum.Enum):
      return dict(__cls=_cls_key(type(o)), value=o.name)

    if (field_names := _CLASS_FIELD_NAMES.get(type(o))) is not None:
      fields = {field: getattr(o, field) for field in field_names}
      return dict(__cls=_cls_key(type(o)), **fields)

    if dataclasses.is_dataclass(o):
      if o.__class__.__name__ == "BoundArguments":
        class_full_path = (
            o.op.__class__.__module__ + "." + o.op.__class__.__name__
        )
        vjp_full_path = o.op.vjp.__module__ + "." + o.op.vjp.__class__.__name__
        fields = dataclasses.asdict(o)
        fields["tokamax_op"] = class_full_path
        fields["tokamax_op_vjp"] = vjp_full_path
        return dict(__cls=_cls_key(type(o)), **fields)

      fields = dataclasses.asdict(o)
      return dict(__cls=_cls_key(type(o)), **fields)

    if isinstance(o, pydantic.BaseModel):
      return dict(__cls=_cls_key(type(o)), **o.model_dump())

    if isinstance(o, jnp.dtype):
      return dict(__cls="jnp.dtype", name=o.name)

    if isinstance(o, type):
      module = inspect.getmodule(o).__name__
      return dict(__cls="type", name=f"{module}.{o.__name__}")

    if isinstance(o, functools.partial):
      return dict(
          __cls=_cls_key(type(o)), func=o.func, args=o.args, kwargs=o.keywords
      )

    if callable(o):
      module = inspect.getmodule(o).__name__
      name = getattr(o, "__name__", str(o))
      return dict(__cls="function", name=f"{module}.{name}")

    if isinstance(o, Mapping):
      # Serialize items with non-string keys as a list.
      ret = dict(__cls=_cls_key(type(o)))
      non_str_key_items = []
      for k, v in o.items():
        if isinstance(k, str):
          ret[k] = v
        else:
          non_str_key_items.append((k, v))
      if non_str_key_items:
        ret["__items"] = non_str_key_items
      return ret

    return super().default(o)


def get_module_member(name):
  module_name, name = name.rsplit(".", 1)
  return getattr(importlib.import_module(module_name), name)


def _mapping_handler(cls, o):
  items = o.pop("__items", [])
  return cls({k: tuple(v) if isinstance(v, list) else v for k, v in items}, **o)


_HANDLERS = {
    "type": lambda o: get_module_member(o["name"]),
    "function": lambda o: get_module_member(o["name"]),
    "jnp.dtype": lambda o: jnp.dtype(o["name"]),
}


def _cls_key(cls) -> str:
  return ".".join((cls.__module__, cls.__name__))


def _bound_arguments_handler(o, cls_key):
  cls = get_module_member(cls_key)
  class_full_path = o.pop("tokamax_op")
  vjp_full_path = o.pop("tokamax_op_vjp")
  op_config = o.pop("op")
  vjp = get_module_member(vjp_full_path)(**op_config.pop("vjp"))
  op = get_module_member(class_full_path)(**{"vjp": vjp, **op_config})
  final_cls = cls(**{"op": op, "arguments": o["arguments"]})
  return final_cls


def _get_handler(cls_key):
  """Returns a handler for a class."""
  cls = get_module_member(cls_key)
  if "BoundArguments" in cls.__name__:
    return _bound_arguments_handler
  if dataclasses.is_dataclass(cls) or (cls in _CLASS_FIELD_NAMES):
    return lambda o: cls(**o)
  if issubclass(cls, pydantic.BaseModel):
    return cls.model_validate
  if issubclass(cls, enum.Enum):
    return lambda o: cls[o["value"]]
  if issubclass(cls, Mapping):
    return functools.partial(_mapping_handler, cls)
  if issubclass(cls, functools.partial):
    return lambda o: cls(o["func"], *o["args"], **o["kwargs"])
  raise ValueError(f"Unsupported type: {cls}")


def _lists_to_tuples(o):
  is_list = lambda x: isinstance(x, list)

  def list_to_tuple(x):
    return tuple(map(_lists_to_tuples, x)) if is_list(x) else x

  return jax.tree.map(list_to_tuple, o, is_leaf=is_list)


def _object_hook(o: dict[str, Any]) -> Any:
  """Object hook for a JSON decoder."""
  o = _lists_to_tuples(o)

  if (cls_key := o.pop("__cls", None)) is None:
    return o

  if (handler := _HANDLERS.get(cls_key)) is None:
    _HANDLERS[cls_key] = handler = _get_handler(cls_key)
  if "BoundArguments" in cls_key:
    return handler(o, cls_key)
  return handler(o)


class JsonDecoder(json.JSONDecoder):
  """JSON decoder for Tokamax types."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, object_hook=_object_hook, **kwargs)
