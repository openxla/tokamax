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
"""Autotuning API."""

from collections.abc import Callable, Mapping
import dataclasses
import inspect
from typing import Annotated, Any, Final, Self, Sequence, TypeAlias

from absl import logging
import immutabledict
import jax
import pydantic
from tokamax._src import benchmarking
from tokamax._src import hlo_utils
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.autotuning import autotuner
from tokamax._src.ops import op as op_base
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.attention import base as attention_base
from tokamax._src.ops.gated_linear_unit import api as glu_api
from tokamax._src.ops.gated_linear_unit import base as glu_base
from tokamax._src.ops.normalization import api as normalization_api
from tokamax._src.ops.normalization import base as normalization_base
from tokamax._src.ops.ragged_dot import api as ragged_dot_api
from tokamax._src.ops.ragged_dot import base as ragged_dot_base

from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import


HloComputation: TypeAlias = (
    jax.stages.Lowered
    | hlo_pb2.HloModuleProto
    | Sequence[hlo_pb2.HloModuleProto]
)
BoundArgsAutotuningData: TypeAlias = tuple[
    op_base.BoundArguments, autotuner.AutotuningData[Any]
]


def _serialize_bound_args_autotuning_data(
    value: BoundArgsAutotuningData, info
) -> tuple[dict[str, Any], dict[str, Any]]:
  ba, data = value
  ba_data = _BOUND_ARGS_ADAPTER.dump_python(ba, info)
  del ba_data["op"]["config"]
  del ba_data["op"]["vjp"]
  config_cls = ba.op.config_cls
  data_adapter = pydantic_lib.get_adapter(autotuner.AutotuningData[config_cls])
  data = data_adapter.dump_python(data, info, round_trip=True)
  return ba_data, data


def _validate_bound_args_autotuning_data(value: Any) -> BoundArgsAutotuningData:
  ba, data = value
  if isinstance(ba, op_base.BoundArguments):
    assert isinstance(data, autotuner.AutotuningData)
    return ba, data
  ba = _BOUND_ARGS_ADAPTER.validate_python(ba)
  config_cls = ba.op.config_cls
  data_adapter = pydantic_lib.get_adapter(autotuner.AutotuningData[config_cls])
  return ba, autotuner.AutotuningData(data_adapter.validate_python(data))


@dataclasses.dataclass(frozen=True)
class AutotuningResult:
  """Autotuning results.

  `AutotuningResult`s can be used as a context manager, whereby it will act as
  an overlay for the autotuning cache within the scope of the context; i.e. the
  `AutotuningResult` will be checked first for a matching config, but it will
  fallback to the default autotuning cache if not found. Multiple
  `AutotuningResult`s contexts can be stacked, with the innermost one taking
  precedence.
  """

  device_kind: str
  data: tuple[
      Annotated[
          BoundArgsAutotuningData,
          pydantic.PlainValidator(_validate_bound_args_autotuning_data),
          pydantic.PlainSerializer(_serialize_bound_args_autotuning_data),
      ],
      ...,
  ]

  def dump(self, fp):
    fp.write(self.dumps())

  def dumps(self) -> str:
    return str(_AUTOTUNING_RESULT_ADAPTER.dump_json(self), "utf-8")

  @classmethod
  def load(cls, fp) -> Self:
    return cls.loads(fp.read())

  @classmethod
  def loads(cls, json_data: str) -> Self:
    return _AUTOTUNING_RESULT_ADAPTER.validate_json(json_data)

  def __enter__(self):
    overlay = {}
    for ba, data in self.data:
      key = ba.autotuning_cache_key
      overlay.setdefault(ba.op, {}).setdefault(self.device_kind, {})[key] = data
    op_base.get_autotuning_cache_overlay().append(overlay)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    op_base.get_autotuning_cache_overlay().pop()


_AUTOTUNING_RESULT_ADAPTER = pydantic.TypeAdapter(AutotuningResult)
_BOUND_ARGS_ADAPTER = pydantic_lib.TypeAdapter(op_base.PydanticBoundArguments)


def get_bound_args(
    x: HloComputation,
) -> tuple[op_base.BoundArguments, ...]:
  """Returns a tuple of unique BoundArguments for all Tokamax ops in x.

  Args:
    x: Either an HLO computation or an XprofId.

  Returns:
    A tuple of unique BoundArguments for all Tokamax ops in x.
  """
  if isinstance(x, hlo_pb2.HloModuleProto):
    hlo_modules = (x,)
  elif isinstance(x, (list, tuple)):
    hlo_modules = tuple(x)
  elif isinstance(x, jax.stages.Lowered):
    hlo_modules = tuple(
        hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
        for hlo in x.compile().runtime_executable().hlo_modules()
    )
  else:
    raise ValueError(f"Unsupported HLO computation type {type(x)}")

  # Filter out bound args so that only unique ones remain.
  seen_keys = set()
  unique_bound_args = []
  for bound_arg in hlo_utils.get_opspecs(hlo_modules):
    key = bound_arg.autotuning_cache_key
    if (bound_arg.op, key) not in seen_keys:
      seen_keys.add((bound_arg.op, key))
      unique_bound_args.append(bound_arg)
  return tuple(unique_bound_args)


_API_IMPLEMENTATIONS: Final[
    Mapping[type[op_base.Op], Mapping[str, Callable[..., Any]]]
] = immutabledict.immutabledict({
    normalization_base.Normalization: normalization_api.IMPLEMENTATIONS,
    glu_base.GatedLinearUnit: glu_api.IMPLEMENTATIONS,
    ragged_dot_base.RaggedDot: ragged_dot_api.IMPLEMENTATIONS,
    attention_base.DotProductAttention: attention_api.IMPLEMENTATIONS,
})


def get_op_implementations(op: op_base.Op) -> dict[str, Callable[..., Any]]:
  """Returns all implementations of the given op.

  Args:
    op: The op for which to get the implementations.

  Returns:
    An (implementation name, implementation) mapping.
  """
  mro = inspect.getmro(op.__class__)
  return dict(_API_IMPLEMENTATIONS.get(mro[mro.index(op_base.Op) - 1], {}))


def autotune(
    x: Sequence[op_base.BoundArguments] | HloComputation,
    ignore_cache: bool = False,
    ignore_errors: bool = False,
    all_implementations: bool = True,
) -> AutotuningResult:
  """Autotunes all captured ops in x.

  Args:
    x: Either a list of bound arguments or an HLO computation.
    ignore_cache: Whether to ignore the autotuningcache and re-autotune.
    ignore_errors: Whether to ignore errors when autotuning.
    all_implementations: Whether to autotune all implementations of the op.

  Returns:
    An `AutotuningResult` of the autotuned ops.
  """
  # TODO: Implement `ignore_cache=True`.
  if ignore_cache:
    raise NotImplementedError("`ignore_cache=True` is not implemented.")

  if isinstance(x, (list, tuple)) and isinstance(x[0], op_base.BoundArguments):
    bound_args = tuple(x)
  else:
    bound_args = get_bound_args(x)

  if all_implementations:
    bound_args = tuple(
        op_base.BoundArguments(op, ba.arguments)  # pylint: disable=g-complex-comprehension
        for ba in bound_args
        for op in get_op_implementations(ba.op).values()
        if isinstance(op, op_base.Op)
    )

  data = []
  for bound_arg in bound_args:
    try:
      data.append((bound_arg, bound_arg.autotune()))
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to autotune for op %s", bound_arg.op)
      if not ignore_errors:
        raise

  device_kind = jax.devices()[0].device_kind
  return AutotuningResult(device_kind, tuple(data))
