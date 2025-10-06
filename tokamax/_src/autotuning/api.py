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
from typing import Annotated, Any, Final, ParamSpec, Self, Sequence, TypeAlias

from absl import logging
import immutabledict
import jax
from jax.extend import backend
import pydantic
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
import tqdm

from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import


HloComputation: TypeAlias = (
    jax.stages.Lowered
    | hlo_pb2.HloModuleProto
    | Sequence[hlo_pb2.HloModuleProto]
    | hlo_pb2.HloProto
    | Sequence[hlo_pb2.HloProto]
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
    state = op_base.get_autotuning_cache_overlay_state()
    state.stack.append(overlay)
    context = state.context(state.context.value + (id(self),))
    context.__enter__()
    object.__setattr__(self, "_context", context)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self._context.__exit__(exc_type, exc_value, traceback)  # pytype: disable=attribute-error
    object.__delattr__(self, "_context")
    op_base.get_autotuning_cache_overlay_state().stack.pop()


_AUTOTUNING_RESULT_ADAPTER = pydantic.TypeAdapter(AutotuningResult)
_BOUND_ARGS_ADAPTER = pydantic_lib.TypeAdapter(op_base.BoundArguments)
_P = ParamSpec("_P")


def get_bound_args(
    f: (
        Callable[_P, Any]
        | HloComputation
    ),
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> tuple[op_base.BoundArguments, ...]:
  """Returns a tuple of unique BoundArguments for all Tokamax ops in `f`.

  Args:
    f: A callable, an HLO computation, or an XprofId.
    *args: Positional arguments to `f` (only valid if `f` is callable).
    **kwargs: Keyword arguments to `f` (only valid if `f` is callable).

  Returns:
    A tuple of unique BoundArguments for all Tokamax ops in `f`.
  """
  if callable(f):
    if not isinstance(f, jax.stages.Wrapped):
      f = jax.jit(f)
    f = f.lower(*args, **kwargs)
  elif args or kwargs:
    raise ValueError("`args` / `kwargs` are only supported if `f` is callable.")

  hlo_modules = _get_hlo_modules(f)
  # Filter out bound args so that only unique ones remain.
  seen_keys = set()
  unique_bound_args = []
  for bound_arg in hlo_utils.get_opspecs(hlo_modules):
    # The chosen config is serialized into the HLO - remove it here.
    object.__setattr__(bound_arg.op, "config", None)
    key = bound_arg.autotuning_cache_key
    if (bound_arg.op, key) not in seen_keys:
      seen_keys.add((bound_arg.op, key))
      unique_bound_args.append(bound_arg)
  return tuple(unique_bound_args)


_convert_hlo_module = (
    lambda x: x.hlo_module if isinstance(x, hlo_pb2.HloProto) else x
)


def _get_hlo_modules(
    x: HloComputation,
) -> tuple[hlo_pb2.HloModuleProto, ...]:
  """Converts x to a tuple of HLO module protos."""

  if isinstance(x, hlo_pb2.HloModuleProto):
    return (x,)
  elif isinstance(x, hlo_pb2.HloProto):
    return (x.hlo_module,)
  elif isinstance(x, (tuple, list)):
    return tuple(_convert_hlo_module(hlo) for hlo in x)
  elif isinstance(x, jax.stages.Lowered):
    return tuple(
        hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
        for hlo in x.compile().runtime_executable().hlo_modules()
    )
  else:
    raise ValueError(f"Unsupported HLO computation type {type(x)}")


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
    f: (
        Callable[..., Any]
        | Sequence[op_base.BoundArguments]
        | HloComputation
    ),
    *args,
    ignore_cache: bool = False,
    all_implementations: bool = True,
    progress_bar: bool = True,
) -> AutotuningResult:
  """Autotunes all captured ops in x.

  Args:
    f: A callable, a list of bound arguments, or an HLO computation.
    *args: Positional arguments to `f` (only valid if `f` is callable). NOTE -
      To autotune a callable with keyword arguments, pass the results of
      `tokamax.get_bound_args(f, *args, **kwargs)` to `autotune`.
    ignore_cache: Whether to ignore the autotuningcache and re-autotune.
    all_implementations: Whether to autotune all implementations of the op.
    progress_bar: Whether to show a progress bar (default: `True`).

  Returns:
    An `AutotuningResult` of the autotuned ops.
  """
  # TODO: Implement `ignore_cache=True`.
  if ignore_cache:
    raise NotImplementedError("`ignore_cache=True` is not implemented.")

  if isinstance(f, (list, tuple)) and isinstance(f[0], op_base.BoundArguments):
    if args:
      raise ValueError("`args` are only supported if `f` is callable.")
    bound_args = tuple(f)
  else:
    bound_args = get_bound_args(f, *args)  # pytype: disable=paramspec-error

  device_kinds = map(op_base.infer_device_kind, bound_args)
  device_kinds = {k for k in device_kinds if k is not None}
  if not device_kinds:
    device_kind = backend.get_default_device().device_kind
  elif len(device_kinds) == 1:
    device_kind = device_kinds.pop()
  else:
    raise ValueError(f"Multiple device kinds found: {device_kinds}")

  if all_implementations:
    bound_args = tuple(
        op_base.BoundArguments(op, ba.arguments)  # pylint: disable=g-complex-comprehension
        for ba in bound_args
        for op in get_op_implementations(ba.op).values()
        if isinstance(op, op_base.Op)
    )

  data = []
  if progress_bar:
    bound_args = tqdm.tqdm(
        bound_args,
        desc="Autotuning",
        unit=" op calls",
    )

  total_configs = _count_configs(bound_args)
  for bound_arg in tqdm.tqdm(
      bound_args,
      desc="Autotuning",
      unit=" op calls",
      postfix={"Total microbenchmarks": total_configs},
  ):
    try:
      data.append((bound_arg, bound_arg.autotune()))
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to autotune for op %s", bound_arg.op)

  return AutotuningResult(device_kind, tuple(data))


def _count_configs(bound_args: tuple[op_base.BoundArguments, ...]) -> int:
  """Returns the number of unique configs in the given bound arguments."""
  # For ops without explicit configs, we consider there to be a single config.
  count = lambda x: len(x) if x else 1
  return sum(count(ba.autotuning_configs) for ba in bound_args)
