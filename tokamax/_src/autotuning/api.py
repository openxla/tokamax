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
from dataclasses import dataclass
import json
from typing import Any, Self, Sequence, TypeAlias, Union

from absl import logging
import jax
from tokamax._src import autotuning
from tokamax._src import hlo_utils
from tokamax._src import serialization
from tokamax._src.autotuning import autotuning_utils
from tokamax._src.ops import op

from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import

HloComputation = Union[
    jax.stages.Lowered,
    hlo_pb2.HloModuleProto,
    Sequence[hlo_pb2.HloModuleProto],
]


# TODO: Add context manager in a separate CL.
@dataclass(frozen=True)
class AutotuningResult:
  """Context manager for autotuning results."""

  device_kind: str
  result: tuple[tuple[op.BoundArguments, autotuning.AutotuningData[Any]], ...]

  def dump(self, fp):
    dump_str = self.dumps()
    fp.write(dump_str)

  def dumps(self) -> str:
    data = {"device_kind": self.device_kind, "result": self.result}
    return json.dumps(data, cls=serialization.JsonEncoder)

  @classmethod
  def load(cls, fp) -> Self:
    return cls.loads(fp.read())

  @classmethod
  def loads(cls, json_str: str) -> Self:
    x = json.loads(json_str, cls=serialization.JsonDecoder)
    return cls(**x)


def get_bound_args(
    x: HloComputation,
) -> tuple[op.BoundArguments, ...]:
  """Returns a tuple of unique BoundArguments for all Tokamax ops in x.

  Args:
    x: Either an HLO computation or an XprofId.

  Returns:
    A tuple of unique BoundArguments for all Tokamax ops in x.

  Raises:
    ValueError: If x is not a supported type.
  """
  hlo_modules = []
  if isinstance(x, hlo_pb2.HloModuleProto):
    hlo_modules.append(x)
  elif isinstance(x, list):
    hlo_modules = x
  elif isinstance(x, jax.stages.Lowered):
    hlo_modules = x.compile().runtime_executable().hlo_modules()
    hlo_modules = [
        hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
        for hlo in hlo_modules
    ]
  else:
    raise ValueError(f"Unsupported HLO computation type {type(x)}")

  total_bound_args = hlo_utils.get_opspecs(hlo_modules)

  # Filter out bound args so that only unique ones remain.
  seen_keys = set()
  unique_bound_args = []
  for bound_arg in total_bound_args:
    key = bound_arg.autotuning_cache_key
    if key not in seen_keys:
      seen_keys.add(key)
      unique_bound_args.append(bound_arg)

  return tuple(unique_bound_args)


def autotune(
    x: tuple[op.BoundArguments, ...] | HloComputation,
    ignore_cache: bool = False,
    ignore_errors: bool = False,
    all_implementations: bool = True,
) -> AutotuningResult:
  """Autotunes all captured ops in x.

  Args:
    x: Either a list of bound arguments or an HLO computation or an XprofId.
    ignore_cache: Whether to ignore the autotuningcache and re-autotune.
    ignore_errors: Whether to ignore errors when autotuning.
    all_implementations: Whether to autotune all implementations of the op.

  Returns:
    An AutotuneResult of the autotuned ops.
  """
  bound_args = x if isinstance(x, tuple) else get_bound_args(x)

  ops_to_autotune = []
  if all_implementations:
    for bound_arg in bound_args:
      # Find all implementations of the op and autotune each one.
      op_implementations = (
          autotuning_utils.get_all_api_implementations_with_specs(bound_arg)
      )
      ops_to_autotune += op_implementations
  else:
    ops_to_autotune += bound_args

  autotuned_ops = []
  for bound_arg in ops_to_autotune:
    try:
      autotuned_data = bound_arg.autotune()
      autotuned_ops.append((bound_arg, autotuned_data))
    except Exception as e:
      logging.exception("Failed to autotune for op %s", bound_arg.op)
      if not ignore_errors:
        raise e

  return AutotuningResult(
      device_kind=jax.devices()[0].device_kind, result=tuple(autotuned_ops)
  )
