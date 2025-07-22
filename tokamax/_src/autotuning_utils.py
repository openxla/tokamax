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
"""Utilities for autotuning Tokamax ops."""

import collections
from collections.abc import Callable, Mapping
import inspect
from typing import Any, Final, Sequence, TypeVar
from absl import logging
import immutabledict
from tokamax._src import autotuning
from tokamax._src.ops import op as op_base
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.attention import base as attention_base
from tokamax._src.ops.gated_linear_unit import api as glu_api
from tokamax._src.ops.gated_linear_unit import base as glu_base
from tokamax._src.ops.normalization import api as normalization_api
from tokamax._src.ops.normalization import base as normalization_base
from tokamax._src.ops.ragged_dot import api as ragged_dot_api
from tokamax._src.ops.ragged_dot import base as ragged_dot_base


API_IMPLEMENTATIONS: Final[
    Mapping[type[op_base.Op], Mapping[str, Callable[..., Any]]]
] = immutabledict.immutabledict({
    normalization_base.Normalization: normalization_api.IMPLEMENTATIONS,
    glu_base.GatedLinearUnit: glu_api.IMPLEMENTATIONS,
    ragged_dot_base.RaggedDot: ragged_dot_api.IMPLEMENTATIONS,
    attention_base.DotProductAttention: attention_api.IMPLEMENTATIONS,
})

_Config = TypeVar('_Config')

def get_op_api_implementations(
    op: op_base.Op,
) -> Mapping[str, Callable[..., Any]] | None:
  """Given a Tokamax op, return the API implementations of this op by searching the reverse map.

  Args:
    op: The Tokamax op to get the implementation name for.

  Returns:
    The (implementation name, all implementations) tuple for the given op spec,
    or None.
  """
  mro = inspect.getmro(op.__class__)
  try:
    base_op_class = mro[mro.index(op_base.Op) - 1]
  except ValueError:
    return None
  return API_IMPLEMENTATIONS.get(base_op_class)


def get_all_api_implementations_with_specs(
    op_spec: op_base.BoundArguments,
) -> tuple[op_base.BoundArguments, ...]:
  """Retrieves all API implementations of that op given an op spec.

  The API implementations found are combined with the op spec's arguments and
  returned as tuples. If op_spec.op's base class is not in the API map,
  returns an empty tuple.

  Args:
    op_spec: The op spec to get all implementations for.

  Returns:
    Tuples of op_base.BoundArguments pairs where each BoundArguments.op is a
    different API implementation of op_spec.op and the arguments are the same as
    in op_spec.
  """
  implementations = get_op_api_implementations(op_spec.op)
  if implementations is None:
    return ()

  args = op_spec.arguments
  return tuple(op_base.BoundArguments(op, args) for op in implementations.values())  # pytype: disable=wrong-arg-types


def autotune_all_captured_ops(
    bound_args_list: Sequence[op_base.BoundArguments],
) -> list[tuple[op_base.BoundArguments, autotuning.AutotuningData[Any]]]:
  """Autotunes all captured ops.

  Args:
    bound_args_list: The opspec tuple that will be autotuned. Should be obtained
      by calling `get_all_api_implementations_with_specs`.

  Returns:
    A list of (bound_args, autotuning results). Each entry in the
    list corresponds to a different spec of the op and its autotuning results.
  """
  autotuned_results = []
  for opspec in bound_args_list:
    try:
      autotuned_data = opspec.autotune()
    except Exception:
      logging.exception('Failed to autotune for op %s', opspec.op)
      continue

    autotuned_results.append((opspec, autotuned_data))
  return autotuned_results
