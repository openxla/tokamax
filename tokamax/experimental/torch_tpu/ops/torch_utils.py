# Copyright 2026 Google LLC
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
"""Torch Tokamax utility functions."""

import ast
from collections.abc import Sequence
import inspect
import textwrap
from typing import Any, Callable, Generic, Optional, TypeVar, overload
import jax
from tokamax._src.ops import op as jax_tokamax_op
import torch


def convert_torch_to_jax_via_meta(
    torch_tensor: torch.Tensor,
) -> jax.ShapeDtypeStruct:
  """Converts a Torch tensor to a JAX ShapeDtypeStruct via metadata.

  Note that this does not copy the data from the Torch tensor to the JAX
  array. This is meant for abstract purposes such as creating BoundArguments
  for serialization and autotuning cache lookup.

  Args:
    torch_tensor: The Torch tensor to convert.

  Returns:
    A JAX ShapeDtypeStruct with the same shape and dtype as the Torch tensor.
  """
  x_meta = torch_tensor.to("meta")
  dtype_str = str(x_meta.dtype).split(".")[-1]
  jax_dtype = getattr(jax.numpy, dtype_str)
  return jax.ShapeDtypeStruct(
      shape=tuple(x_meta.shape),
      dtype=jax_dtype,
  )


def get_configs(
    torch_op: Any,
    *args: Any,
    from_autotuning_cache: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any]:
  """Returns the configs for the forward and backward pass.

  Args:
    torch_op: The Torch op to get the configs for. This should be the forward
      op.
    *args: Positional arguments to the JAX Tokamax op.
    from_autotuning_cache: Whether to use the autotuning cache on disk. If
      false, we fall back to a heuristic defined in get_heuristics_config.
      Failing that, we fall back to a default value in the config constructor.
    **kwargs: Keyword arguments to the JAX Tokamax op.

  Returns:
    A tuple of the forward and backward pass configs.
  """

  fwd_bound_args = torch_op.get_bound_args(*args, **kwargs)
  fwd_config = (
      fwd_bound_args.default_config
      if from_autotuning_cache
      else fwd_bound_args.heuristics_config
  )
  vjp_config = None
  if torch_op.backward_op_torch is not None:
    vjp_bound_args = torch_op.backward_op_torch.get_bound_args(*args, **kwargs)
    vjp_config = (
        vjp_bound_args.default_config
        if from_autotuning_cache
        else vjp_bound_args.heuristics_config
    )
  return fwd_config, vjp_config


def inspect_for_attribute(
    fn: Callable[..., Any],
    attribute_name: str,
) -> None:
  """Verifies that an attribute is used in the given function.

  Args:
    fn: The function or method to inspect.
    attribute_name: The name of the attribute or variable to check for.

  Raises:
    AssertionError: If attribute_name is not used in fn.
  """
  unwrapped_fn = inspect.unwrap(fn)
  is_used = False

  # Check source code via AST if available.
  try:
    source = inspect.getsource(unwrapped_fn)
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
      if isinstance(node, ast.Attribute) and node.attr == attribute_name:
        is_used = True
        break
      if isinstance(node, ast.Name) and node.id == attribute_name:
        is_used = True
        break
  except (OSError, TypeError, SyntaxError):
    pass

  # Fallback to code object inspection (e.g. for dynamic or REPL functions).
  if not is_used:
    code = getattr(unwrapped_fn, "__code__", None)
    if code is not None and attribute_name in code.co_names:
      is_used = True

  fn_name = getattr(fn, "__name__", str(fn))
  assert is_used, (
      f"{attribute_name} is not used in {fn_name}. The inheriting class must"
      f" invoke {attribute_name} in {fn_name}."
  )
