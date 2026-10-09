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
from collections.abc import Iterable
import inspect
import re
import textwrap
from typing import Any, Callable
import jax
from jax import numpy as jnp
import torch

# Attributes defined on the base `TorchOp` class. Subclasses may only read/write
# these attributes (and may only assign them inside `__init__`) so that
# TorchDynamo does not observe instance state mutations or custom attributes on
# `self` during tracing.
ALLOWED_TORCH_OP_ATTRIBUTES: frozenset[str] = frozenset({
    "op_impl_jax",
    "_torch_tokamax_op",
    "backward_op_torch",
    "jax_op_name",
    "is_vjp",
    "donate_argnums",
    "fake_impl",
    "_fake_impl",
    "_ops_registered",
})

# Methods defined on the base `TorchOp` class.
ALLOWED_TORCH_OP_METHODS: frozenset[str] = frozenset({
    "__init__",
    "__init_subclass__",
    "get_bound_args",
    "_register_ops",
    "__call__",
    "deconstruct_config",
    "reconstruct_config",
    "op_impl_call_config_setup",
    "op_impl_call",
    "setup_context",
    "backward",
    "derive_backward_shapes",
})

_DUNDER_CLASS_ATTRIBUTE_REGEX: re.Pattern[str] = re.compile(r"^__.*__$")


def dtype_to_str(
    dtype: torch.dtype | jax.typing.DTypeLike | str | None,
) -> str | None:
  """Converts a PyTorch or JAX dtype to its string name for static_argnums."""
  if dtype is None:
    return None
  if isinstance(dtype, torch.dtype):
    return str(dtype).split(".")[-1]
  if isinstance(dtype, str):
    return dtype
  return jnp.dtype(dtype).name


def str_to_jax_dtype(
    dtype: torch.dtype | jax.typing.DTypeLike | str | None,
) -> jnp.dtype | None:
  """Converts a PyTorch, JAX, or string dtype to a jnp.dtype."""
  dtype_str = dtype_to_str(dtype)
  return jnp.dtype(dtype_str) if dtype_str is not None else None


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


def _inspect_method_for_no_self_attributes(
    cls_name: str,
    method_name: str,
    fn: Callable[..., Any],
    allowed_names: frozenset[str],
) -> None:
  """Inspects a single method on a TorchOp subclass for `self.<attr>` usage."""
  unwrapped_fn = inspect.unwrap(fn)
  allow_mutation = method_name == "__init__"

  try:
    source = inspect.getsource(unwrapped_fn)
    tree = ast.parse(textwrap.dedent(source))
  except (OSError, TypeError, SyntaxError):
    return

  for node in ast.walk(tree):
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
      if isinstance(node.ctx, (ast.Store, ast.Del)) and not allow_mutation:
        raise AssertionError(
            f"{cls_name}.{method_name} mutates 'self.{node.attr}', which is"
            " disallowed outside __init__ because it causes TorchDynamo side"
            " effects."
        )
      if node.attr not in allowed_names:
        raise AssertionError(
            f"{cls_name}.{method_name} uses disallowed attribute"
            f" 'self.{node.attr}'. Only base TorchOp attributes and methods are"
            " allowed to prevent TorchDynamo side effects."
        )
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("setattr", "delattr", "getattr")
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
    ):
      func_name = node.func.id
      attr_name = (
          node.args[1].value
          if len(node.args) >= 2
          and isinstance(node.args[1], ast.Constant)
          and isinstance(node.args[1].value, str)
          else "<dynamic>"
      )
      if func_name in ("setattr", "delattr") and not allow_mutation:
        raise AssertionError(
            f"{cls_name}.{method_name} mutates 'self.{attr_name}' via"
            f" {func_name}(), which is disallowed outside __init__ because it"
            " causes TorchDynamo side effects."
        )
      if attr_name not in allowed_names:
        raise AssertionError(
            f"{cls_name}.{method_name} uses disallowed attribute"
            f" 'self.{attr_name}' via {func_name}(). Only base TorchOp"
            " attributes and methods are allowed to prevent TorchDynamo side"
            " effects."
        )


def inspect_for_no_self_attributes(
    op_or_cls: Any,
    *,
    allowed_attributes: Iterable[str] = ALLOWED_TORCH_OP_ATTRIBUTES,
    allowed_methods: Iterable[str] = ALLOWED_TORCH_OP_METHODS,
) -> None:
  """Verifies that a `TorchOp` class or instance does not use custom `self.<attr>`.

  Checks three invariants to prevent TorchDynamo side effects:
  1. If an instance is provided, `instance.__dict__` only contains attributes
     defined by the base `TorchOp` class (`allowed_attributes`).
  2. Each subclass in the MRO up to `TorchOp` only defines methods from
     `allowed_methods` (no custom class attributes, properties, or helper
     methods on `self`).
  3. Inside every method defined on the subclass, `self.<attr>` only references
     names in `allowed_attributes | allowed_methods`, and `self.<attr>` is never
     mutated (`self.x = ...`, `del self.x`, `setattr(self, ...)`, etc.) outside
     `__init__`.

  Args:
    op_or_cls: A `TorchOp` subclass or instance to inspect.
    allowed_attributes: Attribute names allowed on `self`.
    allowed_methods: Method names allowed on `self`.

  Raises:
    AssertionError: If any disallowed `self.<attribute>` usage or mutation is
      found.
  """
  allowed_attr_set = frozenset(allowed_attributes)
  allowed_method_set = frozenset(allowed_methods)
  allowed_names = allowed_attr_set | allowed_method_set

  cls = op_or_cls if isinstance(op_or_cls, type) else type(op_or_cls)

  if not isinstance(op_or_cls, type):
    for attr_name in op_or_cls.__dict__:
      if attr_name not in allowed_attr_set:
        raise AssertionError(
            f"{cls.__name__} defines disallowed instance attribute"
            f" 'self.{attr_name}'. Only base TorchOp attributes"
            f" {sorted(allowed_attr_set)} are allowed to prevent TorchDynamo"
            " side effects."
        )

  for mro_cls in cls.__mro__:
    if mro_cls is object or mro_cls.__name__ == "TorchOp":
      break
    for attr_name, attr_val in mro_cls.__dict__.items():
      if attr_name not in allowed_names:
        if _DUNDER_CLASS_ATTRIBUTE_REGEX.match(attr_name):
          continue
        raise AssertionError(
            f"{cls.__name__} defines disallowed attribute or method"
            f" '{attr_name}' (accessed via 'self.{attr_name}'). Only base"
            " TorchOp attributes and methods are allowed to prevent"
            " TorchDynamo side effects."
        )
      fn = getattr(attr_val, "_fn", attr_val)
      if callable(fn):
        _inspect_method_for_no_self_attributes(
            cls.__name__, attr_name, fn, allowed_names
        )
