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
"""Ragged Dot PyTorch Op API using reference implementation."""

from collections.abc import Sequence
from typing import Any
import jax
from jax import numpy as jnp
from tokamax._src import precision as precision_lib
from tokamax._src.ops.ragged_dot import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch
from typing_extensions import override


def dim_nums_to_tuple(
    dim_nums: jax.lax.RaggedDotDimensionNumbers | Sequence[int] | None,
) -> tuple[int, ...]:
  """Serializes RaggedDotDimensionNumbers into a flat tuple of ints for jax_op."""
  if dim_nums is None:
    dim_nums = jax_base.DEFAULT_RAGGED_DOT_DIM_NUMS
  if not isinstance(dim_nums, jax.lax.RaggedDotDimensionNumbers):
    return tuple(dim_nums)
  (lhs_c, rhs_c), (lhs_b, rhs_b) = dim_nums.dot_dimension_numbers
  lhs_r = dim_nums.lhs_ragged_dimensions
  rhs_g = dim_nums.rhs_group_dimensions
  parts = (lhs_c, rhs_c, lhs_b, rhs_b, lhs_r, rhs_g)
  out: list[int] = [len(p) for p in parts]
  for p in parts:
    out.extend(p)
  return tuple(out)


def tuple_to_dim_nums(
    dim_nums: jax.lax.RaggedDotDimensionNumbers | Sequence[int] | None,
) -> jax.lax.RaggedDotDimensionNumbers:
  """Reconstructs RaggedDotDimensionNumbers from its flat int-tuple encoding."""
  if dim_nums is None:
    return jax_base.DEFAULT_RAGGED_DOT_DIM_NUMS
  if isinstance(dim_nums, jax.lax.RaggedDotDimensionNumbers):
    return dim_nums
  lengths = dim_nums[:6]
  idx = 6
  parts: list[list[int]] = []
  for length in lengths:
    parts.append(list(dim_nums[idx : idx + length]))
    idx += length
  return jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=((parts[0], parts[1]), (parts[2], parts[3])),
      lhs_ragged_dimensions=parts[4],
      rhs_group_dimensions=parts[5],
  )


def precision_to_str(
    precision: jax.lax.PrecisionLike,
) -> str | None:
  """Serializes PrecisionLike into a string for jax_op."""
  if precision is None:
    return None
  if isinstance(precision, str) and "," in precision:
    return precision
  canon = precision_lib.canonicalize_precision(precision)
  if isinstance(canon, jax.lax.DotAlgorithmPreset):
    return canon.name
  if (
      isinstance(canon, tuple)
      and len(canon) == 2
      and isinstance(canon[0], jax.lax.Precision)
      and isinstance(canon[1], jax.lax.Precision)
  ):
    return f"{canon[0].name.lower()},{canon[1].name.lower()}"
  raise NotImplementedError(f"Unsupported precision: {precision}")


def str_to_precision(
    precision: jax.lax.PrecisionLike,
) -> jax_base.CanonicalPrecision:
  """Reconstructs CanonicalPrecision from its string representation."""
  if isinstance(precision, str) and "," in precision:
    p0, p1 = precision.split(",", 1)
    return precision_lib.canonicalize_precision((p0, p1))
  return precision_lib.canonicalize_precision(precision)


# Why `_normalize_bound_args_inputs` is needed:
#
# `TorchOp.get_bound_args` constructs `BoundArguments(self.op_impl_jax, ...)`
# directly from `op_impl_jax._fwd`'s signature without calling
# `op_impl_jax.bind(...)`. However, `jax_base.RaggedDot`'s config lookup
# (`default_config` / `heuristics_config` / autotuning cache keys) expects the
# argument canonicalization normally performed in `RaggedDot.bind`:
#   1. `group_sizes` must be wrapped in `jax_base.GroupSizes` (carrying either
#      explicit group sizes or `lhs.shape[0]` as total size) so that
#      `BoundArguments` is hashable for autotuning cache lookup and has
#      representative group sizes for heuristics.
#   2. `ragged_dot_dimension_numbers` must be reconstructed from its serialized
#      int tuple (or defaulted to `DEFAULT_RAGGED_DOT_DIM_NUMS` when `None`),
#      and `precision` must be canonicalized via
#      `precision_lib.canonicalize_precision` (e.g. `None` ->
#      `(Precision.DEFAULT, Precision.DEFAULT)`).
#   3. `preferred_element_type` must be converted from a `torch.dtype` or string
#      into a `jnp.dtype`.
#
# Furthermore, `get_bound_args` is invoked from two different call sites:
#   - User-facing `torch_utils.get_configs(op, lhs, rhs, group_sizes, ...)`,
#     where arguments are in user-facing PyTorch/JAX types.
#   - Internal `TorchOp.op_impl_call_config_setup`, which receives custom_op
#     arguments serialized as tuples/strings alongside `config` and
#     `bwd_config`.
#
# `_normalize_bound_args_inputs` reconciles both call signatures and applies
# `RaggedDot.bind`'s canonicalization so `super().get_bound_args` produces a
# valid `BoundArguments` instance.
def _normalize_bound_args_inputs(
    *args: Any, **kwargs: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes positional/keyword args for RaggedDot.bind signature."""
  pos_names = (
      "lhs",
      "rhs",
      "group_sizes",
      "ragged_dot_dimension_numbers",
      "precision",
      "preferred_element_type",
      "return_residuals",
      "config",
      "bwd_config",
  )

  normalized_kwargs = dict(kwargs)
  for name, val in zip(pos_names, args):
    normalized_kwargs[name] = val
  if "lhs" in normalized_kwargs and "rhs" in normalized_kwargs:
    args = (normalized_kwargs.pop("lhs"), normalized_kwargs.pop("rhs"))
  else:
    args = args[:2]

  for ignored in (
      "return_residuals",
      "config",
      "bwd_config",
  ):
    normalized_kwargs.pop(ignored, None)

  ragged_dot_dimension_numbers = tuple_to_dim_nums(
      normalized_kwargs.get("ragged_dot_dimension_numbers")
  )
  normalized_kwargs["ragged_dot_dimension_numbers"] = (
      ragged_dot_dimension_numbers
  )
  normalized_kwargs["precision"] = str_to_precision(
      normalized_kwargs.get("precision")
  )
  normalized_kwargs["preferred_element_type"] = torch_utils.str_to_jax_dtype(
      normalized_kwargs.get("preferred_element_type")
  )
  normalized_kwargs["activation"] = None
  group_sizes = normalized_kwargs.get("group_sizes")
  if isinstance(group_sizes, torch.Tensor):
    group_sizes = torch_utils.convert_torch_to_jax_via_meta(group_sizes)
  if isinstance(group_sizes, Sequence):
    group_sizes = tuple(group_sizes)
    group_sizes = jax_base.GroupSizes(
        jnp.array(group_sizes, jnp.int32), group_sizes
    )
  if (
      ragged_dot_dimension_numbers == jax_base.DEFAULT_RAGGED_DOT_DIM_NUMS
      and group_sizes is not None
      and not isinstance(group_sizes, jax_base.GroupSizes)
  ):
    lhs = args[0] if args else normalized_kwargs["lhs"]
    group_sizes = jax_base.GroupSizes(group_sizes, lhs.shape[0])
  if group_sizes is not None:
    normalized_kwargs["group_sizes"] = group_sizes
  return args, normalized_kwargs


class _RaggedDot[Config](torch_op.TorchOp[Config]):
  """Ragged Dot PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    # By default, JAX lowers `ragged_dot` into a specialized XLA instruction
    # (`chlo.ragged_dot`) that `torch_tpu`'s compiler pipeline does not yet
    # support. Disabling this flag tells JAX to break `ragged_dot` down into
    # standard matrix and array operations that `torch_tpu` can compile.
    jax.config.update("jax_ragged_dot_use_ragged_dot_instruction", False)
    self.op_impl_jax = jax_base.RaggedDot()
    self.jax_op_name = "base_ragged_dot"
    # Ragged dot is a special case where the forward and backward ops are the
    # same but with different dimension numbers and inputs. See the backward
    # implementation for how dlhs and drhs are computed.
    self.backward_op_torch = self

  @override
  def get_bound_args(
      self,
      *args: Any,
      **kwargs: Any,
  ) -> Any:
    norm_args, norm_kwargs = _normalize_bound_args_inputs(*args, **kwargs)
    return super().get_bound_args(*norm_args, **norm_kwargs)

  @override
  def __call__(
      self,
      lhs: torch.Tensor,
      rhs: torch.Tensor,
      group_sizes: torch.Tensor,
      ragged_dot_dimension_numbers: (
          jax.lax.RaggedDotDimensionNumbers | Sequence[int] | None
      ) = None,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: (
          torch.dtype | jax.typing.DTypeLike | str | None
      ) = None,
      return_residuals: bool = False,
      activation: jax_base.ActivationFunction | None = None,
      configs: tuple[Config | None, Config | None] | None = None,
  ) -> Any:
    if activation is not None:
      raise NotImplementedError(
          "activations are not supported on Torch TPU Tokamax"
      )
    dim_nums = tuple_to_dim_nums(ragged_dot_dimension_numbers)
    dim_nums_tuple = dim_nums_to_tuple(dim_nums)
    precision_str = precision_to_str(precision)
    preferred_element_type_str = torch_utils.dtype_to_str(
        preferred_element_type
    )
    fwd_config, bwd_config = (None, None) if configs is None else configs

    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert self.backward_op_torch is not None, "Backward op not set."
    is_fwd = dim_nums == jax_base.DEFAULT_RAGGED_DOT_DIM_NUMS
    out, residuals = self._torch_tokamax_op(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=dim_nums_tuple,
        precision=precision_str,
        preferred_element_type=preferred_element_type_str,
        return_residuals=is_fwd,
        config=self.deconstruct_config(fwd_config),
        bwd_config=self.backward_op_torch.deconstruct_config(bwd_config),
    )
    return (out, residuals) if return_residuals else out

  @override
  def op_impl_call(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      group_sizes: jax.Array,
      ragged_dot_dimension_numbers: tuple[int, ...] | None = None,
      precision: str | None = None,
      preferred_element_type: str | None = None,
      return_residuals: bool = True,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    del config, bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    dim_nums = tuple_to_dim_nums(ragged_dot_dimension_numbers)
    canon_precision = str_to_precision(precision)
    preferred_element_type_dtype = torch_utils.str_to_jax_dtype(
        preferred_element_type
    )

    out, residuals = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        lhs,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dim_nums,
        precision=canon_precision,
        preferred_element_type=preferred_element_type_dtype,
        return_residuals=return_residuals,
        config=None,
        activation=None,
    )
    return out, (out if residuals is None else residuals)

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: tuple[torch.Tensor, torch.Tensor],
  ) -> None:
    """Saves tensors and attributes to the context for the backward pass."""
    (
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers,
        precision,
        preferred_element_type,
        _,
        _,
        bwd_config,
    ) = inputs
    out, residuals = output
    ctx.save_for_backward(lhs, rhs, group_sizes, residuals, out)
    ctx.ragged_dot_dimension_numbers = ragged_dot_dimension_numbers
    ctx.precision = precision
    ctx.preferred_element_type = preferred_element_type
    ctx.bwd_config = bwd_config

  @override
  def backward(
      self,
      ctx: Any,
      dout: torch.Tensor,
      dresiduals: torch.Tensor | None = None,
  ) -> tuple[
      torch.Tensor, torch.Tensor, None, None, None, None, None, None, None
  ]:
    """Callback for torch register_autograd."""
    del dresiduals  # Unused.
    assert self.backward_op_torch is not None, "Backward op not set."
    lhs, rhs, group_sizes, _, _ = ctx.saved_tensors

    dlhs_dim_nums = jax_base.TRANS_RHS_RAGGED_DOT_DIM_NUMS
    drhs_dim_nums = jax_base.RAGGED_CONTRACTING_DOT_DIM_NUMS

    # Execute dlhs and drhs using the ragged_dot op itself (since
    # self.backward_op_torch is self), retrieving the config for each backward
    # call via torch_utils.get_configs without mutating any attributes on
    # `self`.
    dlhs_config, _ = torch_utils.get_configs(
        self.backward_op_torch,
        dout,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dlhs_dim_nums,
        precision=ctx.precision,
        preferred_element_type=lhs.dtype,
    )
    dlhs = self.backward_op_torch(
        dout,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dlhs_dim_nums,
        precision=ctx.precision,
        preferred_element_type=lhs.dtype,
        return_residuals=False,
        configs=(dlhs_config, None),
    )
    drhs_config, _ = torch_utils.get_configs(
        self.backward_op_torch,
        lhs,
        dout,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=drhs_dim_nums,
        precision=ctx.precision,
        preferred_element_type=rhs.dtype,
    )
    drhs = self.backward_op_torch(
        lhs,
        dout,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=drhs_dim_nums,
        precision=ctx.precision,
        preferred_element_type=rhs.dtype,
        return_residuals=False,
        configs=(drhs_config, None),
    )
    # Return gradients for (lhs, rhs, group_sizes, ragged_dot_dimension_numbers,
    # precision, preferred_element_type, return_residuals, config, bwd_config).
    return dlhs, drhs, None, None, None, None, None, None, None


# Singleton instance of the RaggedDot op.
RaggedDot = _RaggedDot[Any]()  # pylint: disable=invalid-name
