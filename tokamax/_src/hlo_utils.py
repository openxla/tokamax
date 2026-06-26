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
"""Utilities for extracting kernel information from StableHLO."""

from collections.abc import Callable
import functools
from typing import Any, Final, Sequence, TypeAlias

import immutabledict
import jax
from jax import export
from jax.interpreters import mlir
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import stablehlo
from tokamax._src import hlo_utils_common
from tokamax._src.ops import op as op_lib

DISABLE_JAX_EXPORT_CHECKS: Final[tuple[export.DisabledSafetyCheck, ...]] = (
    *(
        export.DisabledSafetyCheck.custom_call(key)
        for key in hlo_utils_common.PALLAS_TRITON_KEYS
    ),
    export.DisabledSafetyCheck.custom_call(hlo_utils_common.MOSAIC_GPU_KEY),
    export.DisabledSafetyCheck.custom_call(hlo_utils_common.MOSAIC_TPU_KEY),
    export.DisabledSafetyCheck.custom_call(hlo_utils_common.TRITON_KEY),
    export.DisabledSafetyCheck.custom_call(hlo_utils_common.TRITON_FFI_KEY),
)

HloComputation: TypeAlias = (
    jax.stages.Lowered
    | ir.Module
)

TritonKernelInfo: TypeAlias = hlo_utils_common.TritonKernelInfo
KernelInfoBase: TypeAlias = hlo_utils_common.KernelInfoBase


def get_kernel_info(
    x: HloComputation, include_xla_kernels: bool = True
) -> tuple[KernelInfoBase, ...]:
  """Extracts accelerator kernel information from HLO.

  Args:
    x: The lowered JAX function from which to extract kernel information.
    include_xla_kernels: Whether to include XLA kernels in the output.

  Returns:
    A tuple of objects inheriting from `KernelInfoBase`.
  """
  if isinstance(x, (jax.stages.Lowered, ir.Module)):
    return _get_kernel_info_stablehlo(
        x, include_xla_kernels=include_xla_kernels
    )


def get_opspecs(
    x: HloComputation | KernelInfoBase,
    include_xla_kernels: bool = True,
) -> tuple[op_lib.BoundArguments, ...]:
  """Returns `BoundArguments` for all Tokamax ops in a given computation."""

  kernel_infos = (
      [x]
      if isinstance(x, KernelInfoBase)
      else get_kernel_info(x, include_xla_kernels)
  )

  op_specs = []
  for kernel in kernel_infos:
    json_data = hlo_utils_common.get_json_from_name(kernel.op_name)
    if json_data is None:
      continue

    op_specs.append(op_lib.BOUND_ARGS_ADAPTER.validate_json(json_data))

  return tuple(op_specs)


# Adapted from `GetNameFromLocImpl` in `mhlo_to_hlo/location_exporter.cc`.
def _get_op_name(loc: ir.Location) -> str:
  if isinstance(loc, ir.NameLoc):
    name = loc.name_str.split('@', maxsplit=1)[0]
    if name.endswith(':'):
      name = _get_op_name(loc.child_loc)
    return name
  if isinstance(loc, ir.CallSiteLoc):
    return _get_op_name(loc.callee)
  if isinstance(loc, ir.FusedLoc):
    return ';'.join(filter(bool, map(_get_op_name, loc.locations)))
  return ''


# Lifted from `jax._src.interpreters.mlir`.
_ALL_DTYPES: Final[tuple[jnp.dtype, ...]] = (
    jnp.dtype(jnp.bool_),
    jnp.dtype(jnp.int4),
    jnp.dtype(jnp.int8),
    jnp.dtype(jnp.int16),
    jnp.dtype(jnp.int32),
    jnp.dtype(jnp.int64),
    jnp.dtype(jnp.uint4),
    jnp.dtype(jnp.uint8),
    jnp.dtype(jnp.uint16),
    jnp.dtype(jnp.uint32),
    jnp.dtype(jnp.uint64),
    jnp.dtype(jnp.float8_e4m3b11fnuz),
    jnp.dtype(jnp.float8_e4m3fn),
    jnp.dtype(jnp.float8_e4m3fnuz),
    jnp.dtype(jnp.float8_e5m2),
    jnp.dtype(jnp.float8_e5m2fnuz),
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float16),
    jnp.dtype(jnp.float32),
    jnp.dtype(jnp.float64),
    jnp.dtype(jnp.complex64),
    jnp.dtype(jnp.complex128),
    jnp.dtype(jnp.int2),
    jnp.dtype(jnp.uint2),
    jnp.dtype(jnp.float8_e3m4),
    jnp.dtype(jnp.float8_e4m3),
    jnp.dtype(jnp.float8_e8m0fnu),
    jnp.dtype(jnp.float4_e2m1fn),
)


def _get_shape_dtype(ty: ir.Type) -> jax.ShapeDtypeStruct:
  ty = ir.ShapedType(ty)
  with ty.context:
    for dtype in _ALL_DTYPES:
      if mlir.dtype_to_ir_type(dtype) == ty.element_type:
        return jax.ShapeDtypeStruct(ty.shape, dtype)
  raise ValueError(f'Unknown type {ty}.')


def _get_source_file_line(loc: ir.Location) -> tuple[str, int]:
  """Returns the source file and line number of a location."""
  if isinstance(loc, ir.FileLineColLoc):
    return loc.filename, loc.start_line
  if isinstance(loc, ir.NameLoc):
    return _get_source_file_line(loc.child_loc)
  if isinstance(loc, ir.CallSiteLoc):
    return _get_source_file_line(loc.callee)
  if isinstance(loc, ir.FusedLoc):
    for inner_loc in reversed(loc.locations):
      file, line = _get_source_file_line(inner_loc)
      if file:
        return file, line
  return '', -1


def _get_common_kernel_info(
    op: ir.OpView, call_stack: tuple[str, ...]
) -> dict[str, Any]:
  """Extracts common kernel information from a `stablehlo` op."""
  source_file, source_line = _get_source_file_line(op.location)

  assert (parent := op.parent) is not None
  while (grandparent := parent.parent) is not None:
    parent = grandparent

  # Capture input / output layouts?
  return dict(
      name=op.name[len('stablehlo.') :],
      inputs=tuple(_get_shape_dtype(operand.type) for operand in op.operands),
      outputs=tuple(_get_shape_dtype(result.type) for result in op.results),
      op_name=';'.join(filter(bool, call_stack + (_get_op_name(op.location),))),
      source_line=source_line,
      source_file=source_file,
      hlo_module_name=parent.opview.sym_name.value,  # pytype: disable=attribute-error
  )


def _kernel_info_getter(cls):
  return lambda op, call_stack: cls(**_get_common_kernel_info(op, call_stack))


_KERNEL_GETTER_DICT = {
    hlo_utils_common.MOSAIC_GPU_KEY: _kernel_info_getter(
        hlo_utils_common.MosaicGpuKernelInfo
    ),
    hlo_utils_common.MOSAIC_TPU_KEY: _kernel_info_getter(
        hlo_utils_common.MosaicTpuKernelInfo
    ),
}
for _key in hlo_utils_common.PALLAS_TRITON_KEYS:
  _KERNEL_GETTER_DICT[_key] = _kernel_info_getter(
      hlo_utils_common.TritonKernelInfo
  )

_KERNEL_GETTER: Final[
    immutabledict.immutabledict[
        str,
        Callable[
            [stablehlo.CustomCallOp, tuple[str, ...]],
            KernelInfoBase,
        ],
    ]
] = immutabledict.immutabledict(_KERNEL_GETTER_DICT)
_get_tokamax_xla_kernel_info = _kernel_info_getter(
    hlo_utils_common.TokamaxXlaKernelInfo
)


def _get_kernel_info_stablehlo(
    x: jax.stages.Lowered | ir.Module,
    include_xla_kernels: bool = True,
) -> tuple[KernelInfoBase, ...]:
  """Extracts accelerator kernel information from a lowered JAX function.

  Args:
    x: The lowered JAX function from which to extract kernel information.
    include_xla_kernels: Whether to include XLA kernels in the output.

  Returns:
    A tuple of KernelInfoBase objects.
  """
  if isinstance(x, jax.stages.Lowered):
    x = hlo_utils_common.ir_module_from_lowered(x)

  symbol_table = ir.SymbolTable(x.operation)
  infos = []

  def handle_op(
      op: ir.Operation, call_stack: tuple[str, ...] = ()
  ) -> ir.WalkResult:
    op_ = op.opview

    if isinstance(op_, stablehlo.CustomCallOp):
      if (getter := _KERNEL_GETTER.get(op_.call_target_name.value)) is not None:  # pytype: disable=attribute-error
        infos.append(getter(op_, call_stack))
    elif isinstance(op_, func.CallOp):
      callee = symbol_table[op_.callee.value]  # pytype: disable=attribute-error
      call_stack = call_stack + (_get_op_name(op_.location),)
      callee.operation.walk(functools.partial(handle_op, call_stack=call_stack))
    elif isinstance(op_, func.FuncOp):
      if op_.name.value != 'main':  # pytype: disable=attribute-error
        return ir.WalkResult.SKIP
    elif (
        include_xla_kernels
        and isinstance(op_.name, str)  # `FuncOp` returns `StringAttr`.
        and op_.name.startswith('stablehlo.')
        and op_.name[len('stablehlo.') :]
        not in hlo_utils_common.XLA_NOISE_OPCODES
        and (
            any(hlo_utils_common.TOKAMAX_NAME in name for name in call_stack)
            or hlo_utils_common.TOKAMAX_NAME in _get_op_name(op_.location)
        )
    ):
      infos.append(_get_tokamax_xla_kernel_info(op_, call_stack))
    return ir.WalkResult.ADVANCE

  x.operation.walk(handle_op, ir.WalkOrder.PRE_ORDER)
  return tuple(infos)
