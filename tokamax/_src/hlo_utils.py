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
"""Utilities for extracting kernel information from HLO."""
from collections.abc import Callable
import dataclasses
import functools
from typing import Any, Final
import zlib

import immutabledict
import jax
from jax import export
from jax.extend import mlir as jex_mlir
from jax.interpreters import mlir
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import stablehlo
from tokamax._src.ops import op as op_lib

_PALLAS_TRITON_KEY: Final[str] = '__gpu$xla.gpu.triton'
_MOSAIC_GPU_KEY: Final[str] = 'mosaic_gpu_v2'
_MOSAIC_TPU_KEY: Final[str] = 'tpu_custom_call'
_TRITON_KEY: Final[str] = 'triton_kernel_call'

DISABLE_JAX_EXPORT_CHECKS: Final[tuple[export.DisabledSafetyCheck, ...]] = (
    export.DisabledSafetyCheck.custom_call(_PALLAS_TRITON_KEY),
    export.DisabledSafetyCheck.custom_call(_MOSAIC_GPU_KEY),
    export.DisabledSafetyCheck.custom_call(_MOSAIC_TPU_KEY),
    export.DisabledSafetyCheck.custom_call(_TRITON_KEY),
)

_XLA_NOISE_OPCODES: Final[set[str]] = {
    'concatenate',
    'constant',
    'convert',
    'broadcast',
    'broadcast_in_dim',
    'reduce',
    'reshape',
    'slice',
    'transpose',
}
_TOKAMAX_NAME: Final[str] = 'tokamax'


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class KernelInfoBase:
  """Kernel information base class."""

  name: str
  inputs: tuple[jax.ShapeDtypeStruct, ...]
  outputs: tuple[jax.ShapeDtypeStruct, ...]
  op_name: str
  source_file: str
  source_line: int
  hlo_module_name: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TritonKernelInfo(KernelInfoBase):
  """Triton kernel information."""

  kernel_name: str
  num_warps: int
  grid: tuple[int, int, int]
  num_stages: int | None
  compute_capability: int | None
  metadata: bytes


# TODO: Add fields for Mosaic TPU kernel information.
@dataclasses.dataclass(frozen=True, slots=True)
class MosaicTpuKernelInfo(KernelInfoBase):
  """Mosaic TPU kernel information."""


@dataclasses.dataclass(frozen=True, slots=True)
class MosaicGpuKernelInfo(KernelInfoBase):
  """Mosaic GPU kernel information."""


@dataclasses.dataclass(frozen=True, slots=True)
class TokamaxXlaKernelInfo(KernelInfoBase):
  """Tokamax XLA kernel information."""


# Adapted from `GetNameFromLocImpl` in `mhlo_to_hlo/location_exporter.cc`.
def _get_op_name(loc: ir.Location) -> str:
  if loc.is_a_name():
    name = loc.name_str.split('@', maxsplit=1)[0]
    if name.endswith(':'):
      name = _get_op_name(loc.child_loc)
    return name
  if loc.is_a_callsite():
    return _get_op_name(loc.callee)
  if loc.is_a_fused():
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
  if loc.is_a_file():
    return loc.filename, loc.start_line
  if loc.is_a_name():
    return _get_source_file_line(loc.child_loc)
  if loc.is_a_callsite():
    return _get_source_file_line(loc.callee)
  if loc.is_a_fused():
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

  parent = op.parent
  while parent.parent is not None:
    parent = parent.parent

  # Capture input / output layouts?
  return dict(
      name=op.name[len('stablehlo.') :],
      inputs=tuple(_get_shape_dtype(operand.type) for operand in op.operands),
      outputs=tuple(_get_shape_dtype(result.type) for result in op.results),
      op_name=';'.join(filter(bool, call_stack + (_get_op_name(op.location),))),
      source_line=source_line,
      source_file=source_file,
      hlo_module_name=parent.opview.sym_name.value,
  )


def _get_pallas_kernel_info(
    op: stablehlo.CustomCallOp, call_stack: tuple[str, ...]
) -> TritonKernelInfo:
  """Get Pallas kernel info from a `stablehlo.CustomCallOp`."""
  if (config := op.backend_config) is None:
    config = op.attributes['mhlo.backend_config']
  elif isinstance(config, ir.StringAttr):
    assert not ir.StringAttr(config).value
    config = op.attributes['mhlo.backend_config']
  return TritonKernelInfo(
      **_get_common_kernel_info(op, call_stack),
      kernel_name=config['name'].value,
      num_warps=config['num_warps'].value,
      num_stages=config['num_stages'].value,
      grid=tuple(config[f'grid_{dim}'].value for dim in ('x', 'y', 'z')),
      compute_capability=None,
      metadata=b'',
  )


def _kernel_info_getter(cls):
  return lambda op, call_stack: cls(**_get_common_kernel_info(op, call_stack))


_KERNEL_GETTER: Final[
    immutabledict.immutabledict[
        str, Callable[[stablehlo.CustomCallOp, tuple[str, ...]], KernelInfoBase]
    ]
] = immutabledict.immutabledict({
    _MOSAIC_GPU_KEY: _kernel_info_getter(MosaicGpuKernelInfo),
    _MOSAIC_TPU_KEY: _kernel_info_getter(MosaicTpuKernelInfo),
    _PALLAS_TRITON_KEY: _get_pallas_kernel_info,
})
_get_tokamax_xla_kernel_info = _kernel_info_getter(TokamaxXlaKernelInfo)


def get_kernel_info(
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
    x = x.compiler_ir('stablehlo')

  symbol_table = ir.SymbolTable(x.operation)
  infos = []

  def handle_op(
      op: ir.Operation, call_stack: tuple[str, ...] = ()
  ) -> ir.WalkResult:
    op = op.opview

    if isinstance(op, stablehlo.CustomCallOp):
      if (getter := _KERNEL_GETTER.get(op.call_target_name.value)) is not None:
        infos.append(getter(op, call_stack))
    elif isinstance(op, func.CallOp):
      callee = symbol_table[op.callee.value]
      call_stack = call_stack + (_get_op_name(op.location),)
      callee.operation.walk(functools.partial(handle_op, call_stack=call_stack))
    elif isinstance(op, func.FuncOp):
      if op.name.value != 'main':
        return ir.WalkResult.SKIP
    elif (
        include_xla_kernels
        and isinstance(op.name, str)  # `FuncOp` returns `StringAttr`.
        and op.name.startswith('stablehlo.')
        and op.name[len('stablehlo.') :] not in _XLA_NOISE_OPCODES
        and (
            any(_TOKAMAX_NAME in name for name in call_stack)
            or _TOKAMAX_NAME in _get_op_name(op.location)
        )
    ):
      infos.append(_get_tokamax_xla_kernel_info(op, call_stack))
    return ir.WalkResult.ADVANCE

  x.operation.walk(handle_op, ir.WalkOrder.PRE_ORDER)
  return tuple(infos)


def get_opspecs(
    x: jax.stages.Lowered | ir.Module,
    include_xla_kernels: bool = True,
) -> tuple[op_lib.BoundArguments, ...]:
  """Returns `BoundArguments` for all Tokamax ops in a lowered JAX function."""

  op_specs = []
  for kernel in get_kernel_info(x, include_xla_kernels=include_xla_kernels):
    marker = _TOKAMAX_NAME + ':'
    idx = kernel.op_name.find(marker)
    # For XLA kernels, sometimes the op info is not present, eg.
    # jit(tokamax_norm_and_glu)/convert_element_type.
    if idx == -1:
      continue
    json_data = kernel.op_name[idx + len(marker) :]
    count = 0
    # A VJP op may have multiple op specs in the HLO. Find the position of the
    # end brace for the first op spec. We only return the first op (the VJP), as
    # the forward op will be present in the HLO elsewhere.
    for i, c in enumerate(json_data):
      if c == '{':
        count += 1
      elif c == '}':
        count -= 1
        if count < 1:
          # This might mean that we have more end braces than opening braces,
          # but in that case the `validate_json` call below will fail.
          json_data = json_data[: i + 1]
          break
    op_specs.append(op_lib.BOUND_ARGS_ADAPTER.validate_json(json_data))

  return tuple(op_specs)
