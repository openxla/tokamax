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
from collections.abc import Callable, Sequence
import dataclasses
import re
from typing import Any, Final
import zlib

from google.protobuf import json_format
import immutabledict
import jax
from jax import export
from jax.interpreters.mlir import ir
import jax.numpy as jnp
from tokamax._src.ops import op as op_base

from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import

_TRITON_PALLAS_KEY: Final[str] = '__gpu$xla.gpu.triton'
_MOSAIC_GPU_KEY: Final[str] = 'mosaic_gpu_v2'
_MOSAIC_TPU_KEY: Final[str] = 'tpu_custom_call'
_TRITON_KEY: Final[str] = 'triton_kernel_call'

DISABLE_JAX_EXPORT_CHECKS: Final[tuple[export.DisabledSafetyCheck, ...]] = (
    export.DisabledSafetyCheck.custom_call(_TRITON_PALLAS_KEY),
    export.DisabledSafetyCheck.custom_call(_MOSAIC_GPU_KEY),
    export.DisabledSafetyCheck.custom_call(_MOSAIC_TPU_KEY),
    export.DisabledSafetyCheck.custom_call(_TRITON_KEY),
)

_HLO_JAX_DTYPE_MAP: Final[immutabledict.immutabledict[str, type(Any)]] = (
    immutabledict.immutabledict({
        # Predicates are two-state booleans.
        'PRED': jnp.bool_,
        # Signed integral values of fixed width.
        'S4': jnp.int4,
        'S8': jnp.int8,
        'S16': jnp.int16,
        'S32': jnp.int32,
        'S64': jnp.int64,
        # Unsigned integral values of fixed width.
        'U8': jnp.uint8,
        'U16': jnp.uint16,
        'U32': jnp.uint32,
        'U64': jnp.uint64,
        # Floating-point values of fixed width.
        'BF16': jnp.bfloat16,
        'F16': jnp.float16,
        'F32': jnp.float32,
        'F64': jnp.float64,
    })
)

_XLA_NOISE_OPCODES: Final[set[str]] = {
    'parameter',
    'get-tuple-element',
    'broadcast',
    'reduce',
    'bitcast',
}

_TOKAMAX_NAME: Final[str] = 'tokamax'


@dataclasses.dataclass(frozen=True)
class KernelInfoBase:
  """Kernel information base class."""

  name: str
  inputs: tuple[jax.ShapeDtypeStruct, ...]
  outputs: tuple[jax.ShapeDtypeStruct, ...]
  op_name: str
  source_file: str
  source_line: int
  hlo_module_name: str


@dataclasses.dataclass(frozen=True)
class TritonKernelInfo(KernelInfoBase):
  """Triton kernel information."""

  kernel_name: str
  num_warps: int
  grid: tuple[int, int, int]
  num_stages: int | None
  cluster_dim: tuple[int, int, int] | None
  compute_capability: int | None
  metadata: bytes | None


# TODO: Add fields for Mosaic TPU kernel information.
@dataclasses.dataclass(frozen=True)
class MosaicTpuKernelInfo(KernelInfoBase):
  """Mosaic TPU kernel information."""


@dataclasses.dataclass(frozen=True)
class MosaicGpuKernelInfo(KernelInfoBase):
  """Mosaic GPU kernel information."""


@dataclasses.dataclass(frozen=True)
class TokamaxXlaKernelInfo(KernelInfoBase):
  """Tokamax XLA kernel information."""


def _is_tokamax_kernel(
    kernel: hlo_pb2.HloInstructionProto | KernelInfoBase,
) -> bool:
  """Returns True if a kernel is a Tokamax kernel."""
  if isinstance(kernel, hlo_pb2.HloInstructionProto):
    op_name = kernel.metadata.op_name
  elif isinstance(kernel, KernelInfoBase):
    op_name = kernel.op_name
  else:
    raise ValueError(f'Unsupported kernel type {type(kernel)}')

  return _TOKAMAX_NAME in op_name


def _get_generic_kernel_info(
    instruction: hlo_pb2.HloInstructionProto,
) -> dict[str, Any]:

  return dict(
      name=instruction.name,
      source_line=instruction.metadata.source_line,
      source_file=instruction.metadata.source_file,
      op_name=instruction.metadata.op_name,
      inputs=_parse_shapes(instruction.operand_shapes_with_layout),
      outputs=_parse_shapes(instruction.shape),
  )


def _instruction_get_pallas_kernel(
    instruction: hlo_pb2.HloInstructionProto, module_name: str
) -> TritonKernelInfo:
  """Get Pallas kernel info from an HLO instruction."""

  mlir_ctx = ir.Context()

  def parse_ctx(name):
    backend_config = instruction.backend_config.decode('utf-8')
    return ir.DictAttr.parse(backend_config, mlir_ctx)[name].value

  grid = (
      parse_ctx('grid_x'),
      parse_ctx('grid_y'),
      parse_ctx('grid_z'),
  )
  return TritonKernelInfo(
      **_get_generic_kernel_info(instruction),
      kernel_name=parse_ctx('name'),
      num_warps=parse_ctx('num_warps'),
      num_stages=parse_ctx('num_stages'),
      grid=grid,
      hlo_module_name=module_name,
      compute_capability=None,
      cluster_dim=None,
      metadata=None,
  )


def _instruction_get_mosaic_gpu_kernel(
    instruction: hlo_pb2.HloInstructionProto, module_name: str
) -> MosaicGpuKernelInfo:
  """Get Mosaic GPU kernel info from an HLO instruction."""

  return MosaicGpuKernelInfo(
      **_get_generic_kernel_info(instruction),
      hlo_module_name=module_name,
  )


def _instruction_get_mosaic_tpu_kernel(
    instruction: hlo_pb2.HloInstructionProto, module_name: str
) -> MosaicTpuKernelInfo:
  """Get Mosaic GPU kernel info from an HLO instruction."""

  return MosaicTpuKernelInfo(
      **_get_generic_kernel_info(instruction),
      hlo_module_name=module_name,
  )


def _instruction_get_triton_kernel(
    instruction: hlo_pb2.HloInstructionProto, module_name: str
) -> TritonKernelInfo:
  """Get Triton kernel info from an HLO instruction."""

  cfg = instruction.backend_config
  grid = (0, 0, 0)
  name = ''
  compute_capability = 0
  num_warps = 0
  cluster_dim = (0, 0, 0)
  metadata = ''

  return TritonKernelInfo(
      **_get_generic_kernel_info(instruction),
      kernel_name=name,
      compute_capability=compute_capability,
      num_warps=num_warps,
      num_stages=None,  # This is not currently in triton.proto.
      grid=grid,
      cluster_dim=cluster_dim,
      metadata=metadata,
      hlo_module_name=module_name,
  )


_KERNEL_GETTER: Final[
    immutabledict.immutabledict[
        str, Callable[[hlo_pb2.HloInstructionProto, str], KernelInfoBase]
    ]
] = immutabledict.immutabledict({
    _MOSAIC_GPU_KEY: _instruction_get_mosaic_gpu_kernel,
    _MOSAIC_TPU_KEY: _instruction_get_mosaic_tpu_kernel,
    _TRITON_PALLAS_KEY: _instruction_get_pallas_kernel,
    _TRITON_KEY: _instruction_get_triton_kernel,
})


def _get_kernel_from_instruction(
    instruction: hlo_pb2.HloInstructionProto, module_name: str
) -> KernelInfoBase:
  target = getattr(instruction, 'custom_call_target', None)
  if (getter := _KERNEL_GETTER.get(target)) is not None:
    return getter(instruction, module_name)
  else:
    return TokamaxXlaKernelInfo(
        **_get_generic_kernel_info(instruction),
        hlo_module_name=module_name,
    )


def get_kernel_info(
    x: (
        hlo_pb2.HloModuleProto
        | Sequence[hlo_pb2.HloModuleProto]
        | jax.stages.Lowered
    ),
    include_xla_kernels: bool = True,
) -> tuple[KernelInfoBase, ...]:
  """Extracts accelerator kernel information from an HLO module.

  Args:
    x: The HLO proto or module proto or JAX lowered function to extract kernels
      from.
    include_xla_kernels: Whether to include XLA kernels in the output.

  Returns:
    A tuple of KernelInfoBase objects.
  """

  if isinstance(x, jax.stages.Lowered):
    out = _get_kernel_info_from_lowered(x)
  else:
    out = _get_kernel_info_from_hlo(x)

  if include_xla_kernels:
    return out
  else:
    return tuple(
        kernel for kernel in out if not isinstance(kernel, TokamaxXlaKernelInfo)
    )


def _get_kernel_info_from_hlo(
    hlo: hlo_pb2.HloModuleProto | Sequence[hlo_pb2.HloModuleProto],
) -> tuple[KernelInfoBase, ...]:
  """Extracts accelerator kernel information from an HLO module."""

  hlos = [hlo] if isinstance(hlo, hlo_pb2.HloModuleProto) else hlo
  out = []

  for hlo in hlos:
    module_name = hlo.name
    for instruction in _get_instructions(hlo):
      out.append(_get_kernel_from_instruction(instruction, module_name))

  return tuple(out)


def _get_kernel_info_from_lowered(
    f: jax.stages.Lowered,
) -> tuple[KernelInfoBase, ...]:
  """Extracts accelerator kernel information from a lowered JITted function."""
  hlos = f.compile().runtime_executable().hlo_modules()
  # TODO: Figure out how to obtain this without serializing and
  # deserializing.
  hlos = [
      hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
      for hlo in hlos
  ]
  return _get_kernel_info_from_hlo(hlos)


def get_opspecs(  # pytype: disable=invalid-annotation
    x: (
        jax.stages.Lowered
        | hlo_pb2.HloModuleProto
        | Sequence[hlo_pb2.HloModuleProto],
    ),
    include_xla_kernels: bool = True,
) -> tuple[op_base.BoundArguments, ...]:
  """Returns a tuple of BoundArguments for all Tokamax ops in the HLO."""

  kernels = get_kernel_info(x, include_xla_kernels=include_xla_kernels)
  op_specs = []
  for kernel in kernels:
    if not _is_tokamax_kernel(kernel):
      continue
    marker = _TOKAMAX_NAME + ':'
    idx = kernel.op_name.find(marker)
    # For XLA kernels, sometimes the op info is not present, eg.
    # jit(tokamax_norm_and_glu)/convert_element_type.
    if idx == -1:
      continue
    json_data = kernel.op_name[idx + len(marker):]
    count = 0
    # A VJP op may have multiple op specs in the HLO. Find the position of the
    # end brace for the first op spec. We only return the first op (the VJP), as
    # the forward op will be present in the HLO elsewhere.
    for i, c in enumerate(json_data):
      if c == '{':
        count += 1
      if c == '}':
        count -= 1
        if count < 1:
          # This might mean that we have more end braces than opening braces,
          # but in that case the `validate_json` call below will fail.
          json_data = json_data[:i + 1]
          break
    op_specs.append(op_base.BOUND_ARGS_ADAPTER.validate_json(json_data))

  return tuple(op_specs)


def _parse_shapes(shapes) -> tuple[jax.ShapeDtypeStruct, ...]:
  out = _parse_shapes_recursive(shapes)
  return tuple(jax.tree.leaves(out))


# TODO: HLO Protos for TPU have a different format for its ShapeProto with different attributes.Fix this for TPU HLOs in a separate update.
def _parse_shapes_recursive(shapes):
  """Parse xla.ShapeProto."""

  # Ideally use isinstance on the type, but this type is not visible.
  if (
      isinstance(shapes, list)
      or 'RepeatedCompositeContainer' in str(type(shapes))
      or 'RepeatedCompositeFieldContainer' in str(type(shapes))
  ):
    return tuple([_parse_shapes(shape) for shape in shapes])  # pytype: disable=attribute-error

  # Ideally use isinstance on the type, but this type is not visible.
  elif 'ShapeProto' in str(type(shapes)):
    shapes = json_format.MessageToDict(shapes)
    if shapes['elementType'] == 'TUPLE' and 'tupleShapes' in shapes:
      return tuple(_process_shape(shape) for shape in shapes['tupleShapes'])
    else:
      return _process_shape(shapes)
  else:
    raise ValueError(f'Unsupported shape type {type(shapes)}')


def _process_shape(shape: dict[str, Any]) -> jax.ShapeDtypeStruct:
  if shape['elementType'] not in _HLO_JAX_DTYPE_MAP:
    raise ValueError(f'Unsupported element type: {shape["elementType"]}')
  dtype = _HLO_JAX_DTYPE_MAP[shape['elementType']]
  if 'dimensions' not in shape:
    # If there are no dimensions, the shape is a scalar.
    return jax.ShapeDtypeStruct(shape=(), dtype=dtype)
  shape = tuple([int(i) for i in shape['dimensions']])
  return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)


def _is_tokamax_xla_hlo(
    hlo_instruction_proto: hlo_pb2.HloInstructionProto,
) -> bool:
  """Whether the HLO instruction is a Tokamax XLA op."""

  if hlo_instruction_proto.opcode in _XLA_NOISE_OPCODES:
    return False

  try:
    # This can raise a ValueError if there is no 'dimensions' field. Filter
    # out such ops.
    _parse_shapes(hlo_instruction_proto.shape)
    return _is_tokamax_kernel(hlo_instruction_proto)
  except ValueError:
    return False


def _collect_hlo(hlo_instruction_proto: hlo_pb2.HloInstructionProto) -> bool:
  """Whether to collect the HLO instruction."""

  return (
      hlo_instruction_proto.custom_call_target in _KERNEL_GETTER
      or _is_tokamax_xla_hlo(hlo_instruction_proto)
  )


def _get_instructions(hlo_module_proto):
  instructions = []
  for computation in hlo_module_proto.computations:
    instructions.extend(computation.instructions)
  return list(filter(_collect_hlo, instructions))
