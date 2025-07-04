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
"""Utilities for extracting GPU kernels from HLO."""

from collections.abc import Sequence
import dataclasses
import json
import re
from typing import Any, Final, Literal
import zlib

from google.protobuf import json_format
import jax
from jax.interpreters.mlir import ir
from jax.jaxlib.gpu import triton_pb2
import jax.numpy as jnp
from tokamax._src import serialization
from tokamax._src.ops import op

from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import

_TRITON_PALLAS_KEY: Final[str] = '__gpu$xla.gpu.triton'
_MOSAIC_GPU_KEY: Final[str] = 'mosaic_gpu_v2'
_TRITON_KEY: Final[str] = 'triton_kernel_call'
_GPU_KERNEL_TARGETS: Final[tuple[str, str, str]] = (
    _TRITON_PALLAS_KEY,
    _MOSAIC_GPU_KEY,
    _TRITON_KEY,
)

_HLO_JAX_DTYPE_MAP = {
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
}

_PATTERN: Final[re.Pattern[str]] = re.compile(r'tokamax:([\w\.]+)\(({.+})\)')


@dataclasses.dataclass(frozen=True)
class GpuKernelInfo:
  """GPU kernel information."""

  name: str
  kernel_name: str

  op_name: str
  source_file: str
  source_line: int
  num_warps: int
  grid: tuple[int, int, int]
  inputs: tuple[jax.ShapeDtypeStruct, ...]
  outputs: tuple[jax.ShapeDtypeStruct, ...]
  hlo_module_name: str
  type: Literal['pallas_triton', 'triton', 'mosaic_gpu']
  num_stages: int | None = None
  cluster_dim: tuple[int, int, int] | None = None
  metadata: bytes | None = None
  compute_capability: int | None = (
      None  # Remove None once all platforms support this.
  )


def get_gpu_kernel_info_from_hlo(
    hlo: hlo_pb2.HloModuleProto | Sequence[hlo_pb2.HloModuleProto],
) -> tuple[GpuKernelInfo, ...]:
  """Extracts GPU kernel information from an HLO module.

  Args:
    hlo: The HLO proto or module proto to extract GPU kernels from.

  Returns:
    A list of dictionaries containing the GPU kernel information.
  """
  if isinstance(hlo, hlo_pb2.HloModuleProto):
    hlo = [hlo]

  out = []
  for proto in hlo:
    out.extend(_hlo_module_get_gpu_kernels(proto))
  return tuple(out)


def get_gpu_kernel_info_from_lowered(
    f: jax.stages.Lowered,
) -> tuple[GpuKernelInfo, ...]:
  """Extracts GPU kernel information from a lowered JITted function.

  Args:
    f: lowered JITted function.

  Returns:
    A list of dictionaries containing the GPU kernel information.
  """
  hlos = f.compile().runtime_executable().hlo_modules()
  hlos = [
      hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
      for hlo in hlos
  ]
  return get_gpu_kernel_info_from_hlo(hlos)


def _hlo_module_get_gpu_kernels(
    hlo: hlo_pb2.HloModuleProto,
) -> tuple[GpuKernelInfo, ...]:
  """Extracts Triton kernel information from an HLO module.

  Args:
    hlo: The HLO module to extract Triton kernels from.

  Returns:
    A list of dictionaries containing the Triton kernel information.
  """
  module_name = hlo.name
  out = []
  for instruction in _get_gpu_instructions(hlo):
    if instruction.custom_call_target == _TRITON_PALLAS_KEY:
      res = _instruction_get_pallas_kernel(instruction, module_name=module_name)
    elif instruction.custom_call_target == _MOSAIC_GPU_KEY:
      res = _instruction_get_mosaic_kernel(instruction, module_name=module_name)
    elif instruction.custom_call_target == _TRITON_KEY:
      res = _instruction_get_triton_kernel(instruction, module_name=module_name)
    else:
      raise ValueError(
          f'Unsupported custom call target: {instruction.custom_call_target}'
      )
    out.append(res)

  return tuple(out)


def _instruction_get_pallas_kernel(
    instruction, module_name: str
) -> GpuKernelInfo:
  """Get Pallas kernel info from an HLO instruction."""

  mlir_ctx = ir.Context()

  def parse_ctx(name):
    backend_config = instruction.backend_config.decode('utf-8')
    return ir.DictAttr.parse(backend_config, mlir_ctx)[name].value

  in_shapes = _parse_shapes(instruction.operand_shapes_with_layout)
  out_shapes = _parse_shapes(instruction.shape)
  grid = (
      parse_ctx('grid_x'),
      parse_ctx('grid_y'),
      parse_ctx('grid_z'),
  )
  return GpuKernelInfo(
      name=instruction.name,
      kernel_name=parse_ctx('name'),
      source_file=instruction.metadata.source_file,
      source_line=instruction.metadata.source_line,
      op_name=instruction.metadata.op_name,
      num_warps=parse_ctx('num_warps'),
      num_stages=parse_ctx('num_stages'),
      grid=grid,
      inputs=in_shapes,
      outputs=out_shapes,
      hlo_module_name=module_name,
      type='pallas_triton',
  )


def _instruction_get_mosaic_kernel(
    instruction, module_name: str
) -> GpuKernelInfo:
  """Get Mosaic GPU kernel info from an HLO instruction."""

  in_shapes = _parse_shapes(instruction.operand_shapes_with_layout)
  out_shapes = _parse_shapes(instruction.shape)

  return GpuKernelInfo(
      name=instruction.name,
      kernel_name='',  # TODO(sbodenstein): add support for kernel_name.
      source_file=instruction.metadata.source_file,
      source_line=instruction.metadata.source_line,
      op_name=instruction.metadata.op_name,
      num_warps=1,  # TODO(sbodenstein): add support for num_warps.
      grid=(1, 1, 1),  # TODO(sbodenstein): add support for grid.
      inputs=in_shapes,
      outputs=out_shapes,
      hlo_module_name=module_name,
      type='mosaic_gpu',
  )


def _instruction_get_triton_kernel(
    instruction, module_name: str
) -> GpuKernelInfo:
  """Get Triton kernel info from an HLO instruction."""

  cfg = instruction.backend_config
  temp = zlib.decompress(cfg)
  proto = triton_pb2.TritonAnyKernelCall.FromString(temp)
  in_shapes = _parse_shapes(instruction.operand_shapes_with_layout)
  out_shapes = _parse_shapes(instruction.shape)

  grid = (
      proto.kernel_call.grid_0,
      proto.kernel_call.grid_1,
      proto.kernel_call.grid_2,
  )
  kernel = proto.kernel_call.kernel

  return GpuKernelInfo(
      name=instruction.name,
      kernel_name=kernel.kernel_name,
      compute_capability=kernel.compute_capability,
      source_file=instruction.metadata.source_file,
      source_line=instruction.metadata.source_line,
      op_name=instruction.metadata.op_name,
      num_warps=kernel.num_warps,
      num_stages=None,  # This is not currently in triton.proto.
      grid=grid,
      inputs=in_shapes,
      outputs=out_shapes,
      cluster_dim=(
          kernel.cluster_dim_0,
          kernel.cluster_dim_1,
          kernel.cluster_dim_2,
      ),
      metadata=proto.metadata,
      hlo_module_name=module_name,
      type='triton',
  )


def get_opspecs(
    x: (
        jax.stages.Lowered
        | hlo_pb2.HloModuleProto
        | Sequence[hlo_pb2.HloModuleProto]
    ),
) -> tuple[op.BoundArguments, ...]:
  """Get Tokamax ops from either a lowered jax function or HLO modules.

  Returns a tuple of BoundArguments.
  """
  # TODO(stzeng): Currently does not work for XLA ops. Add support for XLA op specs.

  if isinstance(x, jax.stages.Lowered):
    gpu_kernels = get_gpu_kernel_info_from_lowered(x)
  else:
    gpu_kernels = get_gpu_kernel_info_from_hlo(x)

  op_specs = []
  for kernel in gpu_kernels:
    if 'tokamax' not in kernel.op_name:
      continue
    match = _PATTERN.search(kernel.op_name)
    if match is None:
      raise ValueError(
          f'Could not extract op name and json str from  {kernel.op_name}.'
      )
    op_name, json_str = match.groups()
    op_class = serialization.get_module_member(op_name)
    spec = json.loads(json_str, cls=serialization.JsonDecoder)
    op_spec = op.BoundArguments(op=op_class(), arguments=spec)
    op_specs.append(op_spec)

  return tuple(op_specs)


def _parse_shapes(shapes) -> tuple[jax.ShapeDtypeStruct, ...]:
  out = _parse_shapes_recursive(shapes)
  return tuple(jax.tree.leaves(out))


def _parse_shapes_recursive(shapes):
  """Parse xla.ShapeProto."""

  # Ideally use isinstance on the type, but this type is not visible.
  if isinstance(shapes, list) or 'RepeatedCompositeContainer' in str(
      type(shapes)
  ):
    return tuple([_parse_shapes(shape) for shape in shapes])  # pytype: disable=attribute-error

  # Ideally use isinstance on the type, but this type is not visible.
  elif 'ShapeProto' in str(type(shapes)):
    shapes = json_format.MessageToDict(shapes)
    if shapes['elementType'] == 'TUPLE':
      return tuple(_process_shape(shape) for shape in shapes['tupleShapes'])
    else:
      return _process_shape(shapes)
  else:
    raise ValueError(f'Unsupported shape type {type(shapes)}')


def _process_shape(shape: dict[str, Any]) -> jax.ShapeDtypeStruct:
  if shape['elementType'] not in _HLO_JAX_DTYPE_MAP:
    raise ValueError(f'Unsupported element type: {shape["elementType"]}')
  dtype = _HLO_JAX_DTYPE_MAP[shape['elementType']]
  shape = tuple([int(i) for i in shape['dimensions']])
  return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)


def _has_custom_gpu_kernel(
    hlo_instruction_proto: hlo_pb2.HloInstructionProto,
) -> bool:

  return (
      hlo_instruction_proto.name.startswith('custom-call')
      and hlo_instruction_proto.custom_call_target in _GPU_KERNEL_TARGETS
  )


def _get_gpu_instructions(hlo_module_proto):
  instructions = []
  for computation in hlo_module_proto.computations:
    instructions.extend(computation.instructions)
  return list(filter(_has_custom_gpu_kernel, instructions))
