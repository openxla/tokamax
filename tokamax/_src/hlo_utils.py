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
from typing import Any, Final, TypeAlias
import zlib

import immutabledict
import jax
from jax import export
from jax.interpreters.mlir import ir
import jax.numpy as jnp
from tokamax._src.ops import op as op_lib

from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import

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


XlaShape: TypeAlias = jax.ShapeDtypeStruct | tuple['XlaShape', ...]


@dataclasses.dataclass(frozen=True, slots=True)
class KernelInfoBase:
  """Kernel information base class."""

  name: str
  inputs: tuple[XlaShape, ...]
  output: XlaShape
  op_name: str
  source_file: str
  source_line: int
  hlo_module_name: str


@dataclasses.dataclass(frozen=True, slots=True)
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


def _parse_shape(shape) -> XlaShape:
  elem_type_enum = shape.DESCRIPTOR.fields_by_name['element_type'].enum_type
  elem_type = elem_type_enum.values_by_number[shape.element_type].name
  if elem_type == 'TUPLE':
    return tuple(map(_parse_shape, shape.tuple_shapes))
  return jax.ShapeDtypeStruct(shape.dimensions, _HLO_JAX_DTYPE_MAP[elem_type])


def _get_generic_kernel_info(
    instruction: hlo_pb2.HloInstructionProto,
) -> dict[str, Any]:

  return dict(
      name=instruction.name,
      source_line=instruction.metadata.source_line,
      source_file=instruction.metadata.source_file,
      op_name=instruction.metadata.op_name,
      inputs=tuple(map(_parse_shape, instruction.operand_shapes_with_layout)),
      output=_parse_shape(instruction.shape),
  )


def _get_pallas_kernel_info(
    instruction: hlo_pb2.HloInstructionProto, module_name: str
) -> TritonKernelInfo:
  """Get Pallas kernel info from an HLO instruction."""
  ctx = ir.Context()
  config = ir.DictAttr.parse(instruction.backend_config.decode('utf-8'), ctx)
  return TritonKernelInfo(
      **_get_generic_kernel_info(instruction),
      kernel_name=config['name'].value,
      num_warps=config['num_warps'].value,
      num_stages=config['num_stages'].value,
      grid=tuple(config[f'grid_{dim}'].value for dim in ('x', 'y', 'z')),
      hlo_module_name=module_name,
      compute_capability=None,
      metadata=b'',
  )


def _kernel_info_getter(cls):
  return lambda i, m: cls(**_get_generic_kernel_info(i), hlo_module_name=m)


_KERNEL_GETTER: Final[
    immutabledict.immutabledict[
        str, Callable[[hlo_pb2.HloInstructionProto, str], KernelInfoBase]
    ]
] = immutabledict.immutabledict({
    _MOSAIC_GPU_KEY: _kernel_info_getter(MosaicGpuKernelInfo),
    _MOSAIC_TPU_KEY: _kernel_info_getter(MosaicTpuKernelInfo),
    _PALLAS_TRITON_KEY: _get_pallas_kernel_info,
})
_get_tokamax_xla_kernel_info = _kernel_info_getter(TokamaxXlaKernelInfo)


def get_kernel_info(
    x: (
        jax.stages.Lowered
        | hlo_pb2.HloModuleProto
        | Sequence[hlo_pb2.HloModuleProto]
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
    # TODO: Figure out how to obtain this without serializing and
    # deserializing.
    hlos = [
        hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
        for hlo in x.compile().runtime_executable().hlo_modules()
    ]
  elif isinstance(x, hlo_pb2.HloModuleProto):
    hlos = [x]
  else:
    hlos = x

  infos = []
  for hlo in hlos:
    for computation in hlo.computations:
      for instruction in computation.instructions:
        target = getattr(instruction, 'custom_call_target', None)
        if (getter := _KERNEL_GETTER.get(target)) is not None:
          infos.append(getter(instruction, hlo.name))
        elif include_xla_kernels:
          is_noise = instruction.opcode in _XLA_NOISE_OPCODES
          if (not is_noise) and _TOKAMAX_NAME in instruction.metadata.op_name:
            infos.append(_get_tokamax_xla_kernel_info(instruction, hlo.name))

  return tuple(infos)


def get_opspecs(  # pytype: disable=invalid-annotation
    x: (
        jax.stages.Lowered
        | hlo_pb2.HloModuleProto
        | Sequence[hlo_pb2.HloModuleProto],
    ),
    include_xla_kernels: bool = True,
) -> tuple[op_lib.BoundArguments, ...]:
  """Returns a tuple of BoundArguments for all Tokamax ops in the HLO."""

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
      if c == '}':
        count -= 1
        if count < 1:
          # This might mean that we have more end braces than opening braces,
          # but in that case the `validate_json` call below will fail.
          json_data = json_data[: i + 1]
          break
    op_specs.append(op_lib.BOUND_ARGS_ADAPTER.validate_json(json_data))

  return tuple(op_specs)
