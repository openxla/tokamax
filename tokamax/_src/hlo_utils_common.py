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

"""Common utilities for HLO utils."""

import dataclasses
from typing import Final, cast
import jax
from jaxlib.mlir import ir

TOKAMAX_NAME: Final[str] = 'tokamax'

PALLAS_TRITON_KEY: Final[str] = '__gpu$xla.gpu.triton'
MOSAIC_GPU_KEY: Final[str] = 'mosaic_gpu_v2'
MOSAIC_TPU_KEY: Final[str] = 'tpu_custom_call'
TRITON_KEY: Final[str] = 'triton_kernel_call'
TRITON_FFI_KEY: Final[str] = 'triton_kernel_call_ffi'


XLA_NOISE_OPCODES: Final[set[str]] = {
    'concatenate',
    'constant',
    'convert',
    'broadcast',
    'broadcast_in_dim',
    'reduce',
    'reshape',
    'slice',
    'transpose',
    'parameter',
    'get-tuple-element',
    'bitcast',
}


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


def get_json_from_name(op_name: str) -> str | None:
  """Returns the JSON data from the op name."""
  marker = TOKAMAX_NAME + ':'
  idx = op_name.find(marker)
  # For XLA kernels, sometimes the op info is not present, eg.
  # jit(tokamax_norm_and_glu)/convert_element_type.
  if idx == -1:
    return None
  json_data = op_name[idx + len(marker) :]
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
  return json_data


def ir_module_from_lowered(
    lowered: jax.stages.Lowered,
) -> ir.Module:
  """Returns an `ir.Module` from a lowered JAX function."""
  assert (module := lowered.compiler_ir('stablehlo')) is not None
  return cast(ir.Module, module)
