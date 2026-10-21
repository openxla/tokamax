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
from typing import Callable, Final, Iterable, cast
import jax
from jaxlib.mlir import ir

TOKAMAX_NAME: Final[str] = 'tokamax'

MOSAIC_GPU_KEY: Final[str] = 'mosaic_gpu_v2'
MOSAIC_TPU_KEY: Final[str] = 'tpu_custom_call'
# Exposed in jax_triton, but we don't want a dependency on it here. So the
# equivalence is tested against.
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
  # TODO: Remove `None` once the migration is complete.
  metadata_payload: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TritonKernelInfo(KernelInfoBase):
  """Triton kernel information."""


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


def get_jsons_from_name(op_name: str) -> list[str]:
  """Returns all JSON data payloads from the op name in order."""
  marker = TOKAMAX_NAME + ':'
  results = []
  pos = 0
  while (idx := op_name.find(marker, pos)) != -1:
    json_data = op_name[idx + len(marker) :]
    count = 0
    matched = False
    for i, c in enumerate(json_data):
      if c == '{':
        count += 1
      elif c == '}':
        count -= 1
        if count < 1:
          results.append(json_data[: i + 1])
          pos = idx + len(marker) + i + 1
          matched = True
          break
    if not matched:
      break
  return results


def get_json_from_name(op_name: str) -> str | None:
  """Returns the first JSON data from the op name."""
  jsons = get_jsons_from_name(op_name)
  return jsons[0] if jsons else None


def ir_module_from_lowered(
    lowered: jax.stages.Lowered,
) -> ir.Module:
  """Returns an `ir.Module` from a lowered JAX function."""
  assert (module := lowered.compiler_ir('stablehlo')) is not None
  return cast(ir.Module, module)


@dataclasses.dataclass(frozen=True)
class Record:
  emit_fn: Callable[[], KernelInfoBase]
  is_noise: bool
  payload: str | None


def dedupe_wrapper_kernels(
    records: Iterable[Record],
) -> tuple[KernelInfoBase, ...]:
  """Deduplicates wrapper operations based on their metadata payload.

  Always returns non-noise ops. For noise ops, they are only returned if they
  carry a unique payload that isn't already covered by a non-noise op and
  hasn't been seen previously.
  """
  payloads = {r.payload for r in records if not r.is_noise and r.payload}
  infos = []
  for r in records:
    if r.is_noise:
      if not r.payload or r.payload in payloads:
        continue
      payloads.add(r.payload)
    infos.append(r.emit_fn())
  return tuple(infos)
