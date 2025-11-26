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
"""GPU utilities and constants."""

from typing import Final
import jax
from jax.extend import backend
from tokamax._src import config as config_lib

NUM_REGISTERS_PER_SM: Final[int] = 64 * 1024  # P100, V100, A100, H100, B200
CACHE_LINE_SIZE_BYTES: Final[int] = 128


def _compute_capability(device: jax.Device | None = None) -> float | None:
  """Returns the compute capability of the device."""

  if device is None:
    device = backend.get_default_device()

  if device.platform != 'gpu':
    return None

  return float(getattr(device, 'compute_capability', None))


def is_sm80(device: jax.Device | None = None) -> bool:
  """Returns True if the device is an Ampere GPU and False otherwise."""

  compute_capability = _compute_capability(device)
  if compute_capability is None:
    return False

  # https://developer.nvidia.com/cuda-gpus
  return 8 <= compute_capability < 9


def is_sm90(device: jax.Device | None = None) -> bool:
  """Returns True if the device is a Hopper GPU and False otherwise."""

  compute_capability = _compute_capability(device)
  if compute_capability is None:
    return False

  # https://developer.nvidia.com/cuda-gpus
  return 9 <= compute_capability < 10


def is_sm100(device: jax.Device | None = None) -> bool:
  """Returns True if the device is a Blackwell GPU and False otherwise."""
  compute_capability = _compute_capability(device)
  if compute_capability is None:
    return False

  # https://developer.nvidia.com/cuda-gpus
  return 10 <= compute_capability < 11


def has_mosaic_gpu_support(device: jax.Device | None = None) -> bool:
  if config_lib.cross_compile.value:
    return True
  if device is None:
    device = backend.get_default_device()

  if device.platform != 'gpu':
    return False

  # Only currently supported for Hopper and above.
  return float(device.compute_capability) >= 9.0


def has_triton_support(device: jax.Device | None = None) -> bool:
  """Returns True if Triton is supported by the default JAX device."""

  if config_lib.cross_compile.value:
    return True
  if device is None:
    device = backend.get_default_device()

  if device.platform != 'gpu':
    return False

  # TODO: A temporary workaround for Pathways arrays not having a
  # compute_capability attribute.
  if not hasattr(device, 'compute_capability'):
    return True

  # Only currently supported for Ampere and above.
  return float(device.compute_capability) >= 8.0
