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

import re
from typing import Final
import jax
from jax.extend import backend
from tokamax._src import config as config_lib

NUM_REGISTERS_PER_SM: Final[int] = 64 * 1024  # P100, V100, A100, H100, B200
CACHE_LINE_SIZE_BYTES: Final[int] = 128

_DEVICE_KIND_COMPUTE_CAPABILITIES: Final[tuple[tuple[str, float], ...]] = (
    ('VR200', 10.7),
    ('GB300', 10.3),
    ('B300', 10.3),
    ('GB200', 10.0),
    ('B200', 10.0),
    ('GH200', 9.0),
    ('H200', 9.0),
    ('H100', 9.0),
    ('L40S', 8.9),
    ('L40', 8.9),
    ('L4', 8.9),
    ('A100', 8.0),
    ('T4', 7.5),
    ('V100', 7.0),
    ('P100', 6.0),
    ('P40', 6.1),
    ('P4', 6.1),
    ('K80', 3.7),
)
_DEVICE_KIND_CC_PATTERNS: Final[tuple[tuple[re.Pattern[str], float], ...]] = (
    tuple(
        (
            re.compile(
                rf'(?:^|[^A-Za-z0-9]){re.escape(kind)}(?:[^A-Za-z0-9]|$)'
            ),
            cc,
        )
        for kind, cc in _DEVICE_KIND_COMPUTE_CAPABILITIES
    )
)


def _compute_capability(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> float | None:
  """Returns the compute capability of the device."""

  if device is None:
    device = backend.get_default_device()

  if device.platform != 'gpu':
    return None

  cc = getattr(device, 'compute_capability', None)
  if cc is not None:
    return float(cc)

  device_kind = getattr(device, 'device_kind', '')
  default_device = backend.get_default_device()
  if (
      default_device.platform == 'gpu'
      and getattr(default_device, 'device_kind', None) == device_kind
      and (default_cc := getattr(default_device, 'compute_capability', None))
      is not None
  ):
    return float(default_cc)

  for pattern, inferred_cc in _DEVICE_KIND_CC_PATTERNS:
    if pattern.search(device_kind):
      return inferred_cc

  return None


def _cc_between(
    lower: float,
    upper: float,
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if the device's compute capabilities is within [lower, upper)."""
  if (cc := _compute_capability(device)) is None:
    return False
  return cc >= lower and cc < upper


def is_sm80(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if the device is an Ampere GPU and False otherwise."""
  return _cc_between(8.0, 9.0, device)


def is_sm90(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if the device is a Hopper GPU and False otherwise."""
  return _cc_between(9.0, 10.0, device)


def is_sm100(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if the device is a Blackwell/SM100-family (10.x) GPU."""
  return _cc_between(10.0, 11.0, device)


def is_sm103(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if the device is a GB300 (sm103) GPU and False otherwise."""
  return _cc_between(10.3, 10.4, device)


def is_sm107(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if the device is a VR200 (sm107) GPU and False otherwise."""
  return _cc_between(10.7, 10.8, device)


def has_tcgen05_int8_support(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if the device supports tcgen05 .kind::i8 (int8 x int4) MMA."""
  return _cc_between(10.0, 10.2, device)


def has_mosaic_gpu_support(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if Mosaic GPU is supported by the JAX device."""
  if config_lib.cross_compile.value:
    return True
  if device is None:
    device = backend.get_default_device()

  if device.platform != 'gpu':
    return False

  # TODO: Pathways and virtual/abstract GPU devices may omit
  # compute_capability.
  if (cc := _compute_capability(device)) is None:
    return True

  # Only supported for Ampere and above. SM80 support is experimental.
  return cc >= 8.0


def has_triton_support(
    device: jax.Device | jax.sharding.AbstractDevice | None = None,
) -> bool:
  """Returns True if Triton is supported by the default JAX device."""

  if config_lib.cross_compile.value:
    return True
  if device is None:
    device = backend.get_default_device()

  if device.platform != 'gpu':
    return False

  # TODO: A temporary workaround for Pathways arrays and
  # virtual/abstract GPU devices not having a compute_capability attribute.
  if (cc := _compute_capability(device)) is None:
    return True

  # Only currently supported for Ampere and above.
  return cc >= 8.0
