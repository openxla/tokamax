# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Lazy JAX hardware limits for the compact KDA kernels."""

import dataclasses
from jax.experimental.pallas import tpu as pltpu


@dataclasses.dataclass(frozen=True)
class _Limits:
  vmem_limit_bytes: int
  vmem_hw_limit_bytes: int
  block_align_minor: int
  block_align_major: int


def get_tpu_config():
  info = pltpu.get_tpu_info()
  capacity = int(info.vmem_capacity_bytes)
  # libtpu reserves 64 KiB of v7's physical VMEM from compiler requests.
  reserve = 64 * 1024 if getattr(info, "generation", 7) == 7 else 0
  return _Limits(
      capacity - reserve, capacity, int(info.num_sublanes), int(info.num_lanes)
  )
