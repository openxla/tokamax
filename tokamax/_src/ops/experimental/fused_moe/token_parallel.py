# Copyright 2026 Google LLC
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
"""Disjoint token ownership within groups of replicated MoE inputs."""
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class TokenReplicaLayout:
    """Groups are ordered mesh coordinates, independent of expert ownership.

    Every member supplies identical input rows. Each computes one contiguous
    slice, then all members recover the original row order. Singleton groups
    leave the input and output unchanged, including unaligned row counts.
    """
    groups: tuple[tuple[int, ...], ...]
    slots: tuple[int, ...]
    input_rows: int
    local_rows: int

    @classmethod
    def create(cls,
               *,
               ep: int,
               input_rows: int,
               groups: tuple[tuple[int, ...], ...] | None = None,
               row_alignment: int = 1) -> Self:
        if ep < 1 or input_rows < 1:
            raise ValueError("EP width and input row count must be positive")
        if row_alignment < 1 or row_alignment & (row_alignment - 1):
            raise ValueError("row alignment must be a positive power of two")
        groups = (tuple((i, ) for i in range(ep)) if groups is None else tuple(
            tuple(int(i) for i in group) for group in groups))
        if (not groups or not groups[0]
                or any(len(g) != len(groups[0]) for g in groups)
                or sorted(i for g in groups for i in g) != list(range(ep))):
            raise ValueError("token replica groups must partition the mesh "
                             "into equally sized, nonempty ordered groups")
        slots = [0] * ep
        for group in groups:
            for slot, rank in enumerate(group):
                slots[rank] = slot
        width = len(groups[0])
        local_rows = (input_rows + width - 1) // width
        # Pallas routing tiles need aligned row blocks. Below one block,
        # keep a power-of-two whole-array tile rather than widening to eight.
        alignment = (min(row_alignment, 1 <<
                         (local_rows - 1).bit_length()) if width > 1 else 1)
        local_rows = (local_rows + alignment - 1) // alignment * alignment
        return cls(groups, tuple(slots), input_rows, local_rows)

    def split(self,
              rows: jax.Array,
              rank: jax.Array,
              *,
              fill_value: float = 0) -> jax.Array:
        if len(self.groups[0]) == 1:
            return rows
        padding = self.local_rows * len(self.groups[0]) - self.input_rows
        rows = jnp.pad(rows, ((0, padding), (0, 0)),
                       constant_values=fill_value)
        slot = jnp.asarray(self.slots, dtype=jnp.int32)[rank]
        return lax.dynamic_slice_in_dim(rows,
                                        slot * self.local_rows,
                                        self.local_rows,
                                        axis=0)

    def restore(self, rows: jax.Array, axis: str) -> jax.Array:
        if len(self.groups[0]) == 1:
            return rows
        gathered = lax.all_gather(rows,
                                  axis,
                                  axis=0,
                                  tiled=True,
                                  axis_index_groups=self.groups)
        return gathered[:self.input_rows]
