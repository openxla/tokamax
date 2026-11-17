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
"""SMEM page_indices windowing and Context Parallel (CP) page ownership."""

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs


@dataclasses.dataclass(frozen=True)
class PageOwnership:
  """Plain-data page striping rule mapping logical pages to local HBM slots."""

  group_size: int = 1
  rank: Any = 0

  @property
  def is_striped(self) -> bool:
    return self.group_size > 1 or not isinstance(self.rank, int)

  def local_page(self, logical_page: Any) -> Any:
    """Maps a logical sequence page index to this rank's local page slot."""
    if self.group_size == 1:
      return logical_page
    return logical_page // self.group_size

  def owns(self, logical_page: Any) -> Any:
    """Returns whether `logical_page` resides in this rank's local KV cache."""
    if not self.is_striped and self.rank == 0:
      return True
    return (logical_page % self.group_size) == self.rank

  def max_logical_pages(self, pages_per_seq: int) -> int:
    """Returns the total logical page capacity across the CP group."""
    return pages_per_seq * self.group_size


@dataclasses.dataclass(frozen=True)
class PageWindowBuffer:
  """Double-buffered SMEM window over the HBM `page_indices` table."""

  hbm_ref: Any
  smem_ref: Any
  sem_ref: Any
  cfgs: configs.RpaConfigs
  ownership: PageOwnership = dataclasses.field(default_factory=PageOwnership)

  @staticmethod
  def scratch_shapes(cfgs: configs.RpaConfigs) -> tuple[Any, Any]:
    """Returns `(page_indices_smem, page_indices_dma_sem)` scratch shapes."""
    return (
        pltpu.SMEM((2 * cfgs.page_window_slot_len,), jnp.int32),
        pltpu.SemaphoreType.DMA((2,)),
    )

  @property
  def max_logical_page(self) -> int:
    """Maximum valid logical page index across the CP group."""
    return self.ownership.max_logical_pages(self.cfgs.serve.pages_per_seq) - 1

  def window_idx(self, s_idx: Any) -> Any:
    """Returns the window index containing sequence `s_idx`'s page row."""
    return s_idx >> (self.cfgs.page_window_seqs.bit_length() - 1)

  def start_window(self, win_idx: Any, pred: Any) -> None:
    """Starts async DMA of window `win_idx` into SMEM slot `win_idx & 1`."""
    num_lanes = self.cfgs.layout.num_lanes
    slot_len = self.cfgs.page_window_slot_len
    start = win_idx * self.cfgs.page_window_entries
    if self.cfgs.page_window_entries % num_lanes:
      start = start // num_lanes * num_lanes
    slot = win_idx & 1

    @pl.when(pred)
    def _copy():
      pltpu.make_async_copy(
          self.hbm_ref.at[pl.ds(pl.multiple_of(start, num_lanes), slot_len)],
          self.smem_ref.at[
              pl.ds(pl.multiple_of(slot * slot_len, num_lanes), slot_len)
          ],
          self.sem_ref.at[slot],
      ).start()

  def wait_window(self, win_idx: Any) -> None:
    """Waits for window `win_idx` DMA to finish in SMEM."""
    win = self.smem_ref.at[pl.ds(0, self.cfgs.page_window_slot_len)]
    pltpu.make_async_copy(win, win, self.sem_ref.at[win_idx & 1]).wait()

  def seq_row_offset(self, s_idx: Any) -> Any:
    """Returns the SMEM index where sequence `s_idx`'s page row begins."""
    num_lanes = self.cfgs.layout.num_lanes
    win_idx = self.window_idx(s_idx)
    offset = (win_idx & 1) * self.cfgs.page_window_slot_len
    if self.cfgs.page_window_entries % num_lanes:
      offset += (win_idx * self.cfgs.page_window_entries) & (num_lanes - 1)
    return (
        offset
        + (s_idx & (self.cfgs.page_window_seqs - 1))
        * self.cfgs.serve.pages_per_seq
    )

  def enter_seq(self, s_idx: Any, start_seq_idx: Any, end_seq_idx: Any) -> Any:
    """Prefetches window `win_idx + 1`, waits on `win_idx`, returns row offset."""
    win_idx = self.window_idx(s_idx)
    win_seqs = self.cfgs.page_window_seqs

    def _enter():
      self.start_window(win_idx + 1, (win_idx + 1) * win_seqs < end_seq_idx)
      self.wait_window(win_idx)

    if win_seqs == 1:
      _enter()
    else:
      first_in_window = (s_idx & (win_seqs - 1)) == 0
      pl.when((s_idx == start_seq_idx) | first_in_window)(_enter)
    return self.seq_row_offset(s_idx)

  def phys_page(self, row_offset: Any, logical_page: Any) -> jax.Array:
    """Maps `logical_page` to a clamped local slot and reads its physical page."""
    local_p = jnp.minimum(
        self.ownership.local_page(logical_page),
        self.cfgs.serve.pages_per_seq - 1,
    )
    return self.smem_ref[row_offset + local_p]

  def owns(self, logical_page: Any) -> Any:
    """Returns whether this rank owns `logical_page`."""
    return self.ownership.owns(logical_page)
