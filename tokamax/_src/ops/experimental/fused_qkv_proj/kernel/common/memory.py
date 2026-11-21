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
"""Where kernel operands live, and the weights a kernel streams into VMEM.

Memory spaces: per-tile inputs stream from HBM through the pipeline (a single
tile stays in VMEM, aliased instead of copied); outputs are always HBM; small
weights are pinned to VMEM by XLA before the kernel; the large projection
weight stays in HBM and the kernel copies it in itself (StreamedWeight).

No numerics here.
"""

from collections.abc import Sequence
import dataclasses
from typing import Protocol, TypeVar

import jax
from jax.experimental import layout
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

Tree = TypeVar("Tree")
ColumnChunks = tuple[tuple[int, int], ...]  # (first column, width) per chunk.


class AsyncCopy(Protocol):
  """What pltpu.make_async_copy returns: a DMA to start, then wait for."""

  def start(self) -> None:
    """Starts the DMA."""
    ...

  def wait(self) -> None:
    """Blocks until the DMA is done."""
    ...


# Memory spaces.


def specs(tree: Tree, memory_space: pltpu.MemorySpace) -> Tree:
  """`tree` with each array replaced by a BlockSpec in memory_space."""
  spec = pl.BlockSpec(memory_space=memory_space)
  return jax.tree.map(lambda _: spec, tree)


def tiled_input_space(num_tiles: int) -> pltpu.MemorySpace:
  """VMEM for one tile (its block is the whole array, aliased), else HBM."""
  return pltpu.VMEM if num_tiles == 1 else pltpu.HBM


def constrain(tree: Tree, memory_space: pltpu.MemorySpace) -> Tree:
  """Asks XLA to place every array of `tree` in memory_space.

  NOTE: A BlockSpec only sets the kernel's view; XLA picks the operand's
  memory space from this constraint. Without it, a streamed weight lands in
  HBM anyway but small weights do too, and the kernel reads them slowly.

  Args:
    tree: a pytree of arrays.
    memory_space: where XLA should place them.

  Returns:
    tree, each array constrained.
  """
  return jax.tree.map(
      lambda x: pltpu.with_memory_space_constraint(x, memory_space), tree
  )


def place(tree: Tree, memory_space: pltpu.MemorySpace) -> tuple[Tree, Tree]:
  """(tree's arrays pinned to memory_space, their matching BlockSpecs).

  The constraint makes XLA put the buffers there; the BlockSpecs tell the
  kernel which memory space its refs are in. Both must agree.

  Args:
    tree: a pytree of arrays.
    memory_space: where to put them.

  Returns:
    (constrained tree, tree of BlockSpecs).
  """
  return constrain(tree, memory_space), specs(tree, memory_space)


def large_tiled(x: jax.Array | None) -> jax.Array | None:
  """bf16 x in the large (16, 128) tiling XLA picks for the residual stream.

  The post-attention norm between layers wants the residual in (16, 128) for
  the MLP matmul it feeds. XLA passes a 16-bit Mosaic operand in (16, 128) only
  when it comes from a layout constraint (b/479188348), so fused_qkv/fused_gdn
  constrain their residual input and output to it; otherwise XLA relayouts the
  residual between layers.

  Args:
    x: a (T, K) bf16 array, or None.

  Returns:
    x, constrained to the large tiling.
  """
  if x is None:
    return None
  tiling = layout.Layout((0, 1), tiling=((16, 128), (2, 1)))
  return layout.with_layout_constraint(x, tiling)


# Streamed weights.


def stream_chunks(
    blocks: Sequence[tuple[int, int]], max_width: int
) -> ColumnChunks:
  """DMA chunks of a streamed weight, in read order.

  A chunk joins consecutive blocks that are also adjacent in columns, so each
  chunk starts at the column read from it first.

  Args:
    blocks: (first column, width) of each block the kernel reads, in order.
    max_width: most columns per chunk.

  Returns:
    (first column, width) per chunk.
  """
  chunks = []
  for col, width in blocks:
    if chunks:
      start, size = chunks[-1]
      if start + size == col and size < max_width:
        chunks[-1] = (start, size + width)
        continue
    chunks.append((col, width))
  return tuple(chunks)


def stream_scratch(
    w: jax.Array | jax.ShapeDtypeStruct, chunks: ColumnChunks
) -> tuple[pl.MemoryRef, pl.MemoryRef]:
  """Scratch of a streamed weight: (its VMEM copy, one DMA sem per chunk).

  TODO: the copy is the weight's full size, so the chunked DMA
  overlaps compute but the whole weight must fit in VMEM (e.g. Qwen3.5 GDN at
  TP1 doesn't). A few reused chunk slots would lift that limit.
  """
  return pltpu.VMEM(w.shape, w.dtype), pltpu.SemaphoreType.DMA((len(chunks),))


def nbytes(w: jax.Array | jax.ShapeDtypeStruct) -> int:
  """Bytes of w; works on ShapeDtypeStructs too."""
  return w.size * w.dtype.itemsize


@dataclasses.dataclass
class StreamedWeight:
  """Copies a weight from HBM to VMEM in column chunks, inside the kernel.

  Usage:
    1. start() starts one DMA per chunk.
    2. Either call wait_for(col) just before reading the chunk at col, so the
       copies overlap compute, or call wait_all() once before any read. The
       kernel configs' overlap_weight_dma picks which.

  Each chunk is waited on once; a second wait_for on it emits nothing.
  """

  hbm: jax.Ref  # (K, N) the weight as passed to the kernel.
  vmem: jax.Ref  # (K, N) its VMEM copy; what the kernel reads.
  sems: jax.Ref  # (num_chunks,) DMA semaphores.
  chunks: ColumnChunks
  # Started DMAs that haven't been waited on yet, keyed by their chunk's first
  # column. wait_for() and wait_all() remove the ones they wait on, so no
  # chunk is waited on twice.
  _pending: dict[int, AsyncCopy] = dataclasses.field(default_factory=dict)

  def start(self) -> None:
    """Starts every chunk's DMA."""
    for i, (start, size) in enumerate(self.chunks):
      cols = pl.ds(start, size)
      copy = pltpu.make_async_copy(
          self.hbm.at[:, cols], self.vmem.at[:, cols], self.sems.at[i]
      )
      copy.start()
      self._pending[start] = copy

  def wait_all(self) -> None:
    """Waits for every chunk not waited on yet."""
    for copy in self._pending.values():
      copy.wait()
    self._pending.clear()

  def wait_for(self, col: int) -> None:
    """Waits for the chunk that starts at column col, unless already waited on."""
    copy = self._pending.pop(col, None)
    if copy is not None:
      copy.wait()
