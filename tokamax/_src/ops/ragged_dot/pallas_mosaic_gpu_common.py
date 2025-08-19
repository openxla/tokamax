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

"""Common Pallas Mosaic GPU utilities."""

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Self
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import memref
import pydantic


def find_swizzle(dim_size_bits: int, what: str) -> int:
  for swizzle_bytes in (128, 64, 32, 16):
    if dim_size_bits % (swizzle_bytes * 8) == 0:
      return swizzle_bytes
  raise ValueError(
      f"No valid out swizzle for {what}: its minor dimension has"
      f" {dim_size_bits} bits, which is not a multiple of 128."
  )


class Config(pydantic.BaseModel, frozen=True):
  block_m: pydantic.conint(multiple_of=8, gt=0)
  block_n: pydantic.PositiveInt
  block_k: pydantic.PositiveInt
  num_stages: pydantic.PositiveInt
  split_k: pydantic.PositiveInt
  grid_block_n: pydantic.PositiveInt = 1
  warp_specialized: bool = True
  persistent: bool = True
  collective: bool = False  # B200 collective MMA


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GroupInfo:
  """Information regarding the group being processed in a block."""
  tile_size: int
  gmem_tile_size: int
  # Part of the block occupied by the current group.
  bsize: jax.Array
  current_step: jax.Array
  group_id: jax.Array
  in_block_offset: jax.Array
  offset: jax.Array
  remaining_rows_in_group: jax.Array
  total_steps: jax.Array

  @staticmethod
  def steps(group_sizes, tile_size) -> jax.Array:
    """Calculates the total number of steps for all groups."""
    # sum(cdiv(cumsum(group_sizes) % 8 + group_sizes, tile_size))
    off = steps = jnp.int32(0)
    for i in range(len(group_sizes)):
      grp = jnp.int32(group_sizes[i])
      steps += jnp.where(grp == 0, 0, pl.cdiv(grp + off % 8, tile_size))
      off += grp

    return steps

  def next_single_step(self, group_sizes) -> "GroupInfo":
    """Advances the group info to the next step."""
    finished = self.remaining_rows_in_group <= self.bsize
    group_id, remaining_rows_in_group = lax.cond(
        finished,
        functools.partial(self.next_nonzero_group, group_sizes, self.group_id),
        lambda: (self.group_id, self.remaining_rows_in_group - self.bsize),
    )
    offset = self.offset + self.bsize + self.in_block_offset
    in_block_offset = lax.rem(offset, self.gmem_tile_size)
    in_block_offset = jnp.where(finished, in_block_offset, jnp.int32(0))
    bsize = jnp.where(
        group_id >= len(group_sizes),
        jnp.int32(0),
        jnp.minimum(
            remaining_rows_in_group, self.tile_size - in_block_offset
        )
    )
    return dataclasses.replace(
        self,
        group_id=group_id,
        bsize=bsize,
        offset=lax.div(offset, self.gmem_tile_size) * self.gmem_tile_size,
        remaining_rows_in_group=remaining_rows_in_group,
        current_step=self.current_step + 1,
        in_block_offset=in_block_offset,
    )

  def reset(self, group_sizes):
    """Restores this object to its original state after flattening.

    Args:
      group_sizes: The group sizes array.

    Returns:
      A new `GroupInfo` object.
    """
    zero = jnp.int32(0)
    grp0 = jnp.int32(group_sizes[0])
    return dataclasses.replace(
        self,
        remaining_rows_in_group=grp0,
        group_id=zero,
        bsize=jnp.minimum(grp0, self.tile_size),
        offset=zero,
        current_step=zero,
        in_block_offset=zero,
    )

  @staticmethod
  def next_nonzero_group(group_sizes, cur_id) -> tuple[jax.Array, jax.Array]:
    def cond_fn(state):
      i, size = state
      return (i < group_sizes.shape[0]) & (size == 0)

    def body_fn(state):
      i = state[0] + 1
      return i, jnp.int32(group_sizes[i])

    return lax.while_loop(
        cond_fn, body_fn, (cur_id + 1, jnp.int32(group_sizes[cur_id + 1]))
    )

  def to_step(self, group_sizes, step: jax.Array) -> Self:
    loop_body = lambda _, x: x.next_single_step(group_sizes)
    return lax.cond(
        step >= self.current_step,
        lambda: lax.fori_loop(self.current_step, step, loop_body, self),
        lambda: lax.fori_loop(0, step, loop_body, self.reset(group_sizes)),
    )

  @classmethod
  def create(
      cls, group_sizes: jax.Array, tile_size: int, gmem_tile_size: int = 8
  ) -> Self:
    """Get the group info for the current block."""
    zero = jnp.int32(0)
    grp0 = jnp.int32(group_sizes[0])
    return cls(
        tile_size=tile_size,
        remaining_rows_in_group=grp0,
        group_id=zero,
        bsize=jnp.minimum(grp0, tile_size),
        offset=zero,
        current_step=zero,
        total_steps=cls.steps(group_sizes, tile_size),
        in_block_offset=zero,
        gmem_tile_size=gmem_tile_size,
    )


def dequant(s_ref, w):
  """Dequantize the array `w` using a 1D ref `s_ref`."""

  @plgpu.inline_mgpu(
      arg_types=(plgpu.RefType(), plgpu.Layout.WGMMA),
      return_type=plgpu.ShapeDtypeStruct(
          w.shape,
          s_ref.dtype,
          plgpu.Layout.WGMMA,
      ),
  )
  def scaled_w(_, s_smem, w):
    def scale(w_val, idx):
      assert s_smem.type.shape == [w.shape[0]]
      return arith.mulf(memref.load(s_smem, (idx[0],)), w_val)

    return w.foreach(scale, create_array=True)

  return scaled_w(s_ref, w.astype(s_ref.dtype))


# TODO: Unify this with the non_quant store.
def store_acc_transposed(
    acc,
    o_gmem,
    ni: jax.Array,
    m: int,
    group_info: GroupInfo,
    config: Config,
    o_smem_swizzled,
):
  """Stores the accumulator into the output gmem.

  It does so by first storing the accumulator into a swizzled shared memory
  array, then copying that to the output gmem. This is done to allow for
  coalesced writes.

  Args:
    acc: The accumulator to store.
    o_gmem: The output gmem.
    ni: The current n index.
    m: The total m dimension.
    group_info: The group info for the current block.
    config: The kernel config.
    o_smem_swizzled: The swizzled shared memory array to use.
  """
  out_elem_bits = jnp.finfo(o_gmem.dtype).bits
  swizzle_out = find_swizzle(out_elem_bits * config.block_n, "out")
  out_swizzle_elems = (swizzle_out * 8) // out_elem_bits

  o_smem_t = o_smem_swizzled.reshape(config.block_m // 8, 1, 8, config.block_n)
  o_smem_t = plgpu.untile_ref(o_smem_t, (8, config.block_n))
  o_smem_t = plgpu.transpose_ref(o_smem_t, (1, 0))
  o_smem_t[...] = plgpu.layout_cast(
      acc.astype(o_gmem.dtype), plgpu.Layout.WGMMA_TRANSPOSED
  )
  plgpu.commit_smem()
  del o_smem_t
  o_smem0 = o_smem_swizzled.reshape(
      config.block_m, config.block_n // out_swizzle_elems, out_swizzle_elems
  )
  # Write out the largest power of two rows first, then the next largest,
  # etc. This allows us to coalesce writes as much as possible.
  offset = group_info.in_block_offset
  size = 1 << (min(config.block_m, m).bit_length() - 1)
  while size > 0:
    @pl.when(group_info.bsize & size != 0)
    def _():
      o_smem = o_smem0.at[pl.ds(offset, size)]
      o_smem = plgpu.untile_ref(o_smem, (out_swizzle_elems,))
      o_gref_slice = o_gmem.at[
          pl.ds(group_info.offset + offset, size),
          pl.ds(ni * config.block_n, config.block_n),
      ]
      plgpu.copy_smem_to_gmem(o_smem, o_gref_slice, commit_group=False)

    offset += group_info.bsize & size
    size //= 2
  plgpu.commit_smem_to_gmem_group()
  plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def ragged_kernel(
    body, *, g, m, n, out_dtype, config, thread_axis=None
) -> Callable[..., jax.Array]:
  """Returns a Pallas kernel for ragged matmul.

  This kernel computes a ragged matmul, where the LHS is a dense
  matrix of shape (m, k) and the RHS is a ragged matrix of shape (g,
  k, n), where g is the number of groups. The output is a dense matrix
  of shape (m, n).

  The kernel uses a persistent kernel if config.persistent is True,
  otherwise it uses a non-persistent kernel.

  Args:
    body: The body of the kernel. This function will be called with the group
      info, the current m index, the current n index, and the arguments to the
      kernel.
    g: The number of groups.
    m: The m dimension of the LHS matrix.
    n: The n dimension of the RHS matrix.
    out_dtype: The dtype of the output matrix.
    config: The kernel config.
    thread_axis: The name of the thread axis to use for warp specialization. If
      None, warp specialization is not used.

  Returns:
    A Pallas kernel for ragged matmul.
  """

  num_compute_threads = 1 if thread_axis is None else 2
  inner_grid = (
      config.grid_block_n,
      pl.cdiv(m, config.block_m) + g - 1,
      pl.cdiv(n, config.grid_block_n * config.block_n * num_compute_threads),
  )

  def kernel_body(group_sizes_gmem, *args):
    initial_group_info = GroupInfo.create(group_sizes_gmem, config.block_m)
    def loop_body(
        idx: Sequence[jax.Array], carry: GroupInfo
    ):
      group_info = carry
      block_ni, step, remainder_ni = idx
      ni = (
          block_ni
          * pl.cdiv(
              n, config.block_n * config.grid_block_n * num_compute_threads
          )
          + remainder_ni
      )
      group_info = group_info.to_step(group_sizes_gmem, step)
      @pl.when(group_info.bsize > 0)
      def _():
        body(group_info, ni, *args)

      return group_info

    if config.persistent:
      inner_grid_l = list(inner_grid)
      # Now we know the exact number of steps required.
      inner_grid_l[1] = initial_group_info.total_steps
      plgpu.nd_loop(
          tuple(inner_grid_l),
          collective_axes="sm",
          init_carry=initial_group_info,
      )(loop_body)
    else:
      loop_body(
          tuple(map(lax.axis_index, ("remainder_n", "m", "block_n"))),
          initial_group_info,
      )

  if config.persistent:
    # TODO: Detect this number from device.
    grid = (132,)
    grid_names = ("sm",)
  else:
    grid = inner_grid
    grid_names = ("remainder_n", "m", "block_n")

  return plgpu.kernel(
      kernel_body,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      grid=grid,
      grid_names=grid_names,
      thread_name=thread_axis,
      num_threads=thread_axis and (num_compute_threads + 1),
  )
