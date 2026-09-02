# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Ragged dot SM80 kernel using Pallas MGPU."""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import mosaic_gpu as tk_mgpu
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common

# The epilogue reads back one output row (`block_n` elements) as a 1D array
# sharded across the warpgroup, so `block_n` must be a multiple of the warpgroup
# size.
_BLOCK_N_MULTIPLE = 128
# Block start alignment for the ragged tiling (see `GroupInfo.create_aligned`).
_ALIGN_TILE = 8


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  del ba
  return common.Config(
      block_m=128,
      block_n=128,
      block_k=32,
      num_stages=3,
      split_k=1,
      persistent=False,
      grid_tile_width=8,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  lhs, rhs, _ = ba.args
  _, k = lhs.shape
  n = rhs.shape[-1]

  configs = set()
  for block_m in (64, 128):
    for block_n in (128, 256):
      if block_n % _BLOCK_N_MULTIPLE or n % block_n:
        continue
      for block_k in (32, 64, 128):
        if k % block_k:
          continue
        for num_stages in (2,) + tuple(n for n in (3, 4) if n <= k // block_k):
          for grid_tile_width in (1, 2, 4, 8):
            configs.add(
                common.Config(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_stages=num_stages,
                    split_k=1,
                    persistent=False,
                    grid_tile_width=grid_tile_width,
                )
            )
  return configs


def ragged_dot_kernel(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> jax.Array:
  """Ragged dot for SM80."""
  common.check_bf16xbf16_or_f16xf16(lhs, rhs)

  if config.split_k != 1:
    raise NotImplementedError("`split_k != 1` not supported on SM80.")

  m, k = lhs.shape
  g, _, n = rhs.shape

  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  num_stages = config.num_stages

  if block_n % _BLOCK_N_MULTIPLE != 0:
    raise NotImplementedError(
        f"{block_n=} must be a multiple of {_BLOCK_N_MULTIPLE}."
    )
  if k % block_k != 0:
    raise NotImplementedError(f"{k=} must be divisible by {block_k=}.")

  n_pad = pl.align_to(n, block_n)
  if n_pad != n:
    rhs = jnp.concatenate(
        [rhs, jnp.zeros((g, k, n_pad - n), rhs.dtype)], axis=2
    )

  m_iters = pl.cdiv(m, block_m) + g - 1
  n_iters = n_pad // block_n
  k_iters = k // block_k

  m_pad = max(block_m, pl.align_to(m, _ALIGN_TILE))
  if pad := m_pad - m:
    lhs = jnp.concatenate([lhs, jnp.zeros((pad, k), lhs.dtype)], axis=0)
  win_hi = m_pad - block_m  # max valid window start (>= 0).
  dtype = lhs.dtype

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n_pad), out_dtype),
      scratch_types=[
          plgpu.SMEM((block_m, block_n), out_dtype),
      ],
      grid=(m_iters * n_iters,),
      grid_names=("mn",),
      kernel_name="ragged_dot_sm80",
  )
  def kernel(
      lhs_gmem,
      rhs_gmem,
      group_id_gmem,
      block_start_gmem,
      start_within_block_gmem,
      actual_size_gmem,
      out_gmem,
      out_smem,
  ):
    mn = lax.axis_index("mn")
    ti, ni = plgpu.planar_snake(
        mn, (m_iters, n_iters), config.grid_minor_dim, config.grid_tile_width
    )
    lhs_spec = tk_mgpu.tiled_swizzled_block_spec(
        (pl.Element(block_m), block_k),
        dtype,
        lambda ki: (win_start, ki),
        "lhs",
        oob_fill_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
    )
    rhs_spec = tk_mgpu.tiled_swizzled_block_spec(
        (block_k, block_n), dtype, lambda ki: (ki, ni), "rhs"
    )

    gi = group_id_gmem[ti]
    block_start = block_start_gmem[ti]
    start_within_block = start_within_block_gmem[ti]
    actual_size = actual_size_gmem[ti]
    actual_start = block_start + start_within_block

    win_start = pl.multiple_of(lax.min(block_start, win_hi), _ALIGN_TILE)
    smem_base = actual_start - win_start

    n_slice = pl.ds(ni * block_n, block_n)

    @pl.when(actual_size > 0)
    def _():
      acc = jnp.zeros((block_m, block_n), jnp.float32)

      @functools.partial(
          plgpu.emit_pipeline,
          grid=(k_iters,),
          in_specs=(lhs_spec, rhs_spec),
          max_concurrent_steps=num_stages,
          init_carry=acc,
      )
      def pipelined_body(_, lhs_smem, rhs_smem, acc):
        with jax.named_scope("load"):
          a = lhs_smem[...]
          b = rhs_smem[...]
        with jax.named_scope("mma"):
          return plgpu.mma(acc, a, b)

      acc = pipelined_body(lhs_gmem, rhs_gmem.at[gi])

      with jax.named_scope("epilogue"):
        if activation is not None:
          acc = activation(acc)

        # The epilogue stages the accumulator through plain SMEM and writes
        # back only this tile's valid rows one row at a time, so a tile
        # straddling a group boundary never clobbers the neighbouring group's
        # rows.
        out_smem[...] = acc.astype(out_smem.dtype)

        @pl.loop(0, actual_size)
        def _store_row(r):
          row = out_smem[smem_base + r]
          out_gmem[actual_start + r, n_slice] = row.astype(out_gmem.dtype)

  group_info = common.GroupInfo.create_aligned(
      group_sizes, block_m, m_iters, _ALIGN_TILE
  )

  out = kernel(
      lhs,
      rhs,
      group_info.group_id,
      group_info.block_start,
      group_info.start_within_block,
      group_info.actual_size,
  )
  return out[:, :n]
