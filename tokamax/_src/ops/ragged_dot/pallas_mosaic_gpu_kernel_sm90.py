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
"""Ragged dot Pallas-Mosaic-GPU Non-Quantized Kernel."""
import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


_WGMMA = plgpu.Layout.WGMMA


def _kernel_body(
    group_info,
    mi,
    ni,
    lhs_gmem,
    rhs_gmem,
    o_gmem,
    *,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
):
  """Pallas kernel body for non-quantized ragged dot."""

  del mi
  m, k = lhs_gmem.shape
  block_m = config.block_m
  block_n = config.block_n
  block_k = min(k, config.block_k)

  def compute_acc(acc):
    mi = group_info.block
    spec = functools.partial(common.tiled_swizzled_block_spec, delay_release=1)
    lhs_spec = spec(
        (block_m, block_k), lhs_gmem.dtype, lambda ki: (mi, ki), "lhs"
    )
    rhs_spec = spec(
        (block_k, block_n), rhs_gmem.dtype, lambda ki: (ki, ni), "rhs"
    )
    plgpu.emit_pipeline(
        lambda _, lhs_smem, rhs_smem: plgpu.wgmma(acc, lhs_smem, rhs_smem),
        grid=(k // block_k,),
        in_specs=(lhs_spec, rhs_spec),
        max_concurrent_steps=config.num_stages,
    )(lhs_gmem, rhs_gmem.at[group_info.group_id])
    return acc[...]

  acc = pl.run_scoped(compute_acc, plgpu.ACC((block_m, block_n)))

  o_smem_type = common.tiled_swizzled_smem(
      (block_m, block_n), o_gmem.dtype, tiling_prefix=(1,), what="out"
  )

  @functools.partial(pl.run_scoped, o_smem=o_smem_type)
  def epilogue(o_smem):
    acc_ = acc if activation is None else activation(acc)
    o_smem[...] = acc_.astype(o_smem.dtype)
    plgpu.commit_smem()

    smem_start = group_info.start_within_block
    remaining_rows = min(block_m, m)
    while remaining_rows > 0:
      const_rows_len = 1 << int(math.log2(remaining_rows))
      remaining_rows //= 2

      @pl.when(group_info.actual_size & const_rows_len != 0)
      def _():
        o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
        o_gmem_slice = o_gmem.at[
            pl.ds(group_info.block_start + smem_start, const_rows_len),
            pl.ds(ni * block_n, block_n),
        ]
        plgpu.copy_smem_to_gmem(o_smem_slice, o_gmem_slice, commit_group=False)

      smem_start += group_info.actual_size & const_rows_len
    plgpu.commit_smem_to_gmem_group()
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


@jaxtyping.jaxtyped
def ragged_dot_kernel(
    lhs: Float[Array, "M K"],
    rhs: Float[Array, "G K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "M N"]:
  """Pallas kernel for ragged dot with non-quantized inputs."""
  common.check_bf16xbf16_or_f16xf16(lhs, rhs)

  m, _ = lhs.shape
  g, _, n = rhs.shape

  body = functools.partial(_kernel_body, config=config, activation=activation)
  kernel = common.ragged_kernel(
      body, g=g, m=m, n=n, out_dtype=out_dtype, config=config
  )
  group_info = common.GroupInfo.create(
      group_sizes, config.block_m, pl.cdiv(m, config.block_m) + g - 1
  )
  return kernel(
      group_info.group_id,
      group_info.block,
      group_info.block_start,
      group_info.actual_start,
      group_info.actual_end,
      group_info.start_within_block,
      group_info.actual_size,
      lhs,
      rhs,
  )


def _ragged_contracting_dim_dot_kernel_body(
    group_sizes_gmem,
    group_sizes_starts_gmem,
    lhs_gmem,
    rhs_gmem,
    o_gmem,
    *,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
):
  """Pallas kernel body for non-quantized ragged contracting dim dot."""
  block_m = config.block_m
  block_k = config.block_k
  block_n = config.block_n

  mi, ni, gi = map(jax.lax.axis_index, ("m", "n", "g"))
  group_start = group_sizes_starts_gmem[gi]
  group_end = group_start + group_sizes_gmem[gi]

  lb = jax.lax.div(group_start.astype(jnp.int32), block_k)
  ub = pl.cdiv(group_end.astype(jnp.int32), block_k)

  def acc_scope(acc):
    def body(idxs, lhs_smem, rhs_smem):
      (ki,) = idxs
      k_start = (lb + ki) * block_k
      kx = plgpu.broadcasted_iota(jnp.int32, lhs_smem.shape, 1, layout=_WGMMA)
      mask = (kx >= (group_start - k_start)) & (kx < (group_end - k_start))

      # TODO: Only mask out the 1st and last iteration.
      plgpu.wgmma(acc, lhs_smem[...] * mask, rhs_smem)
      plgpu.wgmma_wait(1)

    spec = functools.partial(common.tiled_swizzled_block_spec, delay_release=1)
    lhs_spec = spec(
        (block_m, block_k), lhs_gmem.dtype, lambda ki: (mi, lb + ki), "lhs"
    )
    rhs_spec = spec(
        (block_k, block_n), rhs_gmem.dtype, lambda ki: (lb + ki, ni), "rhs"
    )
    plgpu.emit_pipeline(
        body,
        grid=(ub - lb,),
        in_specs=(lhs_spec, rhs_spec),
        max_concurrent_steps=config.num_stages,
    )(lhs_gmem, rhs_gmem)
    return acc[...]

  acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

  transforms = common.tile_swizzle_transforms((block_m, block_n), o_gmem.dtype)
  o_smem_type = plgpu.SMEM(
      (block_m, block_n), o_gmem.dtype, transforms=transforms
  )

  @functools.partial(pl.run_scoped, o_smem=o_smem_type)
  def epilogue(o_smem):
    acc_ = acc if activation is None else activation(acc)
    o_smem[...] = acc_.astype(o_smem.dtype)
    plgpu.commit_smem()
    ms = pl.ds(mi * block_m, block_m)
    ns = pl.ds(ni * block_n, block_n)
    plgpu.copy_smem_to_gmem(o_smem, o_gmem.at[gi, ms, ns])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


@jaxtyping.jaxtyped
def ragged_contracting_dim_dot_kernel(
    lhs: Float[Array, "K M"],
    rhs: Float[Array, "K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "G M N"]:
  """Pallas kernel for ragged contracting dim dot with non-quantized inputs."""
  common.check_bf16xbf16_or_f16xf16(lhs, rhs)

  _, m = lhs.shape
  _, n = rhs.shape
  (g,) = group_sizes.shape

  body = functools.partial(
      _ragged_contracting_dim_dot_kernel_body,
      config=config,
      activation=activation,
  )
  kernel = plgpu.kernel(
      body,
      out_shape=jax.ShapeDtypeStruct((g, m, n), out_dtype),
      grid=(pl.cdiv(m, config.block_m), pl.cdiv(n, config.block_n), g),
      grid_names=("m", "n", "g"),
  )

  group_sizes_starts = jnp.cumulative_sum(group_sizes, include_initial=True)
  # TODO: Fuse transpose into kernel.
  return kernel(group_sizes, group_sizes_starts[:-1], lhs.T, rhs)
