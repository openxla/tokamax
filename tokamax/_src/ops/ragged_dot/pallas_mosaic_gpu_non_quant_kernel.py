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
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


QuantizedArray = quantization.QuantizedArray


def ragged_dot_non_quantized_kernel_body(
    group_info,
    ni,
    lhs_gmem,
    rhs_gmem,
    o_gmem,
    *,
    swizzle: int,
    out_dtype: jnp.dtype,
    config: common.Config,
):
  """Pallas kernel body for non-quantized ragged dot."""

  m, k = lhs_gmem.shape
  out_elem_bits = jnp.finfo(out_dtype).bits
  elem_bits = jnp.finfo(lhs_gmem.dtype).bits
  swizzle_elems = swizzle * 8 // elem_bits
  out_swizzle_elems = swizzle * 8 // out_elem_bits

  block_m, block_n = config.block_m, config.block_n
  block_k = min(k, config.block_k)
  if block_k % swizzle_elems:
    raise ValueError(
        f"block_k {block_k} must be a multiple of swizzle_elems {swizzle_elems}"
    )

  def compute_acc(acc_ref):
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )
    lhs_spec = plgpu.BlockSpec(
        (block_m, block_k),
        lambda ki: (0, ki),
        transforms=transforms,
        delay_release=1,
    )
    rhs_spec = plgpu.BlockSpec(
        (block_k, block_n),
        lambda ki: (ki, ni),
        transforms=transforms,
        delay_release=1,
    )
    plgpu.emit_pipeline(
        lambda _, lhs_smem, rhs_smem: plgpu.wgmma(acc_ref, lhs_smem, rhs_smem),
        grid=(k // block_k,),
        in_specs=(lhs_spec, rhs_spec),
        max_concurrent_steps=config.num_stages,
    )(
        lhs_gmem.at[pl.ds(group_info.offset, block_m)],
        rhs_gmem.at[group_info.group_id],
    )
    return acc_ref[...]

  acc = pl.run_scoped(compute_acc, plgpu.ACC((block_m, block_n)))

  transforms = (
      plgpu.TilingTransform((1, out_swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  o_smem_type = plgpu.SMEM((block_m, block_n), out_dtype, transforms=transforms)

  @functools.partial(pl.run_scoped, o_smem=o_smem_type)
  def _store_scope(o_smem):
    o_smem[...] = acc.astype(out_dtype)
    plgpu.commit_smem()

    smem_start = group_info.in_block_offset
    remaining_rows = min(block_m, m)
    while remaining_rows > 0:
      const_rows_len = 1 << int(math.log2(remaining_rows))
      remaining_rows //= 2

      @pl.when(group_info.bsize & const_rows_len != 0)
      def _():
        o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
        o_gmem_slice = o_gmem.at[
            pl.ds(group_info.offset + smem_start, const_rows_len),
            pl.ds(ni * block_n, block_n),
        ]
        plgpu.copy_smem_to_gmem(o_smem_slice, o_gmem_slice, commit_group=False)

      smem_start += group_info.bsize & const_rows_len
    plgpu.commit_smem_to_gmem_group()
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def ragged_dot_non_quantized_kernel(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    out_dtype: jnp.dtype,
    config: common.Config,
) -> jax.Array:
  """Pallas kernel for ragged dot with non-quantized inputs."""

  m, k = lhs.shape
  g, k2, n = rhs.shape
  assert k == k2

  if lhs.dtype != rhs.dtype:
    raise ValueError(
        f"lhs and rhs must have the same dtype. Got {lhs.dtype=} and"
        f" {rhs.dtype=}"
    )

  if lhs.dtype not in (jnp.bfloat16, jnp.float16):
    raise NotImplementedError(
        f"Only bfloat16/float16 inputs are supported. Got {lhs.dtype=}"
    )

  elem_bits = jnp.finfo(lhs.dtype).bits
  swizzle = common.find_swizzle(elem_bits * config.block_k, "lhs")

  if group_sizes.shape[0] != g:
    raise ValueError(
        f"Expected group_sizes to have shape {(g,)} but got {group_sizes.shape}"
    )

  body_fn = functools.partial(
      ragged_dot_non_quantized_kernel_body,
      swizzle=swizzle,
      config=config,  # This config's block_k must work with swizzle
      out_dtype=out_dtype,
  )

  kernel = common.ragged_kernel(
      body_fn,
      g=g,
      m=m,
      n=n,
      out_dtype=out_dtype,
      config=config,
  )
  return kernel(group_sizes, lhs, rhs)
