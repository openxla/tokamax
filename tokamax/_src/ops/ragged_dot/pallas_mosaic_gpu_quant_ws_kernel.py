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
import dataclasses

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


QuantizedArray = quantization.QuantizedArray


def body(
    group_info: common.GroupInfo,
    mi,
    ni,
    weights_gmem,
    x_gmem,
    scales_gmem,
    o_gmem,
    schedule_barrier,
    *,
    config: common.Config,
):
  """The main kernel function for ragged dot-product."""
  del mi

  m = x_gmem.shape[0]
  w_elem_bits = jnp.iinfo(weights_gmem.dtype).bits
  x_elem_bits = jnp.dtype(x_gmem.dtype).itemsize * 8
  out_elem_bits = jnp.dtype(o_gmem.dtype).itemsize * 8

  # K is the contiguous dimension
  try:
    swizzle_w = plgpu.find_swizzle(w_elem_bits * config.block_k, "lhs")
  except ValueError as e:
    raise NotImplementedError("No possible swizzle.") from e
  swizzle_x = plgpu.find_swizzle(x_elem_bits * config.block_k, "rhs")
  swizzle_out = plgpu.find_swizzle(out_elem_bits * config.block_n, "out")

  x_swizzle_elems = (swizzle_x * 8) // x_elem_bits
  w_swizzle_elems = (swizzle_w * 8) // w_elem_bits

  wg = lax.axis_index("wg")
  off = lax.select(wg == 0, 0, config.block_n)

  def schedule():
    plgpu.barrier_arrive(schedule_barrier)
    plgpu.barrier_wait(schedule_barrier)

  def store_scope(acc, o_smem_swizzled):
    assert config.block_n % 8 == 0
    o_smem_swizzled = o_smem_swizzled.at[wg]
    common.store_acc_transposed(
        acc, o_gmem, ni * 2 + wg, m, group_info, config, o_smem_swizzled
    )

  def accumulator_carry(cb):
    acc = pl.run_scoped(
        lambda acc_ref: cb(acc_ref)[...],
        plgpu.ACC((config.block_n, config.block_m)),
    )
    pl.run_scoped(
        functools.partial(store_scope, acc),
        plgpu.SMEM(
            (2, config.block_m, config.block_n),
            dtype=o_gmem.dtype,
            transforms=(plgpu.SwizzleTransform(swizzle_out),),
        ),
        collective_axes=("wg",),
    )

  def compute_acc_ref(_, w_smem, x_smem, s_smem, acc_ref):
    w_smem = w_smem.at[pl.ds(off, config.block_n)]
    s_smem = s_smem.at[0, pl.ds(off, config.block_n)]
    pl.when(wg == 0)(schedule)
    plgpu.wgmma_wait(0)
    w = w_smem[...]
    w = common.dequant(s_smem, w)
    schedule()
    plgpu.wgmma(
        acc_ref,
        w,
        plgpu.transpose_ref(x_smem, (1, 0)),
    )
    pl.when(wg == 1)(schedule)
    return acc_ref

  try:
    swizzle_w_transform = plgpu.SwizzleTransform(swizzle_w)
  except ValueError as e:
    raise NotImplementedError(f"{swizzle_w=} unsupported.") from e

  x_transforms = (
      plgpu.TilingTransform((8, x_swizzle_elems)),
      plgpu.SwizzleTransform(swizzle_x),
  )
  w_transforms = (
      plgpu.TilingTransform((8, w_swizzle_elems)),
      swizzle_w_transform,
  )
  k_steps = weights_gmem.shape[2] // config.block_k
  with jax.named_scope("pipeline"):
    plgpu.emit_pipeline_warp_specialized(
        compute_acc_ref,
        num_compute_wgs=2,
        wg_axis="wg",
        memory_registers=168 if config.persistent else 40,
        grid=(k_steps,),
        out_specs=[],
        compute_context=accumulator_carry,
        in_specs=[
            plgpu.BlockSpec(
                (config.block_n * 2, config.block_k),
                lambda k: (ni, k),
                transforms=w_transforms,
            ),
            plgpu.BlockSpec(
                (config.block_m, config.block_k),
                lambda k: (group_info.block, k),
                transforms=x_transforms,
            ),
            plgpu.BlockSpec((1, config.block_n * 2), lambda k: (k, ni)),
        ],
        max_concurrent_steps=max(config.num_stages // 2, 2),
    )(
        weights_gmem.at[group_info.group_id],
        x_gmem,
        scales_gmem.at[group_info.group_id],
    )

  # The memory WG does not arrive at the run so we release it here.
  # TODO: Change the run_scoped() API so this is not
  # necessary.
  @pl.when(wg == 2)
  def _():
    pl.run_scoped(lambda _: None, plgpu.SMEM((), jnp.float32), collective_axes="wg")


def ragged_dot_quantized_ws_kernel(
    lhs: jax.Array,
    rhs: QuantizedArray,
    group_sizes: jax.Array,
    out_dtype,
    config: common.Config,
) -> jax.Array:
  """Returns the Pallas kernel for quantized ragged dot."""

  if rhs.tile_shape != (1, config.block_k, 1):
    raise NotImplementedError(
        "Only scaling tile supported is (1, block_k, 1) got:"
        f" {rhs.tile_shape} (block_k={config.block_k})."
    )

  weights, scales, x = rhs.values.transpose(0, 2, 1), rhs.scales, lhs
  (num_groups, n, k), (m, k2) = weights.shape, x.shape
  assert k == k2

  if k != k2:
    raise ValueError(
        f"Contraction dim mismatch: weights.shape[-1]={k}, x.shape[-1]={k2}"
    )
  if group_sizes.shape[0] != num_groups:
    raise ValueError(
        "Expected group_sizes to have shape"
        f" {(num_groups,)} but got {group_sizes.shape}"
    )

  def kernel_entry(*args):
    return pl.run_scoped(
        functools.partial(body, *args, config=config),
        plgpu.Barrier(num_arrivals=2),
        collective_axes="wg",
    )

  assert n % (config.block_n * 2) == 0

  kernel = common.ragged_kernel(
      kernel_entry,
      g=num_groups,
      m=m,
      n=n,
      out_dtype=out_dtype,
      config=config,
      thread_axis="wg",
  )
  return kernel(group_sizes, weights, x, scales)
