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

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Float  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Integer  # pylint: disable=g-multiple-import,g-importing-member
import qwix
from tokamax._src import jaxtyping
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


def body(
    group_info: common.GroupInfo,
    mi,
    ni,
    w_gmem,
    x_gmem,
    w_scales_gmem,
    o_gmem,
    schedule_barrier,
    *,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
):
  """The main kernel function for ragged dot-product."""
  del mi
  block_m, block_n, block_k = config.block_m, config.block_n, config.block_k

  m = x_gmem.shape[0]
  x_elem_bits = jnp.dtype(x_gmem.dtype).itemsize * 8
  w_elem_bits = jnp.iinfo(w_gmem.dtype).bits

  # K is the contiguous dimension
  try:
    swizzle_w = plgpu.find_swizzle(w_elem_bits * block_k, "lhs")
  except ValueError as e:
    raise NotImplementedError("No possible swizzle.") from e
  swizzle_x = plgpu.find_swizzle(x_elem_bits * block_k, "rhs")

  x_swizzle_elems = (swizzle_x * 8) // x_elem_bits
  w_swizzle_elems = (swizzle_w * 8) // w_elem_bits

  wg = lax.axis_index("wg")
  ns = pl.ds(wg * block_n, block_n)

  def schedule():
    plgpu.barrier_arrive(schedule_barrier)
    plgpu.barrier_wait(schedule_barrier)

  def pipeline_body(_, w_smem, x_smem, w_scales_smem, acc_ref):
    pl.when(wg == 0)(schedule)
    w = common.dequant(w_scales_smem.at[0, ns], w_smem[ns])
    schedule()
    plgpu.wgmma(acc_ref, w, x_smem.T)
    pl.when(wg == 1)(schedule)
    plgpu.wgmma_wait(0)
    return acc_ref

  def pipeline_context(cb):
    acc = pl.run_scoped(
        lambda acc_ref: cb(acc_ref)[...], plgpu.ACC((block_n, block_m))
    )

    def store_acc(acc, o_smem):
      assert block_n % 8 == 0
      if activation is not None:
        acc = activation(acc)
      common.store_acc_transposed(
          acc, o_gmem, 2 * ni + wg, m, group_info, o_smem.at[wg]
      )

    o_smem_type = plgpu.SMEM((2, block_m, block_n), o_gmem.dtype)
    pl.run_scoped(
        functools.partial(store_acc, acc), o_smem_type, collective_axes=("wg",)
    )

  try:
    swizzle_w_transform = plgpu.SwizzleTransform(swizzle_w)
  except ValueError as e:
    raise NotImplementedError(f"{swizzle_w=} unsupported.") from e

  mi = group_info.block
  gi = group_info.group_id
  x_transforms = (
      plgpu.TilingTransform((8, x_swizzle_elems)),
      plgpu.SwizzleTransform(swizzle_x),
  )
  w_transforms = (
      plgpu.TilingTransform((8, w_swizzle_elems)),
      swizzle_w_transform,
  )
  x_spec = plgpu.BlockSpec(
      (block_m, block_k), lambda ki: (mi, ki), transforms=x_transforms
  )
  w_spec = plgpu.BlockSpec(
      (2 * block_n, block_k), lambda ki: (ni, ki), transforms=w_transforms
  )
  w_scales_spec = plgpu.BlockSpec((1, 2 * block_n), lambda ki: (ki, ni))

  with jax.named_scope("pipeline"):
    plgpu.emit_pipeline_warp_specialized(
        pipeline_body,
        num_compute_wgs=2,
        wg_axis="wg",
        memory_registers=168 if config.persistent else 40,
        grid=(w_gmem.shape[2] // block_k,),
        compute_context=pipeline_context,
        in_specs=(w_spec, x_spec, w_scales_spec),
        max_concurrent_steps=max(config.num_stages // 2, 2),
    )(w_gmem.at[gi], x_gmem, w_scales_gmem.at[gi])

  # The memory WG does not arrive at the run so we release it here.
  # TODO: Change the run_scoped() API so this is not
  # necessary.
  @pl.when(wg == 2)
  def _():
    pl.run_scoped(
        lambda _: None, plgpu.SMEM((), jnp.float32), collective_axes="wg"
    )


@jaxtyping.jaxtyped
def ragged_dot_quantized_ws_kernel(
    lhs: Float[Array, "M K"],
    rhs: Float[qwix.QArray, "G K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "M N"]:
  """Returns the Pallas kernel for quantized ragged dot."""
  assert rhs.zero_point is None

  m, _ = lhs.shape
  g, _, n = rhs.shape

  if rhs.scale_tile_shape != (1, config.block_k, 1):
    raise NotImplementedError(
        "Only scaling tile supported is (1, block_k, 1) got:"
        f" {rhs.scale_tile_shape} (block_k={config.block_k})."
    )

  if group_sizes.shape != (g,):
    raise ValueError(
        f"Expected group_sizes to have shape {(g,)} but got {group_sizes.shape}"
    )

  def kernel_entry(*args):
    return pl.run_scoped(
        functools.partial(
            body, *args, activation=activation, config=config
        ),
        plgpu.Barrier(num_arrivals=2),
        collective_axes="wg",
    )

  assert n % (config.block_n * 2) == 0

  kernel = common.ragged_kernel(
      kernel_entry,
      g=g,
      m=m,
      n=n,
      out_dtype=out_dtype,
      config=config,
      thread_axis="wg",
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
      rhs.qvalue.mT,
      lhs,
      rhs.scale,
  )
