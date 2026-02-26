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


_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)


def body(
    group_info: common.GroupInfo,
    mi,
    ni,
    w_gmem,
    x_gmem,
    w_scales_gmem,
    o_gmem,
    o_smem,
    schedule_barrier,
    *,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
):
  """The main kernel function for ragged dot-product."""
  del mi
  block_m, block_n, block_k = config.block_m, config.block_n, config.block_k

  m, k = x_gmem.shape
  wg = lax.axis_index("wg")
  ns = pl.ds(wg * block_n, block_n)

  def schedule_barrier_arrive_and_wait():
    plgpu.barrier_arrive(schedule_barrier)
    plgpu.barrier_wait(schedule_barrier)

  def pipeline_body(_, w_smem, x_smem, w_scales_smem, acc):
    pl.when(wg == 0)(schedule_barrier_arrive_and_wait)
    w = w_smem[ns].astype(w_scales_smem.dtype)
    w_scales = plgpu.load(w_scales_smem, (ns,), layout=_WGMMA_ROW)
    w *= lax.broadcast_in_dim(w_scales, w.shape, [0])
    schedule_barrier_arrive_and_wait()
    plgpu.wgmma(acc, w, x_smem.T)
    pl.when(wg == 1)(schedule_barrier_arrive_and_wait)
    plgpu.wgmma_wait(0)
    return acc

  def pipeline_context(pipeline_callback):
    compute_acc = lambda acc: pipeline_callback(acc)[...]
    acc = pl.run_scoped(compute_acc, plgpu.ACC((block_n, block_m)))

    if activation is not None:
      acc = activation(acc)

    ni_ = 2 * ni + wg
    common.store_acc_transposed(acc, o_gmem, ni_, m, group_info, o_smem.at[wg])

  mi = group_info.block
  gi = group_info.group_id

  spec = common.tiled_swizzled_block_spec
  x_spec = spec((block_m, block_k), x_gmem.dtype, lambda ki: (mi, ki), "x")
  w_spec = spec((2 * block_n, block_k), w_gmem.dtype, lambda ki: (ni, ki), "w")
  w_scales_spec = plgpu.BlockSpec(
      (None, 2 * block_n),
      lambda ki: (ki // (k // w_scales_gmem.shape[1] // block_k), ni),
  )

  plgpu.emit_pipeline_warp_specialized(
      pipeline_body,
      num_compute_wgs=2,
      wg_axis="wg",
      memory_registers=168 if config.persistent else 40,
      grid=(k // block_k,),
      compute_context=pipeline_context,
      in_specs=(w_spec, x_spec, w_scales_spec),
      max_concurrent_steps=max(config.num_stages // 2, 2),
  )(w_gmem.at[gi], x_gmem, w_scales_gmem.at[gi])


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

  if (
      rhs.scale_tile_shape[0] != 1
      or rhs.scale_tile_shape[1] % config.block_k != 0
      or rhs.scale_tile_shape[2] != 1
  ):
    raise NotImplementedError(
        "Only scaling tile supported is (1, N * block_k, 1) got:"
        f" {rhs.scale_tile_shape}."
    )

  assert n % (config.block_n * 2) == 0

  kernel = common.ragged_kernel(
      functools.partial(body, activation=activation, config=config),
      g=g,
      m=m,
      n=n,
      out_dtype=out_dtype,
      config=config,
      thread_axis="wg",
      scratch_shapes=(
          plgpu.SMEM((2, config.block_m, config.block_n), out_dtype),
          plgpu.Barrier(num_arrivals=2),
      ),
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
