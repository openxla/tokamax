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
"""Ragged dot Pallas-Mosaic-GPU Quantized Kernel."""

import functools
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


def ragged_dot_quantized_kernel_body(
    group_info,
    mi,
    ni,
    w_gmem,
    x_gmem,
    w_scales_gmem,
    o_gmem,
    *,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
):
  """Pallas kernel for ragged dot with quantized RHS."""
  del mi
  block_m, block_n, block_k = config.block_m, config.block_n, config.block_k
  m, k = x_gmem.shape

  x_elem_bits = jnp.finfo(x_gmem.dtype).bits
  w_elem_bits = jnp.iinfo(w_gmem.dtype).bits
  swizzle_w = plgpu.find_swizzle(w_elem_bits * block_k, "lhs")
  swizzle_x = plgpu.find_swizzle(x_elem_bits * block_k, "rhs")
  x_swizzle_elems = (swizzle_x * 8) // x_elem_bits
  w_swizzle_elems = (swizzle_w * 8) // w_elem_bits

  def compute_acc(acc_ref):
    def pipeline_body(_, w_smem, x_smem, w_scales_smem):
      w = w_smem[...]
      # Tiling along the reduction dimension. This overlaps to some extent
      # scaling/casting with wgmma.
      assert w.shape[1] % x_swizzle_elems == 0
      steps = w.shape[1] // x_swizzle_elems
      if steps == 1:
        # The LHS registers are reused in each loop. Synchronizing here is
        # the only way to make sure they are not overwritten.
        plgpu.wgmma_wait(0)

      for j in range(steps):
        ks = slice(j * x_swizzle_elems, (j + 1) * x_swizzle_elems)
        w_ = common.dequant(w_scales_smem.at[0], w[:, ks])
        plgpu.wgmma(acc_ref, w_, x_smem.at[:, ks].T)
        plgpu.wgmma_wait(1)

    mi = group_info.block
    gi = group_info.group_id
    x_transforms = (
        plgpu.TilingTransform((8, x_swizzle_elems)),
        plgpu.SwizzleTransform(swizzle_x),
    )
    w_transforms = (
        plgpu.TilingTransform((8, w_swizzle_elems)),
        plgpu.SwizzleTransform(swizzle_w),
    )
    x_spec = plgpu.BlockSpec(
        (block_m, block_k),
        lambda ki: (mi, ki),
        transforms=x_transforms,
        delay_release=1,
    )
    w_spec = plgpu.BlockSpec(
        (block_n, block_k),
        lambda ki: (ni, ki),
        transforms=w_transforms,
        delay_release=1,
    )
    w_scales_spec = plgpu.BlockSpec(
        (1, block_n),
        lambda ki: (ki, ni),
        delay_release=1,
    )
    plgpu.emit_pipeline(
        pipeline_body,
        grid=(k // block_k,),
        in_specs=(w_spec, x_spec, w_scales_spec),
        max_concurrent_steps=config.num_stages,
    )(
        w_gmem.at[gi],
        x_gmem,
        w_scales_gmem.at[gi],
    )
    if activation is not None:
      return activation(acc_ref[...])
    return acc_ref[...]

  acc = pl.run_scoped(compute_acc, plgpu.ACC((block_n, block_m)))
  store = functools.partial(
      common.store_acc_transposed, acc, o_gmem, ni, m, group_info
  )
  pl.run_scoped(store, plgpu.SMEM((block_m, block_n), o_gmem.dtype))


@jaxtyping.jaxtyped
def ragged_dot_quantized_kernel(
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
        "Only scaling tile supported is (1, config.block_k, 1) got:"
        f" {rhs.scale_tile_shape}."
    )

  if group_sizes.shape != (g,):
    raise ValueError(
        f"Expected group_sizes to have shape {(g,)} but got {group_sizes.shape}"
    )

  body = functools.partial(
      ragged_dot_quantized_kernel_body,
      activation=activation,
      config=config,
  )
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
      rhs.qvalue.mT,
      lhs,
      rhs.scale,
  )
