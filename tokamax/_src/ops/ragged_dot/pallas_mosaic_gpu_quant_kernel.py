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

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


def ragged_dot_quantized_kernel_body(
    group_info,
    mi,
    ni,
    weights_gmem,
    x_gmem,
    scales_gmem,
    o_gmem,
    *,
    config: common.Config,
):
  """Pallas kernel for ragged dot with quantized RHS."""

  del mi
  m, k = x_gmem.shape

  x_elem_bits = jnp.finfo(x_gmem.dtype).bits
  w_elem_bits = jnp.iinfo(weights_gmem.dtype).bits

  swizzle_w = plgpu.find_swizzle(w_elem_bits * config.block_k, "lhs")
  swizzle_x = plgpu.find_swizzle(x_elem_bits * config.block_k, "rhs")

  x_swizzle_elems = (swizzle_x * 8) // x_elem_bits
  w_swizzle_elems = (swizzle_w * 8) // w_elem_bits

  def acc_scope(acc_ref):
    def compute(_, w_smem, x_smem, s_smem):
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
        sl = slice(j * x_swizzle_elems, (j + 1) * x_swizzle_elems)
        plgpu.wgmma(
            acc_ref,
            common.dequant(s_smem.at[0], w[:, sl]),
            plgpu.transpose_ref(x_smem.at[:, sl], (1, 0)),
        )
        plgpu.wgmma_wait(1)

    x_transforms = (
        plgpu.TilingTransform((8, x_swizzle_elems)),
        plgpu.SwizzleTransform(swizzle_x),
    )
    w_transforms = (
        plgpu.TilingTransform((8, w_swizzle_elems)),
        plgpu.SwizzleTransform(swizzle_w),
    )
    plgpu.emit_pipeline(
        compute,
        grid=(k // config.block_k,),
        in_specs=(
            plgpu.BlockSpec(
                (config.block_n, config.block_k),
                lambda k_idx: (ni, k_idx),
                transforms=w_transforms,
            ),
            plgpu.BlockSpec(
                (config.block_m, config.block_k),
                lambda k_idx: (group_info.block, k_idx),
                transforms=x_transforms,
            ),
            plgpu.BlockSpec((1, config.block_n), lambda k_idx: (k_idx, ni)),
        ),
        max_concurrent_steps=config.num_stages,
        delay_release=1,
    )(
        weights_gmem.at[group_info.group_id],
        x_gmem,
        scales_gmem.at[group_info.group_id],
    )
    return acc_ref[...]

  acc = pl.run_scoped(acc_scope, plgpu.ACC((config.block_n, config.block_m)))
  out_elem_bits = jnp.finfo(o_gmem.dtype).bits
  swizzle_out = plgpu.find_swizzle(out_elem_bits * config.block_n, "out")
  smem = plgpu.SMEM(
      (config.block_m, config.block_n),
      dtype=o_gmem.dtype,
      transforms=(plgpu.SwizzleTransform(swizzle_out),),
  )
  pl.run_scoped(
      functools.partial(
          common.store_acc_transposed, acc, o_gmem, ni, m, group_info, config
      ),
      smem,
  )


def ragged_dot_quantized_kernel(
    lhs: jax.Array,
    rhs_quantized: quantization.QuantizedArray,
    group_sizes: jax.Array,
    out_dtype: jnp.dtype,
    config: common.Config,
) -> jax.Array:
  """Returns the Pallas kernel for quantized ragged dot."""

  if rhs_quantized.tile_shape != (1, config.block_k, 1):
    raise NotImplementedError(
        "Only scaling tile supported is (1, config.block_k, 1) got:"
        f" {rhs_quantized.tile_shape}."
    )

  weights, scales, x = (
      rhs_quantized.values.transpose(0, 2, 1),
      rhs_quantized.scales,
      lhs,
  )
  (num_groups, n, k_weights), (m, k_x) = weights.shape, x.shape
  if k_weights != k_x:
    raise ValueError(
        f"Contraction dim mismatch: weights.shape[-1]={k_weights},"
        f" x.shape[-1]={k_x}"
    )
  if group_sizes.shape[0] != num_groups:
    raise ValueError(
        "Expected group_sizes to have shape"
        f" {(num_groups,)} but got {group_sizes.shape}"
    )

  body = functools.partial(
      ragged_dot_quantized_kernel_body,
      config=config,
  )

  kernel = common.ragged_kernel(
      body,
      g=num_groups,
      m=m,
      n=n,
      out_dtype=out_dtype,
      config=config,
  )
  return kernel(group_sizes, weights, x, scales)
