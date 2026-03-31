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
"""Pallas-Mosaic-GPU Gated Linear Unit SM90 kernel."""

import functools
import math
from typing import Callable
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  del ba
  return common.Config(
      tile_m=128,
      tile_n=64,
      tile_k=64,
      num_stages=4,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns the autotuning configs for the Pallas:MGPU GLU SM90 kernel."""
  del ba
  configs = set()
  tile_n = 64
  epi_tile_m = 64
  grid_minor_dim = common.MatmulDimension.N
  for tile_m in (128, 256):
    for tile_k in (64, 128, 256):
      for num_stages in (1, 2, 4):
        for epi_tile_n in (16, 32):
          for grid_tile_width in (1, 2, 4, 8, 16):
            for wg_dimension in (
                common.MatmulDimension.M, common.MatmulDimension.N
            ):
              for cluster_size_m in (1, 2):
                for cluster_size_n in (1, 2):
                  if cluster_size_m != 1 and cluster_size_n != 1:
                    continue
                  configs.add(
                      common.Config(
                          tile_m=tile_m,
                          tile_n=tile_n,
                          tile_k=tile_k,
                          num_stages=num_stages,
                          epi_tile_n=epi_tile_n,
                          epi_tile_m=epi_tile_m,
                          grid_minor_dim=grid_minor_dim,
                          grid_tile_width=grid_tile_width,
                          wg_dimension=wg_dimension,
                          cluster_size_m=cluster_size_m,
                          cluster_size_n=cluster_size_n,
                      )
                  )
  return configs


@jaxtyping.jaxtyped
def gated_linear_unit(
    x: Float[Array, "*B M K"],
    weights: Float[Array, "K 2 N"],
    *,
    activation: Callable[[jax.Array], jax.Array],
    config: common.Config,
) -> Float[Array, "*B M N"]:
  """Gated Linear Unit implementation for SM90.

  Args:
    x: LHS operand.
    weights: RHS operand.
    activation: The activation function to use.
    config: Optional config. If `None`, a good default is chosen.

  Returns:
    The result of the GLU.
  """
  orig_x_shape = x.shape
  x = jax.lax.collapse(x, 0, -1)
  m, k = x.shape
  _, _, n = weights.shape
  dtype = x.dtype
  if x.dtype != weights.dtype:
    raise ValueError(
        f"Matmul LHS and RHS have incompatible dtypes {x.dtype} vs"
        f" {weights.dtype}"
    )

  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  num_stages = config.num_stages
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  if tile_n % swizzle_elems != 0:
    raise ValueError(f"{tile_n=} must be a multiple of {swizzle_elems=}")
  if tile_k % swizzle_elems != 0:
    raise ValueError(f"{tile_k=} must be a multiple of {swizzle_elems=}")

  lhs_transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  rhs_transforms = (
      plgpu.TransposeTransform((0, 2, 1, 3)),  # (2, k, n)
      # (2, k, n, tk, tn)
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.TransposeTransform((0, 2, 1, 3, 4, 5)),  # (k, 2, n, tk, tn)
      plgpu.SwizzleTransform(swizzle),
  )
  cta_tile_m = tile_m * (1 + (config.wg_dimension == common.MatmulDimension.M))
  cta_tile_n = tile_n * (1 + (config.wg_dimension == common.MatmulDimension.N))
  cluster_tile_m = cta_tile_m * config.cluster_size_m
  cluster_tile_n = cta_tile_n * config.cluster_size_n
  if m % cluster_tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m} for the given config")
  if n % cluster_tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n} for the given config")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  k_iters = k // tile_k

  epi_tile_n = config.epi_tile_n or tile_n
  epi_tile_m = config.epi_tile_m or tile_m

  def kernel(a_gmem, b_gmem, out_gmem, out_smem):

    def get_pipeline(pipeline_body, compute_context):
      return plgpu.emit_pipeline_warp_specialized(
          pipeline_body,
          grid=(k_iters,),
          memory_registers=40,
          in_specs=[
              plgpu.BlockSpec(
                  (cta_tile_m, tile_k),
                  lambda k: (0, k),
                  transforms=lhs_transforms,
                  memory_space=plgpu.SMEM,
                  collective_axes=("cluster",)
                  if config.cluster_size_n > 1
                  else (),
              ),
              plgpu.BlockSpec(
                  (tile_k, 2, cta_tile_n),
                  lambda k: (k, 0, 0),
                  transforms=rhs_transforms,
                  memory_space=plgpu.SMEM,
                  collective_axes=("cluster",)
                  if config.cluster_size_m > 1
                  else (),
              ),
          ],
          wg_axis="wg",
          num_compute_wgs=2,
          max_concurrent_steps=num_stages,
          compute_context=compute_context,
      )

    # Functions don't influence the allocations necessary to run the pipeline.
    ignore = lambda *_, **__: None

    @functools.partial(
        pl.run_scoped,
        pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(
            a_gmem, b_gmem
        ),
        collective_axes="wg",
    )
    def _pipeline_scope(pipeline_allocs):
      wg_idx = lax.axis_index("wg")
      cta_idx = lax.axis_index("cluster")

      @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
      def _mn_loop(loop_info):
        (lin_idx,) = loop_info.index
        m_cluster_idx, n_cluster_idx = plgpu.planar_snake(
            lin_idx,
            (m_iters, n_iters),
            config.grid_minor_dim,
            config.grid_tile_width,
        )
        m_idx = m_cluster_idx
        n_idx = n_cluster_idx
        if config.cluster_size_m > 1:
          m_idx = m_cluster_idx * 2 + cta_idx
        elif config.cluster_size_n > 1:
          n_idx = n_cluster_idx * 2 + cta_idx
        cta_m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
        cta_n_slice = pl.ds(n_idx * cta_tile_n, cta_tile_n)
        if config.wg_dimension == common.MatmulDimension.M:
          wg_m_slice = pl.ds(wg_idx * tile_m, tile_m)
          wg_n_slice = slice(None)
        else:
          wg_m_slice = slice(None)
          wg_n_slice = pl.ds(wg_idx * tile_n, tile_n)

        def compute_context(eval_pipeline):

          @functools.partial(
              pl.run_scoped,
              acc_ref=plgpu.ACC((tile_m, 2 * tile_n), jnp.float32),
          )
          def _acc_scope(acc_ref):
            eval_pipeline(acc_ref)
            acc = acc_ref[...].astype(dtype).astype(jnp.float32)
            acc = activation(acc[:, :tile_n]) * acc[:, tile_n:]
            acc = acc.astype(dtype)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            for epi_mi in range(tile_m // epi_tile_m):
              for epi_ni in range(tile_n // epi_tile_n):
                epi_m_slice = slice(
                    epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m
                )
                epi_n_slice = slice(
                    epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n
                )
                slot = (epi_mi * (tile_n // epi_tile_n) + epi_ni) % 2
                plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                out_smem[wg_idx, slot] = acc[epi_m_slice, epi_n_slice]
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    out_smem.at[wg_idx, slot],
                    out_gmem.at[cta_m_slice, cta_n_slice]
                    .at[wg_m_slice, wg_n_slice]
                    .at[epi_m_slice, epi_n_slice],
                )

        def mma_body(_, a_smem, b_smem, acc_ref):
          reshaped_b = b_smem.at[:, :, wg_n_slice]
          reshaped_b = reshaped_b.reshape(
              reshaped_b.shape[0], math.prod(reshaped_b.shape[1:])
          )
          plgpu.wgmma(acc_ref, a_smem.at[wg_m_slice], reshaped_b)
          plgpu.wgmma_wait(0)
          return acc_ref

        get_pipeline(mma_body, compute_context)(
            a_gmem.at[cta_m_slice, :],
            b_gmem.at[:, :, cta_n_slice],
            allocations=pipeline_allocs,
        )

    # Await all transfers before we exit.
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  # We don't need multiple slots if there's only one epilogue tile.
  num_out_slots = min(2, (tile_m * tile_n) // (epi_tile_m * epi_tile_n))
  out_swizzle = plgpu.find_swizzle(epi_tile_n * jnp.dtype(dtype).itemsize * 8)
  out_swizzle_elems = out_swizzle // jnp.dtype(dtype).itemsize
  out_transforms = (
      plgpu.TilingTransform((8, out_swizzle_elems)),
      plgpu.SwizzleTransform(out_swizzle),
  )
  scratch_shapes = [
      plgpu.SMEM(
          (2, num_out_slots, epi_tile_m, epi_tile_n),
          dtype,
          transforms=out_transforms,
      ),
  ]
  num_sms = backend.get_default_device().core_count
  cluster_size = config.cluster_size_m * config.cluster_size_n
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(num_sms // cluster_size,),
      grid_names=("cluster_grid",),
      cluster=(cluster_size,),
      cluster_names=("cluster",),
      num_threads=3,
      thread_name="wg",
      scratch_shapes=scratch_shapes,
  )
  return jnp.reshape(f(x, weights), (*orig_x_shape[:-1], n))
