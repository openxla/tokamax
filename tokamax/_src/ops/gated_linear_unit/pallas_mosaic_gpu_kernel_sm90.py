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
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common


_SM90_MAX_SMEM_BYTES = 224 * 1024


def _sm90_smem_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    epi_tile_m: int,
    epi_tile_n: int,
    wg_dimension: common.MatmulDimension,
    k: int,
    in_itemsize: int = 2,
    out_itemsize: int = 2,
    return_residuals: bool = False,
) -> int:
  """Returns the shared memory footprint in bytes for an SM90 GLU config."""
  cta_tile_m = tile_m * (1 + (wg_dimension == common.MatmulDimension.M))
  cta_tile_n = tile_n * (1 + (wg_dimension == common.MatmulDimension.N))
  eff_stages = max(1, min(num_stages, k // tile_k))
  in_bytes = (
      eff_stages * (cta_tile_m * tile_k + 2 * tile_k * cta_tile_n) * in_itemsize
  )
  num_epi_tiles = (tile_m * tile_n) // (epi_tile_m * epi_tile_n)
  num_out_slots = min(2, num_epi_tiles * (3 if return_residuals else 1))
  out_bytes = 2 * num_out_slots * epi_tile_m * epi_tile_n * out_itemsize
  return in_bytes + out_bytes


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns a shape- and hardware-aware heuristic Config for SM90."""
  m, n, k, in_dtype, store_dtype, return_residuals = common.get_problem_spec(ba)
  if m == 0 or n < 64 or k < 32:
    return common.Config(tile_m=128, tile_n=64, tile_k=64, num_stages=1)
  num_sms = getattr(backend.get_default_device(), "core_count", 1)
  in_itemsize = in_dtype.itemsize
  out_itemsize = store_dtype.itemsize

  # Choose tile_m and wg_dimension to balance SM occupancy and tile efficiency.
  if n >= 128 and pl.cdiv(m, 128) * (n // 128) < num_sms:
    tile_m = 64
    tile_n = 64
    wg_dimension = common.MatmulDimension.N
  elif m <= 128:
    tile_m = 64
    tile_n = 64
    wg_dimension = common.MatmulDimension.M
  else:
    tile_m = 128
    tile_n = 64
    if n < 128:
      wg_dimension = common.MatmulDimension.M
    elif pl.cdiv(m, 256) * (n // 64) >= num_sms:
      wg_dimension = common.MatmulDimension.M
    else:
      wg_dimension = common.MatmulDimension.N

  cta_tile_m = tile_m * (1 + (wg_dimension == common.MatmulDimension.M))
  cta_tile_n = tile_n * (1 + (wg_dimension == common.MatmulDimension.N))

  # Enable 2-CTA cluster multicast when there are enough tiles to fill all SMs
  # and RHS weights are large enough to benefit from L2/SMEM multicast.
  cluster_size_m = 1
  cluster_size_n = 1
  total_cta_tiles = pl.cdiv(m, cta_tile_m) * (n // cta_tile_n)
  if total_cta_tiles >= 2 * num_sms and k >= 128 and n >= 256:
    if pl.cdiv(m, cta_tile_m * 2) >= 2 and m >= cta_tile_m * 2:
      cluster_size_m = 2
    elif n % (cta_tile_n * 2) == 0:
      cluster_size_n = 2

  # Prefer tile_k=64 with 4 pipeline stages unless CTA tile footprint is small
  # enough to fit >=3 stages at tile_k=128; fall back to tile_k=32 when k % 64.
  if k % 64 != 0:
    tile_k = 32
  elif k % 128 == 0 and k >= 384 and cta_tile_m * cta_tile_n <= 128 * 64:
    tile_k = 128
  else:
    tile_k = 64
  epi_tile_m = 64
  epi_tile_n = 32 if tile_n >= 64 else tile_n

  num_stages = max(1, min(4, k // tile_k))
  while (
      num_stages > 1
      and _sm90_smem_bytes(
          tile_m,
          tile_n,
          tile_k,
          num_stages,
          epi_tile_m,
          epi_tile_n,
          wg_dimension,
          k,
          in_itemsize,
          out_itemsize,
          return_residuals,
      )
      > _SM90_MAX_SMEM_BYTES
  ):
    num_stages -= 1

  has_4x4_grid = (
      pl.cdiv(m, cta_tile_m * cluster_size_m) >= 4
      and (n // (cta_tile_n * cluster_size_n)) >= 4
  )
  grid_tile_width = 4 if has_4x4_grid else 1

  return common.Config(
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      num_stages=num_stages,
      epi_tile_n=epi_tile_n,
      epi_tile_m=epi_tile_m,
      grid_minor_dim=common.MatmulDimension.N,
      grid_tile_width=grid_tile_width,
      wg_dimension=wg_dimension,
      cluster_size_m=cluster_size_m,
      cluster_size_n=cluster_size_n,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns the autotuning configs for the Pallas:MGPU GLU SM90 kernel."""
  m, n, k, in_dtype, store_dtype, return_residuals = common.get_problem_spec(ba)
  if m == 0 or n == 0 or k == 0:
    return set()
  in_itemsize = in_dtype.itemsize
  out_itemsize = store_dtype.itemsize
  configs = set()
  tile_n = 64
  epi_tile_m = 64
  grid_minor_dim = common.MatmulDimension.N
  tile_k_candidates = (64, 128) if k % 64 == 0 else (32,)
  for tile_m in (64, 128):
    for tile_k in tile_k_candidates:
      if k % tile_k != 0:
        continue
      max_stages = k // tile_k
      stage_candidates = (
          (1,)
          if max_stages == 1
          else tuple(s for s in (2, 3, 4) if s <= max_stages)
      )
      for num_stages in stage_candidates:
        for epi_tile_n in (32, 64):
          for wg_dimension in (
              common.MatmulDimension.M,
              common.MatmulDimension.N,
          ):
            if (
                _sm90_smem_bytes(
                    tile_m,
                    tile_n,
                    tile_k,
                    num_stages,
                    epi_tile_m,
                    epi_tile_n,
                    wg_dimension,
                    k,
                    in_itemsize,
                    out_itemsize,
                    return_residuals,
                )
                > _SM90_MAX_SMEM_BYTES
            ):
              continue
            cta_tile_n = tile_n * (
                1 + (wg_dimension == common.MatmulDimension.N)
            )
            for cluster_size_m, cluster_size_n in ((1, 1), (2, 1), (1, 2)):
              cluster_tile_n = cta_tile_n * cluster_size_n
              if n % cluster_tile_n != 0:
                continue
              for grid_tile_width in (1, 4, 8):
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
    weights: base.FusedWeights | base.UnfusedWeights,
    *,
    activation: Callable[[jax.Array], jax.Array],
    config: common.Config,
    out_dtype: jax.typing.DTypeLike | None = None,
    return_residuals: bool = False,
    approx_math: bool = True,
) -> (
    Float[Array, "*B M N"]
    | tuple[Float[Array, "*B M N"], Float[Array, "*B M 2 N"]]
):
  """Gated Linear Unit implementation for SM90.

  Args:
    x: LHS operand.
    weights: RHS operand (fused `(K, 2, N)` or unfused `((K, N), (K, N))`).
    activation: The activation function to use.
    config: Kernel tile and pipeline configuration.
    out_dtype: Output and residual storage dtype (defaults to `x.dtype`).
    return_residuals: Whether to also return `(..., 2, N)` pre-activations.
    approx_math: Whether to allow PTX `approx` math instructions.

  Returns:
    The result of the GLU, or `(out, residuals)` if `return_residuals=True`.
  """
  is_unfused = isinstance(weights, tuple)
  for w in jax.tree.leaves(weights):
    if x.dtype != w.dtype:
      raise ValueError(
          f"Matmul LHS and RHS have incompatible dtypes {x.dtype} vs {w.dtype}"
      )

  orig_x_shape = x.shape
  x = jax.lax.collapse(x, 0, -1)
  m, k = x.shape
  n = weights[0].shape[-1] if is_unfused else weights.shape[-1]
  if is_unfused:
    if weights[0].shape != (k, n) or weights[1].shape != (k, n):
      raise ValueError(
          f"Incompatible shapes: x={x.shape},"
          f" weights=({weights[0].shape}, {weights[1].shape})"
      )
  elif weights.shape != (k, 2, n):
    raise ValueError(
        f"Incompatible shapes: x={x.shape}, weights={weights.shape}"
    )
  dtype = x.dtype
  store_dtype = jnp.dtype(dtype if out_dtype is None else out_dtype)

  if m == 0 or n == 0 or k == 0:
    raise NotImplementedError("Empty dimensions are not supported.")

  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  in_itemsize = jnp.dtype(dtype).itemsize
  out_itemsize = store_dtype.itemsize
  swizzle = min(
      plgpu.find_swizzle(tile_k * in_itemsize * 8),
      plgpu.find_swizzle(tile_n * in_itemsize * 8),
  )
  swizzle_elems = swizzle // in_itemsize
  if tile_n % swizzle_elems != 0:
    raise NotImplementedError(
        f"{tile_n=} must be a multiple of {swizzle_elems=}"
    )
  if tile_k % swizzle_elems != 0:
    raise NotImplementedError(
        f"{tile_k=} must be a multiple of {swizzle_elems=}"
    )

  lhs_transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  rhs_transforms = (
      plgpu.TilingTransform((8, 1, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  rhs_2d_transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  cta_tile_m = tile_m * (1 + (config.wg_dimension == common.MatmulDimension.M))
  cta_tile_n = tile_n * (1 + (config.wg_dimension == common.MatmulDimension.N))
  cluster_tile_m = cta_tile_m * config.cluster_size_m
  cluster_tile_n = cta_tile_n * config.cluster_size_n
  # `NotImplementedError`, not `ValueError`: these say "this kernel cannot
  # serve these shapes", which is the signal `gated_linear_unit` in `api.py`
  # catches to fall through to the next implementation. Raising `ValueError`
  # here escapes that loop, so a caller with awkward shapes gets a hard failure
  # instead of the XLA fallback the default chain promises.
  if n % cluster_tile_n != 0:
    raise NotImplementedError(
        f"{n=} must be divisible by {cluster_tile_n} for the given config"
    )
  if k % tile_k != 0:
    raise NotImplementedError(f"{k=} must be divisible by {tile_k=}")

  # TilingTransform((8, swizzle_elems)) requires the GMEM M dimension to be a
  # multiple of 8; hardware TMA zero-fills out-of-bounds rows up to
  # cluster_tile_m and ignores out-of-bounds rows on TMA store.
  if m % 8 != 0:
    x = jnp.pad(x, ((0, -m % 8), (0, 0)))
  m_padded = x.shape[0]
  m_iters = pl.cdiv(m_padded, cluster_tile_m)
  n_iters = n // cluster_tile_n
  k_iters = k // tile_k
  eff_stages = min(config.num_stages, k_iters)

  epi_tile_n = config.epi_tile_n or tile_n
  epi_tile_m = config.epi_tile_m or tile_m
  if tile_m % epi_tile_m != 0 or tile_n % epi_tile_n != 0:
    raise NotImplementedError(
        f"{tile_m=}, {tile_n=} must be divisible by"
        f" {epi_tile_m=}, {epi_tile_n=}"
    )
  num_epi_tiles = (tile_m * tile_n) // (epi_tile_m * epi_tile_n)
  num_out_slots = min(2, num_epi_tiles * (3 if return_residuals else 1))

  def kernel(a_gmem, b_gmem, *out_refs, out_smem):
    out_gmem = out_refs[0]
    res_gmem = out_refs[1] if return_residuals else None
    # Hardware TMA descriptors support at most 5 strided dimensions. A 3D
    # `(K, 2, N)` weight tensor tiled along both `K` and `N` already uses all 5
    # dimensions, so when `weights` is vmapped (`b_gmem` is a sliced view of a
    # 4D+ GMEM buffer), split gate and up views along the size-2 axis so TMA can
    # collapse the leading squeezed indices into a single dimension.
    use_2d_rhs = is_unfused or bool(getattr(b_gmem, "transforms", ()))
    if not is_unfused and use_2d_rhs:
      b_gmem = (b_gmem.at[:, 0, :], b_gmem.at[:, 1, :])

    def get_pipeline(pipeline_body, compute_context):
      in_specs: list[plgpu.BlockSpec | tuple[plgpu.BlockSpec, ...]] = [
          plgpu.BlockSpec(
              (cta_tile_m, tile_k),
              lambda k: (0, k),
              transforms=lhs_transforms,
              memory_space=plgpu.SMEM,
              collective_axes=("cluster",) if config.cluster_size_n > 1 else (),
          ),
      ]
      rhs_collective_axes = ("cluster",) if config.cluster_size_m > 1 else ()
      for i in range(
          2 if config.wg_dimension == common.MatmulDimension.N else 1
      ):
        if use_2d_rhs:
          rhs_spec = plgpu.BlockSpec(
              (tile_k, tile_n),
              lambda k, i=i: (k, i),
              transforms=rhs_2d_transforms,
              memory_space=plgpu.SMEM,
              collective_axes=rhs_collective_axes,
          )
          in_specs.append((rhs_spec, rhs_spec))
        else:
          in_specs.append(
              plgpu.BlockSpec(
                  (tile_k, 2, tile_n),
                  lambda k, i=i: (k, 0, i),
                  transforms=rhs_transforms,
                  memory_space=plgpu.SMEM,
                  collective_axes=rhs_collective_axes,
              )
          )
      return plgpu.emit_pipeline_warp_specialized(
          pipeline_body,
          grid=(k_iters,),
          memory_registers=40,
          in_specs=in_specs,
          wg_axis="wg",
          num_compute_wgs=2,
          max_concurrent_steps=eff_stages,
          compute_context=compute_context,
      )

    # Functions don't influence the allocations necessary to run the pipeline.
    ignore = lambda *_, **__: None

    if config.wg_dimension == common.MatmulDimension.N:
      alloc_shapes = [a_gmem, b_gmem, b_gmem]
    else:
      alloc_shapes = [a_gmem, b_gmem]

    @functools.partial(
        pl.run_scoped,
        pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(
            *alloc_shapes
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
            acc = acc_ref[...]
            gates = acc[:, :tile_n]
            proj = acc[:, tile_n:]
            out = (activation(gates) * proj).astype(store_dtype)
            gates_out = gates.astype(store_dtype) if return_residuals else None
            proj_out = proj.astype(store_dtype) if return_residuals else None
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            store_idx = 0

            def store_epi_tile(val, dst_ref):
              nonlocal store_idx
              slot = store_idx % num_out_slots
              plgpu.wait_smem_to_gmem(num_out_slots - 1, wait_read_only=True)
              out_smem[wg_idx, slot] = val
              plgpu.commit_smem()
              plgpu.copy_smem_to_gmem(out_smem.at[wg_idx, slot], dst_ref)
              store_idx += 1

            for epi_mi in range(tile_m // epi_tile_m):
              for epi_ni in range(tile_n // epi_tile_n):
                epi_m_slice = slice(
                    epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m
                )
                epi_n_slice = slice(
                    epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n
                )
                if return_residuals:
                  assert (
                      res_gmem is not None
                      and gates_out is not None
                      and proj_out is not None
                  )
                  store_epi_tile(
                      gates_out[epi_m_slice, epi_n_slice],
                      res_gmem.at[cta_m_slice, 0, cta_n_slice]
                      .at[wg_m_slice, wg_n_slice]
                      .at[epi_m_slice, epi_n_slice],
                  )
                  store_epi_tile(
                      proj_out[epi_m_slice, epi_n_slice],
                      res_gmem.at[cta_m_slice, 1, cta_n_slice]
                      .at[wg_m_slice, wg_n_slice]
                      .at[epi_m_slice, epi_n_slice],
                  )
                store_epi_tile(
                    out[epi_m_slice, epi_n_slice],
                    out_gmem.at[cta_m_slice, cta_n_slice]
                    .at[wg_m_slice, wg_n_slice]
                    .at[epi_m_slice, epi_n_slice],
                )

        def do_wgmma(acc_r, a_r, b_smem):
          if use_2d_rhs:
            b_gate, b_proj = b_smem
            plgpu.wgmma(acc_r.at[:, :tile_n], a_r, b_gate)
            plgpu.wgmma(acc_r.at[:, tile_n:], a_r, b_proj)
          else:
            reshaped_b = b_smem.reshape(
                b_smem.shape[0], math.prod(b_smem.shape[1:])
            )
            plgpu.wgmma(acc_r, a_r, reshaped_b)

        def mma_body(_, *refs):
          if config.wg_dimension == common.MatmulDimension.N:
            a_smem, b_smem0, b_smem1, acc_ref = refs

            # TODO: We could use select_ref here, but we don't
            # handle it yet in lowering. I.e.
            # b_smem = pl.select_ref(wg_idx, b_smem0, b_smem1)
            def zero():
              do_wgmma(acc_ref, a_smem, b_smem0)

            def one():
              do_wgmma(acc_ref, a_smem, b_smem1)

            jax.lax.cond(wg_idx == 0, zero, one)
          else:
            a_smem, b_smem, acc_ref = refs
            do_wgmma(acc_ref, a_smem.at[wg_m_slice], b_smem)
          plgpu.wgmma_wait(0)
          return acc_ref

        b_cta_gmem = (
            (b_gmem[0].at[:, cta_n_slice], b_gmem[1].at[:, cta_n_slice])
            if use_2d_rhs
            else b_gmem.at[:, :, cta_n_slice]
        )
        get_pipeline(mma_body, compute_context)(
            a_gmem.at[cta_m_slice, :],
            *(
                [b_cta_gmem]
                * (1 + (config.wg_dimension == common.MatmulDimension.N))
            ),
            allocations=pipeline_allocs,
        )

    # Await all transfers before we exit.
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  out_swizzle = plgpu.find_swizzle(epi_tile_n * out_itemsize * 8)
  out_swizzle_elems = out_swizzle // out_itemsize
  out_transforms = (
      plgpu.TilingTransform((8, out_swizzle_elems)),
      plgpu.SwizzleTransform(out_swizzle),
  )
  scratch_shapes = dict(
      out_smem=plgpu.SMEM(
          (2, num_out_slots, epi_tile_m, epi_tile_n),
          store_dtype,
          transforms=out_transforms,
      ),
  )
  num_sms = backend.get_default_device().core_count
  cluster_size = config.cluster_size_m * config.cluster_size_n
  grid_size = max(1, min(num_sms // cluster_size, m_iters * n_iters))
  out_struct = jax.ShapeDtypeStruct((m_padded, n), store_dtype)
  out_type = (
      (out_struct, jax.ShapeDtypeStruct((m_padded, 2, n), store_dtype))
      if return_residuals
      else out_struct
  )
  f = plgpu.kernel(
      kernel,
      out_type=out_type,
      grid=(grid_size,),
      grid_names=("cluster_grid",),
      cluster=(cluster_size,),
      cluster_names=("cluster",),
      num_threads=3,
      thread_name="wg",
      scratch_types=scratch_shapes,
      compiler_params=plgpu.CompilerParams(approx_math=approx_math),
  )
  result = f(x, weights)
  if return_residuals:
    out, res = result
    if m_padded != m:
      out = out[:m]
      res = res[:m]
    return (
        jnp.reshape(out, (*orig_x_shape[:-1], n)),
        jnp.reshape(res, (*orig_x_shape[:-1], 2, n)),
    )
  if m_padded != m:
    result = result[:m]
  return jnp.reshape(result, (*orig_x_shape[:-1], n))
