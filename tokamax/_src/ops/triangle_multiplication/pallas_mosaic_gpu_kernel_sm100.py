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
"""Pallas-Mosaic-GPU Triangle Multiplication SM100 (Blackwell B200) kernels."""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src.ops import op
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_gpu_common as common


def _extract_dims(ba: op.BoundArguments) -> tuple[int, int, int, int]:
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  return x.shape[-3], x.shape[-1], w_out.shape[-2], w_out.shape[-1]


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns default SM100 forward config."""
  n, c, h, d = _extract_dims(ba)
  if max(c, h, d) > 64:
    return common.Config(
        block_m=128,
        num_warps=4,
        num_stages=2,
        prologue_persistent=True,
        epilogue_persistent=False,
        ctas_per_sm=1,
    )
  del n
  return common.Config(
      block_m=128,
      num_warps=4,
      num_stages=2,
      prologue_persistent=True,
      epilogue_persistent=True,
      ctas_per_sm=2,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns SM100 forward autotuning configs."""
  _, c, h, d = _extract_dims(ba)
  configs = {get_heuristics_config(ba)}
  if max(c, h, d) > 64:
    for persistent in (True, False):
      configs.add(
          common.Config(
              block_m=128,
              num_warps=4,
              num_stages=2,
              prologue_persistent=persistent,
              epilogue_persistent=False,
              ctas_per_sm=1,
          )
      )
    return configs

  for block_m in (64, 128):
    for persistent in (True, False):
      configs.add(
          common.Config(
              block_m=block_m,
              num_warps=4,
              num_stages=2,
              prologue_persistent=persistent,
              epilogue_persistent=persistent,
              ctas_per_sm=2,
          )
      )
  return configs


def get_vjp_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns default SM100 backward config."""
  _, c, h, d = _extract_dims(ba)
  if max(c, h, d) > 64:
    return common.Config(
        block_m=128,
        num_warps=4,
        num_stages=2,
        prologue_persistent=True,
        epilogue_persistent=True,
        ctas_per_sm=1,
    )
  return common.Config(
      block_m=64,
      num_warps=4,
      num_stages=2,
      prologue_persistent=True,
      epilogue_persistent=True,
      ctas_per_sm=2,
  )


def get_vjp_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns SM100 backward autotuning configs."""
  _, c, h, d = _extract_dims(ba)
  configs = {get_vjp_heuristics_config(ba)}
  if max(c, h, d) > 64:
    return configs

  for block_m, ctas_per_sm in ((64, 2), (64, 1), (128, 1)):
    configs.add(
        common.Config(
            block_m=block_m,
            num_warps=4,
            num_stages=2,
            prologue_persistent=True,
            epilogue_persistent=True,
            ctas_per_sm=ctas_per_sm,
        )
    )
  return configs


def _resolve_grid_m(
    num_tiles: int,
    num_sms: int,
    ctas_per_sm: int,
    num_warps: int,
    persistent: bool,
    max_ctas_per_sm: int,
) -> int:
  if not persistent:
    return num_tiles
  eff_ctas = min(max_ctas_per_sm, max(ctas_per_sm, num_warps // 4))
  return min(num_tiles, num_sms * max(1, eff_ctas))


def _smem_for_mma(shape: tuple[int, int], dtype, min_m: int) -> pl.MemoryRef:
  """Allocates tiled/swizzled SMEM compatible with tcgen05_mma for `min_m`."""
  if min_m == 64:
    swizzle = min(64, shape[1] * jnp.dtype(dtype).itemsize)
    return mgpu_lib.tiled_swizzled_smem(shape, dtype, swizzle=swizzle)
  return mgpu_lib.tiled_swizzled_smem(shape, dtype)


def prologue(
    x_flat: jax.Array,
    mask_flat: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    *,
    eps: float,
    c_real: int,
    h_real: int,
    block_m: int,
    m_real: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    persistent: bool = True,
    ctas_per_sm: int = 2,
    **unused_kwargs,
) -> tuple[jax.Array, jax.Array]:
  """SM100 TMA-pipelined prologue kernel using tcgen05_mma & TMEM."""
  del unused_kwargs, h_real
  m, c = x_flat.shape
  m_valid = m if m_real is None else m_real
  h = projection_in_weights.shape[2]
  dtype = x_flat.dtype
  num_sms = backend.get_default_device().core_count

  w_proj_2d = projection_in_weights.reshape(c, 2 * h)
  w_gate_2d = gate_in_weights.reshape(c, 2 * h)

  if max(c, h) > 64:
    tile_m = 64
    stages = 2
    num_tiles = m // tile_m
    grid_m = _resolve_grid_m(
        num_tiles, num_sms, ctas_per_sm, num_warps, persistent, 1
    )
    vec_layout = plgpu.Layout.WGMMA

    def kernel_large_ch(
        x_gmem,
        mask_gmem,
        w_proj_gmem,
        w_gate_gmem,
        ln_scale_gmem,
        ln_offset_gmem,
        left_gmem,
        right_gmem,
        x_smem,
        w_pl_smem,
        w_pr_smem,
        w_gl_smem,
        w_gr_smem,
        act_smem,
        out_smem,
        p_tmem,
        g_tmem,
        x_barrier,
        w_barrier,
        mma_barrier,
    ):
      pid = lax.axis_index("block")
      num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
          pid < lax.rem(num_tiles, grid_m), jnp.int32
      )

      plgpu.copy_gmem_to_smem(w_proj_gmem.at[:, :h], w_pl_smem, w_barrier)
      plgpu.copy_gmem_to_smem(w_proj_gmem.at[:, h:], w_pr_smem, w_barrier)
      plgpu.copy_gmem_to_smem(w_gate_gmem.at[:, :h], w_gl_smem, w_barrier)
      plgpu.copy_gmem_to_smem(w_gate_gmem.at[:, h:], w_gr_smem, w_barrier)

      for s in range(stages):
        init_tile = pid + s * grid_m
        init_slice = pl.ds(init_tile * tile_m, tile_m)
        common.copy_gmem_to_smem(
            x_gmem.at[init_slice, :],
            x_smem.at[s],
            x_barrier.at[s],
            predicate=init_tile < num_tiles,
        )

      ln_scale = plgpu.load(
          ln_scale_gmem, layout=vec_layout.reduce(0), optimized=False
      )
      ln_offset = plgpu.load(
          ln_offset_gmem, layout=vec_layout.reduce(0), optimized=False
      )
      plgpu.barrier_wait(w_barrier)

      def loop_body(step, _):
        slot = lax.rem(step, jnp.int32(stages))
        tile_id = pid + step * grid_m
        row_start = tile_id * tile_m
        next_tile = tile_id + stages * grid_m
        next_slice = pl.ds(next_tile * tile_m, tile_m)
        next_pred = next_tile < num_tiles
        slice_m = pl.ds(row_start, tile_m)

        mask_val = (
            plgpu.load(
                mask_gmem.at[slice_m],
                layout=vec_layout.reduce(1),
                optimized=False,
            )
            != 0
        )
        mask_val = common.mask_rows_1d(
            mask_val, row_start, m_valid, m, vec_layout.reduce(1)
        )
        plgpu.barrier_wait(x_barrier.at[slot])
        x = plgpu.load(x_smem.at[slot], layout=vec_layout)
        x = common.mask_rows_2d(x, row_start, m_valid, m, vec_layout)
        common.warpgroup_barrier_and_commit_smem()
        common.copy_gmem_to_smem(
            x_gmem.at[next_slice, :],
            x_smem.at[slot],
            x_barrier.at[slot],
            predicate=next_pred,
        )

        mask_b = plgpu.layout_cast(
            lax.broadcast_in_dim(mask_val, (tile_m, c), (0,)), vec_layout
        )
        act = common.layer_norm_2d(
            x, ln_scale, ln_offset, eps, c_real, vec_layout
        )
        act = jnp.where(mask_b, act, 0.0).astype(dtype)
        act_smem[...] = act
        plgpu.commit_smem()

        plgpu.tcgen05_mma(
            g_tmem, act_smem, w_gl_smem, mma_barrier.at[0], accumulate=False
        )
        plgpu.tcgen05_mma(
            p_tmem, act_smem, w_pl_smem, mma_barrier.at[1], accumulate=False
        )
        plgpu.barrier_wait(mma_barrier.at[0])
        g_l = plgpu.async_load_tmem(g_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        s_l = common.sigmoid_fwd_f32(g_l, vec_layout)

        plgpu.barrier_wait(mma_barrier.at[1])
        p_l = plgpu.async_load_tmem(p_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        left = (p_l * s_l).astype(dtype)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        out_smem.T[...] = left
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(out_smem, left_gmem.at[:, slice_m])

        plgpu.tcgen05_mma(
            g_tmem, act_smem, w_gr_smem, mma_barrier.at[2], accumulate=False
        )
        plgpu.tcgen05_mma(
            p_tmem, act_smem, w_pr_smem, mma_barrier.at[3], accumulate=False
        )
        plgpu.barrier_wait(mma_barrier.at[2])
        g_r = plgpu.async_load_tmem(g_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        s_r = common.sigmoid_fwd_f32(g_r, vec_layout)

        plgpu.barrier_wait(mma_barrier.at[3])
        p_r = plgpu.async_load_tmem(p_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        right = (p_r * s_r).astype(dtype)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        out_smem.T[...] = right
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(out_smem, right_gmem.at[:, slice_m])
        common.warpgroup_barrier()

      lax.fori_loop(0, num_local_steps, loop_body, None)
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    out_shape = jax.ShapeDtypeStruct((h, m), dtype)
    scratch_types = (
        mgpu_lib.tiled_swizzled_smem((stages, tile_m, c), dtype),
        _smem_for_mma((c, h), dtype, tile_m),
        _smem_for_mma((c, h), dtype, tile_m),
        _smem_for_mma((c, h), dtype, tile_m),
        _smem_for_mma((c, h), dtype, tile_m),
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
        plgpu.TMEM((tile_m, h), jnp.float32),
        plgpu.TMEM((tile_m, h), jnp.float32),
        plgpu.Barrier(num_arrivals=1, num_barriers=stages),
        plgpu.Barrier(num_arrivals=4),
        plgpu.Barrier(num_arrivals=1, num_barriers=4, orders_tensor_core=True),
    )
    return plgpu.kernel(
        kernel_large_ch,
        out_type=(out_shape, out_shape),
        scratch_types=scratch_types,
        grid=(grid_m,),
        grid_names=("block",),
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(
        x_flat,
        mask_flat,
        w_proj_2d,
        w_gate_2d,
        layernorm_in_scale,
        layernorm_in_offset,
    )

  tile_m = max(64, min(128, block_m))
  if m % tile_m != 0:
    tile_m = 64
  max_stages = 2 if tile_m == 128 else 3
  stages = max(2, min(max_stages, num_stages))
  num_tiles = m // tile_m
  grid_m = _resolve_grid_m(
      num_tiles, num_sms, ctas_per_sm, num_warps, persistent, 2
  )
  vec_layout = plgpu.Layout.TCGEN05 if tile_m == 128 else plgpu.Layout.WGMMA

  # 96 KB SMEM footprint (at tile_m=128, stages=2) + 256 TMEM cols -> FULL
  # 2 CTAs/SM on B200 with overlapped 2H MMAs, dual TMA stores, and single-MUFU
  # sigmoid!
  def kernel(
      x_gmem,
      mask_gmem,
      w_proj_gmem,
      w_gate_gmem,
      ln_scale_gmem,
      ln_offset_gmem,
      left_gmem,
      right_gmem,
      x_smem,
      w_proj_smem,
      w_gate_smem,
      left_out_smem,
      right_out_smem,
      p_tmem,
      g_tmem,
      x_barrier,
      w_barrier,
      mma_barrier,
  ):
    pid = lax.axis_index("block")
    num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
        pid < lax.rem(num_tiles, grid_m), jnp.int32
    )

    plgpu.copy_gmem_to_smem(w_proj_gmem, w_proj_smem, w_barrier)
    plgpu.copy_gmem_to_smem(w_gate_gmem, w_gate_smem, w_barrier)

    for s in range(stages):
      init_tile = pid + s * grid_m
      init_slice = pl.ds(init_tile * tile_m, tile_m)
      common.copy_gmem_to_smem(
          x_gmem.at[init_slice, :],
          x_smem.at[s],
          x_barrier.at[s],
          predicate=init_tile < num_tiles,
      )

    ln_scale = plgpu.load(
        ln_scale_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    ln_offset = plgpu.load(
        ln_offset_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    plgpu.barrier_wait(w_barrier)

    def loop_body(step, _):
      slot = lax.rem(step, jnp.int32(stages))
      tile_id = pid + step * grid_m
      row_start = tile_id * tile_m
      next_tile = tile_id + stages * grid_m
      next_slice = pl.ds(next_tile * tile_m, tile_m)
      next_pred = next_tile < num_tiles
      slice_m = pl.ds(row_start, tile_m)

      mask_val = (
          plgpu.load(
              mask_gmem.at[slice_m],
              layout=vec_layout.reduce(1),
              optimized=False,
          )
          != 0
      )
      mask_val = common.mask_rows_1d(
          mask_val, row_start, m_valid, m, vec_layout.reduce(1)
      )
      plgpu.barrier_wait(x_barrier.at[slot])
      x = plgpu.load(x_smem.at[slot], layout=vec_layout)
      x = common.mask_rows_2d(x, row_start, m_valid, m, vec_layout)
      common.warpgroup_barrier()

      mask_b = plgpu.layout_cast(
          lax.broadcast_in_dim(mask_val, (tile_m, c), (0,)), vec_layout
      )
      act = common.layer_norm_2d(
          x, ln_scale, ln_offset, eps, c_real, vec_layout
      )
      act = jnp.where(mask_b, act, 0.0).astype(dtype)
      x_smem.at[slot][...] = act
      plgpu.commit_smem()

      plgpu.tcgen05_mma(
          g_tmem,
          x_smem.at[slot],
          w_gate_smem,
          mma_barrier.at[0],
          accumulate=False,
      )
      plgpu.tcgen05_mma(
          p_tmem,
          x_smem.at[slot],
          w_proj_smem,
          mma_barrier.at[1],
          accumulate=False,
      )
      if tile_m == 128:
        plgpu.barrier_wait(mma_barrier.at[0])
        g_l = plgpu.async_load_tmem(g_tmem.at[:, :h], layout=vec_layout)
        g_r = plgpu.async_load_tmem(g_tmem.at[:, h:], layout=vec_layout)
        plgpu.wait_load_tmem()
        s_l = common.sigmoid_fwd_f32(g_l, vec_layout)
        s_r = common.sigmoid_fwd_f32(g_r, vec_layout)

        plgpu.barrier_wait(mma_barrier.at[1])
        common.copy_gmem_to_smem(
            x_gmem.at[next_slice, :],
            x_smem.at[slot],
            x_barrier.at[slot],
            predicate=next_pred,
        )
        p_l = plgpu.async_load_tmem(p_tmem.at[:, :h], layout=vec_layout)
        p_r = plgpu.async_load_tmem(p_tmem.at[:, h:], layout=vec_layout)
        plgpu.wait_load_tmem()
        left = (p_l * s_l).astype(dtype)
        right = (p_r * s_r).astype(dtype)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        left_out_smem.T[...] = left
        right_out_smem.T[...] = right
      else:
        plgpu.barrier_wait(mma_barrier.at[0])
        g_lr = plgpu.async_load_tmem(g_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        s_lr = common.sigmoid_fwd_f32(g_lr, vec_layout)

        plgpu.barrier_wait(mma_barrier.at[1])
        common.copy_gmem_to_smem(
            x_gmem.at[next_slice, :],
            x_smem.at[slot],
            x_barrier.at[slot],
            predicate=next_pred,
        )
        p_lr = plgpu.async_load_tmem(p_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        lr = (p_lr * s_lr).astype(dtype)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        left_out_smem.T[...] = lr[:, :h]
        right_out_smem.T[...] = lr[:, h:]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          left_out_smem, left_gmem.at[:, slice_m], commit_group=False
      )
      plgpu.copy_smem_to_gmem(right_out_smem, right_gmem.at[:, slice_m])

    lax.fori_loop(0, num_local_steps, loop_body, None)
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  out_shape = jax.ShapeDtypeStruct((h, m), dtype)
  scratch_types = (
      mgpu_lib.tiled_swizzled_smem((stages, tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((c, 2 * h), dtype),
      mgpu_lib.tiled_swizzled_smem((c, 2 * h), dtype),
      mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
      mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
      plgpu.TMEM((tile_m, 2 * h), jnp.float32),
      plgpu.TMEM((tile_m, 2 * h), jnp.float32),
      plgpu.Barrier(num_arrivals=1, num_barriers=stages),
      plgpu.Barrier(num_arrivals=2),
      plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
  )
  return plgpu.kernel(
      kernel,
      out_type=(out_shape, out_shape),
      scratch_types=scratch_types,
      grid=(grid_m,),
      grid_names=("block",),
      compiler_params=plgpu.CompilerParams(approx_math=True),
  )(
      x_flat,
      mask_flat,
      w_proj_2d,
      w_gate_2d,
      layernorm_in_scale,
      layernorm_in_offset,
  )


def epilogue(
    t3_flat: jax.Array,
    x_flat: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    *,
    eps: float,
    c_real: int,
    h_real: int,
    block_m: int,
    d_real: int | None = None,
    m_real: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    persistent: bool = True,
    ctas_per_sm: int = 2,
    **unused_kwargs,
) -> jax.Array:
  """SM100 TMA-pipelined epilogue kernel using tcgen05_mma & TMEM."""
  del unused_kwargs
  m, c = x_flat.shape
  m_valid = m if m_real is None else m_real
  out_m = m if common.is_interpret_mode() else m_valid
  h, d = projection_out_weights.shape
  d_out = d if d_real is None else d_real
  dtype = x_flat.dtype
  num_sms = backend.get_default_device().core_count

  if max(c, h, d) > 64:
    tile_m = 128
    num_tiles = m // tile_m
    grid_m = _resolve_grid_m(
        num_tiles, num_sms, ctas_per_sm, num_warps, persistent, 1
    )
    tc_layout = plgpu.Layout.TCGEN05

    def kernel_large_chd(
        t3_gmem,
        x_gmem,
        w_proj_out_gmem,
        w_gate_out_gmem,
        ln_in_scale_gmem,
        ln_in_offset_gmem,
        ln_out_scale_gmem,
        ln_out_offset_gmem,
        out_gmem,
        x_smem,
        t3_smem,
        w_proj_out_smem,
        w_gate_out_smem,
        left_act_smem,
        act_smem,
        out_smem,
        g_tmem,
        p_tmem,
        in_barrier,
        w_barrier,
        mma_barrier,
    ):
      pid = lax.axis_index("block")
      num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
          pid < lax.rem(num_tiles, grid_m), jnp.int32
      )

      plgpu.copy_gmem_to_smem(w_proj_out_gmem, w_proj_out_smem, w_barrier)
      plgpu.copy_gmem_to_smem(w_gate_out_gmem, w_gate_out_smem, w_barrier)

      ln_in_scale = plgpu.load(
          ln_in_scale_gmem, layout=tc_layout.reduce(0), optimized=False
      )
      ln_in_offset = plgpu.load(
          ln_in_offset_gmem, layout=tc_layout.reduce(0), optimized=False
      )
      ln_out_scale = plgpu.load(
          ln_out_scale_gmem, layout=tc_layout.reduce(0), optimized=False
      )
      ln_out_offset = plgpu.load(
          ln_out_offset_gmem, layout=tc_layout.reduce(0), optimized=False
      )
      plgpu.barrier_wait(w_barrier)

      def loop_body(step, _):
        tile_id = pid + step * grid_m
        row_start = tile_id * tile_m
        slice_m = pl.ds(row_start, tile_m)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.copy_gmem_to_smem(x_gmem.at[slice_m, :], x_smem, in_barrier)
        plgpu.copy_gmem_to_smem(t3_gmem.at[:, slice_m], t3_smem, in_barrier)
        plgpu.barrier_wait(in_barrier)

        x = plgpu.load(x_smem, layout=tc_layout)
        x = common.mask_rows_2d(x, row_start, m_valid, m, tc_layout)
        left_act = common.layer_norm_2d(
            x, ln_in_scale, ln_in_offset, eps, c_real, tc_layout
        ).astype(dtype)
        left_act_smem[...] = left_act
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            g_tmem,
            left_act_smem,
            w_gate_out_smem,
            mma_barrier.at[0],
            accumulate=False,
        )

        t3_t = plgpu.load(t3_smem.T, layout=tc_layout)
        t3_t = common.mask_rows_2d(t3_t, row_start, m_valid, m, tc_layout)
        common.warpgroup_barrier()
        act_out = common.layer_norm_2d(
            t3_t, ln_out_scale, ln_out_offset, eps, h_real, tc_layout
        ).astype(dtype)
        act_smem[...] = act_out
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            p_tmem,
            act_smem,
            w_proj_out_smem,
            mma_barrier.at[1],
            accumulate=False,
        )

        plgpu.barrier_wait(mma_barrier.at[0])
        g = plgpu.async_load_tmem(g_tmem, layout=tc_layout)
        plgpu.wait_load_tmem()
        gate = common.sigmoid_fwd_f32(g, tc_layout)

        plgpu.barrier_wait(mma_barrier.at[1])
        p = plgpu.async_load_tmem(p_tmem, layout=tc_layout)
        plgpu.wait_load_tmem()

        out = (p * gate).astype(dtype)
        out_smem[...] = out[:, :d_out] if d_out < d else out
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[slice_m, :])

      lax.fori_loop(0, num_local_steps, loop_body, None)
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    scratch_types = (
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
        mgpu_lib.tiled_swizzled_smem((h, d), dtype),
        mgpu_lib.tiled_swizzled_smem((c, d), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, h), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, d_out), dtype),
        plgpu.TMEM((tile_m, d), jnp.float32),
        plgpu.TMEM((tile_m, d), jnp.float32),
        plgpu.Barrier(num_arrivals=2),
        plgpu.Barrier(num_arrivals=2),
        plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
    )
    return plgpu.kernel(
        kernel_large_chd,
        out_type=jax.ShapeDtypeStruct((out_m, d_out), dtype),
        scratch_types=scratch_types,
        grid=(grid_m,),
        grid_names=("block",),
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(
        t3_flat,
        x_flat,
        projection_out_weights,
        gate_out_weights,
        layernorm_in_scale,
        layernorm_in_offset,
        layernorm_out_scale,
        layernorm_out_offset,
    )

  tile_m = max(64, min(128, block_m))
  if m % tile_m != 0:
    tile_m = 64
  max_stages = 2 if tile_m == 128 else 3
  stages = max(2, min(max_stages, num_stages))
  num_tiles = m // tile_m
  grid_m = _resolve_grid_m(
      num_tiles, num_sms, ctas_per_sm, num_warps, persistent, 2
  )
  vec_layout = plgpu.Layout.TCGEN05 if tile_m == 128 else plgpu.Layout.WGMMA
  share_work_out = d_out == d

  # 96 KB SMEM footprint (at tile_m=128, stages=2, d_out==d) -> FULL 2 CTAs/SM
  # on B200, overlapping g_tmem MMA with layer_norm(t3) and p_tmem MMA with
  # sigmoid(g)!
  def kernel(
      t3_gmem,
      x_gmem,
      w_proj_out_gmem,
      w_gate_out_gmem,
      ln_in_scale_gmem,
      ln_in_offset_gmem,
      ln_out_scale_gmem,
      ln_out_offset_gmem,
      out_gmem,
      x_smem,
      t3_smem,
      w_proj_out_smem,
      w_gate_out_smem,
      work_smem,
      opt_out_smem,
      g_tmem,
      p_tmem,
      in_barrier,
      w_barrier,
      mma_barrier,
  ):
    pid = lax.axis_index("block")
    num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
        pid < lax.rem(num_tiles, grid_m), jnp.int32
    )

    plgpu.copy_gmem_to_smem(w_proj_out_gmem, w_proj_out_smem, w_barrier)
    plgpu.copy_gmem_to_smem(w_gate_out_gmem, w_gate_out_smem, w_barrier)

    for s in range(stages):
      init_tile = pid + s * grid_m
      init_slice = pl.ds(init_tile * tile_m, tile_m)
      pred = init_tile < num_tiles
      common.copy_gmem_to_smem(
          x_gmem.at[init_slice, :],
          x_smem.at[s],
          in_barrier.at[s],
          predicate=pred,
      )
      common.copy_gmem_to_smem(
          t3_gmem.at[:, init_slice],
          t3_smem.at[s],
          in_barrier.at[s],
          predicate=pred,
      )

    ln_in_scale = plgpu.load(
        ln_in_scale_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    ln_in_offset = plgpu.load(
        ln_in_offset_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    ln_out_scale = plgpu.load(
        ln_out_scale_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    ln_out_offset = plgpu.load(
        ln_out_offset_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    plgpu.barrier_wait(w_barrier)

    def loop_body(step, _):
      slot = lax.rem(step, jnp.int32(stages))
      tile_id = pid + step * grid_m
      row_start = tile_id * tile_m
      next_tile = tile_id + stages * grid_m
      next_slice = pl.ds(next_tile * tile_m, tile_m)
      next_pred = next_tile < num_tiles

      plgpu.barrier_wait(in_barrier.at[slot])
      x = plgpu.load(x_smem.at[slot], layout=vec_layout)
      x = common.mask_rows_2d(x, row_start, m_valid, m, vec_layout)
      t3_t = plgpu.load(t3_smem.at[slot].T, layout=vec_layout)
      t3_t = common.mask_rows_2d(t3_t, row_start, m_valid, m, vec_layout)
      common.warpgroup_barrier_and_commit_smem()
      common.copy_gmem_to_smem(
          t3_gmem.at[:, next_slice],
          t3_smem.at[slot],
          in_barrier.at[slot],
          predicate=next_pred,
      )

      left_act = common.layer_norm_2d(
          x, ln_in_scale, ln_in_offset, eps, c_real, vec_layout
      ).astype(dtype)
      x_smem.at[slot][...] = left_act
      plgpu.commit_smem()
      plgpu.tcgen05_mma(
          g_tmem,
          x_smem.at[slot],
          w_gate_out_smem,
          mma_barrier.at[0],
          accumulate=False,
      )

      act_out = common.layer_norm_2d(
          t3_t, ln_out_scale, ln_out_offset, eps, h_real, vec_layout
      ).astype(dtype)
      if share_work_out:
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
      work_smem.at[:, :h][...] = act_out
      plgpu.commit_smem()
      plgpu.tcgen05_mma(
          p_tmem,
          work_smem.at[:, :h],
          w_proj_out_smem,
          mma_barrier.at[1],
          accumulate=False,
      )

      plgpu.barrier_wait(mma_barrier.at[0])
      common.copy_gmem_to_smem(
          x_gmem.at[next_slice, :],
          x_smem.at[slot],
          in_barrier.at[slot],
          predicate=next_pred,
      )
      g = plgpu.async_load_tmem(g_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()
      gate = common.sigmoid_fwd_f32(g, vec_layout)

      plgpu.barrier_wait(mma_barrier.at[1])
      p = plgpu.async_load_tmem(p_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()

      out = (p * gate).astype(dtype)
      slice_m = pl.ds(row_start, tile_m)
      if share_work_out:
        out_ref = work_smem.at[:, :d]
        out_ref[...] = out
      else:
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        out_ref = opt_out_smem
        out_ref[...] = out[:, :d_out]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(out_ref, out_gmem.at[slice_m, :])

    lax.fori_loop(0, num_local_steps, loop_body, None)
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  max_hd = max(h, d)
  opt_out_scratch = (
      plgpu.Barrier(num_arrivals=1)
      if share_work_out
      else mgpu_lib.tiled_swizzled_smem((tile_m, d_out), dtype)
  )
  scratch_types = (
      mgpu_lib.tiled_swizzled_smem((stages, tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((stages, h, tile_m), dtype),
      _smem_for_mma((h, d), dtype, tile_m),
      _smem_for_mma((c, d), dtype, tile_m),
      mgpu_lib.tiled_swizzled_smem((tile_m, max_hd), dtype),
      opt_out_scratch,
      plgpu.TMEM((tile_m, d), jnp.float32),
      plgpu.TMEM((tile_m, d), jnp.float32),
      plgpu.Barrier(num_arrivals=2, num_barriers=stages),
      plgpu.Barrier(num_arrivals=2),
      plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
  )
  return plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((out_m, d_out), dtype),
      scratch_types=scratch_types,
      grid=(grid_m,),
      grid_names=("block",),
      compiler_params=plgpu.CompilerParams(approx_math=True),
  )(
      t3_flat,
      x_flat,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )


def epilogue_bwd(
    dout_flat: jax.Array,
    t3_flat: jax.Array,
    x_flat: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    *,
    eps: float,
    c_real: int,
    h_real: int,
    block_m: int,
    m_real: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    persistent: bool = True,
    ctas_per_sm: int = 2,
    **unused_kwargs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """SM100 zero-spill native M=64/128 TMA-pipelined epilogue backward kernel."""
  del unused_kwargs
  m, c = x_flat.shape
  m_valid = m if m_real is None else m_real
  h, d = projection_out_weights.shape
  dtype = x_flat.dtype
  num_sms = backend.get_default_device().core_count
  max_hd = max(h, d)

  if max(c, h, d) > 64:
    tile_m = 128
    num_tiles = m // tile_m
    grid_m = _resolve_grid_m(
        num_tiles, num_sms, ctas_per_sm, num_warps, persistent, 1
    )
    tc_layout = plgpu.Layout.TCGEN05
    proj_layout = plgpu.Layout.TCGEN05 if h == 128 else plgpu.Layout.WGMMA
    gate_layout = plgpu.Layout.TCGEN05 if c == 128 else plgpu.Layout.WGMMA
    min_dw_m = min(h, c)

    def kernel_large_chd(
        dout_gmem,
        t3_gmem,
        x_gmem,
        w_proj_out_gmem,
        w_gate_out_gmem,
        ln_in_scale_gmem,
        ln_in_offset_gmem,
        ln_out_scale_gmem,
        ln_out_offset_gmem,
        dt3_gmem,
        dleft_act_epi_gmem,
        grad_part_gmem,
        x_smem,
        t3_smem,
        dout_smem,
        w_proj_out_smem,
        w_gate_out_smem,
        act_smem,
        left_act_smem,
        dw_proj_tmem,
        dw_gate_tmem,
        work0_tmem,
        work1_tmem,
        in_barrier,
        w_barrier,
        mma_barrier,
    ):
      pid = lax.axis_index("block")
      num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
          pid < lax.rem(num_tiles, grid_m), jnp.int32
      )

      plgpu.copy_gmem_to_smem(w_proj_out_gmem, w_proj_out_smem, w_barrier)
      plgpu.copy_gmem_to_smem(w_gate_out_gmem, w_gate_out_smem, w_barrier)

      ln_in_scale = plgpu.load(
          ln_in_scale_gmem, layout=tc_layout.reduce(0), optimized=False
      )
      ln_in_offset = plgpu.load(
          ln_in_offset_gmem, layout=tc_layout.reduce(0), optimized=False
      )
      ln_out_scale = plgpu.load(
          ln_out_scale_gmem, layout=tc_layout.reduce(0), optimized=False
      )
      ln_out_offset = plgpu.load(
          ln_out_offset_gmem, layout=tc_layout.reduce(0), optimized=False
      )
      plgpu.barrier_wait(w_barrier)

      init_acc_s, init_acc_o = common.init_ln_grad_acc(h, tc_layout)

      def loop_body(step, carry):
        acc_s, acc_o = carry
        tile_id = pid + step * grid_m
        row_start = tile_id * tile_m
        slice_m = pl.ds(row_start, tile_m)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.copy_gmem_to_smem(x_gmem.at[slice_m, :], x_smem, in_barrier)
        plgpu.copy_gmem_to_smem(t3_gmem.at[:, slice_m], t3_smem, in_barrier)
        plgpu.copy_gmem_to_smem(dout_gmem.at[slice_m, :], dout_smem, in_barrier)
        plgpu.barrier_wait(in_barrier)

        x = plgpu.load(x_smem, layout=tc_layout)
        x = common.mask_rows_2d(x, row_start, m_valid, m, tc_layout)
        left_act = common.layer_norm_2d(
            x, ln_in_scale, ln_in_offset, eps, c_real, tc_layout
        ).astype(dtype)
        left_act_smem[...] = left_act
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            work0_tmem,
            left_act_smem,
            w_gate_out_smem,
            mma_barrier.at[0],
            accumulate=False,
        )

        t3_t = plgpu.load(t3_smem.T, layout=tc_layout)
        t3_t = common.mask_rows_2d(t3_t, row_start, m_valid, m, tc_layout)
        dout = plgpu.load(dout_smem, layout=tc_layout).astype(jnp.float32)
        dout = common.mask_rows_2d(dout, row_start, m_valid, m, tc_layout)
        common.warpgroup_barrier()

        t_hat, rstd_out = common.layer_norm_hat_rstd(
            t3_t, eps, h_real, tc_layout
        )
        act = common.apply_norm_affine(
            t_hat, ln_out_scale, ln_out_offset, tc_layout
        ).astype(dtype)
        act_smem[...] = act
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            work1_tmem.at[:, :d],
            act_smem,
            w_proj_out_smem,
            mma_barrier.at[1],
            accumulate=False,
        )

        plgpu.barrier_wait(mma_barrier.at[0])
        g_raw = plgpu.async_load_tmem(work0_tmem, layout=tc_layout)
        plgpu.wait_load_tmem()
        gate = common.sigmoid_f32(g_raw)
        dp_f32 = dout * gate
        dp = dp_f32.astype(dtype)
        q = dp_f32 * (1.0 - gate)

        plgpu.barrier_wait(mma_barrier.at[1])
        p = plgpu.async_load_tmem(work1_tmem.at[:, :d], layout=tc_layout)
        plgpu.wait_load_tmem()
        dg = (q * p).astype(dtype)

        acc_flag = step > 0
        dout_smem[...] = dp
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            dw_proj_tmem, act_smem.T, dout_smem, accumulate=acc_flag
        )
        plgpu.tcgen05_mma(
            work1_tmem.at[:, :h],
            dout_smem,
            w_proj_out_smem.T,
            accumulate=False,
        )
        plgpu.tcgen05_commit_arrive(mma_barrier.at[2])
        plgpu.barrier_wait(mma_barrier.at[2])
        dact = plgpu.async_load_tmem(work1_tmem.at[:, :h], layout=tc_layout)
        plgpu.wait_load_tmem()

        dt3_t, acc_s, acc_o = common.layer_norm_2d_bwd_accum(
            dact, t_hat, rstd_out, ln_out_scale, h_real, tc_layout, acc_s, acc_o
        )
        t3_smem.T[...] = dt3_t.astype(dtype)
        dout_smem[...] = dg
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(
            t3_smem, dt3_gmem.at[:, slice_m], commit_group=False
        )

        plgpu.tcgen05_mma(
            dw_gate_tmem, left_act_smem.T, dout_smem, accumulate=acc_flag
        )
        plgpu.tcgen05_mma(
            work1_tmem.at[:, :c],
            dout_smem,
            w_gate_out_smem.T,
            accumulate=False,
        )
        plgpu.tcgen05_commit_arrive(mma_barrier.at[3])
        plgpu.barrier_wait(mma_barrier.at[3])
        dleft_act = plgpu.async_load_tmem(
            work1_tmem.at[:, :c], layout=tc_layout
        )
        plgpu.wait_load_tmem()

        x_smem[...] = dleft_act.astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(x_smem, dleft_act_epi_gmem.at[slice_m, :])
        return acc_s, acc_o

      acc_s, acc_o = lax.fori_loop(
          0, num_local_steps, loop_body, (init_acc_s, init_acc_o)
      )
      dw_proj = plgpu.async_load_tmem(dw_proj_tmem, layout=proj_layout)
      plgpu.wait_load_tmem()
      mgpu_lib.store(
          grad_part_gmem.at[pid, pl.ds(0, h), :d], dw_proj, optimized=False
      )
      dw_gate = plgpu.async_load_tmem(dw_gate_tmem, layout=gate_layout)
      plgpu.wait_load_tmem()
      mgpu_lib.store(
          grad_part_gmem.at[pid, pl.ds(h, c), :d], dw_gate, optimized=False
      )
      mgpu_lib.store(
          grad_part_gmem.at[pid, pl.ds(h + c, 32), :h], acc_s, optimized=False
      )
      mgpu_lib.store(
          grad_part_gmem.at[pid, pl.ds(h + c + 32, 32), :h],
          acc_o,
          optimized=False,
      )
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    max_chd = max(c, h, d)
    out_type = (
        jax.ShapeDtypeStruct((h, m), dtype),
        jax.ShapeDtypeStruct((m, c), dtype),
        jax.ShapeDtypeStruct((grid_m, h + c + 64, max_hd), jnp.float32),
    )
    scratch_types = (
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
        _smem_for_mma((tile_m, d), dtype, min_dw_m),
        mgpu_lib.tiled_swizzled_smem((h, d), dtype),
        mgpu_lib.tiled_swizzled_smem((c, d), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, h), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        plgpu.TMEM((h, d), jnp.float32),
        plgpu.TMEM((c, d), jnp.float32),
        plgpu.TMEM((tile_m, d), jnp.float32),
        plgpu.TMEM((tile_m, max_chd), jnp.float32),
        plgpu.Barrier(num_arrivals=3),
        plgpu.Barrier(num_arrivals=2),
        plgpu.Barrier(num_arrivals=1, num_barriers=4, orders_tensor_core=True),
    )
    dt3_flat, dleft_act_epi, grad_part = plgpu.kernel(
        kernel_large_chd,
        out_type=out_type,
        scratch_types=scratch_types,
        grid=(grid_m,),
        grid_names=("block",),
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(
        dout_flat,
        t3_flat,
        x_flat,
        projection_out_weights,
        gate_out_weights,
        layernorm_in_scale,
        layernorm_in_offset,
        layernorm_out_scale,
        layernorm_out_offset,
    )
    grad_sum = jnp.sum(grad_part, axis=0)
    return (
        dt3_flat,
        dleft_act_epi,
        grad_sum[:h, :d].astype(dtype),
        grad_sum[h : h + c, :d].astype(dtype),
        jnp.sum(grad_sum[h + c : h + c + 32, :h], axis=0).astype(dtype),
        jnp.sum(grad_sum[h + c + 32 : h + c + 64, :h], axis=0).astype(dtype),
    )

  # Native M=64/128 zero-spill TMEM kernel:
  # - At tile_m=64: 104 KB SMEM + 128 TMEM cols + ~200 regs -> FULL 2 CTAs/SM!
  # - At tile_m=128: native M=64 dw_*_tmem (192 TMEM cols, 0 padding, 0 spills)!
  tile_m = max(64, min(128, block_m))
  if m % tile_m != 0:
    tile_m = 64
  stages = 2 if tile_m == 128 else max(2, min(2, num_stages))
  num_tiles = m // tile_m
  max_ctas = 2 if tile_m == 64 else 1
  grid_m = _resolve_grid_m(
      num_tiles, num_sms, ctas_per_sm, num_warps, persistent, max_ctas
  )
  vec_layout = plgpu.Layout.TCGEN05 if tile_m == 128 else plgpu.Layout.WGMMA
  dw_layout = plgpu.Layout.WGMMA

  def kernel(
      dout_gmem,
      t3_gmem,
      x_gmem,
      w_proj_out_gmem,
      w_gate_out_gmem,
      ln_in_scale_gmem,
      ln_in_offset_gmem,
      ln_out_scale_gmem,
      ln_out_offset_gmem,
      dt3_gmem,
      dleft_act_epi_gmem,
      grad_part_gmem,
      x_smem,
      t3_smem,
      dout_smem,
      w_proj_out_smem,
      w_gate_out_smem,
      act_smem,
      left_act_smem,
      dp_smem,
      dg_smem,
      dt3_smem,
      dw_proj_tmem,
      dw_gate_tmem,
      work0_tmem,
      work1_tmem,
      in_barrier,
      w_barrier,
      mma_barrier,
  ):
    pid = lax.axis_index("block")
    num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
        pid < lax.rem(num_tiles, grid_m), jnp.int32
    )

    plgpu.copy_gmem_to_smem(w_proj_out_gmem, w_proj_out_smem, w_barrier)
    plgpu.copy_gmem_to_smem(w_gate_out_gmem, w_gate_out_smem, w_barrier)

    for s in range(stages):
      init_tile = pid + s * grid_m
      init_slice = pl.ds(init_tile * tile_m, tile_m)
      pred = init_tile < num_tiles
      common.copy_gmem_to_smem(
          x_gmem.at[init_slice, :],
          x_smem.at[s],
          in_barrier.at[s],
          predicate=pred,
      )
      common.copy_gmem_to_smem(
          t3_gmem.at[:, init_slice],
          t3_smem.at[s],
          in_barrier.at[s],
          predicate=pred,
      )
      common.copy_gmem_to_smem(
          dout_gmem.at[init_slice, :],
          dout_smem.at[s],
          in_barrier.at[s],
          predicate=pred,
      )

    ln_in_scale = plgpu.load(
        ln_in_scale_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    ln_in_offset = plgpu.load(
        ln_in_offset_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    ln_out_scale = plgpu.load(
        ln_out_scale_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    ln_out_offset = plgpu.load(
        ln_out_offset_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    plgpu.barrier_wait(w_barrier)

    init_acc_s, init_acc_o = common.init_ln_grad_acc(h, vec_layout)

    def loop_body(step, carry):
      acc_s, acc_o = carry
      slot = lax.rem(step, jnp.int32(stages))
      tile_id = pid + step * grid_m
      row_start = tile_id * tile_m
      next_tile = tile_id + stages * grid_m
      next_slice = pl.ds(next_tile * tile_m, tile_m)
      next_pred = next_tile < num_tiles

      plgpu.barrier_wait(in_barrier.at[slot])
      x = plgpu.load(x_smem.at[slot], layout=vec_layout)
      x = common.mask_rows_2d(x, row_start, m_valid, m, vec_layout)
      left_act = common.layer_norm_2d(
          x, ln_in_scale, ln_in_offset, eps, c_real, vec_layout
      ).astype(dtype)
      left_act_smem[...] = left_act
      plgpu.commit_smem()
      plgpu.tcgen05_mma(
          work0_tmem,
          left_act_smem,
          w_gate_out_smem,
          mma_barrier.at[0],
          accumulate=False,
      )

      t3_t = plgpu.load(t3_smem.at[slot].T, layout=vec_layout)
      t3_t = common.mask_rows_2d(t3_t, row_start, m_valid, m, vec_layout)
      dout = plgpu.load(dout_smem.at[slot], layout=vec_layout).astype(
          jnp.float32
      )
      dout = common.mask_rows_2d(dout, row_start, m_valid, m, vec_layout)
      common.warpgroup_barrier_and_commit_smem()
      common.copy_gmem_to_smem(
          x_gmem.at[next_slice, :],
          x_smem.at[slot],
          in_barrier.at[slot],
          predicate=next_pred,
      )
      common.copy_gmem_to_smem(
          t3_gmem.at[:, next_slice],
          t3_smem.at[slot],
          in_barrier.at[slot],
          predicate=next_pred,
      )
      common.copy_gmem_to_smem(
          dout_gmem.at[next_slice, :],
          dout_smem.at[slot],
          in_barrier.at[slot],
          predicate=next_pred,
      )

      t_hat, rstd_out = common.layer_norm_hat_rstd(
          t3_t, eps, h_real, vec_layout
      )
      act = common.apply_norm_affine(
          t_hat, ln_out_scale, ln_out_offset, vec_layout
      ).astype(dtype)
      act_smem[...] = act
      plgpu.commit_smem()
      plgpu.tcgen05_mma(
          work1_tmem,
          act_smem,
          w_proj_out_smem,
          mma_barrier.at[1],
          accumulate=False,
      )

      plgpu.barrier_wait(mma_barrier.at[0])
      g_raw = plgpu.async_load_tmem(work0_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()
      gate = common.sigmoid_f32(g_raw)
      dp_f32 = dout * gate
      dp = dp_f32.astype(dtype)
      q = dp_f32 * (1.0 - gate)

      plgpu.barrier_wait(mma_barrier.at[1])
      p = plgpu.async_load_tmem(work1_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()
      dg = (q * p).astype(dtype)

      plgpu.wait_smem_to_gmem(0, wait_read_only=True)
      dp_smem[...] = dp
      dg_smem[...] = dg
      plgpu.commit_smem()

      acc_flag = step > 0
      plgpu.tcgen05_mma(
          work0_tmem,
          dp_smem,
          w_proj_out_smem.T,
          mma_barrier.at[2],
          accumulate=False,
      )
      plgpu.tcgen05_mma(
          work1_tmem, dg_smem, w_gate_out_smem.T, accumulate=False
      )
      plgpu.tcgen05_mma(dw_proj_tmem, act_smem.T, dp_smem, accumulate=acc_flag)
      plgpu.tcgen05_mma(
          dw_gate_tmem, left_act_smem.T, dg_smem, accumulate=acc_flag
      )
      plgpu.tcgen05_commit_arrive(mma_barrier.at[3])

      plgpu.barrier_wait(mma_barrier.at[2])
      dact = plgpu.async_load_tmem(work0_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()
      dt3_t, acc_s, acc_o = common.layer_norm_2d_bwd_accum(
          dact, t_hat, rstd_out, ln_out_scale, h_real, vec_layout, acc_s, acc_o
      )

      plgpu.barrier_wait(mma_barrier.at[3])
      dleft_act = plgpu.async_load_tmem(work1_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()

      slice_m = pl.ds(tile_id * tile_m, tile_m)
      dt3_smem.T[...] = dt3_t.astype(dtype)
      dg_smem[...] = dleft_act.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          dt3_smem, dt3_gmem.at[:, slice_m], commit_group=False
      )
      plgpu.copy_smem_to_gmem(dg_smem, dleft_act_epi_gmem.at[slice_m, :])
      return acc_s, acc_o

    acc_s, acc_o = lax.fori_loop(
        0, num_local_steps, loop_body, (init_acc_s, init_acc_o)
    )
    dw_proj = plgpu.async_load_tmem(dw_proj_tmem, layout=dw_layout)
    plgpu.wait_load_tmem()
    mgpu_lib.store(
        grad_part_gmem.at[pid, pl.ds(0, h), :d], dw_proj, optimized=False
    )
    dw_gate = plgpu.async_load_tmem(dw_gate_tmem, layout=dw_layout)
    plgpu.wait_load_tmem()
    mgpu_lib.store(
        grad_part_gmem.at[pid, pl.ds(h, c), :d], dw_gate, optimized=False
    )
    mgpu_lib.store(
        grad_part_gmem.at[pid, pl.ds(h + c, 32), :h], acc_s, optimized=False
    )
    mgpu_lib.store(
        grad_part_gmem.at[pid, pl.ds(h + c + 32, 32), :h],
        acc_o,
        optimized=False,
    )
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  out_type = (
      jax.ShapeDtypeStruct((h, m), dtype),
      jax.ShapeDtypeStruct((m, c), dtype),
      jax.ShapeDtypeStruct((grid_m, h + c + 64, max_hd), jnp.float32),
  )
  scratch_types = (
      mgpu_lib.tiled_swizzled_smem((stages, tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((stages, h, tile_m), dtype),
      mgpu_lib.tiled_swizzled_smem((stages, tile_m, d), dtype),
      _smem_for_mma((h, d), dtype, 64),
      _smem_for_mma((c, d), dtype, 64),
      mgpu_lib.tiled_swizzled_smem((tile_m, h), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
      _smem_for_mma((tile_m, d), dtype, 64),
      _smem_for_mma((tile_m, d), dtype, 64),
      mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
      plgpu.TMEM((h, d), jnp.float32),
      plgpu.TMEM((c, d), jnp.float32),
      plgpu.TMEM((tile_m, d), jnp.float32),
      plgpu.TMEM((tile_m, d), jnp.float32),
      plgpu.Barrier(num_arrivals=3, num_barriers=stages),
      plgpu.Barrier(num_arrivals=2),
      plgpu.Barrier(num_arrivals=1, num_barriers=4, orders_tensor_core=True),
  )
  dt3_flat, dleft_act_epi, grad_part = plgpu.kernel(
      kernel,
      out_type=out_type,
      scratch_types=scratch_types,
      grid=(grid_m,),
      grid_names=("block",),
      compiler_params=plgpu.CompilerParams(approx_math=True),
  )(
      dout_flat,
      t3_flat,
      x_flat,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )
  grad_sum = jnp.sum(grad_part, axis=0)
  return (
      dt3_flat,
      dleft_act_epi,
      grad_sum[:h, :d].astype(dtype),
      grad_sum[h : h + c, :d].astype(dtype),
      jnp.sum(grad_sum[h + c : h + c + 32, :h], axis=0).astype(dtype),
      jnp.sum(grad_sum[h + c + 32 : h + c + 64, :h], axis=0).astype(dtype),
  )


def prologue_bwd(
    dleft_flat: jax.Array,
    dright_flat: jax.Array,
    dleft_act_epi: jax.Array,
    x_flat: jax.Array,
    mask_flat: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    *,
    eps: float,
    c_real: int,
    h_real: int,
    block_m: int,
    m_real: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    persistent: bool = True,
    ctas_per_sm: int = 2,
    **unused_kwargs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """SM100 zero-spill native M=64/128 prologue backward kernel."""
  del unused_kwargs, h_real
  m, c = x_flat.shape
  m_valid = m if m_real is None else m_real
  h = projection_in_weights.shape[2]
  dtype = x_flat.dtype
  num_sms = backend.get_default_device().core_count

  w_proj_2d = projection_in_weights.reshape(c, 2 * h)
  w_gate_2d = gate_in_weights.reshape(c, 2 * h)

  if max(c, h) > 64:
    tile_m = 64
    num_tiles = m // tile_m
    grid_m = _resolve_grid_m(
        num_tiles, num_sms, ctas_per_sm, num_warps, persistent, 1
    )
    vec_layout = plgpu.Layout.WGMMA
    dw_layout = plgpu.Layout.TCGEN05 if c == 128 else plgpu.Layout.WGMMA
    distinct_ch = c != h

    def kernel_large_ch(
        dleft_gmem,
        dright_gmem,
        dact_in_gmem,
        x_gmem,
        mask_gmem,
        w_proj_gmem,
        w_gate_gmem,
        ln_scale_gmem,
        ln_offset_gmem,
        dx_gmem,
        grad_part_gmem,
        buf_m_c,
        buf_h_m,
        w_p_smem,
        w_g_smem,
        act_smem,
        dp_dg_smem,
        dw_pl_tmem,
        dw_pr_tmem,
        dw_gl_tmem,
        dw_gr_top_tmem,
        work_h_tmem,
        work_c_opt_tmem,
        bar_l,
        bar_r,
        mma_barrier,
    ):
      pid = lax.axis_index("block")
      num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
          pid < lax.rem(num_tiles, grid_m), jnp.int32
      )
      work_c_tmem = work_c_opt_tmem if distinct_ch else work_h_tmem

      ln_scale = plgpu.load(
          ln_scale_gmem, layout=vec_layout.reduce(0), optimized=False
      )
      ln_offset = plgpu.load(
          ln_offset_gmem, layout=vec_layout.reduce(0), optimized=False
      )

      init_dw_gr_bot = plgpu.layout_cast(
          jnp.zeros((64, h), jnp.float32), plgpu.Layout.WGMMA
      )
      init_acc_s, init_acc_o = common.init_ln_grad_acc(c, vec_layout)

      def loop_body(step, carry):
        dw_gr_bot, acc_s, acc_o = carry
        tile_id = pid + step * grid_m
        row_start = tile_id * tile_m
        slice_m = pl.ds(row_start, tile_m)
        acc_flag = step > 0

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.copy_gmem_to_smem(x_gmem.at[slice_m, :], buf_m_c, bar_l)
        plgpu.copy_gmem_to_smem(dleft_gmem.at[:, slice_m], buf_h_m, bar_l)
        plgpu.copy_gmem_to_smem(w_proj_gmem.at[:, :h], w_p_smem, bar_l)
        plgpu.copy_gmem_to_smem(w_gate_gmem.at[:, :h], w_g_smem, bar_l)
        plgpu.barrier_wait(bar_l)

        x = plgpu.load(buf_m_c, layout=vec_layout)
        x = common.mask_rows_2d(x, row_start, m_valid, m, vec_layout)
        mask_val = (
            plgpu.load(
                mask_gmem.at[slice_m],
                layout=vec_layout.reduce(1),
                optimized=False,
            )
            != 0
        )
        mask_val = common.mask_rows_1d(
            mask_val, row_start, m_valid, m, vec_layout.reduce(1)
        )
        mask_c = plgpu.layout_cast(
            lax.broadcast_in_dim(mask_val, (tile_m, c), (0,)), vec_layout
        )
        mask_h = plgpu.layout_cast(
            lax.broadcast_in_dim(mask_val, (tile_m, h), (0,)), vec_layout
        )

        x_hat, rstd_in = common.layer_norm_hat_rstd(x, eps, c_real, vec_layout)
        act = common.apply_norm_affine(x_hat, ln_scale, ln_offset, vec_layout)
        act = jnp.where(mask_c, act, 0.0).astype(dtype)
        act_smem[...] = act
        plgpu.commit_smem()

        plgpu.tcgen05_mma(
            work_h_tmem,
            act_smem,
            w_g_smem,
            mma_barrier.at[0],
            accumulate=False,
        )
        plgpu.barrier_wait(mma_barrier.at[0])
        g_l_raw = plgpu.async_load_tmem(work_h_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        plgpu.tcgen05_mma(
            work_h_tmem,
            act_smem,
            w_p_smem,
            mma_barrier.at[1],
            accumulate=False,
        )
        s_l = jnp.where(mask_h, common.sigmoid_f32(g_l_raw), 0.0)

        dleft_t = plgpu.load(buf_h_m.T, layout=vec_layout).astype(jnp.float32)
        dleft_t = common.mask_rows_2d(
            dleft_t, row_start, m_valid, m, vec_layout
        )
        dp_l_f32 = dleft_t * s_l
        dp_l = dp_l_f32.astype(dtype)
        q_l = dp_l_f32 * (1.0 - s_l)

        plgpu.barrier_wait(mma_barrier.at[1])
        p_l = plgpu.async_load_tmem(work_h_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        dg_l = (q_l * p_l).astype(dtype)

        dp_dg_smem[...] = dp_l
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            dw_pl_tmem, act_smem.T, dp_dg_smem, accumulate=acc_flag
        )
        plgpu.tcgen05_mma(work_c_tmem, dp_dg_smem, w_p_smem.T, accumulate=False)
        plgpu.tcgen05_commit_arrive(mma_barrier.at[2])
        plgpu.barrier_wait(mma_barrier.at[2])
        dact = plgpu.async_load_tmem(work_c_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()

        dp_dg_smem[...] = dg_l
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            dw_gl_tmem, act_smem.T, dp_dg_smem, accumulate=acc_flag
        )
        plgpu.tcgen05_mma(work_c_tmem, dp_dg_smem, w_g_smem.T, accumulate=False)
        plgpu.tcgen05_commit_arrive(mma_barrier.at[3])
        plgpu.barrier_wait(mma_barrier.at[3])
        dact = dact + plgpu.async_load_tmem(work_c_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()

        common.warpgroup_barrier_and_commit_smem()
        plgpu.copy_gmem_to_smem(dact_in_gmem.at[slice_m, :], buf_m_c, bar_r)
        plgpu.copy_gmem_to_smem(dright_gmem.at[:, slice_m], buf_h_m, bar_r)
        plgpu.copy_gmem_to_smem(w_proj_gmem.at[:, h:], w_p_smem, bar_r)
        plgpu.copy_gmem_to_smem(w_gate_gmem.at[:, h:], w_g_smem, bar_r)
        plgpu.barrier_wait(bar_r)

        dact_in = plgpu.load(buf_m_c, layout=vec_layout).astype(jnp.float32)
        dact_in = common.mask_rows_2d(
            dact_in, row_start, m_valid, m, vec_layout
        )
        dact = dact + dact_in

        plgpu.tcgen05_mma(
            work_h_tmem,
            act_smem,
            w_g_smem,
            mma_barrier.at[4],
            accumulate=False,
        )
        plgpu.barrier_wait(mma_barrier.at[4])
        g_r_raw = plgpu.async_load_tmem(work_h_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        plgpu.tcgen05_mma(
            work_h_tmem,
            act_smem,
            w_p_smem,
            mma_barrier.at[5],
            accumulate=False,
        )
        s_r = jnp.where(mask_h, common.sigmoid_f32(g_r_raw), 0.0)

        dright_t = plgpu.load(buf_h_m.T, layout=vec_layout).astype(jnp.float32)
        dright_t = common.mask_rows_2d(
            dright_t, row_start, m_valid, m, vec_layout
        )
        dp_r_f32 = dright_t * s_r
        dp_r = dp_r_f32.astype(dtype)
        q_r = dp_r_f32 * (1.0 - s_r)

        plgpu.barrier_wait(mma_barrier.at[5])
        p_r = plgpu.async_load_tmem(work_h_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()
        dg_r = (q_r * p_r).astype(dtype)

        dp_dg_smem[...] = dp_r
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            dw_pr_tmem, act_smem.T, dp_dg_smem, accumulate=acc_flag
        )
        plgpu.tcgen05_mma(work_c_tmem, dp_dg_smem, w_p_smem.T, accumulate=False)
        plgpu.tcgen05_commit_arrive(mma_barrier.at[6])
        plgpu.barrier_wait(mma_barrier.at[6])
        dact = dact + plgpu.async_load_tmem(work_c_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()

        dp_dg_smem[...] = dg_r
        plgpu.commit_smem()
        plgpu.tcgen05_mma(
            dw_gr_top_tmem,
            act_smem.at[:, :64].T,
            dp_dg_smem,
            accumulate=acc_flag,
        )
        plgpu.tcgen05_mma(work_c_tmem, dp_dg_smem, w_g_smem.T, accumulate=False)
        plgpu.tcgen05_commit_arrive(mma_barrier.at[7])
        plgpu.barrier_wait(mma_barrier.at[7])
        dact = dact + plgpu.async_load_tmem(work_c_tmem, layout=vec_layout)
        plgpu.wait_load_tmem()

        if c == 128:
          plgpu.tcgen05_mma(
              work_h_tmem,
              act_smem.at[:, 64:].T,
              dp_dg_smem,
              mma_barrier.at[0],
              accumulate=False,
          )
          plgpu.barrier_wait(mma_barrier.at[0])
          dw_gr_bot = dw_gr_bot + plgpu.async_load_tmem(
              work_h_tmem, layout=plgpu.Layout.WGMMA
          )
          plgpu.wait_load_tmem()

        dx, acc_s, acc_o = common.layer_norm_2d_bwd_accum(
            dact, x_hat, rstd_in, ln_scale, c_real, vec_layout, acc_s, acc_o
        )
        common.warpgroup_barrier()
        buf_m_c[...] = dx.astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(buf_m_c, dx_gmem.at[slice_m, :])
        return dw_gr_bot, acc_s, acc_o

      dw_gr_bot, acc_s, acc_o = lax.fori_loop(
          0,
          num_local_steps,
          loop_body,
          (init_dw_gr_bot, init_acc_s, init_acc_o),
      )
      if c == 128:
        mgpu_lib.store(
            grad_part_gmem.at[pid, pl.ds(c + 64, 64), h:],
            dw_gr_bot,
            optimized=False,
        )
      dw_gr_top = plgpu.async_load_tmem(
          dw_gr_top_tmem, layout=plgpu.Layout.WGMMA
      )
      plgpu.wait_load_tmem()
      mgpu_lib.store(
          grad_part_gmem.at[pid, pl.ds(c, 64), h:], dw_gr_top, optimized=False
      )
      dw_pl = plgpu.async_load_tmem(dw_pl_tmem, layout=dw_layout)
      plgpu.wait_load_tmem()
      mgpu_lib.store(grad_part_gmem.at[pid, :c, :h], dw_pl, optimized=False)
      dw_pr = plgpu.async_load_tmem(dw_pr_tmem, layout=dw_layout)
      plgpu.wait_load_tmem()
      mgpu_lib.store(grad_part_gmem.at[pid, :c, h:], dw_pr, optimized=False)
      dw_gl = plgpu.async_load_tmem(dw_gl_tmem, layout=dw_layout)
      plgpu.wait_load_tmem()
      mgpu_lib.store(
          grad_part_gmem.at[pid, pl.ds(c, c), :h], dw_gl, optimized=False
      )
      mgpu_lib.store(
          grad_part_gmem.at[pid, pl.ds(2 * c, 32), :c], acc_s, optimized=False
      )
      mgpu_lib.store(
          grad_part_gmem.at[pid, pl.ds(2 * c + 32, 32), :c],
          acc_o,
          optimized=False,
      )
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    out_type = (
        jax.ShapeDtypeStruct((m, c), dtype),
        jax.ShapeDtypeStruct((grid_m, 2 * c + 64, 2 * h), jnp.float32),
    )
    work_c_scratch = (
        plgpu.TMEM((tile_m, c), jnp.float32)
        if distinct_ch
        else plgpu.Barrier(num_arrivals=1)
    )
    scratch_types = (
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
        _smem_for_mma((c, h), dtype, 64),
        _smem_for_mma((c, h), dtype, 64),
        _smem_for_mma((tile_m, c), dtype, 64),
        _smem_for_mma((tile_m, h), dtype, 64),
        plgpu.TMEM((c, h), jnp.float32),
        plgpu.TMEM((c, h), jnp.float32),
        plgpu.TMEM((c, h), jnp.float32),
        plgpu.TMEM((64, h), jnp.float32),
        plgpu.TMEM((tile_m, h), jnp.float32),
        work_c_scratch,
        plgpu.Barrier(num_arrivals=4),
        plgpu.Barrier(num_arrivals=4),
        plgpu.Barrier(num_arrivals=1, num_barriers=8, orders_tensor_core=True),
    )
    dx_flat, grad_part = plgpu.kernel(
        kernel_large_ch,
        out_type=out_type,
        scratch_types=scratch_types,
        grid=(grid_m,),
        grid_names=("block",),
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(
        dleft_flat,
        dright_flat,
        dleft_act_epi,
        x_flat,
        mask_flat,
        w_proj_2d,
        w_gate_2d,
        layernorm_in_scale,
        layernorm_in_offset,
    )
    grad_sum = jnp.sum(grad_part, axis=0)
    return (
        dx_flat,
        grad_sum[:c].astype(dtype).reshape(c, 2, h),
        grad_sum[c : 2 * c].astype(dtype).reshape(c, 2, h),
        jnp.sum(grad_sum[2 * c : 2 * c + 32, :c], axis=0).astype(dtype),
        jnp.sum(grad_sum[2 * c + 32 : 2 * c + 64, :c], axis=0).astype(dtype),
    )

  # Native M=64/128 zero-spill width-2H kernel:
  # - At tile_m=64: 104 KB SMEM + 256 TMEM cols + ~200 regs -> FULL 2 CTAs/SM!
  # - Overlaps dual `work_g_tmem` + `work_p_tmem` MMAs AND TMA-pipelines `mask`!
  tile_m = max(64, min(128, block_m))
  if m % tile_m != 0:
    tile_m = 64
  stages = 2
  del num_stages
  num_tiles = m // tile_m
  max_ctas = 2 if tile_m == 64 else 1
  grid_m = _resolve_grid_m(
      num_tiles, num_sms, ctas_per_sm, num_warps, persistent, max_ctas
  )
  vec_layout = plgpu.Layout.TCGEN05 if tile_m == 128 else plgpu.Layout.WGMMA
  dw_layout = plgpu.Layout.WGMMA

  def kernel(
      dleft_gmem,
      dright_gmem,
      dact_in_gmem,
      x_gmem,
      mask_gmem,
      w_proj_gmem,
      w_gate_gmem,
      ln_scale_gmem,
      ln_offset_gmem,
      dx_gmem,
      grad_part_gmem,
      x_smem,
      dleft_smem,
      dright_smem,
      w_proj_smem,
      w_gate_smem,
      act_smem,
      dp_dg_smem,
      dw_proj_tmem,
      dw_gate_tmem,
      work0_tmem,
      work1_tmem,
      work2_tmem,
      work3_tmem,
      in_barrier,
      dact_barrier,
      w_barrier,
      mma_barrier,
  ):
    pid = lax.axis_index("block")
    num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
        pid < lax.rem(num_tiles, grid_m), jnp.int32
    )

    plgpu.copy_gmem_to_smem(w_proj_gmem, w_proj_smem, w_barrier)
    plgpu.copy_gmem_to_smem(w_gate_gmem, w_gate_smem, w_barrier)

    for s in range(stages):
      init_tile = pid + s * grid_m
      init_slice = pl.ds(init_tile * tile_m, tile_m)
      pred = init_tile < num_tiles
      common.copy_gmem_to_smem(
          x_gmem.at[init_slice, :],
          x_smem.at[s],
          in_barrier.at[s],
          predicate=pred,
      )
      common.copy_gmem_to_smem(
          dleft_gmem.at[:, init_slice],
          dleft_smem.at[s],
          in_barrier.at[s],
          predicate=pred,
      )
      common.copy_gmem_to_smem(
          dright_gmem.at[:, init_slice],
          dright_smem.at[s],
          in_barrier.at[s],
          predicate=pred,
      )

    ln_scale = plgpu.load(
        ln_scale_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    ln_offset = plgpu.load(
        ln_offset_gmem, layout=vec_layout.reduce(0), optimized=False
    )
    plgpu.barrier_wait(w_barrier)

    init_acc_s, init_acc_o = common.init_ln_grad_acc(c, vec_layout)

    def loop_body(step, carry):
      acc_s, acc_o = carry
      slot = lax.rem(step, jnp.int32(stages))
      tile_id = pid + step * grid_m
      row_start = tile_id * tile_m
      next_tile = tile_id + stages * grid_m
      next_slice = pl.ds(next_tile * tile_m, tile_m)
      next_pred = next_tile < num_tiles
      slice_m = pl.ds(row_start, tile_m)

      mask_val = (
          plgpu.load(
              mask_gmem.at[slice_m],
              layout=vec_layout.reduce(1),
              optimized=False,
          )
          != 0
      )
      mask_val = common.mask_rows_1d(
          mask_val, row_start, m_valid, m, vec_layout.reduce(1)
      )
      plgpu.barrier_wait(in_barrier.at[slot])
      x = plgpu.load(x_smem.at[slot], layout=vec_layout)
      x = common.mask_rows_2d(x, row_start, m_valid, m, vec_layout)
      common.warpgroup_barrier_and_commit_smem()

      # Reuse x_smem.at[slot] to TMA-load dact_in while running branch MMAs.
      plgpu.copy_gmem_to_smem(
          dact_in_gmem.at[slice_m, :], x_smem.at[slot], dact_barrier
      )

      mask_c = plgpu.layout_cast(
          lax.broadcast_in_dim(mask_val, (tile_m, c), (0,)), vec_layout
      )
      mask_h = plgpu.layout_cast(
          lax.broadcast_in_dim(mask_val, (tile_m, h), (0,)), vec_layout
      )

      x_hat, rstd_in = common.layer_norm_hat_rstd(x, eps, c_real, vec_layout)
      act = common.apply_norm_affine(x_hat, ln_scale, ln_offset, vec_layout)
      act = jnp.where(mask_c, act, 0.0).astype(dtype)
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)
      act_smem[...] = act
      plgpu.commit_smem()

      # Issue all 4 forward recompute MMAs back-to-back into work0..work3!
      plgpu.tcgen05_mma(
          work0_tmem, act_smem, w_gate_smem.at[:, :h], accumulate=False
      )
      plgpu.tcgen05_mma(
          work1_tmem, act_smem, w_gate_smem.at[:, h:], accumulate=False
      )
      plgpu.tcgen05_commit_arrive(mma_barrier.at[0])
      plgpu.tcgen05_mma(
          work2_tmem, act_smem, w_proj_smem.at[:, :h], accumulate=False
      )
      plgpu.tcgen05_mma(
          work3_tmem, act_smem, w_proj_smem.at[:, h:], accumulate=False
      )
      plgpu.tcgen05_commit_arrive(mma_barrier.at[1])

      plgpu.barrier_wait(mma_barrier.at[0])
      g_l = plgpu.async_load_tmem(work0_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()
      s_l = jnp.where(mask_h, common.sigmoid_f32(g_l), 0.0)
      dleft_t = plgpu.load(dleft_smem.at[slot].T, layout=vec_layout).astype(
          jnp.float32
      )
      dleft_t = common.mask_rows_2d(dleft_t, row_start, m_valid, m, vec_layout)
      if c == h:
        common.warpgroup_barrier()
        dleft_smem.at[slot].T[...] = x_hat.astype(dtype)
      dp_l_f32 = dleft_t * s_l
      dp_dg_smem.at[:, :h][...] = dp_l_f32.astype(dtype)
      q_l = dp_l_f32 * (1.0 - s_l)

      g_r = plgpu.async_load_tmem(work1_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()
      s_r = jnp.where(mask_h, common.sigmoid_f32(g_r), 0.0)
      dright_t = plgpu.load(dright_smem.at[slot].T, layout=vec_layout).astype(
          jnp.float32
      )
      dright_t = common.mask_rows_2d(
          dright_t, row_start, m_valid, m, vec_layout
      )
      common.warpgroup_barrier_and_commit_smem()
      if c != h:
        common.copy_gmem_to_smem(
            dleft_gmem.at[:, next_slice],
            dleft_smem.at[slot],
            in_barrier.at[slot],
            predicate=next_pred,
        )
      common.copy_gmem_to_smem(
          dright_gmem.at[:, next_slice],
          dright_smem.at[slot],
          in_barrier.at[slot],
          predicate=next_pred,
      )
      dp_r_f32 = dright_t * s_r
      dp_dg_smem.at[:, h:][...] = dp_r_f32.astype(dtype)
      q_r = dp_r_f32 * (1.0 - s_r)

      plgpu.barrier_wait(mma_barrier.at[1])
      p_l = plgpu.async_load_tmem(work2_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()
      dg_l = (q_l * p_l).astype(dtype)
      p_r = plgpu.async_load_tmem(work3_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()
      dg_r = (q_r * p_r).astype(dtype)

      acc_flag = step > 0
      plgpu.commit_smem()
      plgpu.tcgen05_mma(
          dw_proj_tmem, act_smem.T, dp_dg_smem, accumulate=acc_flag
      )
      plgpu.tcgen05_mma(
          work0_tmem,
          dp_dg_smem,
          w_proj_smem.T,
          accumulate=False,
      )
      plgpu.tcgen05_commit_arrive(mma_barrier.at[2])
      plgpu.barrier_wait(mma_barrier.at[2])

      dp_dg_smem.at[:, :h][...] = dg_l
      dp_dg_smem.at[:, h:][...] = dg_r
      plgpu.commit_smem()
      plgpu.tcgen05_mma(
          work0_tmem,
          dp_dg_smem,
          w_gate_smem.T,
          accumulate=True,
      )
      plgpu.tcgen05_commit_arrive(mma_barrier.at[3])
      plgpu.tcgen05_mma(
          dw_gate_tmem,
          act_smem.T,
          dp_dg_smem,
          mma_barrier.at[4],
          accumulate=acc_flag,
      )

      plgpu.barrier_wait(mma_barrier.at[3])
      dact_mma = plgpu.async_load_tmem(work0_tmem, layout=vec_layout)
      plgpu.wait_load_tmem()

      plgpu.barrier_wait(dact_barrier)
      dact_in = plgpu.load(x_smem.at[slot], layout=vec_layout).astype(
          jnp.float32
      )
      dact_in = common.mask_rows_2d(dact_in, row_start, m_valid, m, vec_layout)
      x_hat_use = (
          plgpu.load(dleft_smem.at[slot].T, layout=vec_layout)
          if c == h
          else x_hat
      )
      common.warpgroup_barrier_and_commit_smem()
      common.copy_gmem_to_smem(
          x_gmem.at[next_slice, :],
          x_smem.at[slot],
          in_barrier.at[slot],
          predicate=next_pred,
      )
      if c == h:
        common.copy_gmem_to_smem(
            dleft_gmem.at[:, next_slice],
            dleft_smem.at[slot],
            in_barrier.at[slot],
            predicate=next_pred,
        )

      dact = dact_mma + dact_in
      dx, acc_s, acc_o = common.layer_norm_2d_bwd_accum(
          dact, x_hat_use, rstd_in, ln_scale, c_real, vec_layout, acc_s, acc_o
      )

      plgpu.barrier_wait(mma_barrier.at[4])
      act_smem[...] = dx.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(act_smem, dx_gmem.at[slice_m, :])
      return acc_s, acc_o

    acc_s, acc_o = lax.fori_loop(
        0, num_local_steps, loop_body, (init_acc_s, init_acc_o)
    )
    dw_proj = plgpu.async_load_tmem(dw_proj_tmem, layout=dw_layout)
    plgpu.wait_load_tmem()
    mgpu_lib.store(grad_part_gmem.at[pid, :c, :], dw_proj, optimized=False)
    dw_gate = plgpu.async_load_tmem(dw_gate_tmem, layout=dw_layout)
    plgpu.wait_load_tmem()
    mgpu_lib.store(
        grad_part_gmem.at[pid, pl.ds(c, c), :], dw_gate, optimized=False
    )
    mgpu_lib.store(
        grad_part_gmem.at[pid, pl.ds(2 * c, 32), :c], acc_s, optimized=False
    )
    mgpu_lib.store(
        grad_part_gmem.at[pid, pl.ds(2 * c + 32, 32), :c],
        acc_o,
        optimized=False,
    )
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  out_type = (
      jax.ShapeDtypeStruct((m, c), dtype),
      jax.ShapeDtypeStruct((grid_m, 2 * c + 64, 2 * h), jnp.float32),
  )
  scratch_types = (
      mgpu_lib.tiled_swizzled_smem((stages, tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((stages, h, tile_m), dtype),
      mgpu_lib.tiled_swizzled_smem((stages, h, tile_m), dtype),
      _smem_for_mma((c, 2 * h), dtype, 64),
      _smem_for_mma((c, 2 * h), dtype, 64),
      mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, 2 * h), dtype),
      plgpu.TMEM((c, 2 * h), jnp.float32),
      plgpu.TMEM((c, 2 * h), jnp.float32),
      plgpu.TMEM((tile_m, h), jnp.float32),
      plgpu.TMEM((tile_m, h), jnp.float32),
      plgpu.TMEM((tile_m, h), jnp.float32),
      plgpu.TMEM((tile_m, h), jnp.float32),
      plgpu.Barrier(num_arrivals=3, num_barriers=stages),
      plgpu.Barrier(num_arrivals=1),
      plgpu.Barrier(num_arrivals=2),
      plgpu.Barrier(num_arrivals=1, num_barriers=5, orders_tensor_core=True),
  )
  dx_flat, grad_part = plgpu.kernel(
      kernel,
      out_type=out_type,
      scratch_types=scratch_types,
      grid=(grid_m,),
      grid_names=("block",),
      compiler_params=plgpu.CompilerParams(approx_math=True),
  )(
      dleft_flat,
      dright_flat,
      dleft_act_epi,
      x_flat,
      mask_flat,
      w_proj_2d,
      w_gate_2d,
      layernorm_in_scale,
      layernorm_in_offset,
  )
  grad_sum = jnp.sum(grad_part, axis=0)
  return (
      dx_flat,
      grad_sum[:c].astype(dtype).reshape(c, 2, h),
      grad_sum[c : 2 * c].astype(dtype).reshape(c, 2, h),
      jnp.sum(grad_sum[2 * c : 2 * c + 32, :c], axis=0).astype(dtype),
      jnp.sum(grad_sum[2 * c + 32 : 2 * c + 64, :c], axis=0).astype(dtype),
  )
