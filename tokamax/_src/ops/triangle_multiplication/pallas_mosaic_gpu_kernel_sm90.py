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
"""Pallas-Mosaic-GPU Triangle Multiplication SM90 (Hopper H100) kernels."""

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
  """Returns default SM90 forward config."""
  n, c, h, d = _extract_dims(ba)
  if max(c, h, d) > 64:
    return common.Config(
        block_m=64,
        num_warps=4,
        num_stages=2,
        prologue_persistent=True,
        epilogue_persistent=True,
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
  """Returns SM90 forward autotuning configs."""
  _, c, h, d = _extract_dims(ba)
  configs = {get_heuristics_config(ba)}
  if max(c, h, d) > 64:
    for persistent in (True, False):
      configs.add(
          common.Config(
              block_m=64,
              num_warps=4,
              num_stages=2,
              prologue_persistent=persistent,
              epilogue_persistent=persistent,
              ctas_per_sm=1,
          )
      )
    return configs

  for block_m in (64, 128):
    for num_stages in (2, 3):
      for persistent in (True, False):
        configs.add(
            common.Config(
                block_m=block_m,
                num_warps=4,
                num_stages=num_stages,
                prologue_persistent=persistent,
                epilogue_persistent=persistent,
                ctas_per_sm=2,
            )
        )
  return configs


def get_vjp_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns default SM90 backward config."""
  _, c, h, d = _extract_dims(ba)
  if max(c, h, d) > 64:
    return common.Config(
        block_m=64,
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
  """Returns SM90 backward autotuning configs."""
  _, c, h, d = _extract_dims(ba)
  configs = {get_vjp_heuristics_config(ba)}
  if max(c, h, d) > 64:
    return configs
  for epilogue_block_m in (64, 128):
    for ctas_per_sm in (1, 2):
      configs.add(
          common.Config(
              block_m=64,
              epilogue_block_m=epilogue_block_m,
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


def _wgmma_matmul(a, b_smem, m: int, n: int) -> jax.Array:
  """Computes `a @ b_smem` in FP32 using a WGMMA accumulator."""

  def _scope(acc):
    plgpu.wgmma(acc, a, b_smem)
    return plgpu.wgmma_accumulator_load(acc, wait_n=0)

  return pl.run_scoped(_scope, plgpu.ACC((m, n), jnp.float32))


def _wgmma_glu_pair(
    a_gate,
    w_gate_smem,
    a_proj,
    w_proj_smem,
    m: int,
    n: int,
    dtype,
) -> jax.Array:
  """Computes `(a_proj @ w_proj) * sigmoid(a_gate @ w_gate)` via WGMMA."""

  def _scope(g_acc, p_acc):
    plgpu.wgmma(g_acc, a_gate, w_gate_smem)
    plgpu.wgmma(p_acc, a_proj, w_proj_smem)
    g = plgpu.wgmma_accumulator_load(g_acc, wait_n=1)
    s = common.sigmoid_fwd_f32(g, plgpu.Layout.WGMMA)
    p = plgpu.wgmma_accumulator_load(p_acc, wait_n=0)
    return (p * s).astype(dtype)

  return pl.run_scoped(
      _scope,
      plgpu.ACC((m, n), jnp.float32),
      plgpu.ACC((m, n), jnp.float32),
  )


def _wgmma_accum_and_dgrad(
    dw_acc_ref,
    act_t_smem,
    dout_smem,
    w_t_smem,
    m: int,
    k: int,
) -> jax.Array:
  """Accumulates `act.T @ dout` into `dw_acc_ref` and returns `dout @ w.T`."""

  def _scope(dact_acc):
    plgpu.wgmma(dw_acc_ref, act_t_smem, dout_smem)
    plgpu.wgmma(dact_acc, dout_smem, w_t_smem)
    return plgpu.wgmma_accumulator_load(dact_acc, wait_n=0)

  return pl.run_scoped(_scope, plgpu.ACC((m, k), jnp.float32))


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
  """SM90 TMA-pipelined prologue kernel with overlapped WGMMAs & stores."""
  del unused_kwargs, h_real
  m, c = x_flat.shape
  m_valid = m if m_real is None else m_real
  h = projection_in_weights.shape[2]
  dtype = x_flat.dtype
  wgmma_layout = plgpu.Layout.WGMMA
  num_sms = backend.get_default_device().core_count

  if max(c, h) > 64:
    tile_m = 64
    stages = 2
    max_ctas = 1
  else:
    tile_m = max(64, min(128, block_m))
    if m % tile_m != 0:
      tile_m = 64
    stages = max(2, min(4, num_stages))
    max_ctas = 3

  num_tiles = m // tile_m
  grid_m = _resolve_grid_m(
      num_tiles, num_sms, ctas_per_sm, num_warps, persistent, max_ctas
  )
  w_proj_2d = projection_in_weights.reshape(c, 2 * h)
  w_gate_2d = gate_in_weights.reshape(c, 2 * h)

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
      w_pl_smem,
      w_pr_smem,
      w_gl_smem,
      w_gr_smem,
      left_out_smem,
      right_out_smem,
      x_barrier,
      w_barrier,
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
        ln_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    ln_offset = plgpu.load(
        ln_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    plgpu.barrier_wait(w_barrier)

    def loop_body(step, _):
      slot = lax.rem(step, jnp.int32(stages))
      tile_id = pid + step * grid_m
      next_tile = tile_id + stages * grid_m
      next_slice = pl.ds(next_tile * tile_m, tile_m)
      next_pred = next_tile < num_tiles
      row_start = tile_id * tile_m
      slice_m = pl.ds(row_start, tile_m)

      mask_val = (
          plgpu.load(
              mask_gmem.at[slice_m],
              layout=wgmma_layout.reduce(1),
              optimized=False,
          )
          != 0
      )
      mask_val = common.mask_rows_1d(
          mask_val, row_start, m_valid, m, wgmma_layout.reduce(1)
      )
      plgpu.barrier_wait(x_barrier.at[slot])
      x = plgpu.load(x_smem.at[slot], layout=wgmma_layout)
      x = common.mask_rows_2d(x, row_start, m_valid, m, wgmma_layout)
      common.warpgroup_barrier_and_commit_smem()
      common.copy_gmem_to_smem(
          x_gmem.at[next_slice, :],
          x_smem.at[slot],
          x_barrier.at[slot],
          predicate=next_pred,
      )

      mask_b = plgpu.layout_cast(
          lax.broadcast_in_dim(mask_val, (tile_m, c), (0,)), wgmma_layout
      )
      act = common.layer_norm_2d(
          x, ln_scale, ln_offset, eps, c_real, wgmma_layout
      )
      act = jnp.where(mask_b, act, 0.0).astype(dtype)

      left = _wgmma_glu_pair(act, w_gl_smem, act, w_pl_smem, tile_m, h, dtype)
      plgpu.wait_smem_to_gmem(1, wait_read_only=True)
      left_out_smem.T[...] = left
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(left_out_smem, left_gmem.at[:, slice_m])

      right = _wgmma_glu_pair(act, w_gr_smem, act, w_pr_smem, tile_m, h, dtype)
      plgpu.wait_smem_to_gmem(1, wait_read_only=True)
      right_out_smem.T[...] = right
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(right_out_smem, right_gmem.at[:, slice_m])

    lax.fori_loop(0, num_local_steps, loop_body, None)
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  out_shape = jax.ShapeDtypeStruct((h, m), dtype)
  scratch_types = (
      mgpu_lib.tiled_swizzled_smem((stages, tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((c, h), dtype),
      mgpu_lib.tiled_swizzled_smem((c, h), dtype),
      mgpu_lib.tiled_swizzled_smem((c, h), dtype),
      mgpu_lib.tiled_swizzled_smem((c, h), dtype),
      mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
      mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
      plgpu.Barrier(num_arrivals=1, num_barriers=stages),
      plgpu.Barrier(num_arrivals=4),
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
  """SM90 zero-spill TMA-pipelined epilogue kernel (96 KB SMEM -> 2 CTAs/SM)."""
  del unused_kwargs
  m, c = x_flat.shape
  m_valid = m if m_real is None else m_real
  out_m = m if common.is_interpret_mode() else m_valid
  h, d = projection_out_weights.shape
  d_out = d if d_real is None else d_real
  dtype = x_flat.dtype
  num_sms = backend.get_default_device().core_count
  if max(c, h, d) > 64:
    tile_m = 64
    stages = 2
    max_ctas = 1
  else:
    tile_m = max(64, min(128, block_m))
    if m % tile_m != 0:
      tile_m = 64
    max_stages = 2 if tile_m == 128 else 3
    stages = max(2, min(max_stages, num_stages))
    max_ctas = 2

  num_tiles = m // tile_m
  grid_m = _resolve_grid_m(
      num_tiles, num_sms, ctas_per_sm, num_warps, persistent, max_ctas
  )
  wgmma_layout = plgpu.Layout.WGMMA

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
      out_smem,
      in_barrier,
      w_barrier,
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
        ln_in_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    ln_in_offset = plgpu.load(
        ln_in_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    ln_out_scale = plgpu.load(
        ln_out_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    ln_out_offset = plgpu.load(
        ln_out_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
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
      x = plgpu.load(x_smem.at[slot], layout=wgmma_layout)
      x = common.mask_rows_2d(x, row_start, m_valid, m, wgmma_layout)
      left_act = common.layer_norm_2d(
          x, ln_in_scale, ln_in_offset, eps, c_real, wgmma_layout
      ).astype(dtype)

      t3_t = plgpu.load(t3_smem.at[slot].T, layout=wgmma_layout)
      t3_t = common.mask_rows_2d(t3_t, row_start, m_valid, m, wgmma_layout)
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

      act_out = common.layer_norm_2d(
          t3_t, ln_out_scale, ln_out_offset, eps, h_real, wgmma_layout
      ).astype(dtype)

      out = _wgmma_glu_pair(
          left_act,
          w_gate_out_smem,
          act_out,
          w_proj_out_smem,
          tile_m,
          d,
          dtype,
      )

      slice_m = pl.ds(row_start, tile_m)
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)
      out_smem[...] = out[:, :d_out] if d_out < d else out
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[slice_m, :])

    lax.fori_loop(0, num_local_steps, loop_body, None)
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  scratch_types = (
      mgpu_lib.tiled_swizzled_smem((stages, tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((stages, h, tile_m), dtype),
      mgpu_lib.tiled_swizzled_smem((h, d), dtype),
      mgpu_lib.tiled_swizzled_smem((c, d), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, d_out), dtype),
      plgpu.Barrier(num_arrivals=2, num_barriers=stages),
      plgpu.Barrier(num_arrivals=2),
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
  """SM90 zero-spill TMA-pipelined epilogue backward kernel."""
  del unused_kwargs, num_stages
  m, c = x_flat.shape
  m_valid = m if m_real is None else m_real
  h, d = projection_out_weights.shape
  dtype = x_flat.dtype
  wgmma_layout = plgpu.Layout.WGMMA
  num_sms = backend.get_default_device().core_count

  if max(c, h, d) > 64:
    tile_m = 64
    stages = 1
    max_ctas = 1
  else:
    tile_m = max(64, min(128, block_m))
    if m % tile_m != 0:
      tile_m = 64
    stages = 2
    max_ctas = 2 if tile_m == 64 else 1

  num_tiles = m // tile_m
  grid_m = _resolve_grid_m(
      num_tiles, num_sms, ctas_per_sm, num_warps, persistent, max_ctas
  )
  max_hd = max(h, d)

  if stages == 1:

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
        dp_dg_smem,
        in_barrier,
        w_barrier,
    ):
      pid = lax.axis_index("block")
      num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
          pid < lax.rem(num_tiles, grid_m), jnp.int32
      )

      plgpu.copy_gmem_to_smem(w_proj_out_gmem, w_proj_out_smem, w_barrier)
      plgpu.copy_gmem_to_smem(w_gate_out_gmem, w_gate_out_smem, w_barrier)

      ln_in_scale = plgpu.load(
          ln_in_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
      )
      ln_in_offset = plgpu.load(
          ln_in_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
      )
      ln_out_scale = plgpu.load(
          ln_out_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
      )
      ln_out_offset = plgpu.load(
          ln_out_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
      )
      plgpu.barrier_wait(w_barrier)

      def _run_with_accs(dw_proj_acc_ref, dw_gate_acc_ref):
        init_acc_s, init_acc_o = common.init_ln_grad_acc(h, wgmma_layout)

        def loop_body(step, carry):
          acc_s, acc_o = carry
          tile_id = pid + step * grid_m
          row_start = tile_id * tile_m
          slice_m = pl.ds(row_start, tile_m)

          plgpu.wait_smem_to_gmem(0, wait_read_only=True)
          plgpu.copy_gmem_to_smem(x_gmem.at[slice_m, :], x_smem, in_barrier)
          plgpu.copy_gmem_to_smem(t3_gmem.at[:, slice_m], t3_smem, in_barrier)
          plgpu.copy_gmem_to_smem(
              dout_gmem.at[slice_m, :], dout_smem, in_barrier
          )
          plgpu.barrier_wait(in_barrier)

          x = plgpu.load(x_smem, layout=wgmma_layout)
          x = common.mask_rows_2d(x, row_start, m_valid, m, wgmma_layout)
          left_act = common.layer_norm_2d(
              x, ln_in_scale, ln_in_offset, eps, c_real, wgmma_layout
          ).astype(dtype)
          common.warpgroup_barrier()
          left_act_smem[...] = left_act

          t3_t = plgpu.load(t3_smem.T, layout=wgmma_layout)
          t3_t = common.mask_rows_2d(t3_t, row_start, m_valid, m, wgmma_layout)
          common.warpgroup_barrier()
          t_hat, rstd_out = common.layer_norm_hat_rstd(
              t3_t, eps, h_real, wgmma_layout
          )
          act = common.apply_norm_affine(
              t_hat, ln_out_scale, ln_out_offset, wgmma_layout
          ).astype(dtype)
          t3_smem.T[...] = t_hat.astype(dtype)
          act_smem[...] = act
          plgpu.commit_smem()

          g_raw = _wgmma_matmul(left_act_smem, w_gate_out_smem, tile_m, d)
          gate = common.sigmoid_f32(g_raw)
          dout = plgpu.load(dout_smem, layout=wgmma_layout).astype(jnp.float32)
          dout = common.mask_rows_2d(dout, row_start, m_valid, m, wgmma_layout)
          common.warpgroup_barrier()
          dp_f32 = dout * gate
          dp_dg_smem[...] = dp_f32.astype(dtype)
          q = (dp_f32 * (1.0 - gate)).astype(dtype)
          p = _wgmma_matmul(act_smem, w_proj_out_smem, tile_m, d)
          dg = (q.astype(jnp.float32) * p).astype(dtype)
          plgpu.commit_smem()

          dact = _wgmma_accum_and_dgrad(
              dw_proj_acc_ref,
              act_smem.T,
              dp_dg_smem,
              w_proj_out_smem.T,
              tile_m,
              h,
          )
          t_hat_reload = plgpu.load(t3_smem.T, layout=wgmma_layout)
          dt3_t, acc_s, acc_o = common.layer_norm_2d_bwd_accum(
              dact,
              t_hat_reload,
              rstd_out,
              ln_out_scale,
              h_real,
              wgmma_layout,
              acc_s,
              acc_o,
          )
          common.warpgroup_barrier()
          t3_smem.T[...] = dt3_t.astype(dtype)
          dp_dg_smem[...] = dg
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(
              t3_smem, dt3_gmem.at[:, slice_m], commit_group=False
          )

          dleft_act = _wgmma_accum_and_dgrad(
              dw_gate_acc_ref,
              left_act_smem.T,
              dp_dg_smem,
              w_gate_out_smem.T,
              tile_m,
              c,
          )
          x_smem[...] = dleft_act.astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(x_smem, dleft_act_epi_gmem.at[slice_m, :])
          return acc_s, acc_o

        acc_s, acc_o = lax.fori_loop(
            0, num_local_steps, loop_body, (init_acc_s, init_acc_o)
        )
        dw_proj = plgpu.wgmma_accumulator_load(dw_proj_acc_ref, wait_n=0)
        plgpu.store(
            grad_part_gmem.at[pid, pl.ds(0, h), :d], dw_proj, optimized=False
        )
        dw_gate = plgpu.wgmma_accumulator_load(dw_gate_acc_ref, wait_n=0)
        plgpu.store(
            grad_part_gmem.at[pid, pl.ds(h, c), :d], dw_gate, optimized=False
        )
        plgpu.store(
            grad_part_gmem.at[pid, pl.ds(h + c, 32), :h],
            acc_s,
            optimized=False,
        )
        plgpu.store(
            grad_part_gmem.at[pid, pl.ds(h + c + 32, 32), :h],
            acc_o,
            optimized=False,
        )

      pl.run_scoped(
          _run_with_accs,
          plgpu.ACC((h, d), jnp.float32),
          plgpu.ACC((c, d), jnp.float32),
      )
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    out_type = (
        jax.ShapeDtypeStruct((h, m), dtype),
        jax.ShapeDtypeStruct((m, c), dtype),
        jax.ShapeDtypeStruct((grid_m, h + c + 64, max_hd), jnp.float32),
    )
    scratch_types = (
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, d), dtype),
        mgpu_lib.tiled_swizzled_smem((h, d), dtype),
        mgpu_lib.tiled_swizzled_smem((c, d), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, h), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, d), dtype),
        plgpu.Barrier(num_arrivals=3),
        plgpu.Barrier(num_arrivals=2),
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

  # 104 KB SMEM + zero-spill register budget at tile_m=64 -> full 2 CTAs/SM on
  # H100 by staging `t_hat` in SMEM and running sequential branch WGMMAs.
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
      t_hat_smem,
      left_act_smem,
      dp_dg_smem,
      dt3_smem,
      in_barrier,
      w_barrier,
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
        ln_in_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    ln_in_offset = plgpu.load(
        ln_in_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    ln_out_scale = plgpu.load(
        ln_out_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    ln_out_offset = plgpu.load(
        ln_out_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    plgpu.barrier_wait(w_barrier)

    def _run_with_accs(dw_proj_acc_ref, dw_gate_acc_ref):
      init_acc_s, init_acc_o = common.init_ln_grad_acc(h, wgmma_layout)

      def loop_body(step, carry):
        acc_s, acc_o = carry
        slot = lax.rem(step, jnp.int32(stages))
        tile_id = pid + step * grid_m
        row_start = tile_id * tile_m
        next_tile = tile_id + stages * grid_m
        next_slice = pl.ds(next_tile * tile_m, tile_m)
        next_pred = next_tile < num_tiles
        slice_m = pl.ds(row_start, tile_m)

        plgpu.barrier_wait(in_barrier.at[slot])
        x = plgpu.load(x_smem.at[slot], layout=wgmma_layout)
        x = common.mask_rows_2d(x, row_start, m_valid, m, wgmma_layout)
        left_act = common.layer_norm_2d(
            x, ln_in_scale, ln_in_offset, eps, c_real, wgmma_layout
        ).astype(dtype)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        left_act_smem[...] = left_act
        plgpu.commit_smem()

        g_raw = _wgmma_matmul(left_act_smem, w_gate_out_smem, tile_m, d)
        gate = common.sigmoid_f32(g_raw)
        dout = plgpu.load(dout_smem.at[slot], layout=wgmma_layout).astype(
            jnp.float32
        )
        dout = common.mask_rows_2d(dout, row_start, m_valid, m, wgmma_layout)
        dp_f32 = dout * gate
        dp_dg_smem[...] = dp_f32.astype(dtype)
        q = (dp_f32 * (1.0 - gate)).astype(dtype)

        t3_t = plgpu.load(t3_smem.at[slot].T, layout=wgmma_layout)
        t3_t = common.mask_rows_2d(t3_t, row_start, m_valid, m, wgmma_layout)
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
            t3_t, eps, h_real, wgmma_layout
        )
        act = common.apply_norm_affine(
            t_hat, ln_out_scale, ln_out_offset, wgmma_layout
        ).astype(dtype)
        t_hat_smem[...] = t_hat.astype(dtype)
        act_smem[...] = act
        plgpu.commit_smem()

        p = _wgmma_matmul(act_smem, w_proj_out_smem, tile_m, d)
        dg = (q.astype(jnp.float32) * p).astype(dtype)

        dact = _wgmma_accum_and_dgrad(
            dw_proj_acc_ref,
            act_smem.T,
            dp_dg_smem,
            w_proj_out_smem.T,
            tile_m,
            h,
        )
        t_hat_reload = plgpu.load(t_hat_smem, layout=wgmma_layout)
        dt3_t, acc_s, acc_o = common.layer_norm_2d_bwd_accum(
            dact,
            t_hat_reload,
            rstd_out,
            ln_out_scale,
            h_real,
            wgmma_layout,
            acc_s,
            acc_o,
        )

        common.warpgroup_barrier()
        dp_dg_smem[...] = dg
        plgpu.commit_smem()

        dleft_act = _wgmma_accum_and_dgrad(
            dw_gate_acc_ref,
            left_act_smem.T,
            dp_dg_smem,
            w_gate_out_smem.T,
            tile_m,
            c,
        )

        common.warpgroup_barrier()
        dt3_smem.T[...] = dt3_t.astype(dtype)
        left_act_smem[...] = dleft_act.astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(
            dt3_smem, dt3_gmem.at[:, slice_m], commit_group=False
        )
        plgpu.copy_smem_to_gmem(
            left_act_smem, dleft_act_epi_gmem.at[slice_m, :]
        )
        return acc_s, acc_o

      acc_s, acc_o = lax.fori_loop(
          0, num_local_steps, loop_body, (init_acc_s, init_acc_o)
      )
      dw_proj = plgpu.wgmma_accumulator_load(dw_proj_acc_ref, wait_n=0)
      plgpu.store(
          grad_part_gmem.at[pid, pl.ds(0, h), :d], dw_proj, optimized=False
      )
      dw_gate = plgpu.wgmma_accumulator_load(dw_gate_acc_ref, wait_n=0)
      plgpu.store(
          grad_part_gmem.at[pid, pl.ds(h, c), :d], dw_gate, optimized=False
      )
      plgpu.store(
          grad_part_gmem.at[pid, pl.ds(h + c, 32), :h],
          acc_s,
          optimized=False,
      )
      plgpu.store(
          grad_part_gmem.at[pid, pl.ds(h + c + 32, 32), :h],
          acc_o,
          optimized=False,
      )

    pl.run_scoped(
        _run_with_accs,
        plgpu.ACC((h, d), jnp.float32),
        plgpu.ACC((c, d), jnp.float32),
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
      mgpu_lib.tiled_swizzled_smem((h, d), dtype),
      mgpu_lib.tiled_swizzled_smem((c, d), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, h), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, h), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, d), dtype),
      mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
      plgpu.Barrier(num_arrivals=3, num_barriers=stages),
      plgpu.Barrier(num_arrivals=2),
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
  """SM90 zero-spill fused width-2H prologue backward kernel."""
  del unused_kwargs, h_real, block_m, num_stages
  m, c = x_flat.shape
  m_valid = m if m_real is None else m_real
  h = projection_in_weights.shape[2]
  dtype = x_flat.dtype
  wgmma_layout = plgpu.Layout.WGMMA
  num_sms = backend.get_default_device().core_count

  w_proj_2d = projection_in_weights.reshape(c, 2 * h)
  w_gate_2d = gate_in_weights.reshape(c, 2 * h)

  tile_m = 64
  num_tiles = m // tile_m

  if max(c, h) > 64:
    grid_m = _resolve_grid_m(
        num_tiles, num_sms, ctas_per_sm, num_warps, persistent, 1
    )

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
        x_hat_smem,
        dp_dg_smem,
        bar_l,
        bar_r,
    ):
      pid = lax.axis_index("block")
      num_local_steps = lax.div(num_tiles, grid_m) + lax.convert_element_type(
          pid < lax.rem(num_tiles, grid_m), jnp.int32
      )

      ln_scale = plgpu.load(
          ln_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
      )
      ln_offset = plgpu.load(
          ln_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
      )

      def _run_with_accs(
          dw_pl_acc_ref, dw_pr_acc_ref, dw_gl_acc_ref, dw_gr_acc_ref
      ):
        init_acc_s, init_acc_o = common.init_ln_grad_acc(c, wgmma_layout)

        def loop_body(step, carry):
          acc_s, acc_o = carry
          tile_id = pid + step * grid_m
          row_start = tile_id * tile_m
          slice_m = pl.ds(row_start, tile_m)

          plgpu.wait_smem_to_gmem(0, wait_read_only=True)
          plgpu.copy_gmem_to_smem(x_gmem.at[slice_m, :], buf_m_c, bar_l)
          plgpu.copy_gmem_to_smem(dleft_gmem.at[:, slice_m], buf_h_m, bar_l)
          plgpu.copy_gmem_to_smem(w_proj_gmem.at[:, :h], w_p_smem, bar_l)
          plgpu.copy_gmem_to_smem(w_gate_gmem.at[:, :h], w_g_smem, bar_l)
          plgpu.barrier_wait(bar_l)

          x = plgpu.load(buf_m_c, layout=wgmma_layout)
          x = common.mask_rows_2d(x, row_start, m_valid, m, wgmma_layout)
          mask_val = (
              plgpu.load(
                  mask_gmem.at[slice_m],
                  layout=wgmma_layout.reduce(1),
                  optimized=False,
              )
              != 0
          )
          mask_val = common.mask_rows_1d(
              mask_val, row_start, m_valid, m, wgmma_layout.reduce(1)
          )
          x_hat, rstd_in = common.layer_norm_hat_rstd(
              x, eps, c_real, wgmma_layout
          )
          act = common.apply_norm_affine(
              x_hat, ln_scale, ln_offset, wgmma_layout
          )
          mask_c = plgpu.layout_cast(
              lax.broadcast_in_dim(mask_val, (tile_m, c), (0,)), wgmma_layout
          )
          act = jnp.where(mask_c, act, 0.0).astype(dtype)
          x_hat_smem[...] = x_hat.astype(dtype)
          act_smem[...] = act
          plgpu.commit_smem()

          def _run_all_branch_wgmmas(dact_acc):
            mask_h = plgpu.layout_cast(
                lax.broadcast_in_dim(mask_val, (tile_m, h), (0,)), wgmma_layout
            )
            g_l = _wgmma_matmul(act_smem, w_g_smem, tile_m, h)
            s_l = jnp.where(mask_h, common.sigmoid_f32(g_l), 0.0)
            dleft_t = common.mask_rows_2d(
                plgpu.load(buf_h_m.T, layout=wgmma_layout),
                row_start,
                m_valid,
                m,
                wgmma_layout,
            )
            common.warpgroup_barrier()
            dp_l_f32 = dleft_t.astype(jnp.float32) * s_l
            dp_dg_smem[...] = dp_l_f32.astype(dtype)
            q_l = (dp_l_f32 * (1.0 - s_l)).astype(dtype)
            p_l = _wgmma_matmul(act_smem, w_p_smem, tile_m, h)
            dg_l = (q_l.astype(jnp.float32) * p_l).astype(dtype)
            plgpu.commit_smem()

            plgpu.wgmma(dw_pl_acc_ref, act_smem.T, dp_dg_smem)
            plgpu.wgmma(dact_acc, dp_dg_smem, w_p_smem.T)
            if common.is_interpret_mode():
              _ = plgpu.wgmma_accumulator_load(dact_acc, wait_n=0)
            else:
              plgpu.wgmma_wait(0)
            common.warpgroup_barrier()
            dp_dg_smem[...] = dg_l
            plgpu.commit_smem()
            plgpu.wgmma(dw_gl_acc_ref, act_smem.T, dp_dg_smem)
            plgpu.wgmma(dact_acc, dp_dg_smem, w_g_smem.T)
            if common.is_interpret_mode():
              _ = plgpu.wgmma_accumulator_load(dact_acc, wait_n=0)
            else:
              plgpu.wgmma_wait(0)

            common.warpgroup_barrier_and_commit_smem()
            plgpu.copy_gmem_to_smem(dact_in_gmem.at[slice_m, :], buf_m_c, bar_r)
            plgpu.copy_gmem_to_smem(dright_gmem.at[:, slice_m], buf_h_m, bar_r)
            plgpu.copy_gmem_to_smem(w_proj_gmem.at[:, h:], w_p_smem, bar_r)
            plgpu.copy_gmem_to_smem(w_gate_gmem.at[:, h:], w_g_smem, bar_r)
            plgpu.barrier_wait(bar_r)

            g_r = _wgmma_matmul(act_smem, w_g_smem, tile_m, h)
            s_r = jnp.where(mask_h, common.sigmoid_f32(g_r), 0.0)
            dright_t = common.mask_rows_2d(
                plgpu.load(buf_h_m.T, layout=wgmma_layout),
                row_start,
                m_valid,
                m,
                wgmma_layout,
            )
            common.warpgroup_barrier()
            dp_r_f32 = dright_t.astype(jnp.float32) * s_r
            dp_dg_smem[...] = dp_r_f32.astype(dtype)
            q_r = (dp_r_f32 * (1.0 - s_r)).astype(dtype)
            p_r = _wgmma_matmul(act_smem, w_p_smem, tile_m, h)
            dg_r = (q_r.astype(jnp.float32) * p_r).astype(dtype)
            plgpu.commit_smem()

            plgpu.wgmma(dw_pr_acc_ref, act_smem.T, dp_dg_smem)
            plgpu.wgmma(dact_acc, dp_dg_smem, w_p_smem.T)
            if common.is_interpret_mode():
              _ = plgpu.wgmma_accumulator_load(dact_acc, wait_n=0)
            else:
              plgpu.wgmma_wait(0)
            common.warpgroup_barrier()
            dp_dg_smem[...] = dg_r
            plgpu.commit_smem()
            plgpu.wgmma(dw_gr_acc_ref, act_smem.T, dp_dg_smem)
            plgpu.wgmma(dact_acc, dp_dg_smem, w_g_smem.T)
            return plgpu.wgmma_accumulator_load(dact_acc, wait_n=0)

          dact = pl.run_scoped(
              _run_all_branch_wgmmas, plgpu.ACC((tile_m, c), jnp.float32)
          )
          dact_in = plgpu.load(buf_m_c, layout=wgmma_layout).astype(jnp.float32)
          dact_in = common.mask_rows_2d(
              dact_in, row_start, m_valid, m, wgmma_layout
          )
          dact = dact + dact_in
          x_hat_reload = plgpu.load(x_hat_smem, layout=wgmma_layout)
          dx, acc_s, acc_o = common.layer_norm_2d_bwd_accum(
              dact,
              x_hat_reload,
              rstd_in,
              ln_scale,
              c_real,
              wgmma_layout,
              acc_s,
              acc_o,
          )
          common.warpgroup_barrier()
          buf_m_c[...] = dx.astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(buf_m_c, dx_gmem.at[slice_m, :])
          return acc_s, acc_o

        acc_s, acc_o = lax.fori_loop(
            0, num_local_steps, loop_body, (init_acc_s, init_acc_o)
        )
        dw_pl = plgpu.wgmma_accumulator_load(dw_pl_acc_ref, wait_n=0)
        plgpu.store(grad_part_gmem.at[pid, :c, :h], dw_pl, optimized=False)
        dw_pr = plgpu.wgmma_accumulator_load(dw_pr_acc_ref, wait_n=0)
        plgpu.store(grad_part_gmem.at[pid, :c, h:], dw_pr, optimized=False)
        dw_gl = plgpu.wgmma_accumulator_load(dw_gl_acc_ref, wait_n=0)
        plgpu.store(
            grad_part_gmem.at[pid, pl.ds(c, c), :h], dw_gl, optimized=False
        )
        dw_gr = plgpu.wgmma_accumulator_load(dw_gr_acc_ref, wait_n=0)
        plgpu.store(
            grad_part_gmem.at[pid, pl.ds(c, c), h:], dw_gr, optimized=False
        )
        plgpu.store(
            grad_part_gmem.at[pid, pl.ds(2 * c, 32), :c],
            acc_s,
            optimized=False,
        )
        plgpu.store(
            grad_part_gmem.at[pid, pl.ds(2 * c + 32, 32), :c],
            acc_o,
            optimized=False,
        )

      pl.run_scoped(
          _run_with_accs,
          plgpu.ACC((c, h), jnp.float32),
          plgpu.ACC((c, h), jnp.float32),
          plgpu.ACC((c, h), jnp.float32),
          plgpu.ACC((c, h), jnp.float32),
      )
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    out_type = (
        jax.ShapeDtypeStruct((m, c), dtype),
        jax.ShapeDtypeStruct((grid_m, 2 * c + 64, 2 * h), jnp.float32),
    )
    scratch_types = (
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
        mgpu_lib.tiled_swizzled_smem((c, h), dtype),
        mgpu_lib.tiled_swizzled_smem((c, h), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
        mgpu_lib.tiled_swizzled_smem((tile_m, h), dtype),
        plgpu.Barrier(num_arrivals=4),
        plgpu.Barrier(num_arrivals=4),
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

  # Fused width-2H (64x128) zero-spill double-buffered path (96 KB SMEM -> FULL
  # 2 CTAs/SM on H100) staging `x_hat` in `x_hat_smem` across sequential WGMMAs.
  stages = 2
  grid_m = _resolve_grid_m(
      num_tiles, num_sms, ctas_per_sm, num_warps, persistent, 2
  )

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
      x_hat_smem,
      dp_dg_smem,
      x_barrier,
      lr_barrier,
      dact_barrier,
      w_barrier,
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
          x_barrier.at[s],
          predicate=pred,
      )

    ln_scale = plgpu.load(
        ln_scale_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    ln_offset = plgpu.load(
        ln_offset_gmem, layout=wgmma_layout.reduce(0), optimized=False
    )
    plgpu.barrier_wait(w_barrier)

    def _run_with_accs(dw_proj_acc_ref, dw_gate_acc_ref):
      init_acc_s, init_acc_o = common.init_ln_grad_acc(c, wgmma_layout)

      def loop_body(step, carry):
        acc_s, acc_o = carry
        slot = lax.rem(step, jnp.int32(stages))
        tile_id = pid + step * grid_m
        row_start = tile_id * tile_m
        next_tile = tile_id + stages * grid_m
        next_slice = pl.ds(next_tile * tile_m, tile_m)
        next_pred = next_tile < num_tiles
        slice_m = pl.ds(row_start, tile_m)

        plgpu.copy_gmem_to_smem(
            dleft_gmem.at[:, slice_m], dleft_smem, lr_barrier
        )
        plgpu.copy_gmem_to_smem(
            dright_gmem.at[:, slice_m], dright_smem, lr_barrier
        )

        mask_val = (
            plgpu.load(
                mask_gmem.at[slice_m],
                layout=wgmma_layout.reduce(1),
                optimized=False,
            )
            != 0
        )
        mask_val = common.mask_rows_1d(
            mask_val, row_start, m_valid, m, wgmma_layout.reduce(1)
        )
        plgpu.barrier_wait(x_barrier.at[slot])
        x = plgpu.load(x_smem.at[slot], layout=wgmma_layout)
        x = common.mask_rows_2d(x, row_start, m_valid, m, wgmma_layout)
        common.warpgroup_barrier_and_commit_smem()

        plgpu.copy_gmem_to_smem(
            dact_in_gmem.at[slice_m, :], x_smem.at[slot], dact_barrier
        )

        x_hat, rstd_in = common.layer_norm_hat_rstd(
            x, eps, c_real, wgmma_layout
        )
        act = common.apply_norm_affine(x_hat, ln_scale, ln_offset, wgmma_layout)
        mask_c = plgpu.layout_cast(
            lax.broadcast_in_dim(mask_val, (tile_m, c), (0,)), wgmma_layout
        )
        act = jnp.where(mask_c, act, 0.0).astype(dtype)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        x_hat_smem[...] = x_hat.astype(dtype)
        act_smem[...] = act
        plgpu.commit_smem()

        mask_h = plgpu.layout_cast(
            lax.broadcast_in_dim(mask_val, (tile_m, h), (0,)), wgmma_layout
        )
        g_l = _wgmma_matmul(act_smem, w_gate_smem.at[:, :h], tile_m, h)
        s_l = jnp.where(mask_h, common.sigmoid_f32(g_l), 0.0)
        plgpu.barrier_wait(lr_barrier)
        dleft_t = common.mask_rows_2d(
            plgpu.load(dleft_smem.T, layout=wgmma_layout),
            row_start,
            m_valid,
            m,
            wgmma_layout,
        )
        dp_l_f32 = dleft_t.astype(jnp.float32) * s_l
        dp_dg_smem.at[:, :h][...] = dp_l_f32.astype(dtype)
        q_l = (dp_l_f32 * (1.0 - s_l)).astype(dtype)
        p_l = _wgmma_matmul(act_smem, w_proj_smem.at[:, :h], tile_m, h)
        common.warpgroup_barrier()
        dleft_smem[...] = (q_l.astype(jnp.float32) * p_l).astype(dtype)

        g_r = _wgmma_matmul(act_smem, w_gate_smem.at[:, h:], tile_m, h)
        s_r = jnp.where(mask_h, common.sigmoid_f32(g_r), 0.0)
        dright_t = common.mask_rows_2d(
            plgpu.load(dright_smem.T, layout=wgmma_layout),
            row_start,
            m_valid,
            m,
            wgmma_layout,
        )
        common.warpgroup_barrier_and_commit_smem()
        dp_r_f32 = dright_t.astype(jnp.float32) * s_r
        dp_dg_smem.at[:, h:][...] = dp_r_f32.astype(dtype)
        q_r = (dp_r_f32 * (1.0 - s_r)).astype(dtype)
        p_r = _wgmma_matmul(act_smem, w_proj_smem.at[:, h:], tile_m, h)
        dg_r = (q_r.astype(jnp.float32) * p_r).astype(dtype)
        plgpu.commit_smem()

        def _accum_proj_and_gate(dact_acc):
          plgpu.wgmma(dw_proj_acc_ref, act_smem.T, dp_dg_smem)
          plgpu.wgmma(dact_acc, dp_dg_smem, w_proj_smem.T)
          if common.is_interpret_mode():
            _ = plgpu.wgmma_accumulator_load(dact_acc, wait_n=0)
          else:
            plgpu.wgmma_wait(0)
          common.warpgroup_barrier()
          dp_dg_smem.at[:, :h][...] = plgpu.load(
              dleft_smem, layout=wgmma_layout
          )
          dp_dg_smem.at[:, h:][...] = dg_r
          common.warpgroup_barrier_and_commit_smem()
          plgpu.wgmma(dw_gate_acc_ref, act_smem.T, dp_dg_smem)
          plgpu.wgmma(dact_acc, dp_dg_smem, w_gate_smem.T)
          return plgpu.wgmma_accumulator_load(dact_acc, wait_n=0)

        dact = pl.run_scoped(
            _accum_proj_and_gate, plgpu.ACC((tile_m, c), jnp.float32)
        )

        plgpu.barrier_wait(dact_barrier)
        dact_in = plgpu.load(x_smem.at[slot], layout=wgmma_layout).astype(
            jnp.float32
        )
        dact_in = common.mask_rows_2d(
            dact_in, row_start, m_valid, m, wgmma_layout
        )
        common.warpgroup_barrier_and_commit_smem()
        common.copy_gmem_to_smem(
            x_gmem.at[next_slice, :],
            x_smem.at[slot],
            x_barrier.at[slot],
            predicate=next_pred,
        )

        dact = dact + dact_in
        x_hat_reload = plgpu.load(x_hat_smem, layout=wgmma_layout)
        dx, acc_s, acc_o = common.layer_norm_2d_bwd_accum(
            dact,
            x_hat_reload,
            rstd_in,
            ln_scale,
            c_real,
            wgmma_layout,
            acc_s,
            acc_o,
        )

        common.warpgroup_barrier()
        act_smem[...] = dx.astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(act_smem, dx_gmem.at[slice_m, :])
        return acc_s, acc_o

      acc_s, acc_o = lax.fori_loop(
          0, num_local_steps, loop_body, (init_acc_s, init_acc_o)
      )
      dw_proj = plgpu.wgmma_accumulator_load(dw_proj_acc_ref, wait_n=0)
      plgpu.store(grad_part_gmem.at[pid, :c, :], dw_proj, optimized=False)
      dw_gate = plgpu.wgmma_accumulator_load(dw_gate_acc_ref, wait_n=0)
      plgpu.store(
          grad_part_gmem.at[pid, pl.ds(c, c), :], dw_gate, optimized=False
      )
      plgpu.store(
          grad_part_gmem.at[pid, pl.ds(2 * c, 32), :c],
          acc_s,
          optimized=False,
      )
      plgpu.store(
          grad_part_gmem.at[pid, pl.ds(2 * c + 32, 32), :c],
          acc_o,
          optimized=False,
      )

    pl.run_scoped(
        _run_with_accs,
        plgpu.ACC((c, 2 * h), jnp.float32),
        plgpu.ACC((c, 2 * h), jnp.float32),
    )
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  out_type = (
      jax.ShapeDtypeStruct((m, c), dtype),
      jax.ShapeDtypeStruct((grid_m, 2 * c + 64, 2 * h), jnp.float32),
  )
  scratch_types = (
      mgpu_lib.tiled_swizzled_smem((stages, tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
      mgpu_lib.tiled_swizzled_smem((h, tile_m), dtype),
      mgpu_lib.tiled_swizzled_smem((c, 2 * h), dtype),
      mgpu_lib.tiled_swizzled_smem((c, 2 * h), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, c), dtype),
      mgpu_lib.tiled_swizzled_smem((tile_m, 2 * h), dtype),
      plgpu.Barrier(num_arrivals=1, num_barriers=stages),
      plgpu.Barrier(num_arrivals=2),
      plgpu.Barrier(num_arrivals=1),
      plgpu.Barrier(num_arrivals=2),
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

