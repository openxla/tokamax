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
"""Pallas-Mosaic-TPU Triangle Multiplication kernels (v4, v5p, v6e, 7x)."""

import math
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops import op
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_tpu_common as common


def _get_tpu_info() -> tuple[int, int, int, int | None]:
  """Returns `(generation, vmem_mib, num_cores, vmem_limit_bytes)`."""
  try:
    info = pltpu.get_tpu_info()
    return (
        info.generation,
        info.vmem_capacity_bytes // (1024 * 1024),
        info.num_cores,
        int(0.90 * info.vmem_capacity_bytes),
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return 6, 64, 1, None


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns hardware-tailored forward config for TPU v4, v5p, v6e, and 7x."""
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  n, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, _, _ = _get_tpu_info()

  if vmem_mib <= 16 or gen <= 4:
    return common.Config(block_m=256 if max_chd > 64 else 512)

  if max_chd > 64:
    return common.Config(block_m=512)

  if gen <= 6 and n <= 384:
    return common.Config(block_m=512)

  return common.Config(block_m=1024)


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns forward autotuning configs tailored to the current TPU."""
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  _, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, _, _ = _get_tpu_info()

  configs = {get_heuristics_config(ba)}
  if vmem_mib <= 16 or gen <= 4:
    block_m_candidates = (128, 256, 512) if max_chd <= 64 else (128, 256)
  else:
    block_m_candidates = (
        (128, 256, 512, 1024) if max_chd <= 64 else (128, 256, 512)
    )

  for block_m in block_m_candidates:
    configs.add(common.Config(block_m=block_m))
  return configs


def get_vjp_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns hardware-tailored backward config for TPU v4, v5p, v6e, and 7x."""
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  n, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, _, _ = _get_tpu_info()

  if vmem_mib <= 16 or gen <= 4:
    return common.Config(block_m=128 if max_chd > 64 else 512)

  if max_chd > 64:
    return common.Config(block_m=256 if vmem_mib <= 32 else 512)

  if gen <= 6 and n <= 384:
    return common.Config(block_m=512)

  return common.Config(block_m=1024)


def get_vjp_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns backward autotuning configs tailored to the current TPU."""
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  _, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, _, _ = _get_tpu_info()

  configs = {get_vjp_heuristics_config(ba)}
  if vmem_mib <= 16 or gen <= 4:
    block_m_candidates = (128, 256, 512) if max_chd <= 64 else (128,)
  else:
    block_m_candidates = (
        (128, 256, 512, 1024) if max_chd <= 64 else (128, 256, 512)
    )

  for block_m in block_m_candidates:
    configs.add(common.Config(block_m=block_m))
  return configs


def resolve_blocks(
    n_pad: int,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
) -> tuple[int, int]:
  """Maps `block_m` (or explicit overrides) to `(block_i, block_j)`."""
  if block_i is not None and block_j is not None:
    bj = min(block_j, n_pad)
    if n_pad % block_i == 0 and n_pad % bj == 0:
      return block_i, bj

  _, vmem_mib, _, _ = _get_tpu_info()

  if block_m <= 128:
    return (8 if vmem_mib <= 16 else 16), 128

  if block_m == 256:
    if vmem_mib <= 16:
      return (8, 256) if n_pad % 256 == 0 else (8, 128)
    if n_pad % 256 == 0:
      return 16, 256
    return max(16, math.gcd(n_pad, 32)), 128

  if block_m == 512:
    if vmem_mib <= 16:
      return 16, 128
    for cand_j in (640, 512, 384, 256):
      if n_pad % cand_j == 0:
        return 16, cand_j
    return max(16, math.gcd(n_pad, 64)), 128

  top_cands = (
      (2048, 1024, 896, 768, 640, 512, 384, 256)
      if vmem_mib > 64
      else (1024, 896, 768, 640, 512, 384, 256)
  )
  for cand_j in top_cands:
    if n_pad % cand_j == 0:
      bi = 16 if cand_j >= 1024 else min(n_pad, 32)
      if n_pad % bi == 0:
        return bi, cand_j
      return 16, cand_j
  return min(n_pad, 64), 128


_resolve_blocks = resolve_blocks


def _scoped_vmem_limit(
    gen: int,
    block_j: int,
    max_dim: int,
    vmem_mib: int,
    vmem_limit_bytes: int | None,
) -> int | None:
  """Leaves VMEM unreserved for XLA MSA when tiles fit in 32 MiB scoped VMEM."""
  if gen >= 6 and vmem_mib > 16 and block_j <= 512 and max_dim <= 64:
    return None
  return vmem_limit_bytes


def _build_bwd_grid(
    num_i: int, num_j: int, num_cores: int
) -> tuple[tuple[int, int, int], int, bool]:
  """Builds a division-free 3D backward grid `(cores, steps_i, steps_j)`."""
  if num_cores > 1 and num_i % num_cores == 0:
    return (num_cores, num_i // num_cores, num_j), num_cores, True
  if num_cores > 1 and num_j % num_cores == 0:
    return (num_cores, num_i, num_j // num_cores), num_cores, False
  return (1, num_i, num_j), 1, True


def _build_block_diag_out_weights_t(
    gate_out_weights: jax.Array,
    projection_out_weights: jax.Array,
) -> jax.Array:
  """Builds `(2 * D, C + H)` block-diagonal transposed epilogue weights."""
  c, d = gate_out_weights.shape
  h = projection_out_weights.shape[0]
  top = jnp.concatenate(
      [gate_out_weights.T, jnp.zeros((d, h), dtype=gate_out_weights.dtype)],
      axis=1,
  )
  bot = jnp.concatenate(
      [
          jnp.zeros((d, c), dtype=projection_out_weights.dtype),
          projection_out_weights.T,
      ],
      axis=1,
  )
  return jnp.concatenate([top, bot], axis=0)


def prologue(
    x_pad: jax.Array,
    mask_pad: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    *,
    eps: float,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Fused Pallas Mosaic TPU prologue kernel."""
  n_pad, c, _ = x_pad.shape
  h = projection_in_weights.shape[2]
  dtype = x_pad.dtype
  block_i, block_j = resolve_blocks(n_pad, block_m, block_i, block_j)
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  gen, vmem_mib, num_cores, vmem_limit_bytes = _get_tpu_info()
  par_sem = "arbitrary" if (num_cores > 1 and n_pad <= 256) else "parallel"
  vmem_limit = _scoped_vmem_limit(
      gen, block_j, max(c, h), vmem_mib, vmem_limit_bytes
  )

  use_4d_t8 = gen >= 7
  if use_4d_t8:
    x_in = x_pad.reshape(n_pad, c // 8, 8, n_pad)
    x_spec = pl.BlockSpec(
        (block_i, c // 8, 8, block_j), lambda i, j: (i, 0, 0, j)
    )
  else:
    x_in = x_pad
    x_spec = pl.BlockSpec((block_i, c, block_j), lambda i, j: (i, 0, j))

  w_proj_t = projection_in_weights.reshape(c, 2 * h).T
  w_gate_t = gate_in_weights.reshape(c, 2 * h).T
  out_shape = jax.ShapeDtypeStruct((h, n_pad, n_pad), dtype)
  out_spec = pl.BlockSpec((h, block_i, block_j), lambda i, j: (0, i, j))

  if 4 * h <= 256:
    w_pg_arg = jnp.concatenate([w_proj_t, w_gate_t], axis=0)

    def kernel_4h(
        x_ref,
        mask_ref,
        w_pg_ref,
        ln_scale_ref,
        ln_offset_ref,
        left_ref,
        right_ref,
    ):
      x_raw = (
          x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
      )
      act = common.layer_norm_3d_axis1(
          x_raw, ln_scale_ref[...], ln_offset_ref[...], eps
      )
      pg = common.matmul_icj(w_pg_ref[...], act)
      lr = (pg[:, : 2 * h] * jax.nn.sigmoid(pg[:, 2 * h :])).astype(dtype)
      lr = jnp.where(
          lax.broadcast_in_dim(mask_ref[...] != 0, lr.shape, (0, 2)),
          lr,
          0,
      )
      left_ref[...] = jnp.transpose(lr[:, :h, :], (1, 0, 2))
      right_ref[...] = jnp.transpose(lr[:, h:, :], (1, 0, 2))

    return pl.pallas_call(
        kernel_4h,
        out_shape=(out_shape, out_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_i, num_j),
            in_specs=[
                x_spec,
                pl.BlockSpec((block_i, block_j), lambda i, j: (i, j)),
                pl.BlockSpec(w_pg_arg.shape, lambda i, j: (0, 0)),
                pl.BlockSpec((c,), lambda i, j: (0,)),
                pl.BlockSpec((c,), lambda i, j: (0,)),
            ],
            out_specs=(out_spec, out_spec),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(par_sem, par_sem),
            vmem_limit_bytes=vmem_limit,
        ),
    )(
        x_in,
        mask_pad,
        w_pg_arg,
        layernorm_in_scale,
        layernorm_in_offset,
    )

  def kernel_2h(
      x_ref,
      mask_ref,
      w_proj_ref,
      w_gate_ref,
      ln_scale_ref,
      ln_offset_ref,
      left_ref,
      right_ref,
  ):
    x_raw = x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    act = common.layer_norm_3d_axis1(
        x_raw, ln_scale_ref[...], ln_offset_ref[...], eps
    )
    p = common.matmul_icj(w_proj_ref[...], act)
    g = common.matmul_icj(w_gate_ref[...], act)
    lr = (p * jax.nn.sigmoid(g)).astype(dtype)
    lr = jnp.where(
        lax.broadcast_in_dim(mask_ref[...] != 0, lr.shape, (0, 2)),
        lr,
        0,
    )
    left_ref[...] = jnp.transpose(lr[:, :h, :], (1, 0, 2))
    right_ref[...] = jnp.transpose(lr[:, h:, :], (1, 0, 2))

  return pl.pallas_call(
      kernel_2h,
      out_shape=(out_shape, out_shape),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_i, num_j),
          in_specs=[
              x_spec,
              pl.BlockSpec((block_i, block_j), lambda i, j: (i, j)),
              pl.BlockSpec(w_proj_t.shape, lambda i, j: (0, 0)),
              pl.BlockSpec(w_gate_t.shape, lambda i, j: (0, 0)),
              pl.BlockSpec((c,), lambda i, j: (0,)),
              pl.BlockSpec((c,), lambda i, j: (0,)),
          ],
          out_specs=(out_spec, out_spec),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(par_sem, par_sem),
          vmem_limit_bytes=vmem_limit,
      ),
  )(
      x_in,
      mask_pad,
      w_proj_t,
      w_gate_t,
      layernorm_in_scale,
      layernorm_in_offset,
  )


def epilogue(
    t3_pad: jax.Array,
    x_pad: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    *,
    eps: float,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
) -> jax.Array:
  """Fused Pallas Mosaic TPU epilogue kernel."""
  n_pad, c, _ = x_pad.shape
  h, d = projection_out_weights.shape
  dtype = x_pad.dtype
  block_i, block_j = resolve_blocks(n_pad, block_m, block_i, block_j)
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  gen, vmem_mib, num_cores, vmem_limit_bytes = _get_tpu_info()
  par_sem = "arbitrary" if (num_cores > 1 and n_pad <= 256) else "parallel"
  vmem_limit = _scoped_vmem_limit(
      gen, block_j, max(c, h, d), vmem_mib, vmem_limit_bytes
  )

  use_4d_t8 = gen >= 7
  if use_4d_t8:
    x_in = x_pad.reshape(n_pad, c // 8, 8, n_pad)
    x_spec = pl.BlockSpec(
        (block_i, c // 8, 8, block_j), lambda i, j: (i, 0, 0, j)
    )
    out_shape = jax.ShapeDtypeStruct((n_pad, d // 8, 8, n_pad), dtype)
    out_spec = pl.BlockSpec(
        (block_i, d // 8, 8, block_j), lambda i, j: (i, 0, 0, j)
    )
  else:
    x_in = x_pad
    x_spec = pl.BlockSpec((block_i, c, block_j), lambda i, j: (i, 0, j))
    out_shape = jax.ShapeDtypeStruct((n_pad, d, n_pad), dtype)
    out_spec = pl.BlockSpec((block_i, d, block_j), lambda i, j: (i, 0, j))

  w_gp_out_t = _build_block_diag_out_weights_t(
      gate_out_weights, projection_out_weights
  )

  def kernel_3d(
      t3_ref,
      x_ref,
      w_gp_ref,
      ln_in_scale_ref,
      ln_in_offset_ref,
      ln_out_scale_ref,
      ln_out_offset_ref,
      out_ref,
  ):
    x_raw = x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    left_act = common.layer_norm_3d_axis1(
        x_raw, ln_in_scale_ref[...], ln_in_offset_ref[...], eps
    )
    t_hat_3d, _ = common.layer_norm_3d_axis0_hat_rstd(t3_ref[...], eps)
    act_out = jnp.transpose(
        common.apply_norm_affine_axis0(
            t_hat_3d, ln_out_scale_ref[...], ln_out_offset_ref[...]
        ).astype(dtype),
        (1, 0, 2),
    )
    gp = common.matmul_icj(
        w_gp_ref[...], jnp.concatenate([left_act, act_out], axis=1)
    )
    out_idj = (gp[:, d:, :] * jax.nn.sigmoid(gp[:, :d, :])).astype(dtype)
    out_ref[...] = (
        out_idj.reshape(block_i, d // 8, 8, block_j) if use_4d_t8 else out_idj
    )

  out = pl.pallas_call(
      kernel_3d,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_i, num_j),
          in_specs=[
              pl.BlockSpec((h, block_i, block_j), lambda i, j: (0, i, j)),
              x_spec,
              pl.BlockSpec((2 * d, c + h), lambda i, j: (0, 0)),
              pl.BlockSpec((c,), lambda i, j: (0,)),
              pl.BlockSpec((c,), lambda i, j: (0,)),
              pl.BlockSpec((h,), lambda i, j: (0,)),
              pl.BlockSpec((h,), lambda i, j: (0,)),
          ],
          out_specs=out_spec,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(par_sem, par_sem),
          vmem_limit_bytes=vmem_limit,
      ),
  )(
      t3_pad,
      x_in,
      w_gp_out_t,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )
  return out.reshape(n_pad, d, n_pad) if use_4d_t8 else out


def epilogue_bwd(
    dout_pad: jax.Array,
    t3_pad: jax.Array,
    x_pad: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    *,
    eps: float,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
  """Fused Pallas Mosaic TPU epilogue backward kernel."""
  n_pad, c, _ = x_pad.shape
  h, d = projection_out_weights.shape
  dtype = x_pad.dtype
  gen, vmem_mib, num_cores, vmem_limit_bytes = _get_tpu_info()
  max_chd = max(c, h, d)
  if (
      block_i is None
      and block_j is None
      and (vmem_mib <= 16 or gen <= 4 or max_chd > 64)
  ):
    block_m = min(
        block_m,
        128
        if (vmem_mib <= 16 and max_chd > 64)
        else (256 if vmem_mib <= 32 else 512),
    )

  block_i, block_j = resolve_blocks(n_pad, block_m, block_i, block_j)
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  vmem_limit = _scoped_vmem_limit(
      gen, block_j, max_chd, vmem_mib, vmem_limit_bytes
  )
  grid, num_cores_used, split_i = _build_bwd_grid(num_i, num_j, num_cores)
  steps_i = grid[1]
  steps_j = grid[2]

  def idx_i(c_idx, si, sj):
    del sj
    return c_idx * steps_i + si if split_i else si

  def idx_j(c_idx, si, sj):
    del si
    return sj if split_i else c_idx * steps_j + sj

  use_4d_t8 = gen >= 7
  if use_4d_t8:
    x_in = x_pad.reshape(n_pad, c // 8, 8, n_pad)
    dout_in = dout_pad.reshape(n_pad, d // 8, 8, n_pad)
    x_spec = pl.BlockSpec(
        (block_i, c // 8, 8, block_j),
        lambda c_idx, si, sj: (
            idx_i(c_idx, si, sj),
            0,
            0,
            idx_j(c_idx, si, sj),
        ),
    )
    dout_spec = pl.BlockSpec(
        (block_i, d // 8, 8, block_j),
        lambda c_idx, si, sj: (
            idx_i(c_idx, si, sj),
            0,
            0,
            idx_j(c_idx, si, sj),
        ),
    )
    dleft_epi_shape = jax.ShapeDtypeStruct((n_pad, c // 8, 8, n_pad), dtype)
    dleft_epi_spec = x_spec
  else:
    x_in = x_pad
    dout_in = dout_pad
    x_spec = pl.BlockSpec(
        (block_i, c, block_j),
        lambda c_idx, si, sj: (idx_i(c_idx, si, sj), 0, idx_j(c_idx, si, sj)),
    )
    dout_spec = pl.BlockSpec(
        (block_i, d, block_j),
        lambda c_idx, si, sj: (idx_i(c_idx, si, sj), 0, idx_j(c_idx, si, sj)),
    )
    dleft_epi_shape = jax.ShapeDtypeStruct((n_pad, c, n_pad), dtype)
    dleft_epi_spec = x_spec

  hij_spec = pl.BlockSpec(
      (h, block_i, block_j),
      lambda c_idx, si, sj: (0, idx_i(c_idx, si, sj), idx_j(c_idx, si, sj)),
  )

  w_gp_out_t = _build_block_diag_out_weights_t(
      gate_out_weights, projection_out_weights
  )
  dw_shape = w_gp_out_t.shape

  def kernel_bwd_3d(
      dout_ref,
      t3_ref,
      x_ref,
      w_gp_ref,
      ln_in_scale_ref,
      ln_in_offset_ref,
      ln_out_scale_ref,
      ln_out_offset_ref,
      dt3_ref,
      dleft_act_epi_ref,
      dw_gp_ref,
      dln_out_ref,
  ):
    is_first = (
        pl.program_id(1) == 0
        if steps_j == 1
        else (pl.program_id(1) == 0) & (pl.program_id(2) == 0)
    )

    @pl.when(is_first)
    def _init_accs():
      dw_gp_ref[...] = jnp.zeros_like(dw_gp_ref)
      dln_out_ref[...] = jnp.zeros_like(dln_out_ref)

    w_gp_val = w_gp_ref[...]
    ln_out_scale = ln_out_scale_ref[...]

    x_raw = x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    left_act = common.layer_norm_3d_axis1(
        x_raw,
        ln_in_scale_ref[...],
        ln_in_offset_ref[...],
        eps,
        sanitize_nonfinite=True,
    )
    t_hat, rstd_out = common.layer_norm_3d_axis1_hat_rstd(
        jnp.transpose(t3_ref[...], (1, 0, 2)), eps
    )
    act_out = common.apply_norm_affine_axis1(
        t_hat, ln_out_scale, ln_out_offset_ref[...]
    ).astype(dtype)

    la_ao = jnp.concatenate([left_act, act_out], axis=1)
    gp = common.matmul_icj(w_gp_val, la_ao)
    gate = jax.nn.sigmoid(gp[:, :d, :])
    p = gp[:, d:, :]

    dout_raw = (
        dout_ref[...].reshape(block_i, d, block_j)
        if use_4d_t8
        else dout_ref[...]
    )
    dout_f32 = dout_raw.astype(jnp.float32)
    dp = (dout_f32 * gate).astype(dtype)
    dg = (dout_f32 * p * gate * (1.0 - gate)).astype(dtype)
    dg_dp = jnp.concatenate([dg, dp], axis=1)

    dw_gp_ref[...] = dw_gp_ref[...] + common.matmul_dw_icj(
        dg_dp, la_ao
    ).reshape(1, *dw_shape)

    dla_dao = common.matmul_wt_icj(w_gp_val, dg_dp)
    dla_icj = dla_dao[:, :c, :].astype(dtype)
    dleft_act_epi_ref[...] = (
        dla_icj.reshape(block_i, c // 8, 8, block_j) if use_4d_t8 else dla_icj
    )

    dt3_ihj, dln_out_tile = common.layer_norm_3d_axis1_bwd(
        dla_dao[:, c:, :], t_hat, rstd_out, ln_out_scale, dtype
    )
    dt3_ref[...] = jnp.transpose(dt3_ihj, (1, 0, 2))
    dln_out_ref[...] = dln_out_ref[...] + dln_out_tile

  (
      dt3_pad,
      dleft_act_epi_out,
      dw_gp_part,
      dln_out_part,
  ) = pl.pallas_call(
      kernel_bwd_3d,
      out_shape=(
          jax.ShapeDtypeStruct((h, n_pad, n_pad), dtype),
          dleft_epi_shape,
          jax.ShapeDtypeStruct((num_cores_used, *dw_shape), jnp.float32),
          jax.ShapeDtypeStruct((num_cores_used, 2, h), jnp.float32),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              dout_spec,
              hij_spec,
              x_spec,
              pl.BlockSpec(dw_shape, lambda c_idx, si, sj: (0, 0)),
              pl.BlockSpec((c,), lambda c_idx, si, sj: (0,)),
              pl.BlockSpec((c,), lambda c_idx, si, sj: (0,)),
              pl.BlockSpec((h,), lambda c_idx, si, sj: (0,)),
              pl.BlockSpec((h,), lambda c_idx, si, sj: (0,)),
          ],
          out_specs=(
              hij_spec,
              dleft_epi_spec,
              pl.BlockSpec((1, *dw_shape), lambda c_idx, si, sj: (c_idx, 0, 0)),
              pl.BlockSpec((1, 2, h), lambda c_idx, si, sj: (c_idx, 0, 0)),
          ),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=vmem_limit,
      ),
  )(
      dout_in,
      t3_pad,
      x_in,
      w_gp_out_t,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )

  dleft_act_epi_pad = (
      dleft_act_epi_out.reshape(n_pad, c, n_pad)
      if use_4d_t8
      else dleft_act_epi_out
  )
  dw_gp = dw_gp_part[0] if num_cores_used == 1 else jnp.sum(dw_gp_part, axis=0)
  dw_gate_out = dw_gp[:d, :c].T.astype(dtype)
  dw_proj_out = dw_gp[d:, c:].T.astype(dtype)
  dln_out = (
      dln_out_part[0] if num_cores_used == 1 else jnp.sum(dln_out_part, axis=0)
  ).astype(dtype)

  return (
      dt3_pad,
      dleft_act_epi_pad,
      dw_proj_out,
      dw_gate_out,
      dln_out[0],
      dln_out[1],
  )


def prologue_bwd(
    dleft_pad: jax.Array,
    dright_pad: jax.Array,
    dleft_act_epi_pad: jax.Array,
    x_pad: jax.Array,
    mask_pad: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    *,
    eps: float,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Fused Pallas Mosaic TPU prologue backward kernel."""
  n_pad, c, _ = x_pad.shape
  h = projection_in_weights.shape[2]
  dtype = x_pad.dtype
  gen, vmem_mib, num_cores, vmem_limit_bytes = _get_tpu_info()
  max_ch = max(c, h)
  if (
      block_i is None
      and block_j is None
      and (vmem_mib <= 16 or gen <= 4 or max_ch > 64)
  ):
    block_m = min(
        block_m,
        128
        if (vmem_mib <= 16 and max_ch > 64)
        else (256 if vmem_mib <= 32 else 512),
    )

  block_i, block_j = resolve_blocks(n_pad, block_m, block_i, block_j)
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  vmem_limit = _scoped_vmem_limit(
      gen, block_j, max_ch, vmem_mib, vmem_limit_bytes
  )
  grid, num_cores_used, split_i = _build_bwd_grid(num_i, num_j, num_cores)
  steps_i = grid[1]
  steps_j = grid[2]

  def idx_i(c_idx, si, sj):
    del sj
    return c_idx * steps_i + si if split_i else si

  def idx_j(c_idx, si, sj):
    del si
    return sj if split_i else c_idx * steps_j + sj

  w_proj_t = projection_in_weights.reshape(c, 2 * h).T
  w_gate_t = gate_in_weights.reshape(c, 2 * h).T

  use_4d_t8 = gen >= 7
  if use_4d_t8:
    x_in = x_pad.reshape(n_pad, c // 8, 8, n_pad)
    dleft_epi_in = dleft_act_epi_pad.reshape(n_pad, c // 8, 8, n_pad)
    x_spec = pl.BlockSpec(
        (block_i, c // 8, 8, block_j),
        lambda c_idx, si, sj: (
            idx_i(c_idx, si, sj),
            0,
            0,
            idx_j(c_idx, si, sj),
        ),
    )
    dx_shape = jax.ShapeDtypeStruct((n_pad, c // 8, 8, n_pad), dtype)
  else:
    x_in = x_pad
    dleft_epi_in = dleft_act_epi_pad
    x_spec = pl.BlockSpec(
        (block_i, c, block_j),
        lambda c_idx, si, sj: (idx_i(c_idx, si, sj), 0, idx_j(c_idx, si, sj)),
    )
    dx_shape = jax.ShapeDtypeStruct((n_pad, c, n_pad), dtype)

  hij_spec = pl.BlockSpec(
      (h, block_i, block_j),
      lambda c_idx, si, sj: (0, idx_i(c_idx, si, sj), idx_j(c_idx, si, sj)),
  )
  mask_spec = pl.BlockSpec(
      (block_i, block_j),
      lambda c_idx, si, sj: (idx_i(c_idx, si, sj), idx_j(c_idx, si, sj)),
  )

  if 4 * h <= 256:
    w_pg_arg = jnp.concatenate([w_proj_t, w_gate_t], axis=0)
    dw_pg_shape = w_pg_arg.shape

    def kernel_4h(
        dleft_ref,
        dright_ref,
        dleft_act_epi_ref,
        x_ref,
        mask_ref,
        w_pg_ref,
        ln_in_scale_ref,
        ln_in_offset_ref,
        dx_ref,
        dw_pg_ref,
        dln_in_ref,
    ):
      is_first = (
          pl.program_id(1) == 0
          if steps_j == 1
          else (pl.program_id(1) == 0) & (pl.program_id(2) == 0)
      )

      @pl.when(is_first)
      def _init_accs():
        dw_pg_ref[...] = jnp.zeros_like(dw_pg_ref)
        dln_in_ref[...] = jnp.zeros_like(dln_in_ref)

      w_pg_val = w_pg_ref[...]
      ln_in_scale = ln_in_scale_ref[...]

      x_raw = (
          x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
      )
      x_hat, rstd = common.layer_norm_3d_axis1_hat_rstd(
          x_raw, eps, sanitize_nonfinite=True
      )
      act = common.apply_norm_affine_axis1(
          x_hat, ln_in_scale, ln_in_offset_ref[...]
      ).astype(dtype)

      pg = common.matmul_icj(w_pg_val, act)
      p = pg[:, : 2 * h, :]
      sig = jax.nn.sigmoid(pg[:, 2 * h :, :])

      dlr = jnp.transpose(
          jnp.concatenate([dleft_ref[...], dright_ref[...]], axis=0),
          (1, 0, 2),
      )
      dlr = jnp.where(
          lax.broadcast_in_dim(mask_ref[...] != 0, dlr.shape, (0, 2)),
          dlr,
          0,
      ).astype(jnp.float32)

      dp = (dlr * sig).astype(dtype)
      dg = (dlr * p * sig * (1.0 - sig)).astype(dtype)
      dpg = jnp.concatenate([dp, dg], axis=1)

      dw_pg_ref[...] = dw_pg_ref[...] + common.matmul_dw_icj(dpg, act).reshape(
          1, *dw_pg_shape
      )

      dleft_epi_raw = (
          dleft_act_epi_ref[...].reshape(block_i, c, block_j)
          if use_4d_t8
          else dleft_act_epi_ref[...]
      )
      dact_total = common.matmul_wt_icj(w_pg_val, dpg) + dleft_epi_raw.astype(
          jnp.float32
      )
      dx, dln_in_tile = common.layer_norm_3d_axis1_bwd(
          dact_total, x_hat, rstd, ln_in_scale, dtype
      )
      dx_ref[...] = dx.reshape(block_i, c // 8, 8, block_j) if use_4d_t8 else dx
      dln_in_ref[...] = dln_in_ref[...] + dln_in_tile

    (
        dx_out,
        dw_pg_part,
        dln_in_part,
    ) = pl.pallas_call(
        kernel_4h,
        out_shape=(
            dx_shape,
            jax.ShapeDtypeStruct((num_cores_used, *dw_pg_shape), jnp.float32),
            jax.ShapeDtypeStruct((num_cores_used, 2, c), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                hij_spec,
                hij_spec,
                x_spec,
                x_spec,
                mask_spec,
                pl.BlockSpec(dw_pg_shape, lambda c_idx, si, sj: (0, 0)),
                pl.BlockSpec((c,), lambda c_idx, si, sj: (0,)),
                pl.BlockSpec((c,), lambda c_idx, si, sj: (0,)),
            ],
            out_specs=(
                x_spec,
                pl.BlockSpec(
                    (1, *dw_pg_shape), lambda c_idx, si, sj: (c_idx, 0, 0)
                ),
                pl.BlockSpec((1, 2, c), lambda c_idx, si, sj: (c_idx, 0, 0)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit,
        ),
    )(
        dleft_pad,
        dright_pad,
        dleft_epi_in,
        x_in,
        mask_pad,
        w_pg_arg,
        layernorm_in_scale,
        layernorm_in_offset,
    )

    dx_pad = dx_out.reshape(n_pad, c, n_pad) if use_4d_t8 else dx_out
    dw_pg_sum = (
        dw_pg_part[0] if num_cores_used == 1 else jnp.sum(dw_pg_part, axis=0)
    )
    dw_pg = dw_pg_sum.T.astype(dtype)
    dln_in = (
        dln_in_part[0] if num_cores_used == 1 else jnp.sum(dln_in_part, axis=0)
    ).astype(dtype)
    return (
        dx_pad,
        dw_pg[:, : 2 * h].reshape(c, 2, h),
        dw_pg[:, 2 * h :].reshape(c, 2, h),
        dln_in[0],
        dln_in[1],
    )

  dw_2h_shape = w_proj_t.shape

  def kernel_2h(
      dleft_ref,
      dright_ref,
      dleft_act_epi_ref,
      x_ref,
      mask_ref,
      w_proj_ref,
      w_gate_ref,
      ln_in_scale_ref,
      ln_in_offset_ref,
      dx_ref,
      dw_proj_ref,
      dw_gate_ref,
      dln_in_ref,
  ):
    is_first = (
        pl.program_id(1) == 0
        if steps_j == 1
        else (pl.program_id(1) == 0) & (pl.program_id(2) == 0)
    )

    @pl.when(is_first)
    def _init_accs():
      dw_proj_ref[...] = jnp.zeros_like(dw_proj_ref)
      dw_gate_ref[...] = jnp.zeros_like(dw_gate_ref)
      dln_in_ref[...] = jnp.zeros_like(dln_in_ref)

    w_proj_val = w_proj_ref[...]
    w_gate_val = w_gate_ref[...]
    ln_in_scale = ln_in_scale_ref[...]

    x_raw = x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    x_hat, rstd = common.layer_norm_3d_axis1_hat_rstd(
        x_raw, eps, sanitize_nonfinite=True
    )
    act = common.apply_norm_affine_axis1(
        x_hat, ln_in_scale, ln_in_offset_ref[...]
    ).astype(dtype)

    p = common.matmul_icj(w_proj_val, act)
    sig = jax.nn.sigmoid(common.matmul_icj(w_gate_val, act))

    dlr = jnp.transpose(
        jnp.concatenate([dleft_ref[...], dright_ref[...]], axis=0),
        (1, 0, 2),
    )
    dlr = jnp.where(
        lax.broadcast_in_dim(mask_ref[...] != 0, dlr.shape, (0, 2)),
        dlr,
        0,
    ).astype(jnp.float32)

    dp = (dlr * sig).astype(dtype)
    dg = (dlr * p * sig * (1.0 - sig)).astype(dtype)

    dw_proj_ref[...] = dw_proj_ref[...] + common.matmul_dw_icj(
        dp, act
    ).reshape(1, *dw_2h_shape)
    dw_gate_ref[...] = dw_gate_ref[...] + common.matmul_dw_icj(
        dg, act
    ).reshape(1, *dw_2h_shape)

    dleft_epi_raw = (
        dleft_act_epi_ref[...].reshape(block_i, c, block_j)
        if use_4d_t8
        else dleft_act_epi_ref[...]
    )
    dact_total = (
        common.matmul_wt_icj(w_proj_val, dp)
        + common.matmul_wt_icj(w_gate_val, dg)
        + dleft_epi_raw.astype(jnp.float32)
    )
    dx, dln_in_tile = common.layer_norm_3d_axis1_bwd(
        dact_total, x_hat, rstd, ln_in_scale, dtype
    )
    dx_ref[...] = dx.reshape(block_i, c // 8, 8, block_j) if use_4d_t8 else dx
    dln_in_ref[...] = dln_in_ref[...] + dln_in_tile

  (
      dx_out,
      dw_proj_part,
      dw_gate_part,
      dln_in_part,
  ) = pl.pallas_call(
      kernel_2h,
      out_shape=(
          dx_shape,
          jax.ShapeDtypeStruct((num_cores_used, *dw_2h_shape), jnp.float32),
          jax.ShapeDtypeStruct((num_cores_used, *dw_2h_shape), jnp.float32),
          jax.ShapeDtypeStruct((num_cores_used, 2, c), jnp.float32),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              hij_spec,
              hij_spec,
              x_spec,
              x_spec,
              mask_spec,
              pl.BlockSpec(dw_2h_shape, lambda c_idx, si, sj: (0, 0)),
              pl.BlockSpec(dw_2h_shape, lambda c_idx, si, sj: (0, 0)),
              pl.BlockSpec((c,), lambda c_idx, si, sj: (0,)),
              pl.BlockSpec((c,), lambda c_idx, si, sj: (0,)),
          ],
          out_specs=(
              x_spec,
              pl.BlockSpec(
                  (1, *dw_2h_shape), lambda c_idx, si, sj: (c_idx, 0, 0)
              ),
              pl.BlockSpec(
                  (1, *dw_2h_shape), lambda c_idx, si, sj: (c_idx, 0, 0)
              ),
              pl.BlockSpec((1, 2, c), lambda c_idx, si, sj: (c_idx, 0, 0)),
          ),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=vmem_limit,
      ),
  )(
      dleft_pad,
      dright_pad,
      dleft_epi_in,
      x_in,
      mask_pad,
      w_proj_t,
      w_gate_t,
      layernorm_in_scale,
      layernorm_in_offset,
  )

  dx_pad = dx_out.reshape(n_pad, c, n_pad) if use_4d_t8 else dx_out
  dw_p_sum = (
      dw_proj_part[0] if num_cores_used == 1 else jnp.sum(dw_proj_part, axis=0)
  )
  dw_g_sum = (
      dw_gate_part[0] if num_cores_used == 1 else jnp.sum(dw_gate_part, axis=0)
  )
  dln_in = (
      dln_in_part[0] if num_cores_used == 1 else jnp.sum(dln_in_part, axis=0)
  ).astype(dtype)

  return (
      dx_pad,
      dw_p_sum.T.reshape(c, 2, h).astype(dtype),
      dw_g_sum.T.reshape(c, 2, h).astype(dtype),
      dln_in[0],
      dln_in[1],
  )
