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

from typing import Any, Callable
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops import op
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_tpu_common as common


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns hardware-tailored forward config for TPU v4, v5p, v6e, and 7x."""
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  n, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, num_cores, _, _ = common.get_tpu_info(x)

  if vmem_mib <= 16 or gen <= 4:
    return common.Config(block_m=256 if max_chd > 64 else 512)

  if max_chd > 64:
    return common.Config(block_m=512 if vmem_mib <= 64 else 1024)

  n_pad = pl.cdiv(n, 128) * 128
  if gen <= 6 and n_pad <= 384:
    return common.Config(block_m=512)
  if n_pad >= 512:
    if vmem_mib > 64 and n_pad % 2048 == 0:
      return common.Config(
          block_m=1024, epilogue_block_i=32, epilogue_block_j=2048
      )
    if vmem_mib > 64 and n_pad % 1024 == 0:
      return common.Config(
          block_m=1024,
          block_i=32,
          block_j=1024,
          epilogue_block_i=64,
          epilogue_block_j=1024,
      )
    if n_pad % 1024 == 0:
      return common.Config(
          block_m=1024, epilogue_block_i=32, epilogue_block_j=1024
      )
    if n_pad % 512 == 0:
      epi_bi = 64 if n_pad % 64 == 0 else 32
      if n_pad == 512 and num_cores > 1:
        epi_bi = min(epi_bi, n_pad // num_cores)
      return common.Config(
          block_m=1024, epilogue_block_i=epi_bi, epilogue_block_j=512
      )

  return common.Config(block_m=1024)


def _dedup_autotuning_configs(
    seed: common.Config,
    candidates: tuple[common.Config, ...],
    x: Any,
) -> set[common.Config]:
  """Deduplicates configs by resolved `(prologue, epilogue)` tiles on `x`."""
  n_pad = pl.cdiv(x.shape[0], 128) * 128
  configs = {seed}
  seen = {resolve_config_blocks(n_pad, seed, x=x)}
  for cfg in candidates:
    key = resolve_config_blocks(n_pad, cfg, x=x)
    if key not in seen:
      seen.add(key)
      configs.add(cfg)
  return configs


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns forward autotuning configs tailored to the current TPU."""
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  _, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, _, _, _ = common.get_tpu_info(x)
  seed = get_heuristics_config(ba)

  if vmem_mib <= 16 or gen <= 4:
    if max_chd <= 64:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
      )
    else:
      cands = (common.Config(block_m=128), common.Config(block_m=256))
  elif vmem_mib > 64:
    if max_chd <= 64:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
          common.Config(block_m=1024),
          common.Config(
              block_m=1024,
              block_i=64,
              block_j=512,
              epilogue_block_i=64,
              epilogue_block_j=1024,
          ),
          common.Config(block_m=1024, block_i=16, block_j=1024),
      )
    else:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
          common.Config(block_m=1024),
      )
  else:
    if max_chd <= 64:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
          common.Config(block_m=1024),
          common.Config(
              block_m=1024,
              block_i=16,
              block_j=1024,
              epilogue_block_i=32,
              epilogue_block_j=1024,
          ),
      )
    else:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
      )
  return _dedup_autotuning_configs(seed, cands, x)


def get_vjp_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns hardware-tailored backward config for TPU v4, v5p, v6e, and 7x."""
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  n, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, _, _, _ = common.get_tpu_info(x)

  if vmem_mib <= 16 or gen <= 4:
    return common.Config(block_m=128 if max_chd > 64 else 512)

  if max_chd > 64:
    if vmem_mib <= 32:
      return common.Config(block_m=256)
    return common.Config(block_m=512 if vmem_mib <= 64 else 1024)

  if vmem_mib <= 32:
    return common.Config(block_m=512)

  n_pad = pl.cdiv(n, 128) * 128
  if gen <= 6 and n_pad <= 384:
    return common.Config(block_m=512)
  return common.Config(block_m=1024)


def get_vjp_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns backward autotuning configs tailored to the current TPU."""
  x = ba.arguments["x"]
  w_out = ba.arguments["projection_out_weights"]
  _, _, c = x.shape
  h, d = w_out.shape
  max_chd = max(c, h, d)
  gen, vmem_mib, _, _, _ = common.get_tpu_info(x)
  seed = get_vjp_heuristics_config(ba)

  if vmem_mib <= 16 or gen <= 4:
    if max_chd <= 64:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
      )
    else:
      cands = (common.Config(block_m=128),)
  elif vmem_mib > 64:
    if max_chd <= 64:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
          common.Config(block_m=1024),
          common.Config(block_m=1024, block_i=64, block_j=512),
          common.Config(block_m=1024, block_i=16, block_j=1024),
          common.Config(block_m=1024, block_i=16, block_j=2048),
      )
    else:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
          common.Config(block_m=1024),
      )
  else:
    if max_chd <= 64:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
          common.Config(block_m=1024),
          common.Config(block_m=1024, block_i=16, block_j=512),
          common.Config(block_m=1024, block_i=16, block_j=1024),
      )
    else:
      cands = (
          common.Config(block_m=128),
          common.Config(block_m=256),
          common.Config(block_m=512),
      )
  return _dedup_autotuning_configs(seed, cands, x)


def _largest_aligned_divisor(n_pad: int, limit: int, align: int) -> int:
  """Returns largest multiple of `align` in `[align, min(limit, n_pad)]` dividing `n_pad`."""
  cand = (min(limit, n_pad) // align) * align
  while cand >= align:
    if n_pad % cand == 0:
      return cand
    cand -= align
  return align


def _default_blocks(
    n_pad: int, block_m: int, vmem_mib: int, num_cores: int = 1
) -> tuple[int, int]:
  """Hardware-tiered default `(block_i, block_j)` for `block_m`."""
  min_bi = 8 if n_pad < 16 else 16
  if block_m <= 128:
    return (8 if vmem_mib <= 16 or n_pad < 16 else 16), 128

  if block_m == 256:
    if vmem_mib <= 16:
      return (8, 256) if n_pad % 256 == 0 else (8, 128)
    if n_pad % 256 == 0:
      return min_bi, 256
    return _largest_aligned_divisor(n_pad, 32, min_bi), 128

  if block_m == 512:
    if vmem_mib <= 16:
      return min_bi, 128
    for cand_j in (640, 512, 384, 256):
      if n_pad % cand_j == 0:
        return min_bi, cand_j
    max_bi = max(min_bi, n_pad // num_cores) if num_cores > 1 else n_pad
    return _largest_aligned_divisor(n_pad, min(64, max_bi), min_bi), 128

  if vmem_mib <= 16:
    return min_bi, 128

  p_target = 16384
  top_cands = (
      (2048, 1024, 896, 768, 640, 512, 384, 256, 128)
      if vmem_mib > 64
      else (1024, 896, 768, 640, 512, 384, 256, 128)
  )
  for cand_j in top_cands:
    if n_pad % cand_j == 0:
      max_bi = min(64, max(min_bi, p_target // cand_j))
      if cand_j == n_pad and num_cores > 1 and n_pad >= min_bi * num_cores:
        max_bi = min(max_bi, n_pad // num_cores)
      return _largest_aligned_divisor(n_pad, max_bi, min_bi), cand_j
  return min_bi, 128


def resolve_blocks(
    n_pad: int,
    block_m: int,
    block_i: int | None = None,
    block_j: int | None = None,
    *,
    x: Any = None,
) -> tuple[int, int]:
  """Maps `block_m` and optional independent axis overrides to `(block_i, block_j)`."""
  _, vmem_mib, num_cores, _, _ = common.get_tpu_info(x)
  def_bi, def_bj = _default_blocks(n_pad, block_m, vmem_mib, num_cores)
  bi = (
      _largest_aligned_divisor(n_pad, block_i, 8)
      if block_i is not None
      else def_bi
  )
  bj = (
      _largest_aligned_divisor(n_pad, block_j, 128)
      if block_j is not None
      else def_bj
  )
  return bi, bj


_resolve_blocks = resolve_blocks


def resolve_config_blocks(
    n_pad: int,
    cfg: common.Config,
    *,
    x: Any = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
  """Resolves `(prologue_blocks, epilogue_blocks)` for `cfg` at `n_pad`."""
  n_pad = pl.cdiv(n_pad, 128) * 128
  pro = resolve_blocks(n_pad, cfg.block_m, cfg.block_i, cfg.block_j, x=x)
  epi_bm = cfg.epilogue_block_m or cfg.block_m
  epi_bi = (
      cfg.epilogue_block_i
      if cfg.epilogue_block_m is not None
      else (cfg.epilogue_block_i or cfg.block_i)
  )
  epi_bj = (
      cfg.epilogue_block_j
      if cfg.epilogue_block_m is not None
      else (cfg.epilogue_block_j or cfg.block_j)
  )
  epi = resolve_blocks(n_pad, epi_bm, epi_bi, epi_bj, x=x)
  return pro, epi


def _estimate_prologue_vmem(bi: int, bj: int, c: int, h: int) -> int:
  """Estimates live VMEM bytes for prologue kernel."""
  per_elem = max(8 * c + 8 * h + 4, 6 * c + 24 * h + 4)
  return bi * bj * per_elem + 8 * h * c + 64 * c + 512 * 1024


def _estimate_epilogue_vmem(bi: int, bj: int, c: int, h: int, d: int) -> int:
  """Estimates live VMEM bytes for epilogue kernel."""
  if c + h <= 128 and 2 * d <= 256:
    scratch = max(4 * c + 4 * h, 2 * c + 2 * h + 8 * d)
    w_bytes = 4 * d * (c + h)
  else:
    scratch = max(
        4 * c, 2 * c + 4 * h, 2 * c + 2 * h + 4 * d, 2 * h + 8 * d
    )
    w_bytes = 2 * d * (c + h)
  per_elem = 4 * c + 4 * h + 4 * d + scratch
  return bi * bj * per_elem + w_bytes + 64 * (c + h) + 512 * 1024


def _estimate_epilogue_bwd_vmem(
    bi: int, bj: int, c: int, h: int, d: int
) -> int:
  """Estimates live VMEM bytes for epilogue backward kernel."""
  if c + h <= 128 and 2 * d <= 256:
    scratch = max(
        4 * c,
        2 * c + 6 * h + 16 * d + 8,
        4 * c + 8 * h + 8 * d + 8,
        12 * h + 4,
    )
    w_bytes = 12 * d * (c + h)
  else:
    scratch = max(
        4 * c,
        2 * c + 6 * h + 12 * d + 8,
        6 * c + 6 * h + 4 * d + 8,
        2 * c + 10 * h + 4 * d + 8,
        12 * h + 4,
    )
    w_bytes = 6 * d * (c + h)
  per_elem = 8 * c + 8 * h + 4 * d + scratch
  inv_bytes = w_bytes + 64 * (c + h) + 1024 * h
  return bi * bj * per_elem + inv_bytes + 768 * 1024


def _estimate_prologue_bwd_vmem(
    bi: int, bj: int, c: int, h: int, *, alias_dx: bool = True
) -> int:
  """Estimates live VMEM bytes for prologue backward kernel."""
  if 4 * h > 256:
    per_elem = max(24 * c + 36 * h + 16, 32 * c + 12 * h + 12)
  else:
    per_elem = max(18 * c + 28 * h + 12, 28 * c + 8 * h + 8)
  if not alias_dx:
    per_elem += 6 * c
  inv_bytes = 24 * h * c + 64 * c + 1024 * c
  return bi * bj * per_elem + inv_bytes + 768 * 1024


def _clamp_blocks_to_vmem(
    n_pad: int,
    bi: int,
    bj: int,
    vmem_budget_bytes: int | None,
    estimate_fn: Callable[[int, int], int],
) -> tuple[int, int]:
  """Steps `(bi, bj)` down through valid divisors of `n_pad` until under VMEM budget."""
  if vmem_budget_bytes is None or estimate_fn(bi, bj) <= vmem_budget_bytes:
    return bi, bj
  bi_align = 16 if (bi % 16 == 0 and n_pad >= 16) else 8
  while bi > bi_align and estimate_fn(bi, bj) > vmem_budget_bytes:
    bi = _largest_aligned_divisor(n_pad, bi - bi_align, bi_align)
  while bj > 128 and estimate_fn(bi, bj) > vmem_budget_bytes:
    bj = _largest_aligned_divisor(n_pad, bj - 128, 128)
  while (
      bi > 8 and n_pad % 8 == 0 and estimate_fn(bi, bj) > vmem_budget_bytes
  ):
    bi = _largest_aligned_divisor(n_pad, bi - 8, 8)
  return bi, bj


def _scoped_vmem_limit(
    gen: int,
    vmem_mib: int,
    est_vmem_bytes: int,
    vmem_limit_bytes: int | None,
    *,
    bi: int = 64,
    bj: int = 1024,
    max_dim: int = 128,
) -> int | None:
  """Leaves VMEM unreserved for XLA MSA when tiles fit inside default VMEM."""
  if (
      gen >= 6
      and vmem_mib > 16
      and bj <= 512
      and max_dim <= 64
      and bi * bj <= 16384
  ):
    if est_vmem_bytes <= 30 * 1024 * 1024:
      return None
    if est_vmem_bytes <= 38 * 1024 * 1024:
      return 44 * 1024 * 1024
  return vmem_limit_bytes


def _prologue_cost_estimate(n_pad: int, c: int, h: int) -> pl.CostEstimate:
  m = n_pad * n_pad
  return pl.CostEstimate(
      flops=m * (8 * c * h + 7 * c + 4 * h),
      transcendentals=m * (2 * h + 1),
      bytes_accessed=2 * m * (c + 2 * h + 1) + 8 * c * h + 64 * c,
  )


def _epilogue_cost_estimate(
    n_pad: int, c: int, h: int, d: int
) -> pl.CostEstimate:
  m = n_pad * n_pad
  use_fused_gp = c + h <= 128 and 2 * d <= 256
  w_elems = 2 * d * (c + h) if use_fused_gp else d * (c + h)
  return pl.CostEstimate(
      flops=m * (2 * w_elems + 7 * (c + h) + 2 * d),
      transcendentals=m * (d + 2),
      bytes_accessed=2 * m * (c + h + d) + 2 * w_elems + 64 * (c + h),
  )


def _build_bwd_grid(
    num_i: int, num_j: int, num_cores: int
) -> tuple[tuple[int, int, int], int, bool]:
  """Builds a division-free 3D backward grid `(cores, steps_i, steps_j)`."""
  if num_cores > 1 and num_i % num_cores == 0:
    return (num_cores, num_i // num_cores, num_j), num_cores, True
  if num_cores > 1 and num_j % num_cores == 0:
    return (num_cores, num_i, num_j // num_cores), num_cores, False
  return (1, num_i, num_j), 1, True


def _bcast_norm_2d(v: jax.Array) -> jax.Array:
  """Broadcasts 1D `(K,)` norm param to 2D `(16, K)` for vmap & TPU 7x DMA."""
  return lax.broadcast_in_dim(v, (16, v.shape[0]), (1,))


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
  gen, vmem_mib, num_cores, vmem_limit_bytes, vmem_budget_bytes = (
      common.get_tpu_info(x_pad)
  )
  block_i, block_j = resolve_blocks(
      n_pad, block_m, block_i, block_j, x=x_pad
  )
  est_fn = lambda bi, bj: _estimate_prologue_vmem(bi, bj, c, h)
  block_i, block_j = _clamp_blocks_to_vmem(
      n_pad, block_i, block_j, vmem_budget_bytes, est_fn
  )
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  par_sem = "arbitrary" if (num_cores > 1 and n_pad <= 256) else "parallel"
  vmem_limit = _scoped_vmem_limit(
      gen,
      vmem_mib,
      est_fn(block_i, block_j),
      vmem_limit_bytes,
      bi=block_i,
      bj=block_j,
      max_dim=max(c, h),
  )

  use_4d_t8 = gen >= 7
  use_4d_hij = gen >= 7 and (block_i % 16 != 0)
  if use_4d_t8:
    x_in = x_pad.reshape(n_pad, c // 8, 8, n_pad)
    x_spec = pl.BlockSpec(
        (block_i, c // 8, 8, block_j), lambda i, j: (i, 0, 0, j)
    )
  else:
    x_in = x_pad
    x_spec = pl.BlockSpec((block_i, c, block_j), lambda i, j: (i, 0, j))

  if use_4d_hij:
    mask_in = mask_pad.reshape(n_pad // 8, 8, n_pad)
    mask_spec = pl.BlockSpec(
        (block_i // 8, 8, block_j), lambda i, j: (i, 0, j)
    )
    out_shape = jax.ShapeDtypeStruct((h, n_pad // 8, 8, n_pad), dtype)
    out_spec = pl.BlockSpec(
        (h, block_i // 8, 8, block_j), lambda i, j: (0, i, 0, j)
    )
  else:
    mask_in = mask_pad
    mask_spec = pl.BlockSpec((block_i, block_j), lambda i, j: (i, j))
    out_shape = jax.ShapeDtypeStruct((h, n_pad, n_pad), dtype)
    out_spec = pl.BlockSpec((h, block_i, block_j), lambda i, j: (0, i, j))

  w_proj_t = projection_in_weights.reshape(c, 2 * h).T
  w_gate_t = gate_in_weights.reshape(c, 2 * h).T
  ln_scale_in = _bcast_norm_2d(layernorm_in_scale)
  ln_offset_in = _bcast_norm_2d(layernorm_in_offset)
  ln_c_spec = pl.BlockSpec((16, c), lambda i, j: (0, 0))

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
      mask_2d = (
          mask_ref[...].reshape(block_i, block_j)
          if use_4d_hij
          else mask_ref[...]
      )
      act = common.layer_norm_3d_axis1(
          x_raw, ln_scale_ref[0], ln_offset_ref[0], eps
      )
      pg_2d = common.matmul_2d(w_pg_ref[...], common.pack_icj_to_2d(act))
      lr_2d = (pg_2d[: 2 * h] * jax.nn.sigmoid(pg_2d[2 * h :])).astype(dtype)
      lr_hij = jnp.transpose(
          common.unpack_2d_to_icj(lr_2d, block_i, block_j), (1, 0, 2)
      )
      lr_hij = jnp.where(
          lax.broadcast_in_dim(mask_2d == 0, lr_hij.shape, (1, 2)),
          0,
          lr_hij,
      )
      left_val = lr_hij[:h]
      right_val = lr_hij[h:]
      left_ref[...] = (
          left_val.reshape(h, block_i // 8, 8, block_j)
          if use_4d_hij
          else left_val
      )
      right_ref[...] = (
          right_val.reshape(h, block_i // 8, 8, block_j)
          if use_4d_hij
          else right_val
      )

    left_out, right_out = pl.pallas_call(
        kernel_4h,
        out_shape=(out_shape, out_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_i, num_j),
            in_specs=[
                x_spec,
                mask_spec,
                pl.BlockSpec(w_pg_arg.shape, lambda i, j: (0, 0)),
                ln_c_spec,
                ln_c_spec,
            ],
            out_specs=(out_spec, out_spec),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(par_sem, par_sem),
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        cost_estimate=(
            None
            if (gen == 5 and num_cores > 1)
            else _prologue_cost_estimate(n_pad, c, h)
        ),
    )(
        x_in,
        mask_in,
        w_pg_arg,
        ln_scale_in,
        ln_offset_in,
    )
    if use_4d_hij:
      return (
          left_out.reshape(h, n_pad, n_pad),
          right_out.reshape(h, n_pad, n_pad),
      )
    return left_out, right_out

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
    mask_2d = (
        mask_ref[...].reshape(block_i, block_j) if use_4d_hij else mask_ref[...]
    )
    act = common.layer_norm_3d_axis1(
        x_raw, ln_scale_ref[0], ln_offset_ref[0], eps
    )
    act_2d = common.pack_icj_to_2d(act)
    p_2d = common.matmul_2d(w_proj_ref[...], act_2d)
    g_2d = common.matmul_2d(w_gate_ref[...], act_2d)
    lr_2d = (p_2d * jax.nn.sigmoid(g_2d)).astype(dtype)
    lr_hij = jnp.transpose(
        common.unpack_2d_to_icj(lr_2d, block_i, block_j), (1, 0, 2)
    )
    lr_hij = jnp.where(
        lax.broadcast_in_dim(mask_2d == 0, lr_hij.shape, (1, 2)),
        0,
        lr_hij,
    )
    left_val = lr_hij[:h]
    right_val = lr_hij[h:]
    left_ref[...] = (
        left_val.reshape(h, block_i // 8, 8, block_j)
        if use_4d_hij
        else left_val
    )
    right_ref[...] = (
        right_val.reshape(h, block_i // 8, 8, block_j)
        if use_4d_hij
        else right_val
    )

  left_out, right_out = pl.pallas_call(
      kernel_2h,
      out_shape=(out_shape, out_shape),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_i, num_j),
          in_specs=[
              x_spec,
              mask_spec,
              pl.BlockSpec(w_proj_t.shape, lambda i, j: (0, 0)),
              pl.BlockSpec(w_gate_t.shape, lambda i, j: (0, 0)),
              ln_c_spec,
              ln_c_spec,
          ],
          out_specs=(out_spec, out_spec),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(par_sem, par_sem),
          vmem_limit_bytes=vmem_limit,
          disable_bounds_checks=True,
      ),
      cost_estimate=(
          None
          if (gen == 5 and num_cores > 1)
          else _prologue_cost_estimate(n_pad, c, h)
      ),
  )(
      x_in,
      mask_in,
      w_proj_t,
      w_gate_t,
      ln_scale_in,
      ln_offset_in,
  )
  if use_4d_hij:
    return (
        left_out.reshape(h, n_pad, n_pad),
        right_out.reshape(h, n_pad, n_pad),
    )
  return left_out, right_out


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
  gen, vmem_mib, num_cores, vmem_limit_bytes, vmem_budget_bytes = (
      common.get_tpu_info(x_pad)
  )
  block_i, block_j = resolve_blocks(
      n_pad, block_m, block_i, block_j, x=x_pad
  )
  est_fn = lambda bi, bj: _estimate_epilogue_vmem(bi, bj, c, h, d)
  block_i, block_j = _clamp_blocks_to_vmem(
      n_pad, block_i, block_j, vmem_budget_bytes, est_fn
  )
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  par_sem = "arbitrary" if (num_cores > 1 and n_pad <= 256) else "parallel"
  vmem_limit = _scoped_vmem_limit(
      gen,
      vmem_mib,
      est_fn(block_i, block_j),
      vmem_limit_bytes,
      bi=block_i,
      bj=block_j,
      max_dim=max(c, h, d),
  )

  use_4d_t8 = gen >= 7
  use_4d_hij = gen >= 7 and (block_i % 16 != 0)
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

  if use_4d_hij:
    t3_in = t3_pad.reshape(h, n_pad // 8, 8, n_pad)
    t3_spec = pl.BlockSpec(
        (h, block_i // 8, 8, block_j), lambda i, j: (0, i, 0, j)
    )
  else:
    t3_in = t3_pad
    t3_spec = pl.BlockSpec((h, block_i, block_j), lambda i, j: (0, i, j))

  ln_in_scale_in = _bcast_norm_2d(layernorm_in_scale)
  ln_in_offset_in = _bcast_norm_2d(layernorm_in_offset)
  ln_out_scale_in = _bcast_norm_2d(layernorm_out_scale)
  ln_out_offset_in = _bcast_norm_2d(layernorm_out_offset)
  epi_cost = (
      None
      if (gen == 5 and num_cores > 1)
      else _epilogue_cost_estimate(n_pad, c, h, d)
  )

  use_fused_gp = c + h <= 128 and 2 * d <= 256
  if use_fused_gp:
    w_gp_out_t = _build_block_diag_out_weights_t(
        gate_out_weights, projection_out_weights
    )

    def kernel_gp(
        t3_ref,
        x_ref,
        w_gp_ref,
        ln_in_scale_ref,
        ln_in_offset_ref,
        ln_out_scale_ref,
        ln_out_offset_ref,
        out_ref,
    ):
      x_raw = (
          x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
      )
      t3_raw = (
          t3_ref[...].reshape(h, block_i, block_j)
          if use_4d_hij
          else t3_ref[...]
      )
      left_act = common.layer_norm_3d_axis1(
          x_raw, ln_in_scale_ref[0], ln_in_offset_ref[0], eps
      )
      t_hat_3d, _ = common.layer_norm_3d_axis0_hat_rstd(t3_raw, eps)
      act_out = jnp.transpose(
          common.apply_norm_affine_axis0(
              t_hat_3d, ln_out_scale_ref[0], ln_out_offset_ref[0]
          ).astype(dtype),
          (1, 0, 2),
      )
      la_ao_2d = jnp.concatenate(
          [common.pack_icj_to_2d(left_act), common.pack_icj_to_2d(act_out)],
          axis=0,
      )
      gp_2d = common.matmul_2d(w_gp_ref[...], la_ao_2d)
      out_2d = (gp_2d[d:, :] * jax.nn.sigmoid(gp_2d[:d, :])).astype(dtype)
      out_idj = common.unpack_2d_to_icj(out_2d, block_i, block_j)
      out_ref[...] = (
          out_idj.reshape(block_i, d // 8, 8, block_j)
          if use_4d_t8
          else out_idj
      )

    out = pl.pallas_call(
        kernel_gp,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_i, num_j),
            in_specs=[
                t3_spec,
                x_spec,
                pl.BlockSpec((2 * d, c + h), lambda i, j: (0, 0)),
                pl.BlockSpec((16, c), lambda i, j: (0, 0)),
                pl.BlockSpec((16, c), lambda i, j: (0, 0)),
                pl.BlockSpec((16, h), lambda i, j: (0, 0)),
                pl.BlockSpec((16, h), lambda i, j: (0, 0)),
            ],
            out_specs=out_spec,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(par_sem, par_sem),
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        cost_estimate=epi_cost,
    )(
        t3_in,
        x_in,
        w_gp_out_t,
        ln_in_scale_in,
        ln_in_offset_in,
        ln_out_scale_in,
        ln_out_offset_in,
    )
    return out.reshape(n_pad, d, n_pad) if use_4d_t8 else out

  w_gate_t = gate_out_weights.T
  w_proj_t = projection_out_weights.T

  def kernel_3d(
      t3_ref,
      x_ref,
      w_gate_ref,
      w_proj_ref,
      ln_in_scale_ref,
      ln_in_offset_ref,
      ln_out_scale_ref,
      ln_out_offset_ref,
      out_ref,
  ):
    x_raw = x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    t3_raw = (
        t3_ref[...].reshape(h, block_i, block_j) if use_4d_hij else t3_ref[...]
    )
    left_act = common.layer_norm_3d_axis1(
        x_raw, ln_in_scale_ref[0], ln_in_offset_ref[0], eps
    )
    t_hat_3d, _ = common.layer_norm_3d_axis0_hat_rstd(t3_raw, eps)
    act_out = jnp.transpose(
        common.apply_norm_affine_axis0(
            t_hat_3d, ln_out_scale_ref[0], ln_out_offset_ref[0]
        ).astype(dtype),
        (1, 0, 2),
    )
    g_2d = common.matmul_2d(w_gate_ref[...], common.pack_icj_to_2d(left_act))
    p_2d = common.matmul_2d(w_proj_ref[...], common.pack_icj_to_2d(act_out))
    out_2d = (p_2d * jax.nn.sigmoid(g_2d)).astype(dtype)
    out_idj = common.unpack_2d_to_icj(out_2d, block_i, block_j)
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
              t3_spec,
              x_spec,
              pl.BlockSpec((d, c), lambda i, j: (0, 0)),
              pl.BlockSpec((d, h), lambda i, j: (0, 0)),
              pl.BlockSpec((16, c), lambda i, j: (0, 0)),
              pl.BlockSpec((16, c), lambda i, j: (0, 0)),
              pl.BlockSpec((16, h), lambda i, j: (0, 0)),
              pl.BlockSpec((16, h), lambda i, j: (0, 0)),
          ],
          out_specs=out_spec,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(par_sem, par_sem),
          vmem_limit_bytes=vmem_limit,
          disable_bounds_checks=True,
      ),
      cost_estimate=epi_cost,
  )(
      t3_in,
      x_in,
      w_gate_t,
      w_proj_t,
      ln_in_scale_in,
      ln_in_offset_in,
      ln_out_scale_in,
      ln_out_offset_in,
  )
  return out.reshape(n_pad, d, n_pad) if use_4d_t8 else out


def _build_block_diag_out_weights_t(
    gate_out_weights: jax.Array,
    projection_out_weights: jax.Array,
) -> jax.Array:
  """Builds `(2 * D, C + H)` block-diagonal `[[W_gate^T, 0], [0, W_proj^T]]`."""
  c, d = gate_out_weights.shape
  h, _ = projection_out_weights.shape
  dtype = gate_out_weights.dtype
  top = jnp.concatenate(
      [gate_out_weights.T, jnp.zeros((d, h), dtype=dtype)], axis=1
  )
  bot = jnp.concatenate(
      [jnp.zeros((d, c), dtype=dtype), projection_out_weights.T], axis=1
  )
  return jnp.concatenate([top, bot], axis=0)


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
    alias_x_pad: bool = False,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
  """Fused Pallas Mosaic TPU epilogue backward kernel."""
  n_pad, c, _ = x_pad.shape
  h, d = projection_out_weights.shape
  dtype = x_pad.dtype
  gen, vmem_mib, num_cores, vmem_limit_bytes, vmem_budget_bytes = (
      common.get_tpu_info(x_pad)
  )
  block_i, block_j = resolve_blocks(
      n_pad, block_m, block_i, block_j, x=x_pad
  )
  est_fn = lambda bi, bj: _estimate_epilogue_bwd_vmem(bi, bj, c, h, d)
  block_i, block_j = _clamp_blocks_to_vmem(
      n_pad, block_i, block_j, vmem_budget_bytes, est_fn
  )
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  vmem_limit = _scoped_vmem_limit(
      gen,
      vmem_mib,
      est_fn(block_i, block_j),
      vmem_limit_bytes,
      bi=block_i,
      bj=block_j,
      max_dim=max(c, h, d),
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
  use_4d_hij = gen >= 7 and (block_i % 16 != 0)
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

  if use_4d_hij:
    t3_in = t3_pad.reshape(h, n_pad // 8, 8, n_pad)
    dt3_shape = jax.ShapeDtypeStruct((h, n_pad // 8, 8, n_pad), dtype)
    hij_spec = pl.BlockSpec(
        (h, block_i // 8, 8, block_j),
        lambda c_idx, si, sj: (
            0,
            idx_i(c_idx, si, sj),
            0,
            idx_j(c_idx, si, sj),
        ),
    )
  else:
    t3_in = t3_pad
    dt3_shape = jax.ShapeDtypeStruct((h, n_pad, n_pad), dtype)
    hij_spec = pl.BlockSpec(
        (h, block_i, block_j),
        lambda c_idx, si, sj: (0, idx_i(c_idx, si, sj), idx_j(c_idx, si, sj)),
    )

  allow_alias = not (gen == 5 and num_cores_used > 1)
  epi_aliases = {2: 1} if (alias_x_pad and allow_alias) else {}
  ln_in_scale_in = _bcast_norm_2d(layernorm_in_scale)
  ln_in_offset_in = _bcast_norm_2d(layernorm_in_offset)
  ln_out_scale_in = _bcast_norm_2d(layernorm_out_scale)
  ln_out_offset_in = _bcast_norm_2d(layernorm_out_offset)

  use_2d_bwd = (
      gen == 6
      and num_cores == 1
      and block_j >= 1024
      and block_i * block_j <= 16384
  )
  if c + h <= 128 and 2 * d <= 256:
    w_gp_out_t = _build_block_diag_out_weights_t(
        gate_out_weights, projection_out_weights
    )
    dw_gp_shape = w_gp_out_t.shape

    def kernel_bwd_gp(
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

      x_raw = (
          x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
      )
      t3_raw = (
          t3_ref[...].reshape(h, block_i, block_j)
          if use_4d_hij
          else t3_ref[...]
      )
      t3_ihj = jnp.transpose(t3_raw, (1, 0, 2))
      w_gp_val = w_gp_ref[...]
      ln_out_scale = ln_out_scale_ref[0]

      zero_dt = jnp.zeros((), dtype)
      left_act = common.layer_norm_3d_axis1(
          x_raw, ln_in_scale_ref[0], ln_in_offset_ref[0], eps
      )
      t_hat_ihj, rstd_ihj = common.layer_norm_3d_axis1_hat_rstd(t3_ihj, eps)
      act_out = common.apply_norm_affine_axis1(
          t_hat_ihj, ln_out_scale, ln_out_offset_ref[0]
      ).astype(dtype)
      dout_raw = (
          dout_ref[...].reshape(block_i, d, block_j)
          if use_4d_t8
          else dout_ref[...]
      )

      if use_2d_bwd:
        la_ao_2d = jnp.concatenate(
            [common.pack_icj_to_2d(left_act), common.pack_icj_to_2d(act_out)],
            axis=0,
        )
        gp_2d = common.matmul_2d(w_gp_val, la_ao_2d)
        gate_2d = jax.nn.sigmoid(gp_2d[:d, :])
        p_2d = gp_2d[d:, :]
        dout_2d = common.pack_icj_to_2d(dout_raw)
        dout_zero_2d = dout_2d == zero_dt
        dout_gate_2d = dout_2d.astype(jnp.float32) * gate_2d
        dp_2d = jnp.where(dout_zero_2d, zero_dt, dout_gate_2d.astype(dtype))
        dg_2d = jnp.where(
            dout_zero_2d,
            zero_dt,
            (dout_gate_2d * p_2d * (1.0 - gate_2d)).astype(dtype),
        )
        dg_dp_2d = jnp.concatenate([dg_2d, dp_2d], axis=0)
        safe_la_ao_2d = jnp.where(
            la_ao_2d * zero_dt == zero_dt, la_ao_2d, zero_dt
        )
        dw_gp_ref[...] = dw_gp_ref[...] + common.matmul_dw_2d(
            dg_dp_2d, safe_la_ao_2d
        ).reshape(1, *dw_gp_shape)
        dla_dao_2d = common.matmul_wt_2d(w_gp_val, dg_dp_2d)
        dla_icj = common.unpack_2d_to_icj(
            dla_dao_2d[:c, :].astype(dtype), block_i, block_j
        )
        dao_icj = common.unpack_2d_to_icj(dla_dao_2d[c:, :], block_i, block_j)
      else:
        la_ao = jnp.concatenate([left_act, act_out], axis=1)
        gp = common.matmul_3d(w_gp_val, la_ao)
        gate = jax.nn.sigmoid(gp[:, :d, :])
        p = gp[:, d:, :]
        dout_zero = dout_raw == zero_dt
        dout_gate = dout_raw.astype(jnp.float32) * gate
        dp = jnp.where(dout_zero, zero_dt, dout_gate.astype(dtype))
        dg = jnp.where(
            dout_zero, zero_dt, (dout_gate * p * (1.0 - gate)).astype(dtype)
        )
        dg_dp = jnp.concatenate([dg, dp], axis=1)
        safe_la_ao = jnp.where(la_ao * zero_dt == zero_dt, la_ao, zero_dt)
        dw_gp_ref[...] = dw_gp_ref[...] + common.matmul_dw_3d(
            dg_dp, safe_la_ao
        ).reshape(1, *dw_gp_shape)
        dla_dao = common.matmul_wt_3d(w_gp_val, dg_dp)
        dla_icj = dla_dao[:, :c, :].astype(dtype)
        dao_icj = dla_dao[:, c:, :]

      dleft_act_epi_ref[...] = (
          dla_icj.reshape(block_i, c // 8, 8, block_j)
          if use_4d_t8
          else dla_icj
      )
      dt3_ihj, dln_out_tile = common.layer_norm_3d_axis1_bwd(
          dao_icj,
          t_hat_ihj,
          rstd_ihj,
          ln_out_scale,
          dtype,
          from_hat_rstd=True,
      )
      dt3_val = jnp.transpose(dt3_ihj, (1, 0, 2))
      dt3_ref[...] = (
          dt3_val.reshape(h, block_i // 8, 8, block_j)
          if use_4d_hij
          else dt3_val
      )
      dln_out_ref[...] = dln_out_ref[...] + dln_out_tile

    (
        dt3_out,
        dleft_act_epi_out,
        dw_gp_part,
        dln_out_part,
    ) = pl.pallas_call(
        kernel_bwd_gp,
        out_shape=(
            dt3_shape,
            dleft_epi_shape,
            jax.ShapeDtypeStruct((num_cores_used, *dw_gp_shape), jnp.float32),
            jax.ShapeDtypeStruct((num_cores_used, 2 * h, 128), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                dout_spec,
                hij_spec,
                x_spec,
                pl.BlockSpec(dw_gp_shape, lambda c_idx, si, sj: (0, 0)),
                pl.BlockSpec((16, c), lambda c_idx, si, sj: (0, 0)),
                pl.BlockSpec((16, c), lambda c_idx, si, sj: (0, 0)),
                pl.BlockSpec((16, h), lambda c_idx, si, sj: (0, 0)),
                pl.BlockSpec((16, h), lambda c_idx, si, sj: (0, 0)),
            ],
            out_specs=(
                hij_spec,
                dleft_epi_spec,
                pl.BlockSpec(
                    (1, *dw_gp_shape), lambda c_idx, si, sj: (c_idx, 0, 0)
                ),
                pl.BlockSpec(
                    (1, 2 * h, 128), lambda c_idx, si, sj: (c_idx, 0, 0)
                ),
            ),
        ),
        input_output_aliases=epi_aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
    )(
        dout_in,
        t3_in,
        x_in,
        w_gp_out_t,
        ln_in_scale_in,
        ln_in_offset_in,
        ln_out_scale_in,
        ln_out_offset_in,
    )

    dt3_pad = dt3_out.reshape(h, n_pad, n_pad) if use_4d_hij else dt3_out
    dleft_act_epi_pad = (
        dleft_act_epi_out.reshape(n_pad, c, n_pad)
        if use_4d_t8
        else dleft_act_epi_out
    )
    dw_gp_sum = (
        dw_gp_part[0] if num_cores_used == 1 else jnp.sum(dw_gp_part, axis=0)
    )
    dw_gate_out = dw_gp_sum[:d, :c].T.astype(dtype)
    dw_proj_out = dw_gp_sum[d:, c:].T.astype(dtype)
    dln_out = (
        jnp.sum(dln_out_part[0], axis=1)
        if num_cores_used == 1
        else jnp.sum(dln_out_part, axis=(0, 2))
    ).reshape(2, h).astype(dtype)
    return (
        dt3_pad,
        dleft_act_epi_pad,
        dw_proj_out,
        dw_gate_out,
        dln_out[0],
        dln_out[1],
    )

  w_gate_t = gate_out_weights.T
  w_proj_t = projection_out_weights.T

  def kernel_bwd_3d(
      dout_ref,
      t3_ref,
      x_ref,
      w_gate_ref,
      w_proj_ref,
      ln_in_scale_ref,
      ln_in_offset_ref,
      ln_out_scale_ref,
      ln_out_offset_ref,
      dt3_ref,
      dleft_act_epi_ref,
      dw_gate_ref,
      dw_proj_ref,
      dln_out_ref,
  ):
    is_first = (
        pl.program_id(1) == 0
        if steps_j == 1
        else (pl.program_id(1) == 0) & (pl.program_id(2) == 0)
    )

    @pl.when(is_first)
    def _init_accs():
      dw_gate_ref[...] = jnp.zeros_like(dw_gate_ref)
      dw_proj_ref[...] = jnp.zeros_like(dw_proj_ref)
      dln_out_ref[...] = jnp.zeros_like(dln_out_ref)

    x_raw = (
        x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    )
    t3_raw = (
        t3_ref[...].reshape(h, block_i, block_j) if use_4d_hij else t3_ref[...]
    )
    t3_ihj = jnp.transpose(t3_raw, (1, 0, 2))
    w_gate_val = w_gate_ref[...]
    w_proj_val = w_proj_ref[...]
    ln_out_scale = ln_out_scale_ref[0]

    zero_dt = jnp.zeros((), dtype)
    left_act = common.layer_norm_3d_axis1(
        x_raw, ln_in_scale_ref[0], ln_in_offset_ref[0], eps
    )
    t_hat_ihj, rstd_ihj = common.layer_norm_3d_axis1_hat_rstd(t3_ihj, eps)
    act_out = common.apply_norm_affine_axis1(
        t_hat_ihj, ln_out_scale, ln_out_offset_ref[0]
    ).astype(dtype)

    gate = jax.nn.sigmoid(common.matmul_3d(w_gate_val, left_act))
    p = common.matmul_3d(w_proj_val, act_out)
    dout_raw = (
        dout_ref[...].reshape(block_i, d, block_j)
        if use_4d_t8
        else dout_ref[...]
    )
    dout_zero = dout_raw == zero_dt
    dout_gate = dout_raw.astype(jnp.float32) * gate
    dp = jnp.where(dout_zero, zero_dt, dout_gate.astype(dtype))
    dg = jnp.where(
        dout_zero, zero_dt, (dout_gate * p * (1.0 - gate)).astype(dtype)
    )

    safe_left_act = jnp.where(
        left_act * zero_dt == zero_dt, left_act, zero_dt
    )
    safe_act_out = jnp.where(act_out * zero_dt == zero_dt, act_out, zero_dt)
    dw_gate_ref[...] = dw_gate_ref[...] + common.matmul_dw_3d(
        dg, safe_left_act
    ).reshape(1, d, c)
    dw_proj_ref[...] = dw_proj_ref[...] + common.matmul_dw_3d(
        dp, safe_act_out
    ).reshape(1, d, h)

    dla_icj = common.matmul_wt_3d(w_gate_val, dg).astype(dtype)
    dao_icj = common.matmul_wt_3d(w_proj_val, dp)
    dleft_act_epi_ref[...] = (
        dla_icj.reshape(block_i, c // 8, 8, block_j) if use_4d_t8 else dla_icj
    )
    dt3_ihj, dln_out_tile = common.layer_norm_3d_axis1_bwd(
        dao_icj,
        t_hat_ihj,
        rstd_ihj,
        ln_out_scale,
        dtype,
        from_hat_rstd=True,
    )
    dt3_val = jnp.transpose(dt3_ihj, (1, 0, 2))
    dt3_ref[...] = (
        dt3_val.reshape(h, block_i // 8, 8, block_j) if use_4d_hij else dt3_val
    )
    dln_out_ref[...] = dln_out_ref[...] + dln_out_tile

  (
      dt3_out,
      dleft_act_epi_out,
      dw_gate_part,
      dw_proj_part,
      dln_out_part,
  ) = pl.pallas_call(
      kernel_bwd_3d,
      out_shape=(
          dt3_shape,
          dleft_epi_shape,
          jax.ShapeDtypeStruct((num_cores_used, d, c), jnp.float32),
          jax.ShapeDtypeStruct((num_cores_used, d, h), jnp.float32),
          jax.ShapeDtypeStruct((num_cores_used, 2 * h, 128), jnp.float32),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              dout_spec,
              hij_spec,
              x_spec,
              pl.BlockSpec((d, c), lambda c_idx, si, sj: (0, 0)),
              pl.BlockSpec((d, h), lambda c_idx, si, sj: (0, 0)),
              pl.BlockSpec((16, c), lambda c_idx, si, sj: (0, 0)),
              pl.BlockSpec((16, c), lambda c_idx, si, sj: (0, 0)),
              pl.BlockSpec((16, h), lambda c_idx, si, sj: (0, 0)),
              pl.BlockSpec((16, h), lambda c_idx, si, sj: (0, 0)),
          ],
          out_specs=(
              hij_spec,
              dleft_epi_spec,
              pl.BlockSpec((1, d, c), lambda c_idx, si, sj: (c_idx, 0, 0)),
              pl.BlockSpec((1, d, h), lambda c_idx, si, sj: (c_idx, 0, 0)),
              pl.BlockSpec(
                  (1, 2 * h, 128), lambda c_idx, si, sj: (c_idx, 0, 0)
              ),
          ),
      ),
      input_output_aliases=epi_aliases,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=vmem_limit,
          disable_bounds_checks=True,
      ),
  )(
      dout_in,
      t3_in,
      x_in,
      w_gate_t,
      w_proj_t,
      ln_in_scale_in,
      ln_in_offset_in,
      ln_out_scale_in,
      ln_out_offset_in,
  )

  dt3_pad = dt3_out.reshape(h, n_pad, n_pad) if use_4d_hij else dt3_out
  dleft_act_epi_pad = (
      dleft_act_epi_out.reshape(n_pad, c, n_pad)
      if use_4d_t8
      else dleft_act_epi_out
  )
  dw_gate_sum = (
      dw_gate_part[0] if num_cores_used == 1 else jnp.sum(dw_gate_part, axis=0)
  )
  dw_proj_sum = (
      dw_proj_part[0] if num_cores_used == 1 else jnp.sum(dw_proj_part, axis=0)
  )
  dw_gate_out = dw_gate_sum.T.astype(dtype)
  dw_proj_out = dw_proj_sum.T.astype(dtype)
  dln_out = (
      jnp.sum(dln_out_part[0], axis=1)
      if num_cores_used == 1
      else jnp.sum(dln_out_part, axis=(0, 2))
  ).reshape(2, h).astype(dtype)

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
  gen, vmem_mib, num_cores, vmem_limit_bytes, vmem_budget_bytes = (
      common.get_tpu_info(x_pad)
  )
  block_i, block_j = resolve_blocks(
      n_pad, block_m, block_i, block_j, x=x_pad
  )
  allow_alias = not (gen == 5 and num_cores > 1)
  est_fn = lambda bi, bj: _estimate_prologue_bwd_vmem(
      bi, bj, c, h, alias_dx=allow_alias
  )
  block_i, block_j = _clamp_blocks_to_vmem(
      n_pad, block_i, block_j, vmem_budget_bytes, est_fn
  )
  num_i = n_pad // block_i
  num_j = n_pad // block_j
  vmem_limit = _scoped_vmem_limit(
      gen,
      vmem_mib,
      est_fn(block_i, block_j),
      vmem_limit_bytes,
      bi=block_i,
      bj=block_j,
      max_dim=max(c, h),
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
  ln_scale_in = _bcast_norm_2d(layernorm_in_scale)
  ln_offset_in = _bcast_norm_2d(layernorm_in_offset)
  ln_c_spec = pl.BlockSpec((16, c), lambda c_idx, si, sj: (0, 0))

  use_4d_t8 = gen >= 7
  use_4d_hij = gen >= 7 and (block_i % 16 != 0)
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

  if use_4d_hij:
    dleft_in = dleft_pad.reshape(h, n_pad // 8, 8, n_pad)
    dright_in = dright_pad.reshape(h, n_pad // 8, 8, n_pad)
    mask_in = mask_pad.reshape(n_pad // 8, 8, n_pad)
    hij_spec = pl.BlockSpec(
        (h, block_i // 8, 8, block_j),
        lambda c_idx, si, sj: (
            0,
            idx_i(c_idx, si, sj),
            0,
            idx_j(c_idx, si, sj),
        ),
    )
    mask_spec = pl.BlockSpec(
        (block_i // 8, 8, block_j),
        lambda c_idx, si, sj: (idx_i(c_idx, si, sj), 0, idx_j(c_idx, si, sj)),
    )
  else:
    dleft_in = dleft_pad
    dright_in = dright_pad
    mask_in = mask_pad
    hij_spec = pl.BlockSpec(
        (h, block_i, block_j),
        lambda c_idx, si, sj: (0, idx_i(c_idx, si, sj), idx_j(c_idx, si, sj)),
    )
    mask_spec = pl.BlockSpec(
        (block_i, block_j),
        lambda c_idx, si, sj: (idx_i(c_idx, si, sj), idx_j(c_idx, si, sj)),
    )

  pro_aliases = {2: 0} if not (gen == 5 and num_cores_used > 1) else {}
  use_2d_bwd = (
      gen == 6
      and num_cores == 1
      and block_j >= 1024
      and block_i * block_j <= 16384
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
      ln_in_scale = ln_in_scale_ref[0]
      x_raw = (
          x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
      )

      zero_dt = jnp.zeros((), dtype)
      mask_2d = (
          mask_ref[...].reshape(block_i, block_j)
          if use_4d_hij
          else mask_ref[...]
      )
      mask_zero = mask_2d == 0
      x_hat, rstd = common.layer_norm_3d_axis1_hat_rstd(x_raw, eps)
      act_bf16 = common.apply_norm_affine_axis1(
          x_hat, ln_in_scale, ln_in_offset_ref[0]
      ).astype(dtype)
      act_pro = jnp.where(
          lax.broadcast_in_dim(mask_zero, act_bf16.shape, (0, 2)),
          zero_dt,
          act_bf16,
      )
      dleft_3d = (
          dleft_ref[...].reshape(h, block_i, block_j)
          if use_4d_hij
          else dleft_ref[...]
      )
      dright_3d = (
          dright_ref[...].reshape(h, block_i, block_j)
          if use_4d_hij
          else dright_ref[...]
      )
      dlr = jnp.transpose(
          jnp.concatenate([dleft_3d, dright_3d], axis=0), (1, 0, 2)
      )
      dlr = jnp.where(
          lax.broadcast_in_dim(mask_zero, dlr.shape, (0, 2)), zero_dt, dlr
      )

      if use_2d_bwd:
        act_pro_2d = common.pack_icj_to_2d(act_pro)
        pg_2d = common.matmul_2d(w_pg_val, act_pro_2d)
        p_2d = pg_2d[: 2 * h, :]
        sig_2d = jax.nn.sigmoid(pg_2d[2 * h :, :])
        dlr_2d = common.pack_icj_to_2d(dlr)
        dlr_zero_2d = dlr_2d == zero_dt
        dlr_sig_2d = dlr_2d.astype(jnp.float32) * sig_2d
        dp_2d = jnp.where(dlr_zero_2d, zero_dt, dlr_sig_2d.astype(dtype))
        dg_2d = jnp.where(
            dlr_zero_2d,
            zero_dt,
            (dlr_sig_2d * p_2d * (1.0 - sig_2d)).astype(dtype),
        )
        dpg_2d = jnp.concatenate([dp_2d, dg_2d], axis=0)
        safe_act_pro_2d = jnp.where(
            act_pro_2d * zero_dt == zero_dt, act_pro_2d, zero_dt
        )
        dw_pg_ref[...] = dw_pg_ref[...] + common.matmul_dw_2d(
            dpg_2d, safe_act_pro_2d
        ).reshape(1, *dw_pg_shape)
        dact_pro = common.unpack_2d_to_icj(
            common.matmul_wt_2d(w_pg_val, dpg_2d), block_i, block_j
        )
      else:
        pg = common.matmul_3d(w_pg_val, act_pro)
        p = pg[:, : 2 * h, :]
        sig = jax.nn.sigmoid(pg[:, 2 * h :, :])
        dlr_zero = dlr == zero_dt
        dlr_sig = dlr.astype(jnp.float32) * sig
        dp = jnp.where(dlr_zero, zero_dt, dlr_sig.astype(dtype))
        dg = jnp.where(
            dlr_zero, zero_dt, (dlr_sig * p * (1.0 - sig)).astype(dtype)
        )
        dpg = jnp.concatenate([dp, dg], axis=1)
        safe_act_pro = jnp.where(act_pro * zero_dt == zero_dt, act_pro, zero_dt)
        dw_pg_ref[...] = dw_pg_ref[...] + common.matmul_dw_3d(
            dpg, safe_act_pro
        ).reshape(1, *dw_pg_shape)
        dact_pro = common.matmul_wt_3d(w_pg_val, dpg)

      dleft_epi_raw = (
          dleft_act_epi_ref[...].reshape(block_i, c, block_j)
          if use_4d_t8
          else dleft_act_epi_ref[...]
      )
      dact_total = dact_pro + dleft_epi_raw.astype(jnp.float32)
      dx, dln_in_tile = common.layer_norm_3d_axis1_bwd(
          dact_total, x_hat, rstd, ln_in_scale, dtype, from_hat_rstd=True
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
            jax.ShapeDtypeStruct((num_cores_used, 2 * c, 128), jnp.float32),
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
                ln_c_spec,
                ln_c_spec,
            ],
            out_specs=(
                x_spec,
                pl.BlockSpec(
                    (1, *dw_pg_shape), lambda c_idx, si, sj: (c_idx, 0, 0)
                ),
                pl.BlockSpec(
                    (1, 2 * c, 128), lambda c_idx, si, sj: (c_idx, 0, 0)
                ),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        input_output_aliases=pro_aliases,
    )(
        dleft_in,
        dright_in,
        dleft_epi_in,
        x_in,
        mask_in,
        w_pg_arg,
        ln_scale_in,
        ln_offset_in,
    )

    dx_pad = dx_out.reshape(n_pad, c, n_pad) if use_4d_t8 else dx_out
    dw_pg_sum = (
        dw_pg_part[0] if num_cores_used == 1 else jnp.sum(dw_pg_part, axis=0)
    )
    dw_pg = dw_pg_sum.T.astype(dtype)
    dln_in = (
        jnp.sum(dln_in_part[0], axis=1)
        if num_cores_used == 1
        else jnp.sum(dln_in_part, axis=(0, 2))
    ).reshape(2, c).astype(dtype)
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
    ln_in_scale = ln_in_scale_ref[0]
    x_raw = (
        x_ref[...].reshape(block_i, c, block_j) if use_4d_t8 else x_ref[...]
    )

    zero_dt = jnp.zeros((), dtype)
    mask_2d = (
        mask_ref[...].reshape(block_i, block_j)
        if use_4d_hij
        else mask_ref[...]
    )
    mask_zero = mask_2d == 0
    x_hat, rstd = common.layer_norm_3d_axis1_hat_rstd(x_raw, eps)
    act_bf16 = common.apply_norm_affine_axis1(
        x_hat, ln_in_scale, ln_in_offset_ref[0]
    ).astype(dtype)
    act_pro = jnp.where(
        lax.broadcast_in_dim(mask_zero, act_bf16.shape, (0, 2)),
        zero_dt,
        act_bf16,
    )
    p = common.matmul_3d(w_proj_val, act_pro)
    sig = jax.nn.sigmoid(common.matmul_3d(w_gate_val, act_pro))

    dleft_3d = (
        dleft_ref[...].reshape(h, block_i, block_j)
        if use_4d_hij
        else dleft_ref[...]
    )
    dright_3d = (
        dright_ref[...].reshape(h, block_i, block_j)
        if use_4d_hij
        else dright_ref[...]
    )
    dlr = jnp.transpose(
        jnp.concatenate([dleft_3d, dright_3d], axis=0), (1, 0, 2)
    )
    dlr = jnp.where(
        lax.broadcast_in_dim(mask_zero, dlr.shape, (0, 2)), zero_dt, dlr
    )
    dlr_zero = dlr == zero_dt
    dlr_sig = dlr.astype(jnp.float32) * sig
    dp = jnp.where(dlr_zero, zero_dt, dlr_sig.astype(dtype))
    dg = jnp.where(
        dlr_zero, zero_dt, (dlr_sig * p * (1.0 - sig)).astype(dtype)
    )

    safe_act_pro = jnp.where(act_pro * zero_dt == zero_dt, act_pro, zero_dt)
    dw_proj_ref[...] = dw_proj_ref[...] + common.matmul_dw_3d(
        dp, safe_act_pro
    ).reshape(1, *dw_2h_shape)
    dw_gate_ref[...] = dw_gate_ref[...] + common.matmul_dw_3d(
        dg, safe_act_pro
    ).reshape(1, *dw_2h_shape)

    dleft_epi_raw = (
        dleft_act_epi_ref[...].reshape(block_i, c, block_j)
        if use_4d_t8
        else dleft_act_epi_ref[...]
    )
    dact_pro = common.matmul_wt_3d(w_proj_val, dp) + common.matmul_wt_3d(
        w_gate_val, dg
    )
    dact_total = dact_pro + dleft_epi_raw.astype(jnp.float32)
    dx, dln_in_tile = common.layer_norm_3d_axis1_bwd(
        dact_total, x_hat, rstd, ln_in_scale, dtype, from_hat_rstd=True
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
          jax.ShapeDtypeStruct((num_cores_used, 2 * c, 128), jnp.float32),
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
              ln_c_spec,
              ln_c_spec,
          ],
          out_specs=(
              x_spec,
              pl.BlockSpec(
                  (1, *dw_2h_shape), lambda c_idx, si, sj: (c_idx, 0, 0)
              ),
              pl.BlockSpec(
                  (1, *dw_2h_shape), lambda c_idx, si, sj: (c_idx, 0, 0)
              ),
              pl.BlockSpec(
                  (1, 2 * c, 128), lambda c_idx, si, sj: (c_idx, 0, 0)
              ),
          ),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=vmem_limit,
          disable_bounds_checks=True,
      ),
      input_output_aliases=pro_aliases,
  )(
      dleft_in,
      dright_in,
      dleft_epi_in,
      x_in,
      mask_in,
      w_proj_t,
      w_gate_t,
      ln_scale_in,
      ln_offset_in,
  )

  dx_pad = dx_out.reshape(n_pad, c, n_pad) if use_4d_t8 else dx_out
  dw_p_sum = (
      dw_proj_part[0] if num_cores_used == 1 else jnp.sum(dw_proj_part, axis=0)
  )
  dw_g_sum = (
      dw_gate_part[0] if num_cores_used == 1 else jnp.sum(dw_gate_part, axis=0)
  )
  dln_in = (
      jnp.sum(dln_in_part[0], axis=1)
      if num_cores_used == 1
      else jnp.sum(dln_in_part, axis=(0, 2))
  ).reshape(2, c).astype(dtype)

  return (
      dx_pad,
      dw_p_sum.T.reshape(c, 2, h).astype(dtype),
      dw_g_sum.T.reshape(c, 2, h).astype(dtype),
      dln_in[0],
      dln_in[1],
  )
