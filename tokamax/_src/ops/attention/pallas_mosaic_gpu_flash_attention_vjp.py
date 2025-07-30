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
"""Pallas-Mosaic-GPU FlashAttention VJP implementation."""

# pylint: disable=invalid-name

import math
import dataclasses
import functools
from typing import TypeAlias

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.attention import base

Mask = base.Mask
Residuals = base.Residuals
PagingInfo = base.PagingInfo

L: TypeAlias = plgpu.Layout


@dataclasses.dataclass(frozen=True)
class Config:
  block_q_dkv: int
  block_kv_dkv: int
  block_q_dq: int
  block_kv_dq: int
  num_stages: int = 2
  compute_wgs: int = 1

  def __post_init__(self):
    if self.block_q_dkv % 64:
      raise ValueError(f"{self.block_q_dkv=} must be a multiple of 64")
    if self.block_kv_dkv % 64:
      raise ValueError(f"{self.block_kv_dkv=} must be a multiple of 64")
    if self.num_stages < 2:
      raise ValueError(f"{self.num_stages=} must be at least 2")


@jaxtyping.jaxtyped
def _bwd(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*B t h D"],
    v: Float[Array, "*B t h d"],
    residuals: Residuals,
    out: Float[Array, "*B T H d"],
    dout: Float[Array, "*B T H d"],
    *,
    logits_scale: float,
    use_base2: bool,
    config: Config,
) -> tuple[
    Float[Array, "*B T H D"],  # dq
    Float[Array, "*B t h D"],  # dk
    Float[Array, "*B t h d"],  # dv
]:
  orig_q_shape = q.shape
  orig_kv_shape = k.shape
  as_ndim = lambda x, ndim: jax.lax.collapse(
      jax.lax.broadcast_to_rank(x, ndim), 0, -ndim + 1
  )
  as_3d = lambda x: as_ndim(x, 3)
  as_4d = lambda x: as_ndim(x, 4)

  q, k, v, out, dout = map(as_4d, (q, k, v, out, dout))
  m, l = map(as_3d, residuals)

  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, _ = k.shape
  kv_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
  if k.shape != kv_shape:
    raise ValueError(f"Expected {k.shape=} to be {kv_shape} (inferred from q).")
  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(
        f"q, k, and v should all have the same dtype, got: {q.dtype},"
        f" {k.dtype}, {v.dtype}"
    )
  if jnp.dtype(dtype) not in (jnp.float16, jnp.bfloat16):
    raise NotImplementedError(
        f"Only f16 and bf16 are supported, got dtype: {dtype}"
    )
  if head_dim % 64:
    raise ValueError(f"{head_dim=} must be divisible by 64")
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  compute_wgs = config.compute_wgs
  num_q_tiles, rem = divmod(q_seq_len, config.block_q_dq * compute_wgs)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {config.block_q_dq=} *"
        f" {compute_wgs=}"
    )

  num_kv_tiles, rem = divmod(kv_seq_len, config.block_kv_dkv * compute_wgs)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len=} must be a multiple of {config.block_kv_dkv=} *"
        f" {compute_wgs=}"
    )

  num_q_tiles_in_dkv, rem = divmod(q_seq_len, config.block_q_dkv)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {config.block_q_dkv=}"
    )

  num_kv_tiles_in_dq, rem = divmod(kv_seq_len, config.block_kv_dq)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len=} must be a multiple of {config.block_kv_dq=}"
    )

  swizzle = 128
  transforms = (
      plgpu.TilingTransform((8, swizzle // q.dtype.itemsize)),
      plgpu.SwizzleTransform(swizzle),
  )
  delta = jnp.einsum(
      "bqhd,bqhd->bhq", out.astype(jnp.float32), dout.astype(jnp.float32)
  )

  exp = jnp.exp2 if use_base2 else jnp.exp

  def kernel_dq(q_ref, k_ref, v_ref, dout_ref, m_ref, l_ref, delta_ref, dq_ref,
                smem_buffers, buffer_barriers, block_q: int, block_kv: int):
    b_idx = lax.axis_index("batch")
    q_idx = lax.axis_index("q_tiles")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")
    kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))

    q_smems, dout_smems, m_smems, l_smems, delta_smems = smem_buffers
    q_barriers, dout_barriers, m_barriers, l_barriers, delta_barriers = (
        buffer_barriers
    )

    def compute_thread(pipeline_callback):
      q_smem = q_smems.at[wg_idx]
      dout_smem = dout_smems.at[wg_idx]
      m_smem = m_smems.at[wg_idx]
      l_smem = l_smems.at[wg_idx]
      delta_smem = delta_smems.at[wg_idx]

      q_seq_base = q_idx * (compute_wgs * block_q) + wg_idx * block_q
      q_slice = (b_idx, pl.ds(q_seq_base, block_q), q_head)
      res_slice = (b_idx, q_head, pl.ds(q_seq_base, block_q))

      plgpu.copy_gmem_to_smem(q_ref.at[q_slice], q_smem, q_barriers.at[wg_idx])
      plgpu.copy_gmem_to_smem(
          dout_ref.at[q_slice], dout_smem, dout_barriers.at[wg_idx]
      )
      plgpu.copy_gmem_to_smem(
          delta_ref.at[res_slice], delta_smem, delta_barriers.at[wg_idx]
      )
      plgpu.copy_gmem_to_smem(
          m_ref.at[res_slice], m_smem, m_barriers.at[wg_idx]
      )
      plgpu.copy_gmem_to_smem(
          l_ref.at[res_slice], l_smem, l_barriers.at[wg_idx]
      )
      _ = [plgpu.barrier_wait(buffer.at[wg_idx]) for buffer in buffer_barriers]

      delta = plgpu.load(delta_smem, (), layout=L.WGMMA.reduce(1))  # [block_q]
      m = plgpu.load(m_smem, (), layout=L.WGMMA.reduce(1))  # [block_q]
      if use_base2:
        m *= math.log2(math.e)
      l = plgpu.load(l_smem, (), layout=L.WGMMA.reduce(1))  # [block_q]
      dq_acc = plgpu.layout_cast(
          jnp.full((block_q, head_dim), 0, dtype=jnp.float32), L.WGMMA,
      )
      dq, _, _, _ = pipeline_callback((dq_acc, m, l, delta))
      q_smem[...] = dq.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(q_smem, dq_ref.at[q_slice])
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    def kv_pipeline(
        _, k_smem, v_smem, k_consumed_barrier, v_consumed_barrier, carry
    ):
      q_smem, dout_smem = q_smems.at[wg_idx], dout_smems.at[wg_idx]
      (dq_acc, m, l, delta) = carry

      def compute_s(acc_ref):
        plgpu.wgmma(acc_ref, q_smem, plgpu.transpose_ref(k_smem, (1, 0)))
        return acc_ref[...]

      s = pl.run_scoped(compute_s, plgpu.ACC((block_q, block_kv), jnp.float32))

      s_scale = logits_scale
      if use_base2:
        s_scale *= math.log2(math.e)

      s *= s_scale

      broadcast = lambda x: lax.broadcast_in_dim(x, (block_q, block_kv), [0])
      p = exp(s - broadcast(m)) / broadcast(l)

      def compute_dp(acc_ref):
        plgpu.wgmma(acc_ref, dout_smem, plgpu.transpose_ref(v_smem, (1, 0)))
        return acc_ref[...]

      dp = pl.run_scoped(
          compute_dp, plgpu.ACC((block_q, block_kv), jnp.float32)
      )
      plgpu.barrier_arrive(v_consumed_barrier)

      ds = p * (dp - lax.broadcast_in_dim(delta, (block_q, block_kv), [0]))
      ds *= logits_scale

      def compute_dq(acc_ref):
        plgpu.wgmma(acc_ref, ds.astype(k_ref.dtype), k_smem)

      dq_acc = pl.run_state(compute_dq)(plgpu.ACC.init(dq_acc))
      plgpu.barrier_arrive(k_consumed_barrier)

      return (dq_acc, m, l, delta)

    pipeline = plgpu.emit_pipeline_warp_specialized(
        kv_pipeline,
        grid=(num_kv_tiles_in_dq,),
        max_concurrent_steps=min(config.num_stages, num_kv_tiles_in_dq),
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=compute_thread,
        in_specs=[
            plgpu.BlockSpec(  # k
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0),
                transforms=transforms),
            plgpu.BlockSpec(  # v
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0),
                transforms=transforms),
        ])
    k_ref = k_ref.at[b_idx, :, kv_head, :]
    v_ref = v_ref.at[b_idx, :, kv_head, :]
    pipeline(k_ref, v_ref)

  def kernel_dkv(q_ref, k_ref, v_ref, dout_ref, m_ref, l_ref, delta_ref,
                 dk_ref, dv_ref, smem_buffers, buffer_barriers, block_q: int,
                 block_kv: int):
    b_idx = lax.axis_index("batch")
    kv_idx = lax.axis_index("num_kv_tiles")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")
    (k_smems, v_smems) = smem_buffers
    (k_barriers, v_barriers) = buffer_barriers

    def compute_thread(pipeline_callback):
      k_smem, v_smem = k_smems.at[wg_idx], v_smems.at[wg_idx]
      kv_seq_base = kv_idx * (compute_wgs * block_kv) + wg_idx * block_kv
      kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
      kv_slice = (b_idx, pl.ds(kv_seq_base, block_kv), kv_head)
      plgpu.copy_gmem_to_smem(k_ref.at[kv_slice], k_smem, k_barriers.at[wg_idx])
      plgpu.copy_gmem_to_smem(v_ref.at[kv_slice], v_smem, v_barriers.at[wg_idx])
      plgpu.barrier_wait(k_barriers.at[wg_idx])
      plgpu.barrier_wait(v_barriers.at[wg_idx])
      dk_acc = plgpu.layout_cast(
          jnp.full((block_kv, head_dim), 0, dtype=jnp.float32), L.WGMMA,
      )
      dv_acc = plgpu.layout_cast(
          jnp.full((block_kv, head_dim), 0, dtype=jnp.float32), L.WGMMA,
      )
      (dk, dv) = pipeline_callback((dv_acc, dk_acc))
      k_smem[...] = dk.astype(k.dtype)
      v_smem[...] = dv.astype(v.dtype)

      plgpu.commit_smem()
      kv_out_slice = (b_idx, pl.ds(kv_seq_base, block_kv), q_head)
      plgpu.copy_smem_to_gmem(
          k_smem, dk_ref.at[kv_out_slice], commit_group=False)
      plgpu.copy_smem_to_gmem(
          v_smem, dv_ref.at[kv_out_slice], commit_group=False)
      plgpu.commit_smem_to_gmem_group()
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    def q_pipeline(
        _,
        q_smem,
        dout_smem,
        m_smem,
        l_smem,
        delta_smem,
        q_consumed_barrier,
        dout_consumed_barrier,
        m_consumed_barrier,
        l_consumed_barrier,
        delta_consumed_barrier,
        carry,
    ):
      k_smem, v_smem = k_smems.at[wg_idx], v_smems.at[wg_idx]
      dk_acc, dv_acc = carry

      def compute_sT(acc_ref):
        plgpu.wgmma(acc_ref, k_smem, plgpu.transpose_ref(q_smem, (1, 0)))
        return acc_ref[...]

      m = plgpu.load(m_smem, (), layout=L.WGMMA.reduce(0))
      l = plgpu.load(l_smem, (), layout=L.WGMMA.reduce(0))
      plgpu.barrier_arrive(m_consumed_barrier)
      plgpu.barrier_arrive(l_consumed_barrier)

      broadcast = lambda x: lax.broadcast_in_dim(x, (block_kv, block_q), [1])
      sT = pl.run_scoped(
          compute_sT, plgpu.ACC((block_kv, block_q), jnp.float32)
      )

      s_scale = logits_scale
      if use_base2:
        s_scale *= math.log2(math.e)
        m *= math.log2(math.e)

      sT *= s_scale

      pT = exp(sT - broadcast(m)) / broadcast(l)

      def _compute(refs):
        # Combining two WGMMA calls in one block to avoid the unnecessary
        # synchronization from two `wgmma.wait_group` calls.
        dv_acc_ref, dpT_acc_ref = refs
        plgpu.wgmma(dv_acc_ref, pT.astype(dtype), dout_smem)
        plgpu.wgmma(dpT_acc_ref, v_smem, plgpu.transpose_ref(dout_smem, (1, 0)))

      zeros = plgpu.layout_cast(
          jnp.full((block_kv, block_q), 0, dtype=jnp.float32), L.WGMMA,
      )
      dv_acc, dpT = pl.run_state(_compute)(
          (plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros))
      )
      plgpu.barrier_arrive(dout_consumed_barrier)

      delta = plgpu.load(delta_smem, (), layout=L.WGMMA.reduce(0))
      plgpu.barrier_arrive(delta_consumed_barrier)

      dsT = pT * (dpT - broadcast(delta))
      dsT *= logits_scale

      def compute_dk(acc_ref):
        plgpu.wgmma(acc_ref, dsT.astype(dtype), q_smem)

      dk_acc = pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc))
      plgpu.barrier_arrive(q_consumed_barrier)

      return (dk_acc, dv_acc)

    pipeline = plgpu.emit_pipeline_warp_specialized(
        q_pipeline,
        grid=(num_q_tiles_in_dkv,),
        max_concurrent_steps=min([config.num_stages, num_q_tiles_in_dkv]),
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=compute_thread,
        in_specs=[
            plgpu.BlockSpec(  # q
                block_shape=(block_q, head_dim),
                index_map=lambda i: (i, 0),
                transforms=transforms,
            ),
            plgpu.BlockSpec(  # dout
                block_shape=(block_q, head_dim),
                index_map=lambda i: (i, 0),
                transforms=transforms,
            ),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
        ],
    )
    q_ref = q_ref.at[b_idx, :, q_head, :]
    dout_ref = dout_ref.at[b_idx, :, q_head, :]
    m_ref = m_ref.at[b_idx, q_head, :]
    l_ref = l_ref.at[b_idx, q_head, :]
    delta_ref = delta_ref.at[b_idx, q_head, :]
    pipeline(q_ref, dout_ref, m_ref, l_ref, delta_ref)

  q_scratch = dout_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq, head_dim),
      q.dtype,
      transforms=transforms,
  )
  m_scratch = l_scratch = delta_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq), jnp.float32
  )
  # TODO: Optionally fuse the dq and dkv kernels.
  dq = plgpu.kernel(
      functools.partial(
          kernel_dq, block_q=config.block_q_dq, block_kv=config.block_kv_dq
      ),
      out_shape=q,
      scratch_shapes=[
          (q_scratch, dout_scratch, m_scratch, l_scratch, delta_scratch),  # type: ignore
          (plgpu.Barrier(num_barriers=compute_wgs),) * 5,  # type: ignore
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, num_q_tiles, batch_size),
      grid_names=("heads", "q_tiles", "batch"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
  )(q, k, v, dout, m, l, delta)

  k_scratch = v_scratch = plgpu.SMEM(
      (compute_wgs, config.block_kv_dkv, head_dim),
      k.dtype,
      transforms=transforms,
  )
  out_shape_kv = jax.ShapeDtypeStruct(
      (batch_size, kv_seq_len, num_q_heads, head_dim), dtype=k.dtype
  )
  dk, dv = plgpu.kernel(
      functools.partial(
          kernel_dkv, block_q=config.block_q_dkv, block_kv=config.block_kv_dkv
      ),
      out_shape=[out_shape_kv, out_shape_kv],
      scratch_shapes=[
          (k_scratch, v_scratch),  # type: ignore
          (plgpu.Barrier(num_barriers=compute_wgs),) * 2,  # type: ignore
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, num_kv_tiles, batch_size),
      grid_names=("heads", "num_kv_tiles", "batch"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
  )(q, k, v, dout, m, l, delta)

  if q_heads_per_kv_head > 1:
    sum_shape = (*k.shape[:-1], q_heads_per_kv_head, head_dim)
    dk = dk.reshape(sum_shape).astype(jnp.float32).sum(axis=-2).astype(dk.dtype)
    dv = dv.reshape(sum_shape).astype(jnp.float32).sum(axis=-2).astype(dv.dtype)

  dq = dq.reshape(*orig_q_shape)
  dk = dk.reshape(*orig_kv_shape)
  dv = dv.reshape(*orig_kv_shape)

  return dq, dk, dv


_SUPPORTED_PRECISIONS = (
    lax.DotAlgorithmPreset.DEFAULT,
    lax.DotAlgorithmPreset.BF16_BF16_F32,
    lax.DotAlgorithmPreset.F16_F16_F32,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicGpuFlashAttentionVjp(
    base.DotProductAttentionVjp[Config, None]
):
  """Pallas-Triton FlashAttention VJP implementation."""

  use_base2: bool = False

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, "*B T H d"],
      dout: Float[Array, "*B T H d"],
      q: Float[Array, "*B T H D"],
      k: Float[Array, "*B t h D"],
      v: Float[Array, "*B t h d"],
      *,
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      logits_soft_cap: float | None,
      mask: base.Mask,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #H t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[base.DotProductAttentionGrads, None]:
    del dropout_rate

    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    if not normalize_output:
      raise NotImplementedError("`normalize_output=False` not supported.")

    if logits_dtype != jnp.float32:
      raise NotImplementedError("`logits_dtype` must be float32.")

    # TODO: Add support for `bias`.
    if bias is not None:
      raise ValueError("`bias` not supported.")

    # TODO: Add support for `logits_soft_cap`.
    if logits_soft_cap is not None:
      raise ValueError("`logits_soft_cap` not supported.")

    # TODO: Add support for `mask`.
    if mask != base.Mask(None):
      raise ValueError("`mask` not supported.")

    if dropout_mask is not None:
      raise NotImplementedError("dropout is not supported.")

    if q_indices is not None:
      raise NotImplementedError("q_indices is not implemented.")

    if k_indices is not None:
      raise NotImplementedError("k_indices is not implemented.")

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    q_k_dot_precision, weights_v_dot_precision = precision
    if q_k_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f"{q_k_dot_precision=} not supported")
    if weights_v_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f" {weights_v_dot_precision=} not supported")

    f = functools.partial(
        _bwd, logits_scale=logits_scale, use_base2=self.use_base2, config=config
    )

    args = (q, k, v, residuals, out, dout)

    dq, dk, dv = f(*args)
    return base.DotProductAttentionGrads(q=dq, k=dk, v=dv, bias=None), None

  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        block_q_dkv=64,
        block_kv_dkv=64,
        block_q_dq=64,
        block_kv_dq=64,
        num_stages=2,
    )
