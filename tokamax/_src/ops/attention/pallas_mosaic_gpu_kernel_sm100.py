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
"""B200 Flash attention with Mosaic GPU."""

import functools
import itertools
import math

import jax
from jax import lax
from jax.experimental.mosaic import gpu as mgpu
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import vector
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common


DotPrecisionLike = lax.Precision | lax.DotAlgorithmPreset
PagingInfo = base.PagingInfo
QArray = base.QArray
Residuals = base.Residuals


_TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
_TMEM_COL = _TMEM.reduce(0)
_TMEM_ROW = _TMEM.reduce(1)
_TCGEN05 = plgpu.Layout.TCGEN05
_TCGEN05_ROW = _TCGEN05.reduce(1)
_WG_SPLAT = plgpu.Layout.WG_SPLAT
_DEFAULT_MASK_VALUE = -1e30

_MMA_TMA_WG = 0
_SOFTMAX_WG = 1
_SCALE_WG = 2
_MMA_WARP = 0
_TMA_LOAD_QK_WARP = 1
_TMA_LOAD_V_WARP = 2
_TMA_LOAD_MASK_WARP = 3
_L_BARRIER_ID = 7
_ALPHA_BARRIER_OFFSET = 8

_load_bcast = common.load_bcast

_MASK_PACK_DTYPE = jnp.int8
_MASK_PACKED_BITS = mgpu_lib.num_bits(_MASK_PACK_DTYPE)


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=dict(extra="forbid")
)  # pytype: disable=wrong-keyword-args
class Config(common.ConfigBase):
  """Configuration parameters for Pallas-Mosaic-GPU kernels on SM100 GPUs.

  Attributes:
    num_tma_splits: Number of chunks to load each K/V - helpful to better hide
      GMEM load latences as we can notify TMA warp after part of the mma, thus
      giving more time to TMA loads.
    collective: if True - 2 CTA MMA will be run with M=256, N=128
  """

  num_tma_splits: pydantic.PositiveInt = 1
  collective: pydantic.StrictBool = True

  def __post_init__(self):
    block_q_per_cta = self.block_q // 2 if self.collective else self.block_q
    if block_q_per_cta < 128:
      raise ValueError(
          f"For SM100 attention forward, block_q per CTA must be at least 128 "
          f"to support TMEM slicing. Got block_q={self.block_q} with "
          f"collective={self.collective}."
      )


def get_heuristics_config(ba: op.BoundArguments) -> Config:
  """Returns a heuristic configuration for flash attention on SM100 GPUs."""
  q, _, v, *_ = ba.args
  *batch_size, q_seq_len, q_heads, head_dim = q.shape
  head_dim = pl.cdiv(max(head_dim, v.shape[-1]), 64) * 64
  batch_size = math.prod(batch_size)
  kv_seq_len = v.shape[-3]
  num_tma_splits = 2 if head_dim >= 256 else 1
  collective = True
  cluster_size = 1 + int(collective)
  num_stages = max(256 // head_dim, 1) * cluster_size
  block_q = 256 if collective else 128
  if ba.kwargs.get("bias") is not None:
    num_stages = min(num_stages, 4)
  block_kv = 128
  split_k = 1

  mask = ba.kwargs.get("mask", None)
  # We use 0.5 threshold here as a safe choice for automatic K-split usage.
  # For other cases like 0.8 etc. we need a smarter heuristic or autotuning.
  min_load_factor = 0.5
  grid_size = batch_size * pl.cdiv(q_seq_len, block_q) * q_heads
  num_ctas = backend.get_default_device().core_count // cluster_size
  # We do not support k split yet for causal attn or with k ranges
  not_masked = mask is None or not (
      mask.is_causal or mask.k_start is not None or mask.k_end is not None
  )
  is_kv_seq_aligned = kv_seq_len % block_kv == 0
  # TODO fix test failures for non aligned q seq
  is_q_seq_aligned = q_seq_len % block_q == 0
  if (
      grid_size / num_ctas < min_load_factor
      and is_kv_seq_aligned
      and is_q_seq_aligned
      and not_masked
  ):
    split_k = num_ctas // grid_size
    split_k = min(kv_seq_len // block_kv, split_k)
    while kv_seq_len % split_k != 0:
      split_k -= 1

  return Config(
      block_q=block_q,
      block_kv=block_kv,
      collective=collective,
      num_stages=num_stages,
      num_tma_splits=num_tma_splits,
      split_k=split_k,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  """Returns a set of configs for autotuning flash attention on SM100 GPUs."""
  del ba
  configs = set()
  for block_kv in [64, 128]:
    for num_stages in [1, 2, 3, 4]:
      for num_tma_splits in [1, 2, 3, 4]:
        # TODO: Investigate why split_k=2 doesn't work with block_kv=128.
        for split_k in [1, 2] if block_kv == 64 else [1]:
          for collective in [False, True] if split_k == 1 else [False]:
            configs.add(
                Config(
                    block_q=256 if collective else 128,
                    block_kv=block_kv,
                    num_stages=num_stages,
                    num_tma_splits=num_tma_splits,
                    collective=collective,
                    split_k=split_k,
                )
            )
  return configs


@jaxtyping.jaxtyped
def flash_attention_kernel(
    q: Float[Array, "T H D"],
    k: Float[Array, "t h D"],
    v: Float[Array, "t h d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | None,
    k_start: Int[Array, "#H #T"] | None,
    k_end: Int[Array, "#H #T"] | None,
    *,
    is_causal: bool,
    logits_soft_cap: float | None,
    logits_scale: float,
    out_dtype: jnp.dtype,
    normalize_output: bool,
    return_residuals: bool,
    use_stable_softmax: bool,
    rescale_threshold: float,
    config: Config,
) -> tuple[Float[Array, "T H d"], Residuals | None]:
  """SM100 Pallas Mosaic GPU Flash Attention."""

  if not use_stable_softmax:
    raise NotImplementedError("Unstable softmax not supported on sm100.")

  q_seq_len, num_q_heads, _ = q.shape
  dtype = q.dtype

  kv_seq_len, num_kv_heads, orig_head_dim_out = v.shape
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(
        f"Only f16 and bf16 are supported, got dtype: {dtype}"
    )

  q, k, v = map(common.pad_head_dim_to_next_multiple_of_min_swizzle, (q, k, v))
  if config.collective:
    m = 2 * 8 * common.MIN_SWIZZLE // mgpu_lib.num_bits(v.dtype)
    v = shape_lib.pad_to_next_multiple_of(v, m, -1)  # Need >=32 bytes per CTA.
  head_dim = q.shape[-1]
  head_dim_out = v.shape[-1]

  min_mask_cols = 128 // _MASK_PACKED_BITS
  if mask is None:
    use_2d_bool_mask = False
  else:
    bcast_mask_q = mask.shape[-2] == 1
    bcast_mask_k = mask.shape[-1] == 1
    use_2d_bool_mask = not (bcast_mask_q or bcast_mask_k)
    if use_2d_bool_mask:
      mask = jnp.packbits(mask, axis=-1, bitorder="little")
      mask = mask.view(_MASK_PACK_DTYPE)
      # align to 16 bytes
      mask = shape_lib.pad_to_next_multiple_of(mask, m=min_mask_cols, axis=-1)
    else:
      mask = mask.astype(jnp.int8)

  tile_q, block_kv = config.block_q, config.block_kv
  num_q_tiles = pl.cdiv(q_seq_len, tile_q)
  mask_block_kv = block_kv // _MASK_PACKED_BITS
  num_stages = config.num_stages
  num_tma_splits = config.num_tma_splits
  collective = config.collective
  block_q = tile_q // 2 if collective else tile_q
  collective_axis = "cluster" if collective else None
  softmax_slots = 2

  epi_tile_d = 1024 // mgpu_lib.num_bits(out_dtype)
  if head_dim_out % epi_tile_d != 0:
    epi_tile_d = head_dim_out
  num_epi_slots = min(2, head_dim_out // epi_tile_d)

  def kernel(
      q_gmem,
      k_gmem,
      v_gmem,
      bias_gmem,
      mask_gmem,
      k_start_gmem,
      k_end_gmem,
      k_start_minmax_gmems,
      k_end_minmax_gmems,
      o_gmem,
      *residual_gmems,
      qo_smem_union,
      k_smem,
      v_smem,
      mask_smem,
      alpha_smem,
      li_smem,
      qk_acc_tmem,
      p_tmem,
      acc_tmem,
      bias_smem,
      q_barrier,
      k_barrier,
      v_barrier,
      bias_barrier,
      bias_consumed_barrier,
      mask_barrier,
      mask_consumed_barrier,
      qk_mma_barrier,
      k_consumed_barrier,
      qk_consumed_barrier,
      pv_mma_barrier,
      v_consumed_barrier,
      p_produced_barrier,
      out_scaled_barrier,
  ):
    (q_smem, o_smem) = qo_smem_union

    qi = lax.axis_index("q_tiles")
    hi = lax.axis_index("heads")
    wg = lax.axis_index("wg")
    cluster_idx = lax.axis_index("cluster")

    q_base_cluster = qi * tile_q
    q_base = q_base_cluster + cluster_idx * block_q
    qs = pl.ds(q_base, block_q)

    lb = 0
    ub = kv_seq_len // block_kv

    if is_causal:
      ub = lax.min(ub, pl.cdiv(q_base_cluster + tile_q, block_kv))

    def load_k_bound(k_range_ref):
      idx = (
          lax.min(hi, k_range_ref.shape[-2] - 1),
          0 if k_range_ref.shape[-1] == 1 else qi,
      )
      return plgpu.load(k_range_ref, idx=idx, layout=plgpu.Layout.WG_SPLAT)

    if k_start_minmax_gmems is None:
      k_start_max = None
    else:
      k_start_min, k_start_max = map(load_k_bound, k_start_minmax_gmems)
      lb = lax.max(lb, lax.div(k_start_min, block_kv))

    if k_end_minmax_gmems is None:
      k_end_min = None
    else:
      k_end_min, k_end_max = map(load_k_bound, k_end_minmax_gmems)
      ub = lax.min(ub, pl.cdiv(k_end_max, block_kv))

    @pl.when((wg == _MMA_TMA_WG) & (ub > lb))
    def mma_tma_wg():
      plgpu.set_max_registers(80, action="decrease")

      @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
      def per_warp():
        warp_id = lax.axis_index("warp")

        def tma_load_kv(gmem, smem, barrier, partition_axis, ki, split_idx):
          kv_head = lax.div(hi, q_heads_per_kv_head)
          si = lax.rem(ki - lb, num_stages)
          block_d = gmem.shape[-1] // num_tma_splits
          ds = pl.ds(split_idx * block_d, block_d)
          if collective:
            leader_tracked = plgpu.CopyPartition.PARTITIONED(partition_axis)
          else:
            leader_tracked = None
          plgpu.copy_gmem_to_smem(
              gmem.at[pl.ds(ki * block_kv, block_kv), kv_head, ds],
              smem.at[si, split_idx],
              barrier=barrier.at[si],
              collective_axes=collective_axis,
              leader_tracked=leader_tracked,
          )

        def tma_load_kv_warp(
            gmem, smem, barrier, consumed_barrier, partition_axis
        ):
          tma_load = functools.partial(
              tma_load_kv, gmem, smem, barrier, partition_axis
          )

          @pl.loop(lb, lax.min(lb + num_stages, ub))
          def prologue(ki):
            pl.loop(0, num_tma_splits)(functools.partial(tma_load, ki))

          @pl.loop(lb + num_stages, ub)
          def kv_loop(ki):
            si = lax.rem(ki - lb, num_stages)

            @pl.loop(0, num_tma_splits)
            def tma_loop(split_idx):
              slot = si * num_tma_splits + split_idx
              plgpu.barrier_wait(consumed_barrier.at[slot])
              tma_load(ki, split_idx)

        @pl.when(warp_id == _TMA_LOAD_QK_WARP)
        def tma_load_qk_warp():
          if collective:
            leader_tracked = plgpu.CopyPartition.PARTITIONED(0)
          else:
            leader_tracked = None
          plgpu.copy_gmem_to_smem(
              q_gmem.at[pl.ds(q_base_cluster, tile_q), hi],
              q_smem,
              barrier=q_barrier,
              collective_axes=collective_axis,
              leader_tracked=leader_tracked,
          )
          tma_load_kv_warp(k_gmem, k_smem, k_barrier, k_consumed_barrier, 0)

        @pl.when(warp_id == _TMA_LOAD_V_WARP)
        def tma_load_v_warp():
          tma_load_kv_warp(v_gmem, v_smem, v_barrier, v_consumed_barrier, 1)

        if bias_smem is not None or mask_smem is not None:

          @pl.when(warp_id == _TMA_LOAD_MASK_WARP)
          def tma_load_bias_mask_warp():
            @pl.loop(lb, ub)
            def kv_loop(ki):
              if bias_smem is not None:
                hi_ = 0 if bias_gmem.shape[-3] == 1 else hi
                ks = pl.ds(ki * block_kv, block_kv)
                plgpu.barrier_wait(bias_consumed_barrier)
                mgpu_lib.fence_async_shared_cta()  # Ensure smem read complete.
                plgpu.copy_gmem_to_smem(
                    bias_gmem.at[hi_, qs, ks], bias_smem, bias_barrier
                )
              if mask_smem is not None:
                hi_ = 0 if mask_gmem.shape[-3] == 1 else hi
                mask_block_size = max(mask_block_kv, min_mask_cols)
                ks = pl.ds(ki * mask_block_kv, mask_block_size)
                plgpu.barrier_wait(mask_consumed_barrier)
                mgpu_lib.fence_async_shared_cta()  # Ensure smem read complete.
                plgpu.copy_gmem_to_smem(
                    mask_gmem.at[hi_, qs, ks], mask_smem, mask_barrier
                )

        @pl.when((warp_id == _MMA_WARP) & (cluster_idx == 0))
        def mma_warp():

          def qk_mma(ki):
            si = lax.rem(ki - lb, num_stages)
            with jax.named_scope("wait_k"):
              plgpu.barrier_wait(qk_consumed_barrier)
              plgpu.barrier_wait(k_barrier.at[si])

            @pl.loop(0, num_tma_splits)
            def tma_loop(split_idx):
              block_d = head_dim // num_tma_splits
              ds = pl.ds(split_idx * block_d, block_d)
              with jax.named_scope("issuing Q@K.T"):
                plgpu.tcgen05_mma(
                    qk_acc_tmem,
                    q_smem.at[:, ds],
                    k_smem.at[si, split_idx].T,
                    k_consumed_barrier.at[si * num_tma_splits + split_idx],
                    accumulate=split_idx > 0,
                    collective_axis=collective_axis,
                )

            plgpu.tcgen05_commit_arrive(qk_mma_barrier, collective_axis)

          def pv_mma(ki):
            si = lax.rem(ki - lb, num_stages)
            slot = lax.rem(ki - lb, 2)
            with jax.named_scope("wait_v"):
              plgpu.barrier_wait(v_barrier.at[si])
              plgpu.barrier_wait(p_produced_barrier.at[slot])

            @pl.loop(0, num_tma_splits)
            def tma_loop(split_idx):
              barrier_slot = si * num_tma_splits + split_idx
              block_d = head_dim_out // num_tma_splits
              ds = pl.ds(split_idx * block_d, block_d)
              plgpu.barrier_wait(out_scaled_barrier.at[split_idx])
              with jax.named_scope("issuing P@V"):
                plgpu.tcgen05_mma(
                    acc_tmem.at[:, ds],
                    p_tmem.at[:, pl.ds(slot * block_kv, block_kv)],
                    v_smem.at[si, split_idx],
                    v_consumed_barrier.at[barrier_slot],
                    accumulate=(ki != lb),
                    collective_axis=collective_axis,
                )

            plgpu.tcgen05_commit_arrive(pv_mma_barrier, collective_axis)

          plgpu.barrier_wait(q_barrier)
          qk_mma(lb)

          @pl.loop(lb, ub - 1)
          def kv_loop(ki):
            qk_mma(ki + 1)
            pv_mma(ki)

          pv_mma(ub - 1)

    @pl.when((wg == _SOFTMAX_WG) & (ub > lb))
    def softmax_wg():
      plgpu.set_max_registers(256, action="increase")

      m_i = plgpu.layout_cast(
          jnp.full((block_q,), -jnp.inf, jnp.float32), _TMEM_ROW
      )
      l_i = plgpu.layout_cast(jnp.zeros_like(m_i), _TMEM_ROW)

      load_k_range = lambda r: _load_bcast(r, (hi, qs), layout=_TMEM_ROW)
      k_start = None if k_start_gmem is None else load_k_range(k_start_gmem)
      k_end = None if k_end_gmem is None else load_k_range(k_end_gmem)

      def needs_k_range_mask(ki):
        needs_apply = False
        if k_end_min is not None:
          needs_apply |= ki * block_kv + block_kv > k_end_min
        if k_start_max is not None:
          needs_apply |= ki * block_kv < k_start_max
        return needs_apply

      def compute_mask(ki, do_causal):
        k_base = ki * block_kv
        acc_shape = (block_q, block_kv)
        iota = lambda d: plgpu.broadcasted_iota(
            jnp.int32, acc_shape, dimension=d, layout=_TMEM
        )

        if do_causal:
          mask = (q_base + iota(0)) >= (k_base + iota(1))
        else:
          mask = plgpu.layout_cast(jnp.ones(acc_shape, dtype=jnp.bool_), _TMEM)

        if mask_gmem is not None:
          if mask_smem is None:
            ks = pl.ds(k_base, block_kv)
            # TODO: we need to handle Q masks differently
            # broadcasting & using them the way it is done is extremely slow
            mask &= common.load_bcast(mask_gmem, (hi, qs, ks), layout=_TMEM)
          else:
            plgpu.barrier_wait(mask_barrier)
            idx = (slice(None), slice(0, mask_block_kv))
            layout = _TMEM(32 // _MASK_PACKED_BITS)
            mask_ = plgpu.load(mask_smem, idx, layout=layout, optimized=False)
            plgpu.barrier_arrive(mask_consumed_barrier)
            mask &= common.unpack_bool_bits_tmem_native(mask_)

        def k_range_mask(mask):
          bc_range = lambda x: lax.broadcast_in_dim(x, mask.shape, (0,))
          if k_start_gmem is not None:
            mask &= (k_base + iota(1)) >= bc_range(k_start)
          if k_end_gmem is not None:
            mask &= (k_base + iota(1)) < bc_range(k_end)
          return mask

        mask = lax.cond(
            needs_k_range_mask(ki), lambda: k_range_mask(mask), lambda: mask
        )
        return jnp.where(mask, 0, _DEFAULT_MASK_VALUE)

      def kv_loop(ki, carry, *, do_causal=False):
        m_scale, m_i, l_i = carry
        si = lax.rem(ki - lb, 2)
        with jax.named_scope("Q@K"):
          plgpu.barrier_wait(qk_mma_barrier)
        with jax.named_scope("load_qk"):
          s = plgpu.async_load_tmem(qk_acc_tmem, layout=_TMEM)
          scale = logits_scale

          if bias_gmem is None:
            bias = None
          elif bias_smem is None:
            hi_ = 0 if bias_gmem.shape[-3] == 1 else hi
            ks = pl.ds(ki * block_kv, block_kv)
            bias = common.load_bcast(bias_gmem, (hi_, qs, ks), layout=_TMEM)
          else:
            plgpu.barrier_wait(bias_barrier)
            bias = plgpu.load(bias_smem, (), layout=_TMEM, optimized=False)
            plgpu.barrier_arrive(bias_consumed_barrier)

          plgpu.wait_load_tmem()
          plgpu.barrier_arrive(qk_consumed_barrier)

        if bias is not None:
          s = s * scale + bias.astype(s.dtype)
          scale = 1.0

        if logits_soft_cap is not None:
          s, scale = jnp.tanh(s * (scale / logits_soft_cap)), logits_soft_cap

        with jax.named_scope("softmax"):
          scale *= math.log2(math.e)
          s, scale = lax.cond(
              do_causal or mask_gmem is not None or needs_k_range_mask(ki),
              lambda: (s * scale + compute_mask(ki, do_causal), 1.0),
              lambda: (s, scale),
          )
          m_i = jnp.maximum(m_i, s.max(axis=1) * scale)
          alpha = jnp.exp2(m_scale - m_i)

          @pl.when(ki > lb)
          def write_alpha_to_smem():
            alpha_smem.at[si][...] = alpha
            mgpu_lib.bar_arrive(si + _ALPHA_BARRIER_OFFSET, num_threads=256)

          needs_rescale = (
              (rescale_threshold == 1.0)
              | (alpha < rescale_threshold)
              | ((not normalize_output) & (ki == ub - 1))
          )
          m_scale = jnp.where(needs_rescale, m_i, m_scale)
          p = jnp.exp2(s * scale - lax.broadcast_in_dim(m_scale, s.shape, [0]))
          l_i = jnp.where(needs_rescale, l_i * alpha, l_i) + p.sum(axis=1)

          with jax.named_scope("write qk_tmem"):
            ks = pl.ds(si * block_kv, block_kv)
            plgpu.async_store_tmem(p_tmem.at[:, ks], p.astype(p_tmem.dtype))
            plgpu.commit_tmem()
        plgpu.barrier_arrive(p_produced_barrier.at[si])
        return m_scale, m_i, l_i

      # prologue
      plgpu.barrier_arrive(qk_consumed_barrier)
      if bias_consumed_barrier is not None:
        plgpu.barrier_arrive(bias_consumed_barrier)
      if mask_consumed_barrier is not None:
        plgpu.barrier_arrive(mask_consumed_barrier)

      # in 2CTA we have non square blocks hence we may need to process
      # M//N steps with a mask, for M=256, N=128 this means 2 steps
      causal_blocks = int(is_causal) * (tile_q // block_kv)
      carry = lax.fori_loop(lb, ub - causal_blocks, kv_loop, (m_i, m_i, l_i))

      if is_causal:
        causal_kv_loop = functools.partial(kv_loop, do_causal=True)
        carry = lax.fori_loop(ub - causal_blocks, ub, causal_kv_loop, carry)

      m_scale, m_i, l_i = carry

      if normalize_output:
        li_smem[...] = l_i
        mgpu_lib.bar_arrive(_L_BARRIER_ID, num_threads=256)

      if return_residuals:
        if normalize_output and (rescale_threshold != 1.0):
          l_i *= jnp.exp2(m_scale - m_i)
        m_i *= 1 / math.log2(math.e)
        for residual, gmem_ref in zip((m_i, l_i), residual_gmems):
          gmem_ref.at[hi, qs].set(residual.astype(gmem_ref.dtype))

    @pl.when(wg == _SCALE_WG)
    def scale_wg():
      plgpu.set_max_registers(160, action="decrease")

      @pl.when(ub > lb)
      def release_out_scaled_barriers():
        for i in range(num_tma_splits):
          plgpu.barrier_arrive(out_scaled_barrier.at[i])

      def two_in_flight(iterable):
        for a, _ in itertools.pairwise(itertools.chain(iterable, [None])):  # pytype: disable=wrong-arg-types
          yield a

      @pl.loop(lb + 1, ub)
      def kv_loop(ki):
        slot = lax.rem(ki - lb, softmax_slots)

        plgpu.barrier_wait(pv_mma_barrier)
        tile_d = 32

        def load_acc_tiles():
          for d_base in range(0, head_dim_out, tile_d):
            ds = pl.ds(d_base, min(tile_d, head_dim_out - d_base))
            yield ds, plgpu.async_load_tmem(acc_tmem.at[:, ds], layout=_TMEM)

        acc_tiles = two_in_flight(load_acc_tiles())
        ds, acc = next(acc_tiles)
        mgpu_lib.bar_sync(slot + _ALPHA_BARRIER_OFFSET, num_threads=256)
        alpha = plgpu.load(alpha_smem, slot, layout=_TMEM_ROW)
        needs_rescale = (
            (rescale_threshold == 1.0)
            | (alpha < rescale_threshold)
            | ((not normalize_output) & (ki == ub - 1))
        )

        per_warp_type = plgpu.ShapeDtypeStruct((), jnp.bool_, _WG_SPLAT)

        @plgpu.inline_mgpu(arg_types=(_TMEM_ROW,), return_type=per_warp_type)
        def warp_any(_, needs_rescale):
          thread_val = functools.reduce(arith.ori, needs_rescale.registers.flat)
          thread_val = vector.extract(thread_val, [], [0])
          i1 = ir.IntegerType.get_signless(1)
          i32 = ir.IntegerType.get_signless(32)
          mask = arith.constant(i32, 0xFFFFFFFF)
          warp_val = nvvm.vote_sync(i1, mask, thread_val, "any")
          return mgpu.FragmentedArray.splat(warp_val, (), is_signed=False)

        def rescale_acc(ds=ds, acc=acc):
          for i in range(num_tma_splits):
            for _ in range(0, head_dim_out // num_tma_splits, tile_d):
              acc = jnp.where(
                  lax.broadcast_in_dim(needs_rescale, acc.shape, [0]),
                  acc * lax.broadcast_in_dim(alpha, acc.shape, [0]),
                  acc,
              )
              plgpu.async_store_tmem(acc_tmem.at[:, ds], acc)
              try:
                ds, acc = next(acc_tiles)
              except StopIteration:
                break

            plgpu.commit_tmem()
            plgpu.barrier_arrive(out_scaled_barrier.at[i])

        def no_rescale():
          for i in range(num_tma_splits):
            plgpu.barrier_arrive(out_scaled_barrier.at[i])
          for _ in range(num_tma_splits):
            mgpu_lib.warpgroup_barrier()  # To match barrier in `commit_tmem`.

        with jax.named_scope("rescale_acc"):
          # If none of the threads in the warp need to rescale, then we can skip
          # loading the accumulator from tmem entirely as the warps operate
          # independently.
          rescale_warp = (rescale_threshold == 1.0) or warp_any(needs_rescale)
          lax.cond(rescale_warp, rescale_acc, no_rescale)

      with jax.named_scope("epilogue"):
        num_d_tiles = head_dim_out // epi_tile_d
        o_gmem_ = o_gmem.at[qs, hi]

        def write_acc():
          if normalize_output:
            mgpu_lib.bar_sync(_L_BARRIER_ID, num_threads=256)
            l_i = plgpu.load(li_smem, (), layout=_TCGEN05_ROW)
            l_rcp = 1.0 / (l_i + float(jnp.finfo(jnp.float32).tiny))
            l_rcp = lax.broadcast_in_dim(l_rcp, acc_tmem.shape, [0])
          else:
            l_rcp = lax.broadcast_in_dim(1.0, acc_tmem.shape, [])
          plgpu.barrier_wait(pv_mma_barrier)

          def load_acc_tiles():
            for di in range(num_d_tiles):
              si = lax.rem(di, num_epi_slots)
              ds = slice(di * epi_tile_d, (di + 1) * epi_tile_d)
              acc = plgpu.async_load_tmem(acc_tmem.at[:, ds], layout=_TCGEN05)
              yield ds, si, acc

          acc_tiles = two_in_flight(load_acc_tiles())
          for i, (ds, si, acc) in enumerate(acc_tiles):
            acc = (acc * l_rcp[:, ds]).astype(o_smem.dtype)
            if i >= num_epi_slots:
              plgpu.wait_smem_to_gmem(num_epi_slots - 1, wait_read_only=True)
            o_smem[si] = acc
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_smem.at[si], o_gmem_.at[:, ds])

        def write_zeros():
          if return_residuals:
            m_gmem, l_gmem = residual_gmems
            m_gmem[...] = jnp.full_like(m_gmem, -jnp.inf)
            l_gmem[...] = jnp.zeros_like(l_gmem)

          o_smem[0] = plgpu.layout_cast(jnp.zeros_like(o_smem.at[0]), _TCGEN05)
          plgpu.commit_smem()

          for di in range(num_d_tiles):
            ds = slice(di * epi_tile_d, (di + 1) * epi_tile_d)
            plgpu.copy_smem_to_gmem(
                o_smem.at[0], o_gmem_.at[:, ds], commit_group=False
            )
          plgpu.commit_smem_to_gmem_group()

        lax.cond(ub > lb, write_acc, write_zeros)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  def pre_reduce_k_range_per_qtile(range_ref):
    if range_ref is None:
      return None

    def pad_reduce(pad_value: int):
      k_range_ = shape_lib.pad_to_next_multiple_of(
          range_ref, tile_q, -1, pad_value
      )
      return shape_lib.einshape("...(bq)->...bq", q=tile_q)(k_range_)

    return (jnp.min(pad_reduce(kv_seq_len), -1), jnp.max(pad_reduce(0), -1))

  k_start_minmax = pre_reduce_k_range_per_qtile(k_start)
  k_end_minmax = pre_reduce_k_range_per_qtile(k_end)

  out_shape = [jax.ShapeDtypeStruct((*q.shape[:-1], head_dim_out), out_dtype)]
  if return_residuals:
    residuals_shape = (num_q_heads, pl.cdiv(q_seq_len, tile_q) * tile_q)
    out_shape += [jax.ShapeDtypeStruct(residuals_shape, jnp.float32)] * 2

  k_block_shape = (
      num_stages,
      num_tma_splits,
      block_kv // 2 if collective else block_kv,
      head_dim // num_tma_splits,
  )
  v_block_shape = (
      num_stages,
      num_tma_splits,
      block_kv,
      head_dim_out // num_tma_splits // (2 if collective else 1),
  )

  tiled_smem = mgpu_lib.tiled_swizzled_smem
  q_scratch = tiled_smem((block_q, head_dim), q.dtype, "q")
  k_scratch = tiled_smem(k_block_shape, k.dtype, "k")
  v_scratch = tiled_smem(v_block_shape, k.dtype, "v")
  o_scratch = tiled_smem((num_epi_slots, block_q, epi_tile_d), out_dtype, "o")
  if normalize_output:
    li_scratch = plgpu.SMEM((block_q,), jnp.float32)
  else:
    li_scratch = None

  if bias is None or bias.shape[-2] == 1 or bias.shape[-1] == 1:
    bias_scratch = None
  else:
    bias_scratch = tiled_smem((block_q, block_kv), bias.dtype, "bias")

  kv_barrier = plgpu.Barrier(
      num_barriers=num_stages, num_arrivals=num_tma_splits
  )
  kv_consumed_barrier = plgpu.Barrier(
      num_barriers=num_stages * num_tma_splits, orders_tensor_core=True
  )

  if use_2d_bool_mask:
    mask_scratch = plgpu.SMEM(
        (block_q, max(mask_block_kv, min_mask_cols)), _MASK_PACK_DTYPE
    )
  else:
    mask_scratch = None

  def tmem(shape, dtype=jnp.float32, **kwargs):
    return plgpu.TMEM(shape, dtype, collective=collective, **kwargs)

  def maybe_cluster_barrier(**kwargs):
    if collective:
      return plgpu.ClusterBarrier(collective_axes=(collective_axis,), **kwargs)
    return plgpu.Barrier(**kwargs)

  scratch_shapes = dict(
      qo_smem_union=plgpu.RefUnion(q_scratch, o_scratch),
      k_smem=k_scratch,
      v_smem=v_scratch,
      mask_smem=mask_scratch,
      alpha_smem=plgpu.SMEM((softmax_slots, block_q), jnp.float32),
      li_smem=li_scratch,
      qk_acc_tmem=tmem((block_q, block_kv)),
      p_tmem=tmem((block_q, block_kv * softmax_slots), v.dtype, packed=True),
      acc_tmem=tmem((block_q, head_dim_out)),
      bias_smem=bias_scratch,
      q_barrier=plgpu.Barrier(),
      k_barrier=kv_barrier,
      v_barrier=kv_barrier,
      bias_barrier=None if bias_scratch is None else plgpu.Barrier(),
      bias_consumed_barrier=None if bias_scratch is None else plgpu.Barrier(),
      mask_barrier=None if mask_scratch is None else plgpu.Barrier(),
      mask_consumed_barrier=None if mask_scratch is None else plgpu.Barrier(),
      qk_mma_barrier=plgpu.Barrier(orders_tensor_core=True),
      k_consumed_barrier=kv_consumed_barrier,
      qk_consumed_barrier=maybe_cluster_barrier(),
      pv_mma_barrier=plgpu.Barrier(orders_tensor_core=True),
      v_consumed_barrier=kv_consumed_barrier,
      p_produced_barrier=maybe_cluster_barrier(num_barriers=2),
      out_scaled_barrier=maybe_cluster_barrier(num_barriers=num_tma_splits),
  )

  profile = False
  compiler_params = plgpu.CompilerParams(
      approx_math=True,
      unsafe_no_auto_barriers=True,
      profile_space=128 if profile else 0,
      profile_dir="sponge" if profile else "",
  )
  out, *residuals = plgpu.kernel(
      kernel,
      out_shape=out_shape,
      scratch_shapes=scratch_shapes,
      grid=(num_q_heads, num_q_tiles),
      grid_names=("heads", "q_tiles"),
      num_threads=3,
      thread_name="wg",
      cluster=(1 + collective,),
      cluster_names=("cluster",),
      compiler_params=compiler_params,
  )(q, k, v, bias, mask, k_start, k_end, k_start_minmax, k_end_minmax)

  residuals = tuple(res[..., :q_seq_len] for res in residuals)
  return (out[..., :orig_head_dim_out], residuals if residuals else None)
