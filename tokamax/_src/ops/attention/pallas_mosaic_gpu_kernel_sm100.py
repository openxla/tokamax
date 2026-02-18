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
import math

import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
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
_DEFAULT_MASK_VALUE = -1e30

_MMA_TMA_WG = 0
_SOFTMAX_WG = 1
_SCALE_WG = 2
_MMA_WARP = 0
_TMA_LOAD_QK_WARP = 1
_TMA_LOAD_V_WARP = 2
_TMA_LOAD_MASK_WARP = 3

_load_bcast = common.load_bcast


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=dict(extra="forbid")
)  # pytype: disable=wrong-keyword-args
class Config(common.ConfigBase):
  """Configuration parameters for Pallas-Mosaic-GPU kernels on SM100 GPUs.

  Attributes:
    block_d: Block size along head_dim for updating accumulator.
    num_tma_splits: Number of chunks to load each K/V - helpful to better hide
      GMEM load latences as we can notify TMA warp after part of the mma, thus
      giving more time to TMA loads.
    collective: if True - 2 CTA MMA will be run with M=256, N=128
  """

  num_tma_splits: pydantic.PositiveInt = 2
  collective: pydantic.StrictBool = True


def get_heuristics_config(ba: op.BoundArguments) -> Config:
  """Returns a heuristic configuration for flash attention on SM100 GPUs."""
  q, _, v, *_ = ba.args
  *batch_size, q_seq_len, q_heads, head_dim = q.shape
  head_dim = pl.cdiv(max(head_dim, v.shape[-1]), 64) * 64
  batch_size = math.prod(batch_size)
  kv_seq_len = v.shape[-3]
  num_tma_splits = 2 if head_dim == 256 else 1
  collective = True
  cluster_size = 1 + int(collective)
  num_stages = max(256 // head_dim, 1) * cluster_size
  block_q = 256 if collective else 128
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
    use_base2: bool,
    use_stable_softmax: bool,
    config: Config,
) -> tuple[Float[Array, "T H d"], Residuals | None]:
  """SM100 Pallas Mosaic GPU Flash Attention."""

  if bias is not None:
    raise NotImplementedError("Bias is not supported on sm100.")

  if not use_stable_softmax:
    raise NotImplementedError("Unstable softmax not supported on sm100.")

  if out_dtype != q.dtype:
    # TODO: Support other out_dtypes.
    raise NotImplementedError(f"{out_dtype=} != {q.dtype=} unsupported.")

  orig_q_seq_len, num_q_heads, _ = q.shape
  dtype = q.dtype

  kv_seq_len, num_kv_heads, orig_head_dim_out = v.shape
  if kv_seq_len % config.block_kv:
    raise ValueError(f"{kv_seq_len=} must be a multiple of {config.block_kv=}")
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(
        f"Only f16 and bf16 are supported, got dtype: {dtype}"
    )

  pad_head_dim = lambda x: shape_lib.pad_to_next_multiple_of(x, 64, -1)
  q, k, v = map(pad_head_dim, (q, k, v))
  q = shape_lib.pad_to_next_multiple_of(q, 8, -3)
  q_seq_len, _, head_dim = q.shape
  head_dim_out = v.shape[-1]

  if mask is None:
    apply_bool_mask = bcast_mask_q = bcast_mask_k = False
  else:
    apply_bool_mask = True
    bcast_mask_q = mask.shape[-2] == 1
    bcast_mask_k = mask.shape[-1] == 1
    mask = mask.astype(jnp.int8)

  use_2d_bool_mask = apply_bool_mask and not (bcast_mask_k or bcast_mask_q)

  tile_q, block_kv = config.block_q, config.block_kv
  num_q_tiles = pl.cdiv(q_seq_len, tile_q)
  num_stages = config.num_stages
  num_tma_splits = config.num_tma_splits if head_dim >= 128 else 1
  collective = config.collective
  block_q = tile_q // 2 if collective else tile_q
  collective_axis = "x" if collective else None
  softmax_slots = 2

  def kernel(*refs, scoped):
    smem_buffers, buffer_barriers = scoped
    (
        (q_smem, o_smem),
        k_smem,
        v_smem,
        p_tmem,
        mask_smem,
        alpha_smem,
        li_smem,
        acc_tmem,
        qk_acc_tmem,
    ) = smem_buffers
    (
        q_gmem,
        k_gmem,
        v_gmem,
        mask_gmem,
        k_start_gmem,
        k_end_gmem,
        k_start_minmax_gmems,
        k_end_minmax_gmems,
        out_gmem,
        *residual_gmems,
    ) = refs

    (
        q_barrier,
        k_barrier,
        v_barrier,
        mask_produced_barrier,
        mask_consumed_barrier,
        # Q@K
        qk_mma_barrier,
        k_consumed_barrier,
        qk_consumed_barrier,
        # P@V
        pv_mma_barrier,
        v_consumed_barrier,
        p_produced_barrier,
        p_consumed_barrier,
        alpha_produced_barrier,
        out_scaled_barrier,
    ) = buffer_barriers

    qi = lax.axis_index("q_tiles")
    hi = lax.axis_index("heads")
    wg = lax.axis_index("wg")
    cluster_idx = lax.axis_index("x")

    q_base_cluster = qi * tile_q
    q_base = q_base_cluster + cluster_idx * block_q
    qs = pl.ds(q_base, block_q)

    use_k_ranges = k_start_gmem is not None or k_end_gmem is not None

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

        def tma_load_kv(gmem, smem, barrier, partitioned_axis, ki, split_idx):
          kv_head = lax.div(hi, q_heads_per_kv_head)
          si = lax.rem(ki - lb, num_stages)
          block_d = gmem.shape[-1] // num_tma_splits
          ds = pl.ds(split_idx * block_d, block_d)
          plgpu.copy_gmem_to_smem(
              gmem.at[pl.ds(ki * block_kv, block_kv), kv_head, ds],
              smem.at[si, split_idx],
              barrier=barrier.at[si],
              partitioned_axis=partitioned_axis if collective else None,
              collective_axes="x" if collective else None,
          )

        def tma_load_kv_warp(
            gmem, smem, barrier, consumed_barrier, partitioned_axis
        ):
          tma_load = functools.partial(
              tma_load_kv, gmem, smem, barrier, partitioned_axis
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
          plgpu.copy_gmem_to_smem(
              q_gmem.at[pl.ds(q_base_cluster, tile_q), hi],
              q_smem,
              barrier=q_barrier,
              partitioned_axis=0 if collective else None,
              collective_axes=(collective_axis,) if collective else None,
          )
          tma_load_kv_warp(
              k_gmem, k_smem, k_barrier, k_consumed_barrier, partitioned_axis=0
          )

        @pl.when(warp_id == _TMA_LOAD_V_WARP)
        def tma_load_v_warp():
          tma_load_kv_warp(
              v_gmem, v_smem, v_barrier, v_consumed_barrier, partitioned_axis=1
          )

        if use_2d_bool_mask:

          @pl.when(warp_id == _TMA_LOAD_MASK_WARP)
          def tma_load_mask_warp():

            @pl.loop(lb, ub)
            def kv_loop(ki):
              hi_ = 0 if mask_gmem.shape[-3] == 1 else hi
              ks = pl.ds(ki * block_kv, block_kv)
              plgpu.copy_gmem_to_smem(
                  mask_gmem.at[hi_, qs, ks], mask_smem, mask_produced_barrier
              )
              plgpu.barrier_wait(mask_consumed_barrier)

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

            plgpu.tcgen05_commit_arrive(
                qk_mma_barrier, collective_axis=collective_axis
            )

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

            plgpu.tcgen05_commit_arrive(
                pv_mma_barrier, collective_axis=collective_axis
            )
            plgpu.tcgen05_commit_arrive(
                p_consumed_barrier.at[slot], collective_axis=collective_axis
            )

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

      def need_apply_k_range_mask(ki):
        need_apply = False
        if not use_k_ranges:
          return need_apply
        k_base = ki * block_kv
        if k_end is not None:
          need_apply = jnp.logical_or(need_apply, k_base + block_kv > k_end_min)
        if k_start is not None:
          need_apply = jnp.logical_or(need_apply, k_base < k_start_max)
        return need_apply

      def compute_qk_mask(ki, do_causal):
        acc_shape = (block_q, block_kv)
        if not (do_causal or apply_bool_mask or use_k_ranges):
          # not mask needed
          return None
        iota = lambda d: plgpu.broadcasted_iota(
            jnp.int32, acc_shape, dimension=d, layout=_TMEM
        )
        mask = plgpu.layout_cast(jnp.ones(acc_shape, dtype=jnp.bool_), _TMEM)

        if do_causal:
          mask &= (iota(0) + q_base) >= (iota(1) + ki * block_kv)
        if apply_bool_mask:
          if use_2d_bool_mask:
            plgpu.barrier_wait(mask_produced_barrier)
            mask &= plgpu.load(mask_smem, (), layout=_TMEM, optimized=False)
            plgpu.barrier_arrive(mask_consumed_barrier)
          else:
            hi_ = 0 if mask_gmem.shape[-3] == 1 else hi
            if bcast_mask_q:
              idx = (hi_, 0, pl.ds(ki * block_kv, block_kv))
              layout = _TMEM_COL
            else:
              idx = (hi_, qs, 0)
              layout = _TMEM_ROW

            mask_vector = plgpu.load(
                mask_gmem, idx, layout=layout, optimized=False
            )
            bc_dim = 1 if bcast_mask_q else 0
            # TODO: we need to handle Q masks differently
            # broadcasting & using them the way it is done is extremely slow
            mask &= plgpu.layout_cast(
                lax.broadcast_in_dim(mask_vector, acc_shape, (bc_dim,)),
                _TMEM,
            )

        if use_k_ranges:

          def _krange_mask(mask):
            bc_range = lambda x: lax.broadcast_in_dim(x, acc_shape, (0,))
            block_kv_iota = iota(1) + (ki * block_kv)

            if k_start_gmem is not None:
              mask &= bc_range(k_start) <= block_kv_iota
            if k_end_gmem is not None:
              mask &= bc_range(k_end) > block_kv_iota
            return mask

          mask = lax.cond(
              need_apply_k_range_mask(ki),
              lambda: _krange_mask(mask),
              lambda: mask,
          )
        return mask

      def maybe_apply_mask(s, scale, ki, *, do_causal):
        if not (apply_bool_mask or use_k_ranges or do_causal):
          return s, scale
        need_apply_mask = jnp.logical_or(
            do_causal or apply_bool_mask, need_apply_k_range_mask(ki)
        )
        compute_mask_fn = lambda: jnp.where(
            compute_qk_mask(ki, do_causal), 0, _DEFAULT_MASK_VALUE
        )
        return lax.cond(
            need_apply_mask,
            lambda: (s * scale + compute_mask_fn(), 1.0),
            lambda: (s, scale),
        )

      def kv_loop(ki, carry, *, do_causal=False):
        m_i, l_i = carry
        is_last_step = ki == ub - 1
        si = lax.rem(ki - lb, 2)
        with jax.named_scope("Q@K"):
          plgpu.barrier_wait(qk_mma_barrier)
        with jax.named_scope("load_qk"):
          s = plgpu.async_load_tmem(qk_acc_tmem, layout=_TMEM)
          scale = logits_scale
          plgpu.wait_load_tmem()
          plgpu.barrier_arrive(qk_consumed_barrier)

        if logits_soft_cap is not None:
          s, scale = jnp.tanh(s * (scale / logits_soft_cap)), logits_soft_cap

        with jax.named_scope("softmax"):
          exp = jnp.exp2 if use_base2 else jnp.exp
          if use_base2:
            scale *= math.log2(math.e)
          s, scale = maybe_apply_mask(s, scale, ki, do_causal=do_causal)
          m_ij = jnp.maximum(m_i, s.max(axis=1) * scale)
          with jax.named_scope("exp(SFU)"):
            alpha = exp(m_i - m_ij)

          @pl.when(ki > lb)
          def write_alpha_to_smem():
            alpha_smem.at[si][...] = alpha
            plgpu.barrier_arrive(alpha_produced_barrier.at[si])

          m_i = m_ij
          with jax.named_scope("exp(SFU)"):
            p = exp(s * scale - lax.broadcast_in_dim(m_ij, s.shape, [0]))
          l_i *= alpha
          l_i += p.sum(axis=1)
          p16 = p.astype(p_tmem.dtype)

          @pl.when(is_last_step)
          def write_l_to_smem():
            li_smem[...] = l_i

          @pl.when(ki > lb + 1)
          def wait_for_p_consumed():
            with jax.named_scope("wait p_consumed"):
              plgpu.barrier_wait(p_consumed_barrier.at[si])

          with jax.named_scope("write qk_tmem"):
            plgpu.async_store_tmem(
                p_tmem.at[:, pl.ds(si * block_kv, block_kv)], p16
            )
            plgpu.commit_tmem()
        plgpu.barrier_arrive(p_produced_barrier.at[si])
        return m_i, l_i

      # prologue
      plgpu.barrier_arrive(qk_consumed_barrier)

      # in 2CTA we have non square blocks hence we may need to process
      # M//N steps with a mask, for M=256, N=128 this means 2 steps
      causal_blocks = int(is_causal) * (tile_q // block_kv)
      m_i, l_i = lax.fori_loop(lb, ub - causal_blocks, kv_loop, (m_i, l_i))

      if is_causal:
        m_i, l_i = lax.fori_loop(
            ub - causal_blocks,
            ub,
            functools.partial(kv_loop, do_causal=True),
            (m_i, l_i),
        )

      if return_residuals:
        if use_base2:
          m_i *= 1 / math.log2(math.e)
        for residual, gmem_ref in zip((m_i, l_i), residual_gmems):
          gmem_ref.at[hi, qs].set(residual.astype(gmem_ref.dtype))

      l_i = plgpu.load(li_smem, (), layout=_TCGEN05_ROW, optimized=True)
      l_i += float(jnp.finfo(jnp.float32).tiny)

      with jax.named_scope("wait mma"):
        slot = lax.rem(ub - 1 - lb, softmax_slots)
        plgpu.barrier_wait(p_consumed_barrier.at[slot])

      # epilogue for writing GMEM
      with jax.named_scope("TMEM -> SMEM"):
        acc = plgpu.async_load_tmem(acc_tmem, layout=_TCGEN05)
      with jax.named_scope("SMEM -> GMEM"):
        if normalize_output:
          acc *= lax.broadcast_in_dim(1.0 / l_i, acc.shape, [0])
        o_smem[...] = acc.astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, out_gmem.at[qs, hi])
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    @pl.when((wg == _SCALE_WG) & (ub > lb))
    def scale_wg():
      plgpu.set_max_registers(160, action="decrease")

      for i in range(num_tma_splits):
        plgpu.barrier_arrive(out_scaled_barrier.at[i])

      @pl.loop(lb + 1, ub)
      def kv_loop(ki):
        slot = lax.rem(ki - lb, softmax_slots)

        plgpu.barrier_wait(pv_mma_barrier)
        block_d = 32
        ds = pl.ds(0, block_d)
        acc = acc_next = plgpu.async_load_tmem(acc_tmem.at[:, ds], layout=_TMEM)
        plgpu.barrier_wait(alpha_produced_barrier.at[slot])
        alpha = plgpu.load(alpha_smem, slot, layout=_TMEM_ROW)

        with jax.named_scope("scale_acc"):

          for i in range(num_tma_splits):
            for _ in range(0, head_dim_out // num_tma_splits, block_d):
              ds_next = pl.ds(ds.start + block_d, block_d)
              if ds_next.start < head_dim_out:
                acc_next = plgpu.async_load_tmem(
                    acc_tmem.at[:, ds_next], layout=_TMEM
                )

              acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
              plgpu.async_store_tmem(acc_tmem.at[:, ds], acc)
              ds = ds_next
              acc = acc_next

            plgpu.commit_tmem()
            plgpu.barrier_arrive(out_scaled_barrier.at[i])

      plgpu.barrier_wait(pv_mma_barrier)

  def entry(*refs):

    def tiled_smem(shape, dtype):
      transforms = common.tile_swizzle_transforms(shape, dtype)
      return plgpu.SMEM(shape, dtype, transforms=transforms)

    q_scratch = tiled_smem((block_q, head_dim), q.dtype)
    k_scratch = tiled_smem(
        (
            num_stages,
            num_tma_splits,
            block_kv // 2 if collective else block_kv,
            head_dim // num_tma_splits,
        ),
        k.dtype,
    )
    v_scratch = tiled_smem(
        (
            num_stages,
            num_tma_splits,
            block_kv,
            head_dim_out // num_tma_splits // (2 if collective else 1),
        ),
        k.dtype,
    )
    o_scratch = tiled_smem((block_q, head_dim_out), q.dtype)
    p_scratch = plgpu.TMEM(
        (block_q, block_kv * softmax_slots),
        v.dtype,
        packed=True,
        collective=collective,
    )
    acc_scratch = plgpu.TMEM(
        (block_q, head_dim_out), jnp.float32, collective=collective
    )
    qk_acc_scratch = plgpu.TMEM(
        (block_q, block_kv), jnp.float32, collective=collective
    )
    alpha_scratch = plgpu.SMEM((softmax_slots, block_q), jnp.float32)
    li_scratch = plgpu.SMEM((block_q,), jnp.float32)
    mask_scratch = tiled_smem((block_q, block_kv), jnp.int8)

    # TMA barriers
    k_barrier = v_barrier = plgpu.Barrier(
        num_barriers=num_stages, num_arrivals=num_tma_splits
    )
    q_barrier = plgpu.Barrier()
    mask_produced_barrier = plgpu.Barrier()
    mask_consumed_barrier = plgpu.Barrier()
    # Q@K
    qk_mma_barrier = plgpu.Barrier(orders_tensor_core=True)
    k_consumed_barrier = plgpu.Barrier(
        num_barriers=num_stages * num_tma_splits, orders_tensor_core=True
    )
    # P@V
    pv_mma_barrier = plgpu.Barrier(orders_tensor_core=True)
    v_consumed_barrier = plgpu.Barrier(
        num_barriers=num_stages * num_tma_splits, orders_tensor_core=True
    )
    if collective:
      p_produced_barrier = plgpu.ClusterBarrier(
          num_barriers=2, collective_axes=(collective_axis,)
      )
      out_scaled_barrier = plgpu.ClusterBarrier(
          num_barriers=num_tma_splits, collective_axes=(collective_axis,)
      )
      qk_consumed_barrier = plgpu.ClusterBarrier(
          collective_axes=(collective_axis,)
      )
    else:
      p_produced_barrier = plgpu.Barrier(num_barriers=2)
      out_scaled_barrier = plgpu.Barrier(num_barriers=num_tma_splits)
      qk_consumed_barrier = plgpu.Barrier()

    alpha_produced_barrier = plgpu.Barrier(num_barriers=2)
    p_consumed_barrier = plgpu.Barrier(num_barriers=2, orders_tensor_core=True)

    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        (
            plgpu.RefUnion(q_scratch, o_scratch),
            k_scratch,
            v_scratch,
            p_scratch,
            mask_scratch,
            alpha_scratch,
            li_scratch,
            acc_scratch,
            qk_acc_scratch,
        ),
        (
            q_barrier,
            k_barrier,
            v_barrier,
            mask_produced_barrier,
            mask_consumed_barrier,
            # Q@K
            qk_mma_barrier,
            k_consumed_barrier,
            qk_consumed_barrier,
            # P@V
            pv_mma_barrier,
            v_consumed_barrier,
            p_produced_barrier,
            p_consumed_barrier,
            alpha_produced_barrier,
            out_scaled_barrier,
        ),
        collective_axes="wg",
    )

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

  out_shape = [jax.ShapeDtypeStruct((*q.shape[:-1], head_dim_out), q.dtype)]
  if return_residuals:
    residuals_shape = (num_q_heads, pl.cdiv(q_seq_len, tile_q) * tile_q)
    out_shape += [jax.ShapeDtypeStruct(residuals_shape, jnp.float32)] * 2

  profile = False
  compiler_params = plgpu.CompilerParams(
      approx_math=True,
      unsafe_no_auto_barriers=True,
      profile_space=128 if profile else 0,
      profile_dir="sponge" if profile else "",
  )
  out, *residuals = plgpu.kernel(
      entry,
      out_shape=out_shape,
      grid=(num_q_heads, num_q_tiles),
      grid_names=("heads", "q_tiles"),
      num_threads=3,
      thread_name="wg",
      cluster=(1 + collective,),
      cluster_names=("x",),
      compiler_params=compiler_params,
  )(q, k, v, mask, k_start, k_end, k_start_minmax, k_end_minmax)

  residuals = tuple(res[..., :orig_q_seq_len] for res in residuals)
  out = out[..., :orig_q_seq_len, :, :orig_head_dim_out]
  return (out, residuals if residuals else None)
