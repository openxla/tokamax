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
from typing import TypeAlias

import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common


DotPrecisionLike = lax.Precision | lax.DotAlgorithmPreset
L: TypeAlias = plgpu.Layout
PagingInfo = base.PagingInfo
QArray = base.QArray
Residuals = base.Residuals


_TMEM = L.TCGEN05_TMEM_NATIVE
_TMEM_COL = L.TCGEN05_TMEM_NATIVE.reduce(0)
_TMEM_ROW = L.TCGEN05_TMEM_NATIVE.reduce(1)
_TCGEN05_ROW = L.TCGEN05.reduce(1)
_DEFAULT_MASK_VALUE = -1e30

_MMA_WG = 0
_SOFTMAX_WG = 1
_SCALE_WG = 2
_MMA_WARP = 0
_QK_MEMORY_WARP = 1
_PV_MEMORY_WARP = 2
_MASK_MEMORY_WARP = 3


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

  # TODO: Relax block size constraints to multiple of 32.
  block_d: pydantic.conint(multiple_of=8, gt=0) = 128
  num_tma_splits: pydantic.PositiveInt = 2
  collective: pydantic.StrictBool = True


def get_heuristics_config(ba: op.BoundArguments) -> Config:
  """Returns a heuristic configuration for flash attention on SM100 GPUs."""
  q, _, v, *_ = ba.args
  head_dim = max(q.shape[-1], v.shape[-1])
  num_tma_splits = 2 if head_dim == 256 else 1
  collective = True
  num_stages = max(256 // head_dim, 1) * (1 + int(collective))
  return Config(
      block_q=256 if collective else 128,
      block_kv=128,
      block_d=128,
      collective=collective,
      num_stages=num_stages,
      num_tma_splits=num_tma_splits,
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

  q_seq_len, num_q_heads, head_dim = q.shape
  dtype = q.dtype

  if q_seq_len % 8 != 0:
    raise NotImplementedError(f"{q_seq_len=} must be a multiple of 8")

  kv_seq_len, num_kv_heads, _ = k.shape
  orig_head_dim_out = v.shape[-1]
  if kv_seq_len % config.block_kv:
    raise ValueError(f"{kv_seq_len=} must be a multiple of {config.block_kv=}")
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(
        f"Only f16 and bf16 are supported, got dtype: {dtype}"
    )

  # TODO: Handle different head_dims without padding.
  head_dim = head_dim_out = pl.cdiv(max(head_dim, orig_head_dim_out), 64) * 64
  pad_head_dim = lambda x: shape_lib.pad_dim_to(x, head_dim, -1)
  q, k, v = map(pad_head_dim, (q, k, v))

  logits_map = lambda x: x

  if logits_scale != 1.0:
    logits_map = lambda x, logits_map=logits_map: logits_scale * logits_map(x)

  if logits_soft_cap is not None:
    logits_map = lambda x, logits_map=logits_map: logits_soft_cap * jnp.tanh(
        logits_map(x) / logits_soft_cap
    )

  if mask is None:
    apply_bool_mask = bcast_mask_q = bcast_mask_k = False
  else:
    apply_bool_mask = True
    bcast_mask_q = mask.shape[-2] == 1
    bcast_mask_k = mask.shape[-1] == 1
    mask = mask.astype(jnp.int8)

  use_2d_bool_mask = apply_bool_mask and not (bcast_mask_k or bcast_mask_q)

  tile_q, block_kv, block_d = config.block_q, config.block_kv, config.block_d
  # scale_d should be >= head_dim_out
  block_d = min(block_d, head_dim_out)
  num_q_tiles = pl.cdiv(q_seq_len, tile_q)
  num_stages = config.num_stages
  num_tma_splits = config.num_tma_splits if head_dim >= 128 else 1
  tma_chunk_size = head_dim // num_tma_splits
  collective = config.collective
  block_q = tile_q // 2 if collective else tile_q
  collective_axis = "x" if collective else None
  softmax_slots = 2

  def kernel(*refs, scoped):
    smem_buffers, buffer_barriers = scoped
    (
        q_smem,
        k_smem,
        v_smem,
        qk_tmem,
        mask_smem,
        alpha_smem,
        li_smem,
        acc_tmem,
        acc_qk_tmem,
    ) = smem_buffers
    (
        q_ref,
        k_ref,
        v_ref,
        mask_ref,
        k_start_ref,
        k_end_ref,
        k_start_minmax_refs,
        k_end_minmax_refs,
        out_ref,
        *residual_gmem_refs,
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
    is_lead_block = cluster_idx == 0

    q_base_cluster = qi * tile_q
    q_base = q_base_cluster + (~is_lead_block) * block_q
    qs = pl.ds(q_base, block_q)

    use_k_ranges = k_start_ref is not None or k_end_ref is not None

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

    if k_start_minmax_refs is None:
      k_start_max = None
    else:
      k_start_min, k_start_max = map(load_k_bound, k_start_minmax_refs)
      lb = lax.max(lb, lax.div(k_start_min, block_kv))

    if k_end_minmax_refs is None:
      k_end_min = None
    else:
      k_end_min, k_end_max = map(load_k_bound, k_end_minmax_refs)
      ub = lax.min(ub, pl.cdiv(k_end_max, block_kv))

    def tma_load_kv(
        ki, gmem_ref, smem_ref, barrier_ref, split_idx, partitioned_axis
    ):
      kv_head = lax.div(hi, q_heads_per_kv_head)
      stage = lax.rem(ki - lb, num_stages)
      tma_chunk = head_dim // num_tma_splits
      ds = pl.ds(split_idx * tma_chunk, tma_chunk)
      plgpu.copy_gmem_to_smem(
          gmem_ref.at[pl.ds(ki * block_kv, block_kv), kv_head, ds],
          smem_ref.at[stage, split_idx],
          barrier=barrier_ref.at[stage],
          partitioned_axis=partitioned_axis if collective else None,
          collective_axes="x" if collective else None,
      )

    tma_load_k = lambda ki, split_idx: tma_load_kv(
        ki, k_ref, k_smem, k_barrier, split_idx, partitioned_axis=0
    )
    tma_load_v = lambda ki, split_idx: tma_load_kv(
        ki, v_ref, v_smem, v_barrier, split_idx, partitioned_axis=1
    )

    def tma_kv_mask(ki):
      hi_ = 0 if mask_ref.shape[-3] == 1 else hi
      plgpu.copy_gmem_to_smem(
          mask_ref.at[hi_, qs, pl.ds(ki * block_kv, block_kv)],
          mask_smem,
          barrier=mask_produced_barrier,
      )

    @pl.when(jnp.logical_and(wg == _MMA_WG, ub > lb))
    def _compute_mma_wg():
      plgpu.set_max_registers(80, action="decrease")

      @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
      def _per_warp():
        warp_id = lax.axis_index("warp")

        @pl.when(warp_id == _QK_MEMORY_WARP)
        def _qk_memory_warp():

          plgpu.copy_gmem_to_smem(
              q_ref.at[pl.ds(q_base_cluster, tile_q), hi],
              q_smem,
              barrier=q_barrier,
              partitioned_axis=0 if collective else None,
              collective_axes=(collective_axis,) if collective else None,
          )

          @pl.loop(lb, lax.min(lb + num_stages, ub))
          def _prefetch(ki):
            @pl.loop(0, num_tma_splits)
            def _(split_idx):
              tma_load_k(ki, split_idx)

          @pl.loop(lb + num_stages, ub)
          def _kv_loop(ki):
            stage = lax.rem(ki - lb, num_stages)

            @pl.loop(0, num_tma_splits)
            def _(split_idx):
              slot = stage * num_tma_splits + split_idx
              plgpu.barrier_wait(k_consumed_barrier.at[slot])
              tma_load_k(ki, split_idx)

        @pl.when(warp_id == _PV_MEMORY_WARP)
        def _pv_memory_warp():

          @pl.loop(lb, lax.min(lb + num_stages, ub))
          def _prefetch(ki):
            @pl.loop(0, num_tma_splits)
            def _(split_idx):
              tma_load_v(ki, split_idx)

          @pl.loop(lb + num_stages, ub)
          def _kv_loop(ki):
            stage = lax.rem(ki - lb, num_stages)

            @pl.loop(0, num_tma_splits)
            def _(split_idx):
              slot = stage * num_tma_splits + split_idx
              plgpu.barrier_wait(v_consumed_barrier.at[slot])
              tma_load_v(ki, split_idx)

        if use_2d_bool_mask:

          @pl.when(warp_id == _MASK_MEMORY_WARP)
          def _mask_memory_warp():

            @pl.loop(lb, ub)
            def _kv_loop(ki):
              tma_kv_mask(ki)
              plgpu.barrier_wait(mask_consumed_barrier)

        @pl.when(jnp.logical_and(warp_id == _MMA_WARP, is_lead_block))
        def _qk_mma_warp():
          plgpu.barrier_wait(q_barrier)

          def compute_qk(ki):
            stage = lax.rem(ki - lb, num_stages)
            with jax.named_scope("wait_k"):
              plgpu.barrier_wait(qk_consumed_barrier)
              plgpu.barrier_wait(k_barrier.at[stage])

            @pl.loop(0, num_tma_splits)
            def _(split_idx):
              barrier_slot = stage * num_tma_splits + split_idx
              ds = pl.ds(split_idx * tma_chunk_size, tma_chunk_size)
              with jax.named_scope("issuing Q@K.T"):
                plgpu.tcgen05_mma(
                    acc_qk_tmem,
                    q_smem.at[:, ds],
                    k_smem.at[stage, split_idx].T,
                    k_consumed_barrier.at[barrier_slot],
                    accumulate=split_idx > 0,
                    collective_axis=collective_axis,
                )

            plgpu.tcgen05_commit_arrive(
                qk_mma_barrier, collective_axis=collective_axis
            )

          def compute_pv(ki):
            stage = lax.rem(ki - lb, num_stages)
            slot = lax.rem(ki - lb, 2)
            with jax.named_scope("wait_v"):
              plgpu.barrier_wait(p_produced_barrier.at[slot])
              plgpu.barrier_wait(v_barrier.at[stage])

            @pl.loop(0, num_tma_splits)
            def _(split_idx):
              barrier_slot = stage * num_tma_splits + split_idx
              ds = pl.ds(split_idx * tma_chunk_size, tma_chunk_size)
              plgpu.barrier_wait(out_scaled_barrier.at[split_idx])
              with jax.named_scope("issuing P@V"):
                plgpu.tcgen05_mma(
                    acc_tmem.at[:, ds],
                    qk_tmem.at[:, pl.ds(slot * block_kv, block_kv)],
                    v_smem.at[stage, split_idx],
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

          def mma_body(ki, _):
            compute_qk(ki + 1)
            compute_pv(ki)

          compute_qk(lb)
          lax.fori_loop(lb, ub - 1, mma_body, None)
          compute_pv(ub - 1)

    @pl.when(jnp.logical_and(wg == _SOFTMAX_WG, ub > lb))
    def _compute_softmax_wg():
      plgpu.set_max_registers(256, action="increase")

      m_i = plgpu.layout_cast(
          jnp.full((block_q,), -jnp.inf, dtype=jnp.float32),
          _TMEM_ROW,
      )
      l_i = plgpu.layout_cast(jnp.zeros_like(m_i), _TMEM_ROW)

      def load_k_range(k_range_ref) -> jax.Array | None:
        if k_range_ref is None:
          return None
        shape = k_range_ref.shape
        hi_ = 0 if shape[-2] == 1 else hi
        singular_q = shape[-1] == 1

        qs_ = 0 if singular_q else qs
        layout = L.WG_SPLAT if singular_q else _TMEM_ROW
        k_range = plgpu.load(
            k_range_ref, (hi_, qs_), layout=layout, optimized=False
        )
        if singular_q:
          k_range = lax.broadcast_in_dim(k_range, (block_q,), ())
          k_range = plgpu.layout_cast(k_range, _TMEM_ROW)
        return k_range

      k_start, k_end = load_k_range(k_start_ref), load_k_range(k_end_ref)

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
            hi_ = 0 if mask_ref.shape[-3] == 1 else hi
            if bcast_mask_q:
              idx = (hi_, 0, pl.ds(ki * block_kv, block_kv))
              layout = _TMEM_COL
            else:
              idx = (hi_, qs, 0)
              layout = _TMEM_ROW

            mask_vector = plgpu.load(
                mask_ref, idx, layout=layout, optimized=False
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

            if k_start_ref is not None:
              mask &= bc_range(k_start) <= block_kv_iota
            if k_end_ref is not None:
              mask &= bc_range(k_end) > block_kv_iota
            return mask

          mask = lax.cond(
              need_apply_k_range_mask(ki),
              lambda: _krange_mask(mask),
              lambda: mask,
          )
        return mask

      def maybe_apply_mask(qk, ki, *, do_causal):
        if not (apply_bool_mask or use_k_ranges or do_causal):
          return qk
        need_apply_mask = jnp.logical_or(
            do_causal or apply_bool_mask, need_apply_k_range_mask(ki)
        )
        compute_mask_fn = lambda: jnp.where(
            compute_qk_mask(ki, do_causal), 0, _DEFAULT_MASK_VALUE
        )
        return lax.cond(
            need_apply_mask, lambda: qk + compute_mask_fn(), lambda: qk
        )

      def kv_loop(ki, carry, *, do_causal=False):
        m_i, l_i = carry
        is_last_step = ki == ub - 1
        si = lax.rem(ki - lb, 2)
        with jax.named_scope("Q@K"):
          plgpu.barrier_wait(qk_mma_barrier)
        with jax.named_scope("load_qk"):
          qk = plgpu.async_load_tmem(acc_qk_tmem, layout=_TMEM)
          plgpu.wait_load_tmem()
          plgpu.barrier_arrive(qk_consumed_barrier)
        with jax.named_scope("softmax"):
          exp = jnp.exp2 if use_base2 else jnp.exp
          qk = logits_map(qk)
          if use_base2:
            qk *= math.log2(math.e)
          qk = maybe_apply_mask(qk, ki, do_causal=do_causal)
          m_ij = jnp.maximum(qk.max(axis=1), m_i)
          with jax.named_scope("exp(SFU)"):
            alpha = exp(m_i - m_ij)

          @pl.when(ki > lb)
          def _():
            alpha_smem.at[si][...] = alpha
            plgpu.barrier_arrive(alpha_produced_barrier.at[si])

          m_i = m_ij
          with jax.named_scope("exp(SFU)"):
            p = exp(qk - lax.broadcast_in_dim(m_ij, (block_q, block_kv), [0]))
          l_i *= alpha
          l_i += p.sum(axis=1)
          p16 = p.astype(qk_tmem.dtype)

          @pl.when(is_last_step)
          def _():
            li_smem[...] = l_i

          @pl.when(ki > lb + 1)
          def _():
            with jax.named_scope("wait p_consumed"):
              plgpu.barrier_wait(p_consumed_barrier.at[si])

          with jax.named_scope("write qk_tmem"):
            plgpu.async_store_tmem(
                qk_tmem.at[:, pl.ds(si * block_kv, block_kv)], p16
            )
            plgpu.commit_tmem()
        plgpu.barrier_arrive(p_produced_barrier.at[si])
        return m_i, l_i

      # prologue
      plgpu.barrier_arrive(qk_consumed_barrier)

      # in 2CTA we have non square blocks hence we may need to process
      # M//N steps with a mask, for M=256, N=128 this means 2 steps
      causal_blocks = int(is_causal) * (tile_q // block_kv)

      m_i, l_i = lax.fori_loop(
          lb,
          ub - causal_blocks,
          kv_loop,
          (m_i, l_i),
      )
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
        for residual, gmem_ref in zip((m_i, l_i), residual_gmem_refs):
          gmem_ref.at[hi, qs].set(residual.astype(gmem_ref.dtype))

      l_i = plgpu.load(li_smem, (), layout=_TCGEN05_ROW, optimized=True)
      l_i += float(jnp.finfo(jnp.float32).tiny)

      with jax.named_scope("wait mma"):
        slot = lax.rem(ub - 1 - lb, softmax_slots)
        plgpu.barrier_wait(p_consumed_barrier.at[slot])

      # epilogue for writing GMEM
      with jax.named_scope("TMEM -> SMEM"):
        acc = plgpu.async_load_tmem(acc_tmem, layout=L.TCGEN05)
        plgpu.wait_load_tmem()
      with jax.named_scope("SMEM -> GMEM"):
        if normalize_output:
          acc *= lax.broadcast_in_dim(1.0 / l_i, acc.shape, [0])
        q_smem[...] = acc.astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(q_smem, out_ref.at[qs, hi])
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    @pl.when(jnp.logical_and(wg == _SCALE_WG, ub > lb))
    def _scale_wg():
      plgpu.set_max_registers(160, action="decrease")

      def kv_loop(ki, _):
        slot = lax.rem(ki - lb, softmax_slots)

        plgpu.barrier_wait(pv_mma_barrier)
        plgpu.barrier_wait(alpha_produced_barrier.at[slot])
        alpha = plgpu.load(alpha_smem, slot, layout=_TMEM_ROW)

        with jax.named_scope("scale_acc"):

          @pl.loop(0, num_tma_splits)
          def _scale_tma_chunk(i):

            @pl.loop(0, tma_chunk_size // block_d)
            def _scale_d(j):
              ds = pl.ds(i * tma_chunk_size + j * block_d, block_d)
              acc_tmem_ref = acc_tmem.at[:, ds]
              updated_acc = plgpu.async_load_tmem(
                  acc_tmem_ref, layout=_TMEM
              ) * lax.broadcast_in_dim(alpha, (block_q, block_d), [0])
              plgpu.async_store_tmem(acc_tmem_ref, updated_acc)

            plgpu.commit_tmem()
            plgpu.barrier_arrive(out_scaled_barrier.at[i])

      for i in range(num_tma_splits):
        plgpu.barrier_arrive(out_scaled_barrier.at[i])

      lax.fori_loop(lb + 1, ub, kv_loop, None)
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
            tma_chunk_size,
        ),
        k.dtype,
    )
    v_scratch = tiled_smem(
        (
            num_stages,
            num_tma_splits,
            block_kv,
            tma_chunk_size // 2 if collective else tma_chunk_size,
        ),
        k.dtype,
    )
    qk_scratch = plgpu.TMEM(
        (block_q, block_kv * softmax_slots),
        q.dtype,
        packed=True,
        collective=collective,
    )
    acc_scratch = plgpu.TMEM(
        (block_q, head_dim), jnp.float32, collective=collective
    )

    acc_qk_scratch = plgpu.TMEM(
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
            q_scratch,
            k_scratch,
            v_scratch,
            qk_scratch,
            mask_scratch,
            alpha_scratch,
            li_scratch,
            acc_scratch,
            acc_qk_scratch,
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

  residuals = tuple(res[..., :q_seq_len] for res in residuals)
  return (out[..., :orig_head_dim_out], residuals if residuals else None)
