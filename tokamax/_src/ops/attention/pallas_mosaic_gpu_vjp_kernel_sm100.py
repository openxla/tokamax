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
"""Flash Attention Pallas-Mosaic-GPU VJP implementation (SM100)."""

# pylint: disable=invalid-name

import math

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import shape as shape_lib
from tokamax._src.ops.attention import base


_TMEM = plgpu.Layout.TCGEN05
_TMEM_COL = plgpu.Layout.TCGEN05.reduce(0)
_TMEM_ROW = plgpu.Layout.TCGEN05.reduce(1)
_WG_SPLAT = plgpu.Layout.WG_SPLAT

# Warpgroup Indices
_LOAD_MMA_WG = 0
_REDUCE_WG = 1
_COMPUTE_WG_1 = 2
_COMPUTE_WG_2 = 3
_NUM_COMPUTE_WGS = 2

# MMA Warpgroup's Warp Indices
_MEMORY_WARP = 0
_MMA_WARP_1 = 1
_MMA_WARP_2 = 2
_MMA_WARP_3 = 3


def _kernel(body, out_shape, **kernel_kwargs):
  """Interface for SM100 attention VJP kernel.

  Unwraps the custom_vmap() operator because tokamax handles vmap
  already via the batched arguments.

  Args:
    body: The kernel body.
    out_shape: The output shapes.
    **kernel_kwargs: Additional kernel arguments.

  Returns:
    The kernel results.

  """

  if singleton_out := not isinstance(out_shape, (tuple, list)):
    out_shape = (out_shape,)

  def wrapped(*kernel_args):
    @pl.run_state
    def stateful(out_refs):
      def _body(*args, **kwargs):
        body(*args, *out_refs, **kwargs)

      plgpu.kernel(
          _body,
          out_shape=(),
          **kernel_kwargs,
      ).fun(*kernel_args)

    out = stateful(
        jax.tree.map(
            lambda s: s
            if isinstance(s, jax.Array)
            else jax.lax.empty(s.shape, s.dtype),
            out_shape,
        )
    )
    return out[0] if singleton_out else out

  return wrapped


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_kv: pydantic.conint(multiple_of=128, ge=128)
  block_q: pydantic.conint(multiple_of=128, ge=128)
  num_stages: pydantic.conint(gt=1) = 2
  num_stages_dq: pydantic.PositiveInt = 1

  def __post_init__(self):
    # SM100 VJP implementation currently relies on block_q and block_kv
    # being at least 128 for indexing logic.
    if self.block_q < 128 or self.block_kv < 128:
      raise ValueError(
          "For SM100 attention VJP, block_q and block_kv must be at least 128."
          f" Got block_q={self.block_q}, block_kv={self.block_kv}."
      )


def get_heuristics_config(ba) -> Config:
  del ba
  return Config(block_q=128, block_kv=128)


def get_autotuning_configs(ba) -> set[Config]:
  del ba
  return {Config(block_q=128, block_kv=128)}


def _ds(start, size):
  if isinstance(start, int):
    return slice(start, start + size)
  else:
    return pl.ds(start, size)


def _broadcast_to_seq(x, dim_q, q_seq_len, dim_k=None, kv_seq_len=None):
  """Broadcasts tensor x to match sequence lengths."""
  if x is None:
    return None
  shape = list(x.shape)
  if shape[dim_q] == 1 and q_seq_len > 1:
    shape[dim_q] = q_seq_len
  if (
      dim_k is not None
      and kv_seq_len is not None
      and shape[dim_k] == 1
      and kv_seq_len > 1
  ):
    shape[dim_k] = kv_seq_len
  if tuple(shape) != x.shape:
    return jnp.broadcast_to(x, shape)
  return x


def _pad(x, axis, block, constant_values=0, broadcastable=False):
  """Pads an array to a multiple of `block` along `axis`.

  Args:
    x: The array to pad.
    axis: The axis or axes along which to pad.
    block: The block size or sizes to pad to a multiple of.
    constant_values: The value to use for padding.
    broadcastable: If True, do not pad if the dimension size is 1.

  Returns:
    The padded array.
  """
  if x is None:
    return None
  if axis is None:
    return x

  if isinstance(axis, int):
    axis = (axis,)
    block = (block,)

  padding = [(0, 0)] * x.ndim
  should_pad = False
  for ax, blk in zip(axis, block):
    if broadcastable and x.shape[ax] == 1:
      continue
    if x.shape[ax] % blk == 0:
      continue
    pad_len = blk - (x.shape[ax] % blk)
    padding[ax] = (0, pad_len)
    should_pad = True

  if not should_pad:
    return x

  return jnp.pad(x, padding, constant_values=constant_values)


@jaxtyping.jaxtyped
def flash_attention_vjp_kernel(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*B t h D"],
    v: Float[Array, "*B t h d"],
    residuals: base.Residuals,
    out: Float[Array, "*B T H d"],
    dout: Float[Array, "*B T H d"],
    bias: Float[Array, "*#B #H #T #t"] | None,
    mask: Bool[Array, "*#B #H #T #t"] | None,
    k_start: Int[Array, "*#B #H #T"] | None,
    k_end: Int[Array, "*#B #H #T"] | None,
    *,
    logits_scale: float,
    logits_soft_cap: float | None,
    use_base2: bool,
    is_causal: bool,
    ds_dtype: jax.typing.DTypeLike | None,
    config: Config,
) -> tuple[
    Float[Array, "*B T H D"],  # dq
    Float[Array, "*B t h D"],  # dk
    Float[Array, "*B t h d"],  # dv
    Float[Array, "*#B #H #T #t"] | None,  # ds
]:
  # The key steps, computed block-wise over KV, are:
  #
  # Notations:
  #   K' = K * scale
  #   Delta = rowsum(O * dO)
  #
  # Algorithm using transposed intermediates for GPU efficiency:
  # 1. Pre-scale K:
  #    K' = K * scale
  # 2. Recompute P.T (on-the-fly from S.T = K' @ Q.T and LSE stats)
  #    P.T = exp(S.T - LSE.T)
  #    When using exp2, this becomes: P.T = exp2((S.T - LSE.T) * log2(e))
  # 3. Compute dV:
  #    dV = P.T @ dO
  # 4. Compute dP.T:
  #    dP.T = V @ dO.T
  # 5. Compute dS.T:
  #    dS.T = P.T * (dP.T - Delta.T)
  # 6. Compute dQ:
  #    dQ = (dS.T).T @ K'
  # 7. Compute dK:
  #    dK' = dS.T @ Q
  #    dK  = dK' * scale

  if config.block_q < 128:
    raise NotImplementedError(
        f"block_q {config.block_q=} must be larger than 128."
    )

  if config.block_kv < 128:
    raise NotImplementedError(
        f"block_kv {config.block_kv=} must be larger than 128."
    )

  as_ndim = lambda x, ndim: jax.lax.collapse(
      jax.lax.broadcast_to_rank(x, ndim), 0, -ndim + 1
  )
  as_3d = lambda x: as_ndim(x, 3)
  as_4d = lambda x: as_ndim(x, 4)
  pad_head_dim = lambda x: shape_lib.pad_to_next_multiple_of(x, 64, -1)

  orig_q_seq_len = q.shape[-3]
  orig_kv_seq_len = k.shape[-3]
  batch_shape = q.shape[:-3]

  # Padding logic
  block_q, block_kv = config.block_q, config.block_kv
  q = _pad(q, -3, block_q)
  k = _pad(k, -3, block_kv)
  v = _pad(v, -3, block_kv)
  out = _pad(out, -3, block_q)
  dout = _pad(dout, -3, block_q)

  bias = _pad(bias, (-2, -1), (block_q, block_kv), broadcastable=True)
  mask = _pad(mask, (-2, -1), (block_q, block_kv), broadcastable=True)
  k_start = _pad(k_start, -1, block_q, broadcastable=True)
  k_end = _pad(k_end, -1, block_q, broadcastable=True)

  m, l = residuals
  # Pad with safe values to avoid NaNs in P = exp(S - m) / l computation
  m = _pad(as_3d(m), -1, block_q, constant_values=1e9)
  l = _pad(as_3d(l), -1, block_q, constant_values=1)

  q, k, v, out, dout = map(as_4d, (q, k, v, out, dout))
  orig_head_dim = q.shape[-1]
  orig_head_dim_out = v.shape[-1]
  q, k, v, out, dout = map(pad_head_dim, (q, k, v, out, dout))

  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  bk_batch_size, kv_seq_len, num_kv_heads, head_dim_out = v.shape
  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(
        f"q, k, and v should all have the same dtype, got: {q.dtype},"
        f" {k.dtype}, {v.dtype}"
    )
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  if bk_batch_size != batch_size:
    raise ValueError("Batch sizes of Q and K/V must match.")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  num_q_tiles, rem = divmod(q_seq_len, config.block_q)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {config.block_q=}"
    )

  num_kv_tiles, rem = divmod(kv_seq_len, config.block_kv)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len=} must be a multiple of {config.block_kv=}"
    )

  if bias is not None:
    bias = _broadcast_to_seq(as_4d(bias), 2, q_seq_len, 3, kv_seq_len)
  if mask is not None:
    mask = _broadcast_to_seq(
        as_4d(mask).astype(jnp.int8), 2, q_seq_len, 3, kv_seq_len
    )
  k_start = _broadcast_to_seq(
      as_3d(k_start) if k_start is not None else None, 2, q_seq_len
  )
  k_end = _broadcast_to_seq(
      as_3d(k_end) if k_end is not None else None, 2, q_seq_len
  )

  # This is Delta = rowsum(O * dO), part of Step 5.
  delta = jnp.einsum(
      "bqhd,bqhd->bhq", out.astype(jnp.float32), dout.astype(jnp.float32)
  )

  num_stages = config.num_stages
  num_stages_dq = config.num_stages_dq
  block_q = config.block_q
  block_kv = config.block_kv

  def kernel(
      q_ref,
      k_ref,
      v_ref,
      dout_ref,
      lse_ref,
      delta_ref,
      bias_ref,
      mask_ref,
      k_start_ref,
      k_end_ref,
      # Outputs
      dq_ref,
      dk_ref,
      dv_ref,
      ds_ref,
      # Scratch
      q_smem,
      k_smem,
      v_smem,
      do_smem,
      bias_smem,
      k_start_smem,
      k_end_smem,
      mask_smem,
      p_ds_smem,
      lse_smem,
      delta_smem,
      dq_acc_smem,
      s_tmem,
      dp_tmem,
      dq_tmem,
      dk_tmem,
      dv_tmem,
      q_produced_barriers,
      q_consumed_barriers,
      k_produced_barrier,
      v_produced_barrier,
      do_produced_barriers,
      do_consumed_barriers,
      bias_produced_barriers,
      bias_consumed_barriers,
      mask_produced_barriers,
      mask_consumed_barriers,
      lse_produced_barriers,
      lse_consumed_barriers,
      delta_produced_barriers,
      delta_consumed_barriers,
      s_produced_barrier,
      s_consumed_barrier,
      dp_produced_barrier,
      dp_consumed_barrier,
      p_produced_barrier,
      p_consumed_barrier,
      ds_produced_barrier,
      ds_consumed_barrier,
      dq_mma_produced_barrier,
      dq_reduce_consumed_barrier,
      dkv_mma_compute_barriers,
  ):
    wg_id = lax.axis_index("wg")
    kv_tile_idx = lax.axis_index("kv_tiles")
    b_idx = lax.axis_index("batch")
    h_idx = lax.axis_index("heads")
    mask_value = float(jnp.finfo(jnp.float32).min)

    @pl.when(wg_id == _LOAD_MMA_WG)
    def load_mma_wg():
      # TODO: constrain the registers used with plgpu.set_max_registers.
      for _ in range(_NUM_COMPUTE_WGS):
        plgpu.barrier_arrive(s_consumed_barrier)

      plgpu.barrier_arrive(dq_reduce_consumed_barrier)
      plgpu.barrier_arrive(ds_consumed_barrier)
      for _ in range(_NUM_COMPUTE_WGS):
        plgpu.barrier_arrive(dp_consumed_barrier)

      for i in range(num_stages):
        for _ in range(_NUM_COMPUTE_WGS):
          plgpu.barrier_arrive(delta_consumed_barriers.at[i])
          plgpu.barrier_arrive(lse_consumed_barriers.at[i])
          if bias is not None:
            plgpu.barrier_arrive(bias_consumed_barriers.at[i])
          if mask_produced_barriers is not None:
            plgpu.barrier_arrive(mask_consumed_barriers.at[i])
        plgpu.barrier_arrive(
            q_consumed_barriers.at[i]
        )  # Arrive 1 for warp 1
        plgpu.barrier_arrive(q_consumed_barriers.at[i])
        plgpu.barrier_arrive(
            do_consumed_barriers.at[i]
        )  # Arrive 1 for warp 1
        plgpu.barrier_arrive(do_consumed_barriers.at[i])

      @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
      def _per_warp():
        warp_id = lax.axis_index("warp")

        @pl.when(warp_id == _MEMORY_WARP)
        def load_warp():
          # Load K and V once
          kv_head = lax.div(h_idx, jnp.array(q_heads_per_kv_head, h_idx.dtype))
          kv_b = 0 if k_ref.shape[0] == 1 else b_idx
          k_h = 0 if k_ref.shape[2] == 1 else kv_head
          kv_idx = (kv_b, _ds(kv_tile_idx * block_kv, block_kv), k_h)
          plgpu.copy_gmem_to_smem(
              k_ref.at[kv_idx], k_smem, barrier=k_produced_barrier
          )
          plgpu.copy_gmem_to_smem(
              v_ref.at[kv_idx], v_smem, barrier=v_produced_barrier
          )
          for i in range(num_q_tiles):
            stage = i % num_stages
            # Load Q: (B, T, H, D)
            q_b = 0 if q_ref.shape[0] == 1 else b_idx
            q_h = 0 if q_ref.shape[2] == 1 else h_idx
            q_slice = (q_b, _ds(i * block_q, block_q), q_h)
            plgpu.barrier_wait(q_consumed_barriers.at[stage])
            plgpu.copy_gmem_to_smem(
                q_ref.at[q_slice],
                q_smem.at[stage],
                barrier=q_produced_barriers.at[stage],
            )
            # Load dO: (B, T, H, D)
            do_b = 0 if dout_ref.shape[0] == 1 else b_idx
            do_h = 0 if dout_ref.shape[2] == 1 else h_idx
            do_slice = (do_b, _ds(i * block_q, block_q), do_h)
            plgpu.barrier_wait(do_consumed_barriers.at[stage])
            plgpu.copy_gmem_to_smem(
                dout_ref.at[do_slice],
                do_smem.at[stage],
                barrier=do_produced_barriers.at[stage],
            )
            # Load LSE: (B, H, T) -> smem (stage, block_q)
            lse_b = 0 if lse_ref.shape[0] == 1 else b_idx
            lse_h = 0 if lse_ref.shape[1] == 1 else h_idx
            lse_slice = (lse_b, lse_h, _ds(i * block_q, block_q))
            plgpu.barrier_wait(lse_consumed_barriers.at[stage])
            plgpu.copy_gmem_to_smem(
                lse_ref.at[lse_slice],
                lse_smem.at[stage],
                barrier=lse_produced_barriers.at[stage],
            )
            # Load Delta: (B, H, T) -> smem (stage, block_q)
            delta_b = 0 if delta_ref.shape[0] == 1 else b_idx
            delta_h = 0 if delta_ref.shape[1] == 1 else h_idx
            delta_slice = (delta_b, delta_h, _ds(i * block_q, block_q))
            plgpu.barrier_wait(delta_consumed_barriers.at[stage])
            plgpu.copy_gmem_to_smem(
                delta_ref.at[delta_slice],
                delta_smem.at[stage],
                barrier=delta_produced_barriers.at[stage],
            )
            if bias is not None:
              # Load Bias: (B, H, T, S) -> smem (stage, block_q, block_kv)
              bias_b = 0 if bias_ref.shape[0] == 1 else b_idx
              bias_h = 0 if bias_ref.shape[1] == 1 else h_idx
              bias_slice = (
                  bias_b,
                  bias_h,
                  _ds(i * block_q, block_q),
                  _ds(kv_tile_idx * block_kv, block_kv),
              )
              plgpu.barrier_wait(bias_consumed_barriers.at[stage])
              plgpu.copy_gmem_to_smem(
                  bias_ref.at[bias_slice],
                  bias_smem.at[stage],
                  barrier=bias_produced_barriers.at[stage],
              )
            if mask_produced_barriers is not None:
              plgpu.barrier_wait(mask_consumed_barriers.at[stage])
              if k_start_ref is not None:
                ks_b = 0 if k_start_ref.shape[0] == 1 else b_idx
                ks_h = 0 if k_start_ref.shape[1] == 1 else h_idx
                ks_slice = (ks_b, ks_h, _ds(i * block_q, block_q))
                plgpu.copy_gmem_to_smem(
                    k_start_ref.at[ks_slice],
                    k_start_smem.at[stage],
                    barrier=mask_produced_barriers.at[stage],
                )
              if k_end_ref is not None:
                ke_b = 0 if k_end_ref.shape[0] == 1 else b_idx
                ke_h = 0 if k_end_ref.shape[1] == 1 else h_idx
                ke_slice = (ke_b, ke_h, _ds(i * block_q, block_q))
                plgpu.copy_gmem_to_smem(
                    k_end_ref.at[ke_slice],
                    k_end_smem.at[stage],
                    barrier=mask_produced_barriers.at[stage],
                )
              if mask_ref is not None:
                # Load Mask: (B, H, T, S) -> smem (stage, block_q, block_kv)
                mask_b = 0 if mask_ref.shape[0] == 1 else b_idx
                mask_h = 0 if mask_ref.shape[1] == 1 else h_idx
                mask_slice = (
                    mask_b,
                    mask_h,
                    _ds(i * block_q, block_q),
                    _ds(kv_tile_idx * block_kv, block_kv),
                )
                plgpu.copy_gmem_to_smem(
                    mask_ref.at[mask_slice],
                    mask_smem.at[stage],
                    barrier=mask_produced_barriers.at[stage],
                )

        @pl.when(warp_id == _MMA_WARP_1)
        def mma_warp_1():
          plgpu.barrier_wait(k_produced_barrier)
          plgpu.barrier_wait(v_produced_barrier)

          for i in range(num_q_tiles):
            stage = i % num_stages
            # Step 2.1: S = Q @ K'.T
            plgpu.barrier_wait(s_consumed_barrier)
            plgpu.barrier_wait(q_produced_barriers.at[stage])
            plgpu.tcgen05_mma(
                s_tmem,
                q_smem.at[stage],
                k_smem.T,
                accumulate=False,  # Q @ K'.T
            )
            plgpu.tcgen05_commit_arrive(q_consumed_barriers.at[stage])
            plgpu.tcgen05_commit_arrive(s_produced_barrier)

            # Step 4: dP = dO @ V.T
            plgpu.barrier_wait(dp_consumed_barrier)
            plgpu.barrier_wait(do_produced_barriers.at[stage])
            plgpu.tcgen05_mma(
                dp_tmem,
                do_smem.at[stage],
                v_smem.T,
                accumulate=False,  # dO @ V.T
            )
            plgpu.tcgen05_commit_arrive(dp_produced_barrier)
            plgpu.tcgen05_commit_arrive(do_consumed_barriers.at[stage])

            # Step 3: dV += P.T @ dO
            plgpu.barrier_wait(p_produced_barrier)
            plgpu.tcgen05_mma(
                dv_tmem,
                p_ds_smem.T,  # P is (block_q, block_kv) -> P.T is (block_kv, block_q)
                do_smem.at[stage],
                accumulate=i > 0,  # P.T @ dO
            )
            plgpu.tcgen05_commit_arrive(p_consumed_barrier)
            plgpu.tcgen05_commit_arrive(do_consumed_barriers.at[stage])

            # Step 7.1: dK' += dS.T @ Q
            plgpu.barrier_wait(ds_produced_barrier)
            plgpu.tcgen05_mma(
                # dS is (block_q, block_kv) -> dS.T is (block_kv, block_q)
                dk_tmem,
                p_ds_smem.T,
                q_smem.at[stage],
                accumulate=i > 0,  # dS.T @ Q
            )
            plgpu.tcgen05_commit_arrive(q_consumed_barriers.at[stage])

            # Step 6: dQ = dS @ K'
            # We store dQ in the TMEM slot that was previously holding dP.
            plgpu.barrier_wait(dq_reduce_consumed_barrier)
            plgpu.tcgen05_mma(
                dq_tmem, p_ds_smem, k_smem, accumulate=False  # dS @ K'
            )
            plgpu.tcgen05_commit_arrive(dq_mma_produced_barrier)
            plgpu.tcgen05_commit_arrive(ds_consumed_barrier)

          plgpu.tcgen05_commit_arrive(dkv_mma_compute_barriers)

    @pl.when(wg_id >= _COMPUTE_WG_1)
    def compute_wg():
      # TODO: constrain the registers used with plgpu.set_max_registers
      kv_off = (block_kv // 2) * (wg_id - _COMPUTE_WG_1)

      def _compute_p(
          s_val,
          lse_val,
          bias_val,
          k_start_val,
          k_end_val,
          kv_off_chunk,
          q_tile_idx,
          stage,
          mask_smem,
      ):
        # P = exp(S - LSE)
        val = s_val
        if bias_val is not None:
          val += bias_val

        tanh_val = None
        if logits_soft_cap is not None:
          tanh_val = jnp.tanh(val / logits_soft_cap)
          val = tanh_val * logits_soft_cap

        base_val = val - lse_val

        if mask_smem is not None:
          mask_val = plgpu.load(
              mask_smem.at[stage], (slice(None), chunk_slice), layout=_TMEM
          )
          mask_val = plgpu.layout_cast(mask_val, _TMEM)
          base_val = jnp.where(mask_val != 0, base_val, mask_value)

        if is_causal:
          k_iota = (
              plgpu.broadcasted_iota(jnp.int32, base_val.shape, 1, layout=_TMEM)
              + kv_off_chunk
              + kv_tile_idx * block_kv
          )
          q_iota = (
              plgpu.broadcasted_iota(jnp.int32, base_val.shape, 0, layout=_TMEM)
              + q_tile_idx * block_q
          )

          # k <= q
          causal_mask = k_iota <= q_iota
          base_val = jnp.where(causal_mask, base_val, mask_value)

        if k_start_val is not None or k_end_val is not None:
          k_iota = (
              plgpu.broadcasted_iota(jnp.int32, base_val.shape, 1, layout=_TMEM)
              + kv_off_chunk
              + kv_tile_idx * block_kv
          )
          if k_start_val is not None:
            mask = k_iota >= k_start_val
            base_val = jnp.where(mask, base_val, mask_value)
          if k_end_val is not None:
            mask = k_iota < k_end_val
            base_val = jnp.where(mask, base_val, mask_value)

        if use_base2:
          # Log2 change of base: exp2(x * log2(e))
          p_val = jnp.exp2(base_val * math.log2(math.e))
        else:
          p_val = jnp.exp(base_val)
        return p_val, tanh_val

      for i in range(num_q_tiles):
        stage = i % num_stages
        qs = _ds(i * block_q, block_q)

        chunk_size = 32
        num_chunks = (block_kv // 2) // chunk_size

        # Step 2.2: Recompute P from S and LSE.
        plgpu.barrier_wait(s_produced_barrier)
        plgpu.barrier_wait(lse_produced_barriers.at[stage])
        if bias is not None:
          plgpu.barrier_wait(bias_produced_barriers.at[stage])
        if mask_produced_barriers is not None:
          plgpu.barrier_wait(mask_produced_barriers.at[stage])
        plgpu.barrier_wait(ds_consumed_barrier)
        # --- Pass 1: Compute P and store to SMEM (Handover Phase 1) ---
        for chunk_idx in range(num_chunks):
          kv_off_chunk = kv_off + chunk_idx * chunk_size
          chunk_slice = pl.ds(kv_off_chunk, chunk_size)

          s_val = plgpu.async_load_tmem(
              s_tmem.at[:, chunk_slice], layout=_TMEM
          )
          plgpu.wait_load_tmem()
          s_val *= logits_scale

          lse_vec = plgpu.load(lse_smem, stage, layout=_TMEM_ROW)
          lse_val = lax.broadcast_in_dim(lse_vec, s_val.shape, [0])
          lse_val = plgpu.layout_cast(lse_val, _TMEM)
          bias_val = None
          if bias is not None:
            bias_val = plgpu.load(
                bias_smem.at[stage], (slice(None), chunk_slice), layout=_TMEM
            )
            bias_val = plgpu.layout_cast(bias_val, _TMEM)
          k_start_val, k_end_val = None, None
          if k_start_smem is not None:
            k_start_val = plgpu.load(k_start_smem, stage, layout=_TMEM_ROW)
            k_start_val = lax.broadcast_in_dim(k_start_val, s_val.shape, [0])
            k_start_val = plgpu.layout_cast(k_start_val, _TMEM)
          if k_end_smem is not None:
            k_end_val = plgpu.load(k_end_smem, stage, layout=_TMEM_ROW)
            k_end_val = lax.broadcast_in_dim(k_end_val, s_val.shape, [0])
            k_end_val = plgpu.layout_cast(k_end_val, _TMEM)

          p_val, _ = _compute_p(
              s_val,
              lse_val,
              bias_val,
              k_start_val,
              k_end_val,
              kv_off_chunk,
              i,
              stage,
              mask_smem,
          )
          p_ds_smem.at[:, chunk_slice].set(p_val.astype(dtype))
          plgpu.commit_smem()
        plgpu.barrier_arrive(p_produced_barrier)

        # Step 5: Compute dS = P * (dP - Delta)
        # Wait for dP (from MMA) and Delta (from Load)
        plgpu.barrier_wait(dp_produced_barrier)
        plgpu.barrier_wait(delta_produced_barriers.at[stage])
        plgpu.barrier_wait(p_consumed_barrier)
        # --- Pass 2: Recompute P, Load dP, Compute dS (Handover Phase 2) ---
        for chunk_idx in range(num_chunks):
          kv_off_chunk = kv_off + chunk_idx * chunk_size
          chunk_slice = pl.ds(kv_off_chunk, chunk_size)

          # Re-load S and LSE to recompute P in float32
          s_val = plgpu.async_load_tmem(
              s_tmem.at[:, chunk_slice], layout=_TMEM
          )
          plgpu.wait_load_tmem()
          s_val *= logits_scale

          lse_vec = plgpu.load(lse_smem, stage, layout=_TMEM_ROW)
          lse_val = lax.broadcast_in_dim(lse_vec, s_val.shape, [0])
          lse_val = plgpu.layout_cast(lse_val, _TMEM)
          bias_val = None
          if bias is not None:
            bias_val = plgpu.load(
                bias_smem.at[stage], (slice(None), chunk_slice), layout=_TMEM
            )
            bias_val = plgpu.layout_cast(bias_val, _TMEM)
          k_start_val, k_end_val = None, None
          if k_start_smem is not None:
            k_start_val = plgpu.load(k_start_smem, stage, layout=_TMEM_ROW)
            k_start_val = lax.broadcast_in_dim(k_start_val, s_val.shape, [0])
            k_start_val = plgpu.layout_cast(k_start_val, _TMEM)
          if k_end_smem is not None:
            k_end_val = plgpu.load(k_end_smem, stage, layout=_TMEM_ROW)
            k_end_val = lax.broadcast_in_dim(k_end_val, s_val.shape, [0])
            k_end_val = plgpu.layout_cast(k_end_val, _TMEM)

          p_val, tanh_val = _compute_p(
              s_val,
              lse_val,
              bias_val,
              k_start_val,
              k_end_val,
              kv_off_chunk,
              i,
              stage,
              mask_smem,
          )
          # Load dP and Delta
          dp_val = plgpu.async_load_tmem(
              dp_tmem.at[:, chunk_slice], layout=_TMEM
          )
          plgpu.wait_load_tmem()

          delta_vec = plgpu.load(delta_smem, stage, layout=_TMEM_ROW)
          delta_val = lax.broadcast_in_dim(delta_vec, dp_val.shape, [0])
          delta_val = plgpu.layout_cast(delta_val, _TMEM)

          ds_val = p_val * (dp_val - delta_val)
          if logits_soft_cap is not None:
            ds_val *= 1.0 - tanh_val * tanh_val

          ks = pl.ds(kv_tile_idx * block_kv + kv_off_chunk, chunk_size)
          p_ds_smem.at[:, chunk_slice].set(ds_val.astype(p_ds_smem.dtype))
          if ds_ref is not None:
            ds_b = 0 if ds_ref.shape[0] == 1 else b_idx
            ds_h = 0 if ds_ref.shape[1] == 1 else h_idx
            ds_ref.at[ds_b, ds_h, qs, ks].set(ds_val.astype(ds_ref.dtype))
          plgpu.commit_smem()

        plgpu.barrier_arrive(s_consumed_barrier)
        plgpu.barrier_arrive(lse_consumed_barriers.at[stage])
        if bias_ref is not None:
          plgpu.barrier_arrive(bias_consumed_barriers.at[stage])
        if mask_produced_barriers is not None:
          plgpu.barrier_arrive(mask_consumed_barriers.at[stage])
        plgpu.barrier_arrive(dp_consumed_barrier)
        plgpu.barrier_arrive(delta_consumed_barriers.at[stage])
        plgpu.barrier_arrive(ds_produced_barrier)

    @pl.when(wg_id == _REDUCE_WG)
    def reduce_wg():
      # TODO: constrain the registers used with plgpu.set_max_registers
      b_idx = lax.axis_index("batch")
      h_idx = lax.axis_index("heads")

      for i in range(num_q_tiles):
        # Wait for dQ block from MMA WG
        plgpu.barrier_wait(dq_mma_produced_barrier)

        # Wait for previous TMA store from this stage to complete.
        # We allow (num_stages - 1) pending stores.
        plgpu.wait_smem_to_gmem(num_stages_dq - 1)

        # Load dQ from TMEM to Registers -> SMEM
        dq_val = plgpu.async_load_tmem(dq_tmem, layout=_TMEM)
        plgpu.wait_load_tmem()
        dq_val *= logits_scale

        # Signal MMA that TMEM slot for dQ is now consumed
        plgpu.barrier_arrive(dq_reduce_consumed_barrier)
        dq_stage = i % num_stages_dq
        dq_acc_smem.at[dq_stage].set(dq_val.astype(dq_acc_smem.dtype))
        plgpu.commit_smem()

        # Asynchronously add the dQ block to GMEM.
        q_slice = _ds(i * block_q, block_q)
        dq_slice = (b_idx, q_slice, h_idx)
        plgpu.copy_smem_to_gmem(
            dq_acc_smem.at[dq_stage], dq_ref.at[dq_slice], reduction_op="add"
        )

      # Epilogue: Store dK and dV. Done in the reduce wg because it
      # already has the correct number of registers allocated.
      kv_tile_idx = lax.axis_index("kv_tiles")
      b_idx = lax.axis_index("batch")
      h_idx = lax.axis_index("heads")
      plgpu.barrier_wait(dkv_mma_compute_barriers)

      dk_val = plgpu.async_load_tmem(dk_tmem, layout=_TMEM)
      # Step 7.2: dK = dK' * scale
      plgpu.wait_load_tmem()
      k_smem.set((dk_val * logits_scale).astype(k_smem.dtype))
      plgpu.commit_smem()
      kv_idx = (b_idx, _ds(kv_tile_idx * block_kv, block_kv), h_idx)
      plgpu.copy_smem_to_gmem(k_smem, dk_ref.at[kv_idx])

      dv_val = plgpu.async_load_tmem(dv_tmem, layout=_TMEM)
      plgpu.wait_load_tmem()
      v_smem.set(dv_val.astype(v_smem.dtype))
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(v_smem, dv_ref.at[kv_idx])

  def smem_transforms(dtype, swizzle=64):
    return (
        plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)),
        plgpu.SwizzleTransform(swizzle),
    )

  # Output shapes
  # dq, dk, dv, ds
  dq_zeros = jnp.zeros(q.shape, jnp.float32)

  ds_shape = (
      jax.ShapeDtypeStruct(bias.shape, bias.dtype) if bias is not None else None
  )
  dk_shape = (batch_size, kv_seq_len, num_q_heads, head_dim)
  dv_shape = (batch_size, kv_seq_len, num_q_heads, head_dim_out)
  out_shape = (
      # dq, dk, dv, ds
      dq_zeros,
      jax.ShapeDtypeStruct(dk_shape, k.dtype),
      jax.ShapeDtypeStruct(dv_shape, v.dtype),
      ds_shape,
  )

  def _and_smem(arr, shape, **kw):
    if arr is None:
      return None
    return plgpu.SMEM(shape, **dict(dtype=arr.dtype) | kw)

  if k_start is not None or k_end is not None or mask is not None:
    mask_produced_barriers = plgpu.Barrier(
        num_barriers=num_stages,
        num_arrivals=(k_start is not None)
        + (k_end is not None)
        + (mask is not None),
    )
    mask_consumed_barriers = plgpu.Barrier(
        num_barriers=num_stages, num_arrivals=_NUM_COMPUTE_WGS
    )
  else:
    mask_produced_barriers = None
    mask_consumed_barriers = None

  scratch_shapes = dict(
      q_smem=plgpu.SMEM(
          (num_stages, block_q, head_dim),
          q.dtype,
          transforms=smem_transforms(q.dtype),
      ),
      k_smem=plgpu.SMEM(
          (block_kv, head_dim), k.dtype, transforms=smem_transforms(k.dtype)
      ),
      v_smem=plgpu.SMEM(
          (block_kv, head_dim_out), v.dtype, transforms=smem_transforms(v.dtype)
      ),
      do_smem=plgpu.SMEM(
          (num_stages, block_q, head_dim_out),
          dout.dtype,
          transforms=smem_transforms(dout.dtype),
      ),
      bias_smem=_and_smem(bias, (num_stages, block_q, block_kv)),
      k_start_smem=_and_smem(k_start, (num_stages, block_q)),
      k_end_smem=_and_smem(k_end, (num_stages, block_q)),
      mask_smem=_and_smem(
          mask,
          (num_stages, block_q, block_kv),
          dtype=jnp.int8,
          transforms=smem_transforms(jnp.int8, swizzle=32),
      ),
      p_ds_smem=plgpu.SMEM(
          (block_q, block_kv),
          ds_dtype or q.dtype,
          transforms=smem_transforms(ds_dtype or q.dtype),
      ),
      lse_smem=plgpu.SMEM((num_stages, block_q), jnp.float32),
      delta_smem=plgpu.SMEM((num_stages, block_q), jnp.float32),
      dq_acc_smem=plgpu.SMEM(
          (num_stages_dq, block_q, head_dim),
          dq_zeros.dtype,
          transforms=smem_transforms(dq_zeros.dtype, swizzle=32),
      ),
      s_tmem=plgpu.TMEM((block_q, block_kv), jnp.float32),
      dp_tmem=plgpu.TMEM((block_q, block_kv), jnp.float32),
      dq_tmem=plgpu.TMEM((block_q, head_dim), jnp.float32),
      dk_tmem=plgpu.TMEM((block_kv, head_dim), jnp.float32),
      dv_tmem=plgpu.TMEM((block_kv, head_dim_out), jnp.float32),
      q_produced_barriers=plgpu.Barrier(num_barriers=num_stages),
      q_consumed_barriers=plgpu.Barrier(
          num_barriers=num_stages, num_arrivals=2, orders_tensor_core=True
      ),
      k_produced_barrier=plgpu.Barrier(),
      v_produced_barrier=plgpu.Barrier(),
      do_produced_barriers=plgpu.Barrier(num_barriers=num_stages),
      do_consumed_barriers=plgpu.Barrier(
          num_barriers=num_stages, num_arrivals=2, orders_tensor_core=True
      ),
      bias_produced_barriers=None
      if bias is None
      else plgpu.Barrier(num_barriers=num_stages),
      bias_consumed_barriers=None
      if bias is None
      else plgpu.Barrier(
          num_barriers=num_stages, num_arrivals=_NUM_COMPUTE_WGS
      ),
      mask_produced_barriers=mask_produced_barriers,
      mask_consumed_barriers=mask_consumed_barriers,
      lse_produced_barriers=plgpu.Barrier(num_barriers=num_stages),
      lse_consumed_barriers=plgpu.Barrier(
          num_barriers=num_stages, num_arrivals=_NUM_COMPUTE_WGS
      ),
      delta_produced_barriers=plgpu.Barrier(num_barriers=num_stages),
      delta_consumed_barriers=plgpu.Barrier(
          num_barriers=num_stages, num_arrivals=_NUM_COMPUTE_WGS
      ),
      s_produced_barrier=plgpu.Barrier(orders_tensor_core=True),
      s_consumed_barrier=plgpu.Barrier(num_arrivals=_NUM_COMPUTE_WGS),
      dp_produced_barrier=plgpu.Barrier(orders_tensor_core=True),
      dp_consumed_barrier=plgpu.Barrier(num_arrivals=_NUM_COMPUTE_WGS),
      p_produced_barrier=plgpu.Barrier(num_arrivals=_NUM_COMPUTE_WGS),
      p_consumed_barrier=plgpu.Barrier(orders_tensor_core=True),
      ds_produced_barrier=plgpu.Barrier(num_arrivals=_NUM_COMPUTE_WGS),
      ds_consumed_barrier=plgpu.Barrier(orders_tensor_core=True),
      dq_mma_produced_barrier=plgpu.Barrier(orders_tensor_core=True),
      dq_reduce_consumed_barrier=plgpu.Barrier(),
      dkv_mma_compute_barriers=plgpu.Barrier(orders_tensor_core=True),
  )
  grid = (batch_size, num_q_heads, num_kv_tiles)
  (
      dq,
      dk,
      dv,
      ds,
  ) = _kernel(
      kernel,
      out_shape=out_shape,
      grid=grid,
      grid_names=("batch", "heads", "kv_tiles"),
      num_threads=_NUM_COMPUTE_WGS + 2,
      thread_name="wg",
      scratch_shapes=scratch_shapes,
  )(q, k, v, dout, m + jnp.log(l), delta, bias, mask, k_start, k_end)

  if q_heads_per_kv_head > 1:
    dk = dk.reshape(*k.shape[:-1], q_heads_per_kv_head, -1).sum(axis=-2)
    dv = dv.reshape(*v.shape[:-1], q_heads_per_kv_head, -1).sum(axis=-2)

  dq = dq[..., :orig_q_seq_len, :, :orig_head_dim]
  dk = dk[..., :orig_kv_seq_len, :, :orig_head_dim]
  dv = dv[..., :orig_kv_seq_len, :, :orig_head_dim_out]
  if ds is not None:
    ds = ds[..., :orig_q_seq_len, :orig_kv_seq_len]

  dq = dq.reshape(batch_shape + dq.shape[1:])
  dk = dk.reshape(batch_shape + dk.shape[1:])
  dv = dv.reshape(batch_shape + dv.shape[1:])

  return dq.astype(q.dtype), dk, dv, ds
