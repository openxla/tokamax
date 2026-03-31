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
"""Pallas-Mosaic-GPU SM90 forward+backward kernels for linear softmax CE loss.

Algorithm (forward): tiles (B, V) with an inner H pipeline, so the
(b_tile, v_tile) logit matrix never appears in HBM. Two warp groups (wg=0,1)
each handle tile_m rows of the 2*tile_m CTA tile; WGMMA + TMA pipelines
compute the matmul x[b_tile,:] @ w[:,v_tile] and the epilogue reduces to
per-token logsumexp. The correct-class logit is computed outside the kernel as
a cheap O(B*H) XLA einsum (gather + dot).

Algorithm (backward): implemented in pallas_mosaic_gpu.py as a jax.lax.scan
over padded vocabulary chunks, issuing cuBLAS GEMMs per chunk (not WGMMA).
The in-kernel backward (linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_gpu_sm90)
exists and is tested, but is not wired into the Op — it was superseded by the
chunked-scan approach which avoids atomic_add serialisation across CTAs.
"""

import functools
from typing import Literal

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real, Scalar

_WGMMA = plgpu.Layout.WGMMA
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)


def _validate_inputs(
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> None:
  """Validates inputs and tile-size constraints."""
  b_dim, h_dim = x.shape
  v_dim = w.shape[1]
  if b_dim % (2 * tile_m) != 0:
    raise ValueError(
        f"Batch dimension B={b_dim} must be divisible by"
        f" 2 * tile_m={2 * tile_m}."
    )
  if h_dim % tile_k != 0:
    raise ValueError(
        f"Hidden dimension H={h_dim} must be divisible by tile_k={tile_k}."
    )
  if v_dim % tile_n != 0:
    raise ValueError(
        f"Vocab dimension V={v_dim} must be divisible by tile_n={tile_n}."
    )
  if w.shape[0] != h_dim:
    raise ValueError(
        f"w hidden dim {w.shape[0]} must match x hidden dim {h_dim}."
    )
  if labels.shape[0] != b_dim:
    raise ValueError(
        f"labels batch size {labels.shape[0]} must match x batch size {b_dim}."
    )


def linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_gpu_sm90(
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    num_stages: int = 4,
    reduction: Literal["sum", "mean"] = "sum",
) -> tuple[Real[Scalar, ""], Real[Array, "B"]]:
  """Forward pass for linear softmax cross-entropy loss via Pallas/Mosaic-GPU.

  Uses WGMMA + TMA pipelining on SM90 (H100). Two warp groups each handle
  tile_m rows of the current (b_cta, v) tile, accumulating x @ w across the
  H dimension before computing per-token logsumexp and correct-class logit.

  Args:
    x: Hidden states, shape (B, H).
    labels: Integer token indices, shape (B,).
    w: LM head weight matrix, shape (H, V).
    tile_m: Tile size over B. Each CTA uses 2 * tile_m rows; B must be
      divisible by 2 * tile_m.
    tile_n: Tile size over V. V must be divisible by tile_n.
    tile_k: Tile size for the H contraction loop. H must be divisible by
      tile_k.
    num_stages: TMA pipeline depth.
    reduction: "sum" or "mean" over tokens.

  Returns:
    (loss, lse) where lse is the per-token log-sum-exp, shape (B,).
  """
  _validate_inputs(x, labels, w, tile_m, tile_k, tile_n)

  # Mosaic GPU wgmma operates in bfloat16 with float32 accumulation. Downcast
  # float32 inputs to bfloat16 to halve SMEM usage and use the faster bf16
  # wgmma path (same approach as the attention sm90 kernel).
  if x.dtype != jnp.bfloat16:
    x = x.astype(jnp.bfloat16)
  if w.dtype != jnp.bfloat16:
    w = w.astype(jnp.bfloat16)

  b_dim, h_dim = x.shape
  v_dim = w.shape[1]
  dtype = x.dtype  # bfloat16
  elem_bits = jnp.finfo(dtype).bits

  cta_tile_m = 2 * tile_m  # two warp groups each covering tile_m rows
  b_cta_iters = b_dim // cta_tile_m
  v_iters = v_dim // tile_n
  k_iters = h_dim // tile_k

  # Swizzle for lhs (x tiles: last dim = tile_k) and rhs (w tiles: last dim = tile_n).
  # Rule: swizzle = find_swizzle(last_dim * elem_bits) — see attention common.
  lhs_swizzle = plgpu.find_swizzle(tile_k * elem_bits)
  lhs_swizzle_elems = 8 * lhs_swizzle // elem_bits
  lhs_transforms = (
      plgpu.TilingTransform((8, lhs_swizzle_elems)),
      plgpu.SwizzleTransform(lhs_swizzle),
  )

  rhs_swizzle = plgpu.find_swizzle(tile_n * elem_bits)
  rhs_swizzle_elems = 8 * rhs_swizzle // elem_bits
  rhs_transforms = (
      plgpu.TilingTransform((8, rhs_swizzle_elems)),
      plgpu.SwizzleTransform(rhs_swizzle),
  )

  def kernel(
      x_gmem,
      w_gmem,
      tile_lse_gmem,
      lse_smem,
  ):
    """Persistent kernel body.

    Args:
      x_gmem: Input activations, shape (B, H).
      w_gmem: Weight matrix, shape (H, V).
      tile_lse_gmem: Output per-tile logsumexp, shape (v_iters, B).
      lse_smem: Scratch SMEM for lse staging, shape (2, tile_m).
    """

    def get_pipeline(pipeline_body, compute_context):
      return plgpu.emit_pipeline_warp_specialized(
          pipeline_body,
          grid=(k_iters,),
          memory_registers=40,
          in_specs=[
              plgpu.BlockSpec(
                  (cta_tile_m, tile_k),
                  lambda k: (0, k),
                  transforms=lhs_transforms,
                  memory_space=plgpu.SMEM,
              ),
              plgpu.BlockSpec(
                  (tile_k, tile_n),
                  lambda k: (k, 0),
                  transforms=rhs_transforms,
                  memory_space=plgpu.SMEM,
              ),
          ],
          wg_axis="wg",
          num_compute_wgs=2,
          max_concurrent_steps=num_stages,
          compute_context=compute_context,
      )

    ignore = lambda *_, **__: None

    @functools.partial(
        pl.run_scoped,
        pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(
            x_gmem, w_gmem
        ),
        collective_axes="wg",
    )
    def _pipeline_scope(pipeline_allocs):
      wg_idx = lax.axis_index("wg")

      @plgpu.nd_loop((b_cta_iters * v_iters,), collective_axes="cluster_grid")
      def _bv_loop(loop_info):
        (lin_idx,) = loop_info.index
        b_cta_idx = lin_idx // v_iters
        v_idx = lin_idx % v_iters

        b_cta_start = b_cta_idx * cta_tile_m
        v_start = v_idx * tile_n

        # Each wg handles its own tile_m-row slice of the cta_tile_m block.
        wg_b_start = b_cta_start + wg_idx * tile_m
        b_wg_slice = pl.ds(wg_b_start, tile_m)

        def compute_context(eval_pipeline):

          @functools.partial(
              pl.run_scoped,
              acc_ref=plgpu.ACC((tile_m, tile_n), jnp.float32),
          )
          def _acc_scope(acc_ref):
            eval_pipeline(acc_ref)
            acc = acc_ref[...].astype(jnp.float32)  # (tile_m, tile_n) WGMMA

            # Per-token logsumexp over this V tile.
            # - No keepdims: (tile_m, 1) violates WGMMA tile divisibility.
            # - jax.nn.logsumexp is off-limits: calls is_finite internally.
            # - Use lax.broadcast_in_dim to expand back to (tile_m, tile_n).
            amax = jnp.max(acc, axis=-1)  # (tile_m,) WGMMA_ROW
            amax_bcast = lax.broadcast_in_dim(amax, acc.shape, [0])
            tile_lse_vals = amax + jnp.log(
                jnp.sum(jnp.exp(acc - amax_bcast), axis=-1)
            )  # (tile_m,) WGMMA_ROW

            # Stage through SMEM then TMA-store to GMEM.
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            lse_smem[wg_idx] = tile_lse_vals
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                lse_smem.at[wg_idx],
                tile_lse_gmem.at[v_idx, b_wg_slice],
            )

        def mma_body(_, x_smem, w_smem, acc_ref):
          wg_m_slice = pl.ds(wg_idx * tile_m, tile_m)
          # w is (K, N) in SMEM — no transpose needed (cf. v_smem in attention).
          plgpu.wgmma(acc_ref, x_smem.at[wg_m_slice], w_smem)
          plgpu.wgmma_wait(0)
          return acc_ref

        get_pipeline(mma_body, compute_context)(
            x_gmem.at[pl.ds(b_cta_start, cta_tile_m), :],
            w_gmem.at[:, pl.ds(v_start, tile_n)],
            allocations=pipeline_allocs,
        )

    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  num_sms = backend.get_default_device().core_count
  scratch_shapes = [
      plgpu.SMEM((2, tile_m), jnp.float32),  # lse staging, one row per wg
  ]

  f = plgpu.kernel(
      kernel,
      out_shape=[
          jax.ShapeDtypeStruct((v_iters, b_dim), jnp.float32),
      ],
      grid=(num_sms,),
      grid_names=("cluster_grid",),
      cluster=(1,),
      cluster_names=("cluster",),
      num_threads=3,
      thread_name="wg",
      scratch_shapes=scratch_shapes,
  )

  (tile_lse,) = f(x, w)

  # Combine across V tiles; tile_lse is (v_iters, B), reduce over v_iters.
  lse = jax.nn.logsumexp(tile_lse, axis=0)  # (B,)

  # Correct-class logit: O(B*H) XLA gather+dot, much cheaper than the kernel.
  # Using float32 throughout for consistency with the fp32 kernel accumulation.
  x_f32 = x.astype(jnp.float32)
  w_f32 = w.astype(jnp.float32)
  correct_logit = jnp.einsum("bh,hb->b", x_f32, w_f32[:, labels])  # (B,)
  per_token_loss = lse - correct_logit

  if reduction == "sum":
    loss = jnp.sum(per_token_loss)
  else:
    loss = jnp.mean(per_token_loss)

  return loss.astype(jnp.float32), lse

