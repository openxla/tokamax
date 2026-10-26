# Copyright 2026 Google LLC. All Rights Reserved.
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

"""PyTorch interface for Pallas TPU Native Sparse Attention (NSA) operator."""

from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tokamax._src.ops.experimental.nsa import base as jax_base
from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu as jax_pallas
from tokamax.experimental.torch_tpu.ops.nsa import torch_base
from typing_extensions import override


class _PallasTpuNativeSparseAttention(torch_base._NativeSparseAttention):
  """PyTorch wrapper for Pallas TPU NSA operator."""

  def __init__(self):
    super().__init__(
        op_impl_jax=jax_pallas.PallasTpuNativeSparseAttention(),
        jax_op_name="torch_tpu_pallas_nsa",
    )

  def _register_ops(self) -> None:
    if hasattr(super(), "_register_ops"):
      super()._register_ops()
    try:
      from torch_tpu._internal import pallas  # pylint: disable=g-import-not-at-top
    except ImportError:
      pass
    else:
      if self._torch_tokamax_op is not None:
        self._setup_autograd(pallas)

  @override
  def _cpu_fallback(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g_cmp: torch.Tensor,
      g_slc: torch.Tensor,
      g_swa: torch.Tensor,
      chunk_size: int = torch_base._CHUNK_SIZE,
      topk: int = torch_base._TOPK,
      window: int = torch_base._WINDOW,
      cmp_block_size: int = torch_base._CMP_BLOCK_SIZE,
      scale: Optional[float] = None,
  ) -> torch.Tensor:
    """Executes the Pallas TPU NSA kernel in interpret mode via JAX."""
    op_self = self

    class _PallasCpuFn(torch.autograd.Function):

      @staticmethod
      def forward(ctx, q_t, k_t, v_t, gc_t, gs_t, gw_t):
        ctx.save_for_backward(q_t, k_t, v_t, gc_t, gs_t, gw_t)
        q_j = jnp.asarray(q_t.detach().cpu().numpy())
        k_j = jnp.asarray(k_t.detach().cpu().numpy())
        v_j = jnp.asarray(v_t.detach().cpu().numpy())
        gc_j = jnp.asarray(gc_t.detach().cpu().numpy())
        gs_j = jnp.asarray(gs_t.detach().cpu().numpy())
        gw_j = jnp.asarray(gw_t.detach().cpu().numpy())
        out_j = op_self.op_impl_call(
            q_j,
            k_j,
            v_j,
            g_cmp=gc_j,
            g_slc=gs_j,
            g_swa=gw_j,
            chunk_size=chunk_size,
            topk=topk,
            window=window,
            cmp_block_size=cmp_block_size,
            scale=scale,
        )
        return torch.from_numpy(np.array(out_j, copy=True)).to(
            device=q_t.device, dtype=q_t.dtype
        )

      @staticmethod
      def backward(ctx, do_t):
        q_t, k_t, v_t, gc_t, gs_t, gw_t = ctx.saved_tensors
        q_j = jnp.asarray(q_t.detach().cpu().float().numpy())
        k_j = jnp.asarray(k_t.detach().cpu().float().numpy())
        v_j = jnp.asarray(v_t.detach().cpu().float().numpy())
        gc_j = jnp.asarray(gc_t.detach().cpu().float().numpy())
        gs_j = jnp.asarray(gs_t.detach().cpu().float().numpy())
        gw_j = jnp.asarray(gw_t.detach().cpu().float().numpy())
        do_j = jnp.asarray(do_t.detach().cpu().float().numpy())
        ref_op = jax_base.NativeSparseAttention()

        def _fwd_fn(q_, k_, v_, gc_, gs_, gw_):
          out_, _ = ref_op._fwd(
              q_,
              k_,
              v_,
              g_cmp=gc_,
              g_slc=gs_,
              g_swa=gw_,
              chunk_size=chunk_size,
              topk=topk,
              window=window,
              cmp_block_size=cmp_block_size,
              scale=scale,
          )
          return out_

        _, vjp_fn = jax.vjp(_fwd_fn, q_j, k_j, v_j, gc_j, gs_j, gw_j)
        grads_j = vjp_fn(do_j)
        return tuple(
            torch.from_numpy(np.array(g, copy=True)).to(
                device=t.device, dtype=t.dtype
            )
            for g, t in zip(grads_j, (q_t, k_t, v_t, gc_t, gs_t, gw_t))
        )

    return _PallasCpuFn.apply(q, k, v, g_cmp, g_slc, g_swa)

  def _setup_autograd(self, pallas_module):
    ref_op = jax_base.NativeSparseAttention()

    def bwd_jax(
        q,
        k,
        v,
        g_cmp,
        g_slc,
        g_swa,
        do,
        chunk_size=256,
        topk=4,
        window=128,
        cmp_block_size=64,
        scale=None,
    ):
      head_dim = q.shape[-1]
      scale_val = scale if scale is not None else (head_dim ** -0.5)
      orig_dtype = q.dtype

      def single_bwd(carry, inputs):
        q_b, k_b, v_b, gc_b, gs_b, gw_b, do_b = inputs
        q_f = q_b.astype(jnp.float32)
        k_f = k_b.astype(jnp.float32)
        v_f = v_b.astype(jnp.float32)
        gc_f = gc_b.astype(jnp.float32)
        gs_f = gs_b.astype(jnp.float32)
        gw_f = gw_b.astype(jnp.float32)
        do_f = do_b.astype(jnp.float32)

        def loss_fn(q_, k_, v_, gc_, gs_, gw_):
          out, _ = ref_op._fwd(
              q_[None],
              k_[None],
              v_[None],
              gc_[None],
              gs_[None],
              gw_[None],
              chunk_size=chunk_size,
              topk=topk,
              window=window,
              cmp_block_size=cmp_block_size,
              scale=scale_val,
          )
          return jnp.sum(out[0].astype(jnp.float32) * do_f)

        dq_, dk_, dv_, dgc_, dgs_, dgw_ = jax.grad(
            loss_fn, argnums=(0, 1, 2, 3, 4, 5)
        )(q_f, k_f, v_f, gc_f, gs_f, gw_f)
        return carry, (
            dq_.astype(orig_dtype),
            dk_.astype(orig_dtype),
            dv_.astype(orig_dtype),
            dgc_.astype(gc_b.dtype),
            dgs_.astype(gs_b.dtype),
            dgw_.astype(gw_b.dtype),
        )

      _, (dq, dk, dv, dgc, dgs, dgw) = jax.lax.scan(
          single_bwd, None, (q, k, v, g_cmp, g_slc, g_swa, do)
      )
      return dq, dk, dv, dgc, dgs, dgw

    bwd_op = pallas_module.jax_op(f"{self.jax_op_name}_bwd", bwd_jax)

    def setup_context(ctx, inputs, output):
      del output
      ctx.save_for_backward(*inputs[:6])
      ctx.chunk_size = inputs[6] if len(inputs) > 6 else 256
      ctx.topk = inputs[7] if len(inputs) > 7 else 4
      ctx.window = inputs[8] if len(inputs) > 8 else 128
      ctx.cmp_block_size = inputs[9] if len(inputs) > 9 else 64
      ctx.scale = inputs[10] if len(inputs) > 10 else None

    def backward(ctx, do):
      saved = ctx.saved_tensors
      return bwd_op(
          *saved,
          do,
          chunk_size=ctx.chunk_size,
          topk=ctx.topk,
          window=ctx.window,
          cmp_block_size=ctx.cmp_block_size,
          scale=ctx.scale,
      )

    self._torch_tokamax_op.register_autograd(
        backward, setup_context=setup_context
    )


# Singleton instance
nsa = _PallasTpuNativeSparseAttention()
