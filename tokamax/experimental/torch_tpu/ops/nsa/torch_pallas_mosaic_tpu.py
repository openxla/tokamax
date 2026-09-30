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

from typing import Any, Optional
import jax
import jax.numpy as jnp
import torch
from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu as jax_pallas
from tokamax.experimental.torch_tpu.ops.nsa import torch_base
from typing_extensions import override


class _PallasTpuNativeSparseAttention(torch_base._NativeSparseAttention):
  """PyTorch wrapper for Pallas TPU NSA operator."""

  def __init__(self):
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuNativeSparseAttention()
    self.jax_op_name = "torch_tpu_pallas_nsa"
    try:
      from torch_tpu._internal import pallas
      self._torch_tokamax_op = pallas.jax_op(self.jax_op_name, self.op_impl_call)
      self._setup_autograd(pallas)
    except Exception:
      pass

  def _setup_autograd(self, pallas_module):
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
      scale_val = scale if scale is not None else (q.shape[-1] ** -0.5)
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
          (out, _), _ = self.op_impl_jax._fwd(
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

    self._torch_tokamax_op.register_autograd(backward, setup_context=setup_context)


# Singleton instance
nsa = _PallasTpuNativeSparseAttention()
