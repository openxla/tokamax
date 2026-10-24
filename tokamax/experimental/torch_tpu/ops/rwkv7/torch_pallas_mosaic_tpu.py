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

"""PyTorch interface for Pallas TPU RWKV-7 operator."""

import jax
import torch
import torch.nn.functional as F
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu as jax_pallas
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu_bwd_kernel
from tokamax.experimental.torch_tpu.ops.rwkv7 import torch_base
from typing_extensions import override


class _PallasTpuRWKV7(torch_base._RWKV7):
  """PyTorch wrapper for Pallas TPU RWKV-7 operator."""

  def __init__(self):
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuRWKV7()
    self.jax_op_name = "tokamax::torch_tpu_pallas_rwkv7"
    self._torch_tokamax_op = None
    try:
      from torch_tpu._internal import pallas
    except ImportError:
      pallas = None
    else:
      try:
        self._torch_tokamax_op = pallas.jax_op(
            self.jax_op_name, self._op_impl_call_with_residuals
        )
        self._setup_autograd(pallas)
      except (ImportError, AttributeError, RuntimeError):
        self._torch_tokamax_op = None

  def _op_impl_call_with_residuals(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
      chunk_size: int = 64,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    (out, _), residuals = self.op_impl_jax._fwd(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=1.0,
        return_residuals=True,
    )
    _, _, _, _, _, _, a_ab_inv, merged_out, s_in = residuals
    return out, a_ab_inv, merged_out, s_in

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
      chunk_size: int = 64,
  ) -> jax.Array:
    (out, _), _ = self.op_impl_jax._fwd(
        q, k, v, alpha, beta, gk, chunk_size=chunk_size, scale=1.0
    )
    return out

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
      chunk_size: int = 64,
  ) -> torch.Tensor:
    seq_len = q.shape[1]
    if seq_len % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({seq_len}) must be divisible by chunk_size"
          f" ({chunk_size})"
      )

    k = F.normalize(k, dim=-1)
    gk = torch.clamp(gk, max=0.0)

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      out, _a_ab_inv, _merged_out, _s_in = self._torch_tokamax_op(
          q, k, v, alpha, beta, gk, chunk_size=chunk_size
      )
      return out
    return self._cpu_fallback(q, k, v, alpha, beta, gk, chunk_size)

  rwkv7_pallas_delta_rule = __call__

  def _setup_autograd(self, pallas_module):
    def bwd_jax(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        alpha: jax.Array,
        beta: jax.Array,
        gk: jax.Array,
        a_ab_inv: jax.Array,
        merged_out: jax.Array,
        s_in: jax.Array,
        do: jax.Array,
        chunk_size: int = 64,
    ) -> tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ]:
      is_unbatched = q.ndim == 3
      if is_unbatched:
        q, k, v, alpha, beta, gk, do = (
            q[None],
            k[None],
            v[None],
            alpha[None],
            beta[None],
            gk[None],
            do[None],
        )
      dq, dk, dv, dalpha, dbeta, dgk, _ = (
          pallas_mosaic_tpu_bwd_kernel.rwkv7_pallas_bwd(
              q,
              k,
              v,
              alpha,
              beta,
              gk,
              do,
              chunk_size=chunk_size,
              scale=1.0,
              a_ab_inv=a_ab_inv,
              merged_out=merged_out,
              s_in=s_in,
          )
      )
      if is_unbatched:
        dq, dk, dv, dalpha, dbeta, dgk = (
            dq[0],
            dk[0],
            dv[0],
            dalpha[0],
            dbeta[0],
            dgk[0],
        )
      return dq, dk, dv, dalpha, dbeta, dgk

    bwd_op = pallas_module.jax_op(f"{self.jax_op_name}_bwd", bwd_jax)

    def setup_context(ctx, inputs, output):
      _out, a_ab_inv, merged_out, s_in = output
      ctx.save_for_backward(*inputs[:6], a_ab_inv, merged_out, s_in)
      ctx.has_chunk_size_arg = len(inputs) > 6
      ctx.chunk_size = inputs[6] if ctx.has_chunk_size_arg else 64

    def backward(ctx, do, _d_a_ab_inv, _d_merged_out, _d_s_in):
      saved = ctx.saved_tensors
      grads = bwd_op(*saved, do, chunk_size=ctx.chunk_size)
      if ctx.has_chunk_size_arg:
        return (*grads, None)
      return grads

    self._torch_tokamax_op.register_autograd(
        backward, setup_context=setup_context
    )


# Singleton instance
rwkv7 = _PallasTpuRWKV7()
