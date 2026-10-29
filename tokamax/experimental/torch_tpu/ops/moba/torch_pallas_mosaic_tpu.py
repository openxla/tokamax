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

"""PyTorch interface for Pallas TPU Mixture of Block Attention (MoBA) operator."""

from typing import Any, Optional
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.moba import base as jax_base
from tokamax._src.ops.experimental.moba import pallas_mosaic_tpu as jax_pallas
from tokamax.experimental.torch_tpu.ops.moba import torch_base
import torch
from typing_extensions import override


def _pallas_moba_cpu_fallback(
    op_impl_jax: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = torch_base._CHUNK_SIZE,
    topk: int = torch_base._TOPK,
    scale: Optional[float] = None,
) -> torch.Tensor:
  """Executes the Pallas TPU MoBA kernel in interpret mode via JAX."""

  class _PallasCpuFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_t, k_t, v_t):
      ctx.save_for_backward(q_t, k_t, v_t)
      q_j = jnp.asarray(q_t.detach().cpu().numpy())
      k_j = jnp.asarray(k_t.detach().cpu().numpy())
      v_j = jnp.asarray(v_t.detach().cpu().numpy())
      out_j, _ = op_impl_jax._fwd(
          q_j,
          k_j,
          v_j,
          topk=topk,
          chunk_size=chunk_size,
          scale=scale,
      )
      return torch.from_numpy(np.array(out_j, copy=True)).to(
          device=q_t.device, dtype=q_t.dtype
      )

    @staticmethod
    def backward(ctx, do_t):
      q_t, k_t, v_t = ctx.saved_tensors
      q_j = jnp.asarray(q_t.detach().cpu().float().numpy())
      k_j = jnp.asarray(k_t.detach().cpu().float().numpy())
      v_j = jnp.asarray(v_t.detach().cpu().float().numpy())
      do_j = jnp.asarray(do_t.detach().cpu().float().numpy())
      ref_op = jax_base.MixtureOfBlockAttention()

      def _fwd_fn(q_, k_, v_):
        out_, _ = ref_op._fwd(
            q_,
            k_,
            v_,
            topk=topk,
            chunk_size=chunk_size,
            scale=scale,
        )
        return out_

      _, vjp_fn = jax.vjp(_fwd_fn, q_j, k_j, v_j)
      grads_j = vjp_fn(do_j)
      return tuple(
          torch.from_numpy(np.array(g, copy=True)).to(
              device=t.device, dtype=t.dtype
          )
          for g, t in zip(grads_j, (q_t, k_t, v_t))
      )

  return _PallasCpuFn.apply(q, k, v)


def _register_moba_autograd(
    torch_tokamax_op: Any, jax_op_name: str, pallas_module: Any
) -> None:
  """Registers reverse-mode autograd VJP on the MoBA custom op."""
  ref_op = jax_base.MixtureOfBlockAttention()

  def bwd_jax(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      do: jax.Array,
      topk: int = 4,
      chunk_size: int = 256,
      scale: Optional[float] = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    _, _, _, head_dim = q.shape
    scale_val = scale if scale is not None else (head_dim**-0.5)
    orig_dtype = q.dtype

    def single_bwd(carry, inputs):
      q_b, k_b, v_b, do_b = inputs
      q_f = q_b.astype(jnp.float32)
      k_f = k_b.astype(jnp.float32)
      v_f = v_b.astype(jnp.float32)
      do_f = do_b.astype(jnp.float32)

      def loss_fn(q_, k_, v_):
        out, _ = ref_op._fwd(
            q_[None],
            k_[None],
            v_[None],
            topk=topk,
            chunk_size=chunk_size,
            scale=scale_val,
        )
        return jnp.sum(out[0].astype(jnp.float32) * do_f)

      dq_, dk_, dv_ = jax.grad(loss_fn, argnums=(0, 1, 2))(q_f, k_f, v_f)
      return carry, (
          dq_.astype(orig_dtype),
          dk_.astype(orig_dtype),
          dv_.astype(orig_dtype),
      )

    _, (dq, dk, dv) = jax.lax.scan(single_bwd, None, (q, k, v, do))
    return dq, dk, dv

  bwd_name = (
      f"{jax_op_name}_bwd"
      if "::" in jax_op_name
      else f"tokamax::{jax_op_name}_bwd"
  )
  bwd_op = pallas_module.jax_op(bwd_name, bwd_jax)

  def setup_context(ctx, inputs, output):
    del output
    ctx.save_for_backward(*inputs[:3])
    ctx.topk = inputs[3] if len(inputs) > 3 else 4
    ctx.chunk_size = inputs[4] if len(inputs) > 4 else 256
    ctx.scale = inputs[5] if len(inputs) > 5 else None

  def backward(ctx, do):
    saved = ctx.saved_tensors
    return bwd_op(
        *saved,
        do,
        topk=ctx.topk,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
    )

  torch_tokamax_op.register_autograd(backward, setup_context=setup_context)


class _PallasTpuMixtureOfBlockAttention(torch_base._MixtureOfBlockAttention):
  """PyTorch wrapper for Pallas TPU MoBA operator."""

  def __init__(self):
    super().__init__(
        op_impl_jax=jax_pallas.PallasTpuMixtureOfBlockAttention(
            bypass_device_check=True
        ),
        jax_op_name="torch_tpu_pallas_moba",
    )

  @override
  def _register_ops(self) -> None:
    if hasattr(super(), "_register_ops"):
      super()._register_ops()
    try:
      from torch_tpu._internal import pallas  # pylint: disable=g-import-not-at-top
    except ImportError:
      pass
    else:
      if self._torch_tokamax_op is not None:
        _register_moba_autograd(
            self._torch_tokamax_op, self.jax_op_name, pallas
        )

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      chunk_size: int = torch_base._CHUNK_SIZE,
      topk: int = torch_base._TOPK,
      scale: Optional[float] = None,
  ) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = q.shape
    del batch_size, num_heads, head_dim
    if seq_len % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({seq_len}) must be divisible by chunk_size"
          f" ({chunk_size})"
      )

    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      out = self._torch_tokamax_op(
          q_t, k_t, v_t, topk=topk, chunk_size=chunk_size, scale=scale
      )
    else:
      out = _pallas_moba_cpu_fallback(
          self.op_impl_jax, q_t, k_t, v_t, chunk_size, topk, scale
      )

    return out.transpose(1, 2).contiguous()


# Singleton instance
moba = _PallasTpuMixtureOfBlockAttention()
