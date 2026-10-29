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

# Dimensions.
#   B      batch size
#   T      sequence length (nt * tt)
#   H      number of attention heads
#   K      key head dimension
#   V      value head dimension
#   tt     sequence chunk tile extent (chunk_size)
#   nt     grid extent over T: nt = T // tt

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu as jax_pallas
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu_bwd_kernel
from tokamax.experimental.torch_tpu.ops.rwkv7 import torch_base
from typing_extensions import override

Config = jax_pallas.Config


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
  t_cpu = t.detach().cpu()
  if t_cpu.dtype == torch.bfloat16:
    return jnp.asarray(t_cpu.float().numpy()).astype(jnp.bfloat16)
  return jnp.asarray(t_cpu.numpy())


def _jax_to_torch(
    arr: jax.Array, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
  if arr.dtype == jnp.bfloat16:
    np_arr = np.array(arr.astype(jnp.float32), copy=True)
    return torch.from_numpy(np_arr).to(device=device, dtype=dtype)
  return torch.from_numpy(np.array(arr, copy=True)).to(
      device=device, dtype=dtype
  )


class _PallasTpuRWKV7Vjp(torch_base._TorchOpBase[Config]):  # pylint: disable=protected-access
  """PyTorch VJP wrapper for Pallas TPU RWKV-7 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuRWKV7Vjp()
    self.jax_op_name = "torch_tpu_pallas_rwkv7_vjp"
    self.is_vjp = True

  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    if not abstract_argument_dict:
      empty = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
      return {"residuals": (empty, empty, empty), "out": empty, "dout": empty}
    q_meta = abstract_argument_dict["q"]
    v_meta = abstract_argument_dict["v"]
    b_sz, seq_len, h_sz, k_dim = q_meta.shape
    _, _, _, v_dim = v_meta.shape
    tt = abstract_argument_dict.get("chunk_size", 64)
    nt = seq_len // tt
    return {
        "residuals": (
            jax.ShapeDtypeStruct((b_sz, h_sz, nt, tt, tt), q_meta.dtype),
            jax.ShapeDtypeStruct(
                (b_sz, h_sz, nt, tt, k_dim + v_dim), jnp.float32
            ),
            jax.ShapeDtypeStruct((b_sz, h_sz, nt, k_dim, v_dim), jnp.float32),
        ),
        "out": (v_meta, None),
        "dout": (v_meta, None),
    }

  @override
  def op_impl_call(
      self,
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
      scale: float | None = None,
  ) -> tuple[
      jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
  ]:
    assert self.op_impl_jax is not None, "Backward class not set."
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
            scale=scale,
            a_ab_inv=a_ab_inv,
            merged_out=merged_out,
            s_in=s_in,
        )
    )
    return dq, dk, dv, dalpha, dbeta, dgk

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
      a_ab_inv: torch.Tensor,
      merged_out: torch.Tensor,
      s_in: torch.Tensor,
      do: torch.Tensor,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(
          q,
          k,
          v,
          alpha,
          beta,
          gk,
          a_ab_inv,
          merged_out,
          s_in,
          do,
          chunk_size=chunk_size,
          scale=scale,
      )
    grads_jax = self.op_impl_call(
        _torch_to_jax(q),
        _torch_to_jax(k),
        _torch_to_jax(v),
        _torch_to_jax(alpha),
        _torch_to_jax(beta),
        _torch_to_jax(gk),
        _torch_to_jax(a_ab_inv),
        _torch_to_jax(merged_out),
        _torch_to_jax(s_in),
        _torch_to_jax(do),
        chunk_size=chunk_size,
        scale=scale,
    )
    return (
        _jax_to_torch(grads_jax[0], q.device, q.dtype),
        _jax_to_torch(grads_jax[1], k.device, k.dtype),
        _jax_to_torch(grads_jax[2], v.device, v.dtype),
        _jax_to_torch(grads_jax[3], alpha.device, alpha.dtype),
        _jax_to_torch(grads_jax[4], beta.device, beta.dtype),
        _jax_to_torch(grads_jax[5], gk.device, gk.dtype),
    )


class _CpuPallasAutogradFn(torch.autograd.Function):
  """Routes CPU execution through `op_impl_call` and `backward_op_torch`."""

  @staticmethod
  def forward(
      ctx: Any,
      op_self: Any,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
      chunk_size: int,
      scale: float | None,
  ) -> torch.Tensor:
    out_jax, a_ab_inv_jax, merged_out_jax, s_in_jax = op_self.op_impl_call(
        _torch_to_jax(q),
        _torch_to_jax(k),
        _torch_to_jax(v),
        _torch_to_jax(alpha),
        _torch_to_jax(beta),
        _torch_to_jax(gk),
        chunk_size=chunk_size,
        scale=scale,
    )
    out = _jax_to_torch(out_jax, q.device, q.dtype)
    a_ab_inv_t = _jax_to_torch(a_ab_inv_jax, q.device, q.dtype)
    merged_out_t = _jax_to_torch(merged_out_jax, q.device, torch.float32)
    s_in_t = _jax_to_torch(s_in_jax, q.device, torch.float32)
    ctx.save_for_backward(
        q, k, v, alpha, beta, gk, a_ab_inv_t, merged_out_t, s_in_t
    )
    ctx.op_self = op_self
    ctx.chunk_size = chunk_size
    ctx.scale = scale
    return out

  @staticmethod
  def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
    q, k, v, alpha, beta, gk, a_ab_inv_t, merged_out_t, s_in_t = (
        ctx.saved_tensors
    )
    dq, dk, dv, dalpha, dbeta, dgk = ctx.op_self.backward_op_torch(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        a_ab_inv_t,
        merged_out_t,
        s_in_t,
        do,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
    )
    return None, dq, dk, dv, dalpha, dbeta, dgk, None, None


class _PallasTpuRWKV7(torch_base._RWKV7[Config]):  # pylint: disable=protected-access
  """PyTorch wrapper for Pallas TPU RWKV-7 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuRWKV7()
    self.backward_op_torch = _PallasTpuRWKV7Vjp()
    self.jax_op_name = "torch_tpu_pallas_rwkv7"

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
      scale: float | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, _), residuals = self.op_impl_jax._fwd(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=True,
    )
    _, _, _, _, _, _, a_ab_inv, merged_out, s_in = residuals
    return out, a_ab_inv, merged_out, s_in

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
      scale: float | None = None,
  ) -> torch.Tensor:
    _, seq_len, _, _ = q.shape
    if chunk_size <= 0 or seq_len % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({seq_len}) must be divisible by chunk_size"
          f" ({chunk_size})"
      )

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      out, _a_ab_inv, _merged_out, _s_in = self._torch_tokamax_op(
          q, k, v, alpha, beta, gk, chunk_size=chunk_size, scale=scale
      )
      return out
    return _CpuPallasAutogradFn.apply(
        self, q, k, v, alpha, beta, gk, chunk_size, scale
    )

  def setup_context(
      self, ctx: Any, inputs: Sequence[Any], output: Any
  ) -> None:
    out, a_ab_inv, merged_out, s_in = output
    del out
    ctx.save_for_backward(*inputs[:6], a_ab_inv, merged_out, s_in)
    ctx.chunk_size = inputs[6] if len(inputs) > 6 else 64
    ctx.scale = inputs[7] if len(inputs) > 7 else None

  def backward(
      self,
      ctx: Any,
      do: torch.Tensor,
      _d_a_ab_inv: Any = None,
      _d_merged_out: Any = None,
      _d_s_in: Any = None,
  ) -> tuple[Any, ...]:
    assert self.backward_op_torch is not None, "Backward op not set."
    q, k, v, alpha, beta, gk, a_ab_inv, merged_out, s_in = ctx.saved_tensors
    grads = self.backward_op_torch(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        a_ab_inv,
        merged_out,
        s_in,
        do,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
    )
    return (*grads, None, None)


# Singleton instance
rwkv7 = _PallasTpuRWKV7()
object.__setattr__(rwkv7, "rwkv7_pallas_delta_rule", rwkv7)
