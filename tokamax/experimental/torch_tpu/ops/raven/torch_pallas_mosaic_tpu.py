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

"""PyTorch interface for Pallas TPU Raven GSA operator."""

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension
#   tt     tile extent of T (chunk_size)

from collections.abc import Sequence
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu as jax_pallas
from tokamax.experimental.torch_tpu.ops.raven import torch_base
import torch
from typing_extensions import override

Config = jax_pallas.Config


def _torch_to_jax(t: torch.Tensor | None) -> jax.Array | None:
  if t is None:
    return None
  if t.dtype == torch.bfloat16:
    arr = t.detach().cpu().to(torch.float32).numpy()
    return jnp.asarray(arr, dtype=jnp.bfloat16)
  return jnp.asarray(t.detach().cpu().numpy())


def _jax_to_torch(
    a: jax.Array | None, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
  if a is None:
    return None
  np_arr = np.array(
      a,
      dtype=np.float32 if dtype == torch.bfloat16 else None,
      copy=True,
  )
  return torch.as_tensor(np_arr, device=device).to(dtype=dtype)


class _PallasTpuRavenGSAVjp(torch_base._TorchOpBase[Config]):  # pylint: disable=protected-access
  """PyTorch VJP wrapper for Pallas TPU Raven GSA backward operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuRavenGSAVjp()
    self.jax_op_name = "torch_tpu_pallas_raven_gsa_vjp"
    self.is_vjp = True

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      grad_out: jax.Array,
      v: jax.Array | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    assert self.op_impl_jax is not None, "Backward JAX op not set."
    residuals = (
        q,
        k,
        s,
        g,
        v,
        None,
        None,
        None,
        None,
        chunk_size,
        scale,
    )
    grads, _ = self.op_impl_jax._fwd(
        residuals,
        (grad_out, None),
        (grad_out, None),
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        config=Config(chunk_size=chunk_size),
    )
    return (
        grads["q"],
        grads["k"],
        grads["s"],
        grads["g"],
        grads.get("v"),
    )

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      grad_out: torch.Tensor,
      v: torch.Tensor | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor | None,
  ]:
    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(
          q, k, s, g, grad_out, v, chunk_size=chunk_size, scale=scale
      )
    dq, dk, ds, dg, dv = self.op_impl_call(
        _torch_to_jax(q),
        _torch_to_jax(k),
        _torch_to_jax(s),
        _torch_to_jax(g),
        _torch_to_jax(grad_out),
        v=_torch_to_jax(v),
        chunk_size=chunk_size,
        scale=scale,
    )
    return (
        _jax_to_torch(dq, q.device, q.dtype),  # type: ignore[return-value]
        _jax_to_torch(dk, k.device, k.dtype),  # type: ignore[return-value]
        _jax_to_torch(ds, s.device, s.dtype),  # type: ignore[return-value]
        _jax_to_torch(dg, g.device, g.dtype),  # type: ignore[return-value]
        _jax_to_torch(dv, v.device, v.dtype) if v is not None else None,
    )

  @override
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    if not abstract_argument_dict:
      return {
          "residuals": (),
          "out": (),
          "dout": (),
      }
    q_struct = abstract_argument_dict["q"]
    s_struct = abstract_argument_dict["s"]
    v_struct = abstract_argument_dict.get("v")
    out_struct = v_struct if v_struct is not None else s_struct
    return {
        "residuals": (),
        "out": (out_struct, None),
        "dout": (out_struct, None),
        "q": q_struct,
    }


class _RavenPallasInterpretAutograd(torch.autograd.Function):
  """Autograd wrapper executing the Pallas forward/backward ops via JAX."""

  @staticmethod
  def forward(
      ctx: Any,
      owner: "_PallasTpuRavenGSA",
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      v: torch.Tensor | None,
      chunk_size: int,
      scale: float | None,
  ) -> torch.Tensor:
    q_jax = _torch_to_jax(q)
    k_jax = _torch_to_jax(k)
    s_jax = _torch_to_jax(s)
    g_jax = _torch_to_jax(g)
    v_jax = _torch_to_jax(v)
    assert owner.op_impl_jax is not None
    (out_jax, _), residuals = owner.op_impl_jax._fwd(
        q_jax,
        k_jax,
        s_jax,
        g_jax,
        v_jax,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=True,
        config=Config(chunk_size=chunk_size),
    )
    ctx.owner = owner
    ctx.residuals = residuals
    ctx.chunk_size = chunk_size
    ctx.scale = scale
    ctx.has_v = v is not None
    ctx.q_meta = (q.device, q.dtype)
    ctx.k_meta = (k.device, k.dtype)
    ctx.s_meta = (s.device, s.dtype)
    ctx.g_meta = (g.device, g.dtype)
    ctx.v_meta = (v.device, v.dtype) if v is not None else None
    out_dtype = v.dtype if v is not None else s.dtype
    out_tensor = _jax_to_torch(out_jax, q.device, out_dtype)
    assert out_tensor is not None
    return out_tensor

  @staticmethod
  def backward(
      ctx: Any, grad_out: torch.Tensor
  ) -> tuple[
      None,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor | None,
      None,
      None,
  ]:
    owner: _PallasTpuRavenGSA = ctx.owner
    assert owner.backward_op_torch is not None
    assert owner.backward_op_torch.op_impl_jax is not None
    res_q, res_k, res_s, res_g, res_v = ctx.residuals[:5]
    grad_out_jax = _torch_to_jax(grad_out)
    assert grad_out_jax is not None
    grads, _ = owner.backward_op_torch.op_impl_jax._fwd(
        ctx.residuals,
        (grad_out_jax, None),
        (grad_out_jax, None),
        res_q,
        res_k,
        res_s,
        res_g,
        res_v,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
        config=Config(chunk_size=ctx.chunk_size),
    )
    dq_jax = grads["q"]
    dk_jax = grads["k"]
    ds_jax = grads["s"]
    dg_jax = grads["g"]
    dv_jax = grads.get("v")
    dq = _jax_to_torch(dq_jax, *ctx.q_meta)
    dk = _jax_to_torch(dk_jax, *ctx.k_meta)
    ds = _jax_to_torch(ds_jax, *ctx.s_meta)
    dg = _jax_to_torch(dg_jax, *ctx.g_meta)
    dv = (
        _jax_to_torch(dv_jax, *ctx.v_meta)
        if ctx.has_v and ctx.v_meta is not None
        else None
    )
    assert (
        dq is not None and dk is not None and ds is not None and dg is not None
    )
    return None, dq, dk, ds, dg, dv, None, None


class _PallasTpuRavenGSA(torch_base._RavenGSA[Config]):
  """PyTorch wrapper for Pallas TPU Raven GSA operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuRavenGSA()
    self.jax_op_name = "torch_tpu_pallas_raven_gsa"
    self.backward_op_torch = _PallasTpuRavenGSAVjp()
    self.is_vjp = False

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      v: jax.Array | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> jax.Array:
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, _), _ = self.op_impl_jax._fwd(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=False,
        config=Config(chunk_size=chunk_size),
    )
    return out

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    del output
    q, k, s, g = inputs[:4]
    v = inputs[4] if len(inputs) > 4 else None
    chunk_size = inputs[5] if len(inputs) > 5 else 64
    scale = inputs[6] if len(inputs) > 6 else None
    if v is not None:
      ctx.save_for_backward(q, k, s, g, v)
      ctx.has_v = True
    else:
      ctx.save_for_backward(q, k, s, g)
      ctx.has_v = False
    ctx.chunk_size = chunk_size
    ctx.scale = scale

  @override
  def backward(
      self,
      ctx: Any,
      grad_output: torch.Tensor,
  ) -> tuple[Any, ...]:
    assert self.backward_op_torch is not None, "Backward op not set."
    saved = ctx.saved_tensors
    if ctx.has_v:
      q, k, s, g, v = saved
    else:
      q, k, s, g = saved
      v = None
    dq, dk, ds, dg, dv = self.backward_op_torch(
        q,
        k,
        s,
        g,
        grad_output,
        v=v,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
    )
    return dq, dk, ds, dg, dv, None, None

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      v: torch.Tensor | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> torch.Tensor:
    if q.ndim != 4:
      raise ValueError(
          f"Expected 4D input (batch, seq_len, heads, dim), got {q.shape}."
      )
    _, T, _, _ = q.shape
    if chunk_size <= 0 or T % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({T}) must be divisible by positive"
          f" chunk_size ({chunk_size})."
      )

    g = torch.clamp(g, max=0.0)

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(
          q, k, s, g, v, chunk_size=chunk_size, scale=scale
      )
    return _RavenPallasInterpretAutograd.apply(
        self, q, k, s, g, v, chunk_size, scale
    )


# Singleton instance
raven_gsa = _PallasTpuRavenGSA()
object.__setattr__(raven_gsa, "raven_pallas_gsa", raven_gsa)
