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

"""PyTorch interface for Raven Gated Slot Attention (GSA) operator."""

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension
#   tt     tile extent of T (chunk_size)
#   nt     grid extent along T: nt = T // tt

from collections.abc import Sequence
from typing import Any, TypeVar
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.raven import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
from typing_extensions import override

_Config = TypeVar("_Config")


class _RavenGsaVjp[_Config](torch_op.TorchOp[_Config]):
  """PyTorch VJP wrapper for base Raven GSA backward operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.RavenGsaVjp()
    self.jax_op_name = "base_raven_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      v: torch.Tensor | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    del out, dout, chunk_size, scale, return_residuals, config
    dv = torch.empty_like(v) if v is not None else q.new_empty(())
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(s),
        torch.empty_like(g),
        dv,
    )

  @override
  def __call__(
      self,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      v: torch.Tensor | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
      config: Any = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor | None,
  ]:
    assert self._torch_tokamax_op is not None, "Backward op not registered."
    dq, dk, ds, dg, dv_arr = self._torch_tokamax_op(
        out,
        dout,
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=False,
        config=self.deconstruct_config(config),
    )
    dv = dv_arr if v is not None else None
    return dq, dk, ds, dg, dv

  @override
  def op_impl_call(
      self,
      out: jax.Array,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      v: jax.Array | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, "Backward JAX op not set."
    grads, _ = self.op_impl_jax._fwd(
        (),
        (out, None),
        (dout, None),
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=return_residuals,
        config=None,
    )
    dv = grads.get("v")
    dv_arr = dv if dv is not None else jnp.zeros((), dtype=q.dtype)
    return grads["q"], grads["k"], grads["s"], grads["g"], dv_arr

  @override
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    if not abstract_argument_dict:
      empty_struct = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
      return {
          "residuals": (),
          "out": (empty_struct, None),
          "dout": (empty_struct, None),
      }
    s_struct = abstract_argument_dict["s"]
    v_struct = abstract_argument_dict.get("v")
    out_struct = v_struct if v_struct is not None else s_struct
    return {
        "residuals": (),
        "out": (out_struct, None),
        "dout": (out_struct, None),
    }


RavenGsaVjp = _RavenGsaVjp[Any]()


class _RavenGsa[_Config](torch_op.TorchOp[_Config]):
  """PyTorch wrapper for base Raven GSA operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.RavenGsa()
    self.backward_op_torch = RavenGsaVjp
    self.jax_op_name = "base_raven"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      v: torch.Tensor | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> torch.Tensor:
    del q, k, g, chunk_size, scale, return_residuals, config, bwd_config
    return torch.empty_like(v if v is not None else s)

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
      configs: tuple[_Config | None, _Config | None] | None = None,
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
    fwd_config, bwd_config = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    assert self.backward_op_torch is not None, "Backward op not set."
    return self._torch_tokamax_op(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=False,
        config=self.deconstruct_config(fwd_config),
        bwd_config=self.backward_op_torch.deconstruct_config(bwd_config),
    )

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
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del config, bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, _), _ = self.op_impl_jax._fwd(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=return_residuals,
        config=None,
    )
    return out

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    q, k, s, g, v, chunk_size, scale, _, _, bwd_config = inputs
    ctx.save_for_backward(q, k, s, g, v, output)
    ctx.chunk_size = chunk_size
    ctx.scale = scale
    ctx.bwd_config = bwd_config

  @override
  def backward(
      self,
      ctx: Any,
      dout: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor | None,
      None,
      None,
      None,
      None,
      None,
  ]:
    assert self.backward_op_torch is not None, "Backward op not set."
    q, k, s, g, v, out = ctx.saved_tensors
    dq, dk, ds, dg, dv = self.backward_op_torch(
        out,
        dout,
        q,
        k,
        s,
        g,
        v=v,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
        config=ctx.bwd_config,
    )
    return dq, dk, ds, dg, dv, None, None, None, None, None


# Singleton instance for the reference PyTorch op.
RavenGsa = _RavenGsa[Any]()
