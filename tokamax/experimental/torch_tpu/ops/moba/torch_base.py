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

"""PyTorch interface for Mixture of Block Attention (MoBA) base operator."""

from collections.abc import Sequence
from typing import Any, Optional, TypeVar
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.moba import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
from typing_extensions import override

_Config = TypeVar("_Config")

_CHUNK_SIZE = 256
_TOPK = 4


class _MixtureOfBlockAttentionVjp(torch_op.TorchOp[_Config]):
  """PyTorch VJP wrapper for base Mixture of Block Attention (MoBA) operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.MixtureOfBlockAttentionVjp()
    self.jax_op_name = "base_moba_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      topk: int = _TOPK,
      chunk_size: int = _CHUNK_SIZE,
      scale: Optional[float] = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del out, dout, topk, chunk_size, scale, return_residuals, config
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
    )

  @override
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    if not abstract_argument_dict:
      empty_struct = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
      return {
          "residuals": None,
          "out": empty_struct,
          "dout": empty_struct,
      }
    q = abstract_argument_dict["q"]
    out_struct = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    return {
        "residuals": None,
        "out": out_struct,
        "dout": out_struct,
    }

  @override
  def __call__(
      self,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      topk: int = _TOPK,
      chunk_size: int = _CHUNK_SIZE,
      scale: Optional[float] = None,
      config: Any = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (
        self._torch_tokamax_op is not None
    ), "Forward op not registered. This means that self.op_impl_jax is not set"
    " in the constructor."
    return self._torch_tokamax_op(
        out,
        dout,
        q,
        k,
        v,
        topk=topk,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=False,
        config=self.deconstruct_config(config),
    )

  @override
  def op_impl_call(
      self,
      out: jax.Array,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      topk: int = _TOPK,
      chunk_size: int = _CHUNK_SIZE,
      scale: Optional[float] = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, "Forward class not set."
    grads, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        None,
        out,
        dout,
        q,
        k,
        v,
        topk=topk,
        chunk_size=chunk_size,
        scale=scale,
        config=None,
        return_residuals=return_residuals,
    )
    return grads


MixtureOfBlockAttentionVjp = _MixtureOfBlockAttentionVjp[Any]()


class _MixtureOfBlockAttention(torch_op.TorchOp[_Config]):
  """PyTorch wrapper for base Mixture of Block Attention (MoBA) operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.MixtureOfBlockAttention()
    self.backward_op_torch = MixtureOfBlockAttentionVjp
    self.jax_op_name = "base_moba"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      topk: int = _TOPK,
      chunk_size: int = _CHUNK_SIZE,
      scale: Optional[float] = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> torch.Tensor:
    del k, v, topk, chunk_size, scale, return_residuals, config, bwd_config
    return torch.empty_like(q)

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      topk: int = _TOPK,
      chunk_size: int = _CHUNK_SIZE,
      scale: Optional[float] = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del config, bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        k,
        v,
        topk=topk,
        chunk_size=chunk_size,
        scale=scale,
        config=None,
        return_residuals=return_residuals,
    )
    return out

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      scale: Optional[float] = None,
      configs: tuple[_Config | None, _Config | None] | None = None,
  ) -> torch.Tensor:
    fwd_config, bwd_config = (None, None) if configs is None else configs
    batch_size, seq_len, num_heads, head_dim = q.shape
    del batch_size, num_heads, head_dim
    resolved_chunk_size = (
        getattr(fwd_config, "chunk_size", chunk_size)
        if fwd_config is not None
        else chunk_size
    )
    if seq_len % resolved_chunk_size != 0:
      raise ValueError(
          f"Sequence length ({seq_len}) must be divisible by chunk_size"
          f" ({resolved_chunk_size})"
      )

    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    assert (
        self._torch_tokamax_op is not None
    ), "Forward op not registered. This means that self.op_impl_jax is not set"
    " in the constructor."
    assert self.backward_op_torch is not None, "Backward op not set."
    out = self._torch_tokamax_op(
        q_t,
        k_t,
        v_t,
        topk=topk,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=False,
        config=self.deconstruct_config(fwd_config),
        bwd_config=self.backward_op_torch.deconstruct_config(bwd_config),
    )
    return out.transpose(1, 2).contiguous()

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    """Saves tensors and metadata to the context for the backward pass."""
    q, k, v, topk, chunk_size, scale, _, _, bwd_config = inputs
    ctx.save_for_backward(q, k, v, output)
    ctx.topk = topk
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
      None,
      None,
      None,
      None,
      None,
      None,
  ]:
    """Callback for torch register_autograd."""
    assert self.backward_op_torch is not None, "Backward op not set."
    q, k, v, out = ctx.saved_tensors
    dq, dk, dv = self.backward_op_torch(
        out,
        dout,
        q,
        k,
        v,
        topk=ctx.topk,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
        config=ctx.bwd_config,
    )
    return dq, dk, dv, None, None, None, None, None, None


MixtureOfBlockAttention = _MixtureOfBlockAttention[Any]()
