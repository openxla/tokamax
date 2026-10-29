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

"""PyTorch interface for Native Sparse Attention (NSA) base operator."""

# Dimensions.
#   B   batch size
#   T   sequence length (PyTorch caller layout: [B, T, H, D])
#   H   number of attention heads
#   D   head dimension
#   W   local causal sliding window size
#   tt  query/KV block tile extent along T (chunk_size)
#   nt  number of query/KV blocks along T (nt = T // tt)
#   tc  compression block tile extent along T (cmp_block_size)
#   nc  number of compressed blocks along T (nc = T // tc)
#   tk  number of selected historical blocks (min(topk, nt))

from collections.abc import Sequence
from typing import Any
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.nsa import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
from typing_extensions import override

_CHUNK_SIZE = 256
_TOPK = 4
_WINDOW = 128
_CMP_BLOCK_SIZE = 64


class _NativeSparseAttentionVjp[_Config](torch_op.TorchOp[_Config]):
  """Native Sparse Attention (NSA) VJP Op in Torch."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.NativeSparseAttentionVjp()
    self.jax_op_name = "base_native_sparse_attention_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g_cmp: torch.Tensor,
      g_slc: torch.Tensor,
      g_swa: torch.Tensor,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      window: int = _WINDOW,
      cmp_block_size: int = _CMP_BLOCK_SIZE,
      scale: float | None = None,
      config: Sequence[int] | None = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    del dout, chunk_size, topk, window, cmp_block_size, scale, config
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g_cmp),
        torch.empty_like(g_slc),
        torch.empty_like(g_swa),
    )

  @override
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    if not abstract_argument_dict:
      return {"residuals": None, "out": None, "dout": None}
    q_meta = abstract_argument_dict["q"]
    out_struct = jax.ShapeDtypeStruct(shape=q_meta.shape, dtype=q_meta.dtype)
    return {
        "residuals": None,
        "out": out_struct,
        "dout": out_struct,
    }

  @override
  def __call__(
      self,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g_cmp: torch.Tensor,
      g_slc: torch.Tensor,
      g_swa: torch.Tensor,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      window: int = _WINDOW,
      cmp_block_size: int = _CMP_BLOCK_SIZE,
      scale: float | None = None,
      config: _Config | tuple[int, ...] | None = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(
        dout,
        q,
        k,
        v,
        g_cmp,
        g_slc,
        g_swa,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
        config=self.deconstruct_config(config),
    )

  @override
  def op_impl_call(
      self,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g_cmp: jax.Array,
      g_slc: jax.Array,
      g_swa: jax.Array,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      window: int = _WINDOW,
      cmp_block_size: int = _CMP_BLOCK_SIZE,
      scale: float | None = None,
      config: tuple[int, ...] | None = None,
  ) -> tuple[
      jax.Array,
      jax.Array,
      jax.Array,
      jax.Array,
      jax.Array,
      jax.Array,
  ]:
    del config
    assert self.op_impl_jax is not None, "Forward class not set."
    out_dummy = jnp.zeros_like(q)
    (dq, dk, dv, dg_cmp, dg_slc, dg_swa), _ = (
        self.op_impl_jax._fwd(  # pylint: disable=protected-access
            None,
            out_dummy,
            dout,
            q,
            k,
            v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            chunk_size=chunk_size,
            topk=topk,
            window=window,
            cmp_block_size=cmp_block_size,
            scale=scale,
            config=None,
            return_residuals=False,
        )
    )
    assert dg_cmp is not None and dg_slc is not None and dg_swa is not None
    return dq, dk, dv, dg_cmp, dg_slc, dg_swa


class _NativeSparseAttention[_Config](torch_op.TorchOp[_Config]):
  """PyTorch wrapper for base Native Sparse Attention (NSA) operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.NativeSparseAttention()
    self.backward_op_torch = _NativeSparseAttentionVjp()
    self.jax_op_name = "base_native_sparse_attention"
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g_cmp: torch.Tensor,
      g_slc: torch.Tensor,
      g_swa: torch.Tensor,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      window: int = _WINDOW,
      cmp_block_size: int = _CMP_BLOCK_SIZE,
      scale: float | None = None,
      config: Sequence[int] | None = None,
      bwd_config: Sequence[int] | None = None,
  ) -> torch.Tensor:
    del k, v, g_cmp, g_slc, g_swa, chunk_size, topk, window, cmp_block_size
    del scale, config, bwd_config
    return torch.empty_like(q)

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g_cmp: jax.Array,
      g_slc: jax.Array,
      g_swa: jax.Array,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      window: int = _WINDOW,
      cmp_block_size: int = _CMP_BLOCK_SIZE,
      scale: float | None = None,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del config, bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    self.op_impl_jax.bind(
        q,
        k,
        v,
        g_cmp,
        g_slc,
        g_swa,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        k,
        v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
        config=None,
        return_residuals=False,
    )
    return out

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g_cmp: torch.Tensor | None = None,
      g_slc: torch.Tensor | None = None,
      g_swa: torch.Tensor | None = None,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      window: int = _WINDOW,
      cmp_block_size: int = _CMP_BLOCK_SIZE,
      scale: float | None = None,
      configs: tuple[_Config | None, _Config | None] | None = None,
  ) -> torch.Tensor:
    if g_cmp is None:
      g_cmp = torch.full_like(q, 1.0 / 3.0)
    if g_slc is None:
      g_slc = torch.full_like(q, 1.0 / 3.0)
    if g_swa is None:
      g_swa = torch.full_like(q, 1.0 / 3.0)

    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    gc_t = g_cmp.transpose(1, 2).contiguous()
    gs_t = g_slc.transpose(1, 2).contiguous()
    gw_t = g_swa.transpose(1, 2).contiguous()

    fwd_config, bwd_config = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    assert self.backward_op_torch is not None, "Backward op not set."
    out = self._torch_tokamax_op(
        q_t,
        k_t,
        v_t,
        gc_t,
        gs_t,
        gw_t,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
        config=self.deconstruct_config(fwd_config),
        bwd_config=self.backward_op_torch.deconstruct_config(bwd_config),
    )
    return out.transpose(1, 2).contiguous()

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: torch.Tensor,
  ) -> None:
    del output
    (
        q,
        k,
        v,
        g_cmp,
        g_slc,
        g_swa,
        chunk_size,
        topk,
        window,
        cmp_block_size,
        scale,
        _,
        bwd_config,
    ) = inputs
    ctx.save_for_backward(q, k, v, g_cmp, g_slc, g_swa)
    ctx.chunk_size = chunk_size
    ctx.topk = topk
    ctx.window = window
    ctx.cmp_block_size = cmp_block_size
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
      torch.Tensor,
      torch.Tensor,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
  ]:
    assert self.backward_op_torch is not None, "Backward op not set."
    q, k, v, g_cmp, g_slc, g_swa = ctx.saved_tensors
    dq, dk, dv, dg_cmp, dg_slc, dg_swa = self.backward_op_torch(
        dout,
        q,
        k,
        v,
        g_cmp,
        g_slc,
        g_swa,
        chunk_size=ctx.chunk_size,
        topk=ctx.topk,
        window=ctx.window,
        cmp_block_size=ctx.cmp_block_size,
        scale=ctx.scale,
        config=ctx.bwd_config,
    )
    return (
        dq,
        dk,
        dv,
        dg_cmp,
        dg_slc,
        dg_swa,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


# Singleton instance of the NativeSparseAttention op.
NativeSparseAttention = _NativeSparseAttention[Any]()  # pylint: disable=invalid-name
