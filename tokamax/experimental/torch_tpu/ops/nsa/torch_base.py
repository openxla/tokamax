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

from typing import Any, Optional
import jax
import torch
from tokamax._src.ops.experimental.nsa import base as jax_base
try:
  from tokamax.experimental.torch_tpu.ops import torch_op
except ImportError:
  torch_op = None
from typing_extensions import override

_CHUNK_SIZE = 256
_TOPK = 4
_WINDOW = 128
_CMP_BLOCK_SIZE = 64


class _NativeSparseAttention:
  """PyTorch wrapper for base Native Sparse Attention (NSA) operator."""

  def __init__(self):
    self.op_impl_jax = jax_base.NativeSparseAttention()
    self.jax_op_name = "torch_base_nsa"
    self._torch_tokamax_op = None
    try:
      from torch_tpu._internal import pallas
    except ImportError:
      pallas = None
    else:
      try:
        self._torch_tokamax_op = pallas.jax_op(self.jax_op_name, self.op_impl_call)
      except Exception:
        self._torch_tokamax_op = None

  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g_cmp: Optional[jax.Array] = None,
      g_slc: Optional[jax.Array] = None,
      g_swa: Optional[jax.Array] = None,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      window: int = _WINDOW,
      cmp_block_size: int = _CMP_BLOCK_SIZE,
      scale: Optional[float] = None,
  ) -> jax.Array:
    (out, _), _ = self.op_impl_jax._fwd(
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
    )
    return out

  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g_cmp: Optional[torch.Tensor] = None,
      g_slc: Optional[torch.Tensor] = None,
      g_swa: Optional[torch.Tensor] = None,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      window: int = _WINDOW,
      cmp_block_size: int = _CMP_BLOCK_SIZE,
      scale: Optional[float] = None,
  ) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = q.shape
    del batch_size, num_heads, head_dim
    if seq_len % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({seq_len}) must be divisible by chunk_size ({chunk_size})"
      )

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

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
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
      )
    else:
      out = self._cpu_fallback(
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
      )

    return out.transpose(1, 2).contiguous()

  nsa_pallas_attention = __call__
  flash_nsa_pallas = __call__

  def _cpu_fallback(
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
      scale: Optional[float] = None,
  ) -> torch.Tensor:
    """Pure PyTorch fallback evaluating the three NSA branches."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = scale or (head_dim ** -0.5)

    # 1. Coarse compression branch
    num_cmp_blocks = seq_len // cmp_block_size
    k_cmp = k.view(batch_size, num_heads, num_cmp_blocks, cmp_block_size, head_dim).mean(dim=3)
    v_cmp = v.view(batch_size, num_heads, num_cmp_blocks, cmp_block_size, head_dim).mean(dim=3)
    scores_cmp = torch.einsum("bhtd,bhcd->bhtc", q, k_cmp) * scale
    q_pos = torch.arange(seq_len, device=q.device).unsqueeze(1)
    c_pos = (torch.arange(num_cmp_blocks, device=q.device) * cmp_block_size).unsqueeze(0)
    mask_cmp = c_pos < q_pos
    scores_cmp = torch.where(mask_cmp.unsqueeze(0).unsqueeze(0), scores_cmp, -1e9)
    w_cmp = torch.softmax(scores_cmp, dim=-1)
    o_cmp = torch.einsum("bhtc,bhcd->bhtd", w_cmp, v_cmp)

    # 2. Fine-grained selection branch
    num_blocks = seq_len // chunk_size
    effective_topk = min(topk, num_blocks)
    block_k_sub = (effective_topk + 1) * chunk_size
    q_b = q.view(batch_size, num_heads, num_blocks, chunk_size, head_dim)
    k_b = k.view(batch_size, num_heads, num_blocks, chunk_size, head_dim)
    v_b = v.view(batch_size, num_heads, num_blocks, chunk_size, head_dim)
    q_mean = q_b.mean(dim=3)
    k_mean = k_b.mean(dim=3)
    routing = torch.einsum("bhqd,bhnd->bhqn", q_mean, k_mean) * scale
    q_idx = torch.arange(num_blocks, device=q.device).unsqueeze(1)
    k_idx = torch.arange(num_blocks, device=q.device).unsqueeze(0)
    routing = torch.where((k_idx < q_idx).unsqueeze(0).unsqueeze(0), routing, -float("inf"))
    topk_idx = torch.topk(routing, effective_topk, dim=-1).indices

    causal_diag = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool))
    slc_outs = []
    for block_idx in range(num_blocks):
      q_i = q_b[:, :, block_idx]
      k_diag = k_b[:, :, block_idx:block_idx+1]
      v_diag = v_b[:, :, block_idx:block_idx+1]

      if effective_topk > 0:
        slot_idx = topk_idx[:, :, block_idx, :][:, :, :, None, None]
        k_hist = torch.gather(
            k_b, 2, slot_idx.expand(batch_size, num_heads, effective_topk, chunk_size, head_dim)
        )
        v_hist = torch.gather(
            v_b, 2, slot_idx.expand(batch_size, num_heads, effective_topk, chunk_size, head_dim)
        )
        k_sub = torch.cat([k_diag, k_hist], dim=2).view(batch_size, num_heads, block_k_sub, head_dim)
        v_sub = torch.cat([v_diag, v_hist], dim=2).view(batch_size, num_heads, block_k_sub, head_dim)

        ranks = torch.arange(effective_topk, device=q.device)
        valid_hist = (ranks < block_idx).repeat_interleave(chunk_size).view(1, 1, 1, -1).expand(batch_size, num_heads, chunk_size, -1)
        mask_diag = causal_diag.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        mask_sub = torch.cat([mask_diag, valid_hist], dim=-1)
      else:
        k_sub = k_diag.view(batch_size, num_heads, chunk_size, head_dim)
        v_sub = v_diag.view(batch_size, num_heads, chunk_size, head_dim)
        mask_sub = causal_diag.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

      scores = torch.matmul(q_i, k_sub.transpose(-1, -2)) * scale
      scores = torch.where(mask_sub, scores, -1e9)
      w_slc = torch.softmax(scores, dim=-1)
      slc_outs.append(torch.matmul(w_slc, v_sub))
    o_slc = torch.cat(slc_outs, dim=2)

    # 3. Sliding window branch
    scores_swa = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
    s_pos = torch.arange(seq_len, device=q.device).unsqueeze(0)
    mask_swa = (s_pos <= q_pos) & (s_pos >= q_pos - window)
    scores_swa = torch.where(mask_swa.unsqueeze(0).unsqueeze(0), scores_swa, -1e9)
    w_swa = torch.softmax(scores_swa, dim=-1)
    o_swa = torch.einsum("bhts,bhsd->bhtd", w_swa, v)

    return g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
