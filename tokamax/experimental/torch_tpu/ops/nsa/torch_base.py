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
      self._torch_tokamax_op = pallas.jax_op(self.jax_op_name, self.op_impl_call)
    except Exception:
      pass

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
    b, t, h, d = q.shape
    if t % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({t}) must be divisible by chunk_size ({chunk_size})"
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
    b, h, t, d = q.shape
    scale = scale or (d ** -0.5)

    # 1. Coarse compression branch
    nc = t // cmp_block_size
    k_cmp = k.view(b, h, nc, cmp_block_size, d).mean(dim=3)
    v_cmp = v.view(b, h, nc, cmp_block_size, d).mean(dim=3)
    scores_cmp = torch.einsum("bhtd,bhcd->bhtc", q, k_cmp) * scale
    q_pos = torch.arange(t, device=q.device).unsqueeze(1)
    c_pos = (torch.arange(nc, device=q.device) * cmp_block_size).unsqueeze(0)
    mask_cmp = c_pos < q_pos
    scores_cmp = torch.where(mask_cmp.unsqueeze(0).unsqueeze(0), scores_cmp, -1e9)
    w_cmp = torch.softmax(scores_cmp, dim=-1)
    o_cmp = torch.einsum("bhtc,bhcd->bhtd", w_cmp, v_cmp)

    # 2. Fine-grained selection branch
    nb = t // chunk_size
    k_eff = min(topk, nb)
    bk_sub = (k_eff + 1) * chunk_size
    q_b = q.view(b, h, nb, chunk_size, d)
    k_b = k.view(b, h, nb, chunk_size, d)
    v_b = v.view(b, h, nb, chunk_size, d)
    q_mean = q_b.mean(dim=3)
    k_mean = k_b.mean(dim=3)
    routing = torch.einsum("bhqd,bhnd->bhqn", q_mean, k_mean) * scale
    q_idx = torch.arange(nb, device=q.device).unsqueeze(1)
    k_idx = torch.arange(nb, device=q.device).unsqueeze(0)
    routing = torch.where((k_idx < q_idx).unsqueeze(0).unsqueeze(0), routing, -float("inf"))
    topk_idx = torch.topk(routing, k_eff, dim=-1).indices

    causal_diag = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool))
    slc_outs = []
    for i in range(nb):
      q_i = q_b[:, :, i]
      k_diag = k_b[:, :, i:i+1]
      v_diag = v_b[:, :, i:i+1]

      if k_eff > 0:
        slot_idx = topk_idx[:, :, i, :][:, :, :, None, None]
        k_hist = torch.gather(k_b, 2, slot_idx.expand(b, h, k_eff, chunk_size, d))
        v_hist = torch.gather(v_b, 2, slot_idx.expand(b, h, k_eff, chunk_size, d))
        k_sub = torch.cat([k_diag, k_hist], dim=2).view(b, h, bk_sub, d)
        v_sub = torch.cat([v_diag, v_hist], dim=2).view(b, h, bk_sub, d)

        ranks = torch.arange(k_eff, device=q.device)
        valid_hist = (ranks < i).repeat_interleave(chunk_size).view(1, 1, 1, -1).expand(b, h, chunk_size, -1)
        mask_diag = causal_diag.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
        mask_sub = torch.cat([mask_diag, valid_hist], dim=-1)
      else:
        k_sub = k_diag.view(b, h, chunk_size, d)
        v_sub = v_diag.view(b, h, chunk_size, d)
        mask_sub = causal_diag.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)

      scores = torch.matmul(q_i, k_sub.transpose(-1, -2)) * scale
      scores = torch.where(mask_sub, scores, -1e9)
      w_slc = torch.softmax(scores, dim=-1)
      slc_outs.append(torch.matmul(w_slc, v_sub))
    o_slc = torch.cat(slc_outs, dim=2)

    # 3. Sliding window branch
    scores_swa = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
    s_pos = torch.arange(t, device=q.device).unsqueeze(0)
    mask_swa = (s_pos <= q_pos) & (s_pos >= q_pos - window)
    scores_swa = torch.where(mask_swa.unsqueeze(0).unsqueeze(0), scores_swa, -1e9)
    w_swa = torch.softmax(scores_swa, dim=-1)
    o_swa = torch.einsum("bhts,bhsd->bhtd", w_swa, v)

    return g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
