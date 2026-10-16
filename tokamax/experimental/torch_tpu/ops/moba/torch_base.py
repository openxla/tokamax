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

from typing import Any, Optional
import jax
import torch
from tokamax._src.ops.experimental.moba import base as jax_base
try:
  from tokamax.experimental.torch_tpu.ops import torch_op
except ImportError:
  torch_op = None
from typing_extensions import override

_CHUNK_SIZE = 256
_TOPK = 4


class _MixtureOfBlockAttention:
  """PyTorch wrapper for base Mixture of Block Attention (MoBA) operator."""

  def __init__(self):
    self.op_impl_jax = jax_base.MixtureOfBlockAttention()
    self.jax_op_name = "torch_base_moba"
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
      topk: int = _TOPK,
      chunk_size: int = _CHUNK_SIZE,
      scale: Optional[float] = None,
  ) -> jax.Array:
    (out, _), _ = self.op_impl_jax._fwd(
        q, k, v, topk=topk, chunk_size=chunk_size, scale=scale
    )
    return out

  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      scale: Optional[float] = None,
  ) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = q.shape
    del head_dim
    if seq_len % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({seq_len}) must be divisible by chunk_size ({chunk_size})"
      )

    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      out = self._torch_tokamax_op(
          q_t, k_t, v_t, topk=topk, chunk_size=chunk_size, scale=scale
      )
    else:
      out = self._cpu_fallback(q_t, k_t, v_t, chunk_size, topk, scale)

    return out.transpose(1, 2).contiguous()

  moba_pallas_attention = __call__
  flash_moba_pallas = __call__

  def _cpu_fallback(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      chunk_size: int = _CHUNK_SIZE,
      topk: int = _TOPK,
      scale: Optional[float] = None,
  ) -> torch.Tensor:
    """Pure PyTorch vectorized fallback executing block-level MoBA attention."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = scale or (head_dim ** -0.5)
    num_blocks = seq_len // chunk_size
    effective_topk = min(topk, num_blocks)
    block_k_sub = (effective_topk + 1) * chunk_size

    q_blocks = q.view(batch_size, num_heads, num_blocks, chunk_size, head_dim)
    k_blocks = k.view(batch_size, num_heads, num_blocks, chunk_size, head_dim)
    v_blocks = v.view(batch_size, num_heads, num_blocks, chunk_size, head_dim)

    q_mean = q_blocks.mean(dim=3)
    k_mean = k_blocks.mean(dim=3)

    routing = torch.einsum("bhqd,bhnd->bhqn", q_mean, k_mean) * scale
    q_idx = torch.arange(num_blocks, device=q.device).unsqueeze(1)
    k_idx = torch.arange(num_blocks, device=q.device).unsqueeze(0)
    hist_mask = k_idx < q_idx
    routing = torch.where(hist_mask.unsqueeze(0).unsqueeze(0), routing, -float("inf"))
    topk_idx = torch.topk(routing, effective_topk, dim=-1).indices

    causal_diag = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool))

    outs = []
    for block_idx in range(num_blocks):
      q_i = q_blocks[:, :, block_idx]
      k_diag = k_blocks[:, :, block_idx:block_idx+1]
      v_diag = v_blocks[:, :, block_idx:block_idx+1]

      if effective_topk > 0:
        slot_idx = topk_idx[:, :, block_idx, :][:, :, :, None, None]
        k_hist = torch.gather(
            k_blocks, 2, slot_idx.expand(batch_size, num_heads, effective_topk, chunk_size, head_dim)
        )
        v_hist = torch.gather(
            v_blocks, 2, slot_idx.expand(batch_size, num_heads, effective_topk, chunk_size, head_dim)
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
      weights = torch.softmax(scores, dim=-1)
      o_i = torch.matmul(weights, v_sub)
      outs.append(o_i)

    return torch.cat(outs, dim=2)
