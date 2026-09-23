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
      self._torch_tokamax_op = pallas.jax_op(self.jax_op_name, self.op_impl_call)
    except Exception:
      pass

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
    b, t, h, d = q.shape
    if t % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({t}) must be divisible by chunk_size ({chunk_size})"
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
    b, h, t, d = q.shape
    scale = scale or (d ** -0.5)
    nb = t // chunk_size
    k_eff = min(topk, nb)
    bk_sub = (k_eff + 1) * chunk_size

    q_blocks = q.view(b, h, nb, chunk_size, d)
    k_blocks = k.view(b, h, nb, chunk_size, d)
    v_blocks = v.view(b, h, nb, chunk_size, d)

    q_mean = q_blocks.mean(dim=3)
    k_mean = k_blocks.mean(dim=3)

    routing = torch.einsum("bhqd,bhnd->bhqn", q_mean, k_mean) * scale
    q_idx = torch.arange(nb, device=q.device).unsqueeze(1)
    k_idx = torch.arange(nb, device=q.device).unsqueeze(0)
    hist_mask = k_idx < q_idx
    routing = torch.where(hist_mask.unsqueeze(0).unsqueeze(0), routing, -float("inf"))
    topk_idx = torch.topk(routing, k_eff, dim=-1).indices

    causal_diag = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool))

    outs = []
    for i in range(nb):
      q_i = q_blocks[:, :, i]
      k_diag = k_blocks[:, :, i:i+1]
      v_diag = v_blocks[:, :, i:i+1]

      if k_eff > 0:
        slot_idx = topk_idx[:, :, i, :][:, :, :, None, None]
        k_hist = torch.gather(k_blocks, 2, slot_idx.expand(b, h, k_eff, chunk_size, d))
        v_hist = torch.gather(v_blocks, 2, slot_idx.expand(b, h, k_eff, chunk_size, d))

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
      weights = torch.softmax(scores, dim=-1)
      o_i = torch.matmul(weights, v_sub)
      outs.append(o_i)

    return torch.cat(outs, dim=2)
