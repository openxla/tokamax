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

"""PyTorch interface for Gated Delta Net 2 (GDN-2) operator."""

from typing import Any
import jax
import torch
import torch.nn.functional as F
from tokamax._src.ops.experimental.gdn2 import base as jax_base
try:
  from tokamax.experimental.torch_tpu.ops import torch_op
except ImportError:
  torch_op = None
from typing_extensions import override


class _GatedDeltaNet2:
  """PyTorch wrapper for base GDN-2 operator."""

  def __init__(self):
    self.op_impl_jax = jax_base.GatedDeltaNet2()
    self.jax_op_name = "torch_base_gdn2"
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
      g: jax.Array,
      b: jax.Array,
      w: jax.Array,
      chunk_size: int = 64,
  ) -> jax.Array:
    (out, _), _ = self.op_impl_jax._fwd(
        q, k, v, g, b, w, chunk_size=chunk_size
    )
    return out

  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      b: torch.Tensor,
      w: torch.Tensor,
      chunk_size: int = 64,
  ) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = q.shape
    del batch_size, num_heads, head_dim
    if seq_len % chunk_size != 0:
      raise ValueError(f"Sequence length ({seq_len}) must be divisible by chunk_size ({chunk_size})")

    g = torch.clamp(g, max=0.0)
    b = torch.clamp(b, min=0.0, max=1.0)

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(q, k, v, g, b, w, chunk_size=chunk_size)
    return self._cpu_fallback(q, k, v, g, b, w, chunk_size)

  gdn2_pallas_delta_rule = __call__

  def _cpu_fallback(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      b: torch.Tensor,
      w: torch.Tensor,
      chunk_size: int = 64,
  ) -> torch.Tensor:
    """Pure PyTorch fallback for GDN-2."""
    batch_size, seq_len, num_heads, key_dim = q.shape
    val_dim = v.shape[-1]
    num_chunks = seq_len // chunk_size
    outs = []

    state = torch.zeros(batch_size, num_heads, key_dim, val_dim, device=q.device, dtype=torch.float32)
    strictly_lower = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device), diagonal=-1)
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device), diagonal=0)
    eye = torch.eye(chunk_size, device=q.device)

    for chunk_idx in range(num_chunks):
      st = chunk_idx * chunk_size
      en = (chunk_idx + 1) * chunk_size
      qc = q[:, st:en].permute(0, 2, 1, 3).float()
      kc = k[:, st:en].permute(0, 2, 1, 3).float()
      vc = v[:, st:en].permute(0, 2, 1, 3).float()
      gc = g[:, st:en].permute(0, 2, 1, 3).float()
      bc = b[:, st:en].permute(0, 2, 1, 3).float()
      wc = w[:, st:en].permute(0, 2, 1, 3).float()

      gcs = torch.cumsum(gc, dim=2)
      bk = bc * kc
      diff = torch.clamp(gcs.unsqueeze(3) - gcs.unsqueeze(2), max=0.0)
      decay = torch.exp(diff)
      t_mat = torch.sum(bk.unsqueeze(3) * kc.unsqueeze(2) * decay, dim=-1)
      t_mat = -t_mat * strictly_lower

      p_mat = eye + t_mat
      curr = t_mat
      for _ in range(1, 6):
        curr = torch.matmul(curr, curr)
        p_mat = torch.matmul(p_mat, eye + curr)
      a_mat = p_mat

      k_g_b = torch.exp(gcs) * kc * bc
      wv = wc * vc
      merged_rhs = torch.cat([k_g_b, wv], dim=-1)
      merged_out = torch.matmul(a_mat, merged_rhs)
      wwy = merged_out[:, :, :, :k_dim]
      u = merged_out[:, :, :, k_dim:]

      qg = qc.unsqueeze(3) * decay
      aqk = torch.sum(qg * kc.unsqueeze(2), dim=-1) * causal_mask

      vi = u - torch.matmul(wwy, state.to(u.dtype))
      oi = torch.matmul(qc * torch.exp(gcs), state.to(qc.dtype)) + torch.matmul(aqk, vi)
      outs.append(oi.permute(0, 2, 1, 3).to(q.dtype))

      glast = gcs[:, :, -1:, :]
      decayed_k = torch.exp(torch.clamp(glast - gcs, max=0.0)) * kc
      s_update = torch.matmul(decayed_k.transpose(-1, -2), vi)
      state = state * torch.exp(torch.clamp(glast, max=0.0)).transpose(-1, -2) + s_update

    return torch.cat(outs, dim=1)
