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

from typing import Any
import jax
import torch
import torch.nn.functional as F
from tokamax._src.ops.experimental.raven import base as jax_base
try:
  from tokamax.experimental.torch_tpu.ops import torch_op
except ImportError:
  torch_op = None
from typing_extensions import override


class _RavenGSA:
  """PyTorch wrapper for base Raven GSA operator."""

  def __init__(self):
    self.op_impl_jax = jax_base.RavenGSA()
    self.jax_op_name = "torch_base_raven_gsa"
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
      s: jax.Array,
      g: jax.Array,
      chunk_size: int = 64,
  ) -> jax.Array:
    (out, _), _ = self.op_impl_jax._fwd(
        q, k, s, g, chunk_size=chunk_size
    )
    return out

  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      chunk_size: int = 64,
  ) -> torch.Tensor:
    b_dim, t, h, d = q.shape
    if t % chunk_size != 0:
      raise ValueError(f"Sequence length ({t}) must be divisible by chunk_size ({chunk_size})")

    g = torch.clamp(g, max=0.0)

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(q, k, s, g, chunk_size=chunk_size)
    return self._cpu_fallback(q, k, s, g, chunk_size)

  raven_pallas_gsa = __call__

  def _cpu_fallback(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      chunk_size: int = 64,
  ) -> torch.Tensor:
    """Pure PyTorch fallback for Raven Gated Slot Attention."""
    b_dim, t, h, k_dim = q.shape
    m_dim = s.shape[-1]
    nt = t // chunk_size
    outs = []

    state = torch.zeros(b_dim, h, k_dim, m_dim, device=q.device, dtype=torch.float32)

    for i in range(nt):
      st = i * chunk_size
      en = (i + 1) * chunk_size
      q_c = q[:, st:en].permute(0, 2, 1, 3).float()
      k_c = k[:, st:en].permute(0, 2, 1, 3).float()
      s_c = s[:, st:en].permute(0, 2, 1, 3).float()
      g_c = g[:, st:en].permute(0, 2, 1, 3).float()

      g_cs = torch.cumsum(g_c, dim=2)
      row_idx = torch.arange(chunk_size, device=q.device).unsqueeze(1)
      col_idx = torch.arange(chunk_size, device=q.device).unsqueeze(0)
      causal = (col_idx <= row_idx).float()

      qk = torch.matmul(q_c, k_c.transpose(-1, -2)) * causal
      diff = torch.clamp(g_cs.unsqueeze(3) - g_cs.unsqueeze(2), max=0.0)
      decay = torch.exp(diff)
      y_diag = torch.sum(qk.unsqueeze(-1) * decay * s_c.unsqueeze(2), dim=3)

      g_last = g_cs[:, :, -1:, :]
      decay_tail = torch.exp(torch.clamp(g_last - g_cs, max=0.0))
      sc = torch.matmul(k_c.transpose(-1, -2), s_c * decay_tail)

      qs = torch.matmul(q_c, state.to(q.dtype))
      decay_local = torch.exp(torch.clamp(g_cs, max=0.0))
      y_inter = qs * decay_local

      outs.append((y_diag + y_inter).permute(0, 2, 1, 3).to(q.dtype))

      state = state * torch.exp(torch.clamp(g_last, max=0.0)) + sc

    return torch.cat(outs, dim=1)
