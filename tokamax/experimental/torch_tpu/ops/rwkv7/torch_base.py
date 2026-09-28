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

"""PyTorch interface for RWKV-7 (DPLR Delta Rule) operator."""

from typing import Any
import jax
import torch
import torch.nn.functional as F
from tokamax._src.ops.experimental.rwkv7 import base as jax_base
try:
  from tokamax.experimental.torch_tpu.ops import torch_op
except ImportError:
  torch_op = None
from typing_extensions import override


class _RWKV7:
  """PyTorch wrapper for base RWKV-7 operator."""

  def __init__(self):
    self.op_impl_jax = jax_base.RWKV7()
    self.jax_op_name = "torch_base_rwkv7"
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
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
      chunk_size: int = 64,
  ) -> jax.Array:
    (out, _), _ = self.op_impl_jax._fwd(
        q, k, v, alpha, beta, gk, chunk_size=chunk_size
    )
    return out

  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
      chunk_size: int = 64,
  ) -> torch.Tensor:
    b_dim, t, h, d = q.shape
    if t % chunk_size != 0:
      raise ValueError(f"Sequence length ({t}) must be divisible by chunk_size ({chunk_size})")

    k = F.normalize(k, dim=-1)
    gk = torch.clamp(gk, max=0.0)

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(q, k, v, alpha, beta, gk, chunk_size=chunk_size)
    return self._cpu_fallback(q, k, v, alpha, beta, gk, chunk_size)

  rwkv7_pallas_delta_rule = __call__

  def _cpu_fallback(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
      chunk_size: int = 64,
  ) -> torch.Tensor:
    """Pure PyTorch fallback for RWKV-7."""
    b_dim, t, h, d = q.shape
    v_dim = v.shape[-1]
    nt = t // chunk_size
    outs = []

    state = torch.zeros(b_dim, h, d, v_dim, device=q.device, dtype=torch.float32)

    for i in range(nt):
      st = i * chunk_size
      en = (i + 1) * chunk_size
      q_c = q[:, st:en].permute(0, 2, 1, 3).float()
      k_c = k[:, st:en].permute(0, 2, 1, 3).float()
      v_c = v[:, st:en].permute(0, 2, 1, 3).float()
      a_c = alpha[:, st:en].permute(0, 2, 1, 3).float()
      b_c = beta[:, st:en].permute(0, 2, 1, 3).float()
      g_c = gk[:, st:en].permute(0, 2, 1, 3).float()

      g_cs = torch.cumsum(g_c, dim=2)
      row_idx = torch.arange(chunk_size, device=q.device).unsqueeze(1)
      col_idx = torch.arange(chunk_size, device=q.device).unsqueeze(0)
      incl_causal = (col_idx <= row_idx).float()
      strict_causal = (col_idx < row_idx).float()

      diff_incl = torch.clamp(g_cs.unsqueeze(3) - g_cs.unsqueeze(2), max=0.0)
      decay_incl = torch.exp(diff_incl)
      aqk = torch.sum(q_c.unsqueeze(3) * k_c.unsqueeze(2) * decay_incl, dim=-1) * incl_causal
      aqb = torch.sum(q_c.unsqueeze(3) * b_c.unsqueeze(2) * decay_incl, dim=-1) * incl_causal

      g_excl = g_cs - g_c
      diff_excl = torch.clamp(g_excl.unsqueeze(3) - g_cs.unsqueeze(2), max=0.0)
      decay_excl = torch.exp(diff_excl)
      a_ab = torch.sum(a_c.unsqueeze(3) * b_c.unsqueeze(2) * decay_excl, dim=-1) * strict_causal
      a_ak = torch.sum(a_c.unsqueeze(3) * k_c.unsqueeze(2) * decay_excl, dim=-1) * strict_causal

      eye = torch.eye(chunk_size, device=q.device, dtype=torch.float32)
      p_mat = eye + a_ab
      curr = a_ab
      for _ in range(1, 6):
        curr = torch.matmul(curr, curr)
        p_mat = torch.matmul(p_mat, eye + curr)
      a_ab_inv = p_mat

      ak_v = torch.matmul(a_ak, v_c)
      decayed_alpha = torch.exp(torch.clamp(g_excl, max=0.0)) * a_c
      merged_rhs = torch.cat([ak_v, decayed_alpha], dim=-1)
      merged_uw = torch.matmul(a_ab_inv, merged_rhs)
      u_c = merged_uw[:, :, :, :v_dim]
      w_c = merged_uw[:, :, :, v_dim:]

      v2_c = u_c + torch.matmul(w_c, state.to(w_c.dtype))
      o_1 = torch.matmul(aqk, v_c)
      o_2 = torch.matmul(aqb, v2_c)
      o_3 = torch.matmul(q_c * torch.exp(torch.clamp(g_cs, max=0.0)), state.to(q_c.dtype))
      o_c = o_1 + o_2 + o_3
      outs.append(o_c.permute(0, 2, 1, 3).to(q.dtype))

      g_last = g_cs[:, :, -1, :]
      decay_last = torch.exp(torch.clamp(g_last[:, :, None, :] - g_cs, max=0.0))
      state = (
          state * torch.exp(torch.clamp(g_last, max=0.0))[:, :, :, None]
          + torch.matmul((k_c * decay_last).transpose(-1, -2), v_c)
          + torch.matmul((b_c * decay_last).transpose(-1, -2), v2_c)
      )

    return torch.cat(outs, dim=1)
