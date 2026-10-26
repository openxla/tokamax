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

import math
from typing import Any

import jax
import torch
from tokamax._src.ops.experimental.gdn2 import base as jax_base
from typing_extensions import override

try:
  from tokamax.experimental.torch_tpu.ops import torch_op
except ImportError:
  torch_op = None


class _FallbackTorchOp[Config]:
  """Fallback base class when `torch_tpu` is unavailable on CPU hosts."""

  def __init__(self) -> None:
    self._torch_tokamax_op: Any = None
    self.op_impl_jax: Any = None
    self.backward_op_torch: Any = None
    self.jax_op_name: str = ""
    self.is_vjp: bool = False
    self.donate_argnums: Any = None
    self.fake_impl: Any = None

  def op_impl_call(self, *args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


_TorchOpBase = torch_op.TorchOp if torch_op is not None else _FallbackTorchOp


class _GatedDeltaNet2[Config](_TorchOpBase[Config]):
  """PyTorch wrapper for base GDN-2 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.GatedDeltaNet2()
    self.jax_op_name = "torch_base_gdn2"

  def op_impl_call_config_setup(
      self, *args: Any, config: Any = None, **kwargs: Any
  ) -> Any:
    if (
        config is None
        and self.op_impl_jax is not None
        and getattr(self.op_impl_jax, "config_cls", type(None)) is type(None)
    ):
      return None
    if hasattr(super(), "op_impl_call_config_setup"):
      return super().op_impl_call_config_setup(*args, config=config, **kwargs)
    return config

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      b: jax.Array,
      w: jax.Array,
      chunk_size: int = 64,
      scale: float | None = None,
      config: Any = None,
  ) -> jax.Array:
    del config
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, _), _ = self.op_impl_jax._fwd(
        q, k, v, g, b, w, chunk_size=chunk_size, scale=scale
    )
    return out

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      b: torch.Tensor,
      w: torch.Tensor,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> torch.Tensor:
    seq_len = q.shape[1]
    if chunk_size <= 0 or seq_len % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({seq_len}) must be divisible by chunk_size"
          f" ({chunk_size})"
      )

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(
          q, k, v, g, b, w, chunk_size=chunk_size, scale=scale
      )
    return self._cpu_fallback(
        q, k, v, g, b, w, chunk_size=chunk_size, scale=scale
    )

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
      scale: float | None = None,
  ) -> torch.Tensor:
    """Pure PyTorch reference implementation for GDN-2."""
    batch_size, seq_len, num_heads, key_dim = q.shape
    val_dim = v.shape[-1]
    num_chunks = seq_len // chunk_size
    resolved_scale = (key_dim**-0.5) if scale is None else scale
    outs = []

    state = torch.zeros(
        batch_size,
        num_heads,
        key_dim,
        val_dim,
        device=q.device,
        dtype=torch.float32,
    )
    strictly_lower = torch.tril(
        torch.ones(chunk_size, chunk_size, device=q.device), diagonal=-1
    )
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=q.device), diagonal=0
    )
    eye = torch.eye(chunk_size, device=q.device)

    for chunk_idx in range(num_chunks):
      start_idx = chunk_idx * chunk_size
      end_idx = (chunk_idx + 1) * chunk_size
      q_c = q[:, start_idx:end_idx].permute(0, 2, 1, 3).float() * resolved_scale
      k_c = k[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      v_c = v[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      g_c = g[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      b_c = b[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      w_c = w[:, start_idx:end_idx].permute(0, 2, 1, 3).float()

      g_cs = torch.cumsum(g_c, dim=2)
      bk = b_c * k_c
      diff = torch.clamp(g_cs.unsqueeze(3) - g_cs.unsqueeze(2), max=80.0)
      decay = torch.exp(diff)
      t_mat = torch.sum(bk.unsqueeze(3) * k_c.unsqueeze(2) * decay, dim=-1)
      t_mat = -t_mat * strictly_lower

      p_mat = eye + t_mat
      curr = t_mat
      num_doubling_steps = max(0, math.ceil(math.log2(chunk_size)) - 1)
      for _ in range(num_doubling_steps):
        curr = torch.matmul(curr, curr)
        p_mat = torch.matmul(p_mat, eye + curr)
      a_mat = p_mat

      k_g_b = torch.exp(g_cs) * k_c * b_c
      wv = w_c * v_c
      merged_rhs = torch.cat([k_g_b, wv], dim=-1)
      merged_out = torch.matmul(a_mat, merged_rhs)
      wwy = merged_out[:, :, :, :key_dim]
      u = merged_out[:, :, :, key_dim:]

      qg = q_c.unsqueeze(3) * decay
      aqk = torch.sum(qg * k_c.unsqueeze(2), dim=-1) * causal_mask

      vi = u - torch.matmul(wwy, state.to(u.dtype))
      oi = torch.matmul(
          q_c * torch.exp(g_cs), state.to(q_c.dtype)
      ) + torch.matmul(aqk, vi)
      outs.append(oi.permute(0, 2, 1, 3).to(q.dtype))

      g_last = g_cs[:, :, -1:, :]
      decayed_k = torch.exp(g_last - g_cs) * k_c
      s_update = torch.matmul(decayed_k.transpose(-1, -2), vi)
      state = state * torch.exp(g_last).transpose(-1, -2) + s_update

    return torch.cat(outs, dim=1)


# Singleton instance for the base GDN-2 operator.
gated_delta_net_2 = _GatedDeltaNet2[Any]()
