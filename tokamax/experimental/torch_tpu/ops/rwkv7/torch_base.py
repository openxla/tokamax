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

import math
from typing import Any

import jax
import torch
from tokamax._src.ops.experimental.rwkv7 import base as jax_base
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


class _RWKV7[Config](_TorchOpBase[Config]):
  """PyTorch wrapper for base RWKV-7 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.RWKV7()
    self.jax_op_name = "torch_base_rwkv7"

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
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
      chunk_size: int = 64,
      scale: float | None = None,
      config: Any = None,
  ) -> jax.Array:
    del config
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, _), _ = self.op_impl_jax._fwd(
        q, k, v, alpha, beta, gk, chunk_size=chunk_size, scale=scale
    )
    return out

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
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
          q, k, v, alpha, beta, gk, chunk_size=chunk_size, scale=scale
      )
    return self._cpu_fallback(
        q, k, v, alpha, beta, gk, chunk_size=chunk_size, scale=scale
    )

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
      scale: float | None = None,
  ) -> torch.Tensor:
    """Pure PyTorch reference implementation for RWKV-7."""
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

    row_idx = torch.arange(chunk_size, device=q.device).unsqueeze(1)
    col_idx = torch.arange(chunk_size, device=q.device).unsqueeze(0)
    incl_causal = (col_idx <= row_idx).float()
    strict_causal = (col_idx < row_idx).float()
    eye = torch.eye(chunk_size, device=q.device, dtype=torch.float32)

    for chunk_idx in range(num_chunks):
      start_idx = chunk_idx * chunk_size
      end_idx = (chunk_idx + 1) * chunk_size
      q_c = q[:, start_idx:end_idx].permute(0, 2, 1, 3).float() * resolved_scale
      k_c = k[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      v_c = v[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      alpha_c = alpha[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      beta_c = beta[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      gk_c = gk[:, start_idx:end_idx].permute(0, 2, 1, 3).float()

      gk_cs = torch.cumsum(gk_c, dim=2)

      diff_incl = torch.clamp(gk_cs.unsqueeze(3) - gk_cs.unsqueeze(2), max=0.0)
      decay_incl = torch.exp(diff_incl)
      aqk = (
          torch.sum(q_c.unsqueeze(3) * k_c.unsqueeze(2) * decay_incl, dim=-1)
          * incl_causal
      )
      aqb = (
          torch.sum(q_c.unsqueeze(3) * beta_c.unsqueeze(2) * decay_incl, dim=-1)
          * incl_causal
      )

      gk_excl = gk_cs - gk_c
      diff_excl = torch.clamp(
          gk_excl.unsqueeze(3) - gk_cs.unsqueeze(2), max=0.0
      )
      decay_excl = torch.exp(diff_excl)
      a_ab = (
          torch.sum(
              alpha_c.unsqueeze(3) * beta_c.unsqueeze(2) * decay_excl, dim=-1
          )
          * strict_causal
      )
      a_ak = (
          torch.sum(
              alpha_c.unsqueeze(3) * k_c.unsqueeze(2) * decay_excl, dim=-1
          )
          * strict_causal
      )

      p_mat = eye + a_ab
      curr = a_ab
      num_doubling_steps = max(0, math.ceil(math.log2(chunk_size)) - 1)
      for _ in range(num_doubling_steps):
        curr = torch.matmul(curr, curr)
        p_mat = torch.matmul(p_mat, eye + curr)
      a_ab_inv = p_mat

      ak_v = torch.matmul(a_ak, v_c)
      decayed_alpha = torch.exp(torch.clamp(gk_excl, max=0.0)) * alpha_c
      merged_rhs = torch.cat([ak_v, decayed_alpha], dim=-1)
      merged_uw = torch.matmul(a_ab_inv, merged_rhs)
      u_c = merged_uw[:, :, :, :val_dim]
      w_c = merged_uw[:, :, :, val_dim:]

      v2_c = u_c + torch.matmul(w_c, state.to(w_c.dtype))
      o_1 = torch.matmul(aqk, v_c)
      o_2 = torch.matmul(aqb, v2_c)
      o_3 = torch.matmul(
          q_c * torch.exp(torch.clamp(gk_cs, max=0.0)), state.to(q_c.dtype)
      )
      o_c = o_1 + o_2 + o_3
      outs.append(o_c.permute(0, 2, 1, 3).to(q.dtype))

      g_last = gk_cs[:, :, -1, :]
      decay_last = torch.exp(
          torch.clamp(g_last[:, :, None, :] - gk_cs, max=0.0)
      )
      state = (
          state * torch.exp(torch.clamp(g_last, max=0.0))[:, :, :, None]
          + torch.matmul((k_c * decay_last).transpose(-1, -2), v_c)
          + torch.matmul((beta_c * decay_last).transpose(-1, -2), v2_c)
      )

    return torch.cat(outs, dim=1)


# Singleton instance for the base RWKV-7 operator.
rwkv7_delta_rule = _RWKV7[Any]()
