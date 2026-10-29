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

# Dimensions.
#   B   batch size
#   T   sequence length (nt * tt)
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension
#   tt  tile extent of T (chunk_size)
#   nt  grid extent along T (T // tt)

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


def _gdn2_cpu_fallback(
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
  B, T, H, K = q.shape
  _, _, _, V = v.shape
  tt = chunk_size
  nt = T // tt
  resolved_scale = (K**-0.5) if scale is None else scale
  outs = []

  state = torch.zeros(
      B,
      H,
      K,
      V,
      device=q.device,
      dtype=torch.float32,
  )
  strictly_lower = torch.tril(
      torch.ones(tt, tt, device=q.device), diagonal=-1
  )
  causal_mask = torch.tril(
      torch.ones(tt, tt, device=q.device), diagonal=0
  )
  eye = torch.eye(tt, device=q.device)

  for chunk_idx in range(nt):
    start_idx = chunk_idx * tt
    end_idx = (chunk_idx + 1) * tt
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
    num_doubling_steps = max(0, math.ceil(math.log2(tt)) - 1)
    for _ in range(num_doubling_steps):
      curr = torch.matmul(curr, curr)
      p_mat = torch.matmul(p_mat, eye + curr)
    a_mat = p_mat

    k_g_b = torch.exp(g_cs) * k_c * b_c
    wv = w_c * v_c
    merged_rhs = torch.cat([k_g_b, wv], dim=-1)
    merged_out = torch.matmul(a_mat, merged_rhs)
    wwy_chunk = merged_out[:, :, :, :K]
    u_chunk = merged_out[:, :, :, K:]

    qg = q_c.unsqueeze(3) * decay
    aqk = torch.sum(qg * k_c.unsqueeze(2), dim=-1) * causal_mask

    v_chunk = u_chunk - torch.matmul(wwy_chunk, state.to(u_chunk.dtype))
    out_chunk = torch.matmul(
        q_c * torch.exp(g_cs), state.to(q_c.dtype)
    ) + torch.matmul(aqk, v_chunk)
    outs.append(out_chunk.permute(0, 2, 1, 3).to(q.dtype))

    g_last = g_cs[:, :, -1:, :]
    decayed_k = torch.exp(g_last - g_cs) * k_c
    s_update = torch.matmul(decayed_k.transpose(-1, -2), v_chunk)
    state = state * torch.exp(g_last).transpose(-1, -2) + s_update

  return torch.cat(outs, dim=1)


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
  ) -> jax.Array:
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
    _, T, _, _ = q.shape
    if chunk_size <= 0 or T % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({T}) must be divisible by chunk_size"
          f" ({chunk_size})"
      )

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(
          q, k, v, g, b, w, chunk_size=chunk_size, scale=scale
      )
    return _gdn2_cpu_fallback(
        q, k, v, g, b, w, chunk_size=chunk_size, scale=scale
    )


# Singleton instance for the base GDN-2 operator.
gated_delta_net_2 = _GatedDeltaNet2[Any]()
object.__setattr__(
    gated_delta_net_2, "gdn2_pallas_delta_rule", gated_delta_net_2
)
