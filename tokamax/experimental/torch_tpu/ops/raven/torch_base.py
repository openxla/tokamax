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

from collections.abc import Sequence
from typing import Any, Generic, TypeVar
import jax
import torch
from tokamax._src.ops.experimental.raven import base as jax_base
from typing_extensions import override

_Config = TypeVar("_Config")

try:
  from tokamax.experimental.torch_tpu.ops import torch_op  # pylint: disable=g-import-not-at-top

  _TorchOpBase = torch_op.TorchOp
except ImportError:

  class _TorchOpBase(Generic[_Config]):  # type: ignore[no-redef]
    """Fallback base class when ``torch_tpu`` is unavailable on CPU/CUDA hosts."""

    def __init__(self) -> None:
      self.op_impl_jax: Any = None
      self._torch_tokamax_op: Any = None
      self.backward_op_torch: Any = None
      self.jax_op_name: str = ""
      self.is_vjp: bool = False
      self.donate_argnums: Sequence[int] | None = None
      self.fake_impl: Any = None

    def op_impl_call(self, *args: Any, **kwargs: Any) -> Any:
      raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
      raise NotImplementedError

    def setup_context(
        self, ctx: Any, inputs: Sequence[Any], output: Any
    ) -> None:
      del ctx, inputs, output

    def backward(self, *args: Any, **kwargs: Any) -> Any:
      raise NotImplementedError

    def derive_backward_shapes(
        self, abstract_argument_dict: dict[str, Any]
    ) -> dict[str, Any]:
      del abstract_argument_dict
      return {}


class _RavenGSA(_TorchOpBase[_Config]):
  """PyTorch wrapper for base Raven GSA operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.RavenGSA()
    self.jax_op_name = "torch_base_raven_gsa"
    self.is_vjp = False

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      v: jax.Array | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> jax.Array:
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, _), _ = self.op_impl_jax._fwd(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
    )
    return out

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      v: torch.Tensor | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> torch.Tensor:
    if q.ndim != 4:
      raise ValueError(
          f"Expected 4D input (batch, seq_len, heads, dim), got {q.shape}."
      )
    seq_len = q.shape[1]
    if chunk_size <= 0 or seq_len % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({seq_len}) must be divisible by positive"
          f" chunk_size ({chunk_size})."
      )

    g = torch.clamp(g, max=0.0)

    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(
          q, k, s, g, v, chunk_size=chunk_size, scale=scale
      )
    return self._cpu_fallback(
        q, k, s, g, v=v, chunk_size=chunk_size, scale=scale
    )

  raven_pallas_gsa = __call__

  def _cpu_fallback(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      s: torch.Tensor,
      g: torch.Tensor,
      v: torch.Tensor | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> torch.Tensor:
    """Pure PyTorch reference fallback for Raven Gated Slot Attention."""
    batch_size, seq_len, num_heads, key_dim = q.shape
    num_slots = s.shape[-1]
    num_chunks = seq_len // chunk_size

    q_scaled = q * scale if scale is not None else q
    stage1_outs = []

    state1 = torch.zeros(
        batch_size,
        num_heads,
        key_dim,
        num_slots,
        device=q.device,
        dtype=torch.float32,
    )
    row_idx = torch.arange(chunk_size, device=q.device).unsqueeze(1)
    col_idx = torch.arange(chunk_size, device=q.device).unsqueeze(0)
    causal = (col_idx <= row_idx).float()

    for chunk_idx in range(num_chunks):
      start_idx = chunk_idx * chunk_size
      end_idx = (chunk_idx + 1) * chunk_size
      q_c = q_scaled[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      k_c = k[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      s_c = s[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      g_c = g[:, start_idx:end_idx].permute(0, 2, 1, 3).float()

      g_cs = torch.cumsum(g_c, dim=2)

      qk = torch.matmul(q_c, k_c.transpose(-1, -2)) * causal
      diff = torch.clamp(g_cs.unsqueeze(3) - g_cs.unsqueeze(2), max=0.0)
      decay = torch.exp(diff)
      y_diag = torch.sum(qk.unsqueeze(-1) * decay * s_c.unsqueeze(2), dim=3)

      g_last = g_cs[:, :, -1:, :]
      decay_tail = torch.exp(torch.clamp(g_last - g_cs, max=0.0))
      state_contrib = torch.matmul(k_c.transpose(-1, -2), s_c * decay_tail)

      q_state = torch.matmul(q_c, state1)
      decay_local = torch.exp(torch.clamp(g_cs, max=0.0))
      y_inter = q_state * decay_local

      stage1_outs.append((y_diag + y_inter).permute(0, 2, 1, 3).to(s.dtype))
      state1 = state1 * torch.exp(torch.clamp(g_last, max=0.0)) + state_contrib

    ok = torch.cat(stage1_outs, dim=1)
    if v is None:
      return ok

    qv = torch.softmax(ok.float(), dim=-1).to(v.dtype)
    val_dim = v.shape[-1]
    stage2_outs = []
    state2 = torch.zeros(
        batch_size,
        num_heads,
        num_slots,
        val_dim,
        device=q.device,
        dtype=torch.float32,
    )

    for chunk_idx in range(num_chunks):
      start_idx = chunk_idx * chunk_size
      end_idx = (chunk_idx + 1) * chunk_size
      qv_c = qv[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      s_c = s[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      v_c = v[:, start_idx:end_idx].permute(0, 2, 1, 3).float()
      g_c = g[:, start_idx:end_idx].permute(0, 2, 1, 3).float()

      g_cs = torch.cumsum(g_c, dim=2)
      diff = torch.clamp(g_cs.unsqueeze(3) - g_cs.unsqueeze(2), max=0.0)
      decay = torch.exp(diff)
      aqk = (
          torch.sum(qv_c.unsqueeze(3) * decay * s_c.unsqueeze(2), dim=-1)
          * causal
      )
      y_diag = torch.matmul(aqk, v_c)

      g_last = g_cs[:, :, -1:, :]
      decay_tail = torch.exp(torch.clamp(g_last - g_cs, max=0.0))
      state_contrib = torch.matmul((s_c * decay_tail).transpose(-1, -2), v_c)

      y_inter = torch.matmul(qv_c * torch.exp(torch.clamp(g_cs, max=0.0)), state2)
      stage2_outs.append((y_diag + y_inter).permute(0, 2, 1, 3).to(v.dtype))
      state2 = (
          state2 * torch.exp(torch.clamp(g_last.transpose(-1, -2), max=0.0))
          + state_contrib
      )

    return torch.cat(stage2_outs, dim=1)


# Singleton instance for the reference PyTorch op.
raven_gsa = _RavenGSA[Any]()
