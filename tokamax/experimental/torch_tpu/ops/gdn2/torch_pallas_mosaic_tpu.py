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

"""PyTorch interface for Pallas TPU GDN-2 operator."""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tokamax._src.ops.experimental.gdn2 import pallas_mosaic_tpu as jax_pallas
from tokamax._src.ops.experimental.gdn2 import pallas_mosaic_tpu_bwd_kernel
from tokamax.experimental.torch_tpu.ops.gdn2 import torch_base
from typing_extensions import override

Config = jax_pallas.Config


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
  t_cpu = t.detach().cpu()
  if t_cpu.dtype == torch.bfloat16:
    return jnp.asarray(t_cpu.float().numpy()).astype(jnp.bfloat16)
  return jnp.asarray(t_cpu.numpy())


def _jax_to_torch(
    arr: jax.Array, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
  if arr.dtype == jnp.bfloat16:
    np_arr = np.array(arr.astype(jnp.float32), copy=True)
    return torch.from_numpy(np_arr).to(device=device, dtype=dtype)
  return torch.from_numpy(np.array(arr, copy=True)).to(
      device=device, dtype=dtype
  )


class _PallasTpuGatedDeltaNet2Vjp(torch_base._TorchOpBase[Config]):  # pylint: disable=protected-access
  """PyTorch VJP wrapper for Pallas TPU GDN-2 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuGatedDeltaNet2Vjp()
    self.jax_op_name = "torch_tpu_pallas_gdn2_vjp"
    self.is_vjp = True

  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    if not abstract_argument_dict:
      empty = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
      return {"residuals": (empty, empty, empty), "out": empty, "dout": empty}
    q_meta = abstract_argument_dict["q"]
    v_meta = abstract_argument_dict["v"]
    b_sz, seq_len, h_sz, k_dim = q_meta.shape
    v_dim = v_meta.shape[-1]
    bt = abstract_argument_dict.get("chunk_size", 64)
    nt = seq_len // bt
    bh = b_sz * h_sz
    return {
        "residuals": (
            jax.ShapeDtypeStruct((bh, nt, bt, bt), q_meta.dtype),
            jax.ShapeDtypeStruct((bh, nt, bt, k_dim + v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bh, nt, k_dim, v_dim), jnp.float32),
        ),
        "out": (v_meta, None),
        "dout": (v_meta, None),
    }

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      b: jax.Array,
      w: jax.Array,
      a_mat_c: jax.Array,
      merged_out_c: jax.Array,
      s_in_c: jax.Array,
      do: jax.Array,
      chunk_size: int = 64,
      scale: float | None = None,
      config: Any = None,
  ) -> tuple[
      jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
  ]:
    del config
    assert self.op_impl_jax is not None, "Backward class not set."
    dq, dk, dv, dg, db, dw, _ = pallas_mosaic_tpu_bwd_kernel.gdn2_pallas_bwd(
        q,
        k,
        v,
        g,
        b,
        w,
        do,
        chunk_size=chunk_size,
        scale=scale,
        a_mat_c=a_mat_c,
        merged_out_c=merged_out_c,
        s_in_c=s_in_c,
    )
    return dq, dk, dv, dg, db, dw

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      b: torch.Tensor,
      w: torch.Tensor,
      a_mat_c: torch.Tensor,
      merged_out_c: torch.Tensor,
      s_in_c: torch.Tensor,
      do: torch.Tensor,
      chunk_size: int = 64,
      scale: float | None = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    if self._torch_tokamax_op is not None and q.device.type in ("tpu", "xla"):
      return self._torch_tokamax_op(
          q,
          k,
          v,
          g,
          b,
          w,
          a_mat_c,
          merged_out_c,
          s_in_c,
          do,
          chunk_size=chunk_size,
          scale=scale,
      )
    grads_jax = self.op_impl_call(
        _torch_to_jax(q),
        _torch_to_jax(k),
        _torch_to_jax(v),
        _torch_to_jax(g),
        _torch_to_jax(b),
        _torch_to_jax(w),
        _torch_to_jax(a_mat_c),
        _torch_to_jax(merged_out_c),
        _torch_to_jax(s_in_c),
        _torch_to_jax(do),
        chunk_size=chunk_size,
        scale=scale,
    )
    return (
        _jax_to_torch(grads_jax[0], q.device, q.dtype),
        _jax_to_torch(grads_jax[1], k.device, k.dtype),
        _jax_to_torch(grads_jax[2], v.device, v.dtype),
        _jax_to_torch(grads_jax[3], g.device, g.dtype),
        _jax_to_torch(grads_jax[4], b.device, b.dtype),
        _jax_to_torch(grads_jax[5], w.device, w.dtype),
    )


class _CpuPallasAutogradFn(torch.autograd.Function):
  """Routes CPU execution through `op_impl_call` and `backward_op_torch`."""

  @staticmethod
  def forward(
      ctx: Any,
      op_self: Any,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      b: torch.Tensor,
      w: torch.Tensor,
      chunk_size: int,
      scale: float | None,
  ) -> torch.Tensor:
    out_jax, a_mat_jax, merged_out_jax, s_in_jax = op_self.op_impl_call(
        _torch_to_jax(q),
        _torch_to_jax(k),
        _torch_to_jax(v),
        _torch_to_jax(g),
        _torch_to_jax(b),
        _torch_to_jax(w),
        chunk_size=chunk_size,
        scale=scale,
    )
    out = _jax_to_torch(out_jax, q.device, q.dtype)
    a_mat_t = _jax_to_torch(a_mat_jax, q.device, q.dtype)
    merged_out_t = _jax_to_torch(merged_out_jax, q.device, torch.float32)
    s_in_t = _jax_to_torch(s_in_jax, q.device, torch.float32)
    ctx.save_for_backward(q, k, v, g, b, w, a_mat_t, merged_out_t, s_in_t)
    ctx.op_self = op_self
    ctx.chunk_size = chunk_size
    ctx.scale = scale
    return out

  @staticmethod
  def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
    q, k, v, g, b, w, a_mat_t, merged_out_t, s_in_t = ctx.saved_tensors
    dq, dk, dv, dg, db, dw = ctx.op_self.backward_op_torch(
        q,
        k,
        v,
        g,
        b,
        w,
        a_mat_t,
        merged_out_t,
        s_in_t,
        do,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
    )
    return None, dq, dk, dv, dg, db, dw, None, None


class _PallasTpuGatedDeltaNet2(torch_base._GatedDeltaNet2[Config]):  # pylint: disable=protected-access
  """PyTorch wrapper for Pallas TPU GDN-2 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuGatedDeltaNet2()
    self.backward_op_torch = _PallasTpuGatedDeltaNet2Vjp()
    self.jax_op_name = "torch_tpu_pallas_gdn2"

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
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, _), residuals = self.op_impl_jax._fwd(
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=True,
    )
    _, _, _, _, _, _, _, a_mat_c, merged_out_c, s_in_c, _, _ = residuals
    return out, a_mat_c, merged_out_c, s_in_c

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
      out, _a_mat_c, _merged_out_c, _s_in_c = self._torch_tokamax_op(
          q, k, v, g, b, w, chunk_size=chunk_size, scale=scale
      )
      return out
    return _CpuPallasAutogradFn.apply(
        self, q, k, v, g, b, w, chunk_size, scale
    )

  gdn2_pallas_delta_rule = __call__

  def setup_context(
      self, ctx: Any, inputs: Sequence[Any], output: Any
  ) -> None:
    out, a_mat_c, merged_out_c, s_in_c = output
    del out
    ctx.save_for_backward(*inputs[:6], a_mat_c, merged_out_c, s_in_c)
    ctx.chunk_size = inputs[6] if len(inputs) > 6 else 64
    ctx.scale = inputs[7] if len(inputs) > 7 else None

  def backward(
      self,
      ctx: Any,
      do: torch.Tensor,
      _d_a_mat: Any = None,
      _d_merged_out: Any = None,
      _d_s_in: Any = None,
  ) -> tuple[Any, ...]:
    assert self.backward_op_torch is not None, "Backward op not set."
    q, k, v, g, b, w, a_mat_c, merged_out_c, s_in_c = ctx.saved_tensors
    grads = self.backward_op_torch(
        q,
        k,
        v,
        g,
        b,
        w,
        a_mat_c,
        merged_out_c,
        s_in_c,
        do,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
    )
    return (*grads, None, None)


# Singleton instance
gdn2 = _PallasTpuGatedDeltaNet2()
