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

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.gdn2 import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
from typing_extensions import override


def _torch_to_jax(tensor: torch.Tensor) -> jax.Array:
  tensor_cpu = tensor.detach().cpu()
  if tensor_cpu.dtype == torch.bfloat16:
    return jnp.asarray(tensor_cpu.float().numpy()).astype(jnp.bfloat16)
  return jnp.asarray(tensor_cpu.numpy())


def _jax_to_torch(
    arr: jax.Array, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
  if arr.dtype == jnp.bfloat16:
    np_arr = np.array(arr.astype(jnp.float32), copy=True)
    return torch.from_numpy(np_arr).to(device=device, dtype=dtype)
  return torch.from_numpy(np.array(arr, copy=True)).to(
      device=device, dtype=dtype
  )


def _register_cpu_kernel(
    op_self: torch_op.TorchOp[Any],
    is_vjp: bool,
) -> None:
  """Registers a CPU custom_op kernel that dispatches to `op_self.op_impl_call`."""
  assert op_self._torch_tokamax_op is not None  # pylint: disable=protected-access
  if is_vjp:

    def _cpu_bwd(
        a_mat_c: torch.Tensor,
        merged_out_c: torch.Tensor,
        s_in_c: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        chunk_size: int = 64,
        scale: float | None = None,
        return_residuals: bool = False,
        config: Sequence[int] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
      grads_jax = op_self.op_impl_call(
          _torch_to_jax(a_mat_c),
          _torch_to_jax(merged_out_c),
          _torch_to_jax(s_in_c),
          _torch_to_jax(out),
          _torch_to_jax(dout),
          _torch_to_jax(q),
          _torch_to_jax(k),
          _torch_to_jax(v),
          _torch_to_jax(g),
          _torch_to_jax(b),
          _torch_to_jax(w),
          chunk_size=chunk_size,
          scale=scale,
          return_residuals=return_residuals,
          config=None if config is None else tuple(config),
      )
      return (
          _jax_to_torch(grads_jax[0], q.device, q.dtype),
          _jax_to_torch(grads_jax[1], k.device, k.dtype),
          _jax_to_torch(grads_jax[2], v.device, v.dtype),
          _jax_to_torch(grads_jax[3], g.device, g.dtype),
          _jax_to_torch(grads_jax[4], b.device, b.dtype),
          _jax_to_torch(grads_jax[5], w.device, w.dtype),
      )

    op_self._torch_tokamax_op.register_kernel("cpu")(_cpu_bwd)  # pylint: disable=protected-access
  else:

    def _cpu_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        chunk_size: int = 64,
        scale: float | None = None,
        return_residuals: bool = True,
        config: Sequence[int] | None = None,
        bwd_config: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      out_jax, a_mat_jax, merged_out_jax, s_in_jax = op_self.op_impl_call(
          _torch_to_jax(q),
          _torch_to_jax(k),
          _torch_to_jax(v),
          _torch_to_jax(g),
          _torch_to_jax(b),
          _torch_to_jax(w),
          chunk_size=chunk_size,
          scale=scale,
          return_residuals=return_residuals,
          config=None if config is None else tuple(config),
          bwd_config=None if bwd_config is None else tuple(bwd_config),
      )
      return (
          _jax_to_torch(out_jax, v.device, v.dtype),
          _jax_to_torch(a_mat_jax, q.device, q.dtype),
          _jax_to_torch(merged_out_jax, q.device, torch.float32),
          _jax_to_torch(s_in_jax, q.device, torch.float32),
      )

    op_self._torch_tokamax_op.register_kernel("cpu")(_cpu_fwd)  # pylint: disable=protected-access


class _GatedDeltaNet2Vjp[Config](torch_op.TorchOp[Config]):
  """Gated Delta Net 2 (GDN-2) PyTorch Op VJP API using reference impl."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.GatedDeltaNet2Vjp()
    self.jax_op_name = "base_gdn2_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  @override
  def _register_ops(self) -> None:
    super()._register_ops()
    _register_cpu_kernel(self, is_vjp=True)

  def _fake_impl(
      self,
      a_mat_c: torch.Tensor,
      merged_out_c: torch.Tensor,
      s_in_c: torch.Tensor,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      b: torch.Tensor,
      w: torch.Tensor,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    del a_mat_c, merged_out_c, s_in_c, out, dout, chunk_size, scale
    del return_residuals, config
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(b),
        torch.empty_like(w),
    )

  @override
  def __call__(
      self,
      a_mat_c: torch.Tensor,
      merged_out_c: torch.Tensor,
      s_in_c: torch.Tensor,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      b: torch.Tensor,
      w: torch.Tensor,
      chunk_size: int = 64,
      scale: float | None = None,
      config: Any = None,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    assert (
        self._torch_tokamax_op is not None
    ), "Forward op not registered. This means that self.op_impl_jax is not set"
    " in the constructor."
    return self._torch_tokamax_op(
        a_mat_c,
        merged_out_c,
        s_in_c,
        out,
        dout,
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=False,
        config=self.deconstruct_config(config),
    )

  @override
  def op_impl_call(
      self,
      a_mat_c: jax.Array,
      merged_out_c: jax.Array,
      s_in_c: jax.Array,
      out: jax.Array,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      b: jax.Array,
      w: jax.Array,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[
      jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
  ]:
    del config
    assert self.op_impl_jax is not None, "Forward class not set."
    grads, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        (a_mat_c, merged_out_c, s_in_c),
        (out, None),
        (dout, None),
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=return_residuals,
        config=None,
    )
    return (
        grads["q"],
        grads["k"],
        grads["v"],
        grads["g"],
        grads["b"],
        grads["w"],
    )

  @override
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    if not abstract_argument_dict:
      empty = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
      return {
          "residuals": (empty, empty, empty),
          "out": empty,
          "dout": empty,
          "a_mat_c": empty,
          "merged_out_c": empty,
          "s_in_c": empty,
      }
    q_meta = abstract_argument_dict["q"]
    v_meta = abstract_argument_dict["v"]
    B, T, H, K = q_meta.shape
    _, _, _, V = v_meta.shape
    tt = abstract_argument_dict.get("chunk_size", 64)
    nt = T // tt
    BH = B * H
    a_mat_meta = jax.ShapeDtypeStruct((BH, nt, tt, tt), q_meta.dtype)
    merged_out_meta = jax.ShapeDtypeStruct((BH, nt, tt, K + V), jnp.float32)
    s_in_meta = jax.ShapeDtypeStruct((BH, nt, K, V), jnp.float32)
    return {
        "residuals": (a_mat_meta, merged_out_meta, s_in_meta),
        "out": (v_meta, None),
        "dout": (v_meta, None),
    }


class _GatedDeltaNet2[Config](torch_op.TorchOp[Config]):
  """Gated Delta Net 2 (GDN-2) PyTorch Op API using reference impl."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.GatedDeltaNet2()
    self.backward_op_torch = _GatedDeltaNet2Vjp()
    self.jax_op_name = "base_gdn2"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def _register_ops(self) -> None:
    super()._register_ops()
    _register_cpu_kernel(self, is_vjp=False)

  def _fake_impl(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      b: torch.Tensor,
      w: torch.Tensor,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = True,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del k, g, b, w, scale, return_residuals, config, bwd_config
    B, T, H, K = q.shape
    _, _, _, V = v.shape
    tt = chunk_size
    nt = T // tt
    BH = B * H
    return (
        torch.empty_like(v),
        q.new_empty((BH, nt, tt, tt)),
        q.new_empty((BH, nt, tt, K + V), dtype=torch.float32),
        q.new_empty((BH, nt, K, V), dtype=torch.float32),
    )

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
      configs: tuple[Config | None, Config | None] | None = None,
  ) -> torch.Tensor:
    _, T, _, _ = q.shape
    if chunk_size <= 0 or T % chunk_size != 0:
      raise ValueError(
          f"Sequence length ({T}) must be divisible by chunk_size"
          f" ({chunk_size})"
      )
    fwd_config, bwd_config = (None, None) if configs is None else configs
    assert (
        self._torch_tokamax_op is not None
    ), "Forward op not registered. This means that self.op_impl_jax is not set"
    " in the constructor."
    assert self.backward_op_torch is not None, "Backward op not set."
    out, _, _, _ = self._torch_tokamax_op(
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=True,
        config=self.deconstruct_config(fwd_config),
        bwd_config=self.backward_op_torch.deconstruct_config(bwd_config),
    )
    return out

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
      return_residuals: bool = True,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    del config, bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    (out, _), _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=return_residuals,
        config=None,
    )
    B, T, H, K = q.shape
    _, _, _, V = v.shape
    tt = chunk_size
    nt = T // tt
    BH = B * H
    a_mat_c = jnp.zeros((BH, nt, tt, tt), dtype=q.dtype)
    merged_out_c = jnp.zeros((BH, nt, tt, K + V), dtype=jnp.float32)
    s_in_c = jnp.zeros((BH, nt, K, V), dtype=jnp.float32)
    return out, a_mat_c, merged_out_c, s_in_c

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    """Saves tensors to the context for the backward pass."""
    (
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size,
        scale,
        _,
        _,
        bwd_config,
    ) = inputs
    out, a_mat_c, merged_out_c, s_in_c = output
    ctx.save_for_backward(a_mat_c, merged_out_c, s_in_c, out, q, k, v, g, b, w)
    ctx.chunk_size = chunk_size
    ctx.scale = scale
    ctx.bwd_config = bwd_config

  @override
  def backward(
      self,
      ctx: Any,
      dout: torch.Tensor,
      d_a_mat: torch.Tensor,
      d_merged_out: torch.Tensor,
      d_s_in: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      None,
      None,
      None,
      None,
      None,
  ]:
    """Callback for torch register_autograd."""
    del d_a_mat, d_merged_out, d_s_in
    assert self.backward_op_torch is not None, "Backward op not set."
    a_mat_c, merged_out_c, s_in_c, out, q, k, v, g, b, w = ctx.saved_tensors
    dq, dk, dv, dg, db, dw = self.backward_op_torch(
        a_mat_c,
        merged_out_c,
        s_in_c,
        out,
        dout,
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
        config=ctx.bwd_config,
    )
    return dq, dk, dv, dg, db, dw, None, None, None, None, None


# Singleton instance of GatedDeltaNet2.
GatedDeltaNet2 = _GatedDeltaNet2[Any]()
