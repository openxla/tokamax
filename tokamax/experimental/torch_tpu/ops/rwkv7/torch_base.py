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

# Dimensions.
#   B      batch size
#   T      sequence length (nt * tt)
#   H      number of attention heads
#   K      key head dimension
#   V      value head dimension
#   tt     sequence chunk tile extent (chunk_size)
#   nt     grid extent over T: nt = T // tt

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.rwkv7 import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch
from typing_extensions import override


class _Rwkv7Vjp[Config](torch_op.TorchOp[Config]):
  """PyTorch VJP wrapper for base RWKV-7 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.Rwkv7Vjp()
    self.jax_op_name = "base_rwkv7_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      a_ab_inv: torch.Tensor,
      merged_out: torch.Tensor,
      s_in: torch.Tensor,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
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
    del a_ab_inv, merged_out, s_in, out, dout, chunk_size, scale
    del return_residuals, config
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(alpha),
        torch.empty_like(beta),
        torch.empty_like(gk),
    )

  @override
  def __call__(
      self,
      a_ab_inv: torch.Tensor,
      merged_out: torch.Tensor,
      s_in: torch.Tensor,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
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
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        a_ab_inv,
        merged_out,
        s_in,
        out,
        dout,
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=False,
        config=self.deconstruct_config(config),
    )

  @override
  def op_impl_call(
      self,
      a_ab_inv: jax.Array,
      merged_out: jax.Array,
      s_in: jax.Array,
      out: jax.Array,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
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
        (a_ab_inv, merged_out, s_in),
        (out, None),
        (dout, None),
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=return_residuals,
        config=None,
    )
    return (
        grads["q"],
        grads["k"],
        grads["v"],
        grads["alpha"],
        grads["beta"],
        grads["gk"],
    )

  @override
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    """Derives the backward shapes for the RWKV-7 VJP pass."""
    if not abstract_argument_dict:
      empty = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
      return {
          "residuals": (empty, empty, empty),
          "out": empty,
          "dout": empty,
      }
    q_meta = abstract_argument_dict["q"]
    v_meta = abstract_argument_dict["v"]
    b_sz, seq_len, h_sz, k_dim = q_meta.shape
    _, _, _, v_dim = v_meta.shape
    tt = abstract_argument_dict.get("chunk_size", 64)
    nt = seq_len // tt
    out_struct = jax.ShapeDtypeStruct(shape=v_meta.shape, dtype=q_meta.dtype)
    return {
        "residuals": (
            jax.ShapeDtypeStruct((b_sz, h_sz, nt, tt, tt), q_meta.dtype),
            jax.ShapeDtypeStruct(
                (b_sz, h_sz, nt, tt, k_dim + v_dim), jnp.float32
            ),
            jax.ShapeDtypeStruct((b_sz, h_sz, nt, k_dim, v_dim), jnp.float32),
        ),
        "out": (out_struct, None),
        "dout": (out_struct, None),
    }


# Singleton instance of Rwkv7Vjp.
Rwkv7Vjp = _Rwkv7Vjp[Any]()


class _Rwkv7[Config](torch_op.TorchOp[Config]):
  """PyTorch wrapper for base RWKV-7 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.Rwkv7()
    self.backward_op_torch = Rwkv7Vjp
    self.jax_op_name = "base_rwkv7"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      alpha: torch.Tensor,
      beta: torch.Tensor,
      gk: torch.Tensor,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = True,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del k, alpha, beta, gk, scale, return_residuals, config, bwd_config
    assert self.backward_op_torch is not None, "Backward op not set."
    derived = self.backward_op_torch.derive_backward_shapes({
        "q": torch_utils.convert_torch_to_jax_via_meta(q),
        "v": torch_utils.convert_torch_to_jax_via_meta(v),
        "chunk_size": chunk_size,
    })
    out_struct, _ = derived["out"]
    a_ab_inv_struct, merged_out_struct, s_in_struct = derived["residuals"]
    return (
        q.new_empty(out_struct.shape),
        q.new_empty(a_ab_inv_struct.shape),
        q.new_empty(merged_out_struct.shape, dtype=torch.float32),
        q.new_empty(s_in_struct.shape, dtype=torch.float32),
    )

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
      configs: tuple[Config | None, Config | None] | None = None,
  ) -> torch.Tensor:
    fwd_config, bwd_config = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert self.backward_op_torch is not None, "Backward op not set."
    out, _, _, _ = self._torch_tokamax_op(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
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
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = True,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    del config, bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    self.op_impl_jax.bind(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=return_residuals,
    )
    (out, _), (a_ab_inv, merged_out, s_in) = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=scale,
        return_residuals=return_residuals,
        config=None,
    )
    return out, a_ab_inv, merged_out, s_in

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
        alpha,
        beta,
        gk,
        chunk_size,
        scale,
        _,
        _,
        bwd_config,
    ) = inputs
    out, a_ab_inv, merged_out, s_in = output
    ctx.save_for_backward(
        q, k, v, alpha, beta, gk, out, a_ab_inv, merged_out, s_in
    )
    ctx.chunk_size = chunk_size
    ctx.scale = scale
    ctx.bwd_config = bwd_config

  @override
  def backward(
      self,
      ctx: Any,
      dout: torch.Tensor,
      d_a_ab_inv: torch.Tensor,
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
    del d_a_ab_inv, d_merged_out, d_s_in
    assert self.backward_op_torch is not None, "Backward op not set."
    (
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        out,
        a_ab_inv,
        merged_out,
        s_in,
    ) = ctx.saved_tensors
    dq, dk, dv, dalpha, dbeta, dgk = self.backward_op_torch(
        a_ab_inv,
        merged_out,
        s_in,
        out,
        dout,
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=ctx.chunk_size,
        scale=ctx.scale,
        config=ctx.bwd_config,
    )
    return (dq, dk, dv, dalpha, dbeta, dgk, None, None, None, None, None)


# Singleton instance of the base Rwkv7 operator.
Rwkv7 = _Rwkv7[Any]()
