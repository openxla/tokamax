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

"""PyTorch interface for Pallas TPU RWKV-7 operator."""

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
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.rwkv7 import torch_base
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuRwkv7Vjp(
    torch_base._Rwkv7Vjp[Config]  # pylint: disable=protected-access
):
  """PyTorch VJP wrapper for Pallas Mosaic TPU RWKV-7 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuRwkv7Vjp()
    self.jax_op_name = "pallas_mosaic_tpu_rwkv7_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

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
      config: Config | tuple[int, ...] | list[int] | None = None,
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
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Backward config not set."
    kernel_config = self.reconstruct_config(config)
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
        config=kernel_config,
    )
    return (
        grads["q"],
        grads["k"],
        grads["v"],
        grads["alpha"],
        grads["beta"],
        grads["gk"],
    )


# Singleton instance of Pallas Mosaic TPU RWKV-7 VJP.
PallasMosaicTpuRwkv7Vjp = (  # pylint: disable=invalid-name
    _PallasMosaicTpuRwkv7Vjp()
)


class _PallasMosaicTpuRwkv7(
    torch_base._Rwkv7[Config]  # pylint: disable=protected-access
):
  """PyTorch wrapper for Pallas Mosaic TPU RWKV-7 operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuRwkv7()
    self.backward_op_torch = PallasMosaicTpuRwkv7Vjp
    self.jax_op_name = "pallas_mosaic_tpu_rwkv7"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

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
    del bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
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
        config=kernel_config,
    )
    return out, a_ab_inv, merged_out, s_in

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    """Callback for torch register_autograd."""
    super().setup_context(ctx, inputs, output)

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
    bwd_config = (
        None
        if ctx.bwd_config is None
        else self.backward_op_torch.reconstruct_config(ctx.bwd_config)
    )
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
        config=bwd_config,
    )
    return (dq, dk, dv, dalpha, dbeta, dgk, None, None, None, None, None)


# Singleton instance of Pallas Mosaic TPU RWKV-7.
PallasMosaicTpuRwkv7 = (  # pylint: disable=invalid-name
    _PallasMosaicTpuRwkv7()
)
