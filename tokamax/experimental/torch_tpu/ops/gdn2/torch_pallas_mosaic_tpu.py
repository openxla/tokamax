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

"""PyTorch interface for Pallas Mosaic TPU GDN-2 operator."""

# Dimensions.
#   B   batch size
#   T   sequence length (nt * tt)
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension
#   BH  flattened batch-head count (B * H)
#   tt  tile extent of T (chunk_size)
#   nt  grid extent along T (T // tt)

from typing import Any

import jax
from tokamax._src.ops.experimental.gdn2 import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.gdn2 import torch_base
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuGatedDeltaNet2Vjp(
    torch_base._GatedDeltaNet2Vjp[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax GDN-2 VJP Op wrapped for PyTorch."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet2Vjp()
    )
    self.jax_op_name = "pallas_mosaic_tpu_gdn2_vjp"
    self.is_vjp = True

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    config = config_parts[0]
    assert config is not None, "Backward config not set."
    return Config(chunk_size=int(config[0]))

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
      config: Config | tuple[int, ...] | list[int] | None = None,
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
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Backward config not set."
    kernel_config = self.reconstruct_config(config)
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
        config=kernel_config,
        return_residuals=return_residuals,
    )
    return (
        grads["q"],
        grads["k"],
        grads["v"],
        grads["g"],
        grads["b"],
        grads["w"],
    )


class _PallasMosaicTpuGatedDeltaNet2(
    torch_base._GatedDeltaNet2[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax GDN-2 Op wrapped for PyTorch."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet2()
    self.backward_op_torch = _PallasMosaicTpuGatedDeltaNet2Vjp()
    self.jax_op_name = "pallas_mosaic_tpu_gdn2"

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    config = config_parts[0]
    assert config is not None, "Forward config not set."
    return Config(chunk_size=int(config[0]))

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
    ), "Forward op not registered. Call register_ops first."
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
        return_residuals=not self.is_vjp,
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
    del bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    (out, _), (a_mat_c, merged_out_c, s_in_c) = (
        self.op_impl_jax._fwd(  # pylint: disable=protected-access
            q,
            k,
            v,
            g,
            b,
            w,
            chunk_size=chunk_size,
            scale=scale,
            return_residuals=return_residuals,
            config=kernel_config,
        )
    )
    return out, a_mat_c, merged_out_c, s_in_c


# Singleton instance of Pallas Mosaic TPU Gated Delta Net 2.
PallasMosaicTpuGatedDeltaNet2 = _PallasMosaicTpuGatedDeltaNet2()
