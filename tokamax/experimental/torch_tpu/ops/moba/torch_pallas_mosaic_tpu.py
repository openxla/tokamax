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

"""PyTorch interface for Pallas TPU Mixture of Block Attention (MoBA) operator."""

from typing import Any, Optional
import jax
from tokamax._src.ops.experimental.moba import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.moba import torch_base
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


def _deconstruct_config(
    config: Config | tuple[Any, ...] | list[Any] | None,
) -> tuple[int, ...] | None:
  if config is None:
    return None
  if isinstance(config, (tuple, list)):
    return (int(config[0]), int(config[1]))
  return (int(config.chunk_size), int(config.topk))


def _reconstruct_config(*config_parts: Any) -> Config:
  raw = config_parts[0]
  if isinstance(raw, Config):
    return raw
  assert raw is not None, "Config not set."
  elements = raw if len(config_parts) == 1 else config_parts
  return Config(chunk_size=int(elements[0]), topk=int(elements[1]))


class _PallasMosaicTpuMixtureOfBlockAttentionVjp(
    torch_base._MixtureOfBlockAttentionVjp[Config]  # pylint: disable=protected-access
):
  """PyTorch VJP wrapper for Pallas Mosaic TPU MoBA operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuMixtureOfBlockAttentionVjp()
    )
    self.jax_op_name = "pallas_mosaic_tpu_moba_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, ...] | None:
    return _deconstruct_config(config)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    return _reconstruct_config(*config_parts)

  @override
  def op_impl_call(
      self,
      out: jax.Array,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      topk: int = torch_base._TOPK,  # pylint: disable=protected-access
      chunk_size: int = torch_base._CHUNK_SIZE,  # pylint: disable=protected-access
      scale: Optional[float] = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Backward config not set."
    kernel_config = self.reconstruct_config(config)
    grads, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        None,
        out,
        dout,
        q,
        k,
        v,
        topk=topk,
        chunk_size=chunk_size,
        scale=scale,
        config=kernel_config,
        return_residuals=return_residuals,
    )
    return grads


PallasMosaicTpuMixtureOfBlockAttentionVjp = (
    _PallasMosaicTpuMixtureOfBlockAttentionVjp()
)


class _PallasMosaicTpuMixtureOfBlockAttention(
    torch_base._MixtureOfBlockAttention[Config]  # pylint: disable=protected-access
):
  """PyTorch wrapper for Pallas Mosaic TPU MoBA operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuMixtureOfBlockAttention()
    )
    self.backward_op_torch = PallasMosaicTpuMixtureOfBlockAttentionVjp
    self.jax_op_name = "pallas_mosaic_tpu_moba"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, ...] | None:
    return _deconstruct_config(config)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    return _reconstruct_config(*config_parts)

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      topk: int = torch_base._TOPK,  # pylint: disable=protected-access
      chunk_size: int = torch_base._CHUNK_SIZE,  # pylint: disable=protected-access
      scale: Optional[float] = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        k,
        v,
        topk=topk,
        chunk_size=chunk_size,
        scale=scale,
        config=kernel_config,
        return_residuals=return_residuals,
    )
    return out


# Singleton instance
PallasMosaicTpuMixtureOfBlockAttention = (
    _PallasMosaicTpuMixtureOfBlockAttention()
)
