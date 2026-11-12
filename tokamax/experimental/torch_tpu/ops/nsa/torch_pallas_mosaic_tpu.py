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

"""PyTorch interface for Pallas TPU Native Sparse Attention (NSA) operator."""

# Dimensions.
#   B   batch size
#   T   sequence length (PyTorch caller layout: [B, T, H, D])
#   H   number of attention heads
#   D   head dimension

from typing import Any
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.nsa import torch_base
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


def _deconstruct_nsa_config(
    config: Config | tuple[Any, ...] | list[Any] | None,
) -> tuple[int, ...] | None:
  """Deconstructs a Config into a flat tuple of ints for jax_op."""
  if config is None:
    return None
  if isinstance(config, (tuple, list)):
    return tuple(int(x) for x in config)
  return (
      int(config.chunk_size),
      int(config.topk),
      int(config.window),
      int(config.cmp_block_size),
  )


def _reconstruct_nsa_config(*config_parts: Any) -> Config:
  """Reconstructs a Config from its deconstructed representation."""
  if len(config_parts) == 1:
    config = config_parts[0]
    if config is None:
      raise ValueError("Config is None in reconstruct_config.")
    if isinstance(config, Config):
      return config
  else:
    config = config_parts
  chunk_size, topk, window, cmp_block_size = config
  return Config(
      chunk_size=int(chunk_size),
      topk=int(topk),
      window=int(window),
      cmp_block_size=int(cmp_block_size),
  )


class _PallasMosaicTpuNativeSparseAttentionVjp(
    torch_base._NativeSparseAttentionVjp[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Native Sparse Attention VJP wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuNativeSparseAttentionVjp()
    )
    self.jax_op_name = "pallas_mosaic_tpu_native_sparse_attention_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, ...] | None:
    return _deconstruct_nsa_config(config)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    return _reconstruct_nsa_config(*config_parts)

  @override
  def op_impl_call(
      self,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g_cmp: jax.Array,
      g_slc: jax.Array,
      g_swa: jax.Array,
      chunk_size: int = torch_base._CHUNK_SIZE,  # pylint: disable=protected-access
      topk: int = torch_base._TOPK,  # pylint: disable=protected-access
      window: int = torch_base._WINDOW,  # pylint: disable=protected-access
      cmp_block_size: int = torch_base._CMP_BLOCK_SIZE,  # pylint: disable=protected-access
      scale: float | None = None,
      config: tuple[int, ...] | None = None,
  ) -> tuple[
      jax.Array,
      jax.Array,
      jax.Array,
      jax.Array,
      jax.Array,
      jax.Array,
  ]:
    assert config is not None, "Config not set."
    assert self.op_impl_jax is not None, "Forward class not set."
    kernel_config = self.reconstruct_config(config)
    out_dummy = jnp.zeros_like(q)
    (dq, dk, dv, dg_cmp, dg_slc, dg_swa), _ = (
        self.op_impl_jax._fwd(  # pylint: disable=protected-access
            None,
            out_dummy,
            dout,
            q,
            k,
            v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            chunk_size=chunk_size,
            topk=topk,
            window=window,
            cmp_block_size=cmp_block_size,
            scale=scale,
            config=kernel_config,
            return_residuals=False,
        )
    )
    assert dg_cmp is not None and dg_slc is not None and dg_swa is not None
    return dq, dk, dv, dg_cmp, dg_slc, dg_swa


class _PallasMosaicTpuNativeSparseAttention(
    torch_base._NativeSparseAttention[Config]  # pylint: disable=protected-access
):
  """PyTorch wrapper for Pallas Mosaic TPU NSA operator."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuNativeSparseAttention()
    )
    self.backward_op_torch = _PallasMosaicTpuNativeSparseAttentionVjp()
    self.jax_op_name = "pallas_mosaic_tpu_native_sparse_attention"
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, ...] | None:
    return _deconstruct_nsa_config(config)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    return _reconstruct_nsa_config(*config_parts)

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g_cmp: jax.Array,
      g_slc: jax.Array,
      g_swa: jax.Array,
      chunk_size: int = torch_base._CHUNK_SIZE,  # pylint: disable=protected-access
      topk: int = torch_base._TOPK,  # pylint: disable=protected-access
      window: int = torch_base._WINDOW,  # pylint: disable=protected-access
      cmp_block_size: int = torch_base._CMP_BLOCK_SIZE,  # pylint: disable=protected-access
      scale: float | None = None,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del bwd_config
    assert config is not None, "Config not set."
    assert self.op_impl_jax is not None, "Forward class not set."
    self.op_impl_jax.bind(
        q,
        k,
        v,
        g_cmp,
        g_slc,
        g_swa,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
    )
    kernel_config = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        k,
        v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
        config=kernel_config,
        return_residuals=False,
    )
    return out


# Singleton instance of the Pallas Mosaic TPU NativeSparseAttention op.
PallasMosaicTpuNativeSparseAttention = (  # pylint: disable=invalid-name
    _PallasMosaicTpuNativeSparseAttention()
)
