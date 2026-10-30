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

"""PyTorch interface for Pallas TPU Raven GSA operator."""

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension
#   tt     tile extent of T (chunk_size)

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.raven import torch_base
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuRavenGsaVjp(torch_base._RavenGsaVjp[Config]):  # pylint: disable=protected-access
  """PyTorch VJP wrapper for Pallas TPU Raven GSA backward operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuRavenGsaVjp()
    self.jax_op_name = "pallas_mosaic_tpu_raven_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  @override
  def op_impl_call(
      self,
      out: jax.Array,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      v: jax.Array | None = None,
      chunk_size: int = 64,
      scale: float | None = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, "Backward JAX op not set."
    kernel_config = (
        self.reconstruct_config(config)
        if config is not None
        else Config(chunk_size=chunk_size)
    )
    residuals = (
        q,
        k,
        s,
        g,
        v,
        None,
        None,
        None,
        None,
        kernel_config.chunk_size,
        scale,
    )
    grads, _ = self.op_impl_jax._fwd(
        residuals,
        (out, None),
        (dout, None),
        q,
        k,
        s,
        g,
        v,
        chunk_size=kernel_config.chunk_size,
        scale=scale,
        return_residuals=return_residuals,
        config=kernel_config,
    )
    dv = grads.get("v")
    dv_arr = dv if dv is not None else jnp.zeros((), dtype=q.dtype)
    return grads["q"], grads["k"], grads["s"], grads["g"], dv_arr


PallasMosaicTpuRavenGsaVjp = _PallasMosaicTpuRavenGsaVjp()


class _PallasMosaicTpuRavenGsa(torch_base._RavenGsa[Config]):  # pylint: disable=protected-access
  """PyTorch wrapper for Pallas TPU Raven GSA operator."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuRavenGsa()
    self.backward_op_torch = PallasMosaicTpuRavenGsaVjp
    self.jax_op_name = "pallas_mosaic_tpu_raven"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

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
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    kernel_config = (
        self.reconstruct_config(config)
        if config is not None
        else Config(chunk_size=chunk_size)
    )
    (out, _), _ = self.op_impl_jax._fwd(
        q,
        k,
        s,
        g,
        v,
        chunk_size=kernel_config.chunk_size,
        scale=scale,
        return_residuals=return_residuals,
        config=kernel_config,
    )
    return out


# Singleton instance
PallasMosaicTpuRavenGsa = _PallasMosaicTpuRavenGsa()
