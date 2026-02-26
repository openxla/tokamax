# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pallas-Mosaic-GPU implementation of the gated linear unit op."""

from collections.abc import Callable
import dataclasses
import functools
from typing import ClassVar
import jax
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import gpu_utils
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_kernel_sm100
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_kernel_sm90
from typing_extensions import override

Residuals = base.Residuals
Config = common.Config


@dataclasses.dataclass(frozen=True, slots=True)
class PallasMosaicGpuGatedLinearUnit(base.GatedLinearUnit[Config, None]):
  """Pallas-Mosaic-GPU gated linear unit."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.is_sm90(device) or gpu_utils.is_sm100(device)

  @override
  def _fwd(
      self,
      x: Float[Array, "*B M K"],
      weights: base.FusedWeights | base.UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      precision: base.CanonicalPrecision,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B M N"], Residuals | None]:
    weights = (
        jnp.stack(weights, axis=-2) if isinstance(weights, tuple) else weights
    )

    activation = activation or (lambda x: x)
    if return_residuals:
      return base.GatedLinearUnit._fwd(  # pylint: disable=protected-access
          self,
          x,
          weights,
          activation=activation,
          precision=precision,
          return_residuals=True,
          config=config,
      )

    if not precision_lib.is_default(x.dtype, weights.dtype, precision):
      raise NotImplementedError(f"{precision=} is not supported.")

    if gpu_utils.is_sm100():
      glu_fn = pallas_mosaic_gpu_kernel_sm100.gated_linear_unit_sm100
    elif gpu_utils.is_sm90():
      glu_fn = pallas_mosaic_gpu_kernel_sm90.gated_linear_unit_sm90
    else:
      device_kind = backend.get_default_device().device_kind.lower()
      raise NotImplementedError(f"Unsupported device kind: {device_kind}")
    fn = functools.partial(glu_fn, activation=activation, config=config)
    fn = self._with_vmap(fn, fallback_to_sequential=False)
    return fn(x, weights), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        tile_m=128,
        tile_n=64,
        tile_k=64,
        num_stages=4,
    )
