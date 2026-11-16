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
from typing import Any, ClassVar, override

import immutabledict
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_kernel_sm100 as sm100
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_kernel_sm80 as sm80
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_kernel_sm90 as sm90

Residuals = base.Residuals
Config = common.Config
type Key = immutabledict.immutabledict[str, Any]


def _get_kernel_module():
  if not gpu_utils.has_mosaic_gpu_support():
    raise NotImplementedError("Mosaic GPU not supported on this platform.")
  if gpu_utils.is_sm100():
    return sm100
  if gpu_utils.is_sm90():
    return sm90
  if gpu_utils.is_sm80():
    return sm80
  raise NotImplementedError("Only supported for sm80, sm90, and sm100 GPUs.")


@dataclasses.dataclass(frozen=True, slots=True)
class PallasMosaicGpuGatedLinearUnit(base.GatedLinearUnit[Config, Key]):
  """Pallas-Mosaic-GPU gated linear unit."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return (
        gpu_utils.is_sm80(device)
        or gpu_utils.is_sm90(device)
        or gpu_utils.is_sm100(device)
    )

  @jaxtyping.jaxtyped
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
    if isinstance(weights, tuple):
      if jnp.dtype(weights[0].dtype) != jnp.dtype(weights[1].dtype):
        raise NotImplementedError(
            "Mosaic GPU GLU requires matching gate and projection weight"
            f" dtypes, got {weights[0].dtype} vs {weights[1].dtype}."
        )
      w_dtype = weights[0].dtype
    else:
      w_dtype = weights.dtype

    store_dtype = jnp.dtype(x.dtype)
    preset, _, _ = common.resolve_dot_precision(store_dtype, w_dtype, precision)
    approx_math = preset != jax.lax.DotAlgorithmPreset.F32_F32_F32

    activation = activation or (lambda x: x)
    glu_fn = _get_kernel_module().gated_linear_unit

    def fwd_single(
        x_arg: jax.Array, w_arg: jax.Array | tuple[jax.Array, jax.Array]
    ):
      all_dims = (
          *x_arg.shape,
          *(d for w in jax.tree.leaves(w_arg) for d in w.shape),
      )
      if not all(isinstance(d, int) for d in all_dims):
        raise NotImplementedError(
            "Mosaic GPU GLU requires static integer dimensions, got"
            f" {x_arg.shape=} and {[w.shape for w in jax.tree.leaves(w_arg)]}."
        )
      x_dot, w_dot = common.prepare_dot_inputs(x_arg, w_arg, preset)
      return glu_fn(
          x_dot,
          w_dot,
          activation=activation,
          config=config,
          out_dtype=store_dtype,
          return_residuals=return_residuals,
          approx_math=approx_math,
      )

    fn = self._with_vmap(fwd_single, fallback_to_sequential=False)
    if return_residuals:
      return fn(x, weights)
    return fn(x, weights), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return _get_kernel_module().get_heuristics_config(ba)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return _get_kernel_module().get_autotuning_configs(ba)
