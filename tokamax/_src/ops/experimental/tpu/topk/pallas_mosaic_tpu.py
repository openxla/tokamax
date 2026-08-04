# Copyright 2026 Google LLC
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
"""Pallas/Mosaic kernel wrapper for TopK on TPU."""

from typing import ClassVar, override

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Int, Shaped
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.tpu.topk import base
from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu_kernel


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Autotuning and execution configuration for TopK Pallas TPU kernel."""

  num_seq_windows: pydantic.conint(gt=0) = 1
  digit_width: pydantic.conint(gt=0) = 4
  num_digits: pydantic.conint(gt=0) = 8
  poison_scratch: bool = False
  use_tc_tiling_on_sc: bool = False
  debug: bool = False


class PallasTpuTopK(base.TopK):
  """Tokamax operator wrapper for Pallas Mosaic TPU TopK kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      operand: Shaped[Array, "*batch N"],
      k: int,
      values: Int[Array, "*batch N"] | None = None,
      *,
      axis: int = -1,
      is_stable: bool = True,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    del axis, is_stable
    if config is None:
      config = self._get_heuristics_config(None)

    res_keys, res_vals = pallas_mosaic_tpu_kernel.top_k(
        keys=operand,
        values=values,
        k=k,
        num_seq_windows=config.num_seq_windows,
        digit_width=config.digit_width,
        num_digits=config.num_digits,
        poison_scratch=config.poison_scratch,
        use_tc_tiling_on_sc=config.use_tc_tiling_on_sc,
        debug=config.debug,
    )
    # The kernel does not guarantee sorted output, so we sort it here.
    # jax.lax.sort sorts ascending, so we flip it to get descending.
    sorted_keys, sorted_vals = jax.lax.sort((res_keys, res_vals), dimension=-1)
    return (
        jnp.flip(sorted_keys, axis=-1),
        jnp.flip(sorted_vals, axis=-1),
    ), None

  # TODO: Add correct heuristics config and autotuning search space.
  @override
  def _get_heuristics_config(self, ba) -> Config:
    return Config()

  @override
  def _get_autotuning_configs(self, ba) -> set[Config]:
    return {
        Config(num_seq_windows=1, digit_width=4, num_digits=8),
        Config(num_seq_windows=2, digit_width=4, num_digits=8),
    }

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6
