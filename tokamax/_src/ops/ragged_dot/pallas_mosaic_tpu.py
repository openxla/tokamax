# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pallas Mosaic TPU Megablox."""

import dataclasses
import functools
import types
from typing import Any, Callable, Literal, Sequence, Tuple, TypeVar
from absl import logging
import jax
import jax.numpy as jnp
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as backend

conint = pydantic.conint


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Pallas Mosaic TPU Ragged Dot config."""

  gmm_tiling: tuple[
      conint(ge=128, multiple_of=8),
      conint(ge=128, multiple_of=8),
      conint(ge=128, multiple_of=8),
  ]


Residuals = types.NoneType


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuRaggedDot(base.RaggedDot[Config, None]):
  """Pallas-Mosaic-TPU ragged dot implementation.

  TPU Implementation of the Megablocks Paper https://arxiv.org/abs/2211.15841.
  """

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", PallasMosaicTpuRaggedDotVjp)

  def _fwd(  # pytype: disable=signature-mismatch
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      *,
      group_sizes: jax.Array | base.GroupSizes,
      ragged_dot_dimension_numbers: (
          jax.lax.RaggedDotDimensionNumbers | None
      ) = None,
      precision: jax.lax.DotAlgorithmPreset,
      preferred_element_type: jax.typing.DTypeLike,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[jax.Array, None]:
    # TODO: Support more ragged_dot_dimension_numbers configurations.
    del precision
    tiling = config.gmm_tiling
    if ragged_dot_dimension_numbers != base.DEFAULT_RAGGED_DOT_DIM_NUMS:
      raise NotImplementedError(
          "Only default `ragged_dot_dimension_numbers` supported."
      )
    if isinstance(group_sizes, base.GroupSizes):
      group_sizes = jnp.array(group_sizes)
    # TODO: Once we know customer precision requirements, we should
    # use that instead of float32. This is a temporary solution to unblock
    # testing.
    if preferred_element_type is None:
      preferred_element_type = jnp.result_type(lhs.dtype, rhs.dtype)

    out = backend.gmm(
        lhs,
        rhs,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        tiling=tiling,
        group_offset=None,
        existing_out=None,
        transpose_rhs=False,
        interpret=False,
    )

    return out, None

  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba  # Unused.
    # For now, return a basic tile config.
    return Config(
        gmm_tiling=(128, 128, 128),
    )

  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba  # Unused.
    # TODO: Add more configs.
    configs = set()
    tile_range = range(128, 1024, 128)
    for m in tile_range:
      for n in tile_range:
        for k in tile_range:
          configs.add(
              Config(
                  gmm_tiling=(m, n, k),
              )
          )
    return configs


def PallasMosaicTpuRaggedDotVjp(
    residual: Residuals,
    out: jax.Array,
    dout: jax.Array,
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    group_sizes: jax.Array | base.GroupSizes,
    ragged_dot_dimension_numbers: (
        jax.lax.RaggedDotDimensionNumbers | None
    ) = None,
    precision: jax.lax.DotAlgorithmPreset,
    preferred_element_type: jax.typing.DTypeLike,
    dlhs_ragged_dot: Callable[..., jax.Array] = base.RaggedDot(),
    drhs_ragged_dot: Callable[..., jax.Array] = base.RaggedDot(),
) -> tuple[tuple[jax.Array, jax.Array], None]:
  del residual, preferred_element_type
  tiling = (128, 128, 128)
  group_offset = jnp.array(0, dtype=jnp.int32)
  transpose_rhs = False
  interpret = False
  if isinstance(group_sizes, base.GroupSizes):
    group_sizes = jnp.array(group_sizes)

  grad_lhs = backend.gmm(
      dout,
      rhs,
      group_sizes=group_sizes,
      preferred_element_type=lhs.dtype,
      tiling=tiling,
      group_offset=group_offset,
      transpose_rhs=not transpose_rhs,
      interpret=interpret,
  )
  grad_rhs = backend.tgmm(
      lhs.swapaxes(0, 1),
      dout,
      group_sizes,
      preferred_element_type=rhs.dtype,
      tiling=tiling,
      group_offset=group_offset,
      num_actual_groups=rhs.shape[0],
      interpret=interpret,
  )

  # NOTE: If the rhs transposition is fused into the forward pass we need to
  # return the transpose of the rhs gradient that we calculated above.
  #
  # TODO: Fuse this transposition into the tgmm.
  grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs
  # return grad_lhs, grad_rhs, None, None, grad
  return (grad_lhs, grad_rhs), None
