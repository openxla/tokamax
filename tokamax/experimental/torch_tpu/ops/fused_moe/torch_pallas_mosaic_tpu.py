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
"""Tokamax operator wrapper for Pallas Mosaic TPU Fused MoE."""

from typing import Any, override

import jax
import numpy as np
from tokamax._src.ops.experimental.fused_moe import kernel as jax_kernel
from tokamax._src.ops.experimental.fused_moe import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.fused_moe import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuFusedMoe(
    torch_base._FusedMoe[Config]  # pylint: disable=protected-access
):
  """Tokamax operator wrapper for Pallas Mosaic TPU Fused MoE."""

  def __init__(self) -> None:
    super().__init__()
    self.jax_op_name = "pallas_mosaic_tpu_fused_moe"
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuFusedMoe()
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, int, int, int, int] | None:
    """Converts Config into a 5-int tuple for jax_op static_argnums."""
    if config is None:
      return None
    if isinstance(config, Config):
      capacity = config.capacity
      block = config.block
      ragged_stride = config.ragged_stride
      rhs_qb = config.rhs_qb
      sharded_plan = config.sharded_plan
    else:
      capacity, block, ragged_stride, rhs_qb, sharded_plan = config
    return (
        int(capacity),
        -1 if block is None else int(block),
        -1 if ragged_stride is None else int(ragged_stride),
        -1 if rhs_qb is None else int(rhs_qb),
        int(bool(sharded_plan)),
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the 5-int tuple back into a Config."""
    config = config_parts[0]
    assert config is not None, "Config not set."
    capacity, block, ragged_stride, rhs_qb, sharded_plan = config
    # When `config=None` is passed to `__call__`, `deconstruct_config(None)`
    # returns `None` and `TorchOp.op_impl_call_config_setup` populates `config`
    # via `dataclasses.astuple` (bypassing `deconstruct_config`), so optional
    # fields arrive here as `None` rather than `-1`.
    return Config(
        capacity=int(capacity),
        block=None if block is None or block < 0 else int(block),
        ragged_stride=(
            None
            if ragged_stride is None or ragged_stride < 0
            else int(ragged_stride)
        ),
        rhs_qb=None if rhs_qb is None or rhs_qb < 0 else int(rhs_qb),
        sharded_plan=bool(sharded_plan),
    )

  @override
  def __call__(
      self,
      x: torch.Tensor,
      w1: torch.Tensor,
      w2: torch.Tensor,
      gating: torch.Tensor,
      w1_scale: torch.Tensor | None = None,
      w2_scale: torch.Tensor | None = None,
      w1_bias: torch.Tensor | None = None,
      w2_bias: torch.Tensor | None = None,
      *,
      topk: int = 2,
      renormalize: bool = True,
      act_fn: str = "silu",
      config: Config | tuple[Any, ...] | list[Any] | None = None,
  ) -> torch.Tensor:
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. self.op_impl_jax was not set in the"
        " constructor."
    )
    return self._torch_tokamax_op(
        x,
        w1,
        w2,
        gating,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
        config=self.deconstruct_config(config),
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      w1: jax.Array,
      w2: jax.Array,
      gating: jax.Array,
      w1_scale: jax.Array | None = None,
      w2_scale: jax.Array | None = None,
      w1_bias: jax.Array | None = None,
      w2_bias: jax.Array | None = None,
      topk: int = 2,
      renormalize: bool = True,
      act_fn: str = "silu",
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    assert (
        self.op_impl_jax is not None
    ), "Forward class not set. self.op_impl_jax was not set in the constructor."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]), axis_names=(jax_kernel.AXIS,)
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        w1,
        w2,
        gating,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk=topk,
        renormalize=renormalize,
        act_fn=act_fn,
        mesh=mesh,
        config=kernel_config,
    )
    return out


PallasMosaicTpuFusedMoe = _PallasMosaicTpuFusedMoe()  # pylint: disable=invalid-name
