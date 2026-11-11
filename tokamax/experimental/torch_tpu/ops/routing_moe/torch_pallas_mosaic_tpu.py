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
"""Tokamax PyTorch operator wrapper for Pallas Mosaic TPU Routing MoE."""

from typing import Any, override

import jax
from tokamax._src.ops.experimental.routing_moe import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.routing_moe import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


def _encode_opt_int(value: int | None) -> int:
  return -1 if value is None or value < 0 else int(value)


def _encode_opt_bool(value: bool | int | None) -> int:
  if value is None or (not isinstance(value, bool) and value < 0):
    return -1
  return int(bool(value))


class _PallasMosaicTpuRoutingMoe(
    torch_base._RoutingMoe[Config]  # pylint: disable=protected-access
):
  """Tokamax PyTorch operator wrapper for Pallas Mosaic TPU Routing MoE."""

  def __init__(self) -> None:
    super().__init__()
    self.jax_op_name = "pallas_mosaic_tpu_routing_moe"
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, int, int, int, int, int, int] | None:
    """Serializes `Config` into a 7-int tuple for `jax_op`."""
    if config is None:
      return None
    if isinstance(config, Config):
      vals = (
          config.tile_m,
          config.bucket_base,
          config.num_xo_buffers,
          config.num_w_buffers,
          config.x_quant_block_size,
          config.use_fp32_acc,
          config.live_group_walk,
      )
    else:
      vals = tuple(config)
    tile_m, bucket_base, num_xo_bufs, num_w_bufs, x_qb, use_fp32, lgw = vals
    return (
        _encode_opt_int(tile_m),
        _encode_opt_int(bucket_base),
        _encode_opt_int(num_xo_bufs),
        _encode_opt_int(num_w_bufs),
        _encode_opt_int(x_qb),
        int(bool(use_fp32)),
        _encode_opt_bool(lgw),
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds `Config` from its 7-int tuple."""
    config = config_parts[0]
    assert config is not None, "Config not set."
    (
        tile_m,
        bucket_base,
        num_xo_buffers,
        num_w_buffers,
        x_quant_block_size,
        use_fp32_acc,
        lgw_int,
    ) = config
    return Config(
        tile_m=None if tile_m < 0 else int(tile_m),
        bucket_base=None if bucket_base < 0 else int(bucket_base),
        num_xo_buffers=None if num_xo_buffers < 0 else int(num_xo_buffers),
        num_w_buffers=None if num_w_buffers < 0 else int(num_w_buffers),
        x_quant_block_size=(
            None if x_quant_block_size < 0 else int(x_quant_block_size)
        ),
        use_fp32_acc=bool(use_fp32_acc),
        live_group_walk=None if lgw_int < 0 else bool(lgw_int),
    )

  @override
  def __call__(
      self,
      x: torch.Tensor,
      topk_weights: torch.Tensor,
      topk_indices: torch.Tensor,
      w1: torch.Tensor,
      w2: torch.Tensor,
      w1_scale: torch.Tensor | None = None,
      w2_scale: torch.Tensor | None = None,
      w1_bias: torch.Tensor | None = None,
      w2_bias: torch.Tensor | None = None,
      experts_start: torch.Tensor | None = None,
      *,
      global_num_experts: int | None = None,
      act_fn: str = "silu",
      skip_padded_tokens: bool = False,
      config: Config | tuple[Any, ...] | list[Any] | None = None,
  ) -> torch.Tensor:
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(
        x,
        topk_weights,
        topk_indices,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        experts_start,
        global_num_experts=global_num_experts,
        act_fn=act_fn,
        skip_padded_tokens=skip_padded_tokens,
        config=self.deconstruct_config(config),
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      topk_weights: jax.Array,
      topk_indices: jax.Array,
      w1: jax.Array,
      w2: jax.Array,
      w1_scale: jax.Array | None = None,
      w2_scale: jax.Array | None = None,
      w1_bias: jax.Array | None = None,
      w2_bias: jax.Array | None = None,
      experts_start: jax.Array | None = None,
      global_num_experts: int | None = None,
      act_fn: str = "silu",
      skip_padded_tokens: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        topk_weights,
        topk_indices,
        w1,
        w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        experts_start=experts_start,
        global_num_experts=global_num_experts,
        act_fn=act_fn,
        skip_padded_tokens=skip_padded_tokens,
        config=kernel_config,
    )
    return out


PallasMosaicTpuRoutingMoe = _PallasMosaicTpuRoutingMoe()  # pylint: disable=invalid-name
