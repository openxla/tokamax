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
"""Pallas Mosaic TPU operator implementation for Tokamax Routing MoE."""

from typing import ClassVar, override

import jax
from jax.experimental import pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.routing_moe import base
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import kernel as gmm
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan_lib
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.io_relayout import output_reduce

_SUBLANES = 8
_PACKED_SORT_SHIFT = 20
_MAX_PACKED_GROUPS = 1 << (31 - _PACKED_SORT_SHIFT)
_MAX_PACKED_ROWS = 1 << _PACKED_SORT_SHIFT


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Tuning parameters for single-shard / EP-shard Routing MoE on TPU."""

  tile_m: int | None = None
  bucket_base: int | None = None
  num_xo_buffers: int | None = None
  num_w_buffers: int | None = None
  x_quant_block_size: int | None = None
  use_fp32_acc: bool = True
  live_group_walk: bool | None = None


def _default_config_for_tokens(num_tokens: int) -> Config:
  """Selects low-token decode buckets or heuristic defaults from `num_tokens`."""
  aligned = max(_SUBLANES, _SUBLANES * pl.cdiv(num_tokens, _SUBLANES))
  if num_tokens <= 64:
    return Config(
        tile_m=min(16, aligned),
        bucket_base=_SUBLANES,
        num_w_buffers=3,
        use_fp32_acc=True,
    )
  if num_tokens <= 256:
    return Config(
        tile_m=min(32, aligned),
        bucket_base=min(16, aligned),
        num_w_buffers=3,
        use_fp32_acc=True,
    )
  return Config(use_fp32_acc=True)


def _to_gmm_plan(cfg: Config) -> gmm_plan_lib.RoutingGmmPlan:
  """Converts operator `Config` into `RoutingGmmPlan`."""
  return gmm_plan_lib.RoutingGmmPlan(
      tile_m=cfg.tile_m,
      bucket_base=cfg.bucket_base,
      num_xo_buffers=cfg.num_xo_buffers,
      num_w_buffers=cfg.num_w_buffers,
      x_quant_block_size=cfg.x_quant_block_size,
      acc_dtype=jnp.float32 if cfg.use_fp32_acc else jnp.bfloat16,
      live_group_walk=cfg.live_group_walk,
  )


def _sort_routed_indices(
    flat_indices: jax.Array, num_groups: int
) -> tuple[jax.Array, jax.Array]:
  """Sorts `flat_indices` stably, returning `(row_group, argsort_indices)`."""
  num_rows = flat_indices.size
  if num_groups < _MAX_PACKED_GROUPS and num_rows <= _MAX_PACKED_ROWS:
    iota = jnp.arange(num_rows, dtype=jnp.uint32)
    packed = (flat_indices.astype(jnp.uint32) << _PACKED_SORT_SHIFT) | iota
    sorted_packed = jax.lax.sort(packed)
    row_group = (sorted_packed >> _PACKED_SORT_SHIFT).astype(jnp.int32)
    mask = jnp.uint32(_MAX_PACKED_ROWS - 1)
    argsort_indices = (sorted_packed & mask).astype(jnp.int32)
    return row_group, argsort_indices
  return jax.lax.sort_key_val(
      flat_indices,
      jnp.arange(num_rows, dtype=jnp.int32),
      is_stable=True,
  )


def _prepare_routing(
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    *,
    num_experts: int,
    skip_padded_tokens: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds effective indices, sorted `(row_group, argsort)` and `group_sizes`."""
  indices_i32 = topk_indices.astype(jnp.int32)
  in_range = (indices_i32 >= 0) & (indices_i32 < num_experts)
  if skip_padded_tokens:
    valid = in_range & (topk_weights != 0)
  else:
    valid = in_range
  effective_indices = jnp.where(valid, indices_i32, jnp.int32(num_experts))
  flat_indices = effective_indices.reshape(-1)
  row_group, argsort_indices = _sort_routed_indices(
      flat_indices, num_experts + 1
  )
  group_sizes = jax.nn.one_hot(
      flat_indices, num_experts, dtype=jnp.int32
  ).sum(axis=0)
  return effective_indices, row_group, argsort_indices, group_sizes


class PallasMosaicTpuRoutingMoe(base.RoutingMoe[Config]):
  """Tokamax operator running Pallas Mosaic TPU `routing_moe` on one shard."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _fwd(
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
      *,
      global_num_experts: int | None = None,
      act_fn: str = "silu",
      skip_padded_tokens: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    del return_residuals
    if w1_bias is not None or w2_bias is not None:
      raise NotImplementedError("Bias is currently not supported.")

    local_num_experts = w1.shape[0]
    num_experts = (
        local_num_experts
        if global_num_experts is None
        else int(global_num_experts)
    )
    if num_experts < local_num_experts:
      raise ValueError(
          f"global_num_experts ({num_experts}) must be >="
          f" local_num_experts ({local_num_experts})."
      )
    if config is None:
      config = _default_config_for_tokens(x.shape[0])

    group_offset = (
        jnp.zeros((1,), dtype=jnp.int32)
        if experts_start is None
        else experts_start.reshape(1).astype(jnp.int32)
    )
    x_compact = input_pack.input_pack(x)
    effective_indices, row_group, argsort_indices, group_sizes = (
        _prepare_routing(
            topk_weights,
            topk_indices,
            num_experts=num_experts,
            skip_padded_tokens=skip_padded_tokens,
        )
    )
    topk = topk_indices.shape[-1]
    gmm_res = gmm.routing_gmm(
        x=x_compact,
        w=arg.SingleWeightWrapper.create(
            w1, w1_scale, None, fused_gate_up=True
        ),
        w2=arg.SingleWeightWrapper.create(w2, w2_scale, None),
        group_sizes=group_sizes,
        gather=arg.RoutingWrapper.create(argsort_indices // topk),
        scatter=arg.RoutingWrapper.create(argsort_indices),
        fuse_act=act_fn,
        group_offset=group_offset,
        row_group=row_group,
        custom_plan=_to_gmm_plan(config),
    )
    if experts_start is None and num_experts == local_num_experts:
      out = output_reduce.output_reduce_unpack(
          gmm_res, topk_weights, out_dtype=x.dtype
      )
    else:
      out = output_reduce.output_reduce_unpack(
          gmm_res,
          topk_weights,
          effective_indices,
          group_offset,
          local_num_experts,
          out_dtype=x.dtype,
      )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return _default_config_for_tokens(ba.arguments["x"].shape[0])

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return {self._get_heuristics_config(ba), Config(use_fp32_acc=True)}

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
