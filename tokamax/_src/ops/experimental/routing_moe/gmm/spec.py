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

"""Validated problem specification of the routing GMM kernel."""

import dataclasses

from tokamax._src.ops.experimental.routing_moe.common import dataclass_util
from tokamax._src.ops.experimental.routing_moe.common import spec as tensor_spec


def _validate_weights(
    w: tensor_spec.WeightSpec,
    w2: tensor_spec.WeightSpec | None,
    fuse_act: str | None,
) -> None:
  """Validates weight specs and optional gated activation compatibility."""
  if fuse_act is not None and not w.fused_gate_up:
    raise ValueError("fuse_act requires a weight with fused_gate_up=True.")
  if w.has_bias or (w2 is not None and w2.has_bias):
    raise NotImplementedError("Bias is not supported yet.")

  # The fused activation combines a gate column and an up column into one, so
  # the output halves only when fuse_act is applied. fused_gate_up alone does
  # not change the width.
  size_out_n = w.out_size
  if fuse_act is not None:
    size_out_n //= 2
  if w2 is None:
    return

  if fuse_act is None:
    raise ValueError("w2 requires fuse_act to be set.")
  if size_out_n != w2.in_size:
    raise ValueError(f"w {size_out_n=} and {w2.in_size=} do not match.")
  if w.group_size != w2.group_size:
    raise ValueError(f"{w.group_size=} and {w2.group_size=} do not match.")
  if w.in_size != w2.out_size:
    raise ValueError(f"{w.in_size=} and {w2.out_size=} do not match.")
  if w2.fused_gate_up:
    raise ValueError("w2 cannot have fused_gate_up=True.")


def _validate_routing(
    size_m: int,
    gather: tensor_spec.RoutingSpec | None,
    scatter: tensor_spec.RoutingSpec | None,
    size_ep: int,
) -> None:
  """Validates gather, scatter, and expert-parallel shard count."""
  size_out_m = size_m
  if gather is not None:
    if gather.remote_axis is not None:
      raise NotImplementedError("Remote gather is not supported.")
    size_out_m = gather.num_indices
  if scatter is not None and scatter.num_indices != size_out_m:
    raise ValueError(
        f"scatter must map every computed row ({size_out_m}). Got"
        f" num_indices {scatter.num_indices}."
    )
  if size_ep < 1:
    raise ValueError(f"size_ep must be positive. Got {size_ep}.")
  if (scatter is None or scatter.remote_axis is None) and size_ep != 1:
    raise ValueError(f"size_ep ({size_ep}) requires a remote scatter.")
  if size_out_m % size_ep != 0:
    raise ValueError(
        f"Output rows ({size_out_m}) must divide evenly across"
        f" {size_ep} devices."
    )


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingGmmSpec:
  """Validated input specifications and operator semantics of routing GMM."""

  x: tensor_spec.ActivationSpec
  w: tensor_spec.WeightSpec
  w2: tensor_spec.WeightSpec | None
  gather: tensor_spec.RoutingSpec | None
  scatter: tensor_spec.RoutingSpec | None
  num_global_groups: int
  fuse_act: str | None
  # Devices that output rows are split across by a remote scatter.
  size_ep: int

  def __post_init__(self) -> None:
    if self.x.col_size != self.w.in_size:
      raise ValueError(
          f"x col_size ({self.x.col_size}) must match w in_size"
          f" ({self.w.in_size})."
      )
    if self.num_global_groups < self.w.group_size:
      raise ValueError(
          f"num_global_groups ({self.num_global_groups}) must be at least"
          f" w.group_size ({self.w.group_size})."
      )
    _validate_weights(self.w, self.w2, self.fuse_act)
    _validate_routing(self.x.row_size, self.gather, self.scatter, self.size_ep)

  @property
  def is_fused_act(self) -> bool:
    return self.fuse_act is not None

  @property
  def size_k(self) -> int:
    return self.x.col_size

  @property
  def size_k2(self) -> int | None:
    return None if self.w2 is None else self.w2.in_size

  @property
  def size_out_m(self) -> int:
    return self.x.row_size if self.gather is None else self.gather.num_indices

  @property
  def size_out_n(self) -> int:
    if self.w2 is not None:
      return self.w2.out_size
    if self.fuse_act is not None:
      return self.w.out_size // 2
    return self.w.out_size

  @property
  def size_local_groups(self) -> int:
    return self.w.group_size
