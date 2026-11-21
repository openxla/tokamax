# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Shared data contracts for the Pallas/Mosaic TPU KDA implementation."""

import dataclasses
from typing import TypeAlias

import jax
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member


CpMetadata: TypeAlias = (
    tuple[
        jax.Array | bool | None,
        jax.Array | bool | None,
        jax.Array | int | None,
        jax.Array | int | None,
    ]
    | None
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class KdaResiduals:
  """Prepared forward values consumed by the custom backward."""

  q: Float[Array, "H B T_ALIGNED K"]
  k: Float[Array, "H B T_ALIGNED K"]
  v: Float[Array, "H B T_ALIGNED V"]
  beta: Float[Array, "H B T_ALIGNED"]
  g_cumsum: Float[Array, "H B T_ALIGNED GW"] | None
  aqk: Float[Array, "H B T_ALIGNED BT"]
  akk: Float[Array, "H B T_ALIGNED BT"] | None
  initial_state: (
      Float[Array, "B H K V"] | Float[Array, "B N H K V"] | None
  )
  g_org: Float[Array, "H B T_ALIGNED GW"] | None
  a_log: Float[Array, "H"] | None
  delta_time_bias: Float[Array, "H*GW"] | None
  h: Float[Array, "H B NT K V"] | None
  g_dtype_marker: Float[Array, ""]
  q_rstd: Float[Array, "H B T_ALIGNED"] | None
  k_rstd: Float[Array, "H B T_ALIGNED"] | None
  cu_seqlens: Int[Array, "B N_CU"] | Int[Array, "N_CU"] | None
  aligned_cu_seqlens: Int[Array, "B N_CU"] | Int[Array, "N_CU"] | None
  chunk_indices: Int[Array, "B NT 2"] | Int[Array, "NT 2"] | None
  aligned_segment_ids: Int[Array, "B T_ALIGNED"] | None
  segment_ids: Int[Array, "B T_SEG"] | None
  cp_metadata: CpMetadata
  v_new: Float[Array, "H B T_ALIGNED V"] | None = None
  q_raw: Float[Array, "H B T_RAW K"] | None = None
  k_raw: Float[Array, "H B T_RAW K"] | None = None
  v_raw: Float[Array, "H B T_RAW V"] | None = None
  conv_weight_q: Float[Array, "H_Q W K"] | None = None
  conv_weight_k: Float[Array, "H_K W K"] | None = None
  conv_weight_v: Float[Array, "H W V"] | None = None
  conv_bias_q: Float[Array, "H_Q K"] | None = None
  conv_bias_k: Float[Array, "H_K K"] | None = None
  conv_bias_v: Float[Array, "H V"] | None = None
  cp_seg_halo: tuple[Int[Array, "B"], Int[Array, "B T_HALO"] | None] | None = (
      None
  )
  chunk_size: int = dataclasses.field(default=64, metadata=dict(static=True))
  two_pass_bf16: bool = dataclasses.field(
      default=False, metadata=dict(static=True)
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class GdnResiduals:
  """Prepared forward values consumed by the GDN custom backward."""

  qkv_conv: Float[Array, "B T C_QKV"] | None
  q: Float[Array, "B T H_QK K"] | None
  k: Float[Array, "B T H_QK K"] | None
  v: Float[Array, "B T H_V V"] | None
  beta: Float[Array, "H_V B T"]
  g_cumsum: Float[Array, "H_V B T 1"] | None
  akk: Float[Array, "H_V B T BT"] | None
  h: Float[Array, "H_V B NT K V"] | None
  initial_state: Float[Array, "B H_V K V"] | None
  g_org: Float[Array, "H_V B T 1"] | None
  a_log: Float[Array, "H_V"] | None
  delta_time_bias: Float[Array, "H_V"] | None
  qkv_raw: Float[Array, "B T_RAW C_QKV"] | None
  conv_weight: Float[Array, "W C_QKV"] | None
  conv_bias: Float[Array, "C_QKV"] | None
  segment_ids: Int[Array, "B T"] | None
  g_dtype_marker: Float[Array, ""]
  cp_seg_halo: tuple[Int[Array, "B"], Int[Array, "B T_HALO"] | None] | None = (
      None
  )
  chunk_size: int = dataclasses.field(default=64, metadata=dict(static=True))
  beta_dtype: str = dataclasses.field(
      default="float32", metadata=dict(static=True)
  )
  two_pass_bf16: bool = dataclasses.field(
      default=False, metadata=dict(static=True)
  )
