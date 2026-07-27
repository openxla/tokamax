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
"""Shared data contracts for the Pallas TPU KDA implementation."""

import dataclasses
from typing import TypeAlias

import jax
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member


CpMetadata: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
] | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class KdaResiduals:
  """Prepared forward values consumed by the custom backward."""

  q: Float[Array, "H B T_ALIGNED K"]
  k: Float[Array, "H B T_ALIGNED K"]
  v: Float[Array, "H B T_ALIGNED V"]
  beta: Float[Array, "H B T_ALIGNED"]
  g_cumsum: Float[Array, "H B T_ALIGNED K"] | None
  aqk: Float[Array, "H B T_ALIGNED BT"]
  akk: Float[Array, "H B T_ALIGNED BT"]
  initial_state: (
      Float[Array, "B H K V"] | Float[Array, "B N H K V"] | None
  )
  g_org: Float[Array, "H B T_ALIGNED K"] | None
  a_log: Float[Array, "H"] | None
  dt_bias: Float[Array, "H*K"] | None
  h: Float[Array, "H B NT K V"] | None
  g_dtype_marker: Float[Array, ""]
  q_rstd: Float[Array, "H B T_ALIGNED"] | None
  k_rstd: Float[Array, "H B T_ALIGNED"] | None
  cu_seqlens: Int[Array, "B N_CU"] | Int[Array, "N_CU"] | None
  aligned_cu_seqlens: Int[Array, "B N_CU"] | Int[Array, "N_CU"] | None
  chunk_indices: Int[Array, "B NT 2"] | Int[Array, "NT 2"] | None
  aligned_segment_ids: Int[Array, "B T_ALIGNED"] | None
  segment_ids: Int[Array, "B T_ORIG"] | None
  cp_metadata: CpMetadata
