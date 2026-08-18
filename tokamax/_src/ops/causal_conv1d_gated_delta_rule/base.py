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
"""Causal Conv1D Gated Delta Rule tokamax Op API using reference impl."""

import dataclasses
from typing import Any, Optional

import jax
from tokamax._src.ops import op
from tokamax._src.ops.causal_conv1d_gated_delta_rule import reference
from typing_extensions import override


@dataclasses.dataclass(frozen=True, kw_only=True)
class CausalConv1dGatedDeltaRule[C](
    op.Op[
        Any,
        tuple[tuple[jax.Array, jax.Array], jax.Array],
        None,
        C,
        None,
    ]
):
  """Causal Conv1D Gated Delta Rule Tokamax Op using reference implementation."""

  def _fwd(
      self,
      qkv: jax.Array,
      b: jax.Array,
      a: jax.Array,
      conv_state: jax.Array,
      recurrent_state: jax.Array,
      conv_weight: jax.Array,
      conv_bias: Optional[jax.Array],
      a_log: jax.Array,
      dt_bias: jax.Array,
      query_start_loc: jax.Array,
      state_indices: jax.Array,
      distribution: jax.Array,
      seq_lens: jax.Array,
      *,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
      config: C = reference.GdnAttentionConfig(),
      return_residuals: bool = False,
  ) -> tuple[tuple[tuple[jax.Array, jax.Array], jax.Array], None]:
    del return_residuals
    output = reference.run_jax_gdn_attention_local_ref(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        config=config,  # pyrefly: ignore[bad-argument-type]
    )
    return output, None
