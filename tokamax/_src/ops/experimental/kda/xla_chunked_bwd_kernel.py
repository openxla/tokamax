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
"""Pure-JAX/XLA backward for chunked Kimi Delta Attention."""

from __future__ import annotations

import jax

from tokamax._src.ops.experimental.kda.xla_chunked_fwd_kernel import (
    chunk_kda_fwd,
)


def chunk_kda_bwd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    gate: jax.Array,
    beta: jax.Array,
    *,
    a_log: jax.Array | None,
    delta_time_bias: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    output_final_state: bool,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    segment_ids: jax.Array | None,
    lower_bound: float | None,
    max_num_segments: int | None,
    chunk_size: int,
    safe_gate: bool,
    output_cotangent: tuple[jax.Array, jax.Array | None],
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
  """Differentiates the pure-JAX chunk pipeline with a separately lowered VJP."""

  def primal(q, k, v, g, b, decay, bias, state):
    return chunk_kda_fwd(
        q,
        k,
        v,
        g,
        b,
        a_log=decay,
        delta_time_bias=bias,
        scale=scale,
        initial_state=state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        lower_bound=lower_bound,
        max_num_segments=max_num_segments,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
    )

  _, pullback = jax.vjp(
      primal,
      query,
      key,
      value,
      gate,
      beta,
      a_log,
      delta_time_bias,
      initial_state,
  )
  return pullback(output_cotangent)


__all__ = ["chunk_kda_bwd"]
