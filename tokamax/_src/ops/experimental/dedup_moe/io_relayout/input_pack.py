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
"""Input row quantization, token-replica ownership, and the shared expert FFN.

Row quantization (`rowquant_fp8`) and token-replica ownership
(`TokenReplicaLayout`) are shared with `fused_moe`; the clamped activations
come from `gmm_v2`. The replicated shared expert FFN is this layer's own.
"""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel
from tokamax._src.ops.experimental.fused_moe import token_parallel
from tokamax._src.ops.gmm_v2 import gmm_v2

rowquant_fp8 = fused_kernel.rowquant_fp8
TokenReplicaLayout = token_parallel.TokenReplicaLayout
silu_and_mul_with_clamp = gmm_v2.silu_and_mul_with_clamp
swigluoai = gmm_v2.swigluoai

# Activations the shared expert FFN supports.
SHARED_EXPERT_ACT_FNS = ("silu", "gelu", "swigluoai")


def shared_expert_ffn(
    x_l: jax.Array, ws1: jax.Array, ws2: jax.Array, *, act_fn: str = "silu"
) -> jax.Array:
  """Replicated shared expert FFN on the local tokens.

  ws1 [D, 2 * Fs] holds the concatenated [gate, up] projection weights
  (matching the routed experts' w1 [E, D, 2 * F] layout); ws2 [Fs, D] is the
  down projection. "silu" applies the clamped SiLU-and-multiply form.

  The result stays f32 on purpose: the caller adds it to the routed
  combine's f32 sum and rounds the total to the output dtype once, as the
  f32 reference does.

  Args:
    x_l: [t, D] local tokens.
    ws1: [D, 2 * Fs] gate and up projection weights.
    ws2: [Fs, D] down projection weights.
    act_fn: One of SHARED_EXPERT_ACT_FNS.

  Returns:
    [t, D // 128, 128] f32 shared expert output.

  Raises:
    NotImplementedError: If `act_fn` is not supported.
  """
  t_local, hidden = x_l.shape
  fs = ws2.shape[0]
  gate_up = jnp.dot(
      x_l.astype(jnp.bfloat16),
      ws1.astype(jnp.bfloat16),
      preferred_element_type=jnp.float32,
  )
  gate, up = gate_up[:, :fs], gate_up[:, fs:]
  if act_fn == "silu":
    mid = silu_and_mul_with_clamp(gate, up)
  elif act_fn == "gelu":
    mid = jax.nn.gelu(gate) * up
  elif act_fn == "swigluoai":
    mid = swigluoai(gate, up)
  else:
    raise NotImplementedError(
        f"shared expert act_fn must be one of {SHARED_EXPERT_ACT_FNS}; got"
        f" {act_fn!r}"
    )
  out = jnp.dot(
      mid.astype(jnp.bfloat16),
      ws2.astype(jnp.bfloat16),
      preferred_element_type=jnp.float32,
  )
  lanes = fused_host.HIDDEN_LANE_BLOCK
  return out.reshape(t_local, hidden // lanes, lanes)
