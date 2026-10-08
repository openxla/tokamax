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
"""Pure JAX reference of the DeepSeek-V4 fused reverse-RoPE `wo_a` projection.

DeepSeek-V4's attention output `x` is `[T, G * H, head_dim]`: `G` groups of
`H = 8` heads. The output projection first undoes the RoPE on the trailing
`rotary_dim` channels of each head (`rope.reference.rope` with `inverse=True`),
then projects each group's `D = H * head_dim` features with its own
`[D, R]` block of the fp8 `wo_a`, scaled per output column by `wo_a_scale`.
"""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.rope import reference as rope_reference

FP8_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)


def wo_a_projection(
    activations: jax.Array,
    wo_a: jax.Array,
    wo_a_scale: jax.Array,
    *,
    num_groups: int,
    quantize_activations: bool = False,
) -> jax.Array:
  """`DeepseekV4Attention._o_proj`'s wo_a einsum, up to (not incl.) `wo_b`.

  Args:
    activations: `[T, G * H, head_dim]` bf16 activations.
    wo_a: `[D, G * R]` fp8 weights.
    wo_a_scale: `[G * R]` float32 per-column weight scales.
    num_groups: The number of head groups `G`.
    quantize_activations: Whether to quantize each `[D]` activation row to fp8
      (one scale per token and group) before the einsum, as the Pallas kernel
      does. Upstream's reference only covers `False`.

  Returns:
    The `[T, G * R]` bf16 projection.
  """
  num_tokens = activations.shape[0]
  o_f = activations.reshape(num_tokens, num_groups, -1)  # [t, g, d]
  reduction = o_f.shape[-1]
  w = wo_a.reshape(reduction, num_groups, -1)
  s = wo_a_scale.reshape(num_groups, -1)
  if quantize_activations:
    # Per-row dynamic fp8 quantization, as in the kernel.
    amax = jnp.max(jnp.abs(o_f), axis=-1, keepdims=True)
    inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
        jnp.bfloat16
    )
    lhs = (o_f * inv).astype(jnp.float8_e4m3fn)
    z = jnp.einsum(
        "tgd,dgr->tgr", lhs, w, preferred_element_type=jnp.float32
    ) * (1.0 / inv.astype(jnp.float32))
    z = z * s.astype(jnp.bfloat16)[None, ...]
  else:
    z = (
        jnp.einsum(
            "tgd,dgr->tgr",
            o_f,
            w.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        * s.astype(jnp.bfloat16)[None, ...]
    )
  return z.astype(jnp.bfloat16).reshape(num_tokens, -1)


def o_projection(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,
    wo_a_scale: jax.Array,
    *,
    inverse: bool = True,
    quantize_activations: bool = False,
) -> jax.Array:
  """RoPE on `x` (`rope.reference.rope`), then `wo_a_projection`.

  Args:
    x: `[T, G * H, head_dim]` bf16 attention output.
    positions: `[T]` int RoPE position of each token.
    cos_sin_cache: `[max_position, rotary_dim]` packed `[cos | sin]` rows.
    wo_a: `[H * head_dim, G * R]` fp8 weights.
    wo_a_scale: `[G * R]` float32 per-column weight scales.
    inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
    quantize_activations: See `wo_a_projection`.

  Returns:
    The `[T, G * R]` bf16 projection.
  """
  head_dim = x.shape[-1]
  heads_per_group = wo_a.shape[0] // head_dim
  num_groups = x.shape[1] // heads_per_group
  roped = rope_reference.rope(x, positions, cos_sin_cache, inverse=inverse)
  return wo_a_projection(
      roped,
      wo_a,
      wo_a_scale,
      num_groups=num_groups,
      quantize_activations=quantize_activations,
  )
