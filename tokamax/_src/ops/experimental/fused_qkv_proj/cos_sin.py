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
"""Per-token RoPE cos/sin rows (vLLM's cos_sin_cache), for tests and benchmarks.

The kernels take these tables as inputs; how a model builds them doesn't
affect the kernels.
"""

import math

import jax.numpy as jnp
from jaxtyping import Array, BFloat16, Float32, Int  # pylint: disable=g-multiple-import,g-importing-member

CosSin = tuple[BFloat16[Array, "T n_pairs"], BFloat16[Array, "T n_pairs"]]


def _cos_sin(
    positions: Int[Array, "*batch"],
    inv_freq: Float32[Array, "n_pairs"],
    mscale: float = 1.0,
) -> tuple[
    BFloat16[Array, "*batch n_pairs"], BFloat16[Array, "*batch n_pairs"]
]:
  freqs = positions.astype(jnp.float32)[..., None] * inv_freq
  return (
      (jnp.cos(freqs) * mscale).astype(jnp.bfloat16),
      (jnp.sin(freqs) * mscale).astype(jnp.bfloat16),
  )


def _inv_freq(
    theta: float, n_pairs: int, denom: int
) -> Float32[Array, "n_pairs"]:
  return 1.0 / theta ** (
      jnp.arange(0, 2 * n_pairs, 2, dtype=jnp.float32) / denom
  )


def cos_sin_default(
    positions: Int[Array, "T"], rotary_dim: int, theta: float
) -> CosSin:
  """vLLM RotaryEmbedding."""
  return _cos_sin(positions, _inv_freq(theta, rotary_dim // 2, rotary_dim))


def cos_sin_mrope_interleaved(
    positions: Int[Array, "3 T"],
    rotary_dim: int,
    theta: float,
    section: tuple[int, int, int],
) -> CosSin:
  """vLLM MRotaryEmbedding (interleaved): t/h/w positions per token."""
  cos, sin = _cos_sin(positions, _inv_freq(theta, rotary_dim // 2, rotary_dim))
  c = jnp.arange(rotary_dim // 2)
  is_h = (c % 3 == 1) & (c < section[1] * 3)
  is_w = (c % 3 == 2) & (c < section[2] * 3)
  pick = lambda t: jnp.where(is_w, t[2], jnp.where(is_h, t[1], t[0]))
  return pick(cos), pick(sin)


def cos_sin_proportional(
    positions: Int[Array, "T"],
    head_dim: int,
    partial_rotary_factor: float,
    theta: float,
) -> CosSin:
  """vLLM Gemma4RotaryEmbedding, only the non-identity angles.

  The rest have cos 1 and sin 0; Rope(n_pairs, head_dim // 2) leaves those
  lanes unchanged.

  Args:
    positions: (T,) token positions.
    head_dim: D.
    partial_rotary_factor: fraction of each head that rotates.
    theta: the RoPE base.

  Returns:
    (cos, sin), each (T, n_pairs) bf16.
  """
  n_pairs = int(head_dim * partial_rotary_factor) // 2
  return _cos_sin(positions, _inv_freq(theta, n_pairs, head_dim))


def cos_sin_yarn(
    positions: Int[Array, "T"],
    rotary_dim: int,
    theta: float,
    factor: float,
    original_max_position: int,
    beta_fast: int = 32,
    beta_slow: int = 1,
    truncate: bool = True,
) -> CosSin:
  """vLLM YaRNScalingRotaryEmbedding, with mscale folded into cos/sin."""

  def correction_dim(rot):
    return (
        rotary_dim
        * math.log(original_max_position / (rot * 2 * math.pi))
        / (2 * math.log(theta))
    )

  low, high = correction_dim(beta_fast), correction_dim(beta_slow)
  if truncate:
    low, high = math.floor(low), math.ceil(high)
  low, high = max(low, 0), min(high, rotary_dim - 1)
  if low == high:
    high += 0.001
  pos_freqs = theta ** (
      jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim
  )
  ramp = jnp.clip(
      (jnp.arange(rotary_dim // 2, dtype=jnp.float32) - low) / (high - low),
      0,
      1,
  )
  mask = 1 - ramp
  inv_freq = (1.0 / (factor * pos_freqs)) * (1 - mask) + (
      1.0 / pos_freqs
  ) * mask
  mscale = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
  return _cos_sin(positions, inv_freq, mscale)
