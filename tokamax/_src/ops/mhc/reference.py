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
"""Pure JAX reference for DeepSeek-V4 manifold-constrained hyper-connections.

See `api` for what mHC computes, the call order and the shapes.
"""

import math

import jax
from jax import lax
import jax.numpy as jnp


def hc_mult_from_mix_dim(hc_mult3: int) -> int:
  """Returns the stream count `M` given the mix dimension `M * (M + 2)`.

  The mix dimension is the row count of `fn` and the size of `hc_base`.

  Args:
    hc_mult3: The mix dimension.

  Raises:
    ValueError: If `hc_mult3` isn't `M * (M + 2)` for a positive `M`.
  """
  hc_mult = math.isqrt(hc_mult3 + 1) - 1  # M * (M + 2) == (M + 1)**2 - 1.
  if hc_mult < 1 or hc_mult * (hc_mult + 2) != hc_mult3:
    raise ValueError(
        f"fn must have M * (M + 2) rows for some M >= 1, got {hc_mult3}."
    )
  return hc_mult


def mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Gating / softmax / Sinkhorn on the tiny `(T, M3)` mix logits.

  Same math as `utils.mhc_pre_gates`, which the Pallas ops run in XLA after the
  kernel.

  Args:
    mixes: `(T, M3)` f32 raw mix logits, `x2d @ fn.T`.
    sqrsum: `(T, 1)` f32 sum of squares of each token's `M * H` stream values.
    hc_mult: Number of residual streams `M`.
    hidden_size: Hidden size `H` of one stream.
    hc_scale: `(3,)` f32 pre/post/comb logit scales.
    hc_base: `(M3,)` f32 logit biases, pre entries first.
    rms_eps: RMS-norm epsilon.
    hc_pre_eps: Additive floor on the sigmoid pre gates.
    hc_sinkhorn_eps: Sinkhorn epsilon.
    hc_post_mult_value: Scale of the sigmoid post gates.
    sinkhorn_repeat: Number of Sinkhorn normalization rounds.

  Returns:
    `(pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M))`, all f32.
  """
  num_tokens = mixes.shape[0]

  mixes = mixes * lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

  pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
  pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

  post_logits = (
      mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
      + hc_base[hc_mult : 2 * hc_mult]
  )
  post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

  comb_logits = mixes[:, 2 * hc_mult :].reshape(
      num_tokens, hc_mult, hc_mult
  ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
  comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
  comb_mix = comb_mix / (
      jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
  )
  for _ in range(sinkhorn_repeat - 1):
    comb_mix = comb_mix / (
        jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
    )
    comb_mix = comb_mix / (
        jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
    )
  return pre_mix, post_mix, comb_mix


def mhc_pre_mixes(x2d: jax.Array, fn: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Returns f32 `(mixes (T, M3), sqrsum (T, 1))` for `x2d (T, M * H)`."""
  x = x2d.astype(jnp.float32)
  mixes = lax.dot_general(
      x,
      fn,
      dimension_numbers=(((1,), (1,)), ((), ())),
      precision=lax.Precision.HIGHEST,
  )
  sqrsum = jnp.sum(x * x, axis=-1, keepdims=True)
  return mixes, sqrsum


def _collapse(
    pre_mix: jax.Array, x2d: jax.Array, hc_mult: int, hidden_size: int
) -> jax.Array:
  """Collapses the `M` streams of `x2d (T, M * H)` into `(T, H)` bf16."""
  out = pre_mix[:, 0:1] * x2d[:, :hidden_size].astype(jnp.float32)
  for i in range(1, hc_mult):
    out = out + (
        pre_mix[:, i : i + 1]
        * x2d[:, i * hidden_size : (i + 1) * hidden_size].astype(jnp.float32)
    )
  return out.astype(jnp.bfloat16)


def mhc_pre(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Reference `mhc_pre`, in pure JAX.

  Args:
    residual: `(T, M * H)` bf16 flat residual streams.
    fn: `(M3, M * H)` f32 gate projection.
    hc_scale: `(3,)` f32 pre/post/comb logit scales.
    hc_base: `(M3,)` f32 logit biases.
    rms_eps: RMS-norm epsilon.
    hc_pre_eps: Additive floor on the sigmoid pre gates.
    hc_sinkhorn_eps: Sinkhorn epsilon.
    hc_post_mult_value: Scale of the sigmoid post gates.
    sinkhorn_repeat: Number of Sinkhorn normalization rounds.

  Returns:
    `(post_mix (T, M) f32, comb_mix (T, M, M) f32, layer_input (T, H) bf16)`.
  """
  hc_mult = hc_mult_from_mix_dim(fn.shape[0])
  hidden_size = residual.shape[-1] // hc_mult

  mixes, sqrsum = mhc_pre_mixes(residual, fn)
  pre_mix, post_mix, comb_mix = mhc_pre_gates(
      mixes,
      sqrsum,
      hc_mult,
      hidden_size,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      sinkhorn_repeat,
  )
  layer_input = _collapse(pre_mix, residual, hc_mult, hidden_size)
  return post_mix, comb_mix, layer_input


def mhc_post(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
  """Reference `mhc_post`, in pure JAX.

  Args:
    x: `(T, H)` bf16 sublayer output.
    residual: `(T, M * H)` bf16 flat residual streams.
    post_layer_mix: `(T, M)` f32 post gates.
    comb_res_mix: `(T, M, M)` f32 stream-mixing matrix.

  Returns:
    The new `(T, M * H)` flat residual streams, in `residual.dtype`.
  """
  num_tokens, hidden_size = x.shape
  hc_mult = comb_res_mix.shape[-1]
  mixed_residual = jnp.einsum(
      "tij,tih->tjh",
      comb_res_mix.astype(jnp.float32),
      residual.reshape(num_tokens, hc_mult, hidden_size).astype(jnp.float32),
      precision=lax.Precision.HIGHEST,
  )
  post_term = (
      post_layer_mix.astype(jnp.float32)[:, :, None]
      * x.astype(jnp.float32)[:, None, :]
  )
  new_residual = (mixed_residual + post_term).astype(residual.dtype)
  return new_residual.reshape(num_tokens, hc_mult * hidden_size)


def mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Reference seam op: `mhc_post`, then `mhc_pre` on its output.

  Args:
    x: `(T, H)` bf16 sublayer output.
    residual: `(T, M * H)` bf16 flat residual streams.
    post_layer_mix: `(T, M)` f32 post gates.
    comb_res_mix: `(T, M, M)` f32 stream-mixing matrix.
    fn: `(M * (M + 2), M * H)` f32 mixing projection for the next `mhc_pre`.
    hc_scale: `(3,)` f32 gate scales.
    hc_base: `(M * (M + 2),)` f32 gate biases.
    rms_eps: RMS-norm epsilon.
    hc_pre_eps: Epsilon added to the pre (read) gates.
    hc_sinkhorn_eps: Sinkhorn normalization epsilon.
    hc_post_mult_value: Multiplier applied to the post gates.
    sinkhorn_repeat: Number of Sinkhorn iterations.

  Returns:
    `(residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur)`, where
    `residual_cur` is the `mhc_post` output and the other three are the
    `mhc_pre` outputs computed from it.
  """
  residual_cur = mhc_post(x, residual, post_layer_mix, comb_res_mix)
  post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre(
      residual_cur,
      fn,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      sinkhorn_repeat,
  )
  return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur
