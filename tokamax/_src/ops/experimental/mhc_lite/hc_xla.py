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
"""Reference JAX-only (XLA) implementation for static Hyper-Connections (HC)."""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import common


def pre_hc(
    x: jax.Array,
    weights: common.HcWeights,
) -> tuple[jax.Array, common.MHCContext]:
  """Reference JAX-only implementation of the static HC pre-branch stage.

  Matches GUM 1.4 `HyperconnectionsLayers(mode=HCMode.PRE)`:
    - 1D mapping (`w_pre` shape `(k,)`): `einsum('...kd,k->...d', x, w_pre)`
    - 2D mapping (`w_pre` shape `(k, d)`): `einsum('...kd,kd->...d', x, w_pre)`

  Args:
    x: Input streams of shape `(batch, sequence, streams, embedding)`.
    weights: Static Hyper-Connections weight container (`HcWeights`).

  Returns:
    A tuple `(layer_input, context)` for the wrapped branch and `post_hc`.

  Raises:
    common.UnsupportedInputError: If `x` is not rank 4 or `weights.w_pre` is not
      rank 1 or 2.
  """
  if x.ndim != 4:
    raise common.UnsupportedInputError(
        "Expected x to have shape (batch, sequence, streams, embedding); got"
        f" {x.shape}."
    )
  w_pre = weights.w_pre.astype(x.dtype)
  if w_pre.ndim == 2:
    layer_input = jnp.einsum("...kd,kd->...d", x, w_pre)
  elif w_pre.ndim == 1:
    layer_input = jnp.einsum("...kd,k->...d", x, w_pre)
  else:
    raise common.UnsupportedInputError(
        f"Expected w_pre to have rank 1 or 2; got shape {weights.w_pre.shape}."
    )
  return layer_input, common.MHCContext(
      x=x,
      h_post=weights.w_post,
      residual=weights.w_res,
      implementation="hc",
  )


def post_hc(
    layer_output: jax.Array,
    context: common.MHCContext,
) -> jax.Array:
  """Reference JAX-only implementation of the static HC post-branch stage.

  Matches GUM 1.4 `HyperconnectionsLayers` (`POST` + `RESIDUAL`) and
  `ResidualBlock` float32 residual accumulation:
    - `res_out = einsum('...kd,mk->...md', x, w_res)`
    - `post_out = expand_dims(layer_output, axis=-2) * w_post`
    - `output = (float32(res_out) + float32(post_out)).astype(x.dtype)`

  Args:
    layer_output: Output from the wrapped branch of shape `(batch, sequence,
      embedding)`.
    context: Opaque `MHCContext` returned by `pre_hc`.

  Returns:
    Mixed output streams of shape `(batch, sequence, streams, embedding)`.
  """
  x = context.x
  w_res = context.residual.astype(x.dtype)
  res_out = jnp.einsum("...kd,mk->...md", x, w_res)

  w_post = context.h_post.astype(x.dtype)
  if w_post.ndim == 1:
    w_post = w_post[:, None]
  post_out = jnp.expand_dims(layer_output.astype(x.dtype), axis=-2) * w_post
  return (res_out.astype(jnp.float32) + post_out.astype(jnp.float32)).astype(
      x.dtype
  )
