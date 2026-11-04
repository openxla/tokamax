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
"""Example masks for FlexAttention."""

import jax.numpy as jnp
from jaxtyping import Array, Bool  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.flex_attention import api


def causal_mask(shape: tuple[int, int, int, int]) -> Bool[Array, "T t"]:
  """Returns a causal mask for the given shape."""
  out = jnp.tri(shape[-2], shape[-1], dtype=jnp.bool_)
  return jnp.expand_dims(out, axis=(0, 1))


def generate_local_window_mask(
    local_window_size: int | tuple[int, int],
) -> api.MaskMod:
  """Returns a local window mask modification function.

  Replicates the `local_window_size` parameter from
  `jax.nn.dot_product_attention`.

  Args:
    local_window_size: Window sizes to make self attention attend to each
      token's local window. If set, this specifies (left_window_size,
      right_window_size) for each token. If a single int is given, it is
      interpreted as a symmetric window (window_size, window_size).
  """
  if isinstance(local_window_size, int):
    left_window, right_window = local_window_size, local_window_size
  else:
    left_window, right_window = local_window_size

  def local_window_mask_fn(
      shape: tuple[int, int, int, int],
  ) -> Bool[Array, "T t"]:
    q_len = shape[-2]
    kv_len = shape[-1]
    query_pos = jnp.arange(q_len)[:, None]
    key_pos = jnp.arange(kv_len)[None, :]
    left_mask = query_pos <= key_pos + left_window
    right_mask = query_pos >= key_pos - right_window
    out = jnp.logical_and(left_mask, right_mask)
    return jnp.expand_dims(out, axis=(0, 1))

  return local_window_mask_fn
