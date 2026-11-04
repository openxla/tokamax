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


def causal_mask(shape: tuple[int, int, int, int]) -> Bool[Array, "T t"]:
  """Returns a causal mask for the given shape."""
  out = jnp.tri(shape[-2], shape[-1], dtype=jnp.bool_)
  return jnp.expand_dims(out, axis=(0, 1))
