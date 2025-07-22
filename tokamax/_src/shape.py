# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Shape utilities."""

import functools

from einshape.src.jax import jax_ops as einshape_jax
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


einshape = lambda eq, **kw: functools.partial(einshape_jax.einshape, eq, **kw)


def pad_dim_to(x: jax.Array, n: int, axis: int) -> jax.Array:
  """Pads `x` to size `n` along `axis`."""
  if (padding := n - x.shape[axis]) == 0:
    return x
  if padding < 0:
    raise ValueError(f"Cannot pad {x.shape[axis]} to smaller size {n}")
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, padding)
  return jnp.pad(x, pad_width)


def pad_to_next_multiple_of(x: jax.Array, m: int, axis: int = 0) -> jax.Array:
  """Pads `x` to the next multiple of `m` along `axis`."""
  return pad_dim_to(x, pl.cdiv(x.shape[axis], m) * m, axis)
