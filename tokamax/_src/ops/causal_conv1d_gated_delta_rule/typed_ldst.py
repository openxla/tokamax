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
"""Typed views of raw byte blocks inside Pallas TPU kernels.

A block ref holds bytes in the layout of its owning buffer (e.g. one
kernel block of the unified KV pool). Consumers that pack a different
element type into those bytes access them through a typed view: Mosaic
``ref.bitcast`` (which rescales the second-minor dim by the element-size
ratio) plus an optional narrowing reshape, both applied to the *ref*
before the load. Reshaping the ref keeps every access one whole vector
register; splitting lanes after the load costs a lane-crossing relayout
instead (measured on v7: 366 bundles / 448 vrot for the post-load split
against 87 / 0 for the ref-level reshape, on a 256 KiB pool block).
The pool gather/scatter kernels and the GDN V3 state seam share these
helpers so the bytes they exchange are identical.
"""

import jax
import jax.numpy as jnp


def _typed_view(block_ref, view_dtype, lane_split: int):
  """Raw block ref as a ``(rows, lanes // lane_split)`` view_dtype ref."""
  if jnp.dtype(block_ref.dtype) != jnp.dtype(view_dtype):
    block_ref = block_ref.bitcast(jnp.dtype(view_dtype))
  return block_ref.reshape(-1, block_ref.shape[-1] // lane_split)


def load_typed(block_ref, *, view_dtype, lane_split: int = 1) -> jax.Array:
  """Reads a raw block ref as ``(rows, lanes // lane_split)`` view_dtype."""
  return _typed_view(block_ref, view_dtype, lane_split)[...]


def store_typed(block_ref, values: jax.Array, *, lane_split: int = 1) -> None:
  """Inverse of ``load_typed``: stores ``values`` through a typed view of the raw block ref, which is fully written."""
  ref = _typed_view(block_ref, values.dtype, lane_split)
  ref[...] = values.reshape(ref.shape)
