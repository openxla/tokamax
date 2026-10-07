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
"""Compressor projection benchmark argument specifications.

The kernel scatters each token's state without bounds checks, so `slot_mapping`
is concrete: distinct, in-bounds multiples of the rows a token's state
occupies. Inputs are bf16, as in serving. (With f32 weights, upstream's
heuristic tiles for CSA with `hidden_size = 7168` need more than XLA's default
32 MiB of scoped VMEM.)
"""

import dataclasses
from typing import Final, Literal

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct

_HIDDEN_SIZE = 7168


class _HashableNPArray(np.ndarray):
  """Hashable numpy array for use as an ArgSpec argument."""

  def __new__(cls, input_array):
    return np.asarray(input_array).view(cls)

  def __hash__(self):
    return hash((self.tobytes(), self.shape, self.dtype))


@dataclasses.dataclass(frozen=True)
class _Geometry:
  """A compressor's state and cache geometry (see `test_base.Geometry`)."""

  state_width: int
  compress_ratio: int
  lanes: int
  page_size: int
  words: bool = False

  @property
  def rows_per_token(self) -> int:
    return 2 * self.state_width // self.lanes

  @property
  def tokens_per_page(self) -> int:
    return self.page_size // self.rows_per_token

  def cache(self, num_pages: int) -> jax.ShapeDtypeStruct:
    if self.words:
      return ShapeDtype((num_pages, self.page_size, self.lanes), jnp.int32)
    return ShapeDtype((num_pages, self.page_size, 4, self.lanes), jnp.uint8)


# DeepSeek-V4's three compressors, with the page sizes of the upstream tests.
_CSA = _Geometry(
    state_width=1024, compress_ratio=4, lanes=128, page_size=256, words=True
)
_HCA = _Geometry(state_width=512, compress_ratio=128, lanes=128, page_size=256)
_INDEXER = _Geometry(state_width=256, compress_ratio=4, lanes=256, page_size=32)


def _make_argspec(
    *,
    name: str,
    geometry: _Geometry,
    num_tokens: int,
    layout: Literal["decode", "prefill"],
    hidden_size: int = _HIDDEN_SIZE,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests", "forward_only"),
) -> arg_spec.ArgSpec:
  """Makes an argspec for the compressor projection.

  Args:
    name: The arg spec name.
    geometry: The compressor geometry.
    num_tokens: The number of tokens.
    layout: "decode" puts each token (one per request) in its own page, at
      varying offsets; "prefill" packs one request's tokens into consecutive
      slots from the first page.
    hidden_size: The hidden size.
    dtype: The dtype of the hidden states, weights and APE.
    tags: The arg spec tags.

  Returns:
    The arg spec.
  """
  tokens = np.arange(num_tokens)
  if layout == "decode":
    num_pages = num_tokens
    token_slots = (
        tokens * geometry.tokens_per_page
        + (5 * tokens) % geometry.tokens_per_page
    )
  else:
    num_pages = -(-num_tokens // geometry.tokens_per_page)
    token_slots = tokens
  slot_mapping = token_slots * geometry.rows_per_token
  state_dim = 2 * geometry.state_width
  return arg_spec.ArgSpec(
      args={
          "hidden_states": ShapeDtype((num_tokens, hidden_size), dtype),
          "wkv_wgate": ShapeDtype((hidden_size, state_dim), dtype),
          "ape": ShapeDtype(
              (geometry.compress_ratio, geometry.state_width), dtype
          ),
          "positions": numerics.RangedArrayInitializer(
              (num_tokens,), jnp.int32, 0, 1 << 16
          ),
          "slot_mapping": _HashableNPArray(slot_mapping.astype(np.int32)),
          "cache": geometry.cache(num_pages),
          "compress_ratio": geometry.compress_ratio,
      },
      project="inference",
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name=f"dsv4_csa_decode_t128_h{_HIDDEN_SIZE}",
        geometry=_CSA,
        num_tokens=128,
        layout="decode",
    ),
    _make_argspec(
        name=f"dsv4_csa_prefill_t4096_h{_HIDDEN_SIZE}",
        geometry=_CSA,
        num_tokens=4096,
        layout="prefill",
    ),
    _make_argspec(
        name=f"dsv4_hca_prefill_t4096_h{_HIDDEN_SIZE}",
        geometry=_HCA,
        num_tokens=4096,
        layout="prefill",
    ),
    _make_argspec(
        name=f"dsv4_indexer_prefill_t4096_h{_HIDDEN_SIZE}",
        geometry=_INDEXER,
        num_tokens=4096,
        layout="prefill",
    ),
)
