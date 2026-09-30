# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Base operator class for Native Sparse Attention (NSA)."""

import dataclasses
from typing import Any, TypeAlias, TypeVar
import jax
from tokamax._src.ops import op
from tokamax._src.ops.experimental.nsa import reference
from typing_extensions import override

_Config = TypeVar("_Config")
Output: TypeAlias = tuple[jax.Array, None]
Residuals: TypeAlias = Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class NativeSparseAttention[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Native Sparse Attention (NSA).

  Fuses coarse compression, fine-grained top-k block selection, and sliding
  window attention branches via a learned 3-way gate.
  """

  supports_symbolic_shapes = False

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g_cmp: jax.Array | None = None,
      g_slc: jax.Array | None = None,
      g_swa: jax.Array | None = None,
      *,
      chunk_size: int = 256,
      topk: int = 4,
      window: int = 128,
      cmp_block_size: int = 64,
      scale: float | None = None,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del config
    out, _ = reference.nsa_reference(
        q,
        k,
        v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
    )
    return (out, None), None
