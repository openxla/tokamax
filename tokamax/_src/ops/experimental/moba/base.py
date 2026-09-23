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

"""Base operator class for Mixture of Block Attention (MoBA)."""

import dataclasses
from typing import Any, TypeAlias, TypeVar
import jax
from tokamax._src.ops import op
from tokamax._src.ops.experimental.moba import reference
from typing_extensions import override

_Config = TypeVar("_Config")
Output: TypeAlias = tuple[jax.Array, None]
Residuals: TypeAlias = Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class MixtureOfBlockAttention[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Mixture of Block Attention (MoBA).

  Routes each query block to the top-k most relevant historical key/value
  blocks via block-mean routing logits.
  """

  supports_symbolic_shapes = False

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      *,
      topk: int = 4,
      chunk_size: int = 256,
      scale: float | None = None,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del config
    out, _ = reference.moba_reference(
        q,
        k,
        v,
        topk=topk,
        chunk_size=chunk_size,
        scale=scale,
    )
    return (out, None), None
