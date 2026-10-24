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

"""Base operator class for Raven Gated Slot Attention (GSA)."""

import dataclasses
from typing import Any, TypeAlias, TypeVar
import jax
from tokamax._src.ops import op
from tokamax._src.ops.experimental.raven import reference
from typing_extensions import override

_Config = TypeVar("_Config")
Output: TypeAlias = tuple[jax.Array, jax.Array | None]
Residuals: TypeAlias = Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class RavenGSA[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Raven Gated Slot Attention (GSA).

  Decouples the two-stage intra-chunk decay across output and slot axes,
  maintaining compact memory-bounded slot state updates.
  """

  supports_symbolic_shapes = False

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      *,
      chunk_size: int = 64,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del config
    out = reference.raven_stage1_reference(
        q,
        k,
        s,
        g,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return out, None
