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

"""Base operator class for Gated Delta Net 2 (GDN-2)."""

import dataclasses
from typing import Any, TypeAlias, TypeVar
import jax
from jaxtyping import Array, Float
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.gdn2 import reference
from typing_extensions import override

_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
Output: TypeAlias = tuple[jax.Array, jax.Array | None]
Residuals: TypeAlias = Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class GatedDeltaNet2[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Gated Delta Net 2 (GDN-2).

  Evaluates the linear recurrent delta-rule state update with input-dependent
  erase and write gating.
  """

  supports_symbolic_shapes = False

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      b: jax.Array,
      w: jax.Array,
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del config
    out = reference.gdn2_reference(
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return out, None
