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

"""Input relayout helper from large to compact layout."""

import jax
from tokamax._src.ops.experimental.routing_moe.common import arg


@jax.jit
def input_pack(x: jax.Array) -> arg.ActivationWrapper[jax.Array]:
  """Placeholder for input_pack kernel from large to compact layout."""
  # TODO: Replace with custom Pallas input_pack kernel.
  assert x.ndim == 2
  row_size, col_size = x.shape
  spec = arg.ActivationSpec(
      row_size=row_size,
      col_size=col_size,
      dtype=x.dtype,
  )
  return arg.ActivationWrapper(
      data=x.reshape(spec.compact_shape),
      scale=None,
      spec=spec,
  )
