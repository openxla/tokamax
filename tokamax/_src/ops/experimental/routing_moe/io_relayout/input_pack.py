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
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec as spec_lib


@jax.jit(static_argnames=("out_format",))
def input_pack(
    x: jax.Array,
    out_format: out_format_lib.ActivationFormatLike | None = None,
) -> arg.ActivationWrapper[jax.Array]:
  """Placeholder for input_pack kernel from large to compact layout."""
  # TODO: Replace with custom Pallas input_pack kernel.
  resolved_format = out_format_lib.ResolvedActivationFormat.create(
      out_format,
      default_dtype=x.dtype,
  )
  if resolved_format.dtype != x.dtype:
    raise NotImplementedError(
        "input_pack does not support dtype conversion yet."
    )
  if resolved_format.quant is not None:
    raise NotImplementedError("input_pack does not support quantization yet.")
  row_size, col_size = x.shape
  spec = spec_lib.ActivationSpec(
      row_size=row_size, col_size=col_size, dtype=x.dtype, quant=None
  )
  pad_cols = (-col_size) % (layout_util.get_packing(x.dtype) * 128)
  if pad_cols:
    x = jnp.pad(x, ((0, 0), (0, pad_cols)))
  return arg.ActivationWrapper(
      data=x.reshape(spec.physical_compact_shape), scale=None, spec=spec
  )
