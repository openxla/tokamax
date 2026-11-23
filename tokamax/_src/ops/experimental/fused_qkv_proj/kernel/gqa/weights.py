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
"""fused_qkv's weight preparation. Load time: call once per layer, not per step.

Only the input norm weight needs a new layout; fused_qkv takes every other
weight as loaded.
"""

import jax
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import norm


def prepare_input_norm_w(
    input_norm_w: jax.Array | None,
    *,
    fp8: bool,
    input_norm: str,
    act_quant_block: int,
) -> jax.Array | None:
  """Returns input_norm_w in fused_qkv's layout.

  Args:
    input_norm_w: (1, K) as loaded, or None without an input norm.
    fp8: whether the projection weight is fp8.
    input_norm: the decoder input norm, as passed to fused_qkv.
    act_quant_block: as passed to fused_qkv.

  Returns:
    fp8 with an input norm: (K / block, 1, block) u32, each bf16 weight in
    both halves (see norm.packed_bf16_weight). Otherwise input_norm_w as is.
  """
  kind = model_config.Norm(input_norm)
  if not fp8 or kind == model_config.Norm.NONE:
    return input_norm_w
  assert input_norm_w is not None
  return norm.packed_bf16_weight(kind, input_norm_w, act_quant_block)
