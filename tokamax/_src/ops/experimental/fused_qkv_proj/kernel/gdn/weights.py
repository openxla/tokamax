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
"""fused_gdn's weight preparation. Load time: call once per layer, not per step.

in_proj_qkvz and its scale are used as loaded; in_proj_ba and the input norm
weight get new layouts.
"""

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-importing-member
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import norm


def prepare_weights(
    in_proj_ba: Array,
    input_norm_w: Array | None,
    *,
    num_v_heads: int,
    input_norm: str = "rms_one_plus_w",
    norm_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    act_quant_block: int = 1024,
) -> tuple[Array, Array | None]:
  """Returns (ba_w, norm_w), fused_gdn's layouts of the loaded weights.

  Args:
    in_proj_ba: (K, 2 Hv) bf16, [b | a], as loaded.
    input_norm_w: (1, K) as loaded; unused without an input norm.
    num_v_heads: Hv, as passed to fused_gdn.
    input_norm: as passed to fused_gdn.
    norm_dtype: as passed to fused_gdn.
    act_quant_block: as passed to fused_gdn.

  Returns:
    ba_w: (K, 128) bf16, b in lanes 0:Hv and a in lanes 64:64 + Hv, the rest
      zeros, so the ba dot is a single MXU column block.
    norm_w: the input norm weight pre-chopped into quant blocks,
      (K / block, 1, block): u32 packed bf16, or f32 if norm_dtype is f32.
      None without an input norm.
  """
  num_lanes = pltpu.get_tpu_info().num_lanes
  half = num_lanes // 2
  ba_w = jnp.zeros((in_proj_ba.shape[0], num_lanes), in_proj_ba.dtype)
  ba_w = ba_w.at[:, :num_v_heads].set(in_proj_ba[:, :num_v_heads])
  ba_w = ba_w.at[:, half : half + num_v_heads].set(in_proj_ba[:, num_v_heads:])
  kind = model_config.Norm(input_norm)
  if kind == model_config.Norm.NONE:
    return ba_w, None
  assert input_norm_w is not None
  if jnp.dtype(norm_dtype) == jnp.float32:
    layout = norm.f32_weight
  else:
    layout = norm.packed_bf16_weight
  return ba_w, layout(kind, input_norm_w, act_quant_block)
