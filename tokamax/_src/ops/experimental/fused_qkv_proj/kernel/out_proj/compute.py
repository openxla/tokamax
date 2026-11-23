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
"""fused_out_proj's math for Step 0 of pipeline.tile_body.

Step 1 (one quantized_matmul.project_block call per column block) is short
enough to stay inline in tile_body. Step 0 reads and writes the VMEM refs
itself, at the same points a single fused loop would: every value is stored as
soon as it is computed, so few vregs stay live.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, BFloat16  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import memory_ref


def prep_chunk_fp8(
    cfg: config_lib.FusedOutProjConfig,
    inputs: memory_ref.Inputs[jax.Ref],  # Tile blocks.
    scratch: memory_ref.Scratch[jax.Ref],
    rows: pl.Slice,
) -> None:
  """Step 0, fp8: unpack, gate and sub-channel quant of `rows` into x_q."""
  m = cfg.model
  assert scratch.x_scale is not None  # fp8 scratch.
  num_blocks = m.q_dim // cfg.quant_block
  pairs_per_block = cfg.quant_block // (m.q_packing * m.head_dim)
  groups_per_kv_head = m.q_per_kv // m.q_packing
  for block in range(num_blocks):
    pairs = []
    for pair in range(block * pairs_per_block, (block + 1) * pairs_per_block):
      kv_head, group = divmod(pair, groups_per_kv_head)
      pairs.append(head_pair(cfg, inputs, rows, kv_head, group))
    x = pairs[0] if len(pairs) == 1 else jnp.concatenate(pairs, axis=1)
    x_q, x_scale = quantized_matmul.quantize_block(x)
    scratch.x_q[rows, pl.ds(block * cfg.quant_block, cfg.quant_block)] = x_q
    scratch.x_scale[block, rows] = layout_util.broadcast_lanes(x_scale)


def prep_chunk_bf16(
    cfg: config_lib.FusedOutProjConfig,
    inputs: memory_ref.Inputs[jax.Ref],  # Tile blocks.
    scratch: memory_ref.Scratch[jax.Ref],
    rows: pl.Slice,
) -> None:
  """Step 0, bf16: unpack and gate `rows` into x_q."""
  m = cfg.model
  pair_cols = m.q_packing * m.head_dim
  groups_per_kv_head = m.q_per_kv // m.q_packing
  for kv_head in range(m.num_kv_heads):
    for group in range(groups_per_kv_head):
      first_col = (kv_head * groups_per_kv_head + group) * pair_cols
      scratch.x_q[rows, pl.ds(first_col, pair_cols)] = head_pair(
          cfg, inputs, rows, kv_head, group
      )


def head_pair(
    cfg: config_lib.FusedOutProjConfig,
    inputs: memory_ref.Inputs[jax.Ref],
    rows: pl.Slice,
    kv_head: int,
    group: int,
) -> BFloat16[Array, "rows 2*D"]:
  """Q heads (kv_head, group) of `rows` as [head_0 | head_1], gated.

  batched_rpa packs each pair of query heads into one u32 per lane (low bits
  head 0); the pair is read with one strided load and split back apart.

  Args:
    cfg: the kernel's config.
    inputs: this tile's input blocks.
    rows: the rows to read.
    kv_head: the KV head whose query heads these are.
    group: which pair of kv_head's query heads.

  Returns:
    (rows, 2D) bf16.
  """
  m = cfg.model
  head_dim = m.head_dim
  first_col = (kv_head * m.q_per_kv + group * m.q_packing) * head_dim
  packed = layout_util.strided_load_heads(
      inputs.o.at[kv_head], rows, group, head_dim
  )
  if 2 * head_dim == pltpu.get_tpu_info().num_lanes:  # Both heads in one vreg.
    pair = layout_util.unpack_lane_pair(packed, head_dim)
  else:
    pair = jnp.concatenate(
        layout_util.unpack_heads(packed, jnp.bfloat16), axis=1
    )
  if m.gate:
    assert inputs.gate is not None
    gate = inputs.gate[rows, pl.ds(first_col, 2 * head_dim)]
    pair = pair * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(pair.dtype)
  return pair
