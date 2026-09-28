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
"""fused_out_proj: everything between batched_rpa and the TP all-reduce.

One Pallas kernel takes batched_rpa's raw output o, unpacks the query heads
from its packed layout, applies the optional sigmoid output gate and
sub-channel fp8 quant, and runs the o_proj matmul (fp8 or bf16, optional bias)
straight into the (T, K) bf16 layout the all-reduce consumes. Its steps are
listed in pipeline.py's module docstring.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, BFloat16, Float32  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import memory_ref
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import pipeline


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "act_quant_block",
        "num_tokens",
        "tile_size",
    ),
)
def fused_out_proj(
    o: BFloat16[Array, "Hkv T_rpa q_per_kv/q_packing q_packing Dp"],
    w: jax.Array,
    w_scale: Float32[Array, "1 K"] | None,
    bias: BFloat16[Array, "1 K"] | None,
    gate: BFloat16[Array, "T Hq*D"] | None = None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    act_quant_block: int = 1024,
    num_tokens: int | None = None,
    tile_size: int | None = None,
) -> BFloat16[Array, "T K"]:
  """Fused RPA output unpack + optional sigmoid gate + o_proj matmul.

  The kernel itself starts at pipeline.tile_body; this function only sets up
  the call.

  fp8, bias, the output gate and Q's dtype follow from the arrays passed in.

  Args:
    o: batched_rpa's raw output (ragged_paged_attention's
      return_raw_output=True); Dp = align(D, 128). T_rpa may exceed T when
      batched_rpa padded the token dimension.
    w: o_proj (Hq*D, K) fp8 or bf16, as loaded.
    w_scale: per-column scale, fp8 only.
    bias: (1, K), or None.
    gate: Qwen3.5's per-head output gate from fused_qkv, or None.
    num_q_heads: query heads on this chip (after TP).
    num_kv_heads: KV heads on this chip (after TP).
    head_dim: D.
    act_quant_block: fp8 only: columns per activation quant block.
    num_tokens: active token count T <= T_rpa; defaults to gate.shape[0] if gate
      is given, else T_rpa.
    tile_size: overrides the token tile (tiling.TileConfig.create).

  Returns:
    out: (T, K) bf16, ready for the TP all-reduce.
  """
  # Make and check every static decision.
  model = model_config.AttentionConfig(
      num_q_heads=num_q_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      gate=gate is not None,
      fp8=w.dtype == quantized_matmul.F8,
      act_quant_block=act_quant_block,
      bias=bias is not None,
      q_dtype=o.dtype,
  )
  weights = model_config.OutProjWeights(w, w_scale, bias)
  cfg = config_lib.FusedOutProjConfig.create(
      model, o, gate, weights, num_tokens=num_tokens, tile_size=tile_size
  )

  # Place operands: per-tile inputs stream from HBM (or stay in VMEM with one
  # tile); W stays in HBM and the kernel streams it; pinned weights go to VMEM.
  inputs = memory_ref.Inputs(o=o, gate=gate)
  inputs_space = memory.tiled_input_space(cfg.tile.num_tiles)
  w = memory.constrain(weights.w, pltpu.HBM)
  pinned, pinned_specs = memory.place(
      model_config.PinnedOutProjWeights(
          w_scale=weights.w_scale, bias=weights.bias
      ),
      pltpu.VMEM,
  )
  out_shapes = memory_ref.Outputs.shapes(cfg)

  (outputs,) = pl.pallas_call(
      functools.partial(pipeline.outer_pipeline, cfg=cfg),
      name=cfg.kernel_name,
      metadata=cfg.metadata(),
      in_specs=(
          memory.specs(inputs, inputs_space),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pinned_specs,
      ),
      out_specs=(memory.specs(out_shapes, pltpu.HBM),),
      out_shape=(out_shapes,),
      scratch_shapes=(memory_ref.Scratch.shapes(cfg, w),),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfg.vmem_limit_bytes
      ),
  )(inputs, w, pinned)
  return outputs.out
