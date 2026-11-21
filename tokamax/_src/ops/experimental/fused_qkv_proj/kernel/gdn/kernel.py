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
"""fused_gdn: the fused GDN (Gated DeltaNet) prologue, as one Pallas kernel.

For Qwen3-Next and Qwen3.5: residual add, GemmaRMSNorm, sub-channel fp8 quant,
the fp8 in_proj_qkvz matmul and the bf16 in_proj_ba matmul, written straight
into the layouts the GDN kernel (fused_conv1d_gdn) reads. Its steps are
listed in pipeline.py's module docstring.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, Float32  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import memory_ref
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import pipeline


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_k_heads",
        "num_v_heads",
        "head_dim",
        "input_norm",
        "eps",
        "norm_dtype",
        "act_quant_block",
        "tile_size",
    ),
)
def fused_gdn(
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    w: jax.Array,
    w_scale: Float32[Array, "1 N"] | None,
    ba_w: BFloat16[Array, "K lanes"],
    norm_w: jax.Array | None,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int = 128,
    input_norm: str = "rms_one_plus_w",
    eps: float = 1e-6,
    norm_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    act_quant_block: int = 1024,
    tile_size: int | None = None,
) -> memory_ref.Outputs[jax.Array]:
  """Fused GDN prologue: input norm + in_proj_qkvz (fp8) + in_proj_ba (bf16).

  The kernel itself starts at pipeline.tile_body; this function only sets up
  the call.

  fp8 and residual follow from the arrays passed in.

  Args:
    x: decoder hidden states, before the input norm. T % 128 == 0 (pad decode
      batches).
    residual: the residual stream to add before the input norm, or None.
    w: in_proj_qkvz (K, N) fp8, as loaded.
    w_scale: in_proj_qkvz's per-column scale.
    ba_w: in_proj_ba, from weights.prepare_weights at weight load time.
    norm_w: input norm weight, from weights.prepare_weights; None without an
      input norm.
    num_k_heads: heads of q and k on this chip (after TP).
    num_v_heads: value heads on this chip (after TP).
    head_dim: D, shared by key and value heads.
    input_norm: the decoder input norm, a model_config.Norm value.
    eps: norm epsilon.
    norm_dtype: precision of the input norm's x * w (vLLM: f32).
    act_quant_block: columns per fp8 activation quant block.
    tile_size: overrides the token tile (tiling.TileConfig.create).

  Returns:
    (mixed_qkv, z, b, a, residual), laid out as memory_ref.Outputs documents.
  """
  # Make and check every static decision.
  model = model_config.GdnConfig(
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_dim=head_dim,
      fp8=w.dtype == quantized_matmul.F8,
      act_quant_block=act_quant_block,
      input_norm=model_config.Norm(input_norm),
      residual=residual is not None,
      eps=eps,
      norm_dtype=norm_dtype,
  )
  weights = model_config.GdnWeights(w, w_scale, ba_w, norm_w)
  cfg = config_lib.FusedGdnConfig.create(
      model, x, residual, weights, tile_size=tile_size
  )

  # Place operands: per-tile inputs stream from HBM (or stay in VMEM with one
  # tile); both projection weights stay in HBM and the kernel streams them;
  # pinned weights go to VMEM.
  # The residual uses the (16, 128) tiling XLA picks between layers, so XLA
  # doesn't relayout it at the kernel boundary.
  residual = memory.large_tiled(residual)
  inputs = memory_ref.Inputs(x=x, residual=residual)
  inputs_space = memory.tiled_input_space(cfg.tile.num_tiles)
  w = memory.constrain(weights.w, pltpu.HBM)
  ba_w = memory.constrain(weights.ba_w, pltpu.HBM)
  pinned, pinned_specs = memory.place(weights.pinned, pltpu.VMEM)
  out_shapes = memory_ref.Outputs.shapes(cfg)

  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  (outputs,) = pl.pallas_call(
      functools.partial(pipeline.outer_pipeline, cfg=cfg),
      name=cfg.kernel_name,
      metadata=cfg.metadata(),
      in_specs=(
          memory.specs(inputs, inputs_space),
          hbm_spec,
          hbm_spec,
          pinned_specs,
      ),
      out_specs=(memory.specs(out_shapes, pltpu.HBM),),
      out_shape=(out_shapes,),
      scratch_shapes=(memory_ref.Scratch.shapes(cfg, weights),),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfg.vmem_limit_bytes
      ),
  )(inputs, w, ba_w, pinned)
  return outputs._replace(residual=memory.large_tiled(outputs.residual))
