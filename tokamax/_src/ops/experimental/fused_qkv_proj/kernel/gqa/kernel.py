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
"""fused_qkv: the fused GQA attention prologue, as one Pallas kernel.

Input norm, QKV projection, q/k/v norms, RoPE and the static KV fp8 cast,
with Q and the new KV written directly in batched_rpa's input layouts. Its
steps are listed in pipeline.py's module docstring.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, Float32  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.batched_rpa.kernel import configs as rpa_configs
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import config as config_lib
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import memory_ref
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import pipeline


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "gate",
        "v_from_k",
        "input_norm",
        "q_norm",
        "k_norm",
        "v_norm",
        "rope_offset",
        "rope_precision",
        "eps",
        "k_scale",
        "v_scale",
        "act_quant_block",
        "q_dtype",
        "kv_dtype",
        "kv_layout",
        "page_size",
        "tile_size",
    ),
)
def fused_qkv(
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    w: jax.Array,
    w_scale: Float32[Array, "1 N"] | None,
    bias: BFloat16[Array, "1 N"] | None,
    q_norm_w: BFloat16[Array, "1 D"] | None,
    k_norm_w: BFloat16[Array, "1 D"] | None,
    input_norm_w: jax.Array | None,
    cos: BFloat16[Array, "T n_pairs"] | None,
    sin: BFloat16[Array, "T n_pairs"] | None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate: bool = False,
    v_from_k: bool = False,
    input_norm: str = "rms",
    q_norm: str = "none",
    k_norm: str = "none",
    v_norm: str = "none",
    rope_offset: int | None = None,
    rope_precision: str = "fp32",
    eps: float = 1e-6,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    act_quant_block: int = 1024,
    q_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    kv_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    kv_layout: rpa_configs.KVLayout = rpa_configs.KVLayout.SEQ_ALONG_LANE,
    page_size: int = 128,
    tile_size: int | None = None,
) -> memory_ref.Outputs[jax.Array]:
  """Input norm + QKV projection + q/k/v norms + RoPE + KV fp8 cast.

  The kernel itself starts at pipeline.tile_body; this function only sets up
  the call.

  fp8, bias, residual and RoPE's n_pairs follow from the arrays passed in;
  the keyword arguments are the model facts the arrays don't show.

  Args:
    x: decoder hidden states, before the input norm.
    residual: the residual stream to add before the input norm, or None.
    w: (K, N) fp8 or bf16, q/k/v projections side by side, as loaded.
    w_scale: per-column scale, fp8 only.
    bias: (1, N), or None.
    q_norm_w: q norm weight, if q_norm has one.
    k_norm_w: k norm weight, if k_norm has one.
    input_norm_w: from weights.prepare_input_norm_w, at weight load time.
    cos: RoPE cos, (T, n_pairs); None without RoPE.
    sin: RoPE sin, (T, n_pairs); None without RoPE.
    num_q_heads: query heads on this chip (after TP).
    num_kv_heads: KV heads on this chip (after TP).
    head_dim: D.
    gate: q_proj also outputs a per-head output gate (Qwen3.5).
    v_from_k: no v_proj; v is the raw k projection (Gemma 4 full attention).
    input_norm: the decoder input norm, a model_config.Norm value ("none",
      "rms", "rms_one_plus_w" or "rms_noscale").
    q_norm: per-head q norm, as input_norm.
    k_norm: per-head k norm, as input_norm.
    v_norm: per-head v norm, as input_norm.
    rope_offset: distance between the two dims of a RoPE pair; defaults to
      n_pairs (standard NeoX). Gemma 4 full attention: head_dim // 2.
    rope_precision: a model_config.RopePrecision value. fp32 (default) matches
      vLLM TPU's unfused RoPE; bf16 rounds every product.
    eps: norm epsilon.
    k_scale: static fp8 scale of the K cache.
    v_scale: static fp8 scale of the V cache.
    act_quant_block: fp8 only: columns per activation quant block.
    q_dtype: Q's dtype; only bf16 is implemented.
    kv_dtype: the KV cache's dtype; only float8_e4m3fn is implemented.
    kv_layout: batched_rpa KV layout to write.
    page_size: batched_rpa KV cache page size; SEQ_ALONG_LANE pads to it.
    tile_size: overrides the token tile (tiling.TileConfig.create).

  Returns:
    (q, kv, gate, residual), laid out as memory_ref.Outputs documents.
  """
  # Make and check every static decision.
  assert (cos is None) == (sin is None)
  rope: model_config.Rope | None = None
  if cos is not None:
    n_pairs = cos.shape[-1]
    rope = model_config.Rope(
        n_pairs, n_pairs if rope_offset is None else rope_offset
    )
  model = model_config.AttentionConfig(
      num_q_heads=num_q_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      gate=gate,
      v_from_k=v_from_k,
      fp8=w.dtype == quantized_matmul.F8,
      act_quant_block=act_quant_block,
      bias=bias is not None,
      input_norm=model_config.Norm(input_norm),
      residual=residual is not None,
      q_norm=model_config.Norm(q_norm),
      k_norm=model_config.Norm(k_norm),
      v_norm=model_config.Norm(v_norm),
      rope=rope,
      rope_precision=model_config.RopePrecision(rope_precision),
      eps=eps,
      q_dtype=q_dtype,
      kv_dtype=kv_dtype,
      k_scale=k_scale,
      v_scale=v_scale,
  )
  weights = model_config.GqaWeights(
      w, w_scale, bias, q_norm_w, k_norm_w, input_norm_w
  )
  cfg = config_lib.FusedQkvConfig.create(
      model,
      x,
      residual,
      weights,
      kv_layout=kv_layout,
      page_size=page_size,
      tile_size=tile_size,
  )

  # Place operands: per-tile inputs stream from HBM (or stay in VMEM with one
  # tile); W stays in HBM and the kernel streams it; pinned weights go to VMEM.
  # The residual uses the (16, 128) tiling XLA picks between layers, so XLA
  # doesn't relayout it at the kernel boundary.
  residual = memory.large_tiled(residual)
  inputs = memory_ref.Inputs(x=x, residual=residual, cos=cos, sin=sin)
  inputs_space = memory.tiled_input_space(cfg.tile.num_tiles)
  w = memory.constrain(weights.w, pltpu.HBM)
  pinned, pinned_specs = memory.place(weights.pinned, pltpu.VMEM)
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
  return outputs._replace(residual=memory.large_tiled(outputs.residual))
