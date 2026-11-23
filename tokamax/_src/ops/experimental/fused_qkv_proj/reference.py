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
"""Unfused XLA attention and GDN prologues built from the shared components.

`gqa_reference()` computes the GQA attention prologue with one of three input
numerics:

  Numerics.KERNEL: what fused_qkv computes; the kernel must match it exactly.
  Numerics.VLLM: what vllm-torchtpu computes, with per-token fp8 activations.
  Numerics.UNQUANTIZED: f32 activations and matmul; the accuracy reference for
    the other two.

Everything after the QKV projection (q/k/v norms, RoPE, KV cast, layouts) is
the same for all three and follows vLLM.

`gdn_reference()` does the same for the GDN prologue: the in_proj_qkvz and
in_proj_ba projections, laid out for the GDN kernel.
"""

import enum
import functools

import jax
from jax.experimental import layout
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, Float, Float32, Float8e4m3fn  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.batched_rpa.kernel import configs as rpa_configs
from tokamax._src.ops.batched_rpa.kernel import wrapper as rpa_wrapper
from tokamax._src.ops.experimental.fused_qkv_proj import models
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import norm
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import rope
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import memory_ref as gdn_memory_ref
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import memory_ref as gqa_memory_ref
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import config as out_proj_config

# The input-side fields (input_norm, residual, fp8, act_quant_block, eps) are
# shared by both model_config.
_Config = model_config.AttentionConfig | model_config.GdnConfig


class Numerics(enum.Enum):
  """Input numerics; see the module docstring."""

  KERNEL = "kernel"
  VLLM = "vllm"
  UNQUANTIZED = "unquantized"


def project(
    x: Float[Array, "T K"],
    w: Float[Array, "K N"],
    x_scale: Float32[Array, "T 1"] | None = None,
    w_scale: Float32[Array, "1 N"] | None = None,
    bias: Float[Array, "1 N"] | None = None,
    exact: bool = False,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> Float[Array, "T N"]:
  """x @ w in f32, dequantized and biased, rounded to `dtype`.

  Args:
    x: (T, K).
    w: (K, N).
    x_scale: (T, 1) f32, or None.
    w_scale: (1, N) f32, or None.
    bias: (1, N), or None.
    exact: run the dot on f32 inputs at HIGHEST precision.
    dtype: the result's dtype.

  Returns:
    (T, N) in dtype.
  """
  if exact:
    # Keeps XLA from folding the upcast back into a low-precision dot.
    x, w = jax.lax.optimization_barrier(
        (x.astype(jnp.float32), w.astype(jnp.float32))
    )
  out = jnp.dot(
      x,
      w,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST if exact else None,
  )
  if not exact:
    # Pin the f32 result row-major, the layout a standalone dot gets. On
    # TPU7x an fp8 dot is exact or ~4e-3 off (rms relative) depending on the
    # layout XLA picks for its result, which follows the consumers: a change
    # downstream (e.g. fp32 RoPE) moved this reference by 0.1 dB.
    out = layout.with_layout_constraint(
        out, layout.Layout(major_to_minor=(0, 1))
    )
  if x_scale is not None:
    assert w_scale is not None
    out = out * x_scale * w_scale
  if bias is not None:
    out = out + bias.astype(jnp.float32)
  return out.astype(dtype)


def matmul_input(
    config: _Config,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
) -> tuple[
    Float[Array, "T K"],
    Float32[Array, "T 1"] | None,
    BFloat16[Array, "T K"] | None,
]:
  """vllm-torchtpu: residual add + input norm (+ per-token fp8 quant).

  Where each step comes from:

  1. norm.add_norm is the decoder layer's input norm, e.g.
     third_party/py/vllm/model_executor/models/qwen3_moe.py
     Qwen3MoeDecoderLayer.forward:
       hidden_states, residual = self.input_layernorm(hidden_states, residual)
     RMSNorm (and GemmaRMSNorm, with weight + 1) forward_native calls
     third_party/py/vllm/ir/ops/layernorm.py fused_add_rms_norm:
       x = x.to(torch.float32)
       x = x + x_residual.to(torch.float32)
       x_residual = x.to(orig_dtype)
       variance = x_var.pow(2).mean(dim=-1, keepdim=True)
       x = x * torch.rsqrt(variance + epsilon)
       x = x.to(weight.dtype) * weight
  2. quantized_matmul.quantize_fp8_per_token is qkv_proj's activation quant.
     Paths are in
     vllm_torchtpu/layers/vllm/quantization/fp8.py
     VllmFp8LinearMethodTPU.apply:
       out = quantized_matmul(x, weight, layer.weight_scale)
     layers/vllm/linear_common.py quantized_matmul -> _quantized_matmul_jax
     -> (per-channel w_scale) kernels/quantized_matmul/util.py
     xla_quantized_matmul:
       x_q, x_scale = quantize_tensor(x, w_q.dtype)  # quantize_block(x, -1)
     util.py quantize_block:
       abs_max = jnp.max(jnp.abs(data), axis=axis, keepdims=True)
       scale = abs_max / dtype_max
       scale = jnp.where(scale == 0, 1.0, scale)
       data_q = (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)

  Args:
    config: the model config.
    x: decoder hidden states.
    residual: the residual stream, or None.
    weights: as loaded.

  Returns:
    (x, x_scale, new residual): x is fp8 with (T, 1) x_scale if config.fp8,
    else bf16 with x_scale None.
  """
  x, residual = norm.add_norm(
      config.input_norm, x, residual, weights.input_norm_w, config.eps
  )
  x_scale = None
  if config.fp8:
    x, x_scale = quantized_matmul.quantize_fp8_per_token(x)
  return x, x_scale, residual


def fused_matmul_input(
    config: _Config,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
) -> tuple[
    Float8e4m3fn[Array, "T K/block block"],
    Float32[Array, "T K/block 1"],
    Float32[Array, "T 1"] | None,
    BFloat16[Array, "T K"] | None,
]:
  """fused_qkv (fp8): residual add + input norm + sub-channel fp8 quant.

  Differs from matmul_input in four ways. All of them let the kernel keep the
  elementwise work in bf16 (two values per lane) and skip a normalize pass:
    1. The residual add is bf16, and the rms is taken from that bf16 sum (vLLM
       adds and takes the rms in f32).
    2. x is not divided by its rms before the quant. A per-row scale cancels
       in the quant, so 1/rms is returned as `row_scale` and applied to the
       matmul output instead. This is FlashNorm's deferred
       normalization (https://arxiv.org/abs/2407.09577).
    3. One fp8 scale per act_quant_block columns (sub-channel) instead of one
       per token, which tracks the activations more closely.
    4. The norm weight multiply and the quant are bf16: x * w, then
       x * bf16(1 / scale) (vLLM: f32 x / scale; GemmaRMSNorm's 1 + w in f32).

  Args:
    config: the model config.
    x: decoder hidden states.
    residual: the residual stream, or None.
    weights: as loaded.

  Returns:
    (x_q, x_scale, row_scale, new residual); row_scale is None without an
    input norm.
  """
  x, row_scale, residual = fused_normed_input(config, x, residual, weights)
  x_q, x_scale = fused_quant(config, x)
  return x_q, x_scale, row_scale, residual


def fused_normed_input(
    config: _Config,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
) -> tuple[
    BFloat16[Array, "T K"],
    Float32[Array, "T 1"] | None,
    BFloat16[Array, "T K"] | None,
]:
  """Steps 1, 2 and the x * w of step 4 of fused_matmul_input.

  Args:
    config: the model config.
    x: decoder hidden states.
    residual: the residual stream, or None.
    weights: as loaded.

  Returns:
    (x * w, 1/rms, new residual); 1/rms is None without an input norm.
  """
  h = x
  if residual is not None:
    h = h + residual  # 1.
    residual = h
  normed, row_scale = h, None
  if config.input_norm != model_config.Norm.NONE:
    h_f32 = h.astype(jnp.float32)
    sum_sq = jnp.sum(h_f32 * h_f32, axis=-1, keepdims=True)
    row_scale = jax.lax.rsqrt(sum_sq / h.shape[-1] + config.eps)  # 2.
    assert weights.input_norm_w is not None
    w = weights.input_norm_w.astype(jnp.float32)
    if config.input_norm == model_config.Norm.RMS_ONE_PLUS_W:
      w = w + 1.0
    normed = h * w.astype(h.dtype)  # 4.
  return normed, row_scale, residual


def fused_quant(
    config: _Config,
    x: BFloat16[Array, "T K"],
    block: int | None = None,
) -> tuple[
    Float8e4m3fn[Array, "T K/block block"], Float32[Array, "T K/block 1"]
]:
  """Step 3 and the quant of step 4 of fused_matmul_input: (x_q, x_scale)."""
  num_tokens, hidden_size = x.shape
  block = config.act_quant_block if block is None else block
  x_blocks = x.reshape(num_tokens, hidden_size // block, block)  # 3.
  amax = jnp.max(jnp.abs(x_blocks), axis=-1, keepdims=True).astype(jnp.float32)
  x_scale = amax / quantized_matmul.F8_MAX
  x_scale = jnp.where(x_scale == 0, 1.0, x_scale)
  x_q = x_blocks * (1.0 / x_scale).astype(x.dtype)  # 4.
  x_q = x_q.clip(-quantized_matmul.F8_MAX, quantized_matmul.F8_MAX).astype(
      quantized_matmul.F8
  )
  return x_q, x_scale


def fused_project(
    x_q: Float8e4m3fn[Array, "T K/block block"],
    x_scale: Float32[Array, "T K/block 1"],
    row_scale: Float32[Array, "T 1"] | None,
    weights: models.QkvWeights | model_config.OutProjWeights,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> Float[Array, "T N"]:
  """fused_qkv (fp8): the QKV matmul, dequantized one quant block at a time."""
  num_blocks, block_width = x_q.shape[1:]
  acc = 0.0
  for block in range(num_blocks):
    w = weights.w[block * block_width : (block + 1) * block_width]
    acc = acc + x_scale[:, block] * jnp.dot(
        x_q[:, block], w, preferred_element_type=jnp.float32
    )
  if row_scale is not None:
    acc = acc * row_scale
  assert weights.w_scale is not None
  acc = acc * weights.w_scale
  if weights.bias is not None:
    acc = acc + weights.bias.astype(jnp.float32)
  return acc.astype(dtype)


def unquantized_input(
    config: _Config,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
) -> tuple[Float32[Array, "T K"], BFloat16[Array, "T K"] | None]:
  """f32 residual add and input norm.

  All f32: rounding x to bf16 here, as vLLM does, would hide vLLM's rounding
  error and count the kernel's.

  Args:
    config: the model config.
    x: decoder hidden states.
    residual: the residual stream, or None.
    weights: as loaded.

  Returns:
    (normed x, new residual).
  """
  h_f32 = x.astype(jnp.float32)
  if residual is not None:
    h_f32 = h_f32 + residual.astype(jnp.float32)
    residual = h_f32.astype(x.dtype)
  inv_rms = jax.lax.rsqrt(
      jnp.mean(h_f32 * h_f32, axis=-1, keepdims=True) + config.eps
  )
  normed_f32 = h_f32
  if config.input_norm != model_config.Norm.NONE:
    assert weights.input_norm_w is not None
    norm_w = weights.input_norm_w.astype(jnp.float32)
    normed_f32 = norm.scale_norm(
        config.input_norm, h_f32, inv_rms, norm_w, jnp.float32
    )
  return normed_f32, residual


def qkv_projection(
    config: model_config.AttentionConfig,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
    numerics: Numerics,
) -> tuple[BFloat16[Array, "T N"], BFloat16[Array, "T K"] | None]:
  """The QKV projection with `numerics`: (qkv, new residual)."""
  if numerics == Numerics.KERNEL and config.fp8:
    x_q, x_scale, row_scale, residual = fused_matmul_input(
        config, x, residual, weights
    )
    return fused_project(x_q, x_scale, row_scale, weights), residual
  if numerics == Numerics.UNQUANTIZED:
    normed_f32, residual = unquantized_input(config, x, residual, weights)
    w = weights.w.astype(jnp.float32)
    if weights.w_scale is not None:
      w = w * weights.w_scale
    return project(normed_f32, w, bias=weights.bias, exact=True), residual
  # bf16 configs have no quant, so KERNEL numerics are vLLM's.
  x, x_scale, residual = matmul_input(config, x, residual, weights)
  qkv = project(x, weights.w, x_scale, weights.w_scale, weights.bias)
  return qkv, residual


@functools.partial(
    jax.jit,
    static_argnums=0,
    static_argnames=("kv_layout", "numerics", "page_size"),
)
def gqa_reference(
    config: model_config.AttentionConfig,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
    cos: BFloat16[Array, "T n_pairs"] | None,
    sin: BFloat16[Array, "T n_pairs"] | None,
    *,
    kv_layout: rpa_configs.KVLayout = rpa_configs.KVLayout.SEQ_ALONG_LANE,
    numerics: Numerics = Numerics.KERNEL,
    page_size: int = 128,
) -> gqa_memory_ref.Outputs[jax.Array]:
  """The unfused GQA attention prologue: fused_qkv's outputs with `numerics`.

  Args:
    config: the attention config.
    x: decoder hidden states.
    residual: the residual stream, or None.
    weights: as loaded, not prepared.
    cos: RoPE cos table, or None without RoPE.
    sin: RoPE sin table, or None without RoPE.
    kv_layout: batched_rpa KV layout of the returned kv.
    numerics: input numerics; see the module docstring.
    page_size: batched_rpa page size (SEQ_ALONG_LANE KV is padded to it).

  Returns:
    q: batched_rpa's prepared Q; Dp is D aligned to 128.
    kv: batched_rpa's prepared KV, in `kv_layout`.
    gate: per-head output gates, or None without config.gate.
    residual: x + residual, or None.
  """
  num_tokens, head_dim = x.shape[0], config.head_dim
  num_kv_heads = config.num_kv_heads
  qkv, residual = qkv_projection(config, x, residual, weights, numerics)
  q = qkv[:, : config.kv_col].reshape(num_tokens, config.num_q_heads, -1)
  gate = q[..., head_dim:].reshape(num_tokens, -1) if config.gate else None
  q = q[..., :head_dim]
  k = qkv[:, config.k_col(0) : config.k_col(num_kv_heads)]
  k = k.reshape(num_tokens, num_kv_heads, head_dim)
  v = qkv[:, config.v_col(0) : config.v_col(num_kv_heads)]
  v = v.reshape(num_tokens, num_kv_heads, head_dim)
  if cos is not None:
    assert sin is not None
    cos, sin = cos[:, None], sin[:, None]
  # vLLM rotates in fp32; KERNEL numerics follow config.rope_precision.
  precision = (
      config.rope_precision
      if numerics == Numerics.KERNEL
      else model_config.RopePrecision.FP32
  )
  q = norm.norm(config.q_norm, q, weights.q_norm_w, config.eps)
  q = rope.rope(config.rope, q, cos, sin, precision)
  k = norm.norm(config.k_norm, k, weights.k_norm_w, config.eps)
  k = rope.rope(config.rope, k, cos, sin, precision)
  v = norm.norm(config.v_norm, v, None, config.eps)
  q, kv, _ = rpa_wrapper.prepare_inputs(
      q,
      quantized_matmul.quantize_fp8_per_tensor(k, config.k_scale),
      quantized_matmul.quantize_fp8_per_tensor(v, config.v_scale),
      jnp.dtype(config.q_dtype),
      jnp.dtype(config.kv_dtype),
      kv_layout=kv_layout,
      page_size=page_size,
  )
  return gqa_memory_ref.Outputs(q=q, kv=kv, gate=gate, residual=residual)


def gdn_projection(
    config: model_config.GdnConfig,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
    numerics: Numerics,
) -> tuple[
    Float[Array, "T N"], Float[Array, "T 2*Hv"], BFloat16[Array, "T K"] | None
]:
  """in_proj_qkvz and in_proj_ba of the input-normed x.

  KERNEL numerics match fused_gdn exactly.

  Args:
    config: the GDN config.
    x: decoder hidden states.
    residual: the residual stream, or None.
    weights: as loaded.
    numerics: see the module docstring.

  Returns:
    (qkvz, ba, new residual): qkvz is [mixed_qkv | z] and ba is [b | a],
    bf16 for VLLM, else f32.
  """
  if numerics == Numerics.KERNEL:
    normed, row_scale, new_residual = fused_normed_input(
        config, x, residual, weights
    )
    # f32 norm_dtype: h * w in f32, rounded to bf16 once.
    if jnp.dtype(config.norm_dtype) == jnp.float32 and row_scale is not None:
      h = x if residual is None else x + residual
      assert weights.input_norm_w is not None
      w = weights.input_norm_w.astype(jnp.float32)
      if config.input_norm == model_config.Norm.RMS_ONE_PLUS_W:
        w = w + 1.0
      normed = (h.astype(jnp.float32) * w).astype(x.dtype)
    x_q, x_scale = fused_quant(config, normed)
    qkvz = fused_project(x_q, x_scale, row_scale, weights, jnp.float32)
    assert weights.side_w is not None
    ba = project(normed, weights.side_w, exact=True, dtype=jnp.float32)
    if row_scale is not None:
      ba = ba * row_scale
    return qkvz, ba, new_residual
  if numerics == Numerics.UNQUANTIZED:
    normed_f32, residual = unquantized_input(config, x, residual, weights)
    assert weights.w_scale is not None and weights.side_w is not None
    w = weights.w.astype(jnp.float32) * weights.w_scale
    qkvz = project(normed_f32, w, exact=True, dtype=jnp.float32)
    ba = project(normed_f32, weights.side_w, exact=True, dtype=jnp.float32)
    return qkvz, ba, residual
  x, residual = norm.add_norm(
      config.input_norm, x, residual, weights.input_norm_w, config.eps
  )
  x_q, x_scale = quantized_matmul.quantize_fp8_per_token(x)
  qkvz = project(x_q, weights.w, x_scale, weights.w_scale)
  assert weights.side_w is not None
  ba = project(x, weights.side_w)
  return qkvz, ba, residual


def gdn_layout(
    config: model_config.GdnConfig,
    qkvz: Float[Array, "T N"],
    ba: Float[Array, "T 2*Hv"],
    z_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> tuple[
    Float32[Array, "T 1 qkv_cols"],
    Float[Array, "T Hv D"],
    Float32[Array, "T 1 lanes"],
    Float32[Array, "T 1 lanes"],
]:
  """Splits the projections into the GDN kernel's (mixed_qkv, z, b, a)."""
  num_tokens, num_v_heads = qkvz.shape[0], config.num_v_heads
  lanes = 128  # The GDN kernel pads b and a to 128 lanes.
  mixed = qkvz[:, : config.qkv_cols].astype(jnp.float32)
  mixed = mixed.reshape(num_tokens, 1, -1)
  z = qkvz[:, config.z_col :].astype(z_dtype)
  z = z.reshape(num_tokens, num_v_heads, config.head_dim)

  def pad(values: Float[Array, "T Hv"]) -> Float32[Array, "T 1 lanes"]:
    values = jnp.pad(
        values.astype(jnp.float32), ((0, 0), (0, lanes - num_v_heads))
    )
    return values.reshape(num_tokens, 1, lanes)

  return mixed, z, pad(ba[:, :num_v_heads]), pad(ba[:, num_v_heads:])


@functools.partial(jax.jit, static_argnums=0, static_argnames=("numerics",))
def gdn_reference(
    config: model_config.GdnConfig,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
    *,
    numerics: Numerics = Numerics.KERNEL,
) -> gdn_memory_ref.Outputs[jax.Array]:
  """The unfused GDN prologue: fused_gdn's outputs with `numerics`.

  Args:
    config: the GDN config.
    x: decoder hidden states.
    residual: the residual stream, or None.
    weights: as loaded, not prepared.
    numerics: input numerics; see the module docstring.

  Returns:
    mixed_qkv: [q | k | v], f32.
    z: f32 for UNQUANTIZED, else bf16.
    b: f32, Hv lanes zero-padded to 128.
    a: f32, Hv lanes zero-padded to 128.
    residual: x + residual, or None.
  """
  qkvz, ba, residual = gdn_projection(config, x, residual, weights, numerics)
  z_dtype = jnp.float32 if numerics == Numerics.UNQUANTIZED else jnp.bfloat16
  mixed_qkv, z, b, a = gdn_layout(config, qkvz, ba, z_dtype)
  return gdn_memory_ref.Outputs(
      mixed_qkv=mixed_qkv, z=z, b=b, a=a, residual=residual
  )


def out_proj_input(
    config: model_config.AttentionConfig,
    o: BFloat16[Array, "Hkv T_rpa q_per_kv/q_packing q_packing Dp"],
    gate: BFloat16[Array, "T Hq*D"] | None = None,
    num_tokens: int | None = None,
) -> BFloat16[Array, "T Hq*D"]:
  """Unpacks batched_rpa's raw output to (T, Hq * D) and applies sigmoid(gate).

  Matches batched_rpa wrapper.prepare_outputs + slice + transpose + reshape,
  followed by Qwen3NextAttention.forward's attn_output * sigmoid(gate).

  Args:
    config: the attention config.
    o: batched_rpa's raw output.
    gate: (T, Hq * D), or None.
    num_tokens: T; default gate's token count, else T_rpa.

  Returns:
    (T, Hq * D) bf16.
  """
  if num_tokens is None:
    num_tokens = gate.shape[0] if gate is not None else o.shape[1]
  x = rpa_wrapper.prepare_outputs(o)[
      :, :num_tokens, : config.q_per_kv, : config.head_dim
  ]
  x = x.swapaxes(1, 0).reshape(num_tokens, config.q_dim)
  if gate is not None:
    x = x * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(x.dtype)
  return x


@functools.partial(
    jax.jit,
    static_argnums=0,
    static_argnames=("num_tokens", "numerics"),
)
def out_proj_reference(
    config: model_config.AttentionConfig,
    o: BFloat16[Array, "Hkv T_rpa q_per_kv/q_packing q_packing Dp"],
    weights: model_config.OutProjWeights,
    gate: BFloat16[Array, "T Hq*D"] | None = None,
    *,
    num_tokens: int | None = None,
    numerics: Numerics = Numerics.KERNEL,
) -> BFloat16[Array, "T K"]:
  """The unfused attention output projection: fused_out_proj with `numerics`."""
  x = out_proj_input(config, o, gate, num_tokens)
  if numerics == Numerics.KERNEL and config.fp8:
    x_q, x_scale = fused_quant(config, x, out_proj_config.quant_block(config))
    return fused_project(x_q, x_scale, None, weights)
  if numerics == Numerics.UNQUANTIZED:
    x_f32 = x.astype(jnp.float32)
    if gate is not None:
      ungated = out_proj_input(config, o, None, num_tokens).astype(jnp.float32)
      x_f32 = ungated * jax.nn.sigmoid(gate.astype(jnp.float32))
    w = weights.w.astype(jnp.float32)
    if weights.w_scale is not None:
      w = w * weights.w_scale
    return project(x_f32, w, bias=weights.bias, exact=True)
  x_scale = None
  if config.fp8:
    x, x_scale = quantized_matmul.quantize_fp8_per_token(x)
  return project(x, weights.w, x_scale, weights.w_scale, weights.bias)
