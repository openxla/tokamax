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
"""Per-chip attention prologue presets for the models we serve on TPU."""

import dataclasses
import functools
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, Float, Float32  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.batched_rpa.kernel import wrapper as rpa_wrapper
from tokamax._src.ops.experimental.fused_qkv_proj import cos_sin
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import weights as gdn_weights
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import weights as gqa_weights
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import config as out_proj_config

Norm = model_config.Norm
Rope = model_config.Rope
AttentionConfig = model_config.AttentionConfig


class QkvWeights(NamedTuple):
  """Projection and norm weights as loaded, before the kernels' preparation."""

  w: jax.Array  # (K, N) fp8 or bf16.
  w_scale: jax.Array | None = None  # (1, N) f32, fp8 only.
  bias: jax.Array | None = None  # (1, N).
  q_norm_w: jax.Array | None = None  # (1, D).
  k_norm_w: jax.Array | None = None  # (1, D).
  input_norm_w: jax.Array | None = None  # (1, K).
  # GDN only: in_proj_ba (K, 2 Hv) bf16, [b | a], an unquantized projection of
  # the same normed x.
  side_w: jax.Array | None = None


def qkv_kwargs(config: AttentionConfig) -> dict[str, Any]:
  """fused_qkv's model keyword arguments for `config`."""
  rope = config.rope
  return dict(
      num_q_heads=config.num_q_heads,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
      gate=config.gate,
      v_from_k=config.v_from_k,
      input_norm=config.input_norm,
      q_norm=config.q_norm,
      k_norm=config.k_norm,
      v_norm=config.v_norm,
      rope_offset=None if rope is None else rope.offset,
      rope_precision=config.rope_precision.value,
      eps=config.eps,
      k_scale=config.k_scale,
      v_scale=config.v_scale,
      act_quant_block=config.act_quant_block,
      q_dtype=config.q_dtype,
      kv_dtype=config.kv_dtype,
  )


def out_proj_kwargs(config: AttentionConfig) -> dict[str, Any]:
  """fused_out_proj's model keyword arguments for `config`."""
  return dict(
      num_q_heads=config.num_q_heads,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
      act_quant_block=config.act_quant_block,
  )


def gdn_kwargs(config: model_config.GdnConfig) -> dict[str, Any]:
  """fused_gdn's model keyword arguments for `config`."""
  return dict(
      num_k_heads=config.num_k_heads,
      num_v_heads=config.num_v_heads,
      head_dim=config.head_dim,
      input_norm=config.input_norm,
      eps=config.eps,
      norm_dtype=config.norm_dtype,
      act_quant_block=config.act_quant_block,
  )


def prepare_gqa_weights(
    config: AttentionConfig, weights: QkvWeights
) -> model_config.GqaWeights:
  """fused_qkv's weight arguments, prepared as at weight load time."""
  assert weights.side_w is None, "side_w is GDN only"
  return model_config.GqaWeights(
      weights.w,
      weights.w_scale,
      weights.bias,
      weights.q_norm_w,
      weights.k_norm_w,
      gqa_weights.prepare_input_norm_w(
          weights.input_norm_w,
          fp8=config.fp8,
          input_norm=config.input_norm,
          act_quant_block=config.act_quant_block,
      ),
  )


def prepare_gdn_weights(
    config: model_config.GdnConfig, weights: QkvWeights
) -> model_config.GdnWeights:
  """fused_gdn's weight arguments, prepared as at weight load time."""
  assert weights.side_w is not None, "fused_gdn needs side_w (in_proj_ba)"
  ba_w, norm_w = gdn_weights.prepare_weights(
      weights.side_w,
      weights.input_norm_w,
      num_v_heads=config.num_v_heads,
      input_norm=config.input_norm,
      norm_dtype=config.norm_dtype,
      act_quant_block=config.act_quant_block,
  )
  return model_config.GdnWeights(weights.w, weights.w_scale, ba_w, norm_w)


# (x, residual, weights, cos, sin).
Inputs = tuple[
    BFloat16[Array, "T K"],
    BFloat16[Array, "T K"] | None,
    QkvWeights,
    BFloat16[Array, "T n_pairs"] | None,
    BFloat16[Array, "T n_pairs"] | None,
]
# (x, residual, prepared weights, cos, sin): fused_qkv's arguments, grouped.
PreparedInputs = tuple[
    BFloat16[Array, "T K"],
    BFloat16[Array, "T K"] | None,
    model_config.GqaWeights,
    BFloat16[Array, "T n_pairs"] | None,
    BFloat16[Array, "T n_pairs"] | None,
]
# (o, weights, gate).
OutProjInputs = tuple[
    BFloat16[Array, "Hkv T_rpa q_per_kv/q_packing q_packing Dp"],
    model_config.OutProjWeights,
    BFloat16[Array, "T Hq*D"] | None,
]


@dataclasses.dataclass(frozen=True)
class Model:
  name: str
  config: AttentionConfig
  hidden_size: int  # K.
  # positions (T,), or (3, T) with mrope -> (cos, sin), each (T, n_pairs).
  cos_sin: Callable[[jax.Array], tuple[jax.Array, jax.Array]] | None
  mrope: bool = False


# Qwen3-Coder-480B-A35B @TP8.
# Attention: third_party/py/vllm/model_executor/models/qwen3_moe.py
#   Qwen3MoeAttention
# Model config: BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic
QWEN3_CODER_TP8 = Model(
    "qwen3_coder_tp8",
    AttentionConfig(
        num_q_heads=12,
        num_kv_heads=1,
        head_dim=128,
        residual=True,
        q_norm=Norm.RMS,
        k_norm=Norm.RMS,
        rope=Rope.neox(128),
    ),
    hidden_size=6144,
    cos_sin=functools.partial(
        cos_sin.cos_sin_default, rotary_dim=128, theta=1e7
    ),
)


def _qwen3_5(tp: int) -> Model:
  """Qwen3.5's full attention layers at TP tp."""
  return Model(
      f"qwen3_5_tp{tp}",
      AttentionConfig(
          num_q_heads=32 // tp,
          num_kv_heads=max(1, 2 // tp),
          head_dim=256,
          gate=True,
          input_norm=Norm.RMS_ONE_PLUS_W,
          residual=True,
          q_norm=Norm.RMS_ONE_PLUS_W,
          k_norm=Norm.RMS_ONE_PLUS_W,
          rope=Rope.neox(64),
      ),
      hidden_size=4096,
      cos_sin=functools.partial(
          cos_sin.cos_sin_mrope_interleaved,
          rotary_dim=64,
          theta=1e7,
          section=(11, 11, 10),
      ),
      mrope=True,
  )


# Qwen3.5-397B-A17B full-attention layers.
# Attention: third_party/py/vllm/model_executor/models/qwen3_next.py
#   Qwen3NextAttention
# Model config: Qwen/Qwen3.5-397B-A17B
QWEN3_5_TP8 = _qwen3_5(8)
QWEN3_5_TP2 = _qwen3_5(2)

# gemma-4-31B-it @TP4.
# Attention: third_party/py/vllm/model_executor/models/gemma4.py Gemma4Attention
# Model config: google/gemma-4-31b
GEMMA4_SLIDING_TP4 = Model(
    "gemma4_sliding_tp4",
    AttentionConfig(
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=256,
        q_norm=Norm.RMS,
        k_norm=Norm.RMS,
        v_norm=Norm.RMS_NOSCALE,
        act_quant_block=768,  # hidden 5376 = 7 * 768.
        rope=Rope.neox(256),
    ),
    hidden_size=5376,
    cos_sin=functools.partial(
        cos_sin.cos_sin_default, rotary_dim=256, theta=1e4
    ),
)
GEMMA4_FULL_TP4 = Model(
    "gemma4_full_tp4",
    AttentionConfig(
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=512,
        v_from_k=True,
        q_norm=Norm.RMS,
        k_norm=Norm.RMS,
        v_norm=Norm.RMS_NOSCALE,
        act_quant_block=768,  # hidden 5376 = 7 * 768.
        rope=Rope(64, 256),
    ),
    hidden_size=5376,
    cos_sin=functools.partial(
        cos_sin.cos_sin_proportional,
        head_dim=512,
        partial_rotary_factor=0.25,
        theta=1e6,
    ),
)

# gpt-oss-120b @TP4.
# Attention: third_party/py/vllm/model_executor/models/gpt_oss.py OAIAttention
# Model config: openai/gpt-oss-120b
GPT_OSS_TP4 = Model(
    "gpt_oss_tp4",
    AttentionConfig(
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=64,
        fp8=False,
        bias=True,
        residual=True,
        rope=Rope.neox(64),
        eps=1e-5,
    ),
    hidden_size=2880,
    cos_sin=functools.partial(
        cos_sin.cos_sin_yarn,
        rotary_dim=64,
        theta=150000.0,
        factor=32.0,
        original_max_position=4096,
        truncate=False,
    ),
)

MODELS = (
    QWEN3_CODER_TP8,
    QWEN3_5_TP8,
    QWEN3_5_TP2,
    GEMMA4_SLIDING_TP4,
    GEMMA4_FULL_TP4,
    GPT_OSS_TP4,
)
BY_NAME = {m.name: m for m in MODELS}


def make_inputs(model: Model, t: int) -> Inputs:
  """Returns random (x, residual, weights, cos, sin) for t tokens."""
  return _make_inputs(model, t, exact=False)


def make_exact_inputs(model: Model, t: int) -> Inputs:
  """Like make_inputs, but with inputs on which floating point is exact.

  Small integers times powers of two are exact in bf16 and fp8, and their sums
  stay exact in any order while small. So the kernel must match the reference
  bit for bit, and any difference is a bug. This catches structural bugs
  (indexing, layouts, head order, packing, block scales), not rounding;
  test_accuracy covers rounding on random inputs.

  How the sums stay exact:
    - x + residual is ±1 per element, so the input norm's sum of squares is
      exactly the hidden size.
    - Each fp8 quant block has one marker column whose input-norm scale,
      56 * 2^-(block % 4), is the block's amax; its W row is zero. Every
      block's fp8 scale is then a different power of two (1/8 ... 1/64), and
      all quantized values are exact. The other scales are 1 or 2.
    - W is in {-1, 0, 1} with power-of-two W scales.

  Limits: all rows get the same scales, and the block scales repeat every 4
  blocks, so a bug that applies another row's scale, or the scale of a block
  4 away, can still pass.

  Args:
    model: the model preset.
    t: tokens.

  Returns:
    (x, residual, weights, cos, sin), as make_inputs.
  """
  return _make_inputs(model, t, exact=True)


def _make_inputs(model: Model, t: int, exact: bool) -> Inputs:
  """make_inputs, or make_exact_inputs if exact."""
  config = model.config
  k, n, d = model.hidden_size, config.num_cols, config.head_dim
  keys = jax.random.split(jax.random.key(t), 10)
  x, residual, w, w_scale, input_norm_w = _projection_inputs(
      config, t, k, n, keys, exact
  )
  bias = (
      jax.random.normal(keys[3], (1, n), jnp.bfloat16) if config.bias else None
  )
  norm_w = []
  for i, kind in enumerate((config.q_norm, config.k_norm)):
    random_w = 0.1 * jax.random.normal(keys[4 + i], (1, d), jnp.bfloat16)
    norm_w.append(
        None
        if kind in (Norm.NONE, Norm.RMS_NOSCALE)
        else (random_w if kind == Norm.RMS_ONE_PLUS_W else 1 + random_w)
    )
  weights = QkvWeights(w, w_scale, bias, *norm_w, input_norm_w=input_norm_w)
  cos = sin = None
  if config.rope is not None:
    pos = jax.random.randint(keys[6], (3, t) if model.mrope else (t,), 0, 32768)
    assert model.cos_sin is not None
    cos, sin = model.cos_sin(pos)
  return x, residual if config.residual else None, weights, cos, sin


def _projection_inputs(
    config: model_config.AttentionConfig | model_config.GdnConfig,
    t: int,
    k: int,
    n: int,
    keys: jax.Array,
    exact: bool,
) -> tuple[
    BFloat16[Array, "T K"],
    BFloat16[Array, "T K"],
    Float[Array, "K N"],
    Float32[Array, "1 N"] | None,
    BFloat16[Array, "1 K"],
]:
  """(x, residual, w, w_scale, input_norm_w); keys holds 10 PRNG keys."""
  w_dtype = quantized_matmul.F8 if config.fp8 else jnp.bfloat16
  w_scale = None

  def pow2(
      key: jax.Array,
      shape: tuple[int, ...],
      min_exponent: int,
      num_exponents: int = 2,
  ) -> Float32[Array, "..."]:
    """Random 2^e with min_exponent <= e < min_exponent + num_exponents."""
    return jnp.exp2(
        min_exponent + jax.random.randint(key, shape, 0, num_exponents)
    ).astype(jnp.float32)

  if exact:
    total = 2 * jax.random.randint(keys[9], (t, k), 0, 2) - 1  # x + residual.
    x = jax.random.randint(keys[0], (t, k), -1, 2) if config.residual else total
    residual = (total - x).astype(jnp.bfloat16)
    x = x.astype(jnp.bfloat16)
    markers = jnp.arange(0, k, config.act_quant_block if config.fp8 else k)
    marker_scale = 56 * jnp.exp2(-(jnp.arange(len(markers)) % 4))
    input_norm_w = pow2(keys[8], (1, k), 0).at[0, markers].set(marker_scale)
    w = jax.random.randint(keys[1], (k, n), -1, 2).at[markers].set(0)
    w = w.astype(w_dtype)
    if config.fp8:
      w_scale = pow2(keys[3], (1, n), -7)
  else:
    x = jax.random.normal(keys[0], (t, k), jnp.bfloat16)
    residual = jax.random.normal(keys[7], (t, k), jnp.bfloat16)
    input_norm_w = 1 + 0.1 * jax.random.normal(keys[8], (1, k), jnp.float32)
    w = jax.random.normal(keys[1], (k, n), jnp.float32)
    if config.fp8:
      w = w.astype(quantized_matmul.F8)
      w_scale = jax.random.uniform(keys[2], (1, n), jnp.float32, 0.01, 0.02)
    else:
      w = (w * k**-0.5).astype(jnp.bfloat16)
  if config.input_norm == Norm.RMS_ONE_PLUS_W:
    input_norm_w = input_norm_w - 1  # Scales by 1 + w.
  input_norm_w = input_norm_w.astype(jnp.bfloat16)
  return x, residual, w, w_scale, input_norm_w


# GDN prologues.


@dataclasses.dataclass(frozen=True)
class GdnModel:
  name: str
  config: model_config.GdnConfig
  hidden_size: int  # K.


# Qwen3.5-397B-A17B linear-attention layers @TP8.
# Layer: vllm-torchtpu Qwen3_5GDNAttention -> VllmGatedDeltaNetAttention.
# Model config: Qwen/Qwen3.5-397B-A17B
QWEN3_5_GDN_TP8 = GdnModel(
    "qwen3_5_gdn_tp8",
    model_config.GdnConfig(num_k_heads=16 // 8, num_v_heads=64 // 8),
    hidden_size=4096,
)

GDN_MODELS = (QWEN3_5_GDN_TP8,)
GDN_BY_NAME = {m.name: m for m in GDN_MODELS}


def make_gdn_inputs(model: GdnModel, t: int, exact: bool = False) -> tuple[
    BFloat16[Array, "T K"],
    BFloat16[Array, "T K"] | None,
    QkvWeights,
]:
  """Random GDN prologue inputs for t tokens.

  With exact, inputs are as in make_exact_inputs, and side_w is in {-1, 0, 1}
  so the side projection is exact too.

  Args:
    model: the model preset.
    t: tokens.
    exact: whether to make exact inputs.

  Returns:
    (x, residual, weights); residual is None without a residual add.
  """
  config = model.config
  k, n = model.hidden_size, config.num_cols
  keys = jax.random.split(jax.random.key(t), 10)
  x, residual, w, w_scale, input_norm_w = _projection_inputs(
      config, t, k, n, keys, exact
  )
  side_key = jax.random.fold_in(keys[4], 1)
  if exact:
    markers = jnp.arange(0, k, config.act_quant_block)
    side_w = jax.random.randint(side_key, (k, config.side_cols), -1, 2)
    side_w = side_w.at[markers].set(0).astype(jnp.bfloat16)
  else:
    side_w = jax.random.normal(side_key, (k, config.side_cols), jnp.float32)
    side_w = (side_w * k**-0.5).astype(jnp.bfloat16)
  weights = QkvWeights(w, w_scale, input_norm_w=input_norm_w, side_w=side_w)
  return x, residual if config.residual else None, weights


# Attention output projection (o_proj).


def make_out_proj_inputs(
    model: Model,
    t: int,
    *,
    exact: bool = False,
    t_rpa: int | None = None,
) -> OutProjInputs:
  """Random fused_out_proj inputs for t tokens.

  With exact, o is small powers of two with one marker column per fp8 quant
  block (whose W row is zero), gate is 0 so sigmoid(gate) = 1/2, and W is in
  {-1, 0, 1} with power-of-two scales, so every operation is exact.

  Args:
    model: the model preset.
    t: tokens.
    exact: whether to make exact inputs.
    t_rpa: o's token count, >= t; default t.

  Returns:
    (o, weights, gate); gate is None without an output gate.
  """
  config = model.config
  t_rpa = t if t_rpa is None else t_rpa
  hq, hkv, d = config.num_q_heads, config.num_kv_heads, config.head_dim
  keys = jax.random.split(jax.random.key(t), 6)
  make_values = _exact_out_proj_values if exact else _random_out_proj_values
  q, gate, weights = make_values(config, model.hidden_size, t, t_rpa, keys)
  dummy_kv = jnp.zeros((t_rpa, hkv, d), config.kv_dtype)
  o, _, _ = rpa_wrapper.prepare_inputs(
      q.reshape(t_rpa, hq, d),
      dummy_kv,
      dummy_kv,
      jnp.dtype(config.q_dtype),
      jnp.dtype(config.kv_dtype),
  )
  return o, weights, gate


def _exact_out_proj_values(
    config: model_config.AttentionConfig,
    k: int,
    t: int,
    t_rpa: int,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array | None, model_config.OutProjWeights]:
  """make_out_proj_inputs' exact (q, gate, weights); see its docstring."""
  q_dim = config.q_dim
  q = jnp.exp2(jax.random.randint(keys[0], (t_rpa, q_dim), 0, 2))
  q = q * (2 * jax.random.randint(keys[1], (t_rpa, q_dim), 0, 2) - 1)
  quant_block = out_proj_config.quant_block(config) if config.fp8 else q_dim
  markers = jnp.arange(0, q_dim, quant_block)
  marker_scale = 56 * jnp.exp2(-(jnp.arange(len(markers)) % 4))
  q = q.at[:, markers].set(marker_scale).astype(jnp.bfloat16)
  gate = jnp.zeros((t, q_dim), jnp.bfloat16) if config.gate else None
  w_dtype = quantized_matmul.F8 if config.fp8 else jnp.bfloat16
  w = jax.random.randint(keys[2], (q_dim, k), -1, 2).at[markers].set(0)
  w = w.astype(w_dtype)
  w_scale = None
  if config.fp8:
    w_scale = jnp.exp2(-7 + jax.random.randint(keys[3], (1, k), 0, 2)).astype(
        jnp.float32
    )
  bias = None
  if config.bias:
    bias = jax.random.randint(keys[4], (1, k), -1, 2).astype(jnp.bfloat16)
  return q, gate, model_config.OutProjWeights(w, w_scale, bias)


def _random_out_proj_values(
    config: model_config.AttentionConfig,
    k: int,
    t: int,
    t_rpa: int,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array | None, model_config.OutProjWeights]:
  """make_out_proj_inputs' random normal (q, gate, weights)."""
  q_dim = config.q_dim
  q = jax.random.normal(keys[0], (t_rpa, q_dim), jnp.bfloat16)
  gate = None
  if config.gate:
    gate = jax.random.normal(keys[1], (t, q_dim), jnp.bfloat16)
  w = jax.random.normal(keys[2], (q_dim, k), jnp.float32)
  w_scale = None
  if config.fp8:
    w = w.astype(quantized_matmul.F8)
    w_scale = jax.random.uniform(keys[3], (1, k), jnp.float32, 0.01, 0.02)
  else:
    w = (w * q_dim**-0.5).astype(jnp.bfloat16)
  bias = None
  if config.bias:
    bias = jax.random.normal(keys[4], (1, k), jnp.bfloat16)
  return q, gate, model_config.OutProjWeights(w, w_scale, bias)
