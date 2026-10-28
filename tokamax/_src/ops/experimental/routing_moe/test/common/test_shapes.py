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
"""Static MoE layer shapes shared across routing_moe tests."""

import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class MoEShape:
  """Static description of one MoE layer, independent of the token count.

  Attributes:
    name: Short identifier for the shape.
    num_experts: Total number of routed experts.
    top_k: Number of routed experts selected per token.
    hidden_size: Token activation dimension.
    intermediate_size: Per-expert MLP intermediate dimension.
    activation: Gated activation function name.
    renormalize: Whether top-k routing weights are renormalized to sum to 1.
    w_dtype: Weight quantization dtype, or None for unquantized bfloat16.
    quant_block_size: Sub-channel quantization block size, or None for
      per-channel quantization.
    has_bias: Whether expert projections include bias terms.
    has_shared_expert: Whether the layer includes a shared expert.
  """

  name: str
  num_experts: int
  top_k: int
  hidden_size: int
  intermediate_size: int
  activation: str = "silu"
  renormalize: bool = True
  w_dtype: jnp.dtype | None = None
  quant_block_size: int | None = None
  has_bias: bool = False
  has_shared_expert: bool = False


BASIC = MoEShape("basic", 128, 8, 1024, 1024)

GMM_PREFILL = MoEShape("gmm_prefill", 512, 10, 4096, 1024)
GMM_TOP1 = MoEShape("gmm_top1", 1, 1, 4096, 1024)
GMM_DECODE = MoEShape("gmm_decode", 512, 10, 512, 512)
GMM_FEW_EXPERTS = MoEShape("gmm_few_experts", 16, 8, 512, 512)
# EP=4 layer used to compare local and remote scatter.
REMOTE_SCATTER = MoEShape("remote_scatter", 256, 8, 4096, 1024)

QWEN_235 = MoEShape("qwen_235", 128, 8, 4096, 1536)
QWEN_30B_A3B = MoEShape("qwen_30b_a3b", 128, 8, 2048, 768)
QWEN_CODER_480B = MoEShape(
    "qwen_coder_480b",
    160,
    8,
    6144,
    2560,
    w_dtype=jnp.dtype(jnp.float8_e4m3fn),
)
MISTRAL_3_LARGE = MoEShape(
    "mistral_3_large",
    128,
    4,
    7168,
    4096,
    w_dtype=jnp.dtype(jnp.float8_e4m3fn),
)
QWEN_35_397B = MoEShape(
    "qwen_35_397b",
    512,
    10,
    4096,
    1024,
    w_dtype=jnp.dtype(jnp.float8_e4m3fn),
    has_shared_expert=True,
)
# The intermediate size is 640, padded to the next multiple of 256 until
# `test_util.check_moe_supported` allows it.
QWEN_38_FLASH_NEXT = MoEShape(
    "qwen_38_flash_next",
    512,
    10,
    2560,
    768,
    w_dtype=jnp.dtype(jnp.float8_e4m3fn),
    has_shared_expert=True,
)
GPT_OSS_120 = MoEShape(
    "gpt_oss_120",
    128,
    4,
    3072,
    3072,
    activation="swigluoai",
    w_dtype=jnp.dtype(jnp.float4_e2m1fn),
    quant_block_size=512,
    has_bias=True,
)
