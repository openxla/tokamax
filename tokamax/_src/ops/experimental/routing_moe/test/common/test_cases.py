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
"""MoE test cases shared across gmm, io_relayout, and layer."""

import dataclasses

from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes


@dataclasses.dataclass(frozen=True)
class Case:
  """One MoE problem combining a layer shape, token count, and shard count.

  Attributes:
    name: Short identifier for the test case.
    shape: Static MoE model dimensions and quantization settings.
    num_tokens: Number of input tokens.
    ep_size: Number of expert-parallel shards.
    active_expert_ratio: Fraction of experts that receive routing weight.
  """

  name: str
  shape: test_shapes.MoEShape
  num_tokens: int
  ep_size: int = 1
  active_expert_ratio: float = 1.0

  @property
  def num_local_experts(self) -> int:
    """Number of experts per EP shard."""
    return self.shape.num_experts // self.ep_size

  @property
  def num_rows(self) -> int:
    """Routed rows, i.e. `(token, pick)` pairs."""
    return self.num_tokens * self.shape.top_k


BASIC_256 = Case("basic_256", test_shapes.BASIC, 256)
PREFILL_64K = Case("prefill_64k", test_shapes.GMM_PREFILL, 65536, ep_size=8)
PREFILL_8K = Case("prefill_8k", test_shapes.GMM_PREFILL, 8192, ep_size=8)
PREFILL_16E = Case("prefill_16e", test_shapes.GMM_FEW_EXPERTS, 4096)
TOP1_200 = Case("top1_200", test_shapes.GMM_TOP1, 200)
DECODE_1 = Case("decode_1", test_shapes.GMM_DECODE, 1)
DECODE_16 = Case("decode_16", test_shapes.GMM_DECODE, 16)
DECODE_1_EP8 = Case("decode_1_ep8", test_shapes.GMM_DECODE, 1, ep_size=8)
DECODE_64_EP8 = Case("decode_64_ep8", test_shapes.GMM_DECODE, 64, ep_size=8)

PREFILL_CASES = (PREFILL_64K, PREFILL_8K, TOP1_200)
DECODE_CASES = (DECODE_16, DECODE_64_EP8)
UNIT_CASES = (BASIC_256, *PREFILL_CASES, *DECODE_CASES)

PROD_CASES = (
    Case("qwen_235_512", test_shapes.QWEN_235, 512),
    Case("qwen_30b_a3b_512", test_shapes.QWEN_30B_A3B, 512),
    Case(
        "qwen_coder_480b_64",
        test_shapes.QWEN_CODER_480B,
        64,
        ep_size=8,
        active_expert_ratio=0.6,
    ),
    Case(
        "qwen_coder_480b_8k",
        test_shapes.QWEN_CODER_480B,
        8192,
        ep_size=8,
        active_expert_ratio=0.6,
    ),
    Case(
        "mistral_3_large_2k",
        test_shapes.MISTRAL_3_LARGE,
        2048,
        ep_size=8,
        active_expert_ratio=0.6,
    ),
    Case(
        "mistral_3_large_4k",
        test_shapes.MISTRAL_3_LARGE,
        4096,
        ep_size=8,
        active_expert_ratio=0.6,
    ),
    Case(
        "mistral_3_large_32k",
        test_shapes.MISTRAL_3_LARGE,
        32768,
        ep_size=8,
        active_expert_ratio=0.6,
    ),
    Case("qwen_35_397b_8", test_shapes.QWEN_35_397B, 8, ep_size=8),
    Case("qwen_35_397b_32", test_shapes.QWEN_35_397B, 32, ep_size=8),
    Case("qwen_35_397b_256", test_shapes.QWEN_35_397B, 256, ep_size=8),
    Case("qwen_35_397b_8k", test_shapes.QWEN_35_397B, 8192, ep_size=8),
    Case("gpt_oss_120_64", test_shapes.GPT_OSS_120, 64, ep_size=2),
)
