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
import itertools
import math
from typing import NamedTuple

from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes

DP_AXIS = "data"
EP_AXIS = "ep"
TP_AXIS = "tp"
MESH_AXES = (DP_AXIS, EP_AXIS, TP_AXIS)


class MoESharding(NamedTuple):
  """Data-, expert-, and tensor-parallel shard counts."""

  dp: int = 1
  ep: int = 1
  tp: int = 1

  @property
  def num_cores(self) -> int:
    """Total number of devices across DP, EP, and TP."""
    return self.dp * self.ep * self.tp

  @property
  def ep_axis(self) -> str | None:
    """Mesh axis for expert parallelism, or `None` when `ep == 1`."""
    return EP_AXIS if self.ep > 1 else None

  @property
  def tp_axis(self) -> str | None:
    """Mesh axis for tensor parallelism, or `None` when `tp == 1`."""
    return TP_AXIS if self.tp > 1 else None

  @property
  def reduction_axis(self) -> tuple[str, ...]:
    """Mesh axes over which partial MoE outputs are summed."""
    return tuple(ax for ax in (self.ep_axis, self.tp_axis) if ax is not None)


@dataclasses.dataclass(frozen=True)
class Case:
  """One MoE problem combining a layer shape, token count, and sharding.

  Attributes:
    name: Short identifier for the test case.
    shape: Static MoE model dimensions and quantization settings.
    num_tokens: Number of input tokens.
    sharding: Expert- and tensor-parallel shard counts.
    active_expert_ratio: Fraction of experts that receive routing weight.
  """

  name: str
  shape: test_shapes.MoEShape
  num_tokens: int
  sharding: MoESharding = MoESharding()
  active_expert_ratio: float = 1.0

  @property
  def num_local_experts(self) -> int:
    """Number of experts per EP shard."""
    return self.shape.num_experts // self.sharding.ep

  @property
  def local_intermediate_size(self) -> int:
    """Intermediate dimension per TP shard, aligned to the 128-lane width."""
    lane_size = 128
    per_shard = self.shape.intermediate_size // self.sharding.tp
    return math.ceil(per_shard / lane_size) * lane_size

  @property
  def num_rows(self) -> int:
    """Routed rows, i.e. `(token, pick)` pairs."""
    return self.num_tokens * self.shape.top_k


def _prod_sweep(
    shape: test_shapes.MoEShape,
    token_counts: tuple[int, ...],
    shardings: tuple[MoESharding, ...],
    *,
    active_expert_ratio: float = 1.0,
) -> tuple[Case, ...]:
  """Expands `shape` across `token_counts` and `shardings`."""
  return tuple(
      Case(
          f"{shape.name}_{n}_ep{s.ep}_tp{s.tp}",
          shape,
          n,
          sharding=s,
          active_expert_ratio=active_expert_ratio,
      )
      for n, s in itertools.product(token_counts, shardings)
  )


BASIC_256 = Case("basic_256", test_shapes.BASIC, 256)
PREFILL_64K = Case(
    "prefill_64k", test_shapes.GMM_PREFILL, 65536, MoESharding(ep=8)
)
PREFILL_8K = Case(
    "prefill_8k", test_shapes.GMM_PREFILL, 8192, MoESharding(ep=8)
)
PREFILL_16E = Case("prefill_16e", test_shapes.GMM_FEW_EXPERTS, 4096)
TOP1_200 = Case("top1_200", test_shapes.GMM_TOP1, 200)
DECODE_1 = Case("decode_1", test_shapes.GMM_DECODE, 1)
DECODE_16 = Case("decode_16", test_shapes.GMM_DECODE, 16)
DECODE_1_EP8 = Case(
    "decode_1_ep8", test_shapes.GMM_DECODE, 1, MoESharding(ep=8)
)
DECODE_64_EP8 = Case(
    "decode_64_ep8", test_shapes.GMM_DECODE, 64, MoESharding(ep=8)
)

PREFILL_CASES = (PREFILL_64K, PREFILL_8K, TOP1_200)
DECODE_CASES = (DECODE_16, DECODE_64_EP8)
UNIT_CASES = (BASIC_256, *PREFILL_CASES, *DECODE_CASES)

PROD_CASES = (
    *_prod_sweep(
        test_shapes.QWEN_235,
        (512,),
        shardings=(MoESharding(ep=8), MoESharding(tp=8)),
    ),
    *_prod_sweep(
        test_shapes.QWEN_30B_A3B,
        (512,),
        shardings=(MoESharding(ep=8), MoESharding(tp=8)),
    ),
    *_prod_sweep(
        test_shapes.QWEN_CODER_480B,
        (64, 8192),
        shardings=(MoESharding(ep=8), MoESharding(tp=8)),
        active_expert_ratio=0.6,
    ),
    *_prod_sweep(
        test_shapes.MISTRAL_3_LARGE,
        (2048, 4096, 32768),
        shardings=(MoESharding(ep=8), MoESharding(tp=8)),
        active_expert_ratio=0.6,
    ),
    *_prod_sweep(
        test_shapes.QWEN_35_397B,
        (8, 32, 256, 8192),
        shardings=(
            MoESharding(ep=8),
            MoESharding(ep=4, tp=2),
            MoESharding(tp=8),
        ),
    ),
    *_prod_sweep(
        test_shapes.GPT_OSS_120,
        (64,),
        shardings=(MoESharding(ep=2), MoESharding(tp=2)),
    ),
)
