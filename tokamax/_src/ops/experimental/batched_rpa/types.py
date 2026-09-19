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
"""Data types, enums, and configurations for Batched Ragged Paged Attention."""

import dataclasses
import enum
import jax


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tuning parameters for the RPA kernel.

  Attributes:
    bq_sz: Query block size.
    bq_c_sz: Query chunk block size for prefill.
    bkv_sz: Key/Value block size.
    batch_size: Batch size.
    n_buffer: Number of buffers for pipelining.
  """

  bq_sz: int
  bq_c_sz: int
  bkv_sz: int
  batch_size: int
  n_buffer: int


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
  """Model config that will always stay constant.

  Attributes:
    num_q_heads: Number of query attention heads.
    num_kv_heads: Number of key/value attention heads.
    head_dim: Dimensionality of each attention head.
    mask_value: Attention mask value for masked positions.
    sm_scale: Softmax scaling factor.
    soft_cap: Soft-capping threshold for attention logits.
    sliding_window: Size of the sliding attention window.
  """

  num_q_heads: int
  num_kv_heads: int
  head_dim: int
  mask_value: float
  sm_scale: float = 1.0
  soft_cap: float | None = None
  sliding_window: int | None = None

  @property
  def num_q_heads_per_kv_head(self) -> int:
    return self.num_q_heads // self.num_kv_heads


class AttentionScope(enum.StrEnum):
  """Which KV positions to attend to.

  FULL:            attend all positions (default).
  CACHE_ONLY:      attend only cached tokens, skip new tokens.
  NEW_TOKENS_ONLY: attend only new tokens, skip cached tokens.
  """

  FULL = enum.auto()
  CACHE_ONLY = enum.auto()
  NEW_TOKENS_ONLY = enum.auto()


class KVLayout(enum.StrEnum):
  """Represents the different layouts for KV cache.

  - HEAD_ALONG_SUBLANE: Number of heads on sublane, head_dim on lane.
  - SEQ_ALONG_LANE: Sequence is packed along the lane, head_dim on sublane.
  """

  HEAD_ALONG_SUBLANE = enum.auto()
  SEQ_ALONG_LANE = enum.auto()

  @property
  def symbol(self) -> str:
    match self:
      case KVLayout.HEAD_ALONG_SUBLANE:
        return "nhs"
      case KVLayout.SEQ_ALONG_LANE:
        return "snh"


class RpaCase(enum.StrEnum):
  """Represents the different cases for Ragged Paged Attention.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

  DECODE = enum.auto()
  PREFILL = enum.auto()
  MIXED = enum.auto()

  @property
  def symbol(self) -> str:
    match self:
      case RpaCase.DECODE:
        return "d"
      case RpaCase.PREFILL:
        return "p"
      case RpaCase.MIXED:
        return "m"

  def get_range(
      self, distribution: jax.Array
  ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
    assert distribution.shape == (3,)
    match self:
      case RpaCase.DECODE:
        return 0, distribution[0]
      case RpaCase.PREFILL:
        return distribution[0], distribution[1]
      case RpaCase.MIXED:
        return distribution[1], distribution[2]
