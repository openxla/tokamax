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
"""Context Parallel (CP) schedulers expressed via plain-data `PageOwnership`."""

from collections.abc import Sequence
import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import paging
from tokamax._src.ops.batched_rpa.kernel import scheduler
from tokamax._src.ops.batched_rpa.kernel import utils


@dataclasses.dataclass(frozen=True)
class CPScheduler(scheduler.BaseScheduler):
  """Base class for page-striped Context Parallel (CP) schedulers.

  Expects `extra_scalars=(cp_rank,)` where `cp_rank` is an `int32[1]` array
  giving the rank of the current device in the CP group.

  Attributes:
    cp_group_size: Number of devices in the CP group.
  """

  cp_group_size: int

  def get_cp_group_size(self) -> int:
    return self.cp_group_size

  def _get_cp_rank(
      self, extra: Sequence[jax.Array] | Sequence[jax.Ref]
  ) -> jax.Array:
    if not extra:
      raise ValueError(
          f"{type(self).__name__} requires `extra_scalars=(cp_rank,)`."
      )
    return extra[0][0]


@dataclasses.dataclass(frozen=True)
class StripedCacheScheduler(CPScheduler):
  """Attends only to this rank's pre-compacted local KV cache pages."""

  scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.CACHE_ONLY

  def get_kv_lens(
      self,
      kv_lens: jax.Array,
      q_lens: jax.Array,
      page_size: int = 1,
      extra_scalars: Sequence[jax.Array] = (),
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    cp_rank = self._get_cp_rank(extra_scalars)
    global_cache_lens = kv_lens - q_lens
    local_cache_lens = utils.cp_local_cache_len(
        global_cache_lens, self.cp_group_size, cp_rank, page_size
    )
    return local_cache_lens, jnp.zeros_like(q_lens), local_cache_lens


@dataclasses.dataclass(frozen=True)
class StripedNewTokensScheduler(CPScheduler):
  """Attends to new tokens and writes back only pages owned by `cp_rank`."""

  scope: ClassVar[configs.AttentionScope] = (
      configs.AttentionScope.NEW_TOKENS_ONLY
  )

  def page_ownership(
      self, extra_refs: Sequence[jax.Ref] = ()
  ) -> paging.PageOwnership:
    return paging.PageOwnership(
        group_size=self.cp_group_size,
        rank=self._get_cp_rank(extra_refs),
    )
