# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tokamax operations public namespace."""

import importlib
from typing import Any

# pylint: disable=g-importing-member,useless-import-alias
from tokamax._src.ops.attention.api import dot_product_attention as dot_product_attention
from tokamax._src.ops.attention.api import Implementation as DotProductAttentionImplementation
from tokamax._src.ops.csa_gather.api import csa_gather as csa_gather
from tokamax._src.ops.deepseek_rope.api import deepseek_rope as deepseek_rope
from tokamax._src.ops.dsv4_compressor.api import dsv4_compressor as dsv4_compressor
from tokamax._src.ops.gated_linear_unit.api import gated_linear_unit as gated_linear_unit
from tokamax._src.ops.grouped_lowrank_proj.api import grouped_lowrank_proj as grouped_lowrank_proj
from tokamax._src.ops.kda.api import kimi_delta_attention as kimi_delta_attention
from tokamax._src.ops.linear_softmax_cross_entropy_loss.api import linear_softmax_cross_entropy_loss as linear_softmax_cross_entropy_loss
from tokamax._src.ops.masked_dense_mla.api import masked_dense_mla as masked_dense_mla
from tokamax._src.ops.mhc.api import mhc_fused_post_pre as mhc_fused_post_pre
from tokamax._src.ops.mhc.api import mhc_post as mhc_post
from tokamax._src.ops.mhc.api import mhc_pre as mhc_pre
from tokamax._src.ops.normalization.api import layer_norm as layer_norm
from tokamax._src.ops.proj_and_save_state.api import proj_and_save_state as proj_and_save_state
from tokamax._src.ops.ragged_dot.api import ragged_dot as ragged_dot
from tokamax._src.ops.ragged_dot.api import ragged_dot_general as ragged_dot_general
from tokamax._src.ops.ragged_gather.api import ragged_gather as ragged_gather
from tokamax._src.ops.ragged_gather_reduce.api import ragged_gather_reduce as ragged_gather_reduce
from tokamax._src.ops.router_topk.api import router_topk as router_topk
from tokamax._src.ops.sparse_mla.api import sparse_mla as sparse_mla
from tokamax._src.ops.topk.api import top_k as top_k
from tokamax._src.ops.triangle_multiplication.api import triangle_multiplication as triangle_multiplication
from tokamax.ops import experimental as experimental

try:
  from tokamax._src.ops.batched_rpa.api import batched_ragged_paged_attention as batched_ragged_paged_attention  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
  from tokamax._src.ops.batched_rpa.api import ragged_paged_attention as ragged_paged_attention  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
  from tokamax._src.ops.lightning_indexer.api import lightning_indexer as lightning_indexer  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
except ImportError:
  pass
# pylint: enable=g-importing-member,useless-import-alias

# pylint: disable=undefined-all-variable
__all__ = (
    "DotProductAttentionImplementation",
    "batched_ragged_paged_attention",
    "csa_gather",
    "deepseek_rope",
    "dot_product_attention",
    "dsv4_compressor",
    "experimental",
    "gated_linear_unit",
    "gpu",
    "grouped_lowrank_proj",
    "kimi_delta_attention",
    "layer_norm",
    "lightning_indexer",
    "linear_softmax_cross_entropy_loss",
    "masked_dense_mla",
    "mhc_fused_post_pre",
    "mhc_post",
    "mhc_pre",
    "proj_and_save_state",
    "ragged_dot",
    "ragged_dot_general",
    "ragged_gather",
    "ragged_gather_reduce",
    "ragged_paged_attention",
    "router_topk",
    "sparse_mla",
    "top_k",
    "tpu",
    "triangle_multiplication",
)
# pylint: enable=undefined-all-variable


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
  if name in ("tpu", "gpu"):
    return importlib.import_module(f"{__name__}.{name}")
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
