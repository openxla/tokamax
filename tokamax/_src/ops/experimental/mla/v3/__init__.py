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

"""Batched Multi-Head Latent Attention (MLA) v3 for TPU."""

from tokamax._src.ops.experimental.mla.v3 import bref_override
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import flash_attention
from tokamax._src.ops.experimental.mla.v3 import kernel
from tokamax._src.ops.experimental.mla.v3 import mla_wrapper
from tokamax._src.ops.experimental.mla.v3 import schedule
from tokamax._src.ops.experimental.mla.v3 import utils

BlockSizes = configs.BlockSizes
MlaModelConfigs = configs.MlaModelConfigs
ServingConfigs = configs.ServingConfigs
KVLayout = configs.KVLayout
MlaCase = configs.MlaCase
MlaConfigs = configs.MlaConfigs

MlaSchedule = schedule.MlaSchedule
generate_mla_metadata = schedule.generate_mla_metadata

flash_attention_qk_softmax = flash_attention.flash_attention_qk_softmax
flash_attention_pv = flash_attention.flash_attention_pv
chunked_flash_attention = flash_attention.chunked_flash_attention

KVBufferedRefSeqAlongLane = bref_override.KVBufferedRefSeqAlongLane
BatchingQRef = bref_override.BatchingQRef
BatchingORef = bref_override.BatchingORef

mla_ragged_paged_attention = mla_wrapper.mla_ragged_paged_attention
mode_configs = mla_wrapper.mode_configs
build_schedules = mla_wrapper.build_schedules
static_validate_inputs = kernel.static_validate_inputs
prepare_q_nope_inputs = kernel.prepare_q_nope_inputs
prepare_q_pe_inputs = kernel.prepare_q_pe_inputs
prepare_kv_inputs_for_transposed_kv_cache = (
    kernel.prepare_kv_inputs_for_transposed_kv_cache
)
prepare_outputs = kernel.prepare_outputs
transpose_kv_cache_to_seq_along_lane = (
    utils.transpose_kv_cache_to_seq_along_lane
)
transpose_kv_cache_from_seq_along_lane = (
    utils.transpose_kv_cache_from_seq_along_lane
)
