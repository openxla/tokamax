# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""A library of accelerator kernels."""

# pylint: disable=g-importing-member,useless-import-alias
from tokamax import autotuning as autotuning
from tokamax import benchmarking as benchmarking
from tokamax import config as config
from tokamax._src.autotuning.api import autotune as autotune
from tokamax._src.autotuning.api import AutotuningResult as AutotuningResult
from tokamax._src.batching import BatchedShapeDtype as BatchedShapeDtype
from tokamax._src.hlo_utils import DISABLE_JAX_EXPORT_CHECKS as DISABLE_JAX_EXPORT_CHECKS
from tokamax._src.ops.attention.api import dot_product_attention as dot_product_attention
from tokamax._src.ops.gated_linear_unit.api import gated_linear_unit as gated_linear_unit
from tokamax._src.ops.normalization.api import layer_norm as layer_norm
from tokamax._src.ops.op import Op as Op
from tokamax._src.ops.ragged_dot.api import ragged_dot as ragged_dot
from tokamax._src.quantization import QuantizedArray as QuantizedArray
# pylint: enable=g-importing-member,useless-import-alias
