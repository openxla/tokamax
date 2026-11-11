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
"""Standalone Pallas Mosaic TPU ragged causal 1D convolution operator."""

# pylint: disable=g-importing-member,useless-import-alias
from tokamax._src.ops.experimental.ragged_causal_conv1d.causal_conv1d import preprocess_conv1d_metadata as preprocess_conv1d_metadata
from tokamax._src.ops.experimental.ragged_causal_conv1d.causal_conv1d import ragged_causal_conv1d as ragged_causal_conv1d
from tokamax._src.ops.experimental.ragged_causal_conv1d.causal_conv1d import ragged_causal_conv1d_jax as ragged_causal_conv1d_jax

# pylint: enable=g-importing-member,useless-import-alias

__all__ = (
    "preprocess_conv1d_metadata",
    "ragged_causal_conv1d",
    "ragged_causal_conv1d_jax",
)
