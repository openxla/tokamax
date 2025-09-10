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
"""Attention benchmark argument specifications."""

import jax
import jax.numpy as jnp


TEST_SPECS_DICT = dict(
    # TODO: Add more dtypes.
    mixtral_8x7b_bf16=dict(
        q=jax.ShapeDtypeStruct((32, 4096, 32, 128), jnp.bfloat16),
        k=jax.ShapeDtypeStruct((32, 4096, 8, 128), jnp.bfloat16),
        v=jax.ShapeDtypeStruct((32, 4096, 8, 128), jnp.bfloat16),
        is_causal=True,
    ),
    deepseek2_16b_bf16=dict(
        q=jax.ShapeDtypeStruct((512, 1024, 16, 192), jnp.bfloat16),
        k=jax.ShapeDtypeStruct((512, 1024, 16, 192), jnp.bfloat16),
        v=jax.ShapeDtypeStruct((512, 1024, 16, 128), jnp.bfloat16),
        is_causal=True,
    ),
)

ARG_SPECS = {k: lambda v=v: v.copy() for k, v in TEST_SPECS_DICT.items()}
