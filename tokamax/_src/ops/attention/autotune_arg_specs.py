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

"""Attention args for autotuning."""

from typing import Any
from typing import Final
from jax import ShapeDtypeStruct  # pylint: disable=g-importing-member
from jax.lax import DotAlgorithmPreset  # pylint: disable=g-importing-member
import jax.numpy as jnp
from tokamax._src.autotuning import arg_specs_common as common
from tokamax._src.ops.attention import bench_arg_specs
from tokamax._src.ops.attention.base import Mask  # pylint: disable=g-importing-member


ARGS: Final[tuple[common.ArgSpec, ...]] = (
    common.ArgSpec(
        args={
            'q': ShapeDtypeStruct(shape=(384, 384, 4, 32), dtype=jnp.bfloat16),
            'k': ShapeDtypeStruct(shape=(384, 384, 4, 32), dtype=jnp.bfloat16),
            'v': ShapeDtypeStruct(shape=(384, 384, 4, 32), dtype=jnp.bfloat16),
            'precision': (
                DotAlgorithmPreset.BF16_BF16_F32,
                DotAlgorithmPreset.BF16_BF16_F32,
            ),
            'logits_scale': 0.1767766952966369,
            'bias': ShapeDtypeStruct(
                shape=(1, 4, 384, 384), dtype=jnp.bfloat16
            ),
            'mask': Mask(
                bool_mask=ShapeDtypeStruct(shape=(384, 1, 1, 384), dtype=bool),
            ),
        },
        project='alphafold',
    ),
    common.ArgSpec(
        args={
            'q': ShapeDtypeStruct(shape=(384, 384, 4, 64), dtype=jnp.bfloat16),
            'k': ShapeDtypeStruct(shape=(384, 384, 4, 64), dtype=jnp.bfloat16),
            'v': ShapeDtypeStruct(shape=(384, 384, 4, 64), dtype=jnp.bfloat16),
            'precision': (
                DotAlgorithmPreset.BF16_BF16_F32,
                DotAlgorithmPreset.BF16_BF16_F32,
            ),
            'logits_scale': 0.1767766952966369,
            'bias': ShapeDtypeStruct(
                shape=(1, 4, 384, 384), dtype=jnp.bfloat16
            ),
            'mask': Mask(
                bool_mask=ShapeDtypeStruct(shape=(384, 1, 1, 384), dtype=bool),
            ),
        },
        project='alphafold',
    ),
    common.ArgSpec(
        args={
            'q': ShapeDtypeStruct(shape=(768, 768, 4, 64), dtype=jnp.bfloat16),
            'k': ShapeDtypeStruct(shape=(768, 768, 4, 64), dtype=jnp.bfloat16),
            'v': ShapeDtypeStruct(shape=(768, 768, 4, 64), dtype=jnp.bfloat16),
            'precision': (
                DotAlgorithmPreset.BF16_BF16_F32,
                DotAlgorithmPreset.BF16_BF16_F32,
            ),
            'logits_scale': 0.1767766952966369,
            'bias': ShapeDtypeStruct(
                shape=(1, 4, 768, 768), dtype=jnp.bfloat16
            ),
            'mask': Mask(
                bool_mask=ShapeDtypeStruct(shape=(768, 1, 1, 768), dtype=bool),
            ),
        },
        project='alphafold',
    ),
) + bench_arg_specs.ARG_SPECS
