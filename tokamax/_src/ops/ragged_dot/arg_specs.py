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
"""Ragged dot benchmark argument specifications."""

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.ragged_dot import base

SPEC_SHAPES = {
    'compute_bound': (
        8,
        4096,
        4096,
        4096,
        jnp.bfloat16,
        jnp.bfloat16,
        [4096] + [0] * 7,
    ),
    'memory_bound': (8, 8, 4096, 4096, jnp.bfloat16, jnp.bfloat16),
    # FIXME: Use correct dtypes.
    '8x7b': (8, 8192, 14336, 4096, jnp.bfloat16, jnp.bfloat16, None, 'mixtral'),
}


def _generate_group_sizes(target_m: int, g: int) -> tuple[int, ...]:
  """Generate group sizes for a given target m."""
  np.random.seed(0)
  repr_val = np.random.uniform(size=(g,))
  repr_val = np.random.binomial(1, 0.9, (g,)) * repr_val
  repr_val = np.int32((repr_val / np.sum(repr_val)) * target_m)
  repr_val[0] += target_m - np.sum(repr_val)
  return tuple(map(int, repr_val))


def _make_spec(
    name,
    num_groups,
    m,
    n,
    k,
    lhs_dtype,
    rhs_dtype,
    group_sizes=None,
    project='',
):
  lhs = jax.ShapeDtypeStruct((m, k), lhs_dtype)
  rhs = jax.ShapeDtypeStruct((num_groups, k, n), rhs_dtype)
  if group_sizes is None:
    group_sizes = [m // num_groups] * num_groups
  else:
    assert len(group_sizes) == num_groups
  group_sizes = base.GroupSizes(  # pytype: disable=wrong-arg-types
      jax.ShapeDtypeStruct((num_groups,), dtype=jnp.int32),
      representative_value=tuple(group_sizes),
  )
  return arg_spec.ArgSpec(
      name=name,
      args=dict(lhs=lhs, rhs=rhs, group_sizes=group_sizes),
      project=project,
      tags=('primary',),
  )


ARG_SPECS = (
    arg_spec.ArgSpec(
        args={
            'lhs': jax.ShapeDtypeStruct(
                shape=(262144, 7168), dtype=jnp.bfloat16
            ),
            'rhs': jax.ShapeDtypeStruct(
                shape=(256, 7168, 2048), dtype=jnp.bfloat16
            ),
            'group_sizes': base.GroupSizes(  # pytype: disable=wrong-arg-types
                jax.ShapeDtypeStruct((256,), dtype=jnp.int32),
                representative_value=_generate_group_sizes(
                    target_m=262144, g=256
                ),
            ),
        },
        project='maxtext',
        name='deepseek-v3',
        tags=('long',),
    ),
    arg_spec.ArgSpec(
        args={
            'lhs': jax.ShapeDtypeStruct(
                shape=(327680, 2880), dtype=jnp.bfloat16
            ),
            'rhs': jax.ShapeDtypeStruct(
                shape=(128, 2880, 2880), dtype=jnp.bfloat16
            ),
            'group_sizes': base.GroupSizes(  # pytype: disable=wrong-arg-types
                jax.ShapeDtypeStruct((128,), dtype=jnp.int32),
                representative_value=_generate_group_sizes(
                    target_m=327680, g=128
                ),
            ),
        },
        project='maxtext',
        name='gpt-oss-327680x2880',
        tags=('long',),
    ),
    arg_spec.ArgSpec(
        args={
            'lhs': jax.ShapeDtypeStruct(
                shape=(393216, 2048), dtype=jnp.bfloat16
            ),
            'rhs': jax.ShapeDtypeStruct(
                shape=(128, 2048, 768), dtype=jnp.bfloat16
            ),
            'group_sizes': base.GroupSizes(  # pytype: disable=wrong-arg-types
                jax.ShapeDtypeStruct((128,), dtype=jnp.int32),
                representative_value=_generate_group_sizes(
                    target_m=393216, g=128
                ),
            ),
        },
        project='maxtext',
        name='gpt-oss-393216x2048',
        tags=('long',),
    ),
    arg_spec.ArgSpec(
        args={
            'lhs': jax.ShapeDtypeStruct(
                shape=(393216, 768), dtype=jnp.bfloat16
            ),
            'rhs': jax.ShapeDtypeStruct(
                shape=(128, 768, 2048), dtype=jnp.bfloat16
            ),
            'group_sizes': base.GroupSizes(  # pytype: disable=wrong-arg-types
                jax.ShapeDtypeStruct((128,), dtype=jnp.int32),
                representative_value=_generate_group_sizes(
                    target_m=393216, g=128
                ),
            ),
        },
        project='maxtext',
        name='gpt-oss-393216x768',
        tags=('long',),
    ),
) + tuple(_make_spec(name, *args) for name, args in SPEC_SHAPES.items())
