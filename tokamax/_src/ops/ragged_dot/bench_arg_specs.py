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
from tokamax._src.autotuning import arg_specs_common as common
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


def _make_spec(
    name,
    num_groups,
    m,
    n,
    k,
    lhs_dtype,
    rhs_dtype,
    group_sizes=None,
    project=None,
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
  return common.ArgSpec(
      name=name,
      args=dict(lhs=lhs, rhs=rhs, group_sizes=group_sizes),
      project=project,
      tags=('primary',),
  )


ARG_SPECS = tuple(_make_spec(name, *args) for name, args in SPEC_SHAPES.items())
