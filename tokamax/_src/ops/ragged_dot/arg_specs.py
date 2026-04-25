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
    'memory_bound': (8, 8, 4096, 4096, jnp.bfloat16, jnp.bfloat16, [1] * 8),
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
    project='',
    tags=(),
) -> arg_spec.ArgSpec:
  """Make an argument spec for a ragged dot operation."""
  lhs = jax.ShapeDtypeStruct((m, k), lhs_dtype)
  rhs = jax.ShapeDtypeStruct((num_groups, k, n), rhs_dtype)
  if group_sizes is None:
    group_sizes = base.GroupSizes(
        jax.ShapeDtypeStruct((num_groups,), dtype=jnp.int32), m
    )
  else:
    assert len(group_sizes) == num_groups
    group_sizes = base.GroupSizes(
        jax.ShapeDtypeStruct((num_groups,), dtype=jnp.int32), tuple(group_sizes)
    )

  args = dict(lhs=lhs, rhs=rhs, group_sizes=group_sizes)
  return arg_spec.ArgSpec(name=name, args=args, project=project, tags=tags)


def _make_maxtext_spec(name_prefix, num_groups, *, m, n, k) -> arg_spec.ArgSpec:
  return _make_spec(
      name=f'{name_prefix}-{m}x{k}_{num_groups}x{k}x{n}',
      num_groups=num_groups,
      m=m,
      n=n,
      k=k,
      lhs_dtype=jnp.bfloat16,
      rhs_dtype=jnp.bfloat16,
      project='maxtext',
      tags=('primary',),
  )


ARG_SPECS = (
    _make_maxtext_spec('deepseek-v3', 256, m=131072, n=1024, k=7168),
    _make_maxtext_spec('deepseek-v3', 256, m=131072, n=7168, k=1024),
    _make_maxtext_spec('deepseek-v3', 256, m=131072, n=512, k=7168),
    _make_maxtext_spec('deepseek-v3', 256, m=131072, n=7168, k=512),
    _make_maxtext_spec('deepseek-v3', 256, m=262144, n=512, k=7168),
    _make_maxtext_spec('deepseek-v3', 256, m=262144, n=7168, k=256),
    _make_maxtext_spec('deepseek-v3', 256, m=262144, n=7168, k=1024),
    _make_maxtext_spec('deepseek-v3', 256, m=262144, n=1024, k=7168),
    _make_maxtext_spec('deepseek-v3', 256, m=262144, n=2048, k=7168),
    _make_maxtext_spec('gpt-oss', 128, m=327680, n=2880, k=2880),
    _make_maxtext_spec('gpt-oss', 128, m=393216, n=768, k=2048),
    _make_maxtext_spec('gpt-oss', 128, m=393216, n=2048, k=768),
    _make_maxtext_spec('gpt-oss', 128, m=524288, n=1536, k=4096),
    _make_maxtext_spec('gpt-oss', 128, m=524288, n=4096, k=1536),
    _make_maxtext_spec('gpt-oss', 256, m=524288, n=2048, k=7168),
    _make_maxtext_spec('gpt-oss', 128, m=262144, n=1536, k=4096),
    _make_maxtext_spec('gpt-oss', 128, m=262144, n=4096, k=1536),
    _make_maxtext_spec('gpt-oss', 128, m=131072, n=2048, k=7168),
    _make_maxtext_spec('gpt-oss', 128, m=131072, n=768, k=2048),
    _make_maxtext_spec('gpt-oss', 128, m=131072, n=1536, k=4096),
    _make_maxtext_spec('gpt-oss', 128, m=131072, n=4096, k=1536),
    _make_maxtext_spec('gpt-oss', 256, m=131072, n=2048, k=7168),
    _make_maxtext_spec('gpt-oss', 256, m=65536, n=2048, k=7168),
    _make_maxtext_spec('gpt-oss', 256, m=131072, n=512, k=7168),
    _make_maxtext_spec('gpt-oss', 256, m=131072, n=7168, k=512),
    _make_maxtext_spec('gpt-oss', 256, m=262144, n=7168, k=512),
    _make_maxtext_spec('gpt-oss', 256, m=262144, n=512, k=7168),
    _make_maxtext_spec('gpt-oss', 256, m=262144, n=7168, k=256),
    _make_maxtext_spec('gpt-oss', 256, m=262144, n=7168, k=1024),
    _make_maxtext_spec('gpt-oss', 256, m=262144, n=1024, k=7168),
) + tuple(_make_spec(name, *args) for name, args in SPEC_SHAPES.items())
