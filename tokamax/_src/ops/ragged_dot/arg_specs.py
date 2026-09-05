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
        '',
        ('primary', 'ci_tests'),
    ),
    'memory_bound': (
        8,
        8,
        4096,
        4096,
        jnp.bfloat16,
        jnp.bfloat16,
        [1] * 8,
        '',
        ('primary', 'ci_tests'),
    ),
    # FIXME: Use correct dtypes.
    '8x7b': (
        8,
        8192,
        14336,
        4096,
        jnp.bfloat16,
        jnp.bfloat16,
        None,
        'mixtral',
        ('primary', 'ci_tests'),
    ),
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


def _make_maxtext_spec(
    name_prefix, num_groups, *, m, n, k, tags=(),
) -> arg_spec.ArgSpec:
  return _make_spec(
      name=f'{name_prefix}-{m}x{k}_{num_groups}x{k}x{n}',
      num_groups=num_groups,
      m=m,
      n=n,
      k=k,
      lhs_dtype=jnp.bfloat16,
      rhs_dtype=jnp.bfloat16,
      project='maxtext',
      tags=tags,
  )


def _make_maxtext_gmm_v2_spec(
    name: str,
    *,
    num_groups: int,
    m: int,
    n: int,
    k: int,
    block_size: int = 256,
    quantized: bool = True,
    project: str = 'maxtext',
    tags: tuple[arg_spec.Tag, ...] = ('primary',),
) -> arg_spec.ArgSpec:
  """Make a GMM v2 spec based on MaxText shapes."""
  lhs = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
  if quantized:
    rhs = jax.ShapeDtypeStruct((num_groups, k, n), jnp.float8_e4m3fn)
    rhs_scale = jax.ShapeDtypeStruct(
        (num_groups, k // block_size, 1, n), jnp.bfloat16
    )
    args = dict(
        lhs=lhs,
        rhs=rhs,
        group_sizes=base.GroupSizes(
            jax.ShapeDtypeStruct((num_groups,), dtype=jnp.int32), m
        ),
        rhs_scale=rhs_scale,
        maybe_quantize_lhs=True,
        preferred_element_type=jnp.bfloat16,
    )
  else:
    rhs = jax.ShapeDtypeStruct((num_groups, k, n), jnp.bfloat16)
    args = dict(
        lhs=lhs,
        rhs=rhs,
        group_sizes=base.GroupSizes(
            jax.ShapeDtypeStruct((num_groups,), dtype=jnp.int32), m
        ),
        preferred_element_type=jnp.bfloat16,
    )
  return arg_spec.ArgSpec(
      name=name,
      args=args,
      project=project,
      tags=tags,
  )


def _make_ullm_gmm_v2_spec(
    name: str,
    *,
    m: int,
    k: int,
    n: int,
    fuse_act: str | None = None,
    num_groups: int = 64,
    project: str = 'ullm',
    tags: tuple[arg_spec.Tag, ...] = ('primary', 'forward_only'),
) -> arg_spec.ArgSpec:
  """Make a ULLM MoE GMM v2 spec."""
  lhs = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
  rhs = jax.ShapeDtypeStruct((num_groups, k, n), jnp.float8_e4m3fn)
  rhs_scale = jax.ShapeDtypeStruct((num_groups, 1, 1, n), jnp.bfloat16)
  group_sizes = base.GroupSizes(
      jax.ShapeDtypeStruct((num_groups,), dtype=jnp.int32),
      tuple([m // num_groups] * num_groups),
  )
  args = dict(
      lhs=lhs,
      rhs=rhs,
      group_sizes=group_sizes,
      rhs_scale=rhs_scale,
      maybe_quantize_lhs=True,
      zero_initialize=False,
      fuse_gateup_activation=fuse_act,
      preferred_element_type=jnp.bfloat16,
  )
  return arg_spec.ArgSpec(
      name=name,
      args=args,
      project=project,
      tags=tags,
  )


ARG_SPECS = (
    _make_maxtext_spec(
        'deepseek-v3', 256, m=131072, n=1024, k=7168, tags=('primary',)
    ),
    _make_maxtext_spec(
        'deepseek-v3', 256, m=131072, n=7168, k=1024, tags=('primary',)
    ),
    _make_maxtext_spec(
        'deepseek-v3', 256, m=131072, n=512, k=7168, tags=('primary',)
    ),
    _make_maxtext_spec(
        'deepseek-v3', 256, m=131072, n=7168, k=512, tags=('primary',)
    ),
    _make_maxtext_spec(
        'deepseek-v3', 256, m=262144, n=512, k=7168, tags=('primary',)
    ),
    _make_maxtext_spec(
        'deepseek-v3', 256, m=262144, n=7168, k=256, tags=('primary',)
    ),
    _make_maxtext_spec(
        'deepseek-v3', 256, m=262144, n=7168, k=1024, tags=('primary',)
    ),
    _make_maxtext_spec(
        'deepseek-v3', 256, m=262144, n=1024, k=7168, tags=('primary',)
    ),
    _make_maxtext_spec(
        'deepseek-v3', 256, m=262144, n=2048, k=7168, tags=('primary',)
    ),
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
) + tuple(_make_spec(name, *args) for name, args in SPEC_SHAPES.items()) + (
    # Use smaller input shapes to avoid XLA implementation from OOMing.
    _make_maxtext_gmm_v2_spec(
        'gmm_v2_maxtext_8192x7168_64x7168x1024',
        num_groups=64,
        m=8192,
        n=1024,
        k=7168,
        block_size=256,
        quantized=True,
    ),
    _make_ullm_gmm_v2_spec(
        'gmm_v2_ullm_prefill_gate_up',
        m=81920,
        k=4096,
        n=2 * 1024,
        fuse_act='silu',
    ),
    _make_ullm_gmm_v2_spec(
        'gmm_v2_ullm_prefill_down',
        m=81920,
        k=1024,
        n=4096,
        fuse_act=None,
    ),
    _make_ullm_gmm_v2_spec(
        'gmm_v2_ullm_decode_gate_up',
        m=1280,
        k=4096,
        n=2 * 1024,
        fuse_act='silu',
    ),
    _make_ullm_gmm_v2_spec(
        'gmm_v2_ullm_decode_down',
        m=1280,
        k=1024,
        n=4096,
        fuse_act=None,
    ),
)

