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
"""Linear softmax cross entropy loss argument specifications."""

import itertools
from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct

# `jax.Device.device_kind` values for TPU v5e and TPU v5p.
_TPU_V5E_PLATFORMS: Final[tuple[str, ...]] = ('TPU v5 lite', 'TPU v5e')
_TPU_V5P_PLATFORMS: Final[tuple[str, ...]] = ('TPU v5', 'TPU v5p')
# HBM capacity per TPU v5p chip.
_TPU_V5P_HBM_BYTES: Final[int] = 95 * 2**30


def _reference_peak_hbm_bytes(
    batch_dim: int,
    hidden_dim: int,
    output_dim: int,
    x_dtype: jax.typing.DTypeLike,
    w_dtype: jax.typing.DTypeLike,
) -> int:
  """Conservatively estimates the peak HBM usage of the XLA reference impl.

  The reference implementation materializes the full `[B, V]` logits in the
  forward pass and the full `[B, V]` softmax gradient in the backward pass. We
  budget for two such buffers being live at once, plus the inputs and their
  gradients.

  Args:
    batch_dim: The batch dimension `B`.
    hidden_dim: The hidden dimension `H`.
    output_dim: The output (vocabulary) dimension `V`.
    x_dtype: The dtype of `x`.
    w_dtype: The dtype of `w`.

  Returns:
    The estimated peak HBM usage in bytes.
  """
  x_itemsize = jnp.dtype(x_dtype).itemsize
  w_itemsize = jnp.dtype(w_dtype).itemsize
  logits_itemsize = jnp.promote_types(x_dtype, w_dtype).itemsize
  logits_bytes = batch_dim * output_dim * logits_itemsize
  inputs_bytes = (
      batch_dim * hidden_dim * x_itemsize + hidden_dim * output_dim * w_itemsize
  )
  return 2 * logits_bytes + 2 * inputs_bytes


def _create_arg_spec(
    batch_dim: int,
    hidden_dim: int,
    output_dim: int,
    reduction: str,
    x_dtype: jax.typing.DTypeLike,
    w_dtype: jax.typing.DTypeLike,
    project: str = '',
    tags: tuple[arg_spec.Tag, ...] = (),
    excluded_platforms: tuple[str, ...] = (),
) -> arg_spec.ArgSpec:
  excluded = list(excluded_platforms)
  # The reference impl's `[B, V]` intermediate buffers exceed the 16 GiB HBM of
  # TPU v5e for large `B * V`.
  if batch_dim * output_dim >= 10**9:
    excluded.extend(_TPU_V5E_PLATFORMS)
  # TPU v5p has 95 GiB of HBM, so only exclude shapes whose estimated reference
  # footprint does not fit.
  if (
      _reference_peak_hbm_bytes(
          batch_dim, hidden_dim, output_dim, x_dtype, w_dtype
      )
      > _TPU_V5P_HBM_BYTES
  ):
    excluded.extend(_TPU_V5P_PLATFORMS)
  return arg_spec.ArgSpec(
      dict(
          x=ShapeDtype((batch_dim, hidden_dim), x_dtype),
          labels=ShapeDtype((batch_dim,), jnp.int32),
          w=ShapeDtype((hidden_dim, output_dim), w_dtype),
          reduction=reduction,
      ),
      project=project,
      tags=tags,
      excluded_platforms=tuple(dict.fromkeys(excluded)),
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=4096,
            output_dim=151936,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='qwen3-8b',
            tags=(
                ('primary',)
                if batch_dim == 1024
                and reduction == 'mean'
                and x_dtype == jnp.bfloat16
                else ()
            ),
            # TODO: HBM page fault on TPU v5p.
            excluded_platforms=('TPU v5p', 'TPU v5', 'TPU v5 lite', 'TPU v5e'),
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            [1024, 2048, 4096, 16384, 32768, 131072],
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=2560,
            output_dim=262144,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='gemma3-4b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            [1024, 2048, 4096, 16384, 32768, 131072],
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=3840,
            output_dim=262144,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='gemma3-7b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            [1024, 2048, 4096, 16384, 32768, 131072],
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=4096,
            output_dim=128256,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='llama3.1-8b',
            tags=(
                ('primary',)
                if batch_dim == 1024
                and reduction == 'mean'
                and x_dtype == jnp.bfloat16
                else ()
            ),
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            [1024, 2048, 4096, 16384, 32768, 131072],
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=7168,
            output_dim=128256,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='deepseek-v3-671b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            (4096, 8192, 16384, 32768),
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=2880,
            output_dim=201088,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='gpt-oss-120b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            (1024, 2048, 4096, 8192, 16384, 32768),
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=5120,
            output_dim=151936,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='qwen3-32b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            (1024, 2048, 4096, 8192, 16384, 32768),
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
)
