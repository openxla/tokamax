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
"""Public API entrypoints for mHC-lite Pallas TPU kernel."""

from collections.abc import Sequence
from typing import Literal
import jax
from tokamax._src.ops.experimental.mhc_lite import common
from tokamax._src.ops.experimental.mhc_lite import hc_xla
from tokamax._src.ops.experimental.mhc_lite import mhc_kernels_fwd
from tokamax._src.ops.experimental.mhc_lite import mhc_xla

type Implementation = Literal["mosaic_tpu", "xla"]


def _resolve_implementation(
    implementation: Implementation | Sequence[Implementation] | None,
) -> Implementation:
  """Validates and resolves the requested implementation."""
  valid = ("mosaic_tpu", "xla")
  if implementation is None:
    return "mosaic_tpu"
  if isinstance(implementation, str):
    if implementation not in valid:
      raise ValueError(f"Unsupported implementation: '{implementation}'")
    return implementation
  for candidate in implementation:
    if candidate in valid:
      return candidate
  raise ValueError(f"Unsupported implementation: {implementation}")


def mhc_lite_pre(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    *,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> tuple[jax.Array, common.MHCContext]:
  """Computes the branch input and opaque context for an mHC-lite wrapped branch.

  Uses the Pallas TPU kernel when `"mosaic_tpu"` (or `None`) is selected, or
  the pure JAX/XLA mHC-lite reference when `"xla"` is selected. Internal matrix
  multiplications use `bfloat16` operand precision with `float32` accumulation
  (`preferred_element_type=jnp.float32`).

  Args:
    x: Input streams of shape `(batch, sequence, streams, embedding)` or
      `(tokens, streams * embedding)`.
    weights: Structured `common.MhcWeights` for mHC-lite.
    permutations: All permutation matrices of shape `(num_permutations, streams,
      streams)`.
    config: Structured `MhcKernelConfig` tuning and compiler configuration.
    implementation: Preferred implementation (`"mosaic_tpu"` or `"xla"`).

  Returns:
    A tuple `(layer_input, context)` where `layer_input` feeds the wrapped
    model branch, and `context` is passed unchanged to `mhc_lite_post`.
  """
  resolved = _resolve_implementation(implementation)
  permutations = jax.lax.stop_gradient(permutations)
  if resolved == "xla":
    return mhc_xla.pre_xla(x, weights, permutations, config=config)
  return mhc_kernels_fwd.pre(x, weights, permutations, config)


def mhc_lite_post(
    layer_output: jax.Array,
    context: common.MHCContext,
    *,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
) -> jax.Array:
  """Runs the mHC-lite post-gate and residual stream mixing.

  Internal matrix multiplications use `bfloat16` operand precision with
  `float32` accumulation (`preferred_element_type=jnp.float32`), and the final
  mixed stream output is cast to `bfloat16`.

  Args:
    layer_output: Output from the wrapped branch of shape `(batch, sequence,
      embedding)` or `(tokens, embedding)`.
    context: Opaque `common.MHCContext` returned by `mhc_lite_pre`.
    config: Structured `MhcKernelConfig` tuning and compiler configuration.

  Returns:
    Mixed output streams of shape `(batch, sequence, streams, embedding)` or
    `(tokens, streams * embedding)`.
  """
  if context.implementation == "xla":
    return mhc_xla.post_xla(layer_output, context)
  if context.implementation != "mosaic_tpu":
    raise ValueError(
        f"Unsupported implementation in MHCContext: '{context.implementation}'"
    )
  return mhc_kernels_fwd.post(layer_output, context, config)


def mhc_lite_post_and_pre(
    layer_output: jax.Array,
    context: common.MHCContext,
    next_weights: common.MhcWeights,
    permutations: jax.Array,
    *,
    config: common.MhcKernelConfig = common.MhcKernelConfig(),
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> tuple[jax.Array, common.MHCContext]:
  """Runs fused Stage-1 post-mixing and Stage-2 pre-branch projection/gating.

  Args:
    layer_output: Output from the Stage-1 wrapped branch of shape `(batch,
      sequence, embedding)` or `(tokens, embedding)`.
    context: Opaque `common.MHCContext` returned by Stage-1 `mhc_lite_pre` or
      previous `mhc_lite_post_and_pre`.
    next_weights: Stage-2 structured `common.MhcWeights` for mHC-lite.
    permutations: All permutation matrices of shape `(num_permutations, streams,
      streams)`.
    config: Structured `MhcKernelConfig` tuning and compiler configuration.
    implementation: Preferred implementation (`"mosaic_tpu"` or `"xla"`).
      Defaults to `context.implementation` if `None`.

  Returns:
    A tuple `(next_layer_input, next_context)` for Stage 2.
  """
  resolved = _resolve_implementation(
      context.implementation if implementation is None else implementation
  )
  permutations = jax.lax.stop_gradient(permutations)
  if resolved == "xla":
    return mhc_xla.post_and_pre_xla(
        layer_output, context, next_weights, permutations, config=config
    )
  return mhc_kernels_fwd.post_and_pre(
      layer_output, context, next_weights, permutations, config
  )


def hc_pre(
    x: jax.Array,
    weights: common.HcWeights,
) -> tuple[jax.Array, common.MHCContext]:
  """Computes the branch input and context for a static HC wrapped branch.

  Args:
    x: Input streams of shape `(batch, sequence, streams, embedding)` or
      `(tokens, streams * embedding)`.
    weights: Static Hyper-Connections weight container (`common.HcWeights`).

  Returns:
    A tuple `(layer_input, context)` where `layer_input` feeds the wrapped
    model branch, and `context` is passed unchanged to `hc_post`.
  """
  return hc_xla.pre_hc(x, weights)


def hc_post(
    layer_output: jax.Array,
    context: common.MHCContext,
) -> jax.Array:
  """Runs the static HC post-gate and residual stream mixing.

  Args:
    layer_output: Output from the wrapped branch of shape `(batch, sequence,
      embedding)` or `(tokens, embedding)`.
    context: Opaque `common.MHCContext` returned by `hc_pre`.

  Returns:
    Mixed output streams of shape `(batch, sequence, streams, embedding)` or
    `(tokens, streams * embedding)`.
  """
  if context.implementation != "hc":
    raise ValueError(
        "Expected MHCContext with implementation='hc'; got"
        f" '{context.implementation}'"
    )
  return hc_xla.post_hc(layer_output, context)


def hc_post_and_pre(
    layer_output: jax.Array,
    context: common.MHCContext,
    next_weights: common.HcWeights,
) -> tuple[jax.Array, common.MHCContext]:
  """Runs static HC fused Stage-1 post-mixing and Stage-2 pre-branch gating.

  Args:
    layer_output: Output from the Stage-1 wrapped branch of shape `(batch,
      sequence, embedding)` or `(tokens, embedding)`.
    context: Opaque `common.MHCContext` returned by `hc_pre` or
      `hc_post_and_pre`.
    next_weights: Stage-2 static Hyper-Connections weight container
      (`common.HcWeights`).

  Returns:
    A tuple `(next_layer_input, next_context)` for Stage 2.
  """
  if context.implementation != "hc":
    raise ValueError(
        "Expected MHCContext with implementation='hc'; got"
        f" '{context.implementation}'"
    )
  return hc_xla.post_and_pre_hc(layer_output, context, next_weights)
