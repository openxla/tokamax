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

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal
import immutabledict
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import base
from tokamax._src.ops.experimental.mhc_lite import common
from tokamax._src.ops.experimental.mhc_lite import hc_xla

type Implementation = Literal["mosaic_tpu", "xla"]

_PRE_IMPLEMENTATIONS: dict[str, base.MhcLitePre[Any]] = dict(
    xla=base.MhcLitePre(),
)
_POST_IMPLEMENTATIONS: dict[str, base.MhcLitePost[Any]] = dict(
    xla=base.MhcLitePost(),
)
_POST_AND_PRE_IMPLEMENTATIONS: dict[str, base.MhcLitePostAndPre[Any]] = dict(
    xla=base.MhcLitePostAndPre(),
)

try:
  from tokamax._src.ops.experimental.mhc_lite import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _PRE_IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuMhcLitePre()
  _POST_IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuMhcLitePost()
  _POST_AND_PRE_IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasTpuMhcLitePostAndPre()
  )
except ImportError:
  pallas_mosaic_tpu = None

_DEFAULT_IMPLEMENTATIONS: tuple[Implementation, ...] = (
    ("mosaic_tpu", "xla") if "mosaic_tpu" in _PRE_IMPLEMENTATIONS else ("xla",)
)

PRE_IMPLEMENTATIONS: Final[Mapping[str, base.MhcLitePre[Any]]] = (
    immutabledict.immutabledict(_PRE_IMPLEMENTATIONS)
)
POST_IMPLEMENTATIONS: Final[Mapping[str, base.MhcLitePost[Any]]] = (
    immutabledict.immutabledict(_POST_IMPLEMENTATIONS)
)
POST_AND_PRE_IMPLEMENTATIONS: Final[
    Mapping[str, base.MhcLitePostAndPre[Any]]
] = immutabledict.immutabledict(_POST_AND_PRE_IMPLEMENTATIONS)
del _PRE_IMPLEMENTATIONS, _POST_IMPLEMENTATIONS, _POST_AND_PRE_IMPLEMENTATIONS


def _dispatch(
    implementations: Mapping[str, Any],
    implementation: Implementation | Sequence[Implementation] | None,
    call_fn: Callable[[str, Any], Any],
) -> tuple[str, Any]:
  """Calls the first implementation that does not raise `NotImplementedError`."""
  if implementation is None:
    candidates: tuple[str, ...] = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    if implementation not in implementations:
      raise ValueError(f"Unsupported implementation: '{implementation}'")
    candidates = (implementation,)
  else:
    candidates = tuple(implementation)
    if not candidates or any(c not in implementations for c in candidates):
      raise ValueError(f"Unsupported implementation: {implementation}")

  errors = []
  for impl_name in candidates:
    try:
      return impl_name, call_fn(impl_name, implementations[impl_name])
    except NotImplementedError as e:
      if len(candidates) == 1:
        raise
      errors.append(e)
  raise ExceptionGroup("all implementations failed", errors)


def _configure_op(
    impl_name: str,
    op_instance: Any,
    config: common.MhcKernelConfig,
    *,
    tokens: int,
    embedding: int,
) -> Any:
  """Applies user config overrides when dispatching to `"mosaic_tpu"`."""
  if impl_name == "mosaic_tpu" and pallas_mosaic_tpu is not None:
    return pallas_mosaic_tpu.configure_op(
        op_instance, config, tokens=tokens, embedding=embedding
    )
  return op_instance


def _canonicalize_pre_x(
    x: jax.Array, permutations: jax.Array
) -> tuple[jax.Array, int, int, tuple[int, ...]]:
  """Validates rank of `x` and returns `(x_2d, tokens, embedding, out_shape)`."""
  if x.ndim == 2:
    tokens, flat = x.shape
    streams = permutations.shape[1]
    embedding = flat // streams
    return x, tokens, embedding, (tokens, embedding)
  if x.ndim == 4:
    batch, seq, streams, embedding = x.shape
    if streams != permutations.shape[1]:
      raise common.UnsupportedInputError(
          f"Mismatch between x streams ({streams}) and permutations"
          f" ({permutations.shape})."
      )
    tokens = batch * seq
    return (
        x.reshape(tokens, streams * embedding),
        tokens,
        embedding,
        (batch, seq, embedding),
    )
  raise common.UnsupportedInputError(
      "Expected x to have shape (batch, sequence, streams, embedding) or"
      f" (tokens, streams * embedding); got {x.shape}."
  )


def _canonicalize_post_inputs(
    layer_output: jax.Array,
    context: common.MHCContext,
) -> tuple[
    int, int, int, jax.Array, jax.Array, jax.Array, jax.Array, tuple[int, ...]
]:
  """Reshapes `post`/`post_and_pre` inputs to canonical 2D forms."""
  embedding = layer_output.shape[-1]
  streams = context.h_post.shape[-1]
  tokens = (
      layer_output.shape[0]
      if layer_output.ndim == 2
      else layer_output.shape[0] * layer_output.shape[1]
  )
  out_shape = (
      (tokens, streams * embedding)
      if layer_output.ndim == 2
      else (*layer_output.shape[:2], streams, embedding)
  )
  x_2d = context.x.reshape(tokens, streams * embedding)
  lo_2d = layer_output.reshape(tokens, embedding)
  hp_2d = context.h_post.reshape(tokens, streams)
  res_2d = context.residual.reshape(tokens, streams * streams)
  return tokens, streams, embedding, x_2d, lo_2d, hp_2d, res_2d, out_shape


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
  with jax.named_scope("mhc_lite_pre"):
    permutations = jax.lax.stop_gradient(permutations)
    x_2d, tokens, embedding, layer_input_shape = _canonicalize_pre_x(
        x, permutations
    )
    perms_f32 = permutations.astype(jnp.float32)

    def _call(impl_name: str, op_instance: Any):
      op_fn = _configure_op(
          impl_name, op_instance, config, tokens=tokens, embedding=embedding
      )
      return op_fn(
          x_2d,
          weights,
          perms_f32,
          rms_epsilon=config.rms_epsilon,
          pre_mapping_epsilon=config.pre_mapping_epsilon,
      )

    impl_used, (layer_input_2d, x_out, h_post, residual) = _dispatch(
        PRE_IMPLEMENTATIONS, implementation, _call
    )
    return layer_input_2d.reshape(layer_input_shape), common.MHCContext(
        x=x_out,
        h_post=h_post,
        residual=residual,
        implementation=impl_used,
    )


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
  with jax.named_scope("mhc_lite_post"):
    if context.implementation not in POST_IMPLEMENTATIONS:
      raise ValueError(
          "Unsupported implementation in MHCContext:"
          f" '{context.implementation}'"
      )
    tokens, _, embedding, x_2d, lo_2d, hp_2d, res_2d, out_shape = (
        _canonicalize_post_inputs(layer_output, context)
    )
    impl_name = context.implementation
    if impl_name == "mosaic_tpu":
      res_2d = res_2d.astype(x_2d.dtype)
    op_fn = _configure_op(
        impl_name,
        POST_IMPLEMENTATIONS[impl_name],
        config,
        tokens=tokens,
        embedding=embedding,
    )
    out_2d = op_fn(lo_2d, x_2d, hp_2d, res_2d)
    return out_2d.reshape(out_shape)


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
  with jax.named_scope("mhc_lite_post_and_pre"):
    permutations = jax.lax.stop_gradient(permutations)
    tokens, _, embedding, x_2d, lo_2d, hp_2d, res_2d, _ = (
        _canonicalize_post_inputs(layer_output, context)
    )
    perms_f32 = permutations.astype(jnp.float32)
    req_impl: Implementation | Sequence[Implementation] = (
        context.implementation  # type: ignore[assignment]
        if implementation is None
        else implementation
    )

    def _call(impl_name: str, op_instance: Any):
      op_fn = _configure_op(
          impl_name, op_instance, config, tokens=tokens, embedding=embedding
      )
      res = res_2d.astype(x_2d.dtype) if impl_name == "mosaic_tpu" else res_2d
      return op_fn(
          lo_2d,
          x_2d,
          hp_2d,
          res,
          next_weights,
          perms_f32,
          rms_epsilon=config.rms_epsilon,
          pre_mapping_epsilon=config.pre_mapping_epsilon,
      )

    impl_used, (next_li_2d, x_next, next_hp, next_res) = _dispatch(
        POST_AND_PRE_IMPLEMENTATIONS, req_impl, _call
    )
    return next_li_2d.reshape(layer_output.shape), common.MHCContext(
        x=x_next,
        h_post=next_hp,
        residual=next_res,
        implementation=impl_used,
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
  with jax.named_scope("hc_pre"):
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
  with jax.named_scope("hc_post"):
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
  with jax.named_scope("hc_post_and_pre"):
    if context.implementation != "hc":
      raise ValueError(
          "Expected MHCContext with implementation='hc'; got"
          f" '{context.implementation}'"
      )
    return hc_xla.post_and_pre_hc(layer_output, context, next_weights)
