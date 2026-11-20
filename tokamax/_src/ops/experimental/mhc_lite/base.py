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
"""Tokamax base ops and VJP ops for mHC-lite."""

from collections.abc import Mapping
import dataclasses
import types
from typing import Any, override
import jax
import jax.numpy as jnp
from tokamax._src.ops import op
from tokamax._src.ops.experimental.mhc_lite import common
from tokamax._src.ops.experimental.mhc_lite import mhc_xla

type PreOutputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
type PreResiduals = tuple[jax.Array, jax.Array]
type CacheKey = Any


def _streams_from_args(args: Mapping[str, Any]) -> int:
  """Extracts expansion stream count `S` from bound op arguments."""
  if "h_post" in args:
    return int(args["h_post"].shape[-1])
  if "permutations" in args:
    return int(args["permutations"].shape[1])
  return int(args["weights"].b_pre.shape[0])


def dims_from_ba(
    ba: op.BoundArguments,
) -> tuple[int, int, int, jnp.dtype]:
  """Extracts canonical `(tokens, streams, embedding, dtype)` from `ba`."""
  x = ba.arguments["x"]
  if x.ndim == 2:
    tokens, flat = x.shape
    streams = _streams_from_args(ba.arguments)
    embedding = flat // streams
  else:
    tokens = x.shape[0] * x.shape[1]
    streams, embedding = x.shape[2], x.shape[3]
  return tokens, streams, embedding, jnp.dtype(x.dtype)


def _check_pre_args(
    x: Any, weights: common.MhcWeights, permutations: Any
) -> None:
  """Validates stream and shape consistency for pre-branch arguments."""
  if permutations.ndim != 3 or permutations.shape[1] != permutations.shape[2]:
    raise common.UnsupportedInputError(
        "Expected permutations of shape (P, streams, streams); got"
        f" {permutations.shape}."
    )
  streams = permutations.shape[1]
  if x.ndim == 2:
    if x.shape[1] % streams:
      raise common.UnsupportedInputError(
          f"Expected 2D x trailing dim divisible by {streams}; got {x.shape}."
      )
    flat = x.shape[1]
  elif x.ndim == 4:
    if x.shape[2] != streams:
      raise common.UnsupportedInputError(
          f"Mismatch between x streams ({x.shape[2]}) and permutations"
          f" ({permutations.shape})."
      )
    flat = streams * x.shape[3]
  else:
    raise common.UnsupportedInputError(
        "Expected x to have shape (batch, sequence, streams, embedding) or"
        f" (tokens, streams * embedding); got {x.shape}."
    )
  if weights.norm_scale.shape != (flat,) or weights.w_pre.shape != (
      flat,
      streams,
  ):
    raise common.UnsupportedInputError(
        f"Incompatible weights shapes for x={x.shape}, streams={streams}:"
        f" norm_scale={weights.norm_scale.shape}, w_pre={weights.w_pre.shape}."
    )


def _check_post_args(
    layer_output: Any, x: Any, h_post: Any, residual: Any
) -> None:
  """Validates stream and token consistency for post-branch arguments."""
  if layer_output.ndim not in (2, 3):
    raise common.UnsupportedInputError(
        f"Expected layer_output rank 2 or 3; got {layer_output.shape}."
    )
  streams = h_post.shape[-1]
  embedding = layer_output.shape[-1]
  tokens = (
      layer_output.shape[0]
      if layer_output.ndim == 2
      else layer_output.shape[0] * layer_output.shape[1]
  )
  if x.size != tokens * streams * embedding:
    raise common.UnsupportedInputError(
        f"x shape {x.shape} does not match tokens={tokens}, streams={streams},"
        f" embedding={embedding}."
    )
  if h_post.size != tokens * streams or residual.size != tokens * streams**2:
    raise common.UnsupportedInputError(
        f"Incompatible gate shapes h_post={h_post.shape},"
        f" residual={residual.shape} for tokens={tokens}, streams={streams}."
    )


def _canonical_cache_key(op_self: op.Op, ba: op.BoundArguments) -> CacheKey:
  """Returns canonicalized cache key ignoring epsilon and `return_residuals`."""
  new_args = dict(ba.arguments)
  if "return_residuals" in new_args:
    new_args["return_residuals"] = False
  if "rms_epsilon" in new_args:
    new_args["rms_epsilon"] = 1e-6
  if "pre_mapping_epsilon" in new_args:
    new_args["pre_mapping_epsilon"] = 1e-6
  if (perm := new_args.get("permutations")) is not None:
    new_args["permutations"] = jax.ShapeDtypeStruct(perm.shape, jnp.float32)
  return op.Op._get_autotuning_cache_key(  # pylint: disable=protected-access
      op_self, dataclasses.replace(ba, arguments=new_args)
  )


def _pre_fwd_ref(
    x: jax.Array,
    weights: common.MhcWeights,
    permutations: jax.Array,
    *,
    rms_epsilon: float = 1e-5,
    pre_mapping_epsilon: float = 1e-6,
) -> tuple[PreOutputs, PreResiduals]:
  """Computes XLA reference outputs and residuals for `MhcLitePre`."""
  layer_input, x_2d, h_post, residual_f32, h_pre = mhc_xla.pre_xla_raw(
      x,
      weights,
      permutations,
      rms_epsilon=rms_epsilon,
      pre_mapping_epsilon=pre_mapping_epsilon,
  )
  tokens = x_2d.shape[0]
  residual_bf16 = residual_f32.astype(x.dtype).reshape(tokens, -1)
  return (layer_input, x_2d, h_post, residual_bf16), (
      weights.fold_norm_scale(),
      h_pre,
  )


def _post_fwd_ref(
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
) -> jax.Array:
  """Computes XLA reference output for `MhcLitePost`."""
  ctx = common.MHCContext(
      x=x, h_post=h_post, residual=residual, implementation="xla"
  )
  return mhc_xla.post_xla(layer_output, ctx)


def _post_and_pre_fwd_ref(
    layer_output: jax.Array,
    x: jax.Array,
    h_post: jax.Array,
    residual: jax.Array,
    next_weights: common.MhcWeights,
    permutations: jax.Array,
    *,
    rms_epsilon: float = 1e-5,
    pre_mapping_epsilon: float = 1e-6,
) -> tuple[PreOutputs, PreResiduals]:
  """Computes XLA reference outputs and residuals for `MhcLitePostAndPre`."""
  x_next = _post_fwd_ref(layer_output, x, h_post, residual)
  return _pre_fwd_ref(
      x_next,
      next_weights,
      permutations,
      rms_epsilon=rms_epsilon,
      pre_mapping_epsilon=pre_mapping_epsilon,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class MhcLitePre[C](op.Op[Any, PreOutputs, PreResiduals, C, CacheKey]):
  """mHC-lite pre-branch stage op."""

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", MhcLitePreVjp())

  @override
  def bind(
      self,
      x: jax.Array,
      weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    _check_pre_args(x, weights, permutations)
    return super().bind(
        x,
        weights,
        permutations,
        rms_epsilon=rms_epsilon,
        pre_mapping_epsilon=pre_mapping_epsilon,
        return_residuals=return_residuals,
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> CacheKey:
    return _canonical_cache_key(self, ba)

  @override
  def _fwd(
      self,
      x: jax.Array,
      weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool,
      config: C,
  ) -> tuple[PreOutputs, PreResiduals | None]:
    del config
    out, residuals = _pre_fwd_ref(
        x,
        weights,
        permutations,
        rms_epsilon=rms_epsilon,
        pre_mapping_epsilon=pre_mapping_epsilon,
    )
    return out, (residuals if return_residuals else None)


@dataclasses.dataclass(frozen=True, kw_only=True)
class MhcLitePreVjp[C](
    op.Op[
        Any,
        tuple[jax.Array, common.MhcWeights, jax.Array],
        types.NoneType,
        C,
        CacheKey,
    ]
):
  """VJP of `MhcLitePre`."""

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> CacheKey:
    return _canonical_cache_key(self, ba)

  @override
  def _fwd(
      self,
      residuals: PreResiduals,
      out: PreOutputs,
      dout: PreOutputs,
      x: jax.Array,
      weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool,
      config: C,
  ) -> tuple[tuple[jax.Array, common.MhcWeights, jax.Array], None]:
    del residuals, out, return_residuals, config
    _, vjp_fn = jax.vjp(
        lambda x_in, w_in: _pre_fwd_ref(
            x_in,
            w_in,
            permutations,
            rms_epsilon=rms_epsilon,
            pre_mapping_epsilon=pre_mapping_epsilon,
        )[0],
        x,
        weights,
    )
    dx, dweights = vjp_fn(dout)
    return (dx, dweights, jax.numpy.zeros_like(permutations)), None


@dataclasses.dataclass(frozen=True, kw_only=True)
class MhcLitePost[C](op.Op[Any, jax.Array, types.NoneType, C, CacheKey]):
  """mHC-lite post-branch stream mixing op."""

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", MhcLitePostVjp())

  @override
  def bind(
      self,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      *,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    _check_post_args(layer_output, x, h_post, residual)
    return super().bind(
        layer_output,
        x,
        h_post,
        residual,
        return_residuals=return_residuals,
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> CacheKey:
    return _canonical_cache_key(self, ba)

  @override
  def _fwd(
      self,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      *,
      return_residuals: bool,
      config: C,
  ) -> tuple[jax.Array, None]:
    del return_residuals, config
    return _post_fwd_ref(layer_output, x, h_post, residual), None


@dataclasses.dataclass(frozen=True, kw_only=True)
class MhcLitePostVjp[C](
    op.Op[
        Any,
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        types.NoneType,
        C,
        CacheKey,
    ]
):
  """VJP of `MhcLitePost`."""

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> CacheKey:
    return _canonical_cache_key(self, ba)

  @override
  def _fwd(
      self,
      residuals: None,
      out: jax.Array,
      dout: jax.Array,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      *,
      return_residuals: bool,
      config: C,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
    del residuals, out, return_residuals, config
    _, vjp_fn = jax.vjp(_post_fwd_ref, layer_output, x, h_post, residual)
    return vjp_fn(dout), None


@dataclasses.dataclass(frozen=True, kw_only=True)
class MhcLitePostAndPre[C](op.Op[Any, PreOutputs, PreResiduals, C, CacheKey]):
  """Fused Stage-1 post-branch mixing and Stage-2 pre-branch projection/gating op."""

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", MhcLitePostAndPreVjp())

  @override
  def bind(
      self,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      next_weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    _check_post_args(layer_output, x, h_post, residual)
    _check_pre_args(x, next_weights, permutations)
    return super().bind(
        layer_output,
        x,
        h_post,
        residual,
        next_weights,
        permutations,
        rms_epsilon=rms_epsilon,
        pre_mapping_epsilon=pre_mapping_epsilon,
        return_residuals=return_residuals,
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> CacheKey:
    return _canonical_cache_key(self, ba)

  @override
  def _fwd(
      self,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      next_weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool,
      config: C,
  ) -> tuple[PreOutputs, PreResiduals | None]:
    del config
    out, residuals = _post_and_pre_fwd_ref(
        layer_output,
        x,
        h_post,
        residual,
        next_weights,
        permutations,
        rms_epsilon=rms_epsilon,
        pre_mapping_epsilon=pre_mapping_epsilon,
    )
    return out, (residuals if return_residuals else None)


@dataclasses.dataclass(frozen=True, kw_only=True)
class MhcLitePostAndPreVjp[C](
    op.Op[
        Any,
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            common.MhcWeights,
            jax.Array,
        ],
        types.NoneType,
        C,
        CacheKey,
    ]
):
  """VJP of `MhcLitePostAndPre`."""

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> CacheKey:
    return _canonical_cache_key(self, ba)

  @override
  def _fwd(
      self,
      residuals: PreResiduals,
      out: PreOutputs,
      dout: PreOutputs,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      next_weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool,
      config: C,
  ) -> tuple[
      tuple[
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          common.MhcWeights,
          jax.Array,
      ],
      None,
  ]:
    del residuals, out, return_residuals, config
    _, vjp_fn = jax.vjp(
        lambda l_in, x_in, hp_in, res_in, w_in: _post_and_pre_fwd_ref(
            l_in,
            x_in,
            hp_in,
            res_in,
            w_in,
            permutations,
            rms_epsilon=rms_epsilon,
            pre_mapping_epsilon=pre_mapping_epsilon,
        )[0],
        layer_output,
        x,
        h_post,
        residual,
        next_weights,
    )
    d_layer_output, d_x, d_h_post, d_residual, d_next_weights = vjp_fn(dout)
    return (
        d_layer_output,
        d_x,
        d_h_post,
        d_residual,
        d_next_weights,
        jax.numpy.zeros_like(permutations),
    ), None
