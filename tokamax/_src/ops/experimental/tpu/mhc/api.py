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
"""DeepSeek-V4 manifold-constrained hyper-connections (mHC) API.

mHC keeps `M = hc_mult` residual streams per token instead of one. Each
sublayer (attention or MoE) is wrapped by an `mhc_pre` / `mhc_post` pair, with
the sublayer itself running in between:

```
post_mix, comb_mix, layer_input = mhc_pre(residual, fn, hc_scale, ...)
x = sublayer(layer_input)
residual = mhc_post(x, residual, post_mix, comb_mix)
```

* `mhc_pre` reads the streams: a mix GEMM against the projection `fn` gives
  per-token gate logits, which become
  - `pre_mix` (sigmoid), the weights that collapse the streams into the
    sublayer input `layer_input`;
  - `post_mix` (sigmoid), the weights that write the sublayer output back;
  - `comb_mix`, a Sinkhorn-normalized `(M, M)` stream-mixing matrix.
* `mhc_post` writes the sublayer output `x` back:
  `new_residual[j] = post_mix[j] * x + sum_i comb_mix[i, j] * residual[i]`,
  where `residual[i]` is stream `i`.
* `mhc_fused_post_pre` computes one sublayer's `mhc_post` followed by the next
  sublayer's `mhc_pre` with one pass over the residual streams instead of two.
  A model can use it at every seam between consecutive sublayers, so it only
  needs a separate `mhc_pre` before its first sublayer and `mhc_post` after its
  last.

The residual streams use the kernels' native flat layout: `residual` is
`(T, M * H)` bf16 for `T` tokens, `M = hc_mult` streams and `H = hidden_size`,
with stream `i` in columns `[i * H, (i + 1) * H)`. A model that keeps the
streams as `(T, M, H)` can reshape to and from this layout for free. `M` is not
passed explicitly: it comes from the shape of `fn` (pre) or `comb_res_mix`
(post). With `M3 = M * (M + 2)`, `fn` is `(M3, M * H)` f32, `hc_scale` is `(3,)`
f32 (pre/post/comb logit scales) and `hc_base` is `(M3,)` f32 (logit biases, in
pre/post/comb order).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal

import immutabledict
import jax
from tokamax._src.ops.experimental.tpu.mhc import base

type Implementation = Literal["xla", "mosaic_tpu"]
type _ImplementationArg = (
    Implementation | Sequence[Implementation | Callable[..., Any]] | None
)

_PRE_IMPLEMENTATIONS = dict(xla=base.MhcPre())
_POST_IMPLEMENTATIONS = dict(xla=base.MhcPost())
_FUSED_IMPLEMENTATIONS = dict(xla=base.MhcFusedPostPre())

try:
  from tokamax._src.ops.experimental.tpu.mhc import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _PRE_IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuMhcPre()
  _POST_IMPLEMENTATIONS["mosaic_tpu"] = pallas_mosaic_tpu.PallasTpuMhcPost()
  _FUSED_IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasTpuMhcFusedPostPre()
  )
except ImportError:
  pass

_DEFAULT_IMPLEMENTATIONS: tuple[Implementation, ...] = (
    ("mosaic_tpu", "xla") if "mosaic_tpu" in _PRE_IMPLEMENTATIONS else ("xla",)
)

PRE_IMPLEMENTATIONS: Final[
    immutabledict.immutabledict[str, Callable[..., Any]]
] = immutabledict.immutabledict(_PRE_IMPLEMENTATIONS)
POST_IMPLEMENTATIONS: Final[
    immutabledict.immutabledict[str, Callable[..., Any]]
] = immutabledict.immutabledict(_POST_IMPLEMENTATIONS)
FUSED_POST_PRE_IMPLEMENTATIONS: Final[
    immutabledict.immutabledict[str, Callable[..., Any]]
] = immutabledict.immutabledict(_FUSED_IMPLEMENTATIONS)
del _PRE_IMPLEMENTATIONS, _POST_IMPLEMENTATIONS, _FUSED_IMPLEMENTATIONS


def _dispatch(
    implementations: Mapping[str, Callable[..., Any]],
    implementation: _ImplementationArg,
    *args,
) -> Any:
  """Calls the first implementation that doesn't raise NotImplementedError."""
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl not in implementations:
        raise ValueError(
            f"Unknown implementation: {impl}. You may need to add a dependency"
            " on the corresponding backend."
        )
      impl = implementations[impl]

    try:
      return impl(*args)
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)


def mhc_pre(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    implementation: _ImplementationArg = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """mHC pre step, run before each sublayer.

  Reads the `M = hc_mult` residual streams: computes the per-token gates and
  collapses the streams into the sublayer input. See the module docstring for
  the math and the call order.

  Args:
    residual: `(T, M * H)` bf16 flat residual streams.
    fn: `(M * (M + 2), M * H)` f32 gate projection.
    hc_scale: `(3,)` f32 pre/post/comb logit scales.
    hc_base: `(M * (M + 2),)` f32 logit biases, in pre/post/comb order.
    rms_eps: RMS-norm epsilon.
    hc_pre_eps: Additive floor on the sigmoid pre gates.
    hc_sinkhorn_eps: Sinkhorn epsilon.
    hc_post_mult_value: Scale of the sigmoid post gates.
    sinkhorn_repeat: Number of Sinkhorn normalization rounds.
    implementation: The implementation to use. By default, `None` is used, which
      selects the best available backend. If a sequence is passed, the first
      implementation that doesn't raise a `NotImplementedError` is used.

  Returns:
    `(post_mix (T, M) f32, comb_mix (T, M, M) f32, layer_input (T, H) bf16)`.

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise `NotImplementedError`.
  """
  return _dispatch(
      PRE_IMPLEMENTATIONS,
      implementation,
      residual,
      fn,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      sinkhorn_repeat,
  )


def mhc_post(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    implementation: _ImplementationArg = None,
) -> jax.Array:
  """mHC post step, run after each sublayer.

  Writes the sublayer output back into the residual streams:
  `new_residual[j] = post_layer_mix[j] * x + sum_i comb_res_mix[i, j] *
  residual[i]`.

  Args:
    x: `(T, H)` bf16 sublayer output.
    residual: `(T, M * H)` bf16 flat residual streams.
    post_layer_mix: `(T, M)` f32 post gates.
    comb_res_mix: `(T, M, M)` f32 stream-mixing matrix.
    implementation: See `mhc_pre`.

  Returns:
    The new `(T, M * H)` bf16 flat residual streams.

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise `NotImplementedError`.
  """
  return _dispatch(
      POST_IMPLEMENTATIONS,
      implementation,
      x,
      residual,
      post_layer_mix,
      comb_res_mix,
  )


def mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    implementation: _ImplementationArg = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """One layer's `mhc_post` followed by the next layer's `mhc_pre`.

  Equivalent to `mhc_pre(mhc_post(x, residual, post_layer_mix, comb_res_mix),
  fn, ...)`, but the Pallas implementation makes one pass over the residual
  streams instead of two.

  Args:
    x: `(T, H)` bf16 sublayer output.
    residual: `(T, M * H)` bf16 flat residual streams.
    post_layer_mix: `(T, M)` f32 post gates of the finished sublayer.
    comb_res_mix: `(T, M, M)` f32 stream-mixing matrix of the finished sublayer.
    fn: `(M * (M + 2), M * H)` f32 gate projection of the next sublayer.
    hc_scale: `(3,)` f32 logit scales of the next sublayer.
    hc_base: `(M * (M + 2),)` f32 logit biases of the next sublayer.
    rms_eps: RMS-norm epsilon.
    hc_pre_eps: Additive floor on the sigmoid pre gates.
    hc_sinkhorn_eps: Sinkhorn epsilon.
    hc_post_mult_value: Scale of the sigmoid post gates.
    sinkhorn_repeat: Number of Sinkhorn normalization rounds.
    implementation: See `mhc_pre`.

  Returns:
    `(residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur)`: the
    `mhc_post` output, then the three `mhc_pre` outputs computed from it.

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise `NotImplementedError`.
  """
  return _dispatch(
      FUSED_POST_PRE_IMPLEMENTATIONS,
      implementation,
      x,
      residual,
      post_layer_mix,
      comb_res_mix,
      fn,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      sinkhorn_repeat,
  )
