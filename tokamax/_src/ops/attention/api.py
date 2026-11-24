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
"""Dot product attention API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal
import immutabledict
import jax
from jax.extend import backend
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import qwix
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import jax_nn
from tokamax._src.ops.attention import xla_chunked

QArray = qwix.QArray
type Implementation = Literal[
    "mosaic",
    "mosaic_tpu",
    "mosaic_gpu",
    "triton",
    "cudnn",
    "xla",
    "xla_chunked",
]

# TODO: Investigate if `_XLA_CHUNK_SIZE` should be larger on TPU.
_XLA_CHUNK_SIZE: Final[tuple[int, int]] = (512, 1024)

_IMPLEMENTATIONS = dict(
    cudnn=jax_nn.JaxNnDotProductAttention(implementation="cudnn"),
    xla=base.DotProductAttention(),
    xla_chunked=xla_chunked.XlaChunkedDotProductAttention(
        chunk_size=_XLA_CHUNK_SIZE
    ),
)
# TODO: Investigate if xla_chunked be used instead of xla for very
# big sequences lengths. Eg. where xla OOMs.
_DEFAULT_IMPLEMENTATIONS = ("xla",)

try:
  from tokamax._src.ops.attention import triton  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["triton"] = triton.TritonFlashAttention()
  _DEFAULT_IMPLEMENTATIONS = ("triton",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

try:
  from tokamax._src.ops.attention import pallas_mosaic_gpu as pl_mgpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_gpu"] = pl_mgpu.PallasMosaicGpuFlashAttention()
  _DEFAULT_IMPLEMENTATIONS = ("mosaic",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass


try:
  from tokamax._src.ops.attention import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS["mosaic_tpu"] = (
      pallas_mosaic_tpu.PallasMosaicTpuFlashAttention()
  )
  if "mosaic" not in _DEFAULT_IMPLEMENTATIONS:
    _DEFAULT_IMPLEMENTATIONS = ("mosaic",) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


@jaxtyping.jaxtyped
def dot_product_attention(
    query: Float[Array | QArray, "*B T N H"],
    key: Float[Array | QArray, "*B S K H"],
    value: Float[Array | QArray, "*B S K h"],
    bias: Float[Array, "*#B #N #T #S"] | None = None,
    mask: Bool[Array, "*#B #N #T #S"] | None = None,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths: Int[Array, "*#B"] | None = None,
    key_value_seq_lengths: Int[Array, "*#B"] | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    max_logit_value: (
        float | Float[Array, ""] | Float[Array, "*#B #N"] | None
    ) = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: (
        Implementation
        | Callable[..., Any]
        | Sequence[Implementation | Callable[..., Any]]
        | None
    ) = None,
    q_sharding: jax.sharding.NamedSharding | None = None,
    k_sharding: jax.sharding.NamedSharding | None = None,
) -> Float[Array, "*B T N h"]:  # pylint: disable=g-doc-args
  """Scaled dot product attention function.

  See `jax.nn.dot_product_attention` for more details.

  The `jax.nn.dot_product_attention` API is extended here as follows:
    - Support for additional batch dimensions.
    - Support for quantized `query`, `key`, and `value` arrays.
    - Support for different output head dimension (`h`).

  Memory usage: for `implementation='xla'`, memory usage scales
  quadratically in sequence length, whilst scaling linearly
  for all other implementations. `implementation='xla_chunked'` is taken from
  https://arxiv.org/abs/2112.05682, and has similar memory usage to
  `implementation='triton'` and `implementation='mosaic'`, but will typically
  be slower than `implementation='xla'`.

  Args:
    logits_soft_cap: If not `None`, perform `logits = logits_soft_cap *
      tanh(logits / logits_soft_cap)`, where `logits` are `scale * query @ key.T
      + bias`.
    max_logit_value: Optional scalar or per-head fixed softmax shift
      (upper-bound estimate of unmasked logits) replacing the dynamic row-max
      reduction (`exp(logits - max_logit_value)`). Values far below the actual
      maximum logit can overflow `float32` `exp`.
    precision: The precision to use for the dot products.
    implementation: The implementation to use. By default, `None` is used, which
      will automatically select the best available backend, and is guaranteed to
      work on all platforms. If a sequence is passed, the first implementation
      that doesn't raise a `NotImplementedError` is used.
    q_sharding: NamedSharding for `q`. Sharding information for the other
      tensors will be inferred by this.
    k_sharding: NamedSharding for `k` and `v`. If `None`, `q_sharding` is used
      to derive KV head sharding. Requires `q_sharding` to also be specified.

  Returns:
    An array of the attention output.

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise `NotImplementedError`.
  """
  is_default = implementation is None
  if is_default:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str) or callable(implementation):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  if query_seq_lengths is not None:
    query_seq_lengths = query_seq_lengths[..., None, None]
  if key_value_seq_lengths is not None:
    key_value_seq_lengths = key_value_seq_lengths[..., None, None]

  mask_ = base.Mask(
      mask,
      is_causal=is_causal,
      q_end=query_seq_lengths,
      k_end=key_value_seq_lengths,
  )

  if isinstance(local_window_size, int):
    local_window_size = (local_window_size, local_window_size)

  if local_window_size is not None:
    pre, post = local_window_size
    mask_ &= base.CausalMask(offset=post, window_size=post + pre + 1)

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      orig_impl = impl
      if impl == "mosaic":
        device_kind = (
            op.infer_device_kind_from_args(
                (query, key, value, q_sharding, k_sharding)
            )
            or backend.get_default_device().device_kind
        )
        impl = "mosaic_gpu" if "NVIDIA" in device_kind else "mosaic_tpu"
      if impl not in IMPLEMENTATIONS:
        if is_default or (orig_impl == "mosaic" and len(implementation) > 1):
          errors.append(
              NotImplementedError(f"Implementation '{impl}' is not registered.")
          )
          continue
        raise ValueError(f"Unknown implementation: {impl}")
      impl = IMPLEMENTATIONS[impl]

    extra_kwargs: dict[str, Any] = {}
    if k_sharding is not None or isinstance(impl, base.DotProductAttention):
      extra_kwargs["k_sharding"] = k_sharding
    if max_logit_value is not None or isinstance(
        impl, base.DotProductAttention
    ):
      extra_kwargs["max_logit_value"] = max_logit_value
    try:
      return impl(
          query,
          key,
          value,
          bias=bias,
          mask=mask_,
          logits_scale=base.AUTO if scale is None else scale,
          precision=precision,
          logits_soft_cap=logits_soft_cap,
          q_sharding=q_sharding,
          **extra_kwargs,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
