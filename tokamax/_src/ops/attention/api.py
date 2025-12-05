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
from typing import Any, Final, Literal, TypeAlias
import immutabledict
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import qwix
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import jax_nn
from tokamax._src.ops.attention import xla_chunked

QArray = qwix.QArray
Implementation: TypeAlias = Literal[
    "mosaic", "triton", "cudnn", "xla", "xla_chunked"
]

# TODO: Investigate if `_XLA_CHUNK_SIZE` should be larger on TPU.
_XLA_CHUNK_SIZE: Final[int] = 128

IMPLEMENTATIONS = dict(
    cudnn=jax_nn.JaxNnDotProductAttention(implementation="cudnn"),
    xla=base.DotProductAttention(),
    xla_chunked=xla_chunked.XlaChunkedDotProductAttention(
        chunk_size=_XLA_CHUNK_SIZE
    ),
)
# TODO: Investigate if xla_chunked be used instead of xla for very
# big sequences lengths. Eg. where xla OOMs.
_DEFAULT_IMPLEMENTATION = ("xla",)

try:
  from tokamax._src.ops.attention import pallas_triton as pl_triton  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["triton"] = pl_triton.PallasTritonFlashAttention()
  _DEFAULT_IMPLEMENTATION = ("triton",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass

try:
  from tokamax._src.ops.attention import pallas_mosaic_gpu as pl_mgpu  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  IMPLEMENTATIONS["mosaic"] = pl_mgpu.PallasMosaicGpuFlashAttention()
  _DEFAULT_IMPLEMENTATION = ("mosaic",) + _DEFAULT_IMPLEMENTATION
except ImportError:
  pass


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(IMPLEMENTATIONS)
)


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
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
    q_sharding: jax.sharding.NamedSharding | None = None,
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
    precision: The precision to use for the dot products.
    implementation: The implementation to use. By default, `None` is used, which
      will automatically select the best available backend, and is guaranteed to
      work on all platforms. If a sequence is passed, the first implementation
      that doesn't raise a `NotImplementedError` is used.
    q_sharding: NamedSharding for `q`. Sharding information for the other
      tensors will be inferred by this.

  Returns:
    An array of the attention output.

  Raises:
    ExceptionGroup: If multiple implementations are provided and all of them
      raise `NotImplementedError`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATION

  if not isinstance(implementation, (tuple, list)):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  if scale is None:
    scale = base.AUTO

  if query_seq_lengths is not None:
    query_seq_lengths = query_seq_lengths[:, None, None]
  if key_value_seq_lengths is not None:
    key_value_seq_lengths = key_value_seq_lengths[:, None, None]

  mask = base.Mask(
      mask,
      is_causal=is_causal,
      q_end=query_seq_lengths,
      k_end=key_value_seq_lengths,
  )

  if isinstance(local_window_size, int):
    local_window_size = (local_window_size, local_window_size)

  if local_window_size is not None:
    k_indices = jnp.arange(key.shape[-3])
    before, after = local_window_size
    mask &= base.Mask(k_start=k_indices - before, k_end=k_indices + (after + 1))

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {impl}")
      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          query,
          key,
          value,
          bias=bias,
          mask=mask,
          logits_scale=scale,
          precision=precision,
          logits_soft_cap=logits_soft_cap,
          q_sharding=q_sharding,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
