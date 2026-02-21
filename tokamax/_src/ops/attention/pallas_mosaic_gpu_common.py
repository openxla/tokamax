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
"""Common utilities for Mosaic GPU attention implementations."""

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import llvm
import pydantic


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=dict(extra="forbid")
)  # pytype: disable=wrong-keyword-args
class ConfigBase:
  """Common configuration parameters for Pallas-Mosaic-GPU kernels.

  Attributes:
    block_q: Block size along Q sequence length.
    num_stages: Number of tma stages for loading KV.
    fold_q_sequence_heads: Whether to fold seq_q into num_q_heads.
    split_k: Number of chunks to split seq_len_k into to improve parallelism.
  """

  # TODO: Relax block size constraints to multiple of 32.
  block_q: pydantic.conint(multiple_of=64, gt=0) = 64
  block_kv: pydantic.conint(multiple_of=64, gt=0) = 64
  num_stages: pydantic.PositiveInt = 2
  fold_q_sequence_heads: pydantic.StrictBool = False
  split_k: pydantic.PositiveInt = 1

  def __post_init__(self):
    if type(self) is ConfigBase:  # pylint: disable=unidiomatic-typecheck
      raise ValueError("Cannot use ConfigBase directly. Use a subclass.")


def decompose_mask(mask, q, k, q_indices, k_indices):
  """Decomposes `mask` into a mask array, `is_causal`, `k_start` and `k_end`."""
  if mask is None:
    return None, False, None, None

  is_causal = False
  k_start = None
  k_end = None

  if k_indices is None:
    mask, is_causal, k_start, k_end = mask.take("is_causal", "k_start", "k_end")

    # Fold `is_causal` into `k_end`. If `q_indices` is not `None`, then this is
    # necessary for correctness. Otherwise, it is a performance optimization.
    if is_causal and (q_indices is not None or k_end is not None):
      if q_indices is None:
        q_indices = jnp.arange(q.shape[-3])
      k_end_ = q_indices + 1
      k_end = k_end_ if k_end is None else jnp.minimum(k_end, k_end_)
      is_causal = False

    if k_start is not None:
      k_start = jax.lax.broadcast_to_rank(k_start, 2)
    if k_end is not None:
      k_end = jax.lax.broadcast_to_rank(k_end, 2)

  q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
  k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
  mask = mask.as_array(q_len_or_indices, k_len_or_indices)
  return mask, is_causal, k_start, k_end


def load_bcast(
    ref: Any,
    idx: tuple[int | jax.Array | pl.Slice, ...],
    *,
    layout: Any,
    optimized: bool = False,
) -> jax.Array:
  """Loads from a reference, with given index, broadcasting if needed."""
  new_idx = []
  shape = []
  bcast_dims = []
  ld_layout = layout
  # NOTE: We could add support for `idx` shorter than `ref.ndim`.
  for d, ix in zip(ref.shape, idx, strict=True):
    new_idx.append(0 if d == 1 else ix)

    if isinstance(ix, pl.Slice):
      if d == 1:
        ld_layout = ld_layout.reduce(len(shape))
      else:
        bcast_dims.append(len(shape))
      shape.append(ix.size)

  new_idx = tuple(new_idx)
  if bcast_dims:
    value = plgpu.load(ref, new_idx, layout=ld_layout, optimized=optimized)
  else:  # Scalar value.
    value = ref[new_idx]
  value = jax.lax.broadcast_in_dim(value, shape, bcast_dims)
  return value if layout is None else plgpu.layout_cast(value, layout)


def num_bits(dtype: jax.typing.DTypeLike) -> int:
  fn = jnp.finfo if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo
  return fn(dtype).bits


def tile_swizzle_transforms(
    shape: tuple[int, ...], dtype: jax.typing.DTypeLike, what: str = ""
) -> tuple[plgpu.TilingTransform, plgpu.SwizzleTransform]:
  """Returns tiling and swizzling transforms."""
  elem_bits = num_bits(dtype)
  swizzle = plgpu.find_swizzle(shape[-1] * elem_bits, what)
  tiling = (8, 8 * swizzle // elem_bits)
  return plgpu.TilingTransform(tiling), plgpu.SwizzleTransform(swizzle)


def _bar_operation(operation: str, barrier_id: jax.Array, num_threads: int):
  @plgpu.inline_mgpu(arg_types=(plgpu.Layout.WG_SPLAT,))
  def bar_op(_, barrier_id):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [barrier_id.registers[()]],
        f"bar.{operation} $0, {num_threads};",
        "r",
        has_side_effects=True,
    )

  bar_op(barrier_id)


bar_arrive = functools.partial(_bar_operation, "arrive")
bar_sync = functools.partial(_bar_operation, "sync")
