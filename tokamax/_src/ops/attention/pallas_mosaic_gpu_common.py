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

from typing import Annotated, Any

import jax
from jax.experimental import pallas as pl
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import vector
import numpy as np
import pydantic
import qwix
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import precision as precision_lib
from tokamax._src import shape as shape_lib


CanonicalPrecision = precision_lib.CanonicalPrecision
QArray = qwix.QArray


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=dict(extra="forbid")
)  # pytype: disable=wrong-keyword-args
class ConfigBase:
  """Common configuration parameters for Pallas-Mosaic-GPU kernels.

  Attributes:
    block_q: Block size along Q sequence length.
    block_kv: Block size along K/V sequence length.
    num_stages: Number of tma stages for loading KV.
    fold_q_sequence_heads: Whether to fold seq_q into num_q_heads.
    split_k: Number of chunks to split seq_len_k into to improve parallelism.
  """

  # TODO: Relax block size constraints to multiple of 32.
  block_q: Annotated[int, pydantic.Field(multiple_of=64, gt=0)] = 64
  block_kv: Annotated[int, pydantic.Field(multiple_of=64, gt=0)] = 64
  num_stages: pydantic.PositiveInt = 2
  fold_q_sequence_heads: pydantic.StrictBool = False
  split_k: pydantic.PositiveInt = 1

  def __post_init__(self):
    if type(self) is ConfigBase:  # pylint: disable=unidiomatic-typecheck
      raise ValueError("Cannot use ConfigBase directly. Use a subclass.")


MIN_SWIZZLE = 32


# The contracting dimension for `wgmma` / `tcgen05.mma` must be a multiple of
# the minimum swizzle size (in number of elements).
def pad_head_dim_to_next_multiple_of_min_swizzle(x):
  m = 8 * MIN_SWIZZLE // mgpu_lib.num_bits(x.dtype)
  return shape_lib.pad_to_next_multiple_of(x, m, -1)


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


def cast_qkv(
    q: jax.Array | QArray,
    k: jax.Array | QArray,
    v: jax.Array | QArray,
    precision: tuple[CanonicalPrecision, CanonicalPrecision],
) -> tuple[jax.Array | QArray, jax.Array | QArray, jax.Array | QArray]:
  """Casts Q, K, and V to the given precision."""

  def cast(x, precision):
    # Quantized arrays inherently define their precision (e.g., int8) and
    # cannot contain infinities, so we bypass sanitization and downcasting.
    if isinstance(x, QArray):
      return x
    assert precision != jax.lax.DotAlgorithmPreset.DEFAULT
    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      return safe_downcast(x, jnp.bfloat16)
    if precision == jax.lax.DotAlgorithmPreset.F16_F16_F32:
      return safe_downcast(x, jnp.float16)
    raise NotImplementedError(f"Unsupported precision: {precision}")

  q_k_dot_precision, p_v_dot_precision = precision
  q_k_dot_precision = precision_lib.to_dot_algorithm_preset(
      q.dtype, k.dtype, q_k_dot_precision
  )
  p_v_dot_precision = precision_lib.to_dot_algorithm_preset(
      v.dtype, v.dtype, p_v_dot_precision
  )
  q = cast(q, q_k_dot_precision)
  k = cast(k, q_k_dot_precision)
  v = cast(v, p_v_dot_precision)
  return q, k, v


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


def unpack_bool_bits_tmem_native(a):
  """Unpacks boolean bits from an int packed array in TMEM_NATIVE layout."""
  packed_bits = mgpu_lib.num_bits(a.dtype)
  if packed_bits not in {4, 8, 16}:
    raise ValueError("Only 4, 8, 16 boolean packing is supported")
  target_cols = a.shape[1] * packed_bits
  out_layout = plgpu.Layout.TCGEN05_TMEM_NATIVE
  out_mgpu_layout = out_layout.to_mgpu()

  @plgpu.inline_mgpu(
      arg_types=(plgpu.Layout.TCGEN05_TMEM_NATIVE(32 // packed_bits),),
      return_type=plgpu.ShapeDtypeStruct(
          (128, target_cols),
          jnp.bool_,
          out_layout,
      ),
  )
  def unpack_booleans(_, fa: mgpu.FragmentedArray):
    out_registers = np.empty(
        out_mgpu_layout.registers_shape((128, target_cols)),
        dtype=object,
    )
    i32_type = ir.IntegerType.get_signless(32)
    out_i32_ty = ir.VectorType.get([2], i32_type)
    zero_i32_vec = vector.broadcast(out_i32_ty, mgpu.c(0, i32_type))

    for (row_idx, col_idx, *_), reg_a in np.ndenumerate(fa.registers):
      vec1_i32_ty = ir.VectorType.get([1], i32_type)
      reg_v1_i32 = vector.bitcast(vec1_i32_ty, reg_a)
      reg_i32_scalar = vector.extract(
          reg_v1_i32,
          dynamic_position=[],
          static_position=ir.DenseI64ArrayAttr.get([0]),
      )
      reg_v2_i32 = vector.broadcast(out_i32_ty, reg_i32_scalar)

      for i in range(16):
        mask_low = 1 << (i * 2)
        mask_high = 1 << (i * 2 + 1)
        base_mask = vector.broadcast(out_i32_ty, mgpu.c(mask_low, i32_type))
        bit_mask = vector.insert(
            mgpu.c(mask_high, i32_type),
            base_mask,
            dynamic_position=[],
            static_position=ir.DenseI64ArrayAttr.get([1]),
        )
        and_res = arith.andi(reg_v2_i32, bit_mask)
        cmp_res = arith.cmpi(arith.CmpIPredicate.ne, and_res, zero_i32_vec)
        logical_col = col_idx * 16 + i
        out_registers[row_idx, logical_col] = cmp_res

    return mgpu.FragmentedArray(
        _registers=out_registers,
        _layout=out_mgpu_layout,
        _is_signed=False,
    )

  return unpack_booleans(a)

def safe_downcast(
    arr: jax.Array, target_dtype: jax.typing.DTypeLike
) -> jax.Array:
  """Clips the array to the target dtype's range before casting to prevent infinities."""
  if arr.dtype == target_dtype:
    return arr
  finfo = jnp.finfo(target_dtype)
  return jnp.clip(arr, finfo.min, finfo.max).astype(target_dtype)
