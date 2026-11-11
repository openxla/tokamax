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
"""Strided VMEM compact-to-large and large-to-compact relayout helpers."""

# Dimensions.
#   tb   token rows in the VMEM tile buffer
#   D    channel dimension
#   P    sublane packing factor (4 // itemsize; 1 for f32, 2 for bf16)

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# NOTE: When performing strided ldst on TPU, `pltpu.bitcast` on VREG data should
# be avoided because it can trigger unintended relayout; bitcast the VMEM ref
# once outside the row loop instead.


def load_large_to_compact(
    # Inputs.
    vmem_ref,  # [tb, D] src_dtype, one VMEM tile
    dst_dtype: jax.typing.DTypeLike = jnp.float32,
    *,
    interpret: bool = False,
) -> jax.Array:
  """Loads a 2D `(tb, D)` VMEM ref into compact `(tb, 1, D)` VREGs."""
  if vmem_ref.ndim != 2:
    raise ValueError(f"Expected 2D vmem_ref, got shape {vmem_ref.shape}.")

  resolved_dst_dtype = jnp.dtype(dst_dtype)
  tb, _ = vmem_ref.shape
  src_dtype = jnp.dtype(vmem_ref.dtype)
  if interpret:
    return jnp.stack(
        [vmem_ref[i : i + 1].astype(resolved_dst_dtype) for i in range(tb)],
        axis=0,
    )

  should_unpack = resolved_dst_dtype != src_dtype
  if should_unpack and (
      src_dtype.itemsize != 2 or resolved_dst_dtype.itemsize != 4
  ):
    raise ValueError(
        "Sublane unpacking requires 16-bit src and 32-bit dst, got"
        f" {src_dtype} -> {resolved_dst_dtype}."
    )
  packing = 4 // src_dtype.itemsize
  if not should_unpack and packing != 1:
    raise ValueError(
        "Unpacked compact layout requires 32-bit dtype when dst_dtype =="
        f" src_dtype, got {src_dtype}."
    )
  if tb % packing != 0:
    raise ValueError(
        f"Expected vmem_ref row extent ({tb}) to be divisible by "
        f"packing ({packing})."
    )

  unpacked_list = []
  if should_unpack:
    u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
    for row_start in range(0, tb, packing):
      packed_row = row_start // packing
      packed = u32_vmem_ref[packed_row : packed_row + 1]
      for i in range(packing):
        unpacked = pltpu.unpack_elementwise(
            packed,
            index=i,
            packed_dtype=src_dtype,
            unpacked_dtype=resolved_dst_dtype,
        )
        unpacked_list.append(unpacked)
  else:
    for row_start in range(0, tb, packing):
      row_end = row_start + packing
      unpacked_list.append(vmem_ref[row_start:row_end])

  return jnp.stack(unpacked_list, axis=0)


def store_compact_to_large(
    # Outputs.
    vmem_ref,  # [tb, D] dst_dtype, one VMEM tile
    # Inputs.
    vreg: jax.Array,  # [tb, P, D] src_dtype, compact VREG array
    *,
    interpret: bool = False,
) -> None:
  """Stores compact `(tb, 1, D)` VREGs into a 2D `(tb, D)` VMEM ref."""
  src_dtype = jnp.dtype(vreg.dtype)
  dst_dtype = jnp.dtype(vmem_ref.dtype)
  src_packing = 4 // src_dtype.itemsize
  dst_packing = 4 // dst_dtype.itemsize

  if vreg.ndim != 3:
    raise ValueError(
        f"Expected 3D compact vreg with shape[-2]=={src_packing}, "
        f"got {vreg.shape}."
    )
  vreg_rows, vreg_pack, vreg_dim = vreg.shape
  if vreg_pack != src_packing:
    raise ValueError(
        f"Expected 3D compact vreg with shape[-2]=={src_packing}, "
        f"got {vreg.shape}."
    )
  if vmem_ref.ndim != 2 or vmem_ref.shape != (vreg_rows, vreg_dim):
    raise ValueError(
        f"Expected 2D vmem_ref with shape ({vreg_rows}, {vreg_dim}), "
        f"got {vmem_ref.shape}."
    )
  tb, _ = vmem_ref.shape

  if interpret:
    for i in range(tb):
      vmem_ref[i : i + 1] = vreg[i].astype(dst_dtype)
    return

  should_pack = src_dtype != dst_dtype
  if should_pack and (src_dtype.itemsize != 4 or dst_dtype.itemsize != 2):
    raise ValueError(
        "Sublane packing requires 32-bit src and 16-bit dst, got"
        f" {src_dtype} -> {dst_dtype}."
    )
  if not should_pack and dst_packing != 1:
    raise ValueError(
        "Unpacked compact layout requires 32-bit dtype when src_dtype =="
        f" dst_dtype, got {dst_dtype}."
    )
  if tb % dst_packing != 0:
    raise ValueError(
        f"Expected vmem_ref row extent ({tb}) to be divisible by "
        f"packing ({dst_packing})."
    )

  if should_pack:
    u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
    for row_start in range(0, tb, dst_packing):
      row_end = row_start + dst_packing
      packed_row = row_start // dst_packing
      unpacked_list = [vreg[i] for i in range(row_start, row_end)]
      packed = pltpu.pack_elementwise(unpacked_list, packed_dtype=dst_dtype)
      u32_vmem_ref[packed_row : packed_row + 1] = packed
  else:
    for row_start in range(0, tb, dst_packing):
      row_end = row_start + dst_packing
      packed_row = row_start // dst_packing
      vmem_ref[row_start:row_end] = vreg[packed_row]
