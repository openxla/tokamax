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

"""Linear Cross-Entropy kernel implementation."""

import functools
import math
from typing import Annotated, Literal
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import pydantic

Array = jt.Array
Integer = jt.Integer
Real = jt.Real
Scalar = jt.Scalar

partial = functools.partial
reduce = functools.reduce


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """The configuration specific for the Pallas Mosaic TPU kernel.

  Attributes:
    b_block_size: The block size for the batch dimension.
    h_block_size: The block size for the hidden dimension.
    v_block_size: The block size for the vocabulary dimension.
  """

  b_block_size: Annotated[int, pydantic.Field(ge=1024, multiple_of=128)] = 1024
  h_block_size: Annotated[int, pydantic.Field(ge=128, multiple_of=128)] = 512
  v_block_size: Annotated[int, pydantic.Field(ge=128, multiple_of=128)] = 2048


def _calculate_fwd_vmem_bytes(
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    dtype: jnp.dtype = jnp.float32,
) -> int:
  """Calculates VMEM memory usage in bytes for the forward kernel."""
  dtype_bytes = jnp.dtype(dtype).itemsize
  h_alloc = 1 << (h_block_size - 1).bit_length()
  num_bytes = (
      # x tile (B, H) in input dtype (double buffered for pallas_call input)
      2 * b_block_size * h_alloc * dtype_bytes
      # labels tile (B,) in int32/float32 (double buffered)
      + 2 * b_block_size * 4
      # w tile (H, V) in input dtype (double buffered for pallas_call input)
      + 2 * h_alloc * v_block_size * dtype_bytes
      # lse (log-sum-exp) buffer tile (B,) in float32 accumulator (double
      # buffered for pallas_call output)
      + 2 * b_block_size * 4
      # logits/xw tile (B, V) in float32 accumulator (single scratch buffer)
      + b_block_size * v_block_size * 4
  )
  return num_bytes


def _calculate_bwd_vmem_bytes(
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    dtype: jnp.dtype = jnp.float32,
) -> int:
  """Calculates VMEM memory usage in bytes for the backward kernel."""
  dtype_bytes = jnp.dtype(dtype).itemsize
  h_alloc = 1 << (h_block_size - 1).bit_length()
  num_bytes = (
      # x tile (B, H) in input dtype (double buffered for pallas_call input)
      2 * b_block_size * h_alloc * dtype_bytes
      # labels tile (B,) in int32/float32 (double buffered for pallas_call
      # input)
      + 2 * b_block_size * 4
      # w tile (H, V) in input dtype (double buffered for pallas_call input)
      + 2 * h_alloc * v_block_size * dtype_bytes
      # lse (log-sum-exp) buffer tile (B,) in float32 (double buffered for
      # pallas_call input)
      + 2 * b_block_size * 4
      # logits/softmax tile (B, V) in float32 accumulator:
      # We account for 3 simultaneous (B, V) float32 buffers (3 * 4 = 12
      # bytes/elem):
      #   1) xw_scratch_ref (explicit VMEM scratch buffer)
      #   2) labels_one_hot (HLO stack temporary from jax.nn.one_hot in
      #      compute_s)
      #   3) jnp.exp(...) (HLO stack temporary during softmax in compute_s)
      + 3 * b_block_size * v_block_size * 4
      # x gradient tile accumulator (B, H) in float32 (single scratch buffer)
      + b_block_size * h_alloc * 4
      # w gradient tile accumulator (H, V) in float32 (single scratch buffer)
      + h_alloc * v_block_size * 4
  )
  return num_bytes


def _get_vmem_limit_bytes() -> int:
  return int(0.5 * pltpu.get_tpu_info().vmem_capacity_bytes)


def _get_heuristic_config(
    b_dim: int,
    h_dim: int,
    v_dim: int,
    *,
    is_bwd: bool = False,
    dtype: jnp.dtype = jnp.float32,
    vmem_limit_bytes: int | None = None,
) -> Config:
  """Calculates heuristic config based on VMEM size, dtype, and divisibility."""
  if vmem_limit_bytes is None:
    vmem_limit_bytes = _get_vmem_limit_bytes()

  calc_vmem_fn = (
      _calculate_bwd_vmem_bytes if is_bwd else _calculate_fwd_vmem_bytes
  )
  dtype_bytes = jnp.dtype(dtype).itemsize

  # 1. Choose b_block_size: needs at least 1024, multiple of 128.
  # Prefer b_block_size such that b_dim % b_block_size == 0.
  b_candidates = [
      b for b in range(1024, max(1025, b_dim + 1), 128) if b_dim % b == 0
  ]
  if b_candidates:
    b_block_size = b_candidates[0]
  else:
    b_block_size = 1024

  # 2. Choose h_block_size: at least 128, multiple of 128, close to h_dim_max.
  # Prefer h_block_size such that h_dim % h_block_size == 0.
  h_dim_max = (
      pltpu.get_tpu_info().mxu_column_size
      * pltpu.get_tpu_info().num_mxus
      // pltpu.get_tpu_info().num_cores
  )
  h_candidates = [
      h for h in range(128, max(129, h_dim + 1), 128) if h_dim % h == 0
  ]
  if h_candidates:
    h_block_size = min(
        h_candidates,
        key=lambda x: (
            0 if (x & (x - 1)) == 0 else 1,
            abs(x - h_dim_max),
        ),
    )
  else:
    h_block_size = h_dim_max if h_dim >= h_dim_max else 128

  # 3. Choose v_block_size: as large as possible to fit VMEM.
  # Must be >= 128, multiple of 128. Divisible by v_dim if possible.
  if is_bwd:
    fixed_bytes = (
        b_block_size * h_block_size * (2 * dtype_bytes + 4) + 16 * b_block_size
    )
    # per_v_bytes accounts for all VMEM costs per column of V:
    #   - w tile (double-buffered in dtype) + w_grad_tile (float32) =
    #     h_block_size * (2 * dtype_bytes + 4)
    #   - 3 simultaneous float32 (B, V) buffers on the VMEM stack during
    #     compute_s:
    #       1) xw_scratch_ref (explicit VMEM scratch)
    #       2) labels_one_hot (HLO stack temporary from jax.nn.one_hot)
    #       3) jnp.exp(...) (HLO stack temporary during softmax)
    #     Each float32 element is 4 bytes, so 3 * 4 = 12 bytes per element
    #     across b_block_size.
    per_v_bytes = h_block_size * (2 * dtype_bytes + 4) + 12 * b_block_size
  else:
    fixed_bytes = (
        2 * b_block_size * h_block_size * dtype_bytes + 16 * b_block_size
    )
    per_v_bytes = 2 * h_block_size * dtype_bytes + b_block_size * (
        4 + dtype_bytes
    )

  if vmem_limit_bytes > fixed_bytes:
    max_v_vmem = (vmem_limit_bytes - fixed_bytes) // per_v_bytes
  else:
    max_v_vmem = 128

  max_v = min(v_dim, max_v_vmem)
  max_v_aligned = max(128, (max_v // 128) * 128)

  # First try to find a divisor of v_dim that fits in VMEM
  v_block_size = None
  for v in range(max_v_aligned, 127, -128):
    if (
        v_dim % v == 0
        and calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
        <= vmem_limit_bytes
    ):
      v_block_size = v
      break

  # If no divisor fits, pick the largest multiple of 128 that fits
  if v_block_size is None:
    for v in range(max_v_aligned, 127, -128):
      if (
          calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
          <= vmem_limit_bytes
      ):
        v_block_size = v
        break
    if v_block_size is None:
      v_block_size = 128

  return Config(
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )


def get_heuristic_fwd_config(
    b_dim: int,
    h_dim: int,
    v_dim: int,
    dtype: jnp.dtype = jnp.float32,
    vmem_limit_bytes: int | None = None,
) -> Config:
  """Returns heuristic config for forward pass based on VMEM size and dtype."""
  return _get_heuristic_config(
      b_dim,
      h_dim,
      v_dim,
      is_bwd=False,
      dtype=dtype,
      vmem_limit_bytes=vmem_limit_bytes,
  )


def get_heuristic_bwd_config(
    b_dim: int,
    h_dim: int,
    v_dim: int,
    dtype: jnp.dtype = jnp.float32,
    vmem_limit_bytes: int | None = None,
) -> Config:
  """Returns heuristic config for backward pass based on VMEM size and dtype."""
  return _get_heuristic_config(
      b_dim,
      h_dim,
      v_dim,
      is_bwd=True,
      dtype=dtype,
      vmem_limit_bytes=vmem_limit_bytes,
  )


def validate_inputs(
    x: Real[Array, "B H"],
    labels: Real[Array, "B V"],
    w: Real[Array, "H V"],
    b_block_size: int,
    h_block_size: int,  # pylint: disable=unused-argument
    v_block_size: int,  # pylint: disable=unused-argument
):
  """Validates the inputs to the kernels.

  Validate inputs and raise ValueError if the inputs are invalid.

  Args:
    x: The last layer output in the dimension of (B, H) where B is the batch
      dimension, and H is the hidden dimension.
    labels: The ground truth label index in the dimension of (B,).
    w: The linear projection weight matrix in the dimension of (H, V) where V is
      the dimension of the output logits aka vocabulary size.
    b_block_size: The batch block size.
    h_block_size: The hidden block size.
    v_block_size: The vocabulary block size.

  Raises:
    ValueError: If the inputs are invalid.
  """
  if x.shape[0] % b_block_size != 0:
    raise ValueError(
        "The batch dimension of x must be a multiple of the B block size."
    )
  if labels.shape[0] % b_block_size != 0:
    raise ValueError(
        "The batch dimension of labels must be a multiple of the B block size."
    )

  if w.shape[0] % 8 != 0:
    raise ValueError("The hidden dimension of w must be a multiple of 8")


def calculate_xw_tiled(
    x_ref,
    w_ref,
    xw_tiled,
    h_index,
    v_index,
    num_h_blocks,
    num_v_blocks,
    h_dim,
    v_dim,
    preferred_element_type: jnp.dtype,
):
  """Calculates xw_tiled += x@w for forward/backward kernel common logic."""
  h_block_size, v_block_size = x_ref.shape[1], w_ref.shape[1]
  # Padding if V dimension is not aligned to the V block size
  if v_dim % v_block_size != 0:

    @pl.when(v_index == num_v_blocks - 1)
    def pad_non_aligned_v_block():
      rem = v_dim % v_block_size
      iota_mask = jax.lax.broadcasted_iota(
          dtype=jnp.int32, shape=w_ref.shape, dimension=1
      )
      w_ref[...] = jnp.where(iota_mask < rem, w_ref[...], 0)

  # Padding if H dimension is not aligned to the V block size
  if h_dim % h_block_size != 0:

    @pl.when(h_index == num_h_blocks - 1)
    def pad_non_aligned_h_block():
      rem = h_dim % h_block_size
      x_iota_mask = jax.lax.broadcasted_iota(
          dtype=jnp.int32, shape=x_ref.shape, dimension=1
      )
      x_ref[...] = jnp.where(x_iota_mask < rem, x_ref[...], 0)

      w_iota_mask = jax.lax.broadcasted_iota(
          dtype=jnp.int32, shape=w_ref.shape, dimension=0
      )
      w_ref[...] = jnp.where(w_iota_mask < rem, w_ref[...], 0)

  @pl.when(h_index == 0)
  def init_xw():
    xw_tiled[...] = jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=preferred_element_type,
    )

  @pl.when(h_index != 0)
  def accumulate_xw():
    xw_tiled[...] += jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=preferred_element_type,
    )


def linear_softmax_cross_entropy_loss_forward_pallas_kernel(
    x,
    labels,
    w,
    *,
    reduction: Literal["sum", "mean", "none"],
    h_dim: int,
    v_dim: int,
    preferred_element_type: jnp.dtype,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
) -> tuple[Real[Scalar, ""] | Real[Array, "B"], Real[Array, "B"]]:
  """Pallas kernel for the forward pass of Linear Softmax Cross-Entropy Loss.

  This kernel uses a block-wise algorithm on all B, H and V dimensions. The B
  and H dimensions can be accumulated linearly. The accumulation
  on V dimension is using the log linearity of log-sum-exp and log-softmax.
  The kernel will return both loss and additionally the log-sum-exp for the
  backward pass. However the x@w won't be returned to avoid additional buffer so
  backward pass
  will need to re-compute x@w. Overall, this kernel will keep all the
  intermediate buffers in VMEM without logits HBM materialization.

  Args:
    x: Input activations `x` (b_dim, h_dim).
    labels: One-hot encoded labels (b_dim, v_dim).
    w: LM Head projection weights `w` (h_dim, v_dim).
    reduction: The reduction method ("sum", "mean" or "none") for the loss
      accumulation.
    h_dim: Hidden dimension size.
    v_dim: Vocabulary dimension size.
    preferred_element_type: Preferred element type for computation.
    b_block_size: Block size for batch dimension.
    h_block_size: Block size for hidden dimension.
    v_block_size: Block size for vocabulary dimension.

  Returns:
    A tuple of (loss, lse).
  """
  b_dim = x.shape[0]
  num_b_blocks = math.ceil(b_dim / b_block_size)
  num_h_blocks = math.ceil(h_dim / h_block_size)
  num_v_blocks = math.ceil(v_dim / v_block_size)

  out_type = [
      jax.ShapeDtypeStruct(shape=(b_dim,), dtype=jnp.float32),  # Loss
      jax.ShapeDtypeStruct(shape=(b_dim,), dtype=jnp.float32),  # LSE
  ]
  loss_out_spec = pl.BlockSpec(
      (b_block_size,), lambda i, j, k: i, memory_space=pltpu.VMEM
  )

  @pl.kernel(
      out_type=out_type,
      mesh=pltpu.TensorCoreMesh(axis_name="core"),
      scratch_types=(
          pltpu.VMEM(
              (b_block_size, v_block_size), dtype=jnp.float32
          ),  # xw_tiled
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_get_vmem_limit_bytes(),
          disable_bounds_checks=True,
      ),
      name=(
          f"lce_fwd_bt_{b_block_size}_ht_{h_block_size}_vt_{v_block_size}"
      ),
  )
  def fwd_kernel(
      x_hbm_ref,
      labels_hbm_ref,
      w_hbm_ref,
      loss_hbm_ref,
      lse_hbm_ref,
      xw_tiled_ref,
  ):
    def fwd_pipeline(
        x_ref,
        labels_ref,
        w_ref,
        loss_ref,
        lse_ref,
        xw_tiled,
    ):
      b_index, v_index, h_index = (pl.program_id(i) for i in range(3))
      unused_num_b_blocks, num_v_blocks, num_h_blocks = (
          pl.num_programs(i) for i in range(3)
      )

      # xw_tiled += x_ref @ w_ref
      calculate_xw_tiled(
          x_ref,
          w_ref,
          xw_tiled,
          h_index=h_index,
          v_index=v_index,
          num_h_blocks=num_h_blocks,
          num_v_blocks=num_v_blocks,
          h_dim=h_dim,
          v_dim=v_dim,
          preferred_element_type=preferred_element_type,
      )

      @pl.when(jnp.logical_and(v_index == 0, h_index == 0))
      def init_lse():
        lse_ref[...] = jnp.full_like(lse_ref, -jnp.inf)
        loss_ref[...] = jnp.zeros_like(loss_ref)

      @pl.when(h_index == num_h_blocks - 1)
      def accumulate_loss():
        # Convert labels to one-hot, due to chunking on v dimension, the indices
        # needs to be shifted down by the v starting index. Negative or
        # out-of-bound indices are OK since jax.nn.one_hot will set them to 0.
        labels_adjusted = labels_ref[...] - v_index * v_block_size
        labels_one_hot = jax.nn.one_hot(
            labels_adjusted, num_classes=v_block_size, dtype=x_ref.dtype
        )
        loss_ref[...] -= jnp.sum(labels_one_hot * xw_tiled[...], axis=-1)
        lse_block = jax.nn.logsumexp(xw_tiled[...], axis=-1)
        lse_ref[...] = jnp.logaddexp(lse_ref[...], lse_block)

      @pl.when(
          jnp.logical_and(
              v_index == num_v_blocks - 1, h_index == num_h_blocks - 1
          )
      )
      def perform_loss_reduction():
        loss_ref[...] += lse_ref[...]

    pltpu.emit_pipeline(
        fwd_pipeline,
        grid=(num_b_blocks, num_v_blocks, num_h_blocks),
        in_specs=[
            pl.BlockSpec(
                (b_block_size, h_block_size),
                lambda i, j, k: (i, k),
                memory_space=pltpu.VMEM,
            ),  # x
            pl.BlockSpec(
                (b_block_size,),
                lambda i, j, k: (i,),
                memory_space=pltpu.VMEM,
            ),  # labels
            pl.BlockSpec(
                (h_block_size, v_block_size),
                lambda i, j, k: (k, j),
                memory_space=pltpu.VMEM,
            ),  # w
        ],
        out_specs=[
            loss_out_spec,  # loss
            pl.BlockSpec(
                (b_block_size,), lambda i, j, k: i, memory_space=pltpu.VMEM
            ),  # lse
        ],
        # TODO: enable parallel core_axis_name for the kernel.
        # core_axis_name="core",
        dimension_semantics=(
            pltpu.ARBITRARY,
            pltpu.ARBITRARY,
            pltpu.ARBITRARY,
        ),
    )(
        x_hbm_ref,
        labels_hbm_ref,
        w_hbm_ref,
        loss_hbm_ref,
        lse_hbm_ref,
        scratches=(xw_tiled_ref,),
    )

  loss, lse = fwd_kernel(x, labels, w)  # pylint: disable=unpacking-non-sequence
  if reduction == "sum":
    return jnp.sum(loss), lse
  elif reduction == "mean":
    return jnp.mean(loss), lse
  else:
    return loss, lse


@partial(
    jax.jit,
    static_argnames=[
        "b_block_size",
        "h_block_size",
        "v_block_size",
        "reduction",
        "preferred_element_type",
    ],
)
def linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
    reduction: Literal["sum", "mean", "none"] = "sum",
    preferred_element_type: jnp.dtype = jnp.float32,
) -> tuple[Real[Scalar, ""] | Real[Array, "B"], Real[Array, "B"]]:
  """The pallas kernel implementation of linear softmax cross-entropy loss.

  This implementation is chunking the x, labels and w in all B, H and V
  dimensions so it can fit in the TPU VMEM, resulting in almost 0 additional
  buffer overhead. The V dimension chunking is non-linear so this kernel uses
  online softmax algorithm to chunk.

  Args:
    x: The last layer output in the dimension of (B, H) where B is the batch and
      H is the hidden dimension.
    labels: The ground truth labels index in the dimension of (B,)
    w: The linear projection weight matrix in the dimension of (H, V) where V is
      the dimension of the output logits aka vocabulary size.
    b_block_size: The batch block size.
    h_block_size: The hidden block size.
    v_block_size: The vocabulary block size.
    reduction: The reduction method ("sum", "mean" or "none") for the loss
      accumulation.

  Returns:
    The loss in scalar and the log-sum-exp of the used for backward pass.

  Raises:
    ValueError: If the invalid configuration is provided.
  """
  validate_inputs(
      x,
      labels,
      w,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )

  if x.dtype == jnp.float16:
    x = x.astype(preferred_element_type)
  if w.dtype == jnp.float16:
    w = w.astype(preferred_element_type)

  h_dim = x.shape[-1]
  v_dim = w.shape[1]

  # Constrain the memory spaces for x and w to prevent OOB accesses that occur
  # when the memory spaces is placed in VMEM.
  x = pltpu.with_memory_space_constraint(x, memory_space=pltpu.HBM)
  w = pltpu.with_memory_space_constraint(w, memory_space=pltpu.HBM)

  # Forward
  loss, lse = linear_softmax_cross_entropy_loss_forward_pallas_kernel(
      x,
      labels,
      w,
      reduction=reduction,
      h_dim=h_dim,
      v_dim=v_dim,
      preferred_element_type=preferred_element_type,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )
  return loss, lse


def linear_softmax_cross_entropy_loss_backward_pallas_kernel(
    dout,
    x,
    labels,
    w,
    lse,
    *,
    preferred_element_type: jnp.dtype,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
) -> tuple[Real[Array, "B H"], Real[Array, "H V"]]:
  """Pallas kernel for the backward pass of Linear Softmax Cross-Entropy Loss.

  Args:
    dout: Gradient of the loss (b_dim,).
    x: Input activations `x` (b_dim, h_dim).
    labels: One-hot encoded labels (b_dim, v_dim).
    w: LM Head projection weights `w` (h_dim, v_dim).
    lse: Log-sum-exp accumulator per batch item (b_dim,).
    preferred_element_type: Preferred element type for computation.
    b_block_size: Block size for batch dimension.
    h_block_size: Block size for hidden dimension.
    v_block_size: Block size for vocabulary dimension.

  Returns:
    A tuple of (x_grad, w_grad).
  """
  b_dim = x.shape[0]
  h_dim, v_dim = w.shape
  num_b_blocks = math.ceil(b_dim / b_block_size)
  num_h_blocks = math.ceil(h_dim / h_block_size)
  num_v_blocks = math.ceil(v_dim / v_block_size)
  num_stages = 2

  dout_spec = pl.BlockSpec(
      (b_block_size,), lambda i, j, s, k: (i,), memory_space=pltpu.VMEM
  )

  @pl.kernel(
      out_type=[
          jax.ShapeDtypeStruct(x.shape, dtype=jnp.float32),  # x_grad
          jax.ShapeDtypeStruct(w.shape, dtype=jnp.float32),  # w_grad
      ],
      mesh=pltpu.TensorCoreMesh(axis_name="core"),
      scratch_types=(
          pltpu.VMEM(
              (b_block_size, v_block_size), dtype=jnp.float32
          ),  # xw_scratch
          pltpu.VMEM(
              (b_block_size, h_block_size), dtype=jnp.float32
          ),  # x_grad_tile
          pltpu.VMEM(
              (h_block_size, v_block_size), dtype=jnp.float32
          ),  # w_grad_tile
          pltpu.SemaphoreType.DMA,  # x_read_sem
          pltpu.SemaphoreType.DMA,  # w_read_sem
          pltpu.SemaphoreType.DMA,  # x_write_sem
          pltpu.SemaphoreType.DMA,  # w_write_sem
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_get_vmem_limit_bytes(),
          disable_bounds_checks=True,
      ),
      name=(
          f"lce_bwd_bt_{b_block_size}_ht_{h_block_size}_vt_{v_block_size}"
      ),
  )
  def bwd_kernel(
      dout_hbm_ref,
      x_hbm_ref,
      labels_hbm_ref,
      w_hbm_ref,
      lse_hbm_ref,
      x_grad_hbm_ref,
      w_grad_hbm_ref,
      xw_scratch_ref,
      x_grad_tile_ref,
      w_grad_tile_ref,
      x_read_sem,
      w_read_sem,
      x_write_sem,
      w_write_sem,
  ):
    def bwd_pipeline(
        dout_ref,
        x_ref,
        labels_ref,
        w_ref,
        lse_ref,
        x_grad_hbm_ref,
        w_grad_hbm_ref,
        xw_scratch_ref,
        x_grad_tile_ref,
        w_grad_tile_ref,
        x_read_sem,
        w_read_sem,
        x_write_sem,
        w_write_sem,
    ):
      b_index, v_index, stage_index, h_index = (
          pl.program_id(i) for i in range(4)
      )

      # Calculate and accumulate xw_scratch_ref += x_ref @ w_ref as first stage
      @pl.when(stage_index == 0)
      def calculate_xw():
        calculate_xw_tiled(
            x_ref,
            w_ref,
            xw_scratch_ref,
            h_index=h_index,
            v_index=v_index,
            num_h_blocks=num_h_blocks,
            num_v_blocks=num_v_blocks,
            h_dim=h_dim,
            v_dim=v_dim,
            preferred_element_type=preferred_element_type,
        )

      # When xw_scratch_ref is fully accumulated, use it to calculate gradients
      # as second stage
      @pl.when(stage_index == 1)
      def calculate_grads():
        # Calculate actual block size if v_dim not a multiple of v_block_size
        cur_v_block_size = jnp.minimum(
            v_dim - v_block_size * v_index, v_block_size
        )

        # V Block size must be multiple of 128 to perform DMA (copy).
        # Aligning the V block size to 128
        cur_v_block_size = pl.multiple_of(
            (pl.cdiv(cur_v_block_size, 128) * 128).astype(jnp.int32), 128
        )

        # Calculate actual block size if h_dim not a multiple of h_block_size
        cur_h_block_size = jnp.minimum(
            h_dim - h_block_size * h_index, h_block_size
        )

        # H Block size must be multiple of 128 of major dimension, and 8 of
        # minor dimension to perform DMA.
        cur_h_block_128_aligned_size = pl.multiple_of(
            (pl.cdiv(cur_h_block_size, 128) * 128).astype(jnp.int32), 128
        )
        cur_h_block_8_aligned_size = pl.multiple_of(
            (pl.cdiv(cur_h_block_size, 8) * 8).astype(jnp.int32), 8
        )

        # Slicing x_grad and x_grad HBM ref to prepare for tiled read / write
        x_grad_slice = x_grad_hbm_ref.at[
            pl.ds(b_index * b_block_size, b_block_size),
            pl.ds(h_index * h_block_size, cur_h_block_128_aligned_size),
        ]
        w_grad_slice = w_grad_hbm_ref.at[
            pl.ds(h_index * h_block_size, cur_h_block_8_aligned_size),
            pl.ds(v_index * v_block_size, cur_v_block_size),
        ]

        x_grad_tile_slice = x_grad_tile_ref.at[
            pl.ds(0, b_block_size), pl.ds(0, cur_h_block_128_aligned_size)
        ]
        w_grad_tile_slice = w_grad_tile_ref.at[
            pl.ds(0, cur_h_block_8_aligned_size), pl.ds(0, cur_v_block_size)
        ]

        # Async copy ops defined here. Only starts after calling .start().
        x_write_future = pltpu.make_async_copy(
            x_grad_tile_slice, x_grad_slice, sem=x_write_sem
        )
        w_write_future = pltpu.make_async_copy(
            w_grad_tile_slice, w_grad_slice, sem=w_write_sem
        )
        x_read_future = pltpu.make_async_copy(
            x_grad_slice, x_grad_tile_slice, sem=x_read_sem
        )
        w_read_future = pltpu.make_async_copy(
            w_grad_slice, w_grad_tile_slice, sem=w_read_sem
        )

        # Preload x_grad and w_grad async before computing softmax to
        # overlap computation

        # Preload w_grad
        @pl.when(b_index != 0)
        def w_read():
          w_read_future.start()

        @pl.when(v_index != 0)
        def x_read():
          x_read_future.start()

        # Compute Softmax and store s = -labels + softmax(x@w) to xw_scratch_ref
        @pl.when(h_index == 0)
        def compute_s():
          labels_adjusted = labels_ref[...] - v_index * v_block_size
          labels_one_hot = jax.nn.one_hot(
              labels_adjusted, num_classes=v_block_size, dtype=x_ref.dtype
          )
          xw_scratch_ref[...] = -labels_one_hot + jnp.exp(
              xw_scratch_ref[...] - lse_ref[...][:, None]
          )
          xw_scratch_ref[...] *= dout_ref[...][:, None]

        # Init W gradient
        @pl.when(b_index == 0)
        def init_w_grad():
          w_grad_tile_ref[...] = jax.lax.dot_general(
              x_ref[...], xw_scratch_ref[...], (((0,), (0,)), ((), ()))
          )
          w_write_future.start()

        # Init X gradient
        @pl.when(v_index == 0)
        def init_x_grad():
          x_grad_tile_ref[...] = jax.lax.dot_general(
              xw_scratch_ref[...], w_ref[...], (((1,), (1,)), ((), ()))
          )
          x_write_future.start()

        # Accumulate W grad on B dimension
        @pl.when(b_index != 0)
        def accumulate_w_grad():
          res = jax.lax.dot_general(
              x_ref[...], xw_scratch_ref[...], (((0,), (0,)), ((), ()))
          )
          w_read_future.wait()
          w_grad_tile_ref[...] += res
          w_write_future.start()

        # Accumulate X grad on V dimension
        @pl.when(v_index != 0)
        def accumulate_x_grad():
          res = jax.lax.dot_general(
              xw_scratch_ref[...], w_ref[...], (((1,), (1,)), ((), ()))
          )
          x_read_future.wait()
          x_grad_tile_ref[...] += res
          x_write_future.start()

        # Lastly make sure to wait x_grad, w_grad write before next iteration
        w_write_future.wait()
        x_write_future.wait()

    pltpu.emit_pipeline(
        bwd_pipeline,
        grid=(num_b_blocks, num_v_blocks, num_stages, num_h_blocks),
        in_specs=[
            dout_spec,
            pl.BlockSpec(  # x
                (b_block_size, h_block_size),
                lambda i, j, s, k: (i, k),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # labels
                (b_block_size,),
                lambda i, j, s, k: (i,),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # w
                (h_block_size, v_block_size),
                lambda i, j, s, k: (k, j),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # lse
                (b_block_size,),
                lambda i, j, s, k: (i,),
                memory_space=pltpu.VMEM,
            ),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),  # x_grad
            pl.BlockSpec(memory_space=pltpu.HBM),  # w_grad
        ],
        # TODO: enable parallel core_axis_name for the kernel.
        # core_axis_name="core",
        dimension_semantics=(
            pltpu.ARBITRARY,
            pltpu.ARBITRARY,
            pltpu.ARBITRARY,
            pltpu.ARBITRARY,
        ),
    )(
        dout_hbm_ref,
        x_hbm_ref,
        labels_hbm_ref,
        w_hbm_ref,
        lse_hbm_ref,
        x_grad_hbm_ref,
        w_grad_hbm_ref,
        scratches=(
            xw_scratch_ref,
            x_grad_tile_ref,
            w_grad_tile_ref,
            x_read_sem,
            w_read_sem,
            x_write_sem,
            w_write_sem,
        ),
    )

  return bwd_kernel(dout, x, labels, w, lse)


@partial(
    jax.jit,
    static_argnames=[
        "b_block_size",
        "h_block_size",
        "v_block_size",
        "reduction",
        "preferred_element_type",
    ],
)
def linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
    dout: Real[Array, ""] | Real[Array, "B"],
    lse: Real[Array, "B"],
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
    reduction: Literal["sum", "mean", "none"] = "sum",
    preferred_element_type: jnp.dtype = jnp.float32,
) -> tuple[Real[Array, "B H"], Real[Array, "H V"]]:
  """Pallas kernel implementation of Linear Softmax Cross-Entropy Loss backward.

  The backward pass is also chunking the x, labels and w in all B, H and V
  dimensions so it can fit in the TPU VMEM. To not materialize the logits, the
  backward pass will re-compute the logits blockwise and cache in VMEM for the
  gradient calculation. This leads to also almost 0 memory overhead in backward
  pass.

  Args:
    dout: The output's gradient of the Linear Cross-Entropy kernel. Since the
      output is loss, the gradient is usually 1.0 when reduction is "sum" or
      "mean", or shape (B,) when reduction is "none".
    lse: The log-sum-exp of the from the forward pass residuals.
    x: The last layer output in the dimension of (B, H) where B is the batch
      dimension , and H is the hidden dimension.
    labels: The ground truth labels index in the dimension of (B,).
    w: The linear projection weight matrix in the dimension of (H, V) where V is
      the dimension of the output logits aka vocabulary size.
    b_block_size: The block size for the batch dimension.
    h_block_size: The block size for the hidden dimension.
    v_block_size: The block size for the vocabulary dimension.
    reduction: The reduction method for the cross entropy loss. Can be set to
      "sum", "mean" or "none" explicitly.

  Returns:
    The tuple of gradient of the loss with respect to x and w.
  """
  validate_inputs(
      x,
      labels,
      w,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )

  if x.dtype == jnp.float16:
    x = x.astype(preferred_element_type)
  if w.dtype == jnp.float16:
    w = w.astype(preferred_element_type)

  # Prepare dout array of shape (B,) for all reduction modes.
  b_dim = x.shape[0]
  if reduction == "sum":
    dout_scalar = jnp.squeeze(jnp.asarray(dout, dtype=preferred_element_type))
    dout_array = jnp.broadcast_to(dout_scalar, (b_dim,))
  elif reduction == "mean":
    dout_scalar = jnp.squeeze(jnp.asarray(dout, dtype=preferred_element_type))
    dout_array = jnp.broadcast_to(dout_scalar / b_dim, (b_dim,))
  else:
    dout_array = jnp.asarray(dout, dtype=preferred_element_type)

  # Constrain the memory spaces for x and w to prevent OOB accesses that occur
  # when the memory spaces is placed in VMEM.
  x = pltpu.with_memory_space_constraint(x, memory_space=pltpu.HBM)
  w = pltpu.with_memory_space_constraint(w, memory_space=pltpu.HBM)

  # Backward
  # pylint: disable-next=unpacking-non-sequence
  x_grad, w_grad = linear_softmax_cross_entropy_loss_backward_pallas_kernel(
      dout_array,
      x,
      labels,
      w,
      lse,
      preferred_element_type=preferred_element_type,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )

  # There is no gradient for the labels
  return (x_grad, w_grad)
