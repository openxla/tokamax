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
from typing import Annotated, Literal, Sequence
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
  return (
      # x tile (B, H) in input dtype (double buffered for pallas_call input)
      2 * b_block_size * h_alloc * dtype_bytes
      # labels (8 bytes), lse (8 bytes), and loss (8 bytes) tiles (all double
      # buffered for pallas_call inputs/outputs, 24 bytes per batch elem total)
      + 2 * 3 * 4 * b_block_size
      # w tile (H, V) in input dtype (double buffered for pallas_call input)
      + 2 * h_alloc * v_block_size * dtype_bytes
      # logits/xw tile (B, V) in float32 accumulator:
      # We account for 4 simultaneous (B, V) float32 buffers
      # (4 * 4 = 16 bytes/elem):
      #   1) xw_tiled (explicit scratch buffer)
      #   2) labels_one_hot (HLO stack temporary in accumulate_loss)
      #   3) diff = xw_tiled - max (HLO stack temporary in logsumexp)
      #   4) exp(diff) (HLO stack temporary in logsumexp)
      + 4 * b_block_size * v_block_size * 4
  )


def _calculate_bwd_vmem_bytes(
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    dtype: jnp.dtype = jnp.float32,
) -> int:
  """Calculates VMEM memory usage in bytes for the backward kernel."""
  dtype_bytes = jnp.dtype(dtype).itemsize
  h_alloc = 1 << (h_block_size - 1).bit_length()
  return (
      # (B, H) shapes:
      # x tile (B, H) in input dtype (double buff for in_specs: 2 * dt)
      # + x_grad_tile accumulator (B, H) in float32 (single scratch buffer: 4B)
      # + dot_general intermediate result in accumulate_x_grad (float32: 4B)
      b_block_size * h_alloc * (2 * dtype_bytes + 8)
      # (B) shapes
      # labels (8 bytes), lse (8 bytes), and dout (8 bytes) tiles (all double
      # buffered for in_specs inputs, 24 bytes per batch elem total)
      + 2 * 3 * 4 * b_block_size
      # (H, V) shapes:
      # 2x w tiles (w_v & w_vm1 double buff for in_specs: 4 * dt)
      # + w_grad_tile accumulator (H, V) in float32 (single scratch buffer: 4B)
      # + dot_general intermediate result in accumulate_w_grad (float32: 4B)
      + h_alloc * v_block_size * (4 * dtype_bytes + 8)
      # (B, V) shapes:
      # logits/softmax tile (B, V) in float32 accumulator:
      # Ping-pong xw_scratch_ref: 2 x (B, V) explicit float32 buffers (8B/elem)
      # + 1 simultaneous float32 stack buffer during compute_s (4B/elem)
      # In-place target subtraction with broadcasted_iota/jnp.where uses zero
      # extra (B, V) buffers. Total 3 simultaneous (B, V) float32 buffers
      # (12B/elem)
      + 3 * b_block_size * v_block_size * 4
  )


def _get_vmem_limit_bytes() -> int:
  return int(0.90 * pltpu.get_tpu_info().vmem_capacity_bytes)


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
    if h_dim < h_dim_max:
      h_candidates_non_div = [
          h for h in range(128, h_dim_max + 1, 128) if h >= h_dim
      ]
      if h_candidates_non_div:
        h_block_size = min(
            h_candidates_non_div,
            key=lambda x: (
                0 if (x & (x - 1)) == 0 else 1,
                x,
            ),
        )
      else:
        h_block_size = h_dim_max
    else:
      h_block_size = h_dim_max

  # 3. Choose v_block_size: as large as possible to fit VMEM.
  # Must be >= 128, multiple of 128. Divisible by v_dim if possible.
  if is_bwd:
    # fixed_bytes accounts for VMEM costs that do not scale with V:
    #   - x tile (2 * dt) + x_grad_tile (4B) + dot_general res (4B) =
    #          b_block_size * h_block_size * (2 * dtype_bytes + 8)
    #   - labels (8 bytes) + lse (8 bytes) + dout (8 bytes) = 24 * b_block_size
    fixed_bytes = (
        b_block_size * h_block_size * (2 * dtype_bytes + 8) + 24 * b_block_size
    )
    # per_v_bytes accounts for all VMEM costs per column of V:
    #   - 2x w tiles (w_v & w_vm1 double-buffered in dtype: 4 * dt) +
    #       w_grad_tile (4B) + dot_general res (4B) =
    #       h_block_size * (4 * dtype_bytes + 8)
    #   - 3 simultaneous float32 (B, V) buffers on the VMEM stack:
    #       1-2) 2x xw_scratch_ref (explicit VMEM scratch ping-pong: 8B/elem)
    #       3) diff/prob temporary on the stack during compute_s (4B/elem)
    #       Total 3 * 4 = 12 bytes per element across b_block_size
    per_v_bytes = h_block_size * (4 * dtype_bytes + 8) + 12 * b_block_size
  else:
    # fixed_bytes accounts for VMEM costs that do not scale with V:
    #   - x tile = 2 * b_block_size * h_block_size * dtype_bytes
    #   - labels (8 bytes) + lse (8 bytes) + loss (8 bytes) = 24 * b_block_size
    fixed_bytes = (
        2 * b_block_size * h_block_size * dtype_bytes + 24 * b_block_size
    )
    # per_v_bytes accounts for all VMEM costs per column of V:
    #   - w tile (double-buff in dtype: 2 * dt) = 2 * h_block_size * dtype_bytes
    #   - 4 simultaneous float32 (B, V) buffers on the VMEM stack:
    #     (4 * 4 = 16 bytes per element across b_block_size):
    #       1) xw_tiled (explicit scratch buffer)
    #       2) labels_one_hot (HLO stack temporary in accumulate_loss)
    #       3) diff = xw_tiled - max (HLO stack temporary in logsumexp)
    #       4) exp(diff) (HLO stack temporary in logsumexp)
    per_v_bytes = 2 * h_block_size * dtype_bytes + 16 * b_block_size

  if vmem_limit_bytes > fixed_bytes:
    max_v_vmem = (vmem_limit_bytes - fixed_bytes) // per_v_bytes
  else:
    max_v_vmem = 128

  max_v = min(v_dim, max_v_vmem)
  max_v_aligned = max(128, (max_v // 128) * 128)

  try:
    max_cores = pltpu.get_tpu_info().num_cores
  except Exception:  # pylint: disable=broad-except
    max_cores = 1

  def is_mc_compatible(v: int) -> bool:
    if max_cores <= 1:
      return True
    n_v_b = math.ceil(v_dim / v)
    return n_v_b < max_cores or n_v_b % max_cores == 0

  # 1. First try to find a divisor of v_dim that fits in VMEM and is multi-core
  # compatible.
  v_block_size = None
  for v in range(max_v_aligned, 127, -128):
    if (
        v_dim % v == 0
        and is_mc_compatible(v)
        and calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
        <= vmem_limit_bytes
    ):
      v_block_size = v
      break

  # 2. If no divisor fits, pick a multiple of 128 that is multi-core compatible
  if v_block_size is None:
    for v in range(max_v_aligned, 127, -128):
      if (
          is_mc_compatible(v)
          and calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
          <= vmem_limit_bytes
      ):
        v_block_size = v
        break

  # 3. Fallback: try divisor of v_dim that fits in VMEM
  if v_block_size is None:
    for v in range(max_v_aligned, 127, -128):
      if (
          v_dim % v == 0
          and calc_vmem_fn(b_block_size, h_block_size, v, dtype=dtype)
          <= vmem_limit_bytes
      ):
        v_block_size = v
        break

  # 4. Fallback: pick the largest multiple of 128 that fits
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
  del b_block_size  # Currently unused.

  if labels.shape[0] != x.shape[0]:
    raise ValueError(
        f"Batch dimension mismatch: labels batch dimension ({labels.shape[0]})"
        f" != x batch dimension ({x.shape[0]})."
    )

  if x.shape[-1] != w.shape[0]:
    raise ValueError(
        f"Hidden dimension mismatch: x hidden dimension ({x.shape[-1]}) !="
        f" w hidden dimension ({w.shape[0]})."
    )

  if w.shape[0] % 8 != 0:
    raise ValueError("The hidden dimension of w must be a multiple of 8")


@jax.named_scope("calculate_xw_tiled")
def calculate_xw_tiled(
    x_ref,
    w_ref,
    xw_tiled,
    b_index,
    h_index,
    v_index,
    num_b_blocks,
    num_h_blocks,
    num_v_blocks,
    b_dim,
    h_dim,
    v_dim,
    preferred_element_type: jnp.dtype,
):
  """Calculates xw_tiled += x@w for forward/backward kernel common logic."""
  b_block_size, h_block_size, v_block_size = (
      x_ref.shape[0],
      x_ref.shape[1],
      w_ref.shape[1],
  )
  x_val = x_ref[...]
  if b_dim % b_block_size != 0:
    rem_b = b_dim % b_block_size
    row_idx = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(b_block_size, 1), dimension=0
    )
    x_val = jnp.where(
        (b_index == num_b_blocks - 1) & (row_idx >= rem_b), 0.0, x_val
    )
  if h_dim % h_block_size != 0:
    rem_h = h_dim % h_block_size
    col_idx = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(1, h_block_size), dimension=1
    )
    x_val = jnp.where(
        (h_index == num_h_blocks - 1) & (col_idx >= rem_h), 0.0, x_val
    )

  w_val = w_ref[...]
  if h_dim % h_block_size != 0:
    rem_h = h_dim % h_block_size
    row_idx = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(h_block_size, 1), dimension=0
    )
    w_val = jnp.where(
        (h_index == num_h_blocks - 1) & (row_idx >= rem_h), 0.0, w_val
    )
  if v_dim % v_block_size != 0 or num_v_blocks * v_block_size > v_dim:
    col_idx = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(1, v_block_size), dimension=1
    )
    w_val = jnp.where((v_index * v_block_size + col_idx) >= v_dim, 0.0, w_val)

  @pl.when(h_index == 0)
  def init_xw():
    xw_tiled[...] = jax.lax.dot_general(
        x_val,
        w_val,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=preferred_element_type,
    )

  @pl.when(h_index != 0)
  def accumulate_xw():
    xw_tiled[...] += jax.lax.dot_general(
        x_val,
        w_val,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=preferred_element_type,
    )


def _bytes(tensor: jt.Array | jax.ShapeDtypeStruct) -> int:
  """Computes total bytes of a tensor or ShapeDtypeStruct."""
  return math.prod(tensor.shape) * jnp.dtype(tensor.dtype).itemsize


def linear_softmax_cross_entropy_loss_fwd_cost_estimate(
    x: Real[Array, "B H"] | jax.ShapeDtypeStruct,
    labels: Integer[Array, "B"] | jax.ShapeDtypeStruct,
    w: Real[Array, "H V"] | jax.ShapeDtypeStruct,
    out_type: Sequence[jax.ShapeDtypeStruct] | None = None,
) -> pl.CostEstimate:
  """Calculates the theoretical hardware cost estimate for the forward pass kernel.

  This provides an accurate model of computational (FLOPs), transcendental,
  and memory bandwidth (bytes accessed) costs for profilers and runtime cost
  models
  (e.g., XProf / HLO CostAnalysis).

  FLOPs Breakdown:
    1. Logit Projection Matmul (X @ W):
       - Input X: (B, H), Weight W: (H, V)
       - Standard matmul FLOPs: 2 * B * H * V (1 multiply + 1 accumulate per
       element).
    2. Online LogSumExp Accumulation (jax.nn.logsumexp across V blocks):
       - Maximum subtraction: B * V ops
       - Exponentiation: B * V ops (tracked in transcendentals)
       - Reduction sum: B * V ops
       - Logarithm of sum: B ops (tracked in transcendentals)
       - Running logaddexp combining blocks: 2 * B * V ops
       - Total elementwise reduction FLOPs: 2 * B * V.
    3. Sparse Target Loss Reduction:
       - Target extraction / one-hot subtraction: B ops
       - Total loss FLOPs: B.

    Total FLOPs = 2 * B * H * V + 2 * B * V + B.

  Transcendental Operations Breakdown:
    - Exponentiation (exp) in logsumexp across vocabulary V: B * V ops.
    - Logarithm (log) in logsumexp per batch element: B ops.
    Total Transcendentals = B * V + B.

  HBM Memory Traffic Breakdown:
    - Inputs read from HBM:
      - Activations x: B * H * sizeof(dtype_x)
      - Labels: B * sizeof(dtype_labels)
      - Weights w: H * V * sizeof(dtype_w)
    - Outputs written to HBM:
      - Loss: B * sizeof(float32) (4 bytes)
      - LSE: B * sizeof(float32) (4 bytes)
    Total Bytes Accessed = sum(input_bytes) + sum(output_bytes).

  Args:
    x: Input activation tensor or ShapeDtypeStruct with shape (B, H).
    labels: Target label tensor or ShapeDtypeStruct with shape (B,).
    w: Weight projection tensor or ShapeDtypeStruct with shape (H, V).
    out_type: Optional sequence of output ShapeDtypeStructs for loss and LSE.

  Returns:
    A `pl.CostEstimate` object containing flops, transcendentals, and
    bytes_accessed.
  """
  b_dim = x.shape[0]
  h_dim = x.shape[1]
  v_dim = w.shape[1]

  matmul_flops = 2 * b_dim * h_dim * v_dim
  reduction_flops = 2 * b_dim * v_dim + b_dim
  total_flops = matmul_flops + reduction_flops

  transcendentals = b_dim * v_dim + b_dim

  inputs = [x, labels, w]
  input_bytes = sum(_bytes(t) for t in inputs)
  if out_type is not None:
    output_bytes = sum(_bytes(t) for t in out_type)
  else:
    output_bytes = 2 * b_dim * 4

  return pl.CostEstimate(
      flops=int(total_flops),
      transcendentals=int(transcendentals),
      bytes_accessed=int(input_bytes + output_bytes),
  )


def linear_softmax_cross_entropy_loss_bwd_cost_estimate(
    dout: Real[Array, ""] | Real[Array, "B"] | jax.ShapeDtypeStruct,
    x: Real[Array, "B H"] | jax.ShapeDtypeStruct,
    labels: Integer[Array, "B"] | jax.ShapeDtypeStruct,
    w: Real[Array, "H V"] | jax.ShapeDtypeStruct,
    lse: Real[Array, "B"] | jax.ShapeDtypeStruct,
    out_type: Sequence[jax.ShapeDtypeStruct] | None = None,
) -> pl.CostEstimate:
  """Calculates the theoretical hardware cost estimate for the backward pass kernel.

  This accounts for forward logit recomputation, dense softmax derivative
  computation
  with in-kernel target subtraction, and backward matrix multiplications for
  activation
  and weight gradients.

  FLOPs Breakdown:
    1. Forward Logit Recomputation (X @ W):
       - Input X: (B, H), Weight W: (H, V)
       - Matmul FLOPs: 2 * B * H * V.
    2. Dense Softmax Probability & Target Subtraction (compute_s):
       - Subtract LSE from logits: (XW - LSE): B * V ops
       - Exponentiation: exp(XW - LSE): B * V ops (tracked in transcendentals)
       - In-kernel target subtraction and gradient scaling:
         S = (prob - 1.0) * dout for target, prob * dout for non-targets: 2 * B
         * V ops.
       - Total elementwise FLOPs: 3 * B * V.
    3. Weight Gradient Matrix Multiplication (W_grad = X^T @ S):
       - Transposed activations X^T: (H, B), Softmax derivatives S: (B, V)
       - Output W_grad: (H, V)
       - Matmul FLOPs: 2 * B * H * V.
    4. Activation Gradient Matrix Multiplication (X_grad = S @ W^T):
       - Softmax derivatives S: (B, V), Transposed weights W^T: (V, H)
       - Output X_grad: (B, H)
       - Matmul FLOPs: 2 * B * H * V.

    Total FLOPs = 6 * B * H * V + 3 * B * V.

  Transcendental Operations Breakdown:
    - Exponentiation (exp) in softmax probabilities: B * V ops.
    Total Transcendentals = B * V.

  HBM Memory Traffic Breakdown:
    - Inputs read from HBM:
      - Upstream gradient dout: dout.size * sizeof(dtype_dout)
      - Activations x: B * H * sizeof(dtype_x)
      - Labels: B * sizeof(dtype_labels)
      - Weights w: H * V * sizeof(dtype_w)
      - Forward LSE: B * sizeof(dtype_lse)
    - Outputs written to HBM:
      - Activation gradients x_grad: (num_cores * B_aligned * H_aligned) or (B *
      H) * 4 bytes
      - Weight gradients w_grad: (H_aligned * V_aligned) * 4 bytes
    Total Bytes Accessed = sum(input_bytes) + sum(output_bytes).

  Args:
    dout: Upstream gradient tensor or ShapeDtypeStruct.
    x: Input activation tensor or ShapeDtypeStruct with shape (B, H).
    labels: Target label tensor or ShapeDtypeStruct with shape (B,).
    w: Weight projection tensor or ShapeDtypeStruct with shape (H, V).
    lse: Forward log-sum-exp tensor or ShapeDtypeStruct with shape (B,).
    out_type: Optional sequence of output ShapeDtypeStructs for x_grad and
      w_grad.

  Returns:
    A `pl.CostEstimate` object containing flops, transcendentals, and
    bytes_accessed.
  """
  b_dim = x.shape[0]
  h_dim = x.shape[1]
  v_dim = w.shape[1]

  matmul_flops = 3 * (2 * b_dim * h_dim * v_dim)
  softmax_flops = 3 * b_dim * v_dim
  total_flops = matmul_flops + softmax_flops

  transcendentals = b_dim * v_dim

  inputs = [dout, x, labels, w, lse]
  input_bytes = sum(_bytes(t) for t in inputs)
  if out_type is not None:
    output_bytes = sum(_bytes(t) for t in out_type)
  else:
    output_bytes = (b_dim * h_dim + h_dim * v_dim) * 4

  return pl.CostEstimate(
      flops=int(total_flops),
      transcendentals=int(transcendentals),
      bytes_accessed=int(input_bytes + output_bytes),
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
  get_b_ds = lambda i: pl.ds(
      i * b_block_size, jnp.minimum(b_block_size, b_dim - i * b_block_size)
  )
  loss_out_spec = pl.BlockSpec(
      (pl.BoundedSlice(b_block_size),),
      lambda i, j, k: (get_b_ds(i),),
      memory_space=pltpu.VMEM,
  )
  lse_out_spec = pl.BlockSpec(
      (pl.BoundedSlice(b_block_size),),
      lambda i, j, k: (get_b_ds(i),),
      memory_space=pltpu.VMEM,
  )

  cost_estimate = linear_softmax_cross_entropy_loss_fwd_cost_estimate(
      x=x,
      labels=labels,
      w=w,
      out_type=out_type,
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
      cost_estimate=cost_estimate,
      name=f"lce_fwd_bt_{b_block_size}_ht_{h_block_size}_vt_{v_block_size}",
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
      num_b_blocks, num_v_blocks, num_h_blocks = (
          pl.num_programs(i) for i in range(3)
      )

      # xw_tiled += x_ref @ w_ref
      calculate_xw_tiled(
          x_ref,
          w_ref,
          xw_tiled,
          b_index=b_index,
          h_index=h_index,
          v_index=v_index,
          num_b_blocks=num_b_blocks,
          num_h_blocks=num_h_blocks,
          num_v_blocks=num_v_blocks,
          b_dim=b_dim,
          h_dim=h_dim,
          v_dim=v_dim,
          preferred_element_type=preferred_element_type,
      )

      @pl.when(jnp.logical_and(v_index == 0, h_index == 0))
      @jax.named_scope("init_lse")
      def init_lse():
        lse_ref[...] = jnp.full_like(lse_ref, -jnp.inf)
        loss_ref[...] = jnp.zeros_like(loss_ref)

      @pl.when(h_index == num_h_blocks - 1)
      @jax.named_scope("accumulate_loss")
      def accumulate_loss():
        # Convert labels to one-hot, due to chunking on v dimension, the indices
        # needs to be shifted down by the v starting index. Negative or
        # out-of-bound indices are OK since jax.nn.one_hot will set them to 0.
        labels_adjusted = labels_ref[...] - v_index * v_block_size
        labels_one_hot = jax.nn.one_hot(
            labels_adjusted, num_classes=v_block_size, dtype=x_ref.dtype
        )
        if b_dim % b_block_size != 0:
          rem_b = b_dim % b_block_size
          row_idx = jax.lax.broadcasted_iota(
              jnp.int32, labels_one_hot.shape, dimension=0
          )
          labels_one_hot = jnp.where(
              (b_index == num_b_blocks - 1) & (row_idx >= rem_b),
              0.0,
              labels_one_hot,
          )
        loss_ref[...] -= jnp.sum(labels_one_hot * xw_tiled[...], axis=-1)
        if v_dim % v_block_size != 0:
          rem_v = v_dim % v_block_size
          col_idx = jax.lax.broadcasted_iota(
              jnp.int32, xw_tiled.shape, dimension=1
          )
          xw_tiled[...] = jnp.where(
              (v_index == num_v_blocks - 1) & (col_idx >= rem_v),
              -jnp.inf,  # forcing exp(-inf)=0, instead of 1
              xw_tiled[...],
          )
        lse_block = jax.nn.logsumexp(xw_tiled[...], axis=-1)
        if b_dim % b_block_size != 0:
          rem_b = b_dim % b_block_size
          row_idx = jax.lax.broadcasted_iota(
              jnp.int32, lse_block.shape, dimension=0
          )
          lse_block = jnp.where(
              (b_index == num_b_blocks - 1) & (row_idx >= rem_b),
              -jnp.inf,
              lse_block,
          )
        lse_ref[...] = jnp.logaddexp(lse_ref[...], lse_block)

      @pl.when(
          jnp.logical_and(
              v_index == num_v_blocks - 1, h_index == num_h_blocks - 1
          )
      )
      def perform_loss_reduction():
        loss_ref[...] += lse_ref[...]

    get_h_ds = lambda k: pl.ds(
        k * h_block_size, jnp.minimum(h_block_size, h_dim - k * h_block_size)
    )
    get_v_ds = lambda j: pl.ds(
        j * v_block_size, jnp.minimum(v_block_size, v_dim - j * v_block_size)
    )

    pltpu.emit_pipeline(
        fwd_pipeline,
        grid=(num_b_blocks, num_v_blocks, num_h_blocks),
        in_specs=[
            pl.BlockSpec(
                (pl.BoundedSlice(b_block_size), pl.BoundedSlice(h_block_size)),
                lambda i, j, k: (get_b_ds(i), get_h_ds(k)),
                memory_space=pltpu.VMEM,
            ),  # x
            pl.BlockSpec(
                (pl.BoundedSlice(b_block_size),),
                lambda i, j, k: (get_b_ds(i),),
                memory_space=pltpu.VMEM,
            ),  # labels
            pl.BlockSpec(
                (pl.BoundedSlice(h_block_size), pl.BoundedSlice(v_block_size)),
                lambda i, j, k: (get_h_ds(k), get_v_ds(j)),
                memory_space=pltpu.VMEM,
            ),  # w
        ],
        out_specs=[
            loss_out_spec,  # loss
            lse_out_spec,  # lse
        ],
        core_axis_name="core",
        dimension_semantics=(
            pltpu.PARALLEL,
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

  # Constrain the memory spaces for x, labels, and w to prevent OOB accesses
  # that occur when the memory spaces is placed in VMEM.
  x = pltpu.with_memory_space_constraint(x, memory_space=pltpu.HBM)
  labels = pltpu.with_memory_space_constraint(labels, memory_space=pltpu.HBM)
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
    b_dim: int,
    h_dim: int,
    v_dim: int,
    preferred_element_type: jnp.dtype,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
) -> tuple[Real[Array, "B H"], Real[Array, "H V"]]:
  """Pallas kernel for the backward pass of Linear Softmax Cross-Entropy Loss.

  Args:
    dout: Gradient of the loss (b_dim,).
    x: Input activations `x` (b_dim, h_dim).
    labels: Ground truth labels (b_dim,).
    w: LM Head projection weights `w` (h_dim, v_dim).
    lse: Log-sum-exp accumulator per batch item (b_dim,).
    b_dim: Batch dimension size.
    h_dim: Hidden dimension size.
    v_dim: Vocabulary dimension size.
    preferred_element_type: Preferred element type for computation.
    b_block_size: Block size for batch dimension.
    h_block_size: Block size for hidden dimension.
    v_block_size: Block size for vocabulary dimension.

  Returns:
    A tuple of (x_grad, w_grad).
  """
  num_b_blocks = math.ceil(b_dim / b_block_size)
  num_h_blocks = math.ceil(h_dim / h_block_size)
  num_v_blocks = math.ceil(v_dim / v_block_size)
  max_cores = pltpu.get_tpu_info().num_cores
  num_cores = math.gcd(num_v_blocks, max_cores)
  if num_cores == 0:
    num_cores = 1
  num_v_blocks_per_core = num_v_blocks // num_cores
  num_v_steps = num_v_blocks_per_core + 1
  major_align = 32 // x.dtype.itemsize
  b_dim_aligned = int(math.ceil(b_dim / major_align) * major_align)
  h_dim_128_aligned = int(math.ceil(h_dim / 128) * 128)
  h_dim_8_aligned = int(math.ceil(h_dim / 8) * 8)
  v_dim_aligned = int(math.ceil(v_dim / 128) * 128)

  out_type = [
      jax.ShapeDtypeStruct(
          (num_cores, b_dim_aligned, h_dim_128_aligned), dtype=jnp.float32
      ),  # x_grad
      jax.ShapeDtypeStruct(
          (h_dim_8_aligned, v_dim_aligned), dtype=jnp.float32
      ),  # w_grad
  ]
  cost_estimate = linear_softmax_cross_entropy_loss_bwd_cost_estimate(
      dout=dout,
      x=x,
      labels=labels,
      w=w,
      lse=lse,
      out_type=out_type,
  )

  @pl.kernel(
      out_type=out_type,
      mesh=pltpu.TensorCoreMesh(axis_name="core"),
      scratch_types=(
          pltpu.VMEM(
              (2, b_block_size, v_block_size), dtype=jnp.float32
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
      cost_estimate=cost_estimate,
      name=(
          f"lce_bwd_bt_{b_block_size}_ht_{h_block_size}_vt_{v_block_size}"
          f"_cores_{num_cores}"
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
    c_index = jax.lax.axis_index("core")

    def bwd_pipeline(
        dout_ref,
        x_ref,
        labels_ref,
        w_v_ref,
        w_vm1_ref,
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
      b_index, v_step, h_index = (pl.program_id(i) for i in range(3))
      global_v = c_index * num_v_blocks_per_core + v_step
      v_vm1 = jnp.maximum(0, v_step - 1)
      global_v_vm1 = c_index * num_v_blocks_per_core + v_vm1
      v_buf_idx = v_step % 2
      vm1_buf_idx = (v_step - 1) % 2

      # 1. Forward logit accumulation for next tile (v_step)
      # Was: @pl.when(v_step < num_v_blocks_per_core)
      # Note: we let this run on the last V block even though it'll be garbage.
      with jax.named_scope("calculate_xw"):
        calculate_xw_tiled(
            x_ref,
            w_v_ref,
            xw_scratch_ref.at[v_buf_idx],
            b_index=b_index,
            h_index=h_index,
            v_index=global_v,
            num_b_blocks=num_b_blocks,
            num_h_blocks=num_h_blocks,
            num_v_blocks=num_v_blocks,
            b_dim=b_dim,
            h_dim=h_dim,
            v_dim=v_dim,
            preferred_element_type=preferred_element_type,
        )

      # 2. Compute Softmax and store s = -labels*dout + softmax(x@w)*dout to
      # xw_scratch_ref[v_buf_idx] when forward accumulation finishes on H
      @jax.named_scope("compute_s")
      def compute_s():
        lse_val = lse_ref[...]
        if b_dim % b_block_size != 0:
          rem_b = b_dim % b_block_size
          row_idx = jax.lax.broadcasted_iota(
              jnp.int32, lse_val.shape, dimension=0
          )
          lse_val = jnp.where(
              (b_index == num_b_blocks - 1) & (row_idx >= rem_b),
              jnp.inf,
              lse_val,
          )

        # 1. Broadcast dout once to avoid redundant vector staging registers
        dout_val = dout_ref[...][:, None]

        # 2. Compute probabilities: prob = exp(xw - lse)
        prob = jnp.exp(xw_scratch_ref[v_buf_idx] - lse_val[:, None])

        # 3. In-kernel target gradient subtraction: (prob - 1.0) * dout for
        # targets
        labels_adjusted = labels_ref[...] - global_v * v_block_size
        col_idx = jax.lax.broadcasted_iota(
            jnp.int32, (1, v_block_size), dimension=1
        )
        is_target = col_idx == labels_adjusted[:, None]
        prob_sub = jnp.where(is_target, prob - 1.0, prob)
        s_val = prob_sub * dout_val

        # 4. Zero out padding elements only when dimensions are not aligned
        if b_dim % b_block_size != 0:
          rem_b = b_dim % b_block_size
          row_idx = jax.lax.broadcasted_iota(
              jnp.int32, (b_block_size, 1), dimension=0
          )
          s_val = jnp.where(
              (b_index == num_b_blocks - 1) & (row_idx >= rem_b),
              0.0,
              s_val,
          )
        if v_dim % v_block_size != 0 or num_v_blocks * v_block_size > v_dim:
          s_val = jnp.where(
              (global_v * v_block_size + col_idx) >= v_dim,
              0.0,
              s_val,
          )

        xw_scratch_ref[v_buf_idx] = s_val

      # was: @pl.when((h_index == num_h_blocks - 1) &
      #               (v_step < num_v_blocks_per_core))
      # Note: we let this run on the last V block even though it'll be garbage.
      if num_h_blocks == 1:
        compute_s()
      else:
        jax.lax.cond(
            (h_index == num_h_blocks - 1) & (v_step < num_v_blocks_per_core),
            compute_s,
            lambda: None,
        )

      # 3. Gradient calculation for current tile (v_step - 1)
      # Calculate actual block size if v_dim not a multiple of v_block_size
      cur_v_block_size = jnp.maximum(
          0, jnp.minimum(v_dim - v_block_size * global_v_vm1, v_block_size)
      )

      # V Block size must be multiple of 128 to perform DMA (copy).
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

      # Major dimension DMA requires 32-byte alignment (8 for fp32, 16 for
      # bf16/fp16, 32 for int8).
      major_align = 32 // x_ref.dtype.itemsize

      # Calculate actual block size if b_dim not a multiple of b_block_size
      cur_b_block_size = jnp.minimum(
          b_dim - b_block_size * b_index, b_block_size
      )
      cur_b_block_aligned_size = pl.multiple_of(
          (pl.cdiv(cur_b_block_size, major_align) * major_align).astype(
              jnp.int32
          ),
          major_align,
      )

      # Slicing x_grad and w_grad HBM ref to prepare for tiled read / write
      x_grad_slice = x_grad_hbm_ref.at[
          c_index,
          pl.ds(b_index * b_block_size, cur_b_block_aligned_size),
          pl.ds(h_index * h_block_size, cur_h_block_128_aligned_size),
      ]
      w_grad_slice = w_grad_hbm_ref.at[
          pl.ds(h_index * h_block_size, cur_h_block_8_aligned_size),
          pl.ds(global_v_vm1 * v_block_size, cur_v_block_size),
      ]

      x_grad_tile_slice = x_grad_tile_ref.at[
          pl.ds(0, cur_b_block_aligned_size),
          pl.ds(0, cur_h_block_128_aligned_size),
      ]
      w_grad_tile_slice = w_grad_tile_ref.at[
          pl.ds(0, cur_h_block_8_aligned_size), pl.ds(0, cur_v_block_size)
      ]

      # Async copy ops defined here. Only starts after calling .start().
      x_grad_write_future = pltpu.make_async_copy(
          x_grad_tile_slice, x_grad_slice, sem=x_write_sem
      )
      w_grad_write_future = pltpu.make_async_copy(
          w_grad_tile_slice, w_grad_slice, sem=w_write_sem
      )
      x_grad_read_future = pltpu.make_async_copy(
          x_grad_slice, x_grad_tile_slice, sem=x_read_sem
      )
      w_grad_read_future = pltpu.make_async_copy(
          w_grad_slice, w_grad_tile_slice, sem=w_read_sem
      )

      # Preload w_grad async before computing gradients.
      # There's no accumulation on first batch block so we skip the read.
      @pl.when((b_index != 0) & (v_step > 0))
      @jax.named_scope("w_grad_read_start")
      def w_read():
        w_grad_read_future.start()

      @jax.named_scope("get_clean_x")
      def get_clean_x():
        """Zeros out out-of-bounds padding elements in x_ref in VMEM."""
        x_val = x_ref[...]
        if b_dim % b_block_size != 0:
          rem_b = b_dim % b_block_size
          row_idx = jax.lax.broadcasted_iota(
              dtype=jnp.int32, shape=(b_block_size, 1), dimension=0
          )
          x_val = jnp.where(
              (b_index == num_b_blocks - 1) & (row_idx >= rem_b),
              0.0,
              x_val,
          )
        if h_dim % h_block_size != 0:
          rem_h = h_dim % h_block_size
          col_idx = jax.lax.broadcasted_iota(
              dtype=jnp.int32, shape=(1, h_block_size), dimension=1
          )
          x_val = jnp.where(
              (h_index == num_h_blocks - 1) & (col_idx >= rem_h),
              0.0,
              x_val,
          )
        return x_val

      @jax.named_scope("get_clean_w_vm1")
      def get_clean_w_vm1():
        """Zeros out out-of-bounds padding elements in w_vm1_ref in VMEM."""
        w_val = w_vm1_ref[...]
        if h_dim % h_block_size != 0:
          rem_h = h_dim % h_block_size
          row_idx = jax.lax.broadcasted_iota(
              dtype=jnp.int32, shape=(h_block_size, 1), dimension=0
          )
          w_val = jnp.where(
              (h_index == num_h_blocks - 1) & (row_idx >= rem_h),
              0.0,
              w_val,
          )
        if v_dim % v_block_size != 0 or num_v_blocks * v_block_size > v_dim:
          col_idx = jax.lax.broadcasted_iota(
              dtype=jnp.int32, shape=(1, v_block_size), dimension=1
          )
          w_val = jnp.where(
              (global_v_vm1 * v_block_size + col_idx) >= v_dim,
              0.0,
              w_val,
          )
        return w_val

      # Preload x_grad async when accumulating across V (v_step > 1).
      @pl.when(v_step > 1)
      @jax.named_scope("x_grad_read_start")
      def x_read():
        x_grad_read_future.start()

      # Init W gradient
      @pl.when((v_step > 0) & (b_index == 0))
      @jax.named_scope("init_w_grad")
      def init_w_grad():
        w_grad_tile_ref[...] = jax.lax.dot_general(
            get_clean_x(),
            xw_scratch_ref.at[vm1_buf_idx][...],
            (((0,), (0,)), ((), ())),
        )
        w_grad_write_future.start()

      # Init X gradient
      @pl.when(v_step == 1)
      @jax.named_scope("init_x_grad")
      def init_x_grad():
        x_grad_tile_ref[...] = jax.lax.dot_general(
            xw_scratch_ref.at[vm1_buf_idx][...],
            get_clean_w_vm1(),
            (((1,), (1,)), ((), ())),
        )
        x_grad_write_future.start()

      # Accumulate W grad on B dimension
      @pl.when((v_step > 0) & (b_index != 0))
      @jax.named_scope("accumulate_w_grad")
      def accumulate_w_grad():
        res = jax.lax.dot_general(
            get_clean_x(),
            xw_scratch_ref.at[vm1_buf_idx][...],
            (((0,), (0,)), ((), ())),
        )
        w_grad_read_future.wait()
        w_grad_tile_ref[...] += res
        w_grad_write_future.start()

      # Accumulate X grad on V dimension
      @pl.when(v_step > 1)
      @jax.named_scope("accumulate_x_grad")
      def accumulate_x_grad():
        res = jax.lax.dot_general(
            xw_scratch_ref.at[vm1_buf_idx][...],
            get_clean_w_vm1(),
            (((1,), (1,)), ((), ())),
        )
        x_grad_read_future.wait()
        x_grad_tile_ref[...] += res
        x_grad_write_future.start()

      # Lastly make sure to wait x_grad, w_grad write before next iteration
      @pl.when(v_step > 0)
      @jax.named_scope("wait_grads")
      def wait_grads():
        w_grad_write_future.wait()
        x_grad_write_future.wait()

    pltpu.emit_pipeline(
        bwd_pipeline,
        grid=(
            num_b_blocks,
            num_v_steps,
            num_h_blocks,
        ),
        in_specs=[
            pl.BlockSpec(  # dout
                (b_block_size,),
                lambda i, j, k: (i,),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # x
                (b_block_size, h_block_size),
                lambda i, j, k: (i, k),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # labels
                (b_block_size,),
                lambda i, j, k: (i,),
                memory_space=pltpu.VMEM,
            ),
            # Notes for w_v and w_vm1:
            # - V is split across cores
            # - We have one extra V block for additional pipelining in
            #   the v dimension.
            pl.BlockSpec(  # w_v
                (h_block_size, v_block_size),
                lambda i, j, k: (
                    k,
                    c_index * num_v_blocks_per_core
                    + jnp.minimum(j, num_v_blocks_per_core - 1),
                ),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # w_vm1
                (h_block_size, v_block_size),
                lambda i, j, k: (
                    k,
                    c_index * num_v_blocks_per_core + jnp.maximum(0, j - 1),
                ),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(  # lse
                (b_block_size,),
                lambda i, j, k: (i,),
                memory_space=pltpu.VMEM,
            ),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),  # x_grad
            pl.BlockSpec(memory_space=pltpu.HBM),  # w_grad
        ],
    )(
        dout_hbm_ref,
        x_hbm_ref,
        labels_hbm_ref,
        w_hbm_ref,
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

  # pylint: disable-next=unpacking-non-sequence
  x_grad_blocks, w_grad = bwd_kernel(dout, x, labels, w, lse)
  x_grad = jnp.sum(x_grad_blocks, axis=0)[:b_dim, :h_dim]
  w_grad = w_grad[:h_dim, :v_dim]
  return x_grad, w_grad


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

  Computes dense Softmax gradients and applies sparse target label subtraction
  in-place inside Pallas Mosaic TPU kernel.

  Args:
    dout: The output's gradient of the Linear Cross-Entropy kernel. Since the
      output is loss, the gradient is usually 1.0 when reduction is "sum" or
      "mean", or shape (B,) when reduction is "none".
    lse: The log-sum-exp of the from the forward pass residuals.
    x: The last layer output in the dimension of (B, H) where B is the batch
      dimension, and H is the hidden dimension.
    labels: The ground truth labels index in the dimension of (B,).
    w: The linear projection weight matrix in the dimension of (H, V) where V is
      the dimension of the output logits aka vocabulary size.
    b_block_size: The block size for the batch dimension.
    h_block_size: The block size for the hidden dimension.
    v_block_size: The block size for the vocabulary dimension.
    reduction: The reduction method for the cross entropy loss. Can be set to
      "sum", "mean" or "none" explicitly.
    preferred_element_type: Preferred element type for computation.

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

  h_dim = x.shape[-1]
  v_dim = w.shape[1]

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
      b_dim=b_dim,
      h_dim=h_dim,
      v_dim=v_dim,
      preferred_element_type=preferred_element_type,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
  )

  # There is no gradient for the labels
  return (x_grad, w_grad)
