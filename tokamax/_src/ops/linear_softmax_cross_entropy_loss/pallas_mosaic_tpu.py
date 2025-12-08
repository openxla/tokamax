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


from dataclasses import dataclass
from functools import partial, reduce
import math
from typing import Annotated, Literal
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real, Scalar
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
from typing_extensions import override


def validate_inputs(
    x: Real[Array, "B H"],
    labels: Real[Array, "B V"],
    w: Real[Array, "H V"],
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
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
  if x.shape[1] % h_block_size != 0:
    raise ValueError(
        "The hidden dimension of x must be a multiple of the H block size."
    )
  if labels.shape[0] % b_block_size != 0:
    raise ValueError(
        "The batch dimension of labels must be a multiple of the B block size."
    )
  if w.shape[0] % h_block_size != 0:
    raise ValueError(
        "The hidden dimension of w must be a multiple of the H block size."
    )


def linear_softmax_cross_entropy_loss_forward_pallas_kernel(
    x_ref,
    labels_ref,
    w_ref,
    loss_ref,
    lse_ref,
    xw_tiled,
    b_block_loss_ref,
    reduction: Literal["sum", "mean"],
    v_dim: int,
):
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
    x_ref: BlockRef for the input activations `x` (b_block_size, h_block_size).
    labels_ref: BlockRef for the labels (b_block_size, v_block_size).
    w_ref: BlockRef for the weights `w` (h_block_size, v_block_size).
    loss_ref: BlockRef for the scalar loss accumulator (1,).
    lse_ref: BlockRef for the log-sum-exp accumulator per batch item
      (b_block_size,). Used for backward pass
    xw_tiled: Scratch BlockRef for accumulating `x @ w` results (b_block_size,
      v_block_size).
    b_block_loss_ref: Scratch BlockRef for accumulating loss per B block (1,).
    reduction: The reduction method ("sum" or "mean") for the loss accumulation.
  """
  b_index, v_index, h_index = (pl.program_id(i) for i in range(3))
  b_block_size = x_ref.shape[0]
  v_block_size = w_ref.shape[1]
  num_b_blocks, num_v_blocks, num_h_blocks = (
      pl.num_programs(i) for i in range(3)
  )

  # Padding if V dimension is not aligned to the V block size
  @pl.when(v_index == num_v_blocks - 1)
  def pad_non_aligned_v_block():
    if v_dim % v_block_size != 0:
      rem = v_dim % v_block_size
      w_ref[:, rem:] = jnp.zeros(
          (w_ref.shape[0], w_ref.shape[1] - rem), dtype=w_ref.dtype
      )

  @pl.when(reduce(jnp.logical_and, (b_index == 0, v_index == 0, h_index == 0)))
  def _():
    loss_ref[0] = 0.0

  @pl.when(jnp.logical_and(v_index == 0, h_index == 0))
  def _():
    lse_ref[...] = jnp.full_like(lse_ref, -jnp.inf)
    b_block_loss_ref[0] = 0.0

  @pl.when(h_index == 0)
  def _():
    xw_tiled[...] = jnp.zeros_like(xw_tiled)

  xw_tiled[...] += x_ref[...] @ w_ref[...]

  @pl.when(h_index == num_h_blocks - 1)
  def _():
    # Convert labels to one-hot, due to chunking on v dimension, the indices
    # needs to be shifted down by the v starting index. Negative or out-of-bound
    # indices are OK since jax.nn.one_hot will set them to 0.
    labels_adjusted = labels_ref[...] - v_index * v_block_size
    labels_one_hot = jax.nn.one_hot(
        labels_adjusted, num_classes=v_block_size, dtype=x_ref.dtype
    )
    b_block_loss_ref[0] -= jnp.sum(labels_one_hot * xw_tiled[...])
    lse_block = jax.nn.logsumexp(xw_tiled[...], axis=-1)
    lse_ref[...] = jnp.logaddexp(lse_ref[...], lse_block)

  @pl.when(
      jnp.logical_and(v_index == num_v_blocks - 1, h_index == num_h_blocks - 1)
  )
  def _():
    b_block_loss_ref[0] += jnp.sum(lse_ref[...])
    if reduction == "mean":
      # For mean reduction, use online averaging algorithm
      loss_ref[0] = (
          loss_ref[0] * b_index / (b_index + 1)
          + b_block_loss_ref[0] / (b_index + 1) / b_block_size
      )
    else:
      # Sum reduction
      loss_ref[0] += b_block_loss_ref[0]


@partial(
    jax.jit,
    static_argnames=[
        "b_block_size",
        "h_block_size",
        "v_block_size",
        "reduction",
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
    reduction: Literal["sum", "mean"] = "sum",
) -> tuple[Real[Scalar, ""], Real[Array, "B"]]:
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
    reduction: The reduction method ("sum" or "mean") for the loss accumulation.

  Returns:
    The loss in scalar and thehe log-sum-exp of the used for backward pass.

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

  h_dim = x.shape[-1]
  v_dim = w.shape[1]
  b_dim = x.shape[0]

  num_b_blocks = math.ceil(b_dim / b_block_size)
  num_h_blocks = math.ceil(h_dim / h_block_size)
  num_v_blocks = math.ceil(v_dim / v_block_size)

  # Forward
  loss, lse = pl.pallas_call(
      partial(
          linear_softmax_cross_entropy_loss_forward_pallas_kernel,
          reduction=reduction,
          v_dim=v_dim,
      ),
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
          pl.BlockSpec(memory_space=pltpu.SMEM),  # loss
          pl.BlockSpec(
              (b_block_size,), lambda i, j, k: i, memory_space=pltpu.VMEM
          ),  # lse
      ],
      out_shape=[
          jax.ShapeDtypeStruct(shape=(1,), dtype=jnp.float32),  # Loss
          jax.ShapeDtypeStruct(shape=(b_dim,), dtype=jnp.float32),  # LSE
      ],
      scratch_shapes=(
          pltpu.VMEM(
              (b_block_size, v_block_size), dtype=jnp.float32
          ),  # xw_tiled
          pltpu.SMEM((1,), dtype=jnp.float32),  # b_block_loss
      ),
      grid=(num_b_blocks, num_v_blocks, num_h_blocks),
  )(x, labels, w)

  return loss[0], lse


def linear_softmax_cross_entropy_loss_backward_pallas_kernel(
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
    reduction: Literal["sum", "mean"],
):
  """Pallas kernel for the backward pass of Linear Softmax Cross-Entropy Loss.

  Args:
    x_ref: Input activations `x` (b_block_size, h_block_size).
    labels_ref: One-hot encoded labels (b_block_size, v_block_size).
    w_ref: LM Head projection weights `w` (h_block_size, v_block_size).
    lse_ref: BlockRef for the log-sum-exp accumulator per batch item
      (b_block_size,). Used for backward pass
    x_grad_hbm_ref: The output gradient of x (b_dim, h_dim) on HBM
    w_grad_hbm_ref: The output gradient of w (h_dim, v_dim)
    xw_scratch_ref: Scratch BlockRef for accumulating logits results
      (b_block_size, v_block_size).
    x_grad_tile_ref: Scratch BlockRef for accumulating x gradient results
      (b_block_size, h_block_size).
    w_grad_tile_ref: Scratch BlockRef for accumulating w results (h_block_size,
      v_block_size).
    reduction: The reduction method ("sum" or "mean") for the gradient
      accumulation.
  """

  b_index, v_index, stage_index, h_index = (pl.program_id(i) for i in range(4))
  num_b_blocks, num_v_blocks, _, num_h_blocks = (
      pl.num_programs(i) for i in range(4)
  )
  b_block_size, h_block_size = x_ref.shape
  v_block_size = w_ref.shape[1]
  v_dim = w_grad_hbm_ref.shape[-1]

  # Padding if V dimension is not aligned to the V block size
  @pl.when(v_index == num_v_blocks - 1)
  def pad_non_aligned_v_block():
    if v_dim % v_block_size != 0:
      rem = v_dim % v_block_size
      w_ref[:, rem:] = jnp.zeros(
          (w_ref.shape[0], w_ref.shape[1] - rem), dtype=w_ref.dtype
      )

  # Initialize logits x@w
  @pl.when(jnp.logical_and(stage_index == 0, h_index == 0))
  def init_logits():
    xw_scratch_ref[...] = x_ref[...] @ w_ref[...]

  # Accumulate logits x@w
  @pl.when(jnp.logical_and(stage_index == 0, h_index != 0))
  def accumulate_logits():
    xw_scratch_ref[...] += x_ref[...] @ w_ref[...]

  # Calculate the actual block size if v_dim is not a multiple of v_block_size
  cur_v_block_size = jnp.minimum(v_dim - v_block_size * v_index, v_block_size)

  # V Block size must be multiple of 128 to perform DMA (copy).
  # Aligning the V block size to 128
  cur_v_block_size = (jnp.ceil(cur_v_block_size / 128) * 128).astype(jnp.int32)
  cur_v_block_size = pl.multiple_of(cur_v_block_size, 128)

  # Slicing x_grad and x_grad HBM ref to prepare for tiled read / write
  x_grad_slice = x_grad_hbm_ref.at[
      pl.dslice(b_index * b_block_size, b_block_size),
      pl.ds(h_index * h_block_size, h_block_size),
  ]
  w_grad_slice = w_grad_hbm_ref.at[
      pl.ds(h_index * h_block_size, h_block_size),
      pl.dslice(v_index * v_block_size, cur_v_block_size),
  ]
  w_grad_tile_slice = w_grad_tile_ref.at[:, pl.dslice(0, cur_v_block_size)]

  def perform_x_grad_tile_reduction():
    if reduction == "mean":
      # Mean reduction happens on the last accumulation on V for
      # numerical stability
      @pl.when(
          reduce(
              jnp.logical_and, (stage_index == 1, v_index == num_v_blocks - 1)
          )
      )
      def _():
        x_grad_tile_ref[...] /= num_b_blocks * b_block_size

  def perform_w_grad_tile_reduction():
    if reduction == "mean":
      # Mean reduction happens on the last accumulation on B for
      # numerical stability
      @pl.when(
          reduce(
              jnp.logical_and, (stage_index == 1, b_index == num_b_blocks - 1)
          )
      )
      def _():
        w_grad_tile_ref[...] /= num_b_blocks * b_block_size

  # Async copy ops are only defined here. It only starts after calling .start().
  x_write_future = pltpu.make_async_copy(
      x_grad_tile_ref, x_grad_slice, sem=x_write_sem
  )
  w_write_future = pltpu.make_async_copy(
      w_grad_tile_slice, w_grad_slice, sem=w_write_sem
  )
  x_read_future = pltpu.make_async_copy(
      x_grad_slice, x_grad_tile_ref, sem=x_read_sem
  )
  w_read_future = pltpu.make_async_copy(
      w_grad_slice, w_grad_tile_slice, sem=w_read_sem
  )

  # Preload x_grad and w_grad async before computing softmax to
  # overlap computation
  @pl.when(reduce(jnp.logical_and, (stage_index == 1, v_index != 0)))
  def x_read():
    x_read_future.start()

  # Preload w_grad
  @pl.when(reduce(jnp.logical_and, (stage_index == 1, b_index != 0)))
  def w_read():
    w_read_future.start()

  # Compute Softmax and store s = - labels + softmax(x@w) to xw_scratch_ref
  @pl.when(jnp.logical_and(stage_index == 1, h_index == 0))
  def compute_s():
    labels_adjusted = labels_ref[...] - v_index * v_block_size
    labels_one_hot = jax.nn.one_hot(
        labels_adjusted, num_classes=v_block_size, dtype=x_ref.dtype
    )
    xw_scratch_ref[...] = -labels_one_hot + jnp.exp(
        xw_scratch_ref[...] - lse_ref[...][:, None]
    )

  # Init X gradient
  @pl.when(reduce(jnp.logical_and, (stage_index == 1, v_index == 0)))
  def init_x_grad():
    x_grad_tile_ref[...] = jax.lax.dot_general(
        xw_scratch_ref[...], w_ref[...], (((1,), (1,)), ((), ()))
    )
    perform_x_grad_tile_reduction()
    x_write_future.start()

  # Init W gradient
  @pl.when(reduce(jnp.logical_and, (stage_index == 1, b_index == 0)))
  def init_w_grad():
    w_grad_tile_ref[...] = jax.lax.dot_general(
        x_ref[...], xw_scratch_ref[...], (((0,), (0,)), ((), ()))
    )
    perform_w_grad_tile_reduction()
    w_write_future.start()

  # Accumulate X grad on V dimension
  @pl.when(reduce(jnp.logical_and, (stage_index == 1, v_index != 0)))
  def accumulate_x_grad():
    res = jax.lax.dot_general(
        xw_scratch_ref[...], w_ref[...], (((1,), (1,)), ((), ()))
    )
    x_read_future.wait()
    x_grad_tile_ref[...] += res
    perform_x_grad_tile_reduction()
    x_write_future.start()

  # Accumulate W grad on B dimension
  @pl.when(reduce(jnp.logical_and, (stage_index == 1, b_index != 0)))
  def accumulate_w_grad():
    res = jax.lax.dot_general(
        x_ref[...], xw_scratch_ref[...], (((0,), (0,)), ((), ()))
    )
    w_read_future.wait()
    w_grad_tile_ref[...] += res
    perform_w_grad_tile_reduction()
    w_write_future.start()

  @pl.when(stage_index == 1)
  def wait_async_writes():
    x_write_future.wait()
    w_write_future.wait()


@partial(
    jax.jit,
    static_argnames=[
        "b_block_size",
        "h_block_size",
        "v_block_size",
        "reduction",
    ],
)
def linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
    dout: Real[Array, ""],
    lse: Real[Array, "B"],
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
    reduction: Literal["sum", "mean"] = "sum",
) -> tuple[Real[Array, "B H"], Real[Array, "B V"]]:
  """The pallas kernel implementation of the Linear Softmax Cross-Entropy Loss backward kernel.

  The backward pass is also chunking the x, labels and w in all B, H and V
  dimensions so it can fit in the TPU VMEM. To not materialize the logits, the
  backward pass will re-compute the logits blockwise and cache in VMEM for the
  gradient calculation. This leads to also almost 0 memory overhead in backward
  pass.

  Args:
    dout: The output's gradient of the Linear Cross-Entropy kernel. Since the
      output is loss, the gradient is usually 1.0.
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
      "sum" or "mean" explicitly.

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

  v_dim = w.shape[1]
  b_dim = x.shape[0]
  h_dim = x.shape[-1]
  num_b_blocks = math.ceil(b_dim / b_block_size)
  num_v_blocks = math.ceil(v_dim / v_block_size)
  num_h_blocks = math.ceil(h_dim / h_block_size)
  num_stages = 2

  # Backward
  x_grad, w_grad = pl.pallas_call(
      partial(
          linear_softmax_cross_entropy_loss_backward_pallas_kernel,
          reduction=reduction,
      ),
      in_specs=[
          pl.BlockSpec(  # x
              (b_block_size, h_block_size),
              lambda i, j, s, k: (i, k),
              memory_space=pltpu.VMEM,
          ),
          pl.BlockSpec(  # labels
              (b_block_size,),
              lambda i, j, s, k: (i),
              memory_space=pltpu.VMEM,
          ),
          pl.BlockSpec(  # w
              (h_block_size, v_block_size),
              lambda i, j, s, k: (k, j),
              memory_space=pltpu.VMEM,
          ),
          pl.BlockSpec(  # lse
              (b_block_size,), lambda i, j, s, k: (i,), memory_space=pltpu.VMEM
          ),
      ],
      out_specs=[
          pl.BlockSpec(memory_space=pltpu.HBM),  # x_grad
          pl.BlockSpec(memory_space=pltpu.HBM),  # w_grad
      ],
      out_shape=[
          jax.ShapeDtypeStruct(x.shape, dtype=jnp.float32),  # x_grad
          jax.ShapeDtypeStruct(w.shape, dtype=jnp.float32),  # w_grad
      ],
      scratch_shapes=(
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
      grid=(num_b_blocks, num_v_blocks, num_stages, num_h_blocks),
  )(x, labels, w, lse)

  # There is no gradient for the labels
  return (dout * x_grad, dout * w_grad)


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


def get_tpu_specific_default_config() -> Config:
  """Returns the heuristic config for based on TPU version."""
  if pltpu.get_tpu_info().generation >= 6:
    return Config(b_block_size=1024, h_block_size=512, v_block_size=2048)
  else:
    return Config(b_block_size=1024, h_block_size=512, v_block_size=512)


@dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(
    base.LinearSoftmaxCrossEntropyLoss
):
  """Wrapper for the tokamax Op API for Pallas Mosaic TPU kernel."""

  def __post_init__(self):
    object.__setattr__(
        self,
        "vjp",
        PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp(config=self.config),
    )

  def _fwd(
      self,
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean"] = "sum",
      config: Config,
      return_residuals: bool,
  ) -> tuple[jax.Array, base.Residuals]:
    loss, lse = linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
        x,
        labels,
        w,
        b_block_size=config.b_block_size,
        h_block_size=config.h_block_size,
        v_block_size=config.v_block_size,
        reduction=reduction,
    )
    return loss, (lse,)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return get_tpu_specific_default_config()

  # TODO: Implement an autotuning search space.
  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return set()

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5


@dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp(
    base.LinearSoftmaxCrossEntropyLossVjp
):
  """Pallas TPU implementation of Linear Softmax Cross-Entropy Loss VJP."""

  def _fwd(
      self,
      residuals: base.Residuals,
      out: Real[Array, ""],
      dout: Real[Array, ""],
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean"] = "sum",
      config: Config,
      return_residuals: bool,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    """Computes Linear Softmax Cross-Entropy Loss VJP `(dx, dlabels, dw)`."""
    (lse,) = residuals

    x_grad, w_grad = linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
        dout,
        lse,
        x,
        labels,
        w,
        b_block_size=config.b_block_size,
        h_block_size=config.h_block_size,
        v_block_size=config.v_block_size,
        reduction=reduction,
    )
    labels_grad = jnp.zeros_like(labels)
    return (x_grad, labels_grad, w_grad), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return get_tpu_specific_default_config()

  # TODO: Implement an autotuning search space.
  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return set()

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
