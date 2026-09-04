# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Example of a PyTorch model running on TPU calling a Tokamax kernel.

This example highlights the amount of boilerplate required.
"""

import time

from absl import app

### ----------------------------------------------------------------------------
### This section sets up a standard PyTorch model. It is pure PyTorch.
### ----------------------------------------------------------------------------

# pylint: disable=g-import-not-at-top
import torch
from torch import nn

class TrainingTransformer(nn.Module):
  """A simplified decoder transformer model for training.

  This is a simplified transformer model for demonstration purposes. The
  key modules are the embedding and final linear, softmax, and crossentropy
  layers.

  This training model models the following mixed precision. This is not
  strictly necessary for this example, but will be necessary in future work.

  * Master weights: fp32
  * Gradients for master weights: fp32
  * Activations: bf16
  * Matmul ops inside decoder blocks: bf16
  * Norms: n/a but assumed fp32
  * Matmul ops for final linear projection from BSH to BSV: fp32

  Attributes:
    seq_len: Length of the input sequence.
    vocab_size: Size of the vocabulary.
    hidden_dim: Hidden dimension of the model.
    num_decoder_blocks: Number of decoder blocks in the model.
    use_tokamax: Whether to use the custom op.
  """

  def __init__(
      self,
      seq_len: int,
      vocab_size: int,
      hidden_dim: int,
      num_decoder_blocks: int,
      use_tokamax: bool = False,
  ):
    super().__init__()
    self.seq_len = seq_len
    self.use_tokamax = use_tokamax

    self.tok_embedding = nn.Embedding(
        vocab_size, hidden_dim, dtype=torch.float32
    )
    self.pos_embedding = nn.Embedding(seq_len, hidden_dim, dtype=torch.float32)
    self.layers = nn.ModuleList([
        nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float32)
        for _ in range(num_decoder_blocks)
    ])
    self.linear = nn.Linear(hidden_dim, vocab_size, dtype=torch.float32)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward pass for the training decoder.

    float32 matmul precision must be set to medium to enable
    the intended mixed precision training.

    Args:
      input_ids: Input token IDs.

    Returns:
      Total loss.
    """
    if input_ids.size(1) != self.seq_len + 1:
      raise ValueError(
          f"Input length must be {self.seq_len + 1}, got {input_ids.size(1)}"
      )

    device = input_ids.device

    # Call embedding layers.
    tok_embed = self.tok_embedding(input_ids[:, : self.seq_len])
    pos_embed = self.pos_embedding(torch.arange(self.seq_len, device=device))
    embed = (tok_embed + pos_embed).to(torch.bfloat16)

    # Call all but the final decoder block using autocast.
    for layer in self.layers[:-1]:
      with torch.autocast(
          device_type=input_ids.device.type, dtype=torch.bfloat16
      ):
        assert embed.dtype == torch.bfloat16
        assert layer.weight.dtype == torch.float32

        embed = layer(embed)
        assert embed.dtype == torch.bfloat16

    # Call final decoder block. Do not autocast it down.
    # fp32 inputs, bf16 op precision, fp32 accumulations
    assert embed.dtype == torch.bfloat16
    assert self.layers[-1].weight.dtype == torch.float32

    # Do not reset matmul precision here; it may invalidate the compile cache.
    assert torch.get_float32_matmul_precision() == "medium"
    embed = embed.to(torch.float32)
    embed = self.layers[-1](embed)
    assert embed.dtype == torch.float32

    # Final linear, logsoftmax, and crossentropy.
    if self.use_tokamax:
      embed_flat = embed.reshape(-1, embed.size(-1))
      labels_flat = input_ids[:, 1:].reshape(-1).to(torch.int32)

      # Call custom op
      loss, _ = lsce_loss(
          embed_flat,
          labels_flat,
          self.linear.weight.t(),
          512,
          512,
          1024,
          "mean",
          "float32",
      )
    else:
      logits = self.linear(embed)
      assert logits.dtype == torch.float32

      labels = input_ids[:, 1:]
      loss = nn.functional.cross_entropy(
          logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
      )

    assert loss.dtype == torch.float32

    return loss


### ----------------------------------------------------------------------------
### This section uses the TorchTPU backend to train the model.
### ----------------------------------------------------------------------------

# pylint: disable=g-import-not-at-top
import torch_tpu

DEVICE = torch.device("tpu")


@torch.compile(backend="tpu")
def train_step(model, input_ids):
  loss = model(input_ids)
  loss.backward()
  return loss


def train_model(model, input_ids):
  """Instantiates and trains a single batch of data."""

  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, capturable=True)

  for epoch in range(100):
    start_time = time.time()

    loss = train_step(model, input_ids)
    optimizer.step()
    optimizer.zero_grad()

    torch.tpu.synchronize()
    end_time = time.time()
    print(
        f"{epoch=}: loss={loss.item():.4f}, time={end_time - start_time:.4f}s"
    )


### ----------------------------------------------------------------------------
### TorchTPU adapters to call JAX -> Tokamax kernel
### ----------------------------------------------------------------------------

# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
import jax
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu_kernel as tokamax_kernel
import torch_tpu._internal.pallas.pallas


def loss_fwd_jax(
    x: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    reduction: str,
    preferred_element_type: str,
) -> tuple[jax.Array, jax.Array]:
  """Wrapper for tokamax kernel with type hints for torch_tpu...jaxop."""
  dtype = (
      jax.numpy.float32
      if preferred_element_type == "float32"
      else jax.numpy.bfloat16
  )
  return tokamax_kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
      x,
      labels,
      weights,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      reduction=reduction,
      preferred_element_type=dtype,
  )


def loss_bwd_jax(
    dout: jax.Array,
    lse: jax.Array,
    x: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    reduction: str,
    preferred_element_type: str,
) -> tuple[jax.Array, jax.Array]:
  """Wrapper for tokamax kernel with type hints for torch_tpu...jaxop."""
  dtype = (
      jax.numpy.float32
      if preferred_element_type == "float32"
      else jax.numpy.bfloat16
  )
  return tokamax_kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
      dout,
      lse,
      x,
      labels,
      weights,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      reduction=reduction,
      preferred_element_type=dtype,
  )


# Register forward with pallas.jax_op
# pylint: disable=protected-access
lsce_loss: torch._library.custom_ops.CustomOpDef = (
    torch_tpu._internal.pallas.pallas.jax_op(
        "custom_op::lsce_loss", loss_fwd_jax
    )
)

# Register backward with pallas.jax_op
# pylint: disable=protected-access
lsce_loss_backward: torch._library.custom_ops.CustomOpDef = (
    torch_tpu._internal.pallas.pallas.jax_op(
        "custom_op::lsce_loss_bwd", loss_bwd_jax
    )
)


# Register autograd
def setup_context(ctx, inputs, output):
  (
      x,
      labels,
      weights,
      b_block_size,
      h_block_size,
      v_block_size,
      reduction,
      preferred_element_type,
  ) = inputs
  """Callback for torch register_autograd."""
  loss, lse = output
  del loss  # Unused
  ctx.save_for_backward(x, labels, weights, lse)
  ctx.b_block_size = b_block_size
  ctx.h_block_size = h_block_size
  ctx.v_block_size = v_block_size
  ctx.reduction = reduction
  ctx.preferred_element_type = preferred_element_type


def backward(ctx, d_loss, d_lse):
  """Callback for torch register_autograd."""
  del d_lse  # Unused
  x, labels, weights, lse = ctx.saved_tensors
  grad_x, grad_w = lsce_loss_backward(
      d_loss,
      lse,
      x,
      labels,
      weights,
      ctx.b_block_size,
      ctx.h_block_size,
      ctx.v_block_size,
      ctx.reduction,
      ctx.preferred_element_type,
  )
  return grad_x, None, grad_w, None, None, None, None, None


lsce_loss.register_autograd(backward, setup_context=setup_context)

### ----------------------------------------------------------------------------
### Demo of pure PyTorch and Tokamax kernel.
### ----------------------------------------------------------------------------


def demo_train(use_tokamax: bool):
  """Test creating model, data, and training."""
  torch.set_float32_matmul_precision("medium")
  torch.manual_seed(42)
  # Make model
  seq_len: int = 1_024
  vocab_size: int = 30_522
  hidden_dim: int = 2_048
  num_decoder_blocks: int = 3
  model = TrainingTransformer(
      seq_len,
      vocab_size,
      hidden_dim,
      num_decoder_blocks,
      use_tokamax=use_tokamax,
  )
  model = model.to(device=DEVICE)

  # Make data
  batch_size: int = 4
  input_ids = (
      (torch.arange(batch_size * (seq_len + 1)) % vocab_size)
      .view(batch_size, seq_len + 1)
      .to(device=DEVICE)
  )

  train_model(model, input_ids)


def main(argv):
  del argv  # Unused
  print("=== Pure PyTorch Training ===")
  demo_train(use_tokamax=False)
  print("=== Tokamax Kernel Training ===")
  demo_train(use_tokamax=True)


if __name__ == "__main__":
  app.run(main)
