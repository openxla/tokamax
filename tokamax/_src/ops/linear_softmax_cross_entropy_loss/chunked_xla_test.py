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

"""Tests for chunked XLA implementation of linear softmax cross-entropy loss."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.linear_softmax_cross_entropy_loss import chunked_xla
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils


class ChunkedXlaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="small",
          b_dim=128,
          h_dim=512,
          v_dim=1024,
          reduction="sum",
          b_block_sz=64,
          v_block_sz=128,
      ),
      dict(
          testcase_name="medium",
          b_dim=256,
          h_dim=1024,
          v_dim=2048,
          reduction="mean",
          b_block_sz=128,
          v_block_sz=256,
      ),
  )
  def test_bwd_matches_reference(
      self, b_dim, h_dim, v_dim, reduction, b_block_sz, v_block_sz
  ):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(42), b_dim, h_dim, v_dim
    )
    # We need LSE from forward pass for the backward pass.
    # We can use the reference forward pass to get it.
    _, lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, w, reduction=reduction
    )

    dout = jnp.array(1.0, dtype=x.dtype)

    # Run chunked XLA backward
    chunked_dx, chunked_dw = (
        chunked_xla.linear_softmax_cross_entropy_loss_bwd_chunked_xla(
            dout,
            lse,
            x,
            labels,
            w,
            b_block_sz=b_block_sz,
            v_block_sz=v_block_sz,
            reduction=reduction,
        )
    )

    # Run reference backward
    ref_dx, ref_dw = reference.linear_softmax_cross_entropy_loss_bwd_reference(
        dout, lse, x, labels, w, reduction=reduction
    )

    atol = 3e-2
    rtol = 3e-2
    self.assertTrue(
        jnp.allclose(chunked_dx, ref_dx, atol=atol, rtol=rtol),
        f"dX mismatch: max diff {jnp.max(jnp.abs(chunked_dx - ref_dx))}",
    )
    self.assertTrue(
        jnp.allclose(chunked_dw, ref_dw, atol=atol, rtol=rtol),
        f"dW mismatch: max diff {jnp.max(jnp.abs(chunked_dw - ref_dw))}",
    )


if __name__ == "__main__":
  absltest.main()
