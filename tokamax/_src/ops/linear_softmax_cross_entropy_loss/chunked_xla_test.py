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

import itertools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.linear_softmax_cross_entropy_loss import chunked_xla
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils


def _chunked_xla_test_cases():
  reductions = ["sum", "mean"]
  types = [
      ("f16", jnp.float16, None),
      ("f32", jnp.float32, None),
      ("f16_f16", jnp.float16, jnp.float16),
      ("f32_f32", jnp.float32, jnp.float32),
      ("f16_f32", jnp.float16, jnp.float32),
  ]

  for reduction, (type_name, dtype, pref_type) in itertools.product(
      reductions,
      types,
  ):
    yield dict(
        testcase_name=f"{reduction}_{type_name}",
        reduction=reduction,
        dtype=dtype,
        preferred_element_type=pref_type,
    )


class ChunkedXlaTest(parameterized.TestCase):

  @parameterized.named_parameters(*_chunked_xla_test_cases())
  def test_bwd_matches_reference(
      self,
      reduction,
      dtype,
      preferred_element_type,
      b_dim=256,
      h_dim=1024,
      v_dim=2048,
      b_block_sz=128,
      v_block_sz=256,
  ):
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(42), b_dim, h_dim, v_dim, dtype=dtype
    )
    lse = jax.nn.logsumexp(x @ w, axis=-1)
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
            preferred_element_type=preferred_element_type,
        )
    )

    # Run reference backward
    ref_dx, ref_dw = reference.linear_softmax_cross_entropy_loss_bwd_reference(
        dout, lse, x, labels, w, reduction=reduction
    )

    # Use looser tolerances for mixed precision or when accumulating float16
    # sums on GPU.
    is_mixed_precision = (
        preferred_element_type is not None and dtype != preferred_element_type
    )
    is_gpu_f16_sum = (
        jax.default_backend() == "gpu"
        and reduction == "sum"
        and dtype == jnp.float16
    )

    if is_mixed_precision or is_gpu_f16_sum:
      atol = rtol = 0.15
    else:
      atol = rtol = 3e-3
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
