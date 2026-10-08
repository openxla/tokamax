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
"""Tests for the base class of the Ragged Dot PyTorch Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_dot import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.ragged_dot import torch_base
import torch
import torch_tpu  # pylint: disable=unused-import


class RaggedDotBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This op only works on TPU.")

    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)
    self.seed = seed
    torch.manual_seed(seed)

  def _generate_random_data(
      self,
      m: int,
      k: int,
      n: int,
      num_groups: int,
      dtype: torch.dtype = torch.float32,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lhs = torch.randn(
        m,
        k,
        device="tpu",
        dtype=dtype,
        requires_grad=True,
    )
    rhs = torch.randn(
        num_groups,
        k,
        n,
        device="tpu",
        dtype=dtype,
        requires_grad=True,
    )
    gs_tuple = jax_base.generate_group_sizes(
        m=m, num_groups=num_groups, seed=self.seed
    )
    group_sizes = torch.tensor(gs_tuple, device="tpu", dtype=torch.int32)
    dout = torch.randn(m, n, device="tpu", dtype=dtype)
    return lhs, rhs, group_sizes, dout

  def test_ragged_dot_numerics_matches_ref_implementation(self):
    """Tests that the base class numerics matches the JAX reference op."""
    m = 512
    k = 256
    n = 512
    num_groups = 4

    lhs_torch, rhs_torch, group_sizes_torch, dout_torch = (
        self._generate_random_data(m, k, n, num_groups, dtype=torch.float32)
    )

    jax_lhs = jnp.array(lhs_torch.detach().cpu().numpy())
    jax_rhs = jnp.array(rhs_torch.detach().cpu().numpy())
    jax_group_sizes = jnp.array(group_sizes_torch.detach().cpu().numpy())
    jax_dout = jnp.array(dout_torch.detach().cpu().numpy())

    forward_config, backward_config = torch_utils.get_configs(
        torch_base.RaggedDot,
        lhs_torch,
        rhs_torch,
        group_sizes=group_sizes_torch,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(forward_config)
    self.assertIsNotNone(backward_config)

    out_torch = torch_base.RaggedDot(
        lhs_torch,
        rhs_torch,
        group_sizes_torch,
        configs=(forward_config, backward_config),
    )
    out_torch.backward(dout_torch)
    dlhs_torch = lhs_torch.grad.clone()
    drhs_torch = rhs_torch.grad.clone()

    fn_reference = jax_base.RaggedDot()
    out_ref, vjp_fn = jax.vjp(
        lambda l, r: fn_reference(l, r, group_sizes=jax_group_sizes),
        jax_lhs,
        jax_rhs,
    )
    dlhs_ref, drhs_ref = vjp_fn(jax_dout)

    out_ref_as_torch = torch.as_tensor(np.asarray(out_ref), device="tpu")
    dlhs_ref_as_torch = torch.as_tensor(np.asarray(dlhs_ref), device="tpu")
    drhs_ref_as_torch = torch.as_tensor(np.asarray(drhs_ref), device="tpu")

    torch.testing.assert_close(
        out_torch, out_ref_as_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        dlhs_torch, dlhs_ref_as_torch, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        drhs_torch, drhs_ref_as_torch, rtol=1e-4, atol=1e-4
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="small_size_test",
          m=256,
          k=128,
          n=256,
          num_groups=4,
      ),
      dict(
          testcase_name="medium_size_test",
          m=1024,
          k=512,
          n=1024,
          num_groups=8,
      ),
  )
  def test_reference_running_correctly(self, m, k, n, num_groups):
    lhs, rhs, group_sizes, dout = self._generate_random_data(
        m, k, n, num_groups
    )
    configs = torch_utils.get_configs(
        torch_base.RaggedDot,
        lhs,
        rhs,
        group_sizes,
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(configs[0])
    self.assertIsNotNone(configs[1])

    out, residuals = torch_base.RaggedDot(
        lhs,
        rhs,
        group_sizes,
        return_residuals=True,
        configs=configs,
    )
    out.backward(dout)
    self.assertIsNotNone(lhs.grad)
    self.assertIsNotNone(rhs.grad)
    self.assertIsNotNone(out)
    self.assertIsNotNone(residuals)

  def test_activation_raises_error(self):
    lhs_torch, rhs_torch, group_sizes_torch, _ = self._generate_random_data(
        m=256, k=128, n=256, num_groups=4, dtype=torch.float32
    )
    with self.assertRaisesRegex(
        NotImplementedError,
        "activations are not supported on Torch TPU Tokamax",
    ):
      torch_base.RaggedDot(
          lhs_torch,
          rhs_torch,
          group_sizes_torch,
          activation=jax.nn.relu,
      )


if __name__ == "__main__":
  absltest.main()
