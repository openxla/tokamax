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
"""Tests for Torch DeepSeek-V4 RoPE base op."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.deepseek_rope import base as jax_base
from tokamax._src.ops.deepseek_rope import test_base as jax_test_base
from tokamax.experimental.torch_tpu.ops import torch_test_utils
from tokamax.experimental.torch_tpu.ops.deepseek_rope import torch_base
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


def _make_torch_and_jax_inputs(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    seed: int = 0,
    max_position: int = jax_test_base.MAX_POSITION,
    rotary_dim: int = jax_test_base.ROTARY_DIM,
    device: str = "tpu",
) -> tuple[dict[str, torch.Tensor], dict[str, jax.Array]]:
  """Creates PyTorch inputs on `device` and matching JAX inputs."""
  x_jax, positions_jax, cos_sin_cache_jax = jax_test_base.make_inputs(
      shape,
      dtype,
      seed=seed,
      max_position=max_position,
      rotary_dim=rotary_dim,
  )
  torch_inputs = dict(
      x=torch_test_utils.jax_to_torch(x_jax, device=device),
      positions=torch_test_utils.jax_to_torch(positions_jax, device=device),
      cos_sin_cache=torch_test_utils.jax_to_torch(
          cos_sin_cache_jax, device=device
      ),
  )
  jax_inputs = dict(
      x=x_jax,
      positions=positions_jax,
      cos_sin_cache=cos_sin_cache_jax,
  )
  return torch_inputs, jax_inputs


class DeepseekRopeBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    torch.manual_seed(0)

  def test_canonicalize_jax_quant_dtype(self):
    self.assertIsNone(torch_base.canonicalize_jax_quant_dtype("rope", None))
    self.assertEqual(
        torch_base.canonicalize_jax_quant_dtype("rope_quant", None),
        jnp.dtype(jnp.float8_e4m3fn),
    )
    self.assertEqual(
        torch_base.canonicalize_jax_quant_dtype(
            "rope_quant", torch.float8_e5m2
        ),
        jnp.dtype(jnp.float8_e5m2),
    )

  @parameterized.product(
      shape=(
          (8, jax_test_base.NUM_HEADS, jax_test_base.HEAD_DIM),
          (64, jax_test_base.NUM_HEADS, jax_test_base.HEAD_DIM),
          (256, jax_test_base.NUM_HEADS, jax_test_base.HEAD_DIM),
          (8, jax_test_base.HEAD_DIM),
          (128, jax_test_base.HEAD_DIM),
      ),
      inverse=(False, True),
      dtype=(jnp.float32, jnp.bfloat16),
  )
  def test_rope_matches_jax_op(self, shape, inverse, dtype):
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        shape, dtype, seed=42
    )
    actual = torch_base.DeepseekRope(
        **torch_inputs, mode="rope", inverse=inverse
    )
    desired_jax = jax_base.DeepseekRope()(
        **jax_inputs, mode="rope", inverse=inverse
    )
    self.assertIsInstance(desired_jax, jax.Array)
    desired = torch_test_utils.jax_to_torch(desired_jax)

    self.assertIsInstance(actual, torch.Tensor)
    self.assertEqual(actual.shape, desired.shape)
    self.assertEqual(actual.dtype, desired.dtype)
    tol = 1e-6 if dtype == jnp.float32 else 1e-2
    torch.testing.assert_close(actual, desired, rtol=tol, atol=tol)

  def test_nope_channels_untouched(self):
    torch_inputs, _ = _make_torch_and_jax_inputs(
        (64, 8, jax_test_base.HEAD_DIM), jnp.float32, seed=7
    )
    x_orig = torch_inputs["x"].clone()
    actual = torch_base.DeepseekRope(**torch_inputs, mode="rope")
    self.assertIsInstance(actual, torch.Tensor)
    torch.testing.assert_close(
        actual[..., : -jax_test_base.ROTARY_DIM],
        x_orig[..., : -jax_test_base.ROTARY_DIM],
        rtol=0.0,
        atol=0.0,
    )

  @parameterized.product(
      num_tokens=(8, 64, 256),
      inverse=(False, True),
  )
  def test_qnorm_rope_matches_jax_op(self, num_tokens, inverse):
    shape = (num_tokens, 128, jax_test_base.HEAD_DIM)
    eps = 1e-6
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        shape, jnp.bfloat16, seed=13
    )
    actual = torch_base.DeepseekRope(
        **torch_inputs, mode="qnorm_rope", eps=eps, inverse=inverse
    )
    desired_jax = jax_base.DeepseekRope()(
        **jax_inputs, mode="qnorm_rope", eps=eps, inverse=inverse
    )
    self.assertIsInstance(desired_jax, jax.Array)
    desired = torch_test_utils.jax_to_torch(desired_jax)

    self.assertIsInstance(actual, torch.Tensor)
    self.assertEqual(actual.shape, desired.shape)
    self.assertEqual(actual.dtype, desired.dtype)
    torch.testing.assert_close(actual, desired, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      shape=((8, 64, 128), (64, 64, 128), (256, 128)),
      inverse=(False, True),
      quant_dtype=(
          None,
          torch.float8_e4m3fn,
          "float8_e4m3fn",
      ),
  )
  def test_rope_quant_matches_jax_op(self, shape, inverse, quant_dtype):
    torch_inputs, jax_inputs = _make_torch_and_jax_inputs(
        shape, jnp.bfloat16, seed=19
    )
    q_actual, scales_actual = torch_base.DeepseekRope(
        **torch_inputs,
        mode="rope_quant",
        inverse=inverse,
        quant_dtype=quant_dtype,
    )
    q_desired_jax, scales_desired_jax = jax_base.DeepseekRope()(
        **jax_inputs,
        mode="rope_quant",
        inverse=inverse,
        quant_dtype=jnp.float8_e4m3fn,
    )
    scales_desired = torch_test_utils.jax_to_torch(scales_desired_jax)

    self.assertEqual(q_actual.dtype, torch.float8_e4m3fn)
    self.assertEqual(tuple(q_actual.shape), shape)
    self.assertEqual(scales_actual.dtype, torch.float32)
    self.assertEqual(tuple(scales_actual.shape), shape[:-1])
    torch.testing.assert_close(
        scales_actual, scales_desired, rtol=1e-6, atol=1e-6
    )
    jax_test_base.assert_bits_equal(
        q_actual.view(torch.uint8).detach().cpu().numpy(),
        q_desired_jax,
    )

  def test_fake_impl(self):
    with fake_tensor.FakeTensorMode():
      x_3d = torch.empty((16, 8, 512), dtype=torch.bfloat16)
      x_quant = torch.empty((16, 8, 128), dtype=torch.bfloat16)
      positions = torch.empty((16,), dtype=torch.int32)
      cos_sin_cache = torch.empty((4096, 64), dtype=torch.float32)

      out_rope = torch_base.DeepseekRope(
          x_3d, positions, cos_sin_cache, mode="rope"
      )
      out_qnorm = torch_base.DeepseekRope(
          x_3d, positions, cos_sin_cache, mode="qnorm_rope"
      )
      q_default, scales_default = torch_base.DeepseekRope(
          x_quant, positions, cos_sin_cache, mode="rope_quant"
      )
      q_e5m2, scales_e5m2 = torch_base.DeepseekRope(
          x_quant,
          positions,
          cos_sin_cache,
          mode="rope_quant",
          quant_dtype=torch.float8_e5m2,
      )

    self.assertEqual(tuple(out_rope.shape), (16, 8, 512))
    self.assertEqual(out_rope.dtype, torch.bfloat16)
    self.assertEqual(tuple(out_qnorm.shape), (16, 8, 512))
    self.assertEqual(out_qnorm.dtype, torch.bfloat16)
    self.assertEqual(tuple(q_default.shape), (16, 8, 128))
    self.assertEqual(q_default.dtype, torch.float8_e4m3fn)
    self.assertEqual(tuple(scales_default.shape), (16, 8))
    self.assertEqual(scales_default.dtype, torch.float32)
    self.assertEqual(tuple(q_e5m2.shape), (16, 8, 128))
    self.assertEqual(q_e5m2.dtype, torch.float8_e5m2)
    self.assertEqual(tuple(scales_e5m2.shape), (16, 8))
    self.assertEqual(scales_e5m2.dtype, torch.float32)

  def test_fake_impl_symbolic_shapes(self):
    class _RopeModule(torch.nn.Module):

      def forward(
          self,
          x: torch.Tensor,
          positions: torch.Tensor,
          cos_sin_cache: torch.Tensor,
      ) -> torch.Tensor:
        return torch_base.DeepseekRope(
            x, positions, cos_sin_cache, mode="rope"
        )

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    x = torch.ones((16, 8, 512), dtype=torch.bfloat16)
    positions = torch.zeros((16,), dtype=torch.int32)
    cos_sin_cache = torch.ones((4096, 64), dtype=torch.float32)

    exported = torch.export.export(
        _RopeModule(),
        (x, positions, cos_sin_cache),
        dynamic_shapes=({0: num_tokens}, {0: num_tokens}, None),
    )
    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and torch_base.DeepseekRope.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    meta = op_nodes[0].meta["val"][0]
    self.assertIsInstance(meta.shape[0], torch.SymInt)
    self.assertEqual(tuple(meta.shape[1:]), (8, 512))

  @parameterized.parameters("rope", "qnorm_rope", "rope_quant")
  def test_torch_compile(self, mode):
    head_dim = 128 if mode == "rope_quant" else 512
    torch_inputs, _ = _make_torch_and_jax_inputs(
        (64, 8, head_dim), jnp.bfloat16, seed=42
    )

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, positions, cos_sin_cache):
      x = x + 0.1
      return torch_base.DeepseekRope(x, positions, cos_sin_cache, mode=mode)

    actual = compiled_fn(**torch_inputs)
    expected = torch_base.DeepseekRope(
        **{**torch_inputs, "x": torch_inputs["x"] + 0.1},
        mode=mode,
    )
    if mode == "rope_quant":
      q_actual, scales_actual = actual
      q_expected, scales_expected = expected
      torch.testing.assert_close(
          scales_actual, scales_expected, rtol=1e-6, atol=1e-6
      )
      jax_test_base.assert_bits_equal(
          q_actual.view(torch.uint8).detach().cpu().numpy(),
          q_expected.view(torch.uint8).detach().cpu().numpy(),
      )
    else:
      torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  absltest.main()
