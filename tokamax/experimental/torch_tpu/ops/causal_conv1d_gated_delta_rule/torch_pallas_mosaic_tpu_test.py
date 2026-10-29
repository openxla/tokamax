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
"""Tests for Pallas Mosaic TPU Causal Conv1D Gated Delta Rule in PyTorch."""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.causal_conv1d_gated_delta_rule import base as jax_base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.causal_conv1d_gated_delta_rule import torch_pallas_mosaic_tpu
import torch
from torch._subclasses import fake_tensor
import torch_tpu  # pylint: disable=unused-import


def _skip_if_unsupported(test_case: absltest.TestCase) -> None:
  """Skips `test_case` unless running on a TPU v6 or newer."""
  if jax.default_backend() != "tpu":
    test_case.skipTest("Only supported on TPUs.")
  try:
    if not pltpu.get_tpu_info().generation >= 6:
      test_case.skipTest("Pallas TPU kernel requires TPU v6 or newer.")
  except Exception:  # pylint: disable=broad-except
    test_case.skipTest("Failed to get TPU info.")


def _create_inputs(
    max_reqs: int,
    lengths: list[int],
    q_loc: list[int],
    distribution: list[int],
    *,
    n_kq: int = 2,
    n_v: int = 8,
    d_k: int = 128,
    d_v: int = 128,
    kernel_size: int = 4,
    with_bias: bool = True,
    device: str = "tpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Creates matching PyTorch and JAX inputs for Pallas TPU GDN tests."""
  num_tokens = sum(lengths)
  num_blocks = max_reqs + 1
  conv_dim = 2 * n_kq * d_k + n_v * d_v

  qkv_torch = torch.randn(
      (num_tokens, conv_dim), dtype=torch.float32, device=device
  )
  b_torch = torch.randn((num_tokens, n_v), dtype=torch.float32, device=device)
  a_torch = torch.randn((num_tokens, n_v), dtype=torch.float32, device=device)
  conv_state_torch = torch.zeros(
      (num_blocks, kernel_size - 1, conv_dim),
      dtype=torch.float32,
      device=device,
  )
  recurrent_state_torch = torch.zeros(
      (num_blocks, n_v, d_k, d_v), dtype=torch.float32, device=device
  )
  conv_weight_torch = torch.randn(
      (conv_dim, 1, kernel_size), dtype=torch.float32, device=device
  )
  conv_bias_torch = (
      torch.randn((conv_dim,), dtype=torch.float32, device=device)
      if with_bias
      else None
  )
  a_log_torch = torch.randn((n_v,), dtype=torch.float32, device=device)
  dt_bias_torch = torch.randn((n_v,), dtype=torch.float32, device=device)
  query_start_loc_torch = torch.tensor(q_loc, dtype=torch.int32, device=device)
  state_indices_torch = torch.arange(
      1, max_reqs + 1, dtype=torch.int32, device=device
  )
  distribution_torch = torch.tensor(
      distribution, dtype=torch.int32, device=device
  )
  seq_lens_torch = (
      query_start_loc_torch[1 : max_reqs + 1] - query_start_loc_torch[:max_reqs]
  ).to(torch.int32)

  static_kwargs = dict(
      n_kq=n_kq,
      n_v=n_v,
      d_k=d_k,
      d_v=d_v,
      kernel_size=kernel_size,
  )

  torch_kwargs = dict(
      qkv=qkv_torch,
      b=b_torch,
      a=a_torch,
      conv_state=conv_state_torch,
      recurrent_state=recurrent_state_torch,
      conv_weight=conv_weight_torch,
      conv_bias=conv_bias_torch,
      a_log=a_log_torch,
      dt_bias=dt_bias_torch,
      query_start_loc=query_start_loc_torch,
      state_indices=state_indices_torch,
      distribution=distribution_torch,
      seq_lens=seq_lens_torch,
      **static_kwargs,
  )

  def _to_jax(t: torch.Tensor | None) -> jax.Array | None:
    if t is None:
      return None
    return jnp.asarray(t.detach().cpu().numpy())

  jax_kwargs = {
      k: _to_jax(v) if isinstance(v, torch.Tensor) or v is None else v
      for k, v in torch_kwargs.items()
  }
  return torch_kwargs, jax_kwargs


class PallasMosaicTpuCausalConv1dGatedDeltaRuleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _skip_if_unsupported(self)
    torch.manual_seed(0)

  @parameterized.named_parameters(
      dict(
          testcase_name="prefill",
          max_reqs=1,
          lengths=[8192],
          q_loc=[0, 8192],
          distribution=[0, 0, 3],
          with_bias=True,
      ),
      dict(
          testcase_name="mixed",
          max_reqs=3,
          lengths=[256, 128, 128],
          q_loc=[0, 256, 384, 512],
          distribution=[0, 3, 3],
          with_bias=True,
      ),
      dict(
          testcase_name="decode_only",
          max_reqs=64,
          lengths=[1] * 64,
          q_loc=list(range(65)),
          distribution=[64, 64, 64],
          with_bias=True,
      ),
      dict(
          testcase_name="mixed_prefill_decode",
          max_reqs=11,
          lengths=[1] * 8 + [128, 128, 256],
          q_loc=[0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520],
          distribution=[8, 11, 11],
          with_bias=True,
      ),
      dict(
          testcase_name="padded_mixed_prefill",
          max_reqs=16,
          lengths=[128, 64, 32, 16, 8],
          q_loc=[0, 128, 192, 224, 240, 248] + [1] * 11,
          distribution=[0, 5, 5],
          with_bias=True,
      ),
      dict(
          testcase_name="padded_decode_only",
          max_reqs=512,
          lengths=[1] * 64,
          q_loc=list(range(65)) + [1] * 448,
          distribution=[64, 64, 64],
          with_bias=True,
      ),
      dict(
          testcase_name="no_conv_bias",
          max_reqs=3,
          lengths=[256, 128, 128],
          q_loc=[0, 256, 384, 512],
          distribution=[0, 3, 3],
          with_bias=False,
      ),
  )
  def test_pallas_mosaic_tpu_matches_jax_op(
      self,
      max_reqs: int,
      lengths: list[int],
      q_loc: list[int],
      distribution: list[int],
      with_bias: bool,
  ):
    torch_kwargs, jax_kwargs = _create_inputs(
        max_reqs=max_reqs,
        lengths=lengths,
        q_loc=q_loc,
        distribution=distribution,
        with_bias=with_bias,
    )

    (ref_conv, ref_rec), ref_out = jax_base.CausalConv1dGatedDeltaRule()(
        **jax_kwargs
    )
    jax_pallas_op = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule()
    )
    (jax_pallas_conv, jax_pallas_rec), jax_pallas_out = jax_pallas_op(
        **jax_kwargs
    )

    (actual_conv, actual_rec), actual_out = (
        torch_pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule(
            **torch_kwargs
        )
    )

    expected_pallas_out = torch.as_tensor(
        np.asarray(jax_pallas_out, dtype=np.float32),
        device="tpu",
        dtype=torch.float32,
    )
    expected_pallas_conv = torch.as_tensor(
        np.asarray(jax_pallas_conv, dtype=np.float32),
        device="tpu",
        dtype=torch.float32,
    )
    expected_pallas_rec = torch.as_tensor(
        np.asarray(jax_pallas_rec, dtype=np.float32),
        device="tpu",
        dtype=torch.float32,
    )

    expected_ref_out = torch.as_tensor(
        np.asarray(ref_out, dtype=np.float32),
        device="tpu",
        dtype=torch.float32,
    )
    expected_ref_conv = torch.as_tensor(
        np.asarray(ref_conv, dtype=np.float32),
        device="tpu",
        dtype=torch.float32,
    )
    expected_ref_rec = torch.as_tensor(
        np.asarray(ref_rec, dtype=np.float32),
        device="tpu",
        dtype=torch.float32,
    )

    self.assertEqual(actual_out.shape, expected_pallas_out.shape)
    self.assertEqual(actual_conv.shape, expected_pallas_conv.shape)
    self.assertEqual(actual_rec.shape, expected_pallas_rec.shape)

    # Exact match against the underlying JAX Pallas Mosaic TPU op.
    torch.testing.assert_close(
        actual_out, expected_pallas_out, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        actual_conv, expected_pallas_conv, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        actual_rec, expected_pallas_rec, rtol=1e-5, atol=1e-5
    )

    # Match against the numerical reference within kernel tolerances.
    torch.testing.assert_close(
        actual_out, expected_ref_out, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_conv, expected_ref_conv, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_rec, expected_ref_rec, rtol=2e-2, atol=2e-2
    )

  def test_has_initial_state_zeros_stale_slot(self):
    torch_kwargs, _ = _create_inputs(
        max_reqs=2,
        lengths=[64, 64],
        q_loc=[0, 64, 128],
        distribution=[0, 2, 2],
    )
    op = torch_pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule

    (new_conv_fresh, new_rec_fresh), out_fresh = op(**torch_kwargs)

    stale_conv = torch.randn_like(torch_kwargs["conv_state"])
    stale_conv[0].zero_()
    stale_rec = torch.randn_like(torch_kwargs["recurrent_state"])
    stale_rec[0].zero_()

    stale_kwargs = dict(
        torch_kwargs,
        conv_state=stale_conv,
        recurrent_state=stale_rec,
    )
    (new_conv_stale, new_rec_stale), out_stale = op(**stale_kwargs)

    torch.testing.assert_close(out_fresh, out_stale, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        new_conv_fresh[1:3], new_conv_stale[1:3], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        new_rec_fresh[1:3], new_rec_stale[1:3], rtol=1e-5, atol=1e-5
    )

  def test_has_initial_state_preserves_continuation(self):
    torch_kwargs, _ = _create_inputs(
        max_reqs=1,
        lengths=[64],
        q_loc=[0, 64],
        distribution=[0, 1, 1],
    )
    half = 32
    full = 64
    device = "tpu"
    op = torch_pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule

    (_, _), out_ref = op(**torch_kwargs)

    step_a_kwargs = dict(
        torch_kwargs,
        qkv=torch_kwargs["qkv"][:half],
        b=torch_kwargs["b"][:half],
        a=torch_kwargs["a"][:half],
        query_start_loc=torch.tensor(
            [0, half], dtype=torch.int32, device=device
        ),
        seq_lens=torch.tensor([half], dtype=torch.int32, device=device),
    )
    (conv_after_a, rec_after_a), out_a = op(**step_a_kwargs)

    step_b_kwargs = dict(
        torch_kwargs,
        qkv=torch_kwargs["qkv"][half:],
        b=torch_kwargs["b"][half:],
        a=torch_kwargs["a"][half:],
        conv_state=conv_after_a,
        recurrent_state=rec_after_a,
        query_start_loc=torch.tensor(
            [0, half], dtype=torch.int32, device=device
        ),
        seq_lens=torch.tensor([full], dtype=torch.int32, device=device),
    )
    (_, _), out_b = op(**step_b_kwargs)

    torch.testing.assert_close(out_a, out_ref[:half], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_b, out_ref[half:], rtol=2e-2, atol=2e-2)

  @parameterized.named_parameters(
      dict(
          testcase_name="mixed",
          max_reqs=3,
          lengths=[256, 128, 128],
          q_loc=[0, 256, 384, 512],
          distribution=[0, 3, 3],
      ),
      dict(
          testcase_name="decode_only",
          max_reqs=64,
          lengths=[1] * 64,
          q_loc=list(range(65)),
          distribution=[64, 64, 64],
      ),
  )
  def test_torch_compile(
      self,
      max_reqs: int,
      lengths: list[int],
      q_loc: list[int],
      distribution: list[int],
  ):
    torch_kwargs, _ = _create_inputs(
        max_reqs=max_reqs,
        lengths=lengths,
        q_loc=q_loc,
        distribution=distribution,
    )
    op = torch_pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule

    fwd_config, bwd_config = torch_utils.get_configs(
        op,
        torch_kwargs["qkv"],
        torch_kwargs["b"],
        torch_kwargs["a"],
        torch_kwargs["conv_state"],
        torch_kwargs["recurrent_state"],
        torch_kwargs["conv_weight"],
        torch_kwargs["conv_bias"],
        torch_kwargs["a_log"],
        torch_kwargs["dt_bias"],
        torch_kwargs["query_start_loc"],
        torch_kwargs["state_indices"],
        torch_kwargs["distribution"],
        torch_kwargs["seq_lens"],
        n_kq=torch_kwargs["n_kq"],
        n_v=torch_kwargs["n_v"],
        d_k=torch_kwargs["d_k"],
        d_v=torch_kwargs["d_v"],
        kernel_size=torch_kwargs["kernel_size"],
        from_autotuning_cache=False,
    )
    self.assertIsNotNone(fwd_config)
    self.assertIsNone(bwd_config)

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(
        qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      qkv = qkv + 0.5
      (new_conv, new_rec), out = op(
          qkv=qkv,
          b=b,
          a=a,
          conv_state=conv_state,
          recurrent_state=recurrent_state,
          conv_weight=conv_weight,
          conv_bias=conv_bias,
          a_log=a_log,
          dt_bias=dt_bias,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          distribution=distribution,
          seq_lens=seq_lens,
          n_kq=torch_kwargs["n_kq"],
          n_v=torch_kwargs["n_v"],
          d_k=torch_kwargs["d_k"],
          d_v=torch_kwargs["d_v"],
          kernel_size=torch_kwargs["kernel_size"],
      )
      return new_conv, new_rec, out + 1.0

    actual_conv, actual_rec, actual_out = compiled_fn(
        torch_kwargs["qkv"],
        torch_kwargs["b"],
        torch_kwargs["a"],
        torch_kwargs["conv_state"],
        torch_kwargs["recurrent_state"],
        torch_kwargs["conv_weight"],
        torch_kwargs["conv_bias"],
        torch_kwargs["a_log"],
        torch_kwargs["dt_bias"],
        torch_kwargs["query_start_loc"],
        torch_kwargs["state_indices"],
        torch_kwargs["distribution"],
        torch_kwargs["seq_lens"],
    )

    eager_kwargs = dict(torch_kwargs, qkv=torch_kwargs["qkv"] + 0.5)
    (expected_conv, expected_rec), expected_out = op(
        **eager_kwargs, config=fwd_config
    )
    expected_out = expected_out + 1.0

    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_conv, expected_conv, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_rec, expected_rec, rtol=1e-5, atol=1e-5)

  def test_fake_impl_and_export_symbolic_shapes(self):
    n_kq, n_v, d_k, d_v, kernel_size = 2, 8, 128, 128, 4
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    num_blocks = 3
    op = torch_pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule

    with fake_tensor.FakeTensorMode():
      qkv = torch.empty((64, conv_dim), dtype=torch.float32)
      b = torch.empty((64, n_v), dtype=torch.float32)
      a = torch.empty((64, n_v), dtype=torch.float32)
      conv_state = torch.empty(
          (num_blocks, kernel_size - 1, conv_dim), dtype=torch.float32
      )
      recurrent_state = torch.empty(
          (num_blocks, n_v, d_k, d_v), dtype=torch.float32
      )
      conv_weight = torch.empty((conv_dim, 1, kernel_size), dtype=torch.float32)
      conv_bias = torch.empty((conv_dim,), dtype=torch.float32)
      a_log = torch.empty((n_v,), dtype=torch.float32)
      dt_bias = torch.empty((n_v,), dtype=torch.float32)
      query_start_loc = torch.empty((3,), dtype=torch.int32)
      state_indices = torch.empty((2,), dtype=torch.int32)
      distribution = torch.empty((3,), dtype=torch.int32)
      seq_lens = torch.empty((2,), dtype=torch.int32)

      (new_conv, new_rec), out = op(
          qkv=qkv,
          b=b,
          a=a,
          conv_state=conv_state,
          recurrent_state=recurrent_state,
          conv_weight=conv_weight,
          conv_bias=conv_bias,
          a_log=a_log,
          dt_bias=dt_bias,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          distribution=distribution,
          seq_lens=seq_lens,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          kernel_size=kernel_size,
      )

    self.assertEqual(new_conv.shape, (num_blocks, kernel_size - 1, conv_dim))
    self.assertEqual(new_rec.shape, (num_blocks, n_v, d_k, d_v))
    self.assertEqual(out.shape, (64, n_v * d_v))


if __name__ == "__main__":
  absltest.main()
