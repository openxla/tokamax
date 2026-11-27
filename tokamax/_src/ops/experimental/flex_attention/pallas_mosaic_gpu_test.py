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
"""Tests for Pallas Mosaic GPU FlexAttention."""

from typing import override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as attn_common
from tokamax._src.ops.attention import test_base as attn_test_base
from tokamax._src.ops.experimental.flex_attention import pallas_mosaic_gpu
from tokamax._src.ops.experimental.flex_attention import test_base
from tokamax._src.ops.experimental.flex_attention import wrapper_test_base


def _bf16_round_trip(x):
  if x.dtype == jnp.float32:
    return attn_common.safe_downcast(x, jnp.bfloat16).astype(jnp.float32)
  return x


class PallasMosaicGpuFlexAttentionTest(
    test_base.FlexAttentionTestBase, parameterized.TestCase
):
  """Tests for PallasMosaicGpuFlexAttention."""

  def __init__(self, *args):
    super().__init__(
        *args, flex_attn=pallas_mosaic_gpu.PallasMosaicGpuFlexAttention()
    )

  def setUp(self):
    if not gpu_utils.is_sm90():
      self.skipTest("Only supported on SM90 GPUs.")
    super().setUp()

  def _run_bf16_test(self, test_fn):
    orig_flex_attn = self._flex_attn
    orig_assert_close = chex.assert_trees_all_close
    orig_normal = jax.random.normal

    def normal_bf16(key, shape=(), dtype=jnp.bfloat16, *args, **kwargs):
      del dtype
      return orig_normal(key, shape, jnp.bfloat16, *args, **kwargs).astype(
          jnp.float32
      )

    def flex_attn(q, k, v, **kwargs):
      if kwargs.get("precision") == "highest":
        kwargs["precision"] = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
      return orig_flex_attn(q, k, v, **kwargs)

    def assert_close(*args, atol=0.0, **kwargs):
      return orig_assert_close(*args, atol=max(atol, 0.02), **kwargs)

    with (
        mock.patch.object(jax.random, "normal", normal_bf16),
        mock.patch.object(self, "_flex_attn", flex_attn),
        mock.patch.object(chex, "assert_trees_all_close", assert_close),
    ):
      test_fn()

  @override
  def test_bias_upscale(self):
    self._run_bf16_test(super().test_bias_upscale)

  @override
  def test_bias_relative_positional(self):
    self._run_bf16_test(super().test_bias_relative_positional)

  @parameterized.product(
      use_base2=(True, False),
      use_stable_softmax=(True, False),
      rescale_threshold=(1.0, 0.5),
  )
  def test_op_parameters(
      self, use_base2, use_stable_softmax, rescale_threshold
  ):
    impl = pallas_mosaic_gpu.PallasMosaicGpuFlexAttention(
        use_base2=use_base2,
        use_stable_softmax=use_stable_softmax,
        rescale_threshold=rescale_threshold,
    )
    with mock.patch.object(self, "_flex_attn", impl):
      self.test_bias_upscale()
      if use_stable_softmax:
        self.test_bias_relative_positional()

  def test_autotune_configs(self):
    q, k, v, *_ = attn_test_base._create_inputs(
        q_shape=(2, 384, 4, 128), dtype=jnp.bfloat16
    )
    bound_args = self._flex_attn.bind(q, k, v)
    configs = self._flex_attn._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)
    expected = jax.nn.dot_product_attention(q, k, v)
    score_mod = lambda s: s / jnp.sqrt(q.shape[-1])
    for config in configs:
      with self.subTest(f"{config=}"):
        impl = pallas_mosaic_gpu.PallasMosaicGpuFlexAttention(config=config)
        actual = impl(q, k, v, score_mod=score_mod)
        chex.assert_trees_all_close(actual, expected, atol=0.01)

  @parameterized.named_parameters(
      dict(
          testcase_name="score_mod_merged_dims",
          score_mod=lambda s: s
          + (
              jnp.arange(4)[None, None, None, :, None] * (s.shape[-1] // 4)
              + jnp.arange(s.shape[-1] // 4)[None, None, None, None, :]
          )
          .reshape(1, 1, 1, s.shape[-1])
          .astype(s.dtype),
          mask_mod=None,
      ),
      dict(
          testcase_name="mask_mod_merged_dims",
          score_mod=None,
          mask_mod=lambda shape: (
              jnp.arange(shape[-2])[None, None, :, None, None]
              >= (
                  jnp.arange(4)[None, None, None, :, None] * (shape[-1] // 4)
                  + jnp.arange(shape[-1] // 4)[None, None, None, None, :]
              )
          ).reshape(1, 1, shape[-2], shape[-1]),
      ),
      dict(
          testcase_name="score_mod_split_dims",
          score_mod=lambda s: s
          + jnp.arange(s.shape[-2] * s.shape[-1], dtype=s.dtype)[
              None, None, :
          ].reshape(1, 1, s.shape[-2], s.shape[-1]),
          mask_mod=None,
      ),
  )
  def test_unsupported_reshape_raises_not_implemented_error(
      self, score_mod, mask_mod
  ):
    q, k, v, *_ = attn_test_base._create_inputs(
        q_shape=(1, 128, 4, 64), dtype=jnp.bfloat16
    )
    with self.assertRaisesRegex(NotImplementedError, "Unsupported reshape"):
      self._flex_attn(q, k, v, score_mod=score_mod, mask_mod=mask_mod)


class WrappedPallasMosaicGpuFlexAttentionTest(
    wrapper_test_base.WrappedFlexAttentionTestBase
):
  """Wrapped DotProductAttention tests for PallasMosaicGpuFlexAttention."""

  def __init__(self, *args):
    super().__init__(
        *args,
        flex_attn=pallas_mosaic_gpu.PallasMosaicGpuFlexAttention(),
        supports_vjp=False,
        supports_precisions=False,
    )

  def setUp(self):
    if not gpu_utils.is_sm90():
      self.skipTest("Only supported on SM90 GPUs.")
    super().setUp()

  @override
  def _run_test_with_inputs(self, q, k, v, *, bias=None, **kwargs):
    impl_kwargs = kwargs.setdefault("impl_kwargs", {})
    if any(x.dtype == jnp.float32 for x in (q, k, v)):
      precision = impl_kwargs.setdefault(
          "precision", jax.lax.DotAlgorithmPreset.BF16_BF16_F32
      )
      if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
        q, k, v = map(_bf16_round_trip, (q, k, v))

    atol = kwargs.get("atol", 0.0)
    kwargs["atol"] = max(atol, 0.0045)
    super()._run_test_with_inputs(q, k, v, bias=bias, **kwargs)

  @override
  def test_causal_mask(self):
    with attn_test_base.override_test_args(atol={1.0: 0.008, 0.99: 0.006}):
      super().test_causal_mask()

  def test_causal_mask_cross_attention0(self):
    with attn_test_base.override_test_args(atol={1.0: 0.008, 0.99: 0.006}):
      super().test_causal_mask_cross_attention0()  # pyrefly: ignore[missing-attribute]

  def test_causal_mask_cross_attention1(self):
    with attn_test_base.override_test_args(atol={1.0: 0.008, 0.99: 0.006}):
      super().test_causal_mask_cross_attention1()  # pyrefly: ignore[missing-attribute]

  @override
  def test_causal_mask_q_indices(self):
    with attn_test_base.override_test_args(atol=0.007):
      super().test_causal_mask_q_indices()

  @override
  def test_causal_mask_k_indices(self):
    with attn_test_base.override_test_args(atol={1.0: 0.007, 0.99: 0.0045}):
      super().test_causal_mask_k_indices()

  @override
  def _test_mask_api(self, **kwargs):
    with attn_test_base.override_test_args(atol=0.015):
      super()._test_mask_api(**kwargs)

  @override
  def test_local_attention_mask(self):
    with attn_test_base.override_test_args(atol={1.0: 0.008, 0.99: 0.006}):
      super().test_local_attention_mask()

  @override
  def test_padding_mask_with_nans(self):
    self.skipTest("TODO: Fix.")

  @override
  def test_normalize_output(self):
    with attn_test_base.override_test_args(atol=0.02):
      super().test_normalize_output()

  @override
  def _test_quantized_int8(self, channelwise_axes, quantize_q):
    with attn_test_base.override_test_args(atol=0.02):
      super()._test_quantized_int8(channelwise_axes, quantize_q)

  @override
  def _test_quantized_int4(self, subchannel_size):
    with attn_test_base.override_test_args(atol=0.05):
      super()._test_quantized_int4(subchannel_size)

  @override
  def _test_small_sequences(self, seq_q, seq_kv):
    with attn_test_base.override_test_args(atol=0.02):
      super()._test_small_sequences(seq_q, seq_kv)

  @override
  def _test_bench(self, spec):
    try:
      with attn_test_base.override_test_args(atol={0.99: 0.02, 1.0: 0.05}):
        super()._test_bench(spec)
    except ValueError as e:
      if "exceeds available shared memory" in str(e):
        self.skipTest(f"Test exceeds shared memory capacity: {e}")
      raise


if __name__ == "__main__":
  absltest.main()
