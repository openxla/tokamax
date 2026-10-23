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

import sys
from typing import override
from unittest import mock
import pytest
from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import precision as precision_utils
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import test_base

try:
  import jax_triton as jt  # pylint: disable=g-import-not-at-top,unused-import  # pyrefly: ignore[missing-import]
  from tokamax._src.ops.attention import triton as fa  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
  from tokamax._src.ops.attention import triton_vjp as fa_vjp  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
except ImportError:
  jt = None  # pyrefly: ignore[assignment]
  fa = None  # pyrefly: ignore[assignment]
  fa_vjp = None  # pyrefly: ignore[assignment]


class _F32PrecisionXlaAttentionVjp(base.DotProductAttentionVjp):

  def __call__(self, *args, precision, logits_dtype, **kwargs):  # pylint: disable=useless-parent-delegation
    return super().__call__(
        *args,
        precision=(jax.lax.DotAlgorithmPreset.F32_F32_F32,) * 2,
        logits_dtype=jnp.dtype(jnp.float32),
        **kwargs,
    )


@absltest.skipIf(fa is None, "Requires jax_triton.")
class TritonFlashAttentionTest(test_base.AttentionTestBase):

  def __init__(self, *args):
    vjp = _F32PrecisionXlaAttentionVjp()
    attn_fn = fa.TritonFlashAttention(vjp=vjp) if fa is not None else None
    super().__init__(*args, attention_fn=attn_fn)  # pyrefly: ignore[bad-argument-type]

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def _run_test(self, q_shape, *args, **kwargs):
    if q_shape[1] >= 32768:
      self.skipTest("Triton seems to fail for so long sequences (b/384038935)")

    super()._run_test(q_shape, *args, **kwargs)

  @override
  @absltest.skip("Not implemented yet.")
  def _test_quantized_int8(self, channelwise_axes, quantize_q):
    expected_int8_dots = int(quantize_q and 3 not in channelwise_axes)
    actual_int8_dots = 0

    lax = sys.modules["jax._src.lax"]
    orig_dot_general = lax.dot_general

    def my_dot_general(
        lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None
    ):
      nonlocal actual_int8_dots
      a_dt, b_dt, acc_dt = lhs.dtype, rhs.dtype, preferred_element_type
      is_int8 = a_dt == jnp.int8 and b_dt == jnp.int8 and acc_dt == jnp.int32
      actual_int8_dots += int(is_int8)
      return orig_dot_general(
          lhs, rhs, dimension_numbers, precision, preferred_element_type
      )

    with mock.patch.object(lax, "dot_general", my_dot_general):
      super()._test_quantized_int8(channelwise_axes, quantize_q)

    self.assertEqual(expected_int8_dots, actual_int8_dots)

  @override
  @absltest.skip("Not implemented yet.")
  def _test_quantized_int4(self, subchannel_size):
    super()._test_quantized_int4(subchannel_size)

  @absltest.skip("Not implemented yet.")
  def test_block_d(self):
    """Tests `block_d != None` with quantization and different head_dim_out."""
    assert fa is not None
    assert isinstance(self._attention_fn, fa.TritonFlashAttention)
    quantize = lambda x: qwix.quantize(x, jnp.int8, channelwise_axes=(0, 1, 2))
    config = fa.Config(
        block_q=64,
        block_k=64,
        block_d=64,
        block_d_out=32,
        num_warps=4,
        num_stages=1,
    )

    def impl(q, k, v, **kwargs):
      k, v = map(quantize, (k, v))
      return self._attention_fn.replace(config=config)(q, k, v, **kwargs)

    def ref_impl(q, k, v, **kwargs):
      k, v = map(lambda x: qwix.dequantize(quantize(x)), (k, v))
      return nn.dot_product_attention(q, k, v, **kwargs)

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    head_dims = (512, 512, 256)
    x = [jax.random.normal(k, (2, 256, 2, d)) for k, d in zip(keys, head_dims)]
    self._run_test_with_inputs(*x, impl=impl, ref_impl=ref_impl, test_vjp=False)

  def test_block_d_unquantized(self):
    assert fa is not None
    assert isinstance(self._attention_fn, fa.TritonFlashAttention)
    config = fa.Config(
        block_q=64,
        block_k=64,
        block_d=64,
        block_d_out=32,
        num_warps=4,
        num_stages=1,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    head_dims = (512, 512, 256)
    x = [jax.random.normal(k, (2, 256, 2, d)) for k, d in zip(keys, head_dims)]
    self._run_test_with_inputs(
        *x,
        impl=self._attention_fn.replace(config=config),
        atol=5e-6,
        test_vjp=False,
    )

  @parameterized.parameters(1, 2)
  def test_small_block_q(self, block_q: int):
    assert fa is not None
    Config = fa.Config
    config = Config(block_q=block_q, block_k=64, num_warps=4, num_stages=2)
    assert isinstance(self._attention_fn, fa.TritonFlashAttention)
    impl = self._attention_fn.replace(config=config)
    self._run_test((2, 256, 2, 64), impl=impl)

  def test_k_range_non_divisible_seq_len_q(self):
    assert fa is not None
    assert isinstance(self._attention_fn, fa.TritonFlashAttention)
    config = fa.Config(block_q=64, block_k=64, num_warps=4, num_stages=2)
    seq_len = 1000
    q_idx = jnp.arange(seq_len)
    mask = base.Mask(
        k_start=(jnp.minimum(q_idx, 300) // 2)[None],
        k_end=jnp.minimum(q_idx + 1, 500)[None],
    )
    self._run_test(
        (2, seq_len, 4, 64),
        impl=self._attention_fn.replace(config=config),
        impl_kwargs=dict(mask=mask),
        ref_kwargs=dict(mask=mask.as_array(seq_len, seq_len)),
        atol_grads=2e-5,
    )

  def test_k_end_near_int32_max(self):
    seq_len_q, seq_len_k = 64, 128
    # Mixes a small and an "unbounded" `k_end` within one query tile.
    k_end = jnp.where(
        jnp.arange(seq_len_q) % 2 == 0, 37, jnp.iinfo(jnp.int32).max
    )
    mask = base.Mask(k_end=k_end)
    self._run_test(
        (2, seq_len_q, 2, 64),
        kv_shape=(2, seq_len_k, 2, 64),
        impl_kwargs=dict(mask=mask),
        ref_kwargs=dict(mask=mask.as_array(seq_len_q, seq_len_k)),
        test_vjp=False,
    )

  def test_neg_inf_bias(self):
    assert fa is not None
    assert isinstance(self._attention_fn, fa.TritonFlashAttention)
    config = fa.Config(block_q=64, block_k=64, num_warps=4, num_stages=2)
    seq_len = 256
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q, k, v = (jax.random.normal(key, (2, seq_len, 2, 64)) for key in keys)
    # Odd rows see only `-inf` logits in the first k blocks they visit.
    row, col = jnp.arange(seq_len)[:, None], jnp.arange(seq_len)[None, :]
    bias = jnp.where((row % 2 == 1) & (col < 128), float("-inf"), 0.0)
    self._run_test_with_inputs(
        q,
        k,
        v,
        bias=bias[None, None],
        impl=self._attention_fn.replace(config=config),
        test_vjp=False,
    )

  @parameterized.parameters(1, 2)
  def test_use_base2(self, split_k):
    assert fa is not None
    assert isinstance(self._attention_fn, fa.TritonFlashAttention)
    config = fa.Config(
        block_q=64, block_k=64, num_warps=4, num_stages=2, split_k=split_k
    )
    impl = self._attention_fn.replace(use_base2=True, config=config)
    self._run_test((2, 256, 2, 64), impl=impl)

  def test_tanh_clipping_float16(self):
    self._run_test(
        (2, 256, 2, 64),
        dtype=jnp.float16,
        ref_impl=base.DotProductAttention(),
        logits_soft_cap=30.0,
        # Unscaled logits push unshifted `exp(s)` past the float16 range.
        logits_scale=1.0,
        atol=1e-2,
        test_vjp=False,
    )

  @parameterized.parameters(2, 3, 4)
  @absltest.skip("Not implemented yet.")
  def test_split_k(self, split_k):
    self.skipTest("Too slow for OSS")
    assert fa is not None
    quantize = lambda x: qwix.quantize(x, jnp.int8, channelwise_axes=(0, 1, 2))
    assert isinstance(self._attention_fn, fa.TritonFlashAttention)

    def impl(q, k, v, **kwargs):
      k, v = map(quantize, (k, v))
      config = fa.Config(
          block_q=64, block_k=64, num_warps=4, num_stages=2, split_k=split_k
      )
      return self._attention_fn.replace(config=config)(q, k, v, **kwargs)

    def ref_impl(q, k, v, **kwargs):
      k, v = map(lambda x: qwix.dequantize(quantize(x)), (k, v))
      return nn.dot_product_attention(q, k, v, **kwargs)

    shape = (2, 1024, 4, 64)
    mask_shape = (1, 4, 1024, 1)
    self._run_test(
        shape,
        mask_shape=mask_shape,
        impl=impl,
        ref_impl=ref_impl,
        expect_supported=shape[-3] % split_k == 0,
        test_vjp=False,
    )

  @parameterized.parameters(2, 4)
  def test_split_k_k_range(self, split_k):
    assert fa is not None
    assert isinstance(self._attention_fn, fa.TritonFlashAttention)
    config = fa.Config(
        block_q=64, block_k=64, num_warps=4, num_stages=2, split_k=split_k
    )
    seq_len = 512
    q_idx = jnp.arange(seq_len)
    # `is_causal` is folded into `k_end`, so it is allowed with `split_k`.
    mask = base.Mask(
        k_start=jnp.maximum(q_idx - 300, 0)[None],
        k_end=jnp.minimum(q_idx + 200, seq_len)[None],
        is_causal=True,
    )
    self._run_test(
        (2, seq_len, 2, 64),
        impl=self._attention_fn.replace(config=config),
        impl_kwargs=dict(mask=mask),
        ref_kwargs=dict(mask=mask.as_array(seq_len, seq_len)),
        test_vjp=False,
    )


@pytest.mark.skip(reason="Too slow for OSS regression tests.")
@absltest.skipIf(fa is None, "Requires jax_triton.")
class TritonFlashAttentionWithTritonVjpTest(test_base.AttentionTestBase):

  def __init__(self, *args):
    if fa is None or fa_vjp is None:
      attn_fn = None
    else:
      vjp = fa_vjp.TritonFlashAttentionVjp(dbias_intermediate_dtype=jnp.float32)
      attn_fn = fa.TritonFlashAttention(vjp=vjp)
    super().__init__(*args, attention_fn=attn_fn)  # pyrefly: ignore[bad-argument-type]

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def _run_test_with_inputs(self, *args, expect_supported=True, **kwargs):
    if kwargs.get("test_vjp", True):
      # TODO: Add missing features to Triton VJP.
      all_kwargs = kwargs | kwargs.get("impl_kwargs", {})
      if (all_kwargs.get("dropout_mask") is not None) or (
          all_kwargs.get("logits_soft_cap") is not None
      ):
        expect_supported = False
    super()._run_test_with_inputs(
        *args, expect_supported=expect_supported, **kwargs
    )

  def test_normalize_output(self):
    self.skipTest("`normalize_output=False` not supported.")


def _is_precision_supported(precision: jax.lax.DotAlgorithmPreset) -> bool:
  return precision in {
      jax.lax.DotAlgorithmPreset.F32_F32_F32,
      jax.lax.DotAlgorithmPreset.TF32_TF32_F32,
      jax.lax.DotAlgorithmPreset.F16_F16_F32,
      jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
      jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3,
  }

if __name__ == "__main__":
  absltest.main()
