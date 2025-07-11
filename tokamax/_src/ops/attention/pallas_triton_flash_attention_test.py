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

from typing import TypeAlias
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
from tokamax._src import precision as precision_utils
from tokamax._src import quantization
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_triton_flash_attention as flash_attn
from tokamax._src.ops.attention import test_base


class _F32PrecisionXlaAttentionVjp(base.DotProductAttentionVjp):

  def __call__(self, *args, precision, logits_dtype, **kwargs):  # pylint: disable=useless-parent-delegation
    return super().__call__(
        *args,
        precision=(jax.lax.DotAlgorithmPreset.F32_F32_F32,) * 2,
        logits_dtype=jnp.dtype(jnp.float32),
        **kwargs,
    )


class PallasTritonFlashAttentionTest(test_base.AttentionTestBase):

  def __init__(self, *args):
    vjp = _F32PrecisionXlaAttentionVjp()
    super().__init__(
        *args, attention_fn=flash_attn.PallasTritonFlashAttention(vjp=vjp)
    )

  def _run_test(
      self,
      q_shape,
      *args,
      **kwargs,
  ):
    if q_shape[1] >= 32768:
      self.skipTest("Triton seems to fail for so long sequences (b/384038935)")

    super()._run_test(q_shape, *args, **kwargs)

  def test_causal_mask_different_block_sizes(self):
    assert isinstance(self._attention_fn, flash_attn.PallasTritonFlashAttention)
    config = flash_attn.Config(
        block_q=128,
        block_k=32,
        num_warps=4,
        num_stages=1,
    )

    mask = nn.make_causal_mask(jax.ShapeDtypeStruct((2, 1024), jnp.float32))
    self._run_test(
        (2, 1024, 4, 64),
        impl=self._attention_fn.with_config(config),
        impl_kwargs=dict(is_causal=True),
        ref_kwargs=dict(mask=mask),
    )

  @parameterized.parameters(*test_base.base_names_and_params("test_mask_api"))
  def test_mask_api(self, test_name, kwargs):
    kwargs = eval(f"dict{kwargs}")  # pylint: disable=eval-used
    if (k_start := kwargs.get("k_start")) and kwargs.get("is_causal"):
      k_start = jnp.array(k_start if isinstance(k_start, range) else [k_start])
      if jnp.any(k_start > jnp.arange(1, 1024 + 1)):
        self.skipTest("k_start > causality diagonal currently unsupported.")

    materialized_mask = not (
        isinstance(kwargs.get("q_start", -1), int)
        and isinstance(kwargs.get("q_end", -1), int)
    )

    orig_fwd = flash_attn._fwd

    def my_fwd(q, k, v, bias, mask, *args, **kwargs):
      self.assertEqual(mask is not None, materialized_mask)
      return orig_fwd(q, k, v, bias, mask, *args, **kwargs)

    with mock.patch.object(flash_attn, "_fwd", my_fwd):
      getattr(super(), test_name)()

  @parameterized.parameters(*precision_utils.SUPPORTED_PRECISIONS)
  def test_precision(self, precision):
    if precision == jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3:
      self.skipTest("TF32_F32_3X not supported")
    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3:
      self.skipTest("BF16_BF16_F32_X3 not supported")
    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6:
      self.skipTest("BF16_BF16_F32_X6 not supported")

    dtype = precision_utils.precision_input_dtype(precision)

    atol = {
        jax.lax.DotAlgorithmPreset.F32_F32_F32: 3e-6,
        jax.lax.DotAlgorithmPreset.TF32_TF32_F32: 0.0023,
        jax.lax.DotAlgorithmPreset.F16_F16_F32: 0.002,
        jax.lax.DotAlgorithmPreset.BF16_BF16_F32: 0.016,
    }.get(precision, 0.0)
    atol_grad = {jax.lax.DotAlgorithmPreset.F16_F16_F32: 0.006}.get(precision)

    self._run_test(
        (2, 256, 2, 64),
        dtype=dtype,
        impl_kwargs=dict(precision=precision),
        bias_shape=(2, 2, 256, 256),
        atol=atol,
        atol_grads=atol_grad,
        expect_supported=_is_precision_supported(precision),
    )

  @parameterized.parameters(
      *test_base.base_names_and_params("test_quantized_int8")
  )
  def test_quantized_int8(self, test_name, kwargs):
    kwargs = eval(f"dict{kwargs}")  # pylint: disable=eval-used
    tile_shape, quantize_q = kwargs["tile_shape"], kwargs["quantize_q"]
    expected_int8_dots = int(quantize_q and tile_shape[-1] == -1)
    actual_int8_dots = 0

    orig_dot_general = jax.lax.dot_general

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

    with mock.patch.object(jax.lax, "dot_general", my_dot_general):
      getattr(super(), test_name)()
      self.assertEqual(expected_int8_dots, actual_int8_dots)

  def test_block_d(self):
    """Tests `block_d != None` with quantization and different head_dim_out."""
    assert isinstance(self._attention_fn, flash_attn.PallasTritonFlashAttention)
    quantize = quantization.quantize_as(jnp.int8, tile_shape=(1, 1, 1, -1))
    config = flash_attn.Config(
        block_q=64,
        block_k=64,
        block_d=64,
        block_d_out=32,
        num_warps=4,
        num_stages=1,
    )

    def impl(q, k, v, **kwargs):
      k, v = map(quantize, (k, v))
      return self._attention_fn.with_config(config)(q, k, v, **kwargs)

    def ref_impl(q, k, v, **kwargs):
      k, v = map(lambda x: quantize(x).recompose(), (k, v))
      return nn.dot_product_attention(q, k, v, **kwargs)

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    head_dims = (512, 512, 256)
    x = [jax.random.normal(k, (2, 256, 2, d)) for k, d in zip(keys, head_dims)]
    self._run_test_with_inputs(*x, impl=impl, ref_impl=ref_impl, test_vjp=False)

  @parameterized.parameters(1, 2, 4, 8)
  def test_small_block_q(self, block_q: int):
    Config: TypeAlias = flash_attn.Config
    config = Config(block_q=block_q, block_k=64, num_warps=4, num_stages=2)
    assert isinstance(self._attention_fn, flash_attn.PallasTritonFlashAttention)
    self._run_test((2, 256, 2, 64), impl=self._attention_fn.with_config(config))

  @parameterized.parameters(2, 3, 4)
  def test_split_k(self, split_k):
    quantize = quantization.quantize_as(jnp.int8, tile_shape=(1, 1, 1, -1))
    quant_dequant = lambda x: quantize(x).recompose()
    assert isinstance(self._attention_fn, flash_attn.PallasTritonFlashAttention)

    def impl(q, k, v, **kwargs):
      k = quantize(k)
      v = quantize(v)
      config = flash_attn.Config(
          block_q=64, block_k=64, num_warps=4, num_stages=2, split_k=split_k
      )
      return self._attention_fn.with_config(config)(q, k, v, **kwargs)

    def ref_impl(q, k, v, **kwargs):
      k = quant_dequant(k)
      v = quant_dequant(v)
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


class PallasTritonFlashAttentionWithPallasTritonVjpTest(
    test_base.AttentionTestBase
):

  def __init__(self, *args):
    super().__init__(
        *args, attention_fn=flash_attn.PallasTritonFlashAttention()
    )

  def _run_test_with_inputs(self, *args, expect_supported=True, **kwargs):
    if kwargs.get("test_vjp", True):
      # TODO: Add missing features to Pallas-Triton VJP.
      all_kwargs = kwargs | kwargs.get("impl_kwargs", {})
      if (all_kwargs.get("dropout_mask") is not None) or (
          all_kwargs.get("logits_soft_cap") is not None
      ):
        expect_supported = False
    super()._run_test_with_inputs(
        *args, expect_supported=expect_supported, **kwargs
    )

  @parameterized.parameters(*test_base.base_names_and_params("test_mask_api"))
  def test_mask_api(self, test_name, kwargs):
    # TODO: Work out why these tests fail.
    if test_name in ("test_mask_api14", "test_mask_api16"):
      self.skipTest("`q_start` can cause NaNs in VJP.")

    kwargs = eval(f"dict{kwargs}")  # pylint: disable=eval-used
    if (k_start := kwargs.get("k_start")) and kwargs.get("is_causal"):
      k_start = jnp.array(k_start if isinstance(k_start, range) else [k_start])
      if jnp.any(k_start > jnp.arange(1, 1024 + 1)):
        self.skipTest("k_start > causality diagonal currently unsupported.")
    getattr(super(), test_name)()

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


class PallasTritonFlashAttentionManualPartitioningTest(
    test_base.AttentionManualPartitioningTestBase
):

  def __init__(self, *args):
    super().__init__(
        *args,
        attention_fn=flash_attn.PallasTritonFlashAttention(),
        supports_vjp=False,  # TODO: Add vjp support and test
    )


if __name__ == "__main__":
  absltest.main()
