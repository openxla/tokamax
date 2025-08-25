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

import dataclasses
from typing import TypeAlias
from unittest import mock
import pytest
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

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def _run_test(
      self,
      q_shape,
      *args,
      **kwargs,
  ):
    if q_shape[1] >= 32768:
      self.skipTest("Triton seems to fail for so long sequences (b/384038935)")

    super()._run_test(q_shape, *args, **kwargs)

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

  @parameterized.parameters(1, 2)
  def test_small_block_q(self, block_q: int):
    Config: TypeAlias = flash_attn.Config
    config = Config(block_q=block_q, block_k=64, num_warps=4, num_stages=2)
    assert isinstance(self._attention_fn, flash_attn.PallasTritonFlashAttention)
    self._run_test((2, 256, 2, 64), impl=self._attention_fn.with_config(config))

  @parameterized.parameters(2, 3, 4)
  def test_split_k(self, split_k):
    self.skipTest("Too slow for OSS")
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

@pytest.mark.skip(reason="Too slow for OSS regression tests.")
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
