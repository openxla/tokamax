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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import pytest
from tokamax._src.ops.attention import pallas_mosaic_gpu_flash_attention as fa
from tokamax._src.ops.attention import test_base


@pytest.mark.skip(reason="Too slow for OSS regression tests.")
class PallasMosaicGpuFlashAttentionTest(test_base.AttentionTestBase):

  def __init__(
      self,
      *args,
      attention_fn=None,
      supports_decode=False,
      supports_bias=True,
      supports_indices=True,
      supports_vjp=True,
      supports_mask=True,
      supports_tanh_clipping=True,
      supports_is_causal=True,
      supports_vmap=False,
  ):
    attention_fn = attention_fn or fa.PallasMosaicGpuFlashAttention()
    super().__init__(
        *args,
        attention_fn=attention_fn,
        supports_bias=supports_bias,
        supports_vjp=supports_vjp,
        supports_mask=supports_mask,
        supports_tanh_clipping=supports_tanh_clipping,
        supports_indices=supports_indices,
        supports_dropout=False,
        supports_cross_attention=True,
        supports_precisions=False,
        supports_vmap=supports_vmap,
        supports_is_causal=supports_is_causal,
    )
    self._supports_decode = supports_decode

  def _run_test_with_inputs(self, q, k, v, *, bias=None, **kwargs):
    # PallasMosaicGpuFlashAttention doesn't support high precisions,
    # (logits_dtype != f32) and f32 inputs. Override the arguments instead of
    # disabling basicaly most of the tests.
    impl_kwargs = kwargs.setdefault("impl_kwargs", {})
    impl_kwargs["precision"] = jax.lax.DotAlgorithmPreset.DEFAULT
    impl_kwargs["logits_dtype"] = jnp.float32

    def as_bf16(x):
      if isinstance(x, jax.Array) and x.dtype == jnp.float32:
        return x.astype(jnp.bfloat16)
      return x

    q, k, v, bias = map(as_bf16, (q, k, v, bias))
    atol = kwargs.get("atol", 0.0)
    kwargs["atol"] = max(atol, 0.0045 if bias is None else 0.007)
    kwargs["atol_grads"] = None if bias is None else 0.03

    if not impl_kwargs.get("normalize_output", True):
      kwargs["test_vjp"] = False

    if (q.ndim < 3) or (q.shape[-3] % 128 != 0) and not self._supports_decode:
      kwargs["expect_supported"] = False
    super()._run_test_with_inputs(q, k, v, bias=bias, **kwargs)

  def test_causal_mask(self):
    # TODO: Investigate why it's less accurate with causal mask.
    with test_base.override_test_args(atol=0.006, atol_grads=0.015):
      super().test_causal_mask()

  def test_causal_mask_cross_attention0(self):
    with test_base.override_test_args(atol=0.006, atol_grads=0.015):
      super().test_causal_mask_cross_attention0()  # pytype: disable=attribute-error

  def test_causal_mask_cross_attention1(self):
    self.skipTest("TODO: Support k-sequence non-multiple of block_kv.")

  def test_padding_mask_with_nans(self):
    self.skipTest("TODO: Fix.")

  def test_normalize_output(self):
    with test_base.override_test_args(atol=0.02):
      super().test_normalize_output()

  def test_base2(self):
    impl = fa.PallasMosaicGpuFlashAttention(use_base2=True)
    self._run_test((2, 1024, 4, 64), impl=impl)

  def test_unstable_softmax(self):
    if isinstance(self._attention_fn, fa.PallasMosaicGpuFlashAttention):
      impl = dataclasses.replace(self._attention_fn, use_stable_softmax=False)
      self._run_test((2, 1024, 4, 64), impl=impl)

  @parameterized.named_parameters(test_base.NAMED_ARG_SPECS.items())
  def test_bench(self, spec):
    suffix = [k for k, v in test_base.NAMED_ARG_SPECS.items() if v == spec][0]
    if spec.get("bias") is None or spec["q"].dtype == jnp.float32:
      atol_grads = None
    else:
      atol_grads = 0.15
    try:
      with test_base.override_test_args(atol=0.02, atol_grads=atol_grads):
        getattr(super(), "test_bench_" + suffix)()
    except ValueError as e:
      if "exceeds available shared memory" in str(e):
        self.skipTest(f"Test exceeds shared memory capacity: {e}")
      raise


# TODO: Add manual partitioning test.

if __name__ == "__main__":
  absltest.main()
