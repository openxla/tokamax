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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.attention import pallas_mosaic_gpu_flash_attention as flash_attention
from tokamax._src.ops.attention import test_base
from tokamax._src.ops.attention import bench_arg_specs


def _atol_ctx(atol: float):
  orig_run_test = test_base._run_test

  def my_run_test(*args, **kwargs):
    _ = kwargs.pop("atol", None)
    orig_run_test(*args, **(kwargs | dict(atol=atol)))

  return mock.patch.object(test_base, "_run_test", my_run_test)


class PallasMosaicGpuFlashAttentionTest(test_base.AttentionTestBase):

  def __init__(
      self,
      *args,
      attention_fn=None,
      supports_decode=False,
      supports_bias=True,
      supports_indices=True,
      supports_vjp=False,
      supports_mask=True,
      supports_tanh_clipping=True,
      supports_is_causal=True,
      supports_vmap=False,
  ):
    attention_fn = (
        attention_fn or flash_attention.PallasMosaicGpuFlashAttention()
    )
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

  def _run_test_with_inputs(self, q, k, v, *args, **kwargs):
    # PallasMosaicGpuFlashAttention doesn't support high precisions,
    # (logits_dtype != f32) and f32 inputs. Override the arguments instead of
    # disabling basicaly most of the tests.
    kwargs["impl_kwargs"] = kwargs.get("impl_kwargs", {}) | dict(
        precision=jax.lax.DotAlgorithmPreset.DEFAULT,
        logits_dtype=jnp.float32,
    )

    cast = (
        lambda x: x.astype(jnp.bfloat16)
        if isinstance(x, jax.Array) and x.dtype == jnp.float32
        else x
    )
    impl = kwargs.get("impl", self._attention_fn)
    kwargs["impl"] = lambda *args, **kwargs: impl(*map(cast, args), **kwargs)
    kwargs["atol"] = 0.007 if "bias" in kwargs else 0.0045
    super()._run_test_with_inputs(q, k, v, *args, **kwargs)

  def test_causal_mask(self):
    # TODO: Investigate why it's less accurate with causal mask.
    with _atol_ctx(0.01):
      super().test_causal_mask()

  def test_causal_mask_cross_attention0(self):
    with _atol_ctx(0.01):
      super().test_causal_mask_cross_attention0()  # pytype: disable=attribute-error

  def test_causal_mask_cross_attention1(self):
    self.skipTest("TODO: Support k-sequence non-multiple of block_kv.")

  def test_causal_mask_q_indices(self):
    with _atol_ctx(0.015):
      super().test_causal_mask_q_indices()

  def test_causal_mask_k_indices(self):
    with _atol_ctx(0.015):
      super().test_causal_mask_k_indices()

  @parameterized.parameters(*test_base.base_names_and_params("test_mask_api"))
  def test_mask_api(self, test_name, kwargs):
    kwargs = eval(f"dict{kwargs}")  # pylint: disable=eval-used
    if (k_start := kwargs.get("k_start")) and kwargs.get("is_causal"):
      k_start = jnp.array(k_start if isinstance(k_start, range) else [k_start])
      if jnp.any(k_start > jnp.arange(1, 1024 + 1)):
        self.skipTest("k_start > causality diagonal currently unsupported.")

    with _atol_ctx(0.02):
      getattr(super(), test_name)()

  def test_padding_mask(self):
    with _atol_ctx(0.01):
      super().test_padding_mask()

  def test_local_attention_mask(self):
    with _atol_ctx(0.01):
      super().test_local_attention_mask()

  @parameterized.parameters(
      *test_base.base_names_and_params("test_vmap")
  )
  def test_vmap(self, test_name, kwargs):
    with _atol_ctx(0.01):
      getattr(super(), test_name)()

  def test_padding_mask_with_nans(self):
    self.skipTest("TODO: Fix.")

  @parameterized.named_parameters(bench_arg_specs.ARG_SPECS.items())
  def test_bench(self, spec):
    self.skipTest("TODO: Enable benchmark tests.")

  def test_normalize_output(self):
    self.skipTest("TODO: Fix precision when `normalize_output=False`.")

  @parameterized.parameters(
      *test_base.base_names_and_params("test_quantized_int8")
  )
  def test_quantized_int8(self, test_name, kwargs):
    with _atol_ctx(0.02):
      getattr(super(), test_name)()

  @parameterized.parameters(
      *test_base.base_names_and_params("test_quantized_int4")
  )
  def test_quantized_int4(self, test_name, kwargs):
    with _atol_ctx(0.08):
      getattr(super(), test_name)()

  def test_base2(self):
    impl = flash_attention.PallasMosaicGpuFlashAttention(use_base2=True)
    self._run_test((2, 1024, 4, 64), impl=impl)


# TODO: Remove after VJP reaches feature parity with the fwd pass.
class PallasMosaicGpuFlashAttentionVjpTest(PallasMosaicGpuFlashAttentionTest):

  def __init__(self, *args, attention_fn=None):
    attention_fn = (
        attention_fn or flash_attention.PallasMosaicGpuFlashAttention()
    )
    self._supports_decode = False
    super().__init__(
        *args,
        attention_fn=attention_fn,
        supports_bias=False,
        supports_indices=False,
        supports_vjp=True,
        supports_mask=True,
        supports_tanh_clipping=False,
        supports_is_causal=False,
        supports_vmap=False,
    )

  def _run_test_with_inputs(self, *args, **kwargs):
    if not kwargs.get("test_vjp", True):
      self.skipTest("No point in testing forward only.")
    return super()._run_test_with_inputs(*args, **kwargs)

  def test_non_power_of_two_head_dim(self):
    # TODO: Fix non-power-of-two head dimension.
    self.skipTest("Only multiples of 64 head dims are supported.")

  @parameterized.product(input_dim=(24, 128), output_dim=(64, 112))
  def test_different_output_head_dim(self, input_dim, output_dim):
    # TODO: Support different output head dimensions in vjp.
    self.skipTest("Different output head dimension not supported in vjp.")

  def test_local_attention_mask(self):
    with _atol_ctx(0.03):
      super().test_local_attention_mask()

# TODO: Add manual partitioning test.

if __name__ == "__main__":
  absltest.main()
