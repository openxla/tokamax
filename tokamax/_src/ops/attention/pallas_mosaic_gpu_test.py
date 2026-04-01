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
import functools
from types import UnionType  # pylint: disable=g-importing-member
from typing import Union, get_origin
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src import gpu_utils
import pytest
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu as fa
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp as fa_vjp
from tokamax._src.ops.attention import test_base
from typing_extensions import override


@pytest.mark.skip(reason="Too slow for OSS regression tests.")
class PallasMosaicGpuFlashAttentionTest(test_base.AttentionTestBase):

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def __init__(
      self,
      *args,
      attention_fn=None,
      supports_bias=True,
      supports_indices=True,
      supports_vjp=True,
      supports_mask=True,
      supports_tanh_clipping=True,
      supports_is_causal=True,
      supports_vmap=True,
  ):
    if attention_fn is None:
      vjp = fa_vjp.PallasMosaicGpuFlashAttentionVjp(
          dbias_intermediate_dtype=jnp.float32
      )
      attention_fn = fa.PallasMosaicGpuFlashAttention(vjp=vjp)

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
        supports_logits_dtype=False,
    )

  def _run_test_with_inputs(self, q, k, v, *, bias=None, **kwargs):
    # PallasMosaicGpuFlashAttention doesn't support high precisions. Override
    # the arguments instead of disabling most of the tests.
    impl_kwargs = kwargs.setdefault("impl_kwargs", {})
    if any(x.dtype == jnp.float32 for x in (q, k, v)):
      impl_kwargs["precision"] = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
      # Bottleneck precision for reference implementation to match kernel.
      bf16_round_trip = lambda x: x.astype(jnp.bfloat16).astype(jnp.float32)
      q, k, v = map(bf16_round_trip, (q, k, v))

    atol = kwargs.get("atol", 0.0)
    kwargs["atol"] = max(atol, 0.0045)
    kwargs["atol_grads"] = None if bias is None else 0.02
    kwargs["test_vjp_deterministic"] = not gpu_utils.is_sm100()

    if not impl_kwargs.get("normalize_output", True) or (
        bias is not None and gpu_utils.is_sm100()
    ):
      kwargs["test_vjp"] = False

    test_vjp = kwargs.get("test_vjp", self._supports_vjp)
    if gpu_utils.is_sm100():
      # TODO: Head dim > 64 is unsupported at the moment on
      # SM100 because smem and tmem is are both exceeded exceeded.
      if q.shape[-1] > 64 and test_vjp:
        kwargs["expect_supported"] = False

      impl = kwargs.get("impl", self._attention_fn)
      if not getattr(impl, "use_stable_softmax", True):
        kwargs["expect_supported"] = False

    super()._run_test_with_inputs(q, k, v, bias=bias, **kwargs)

  def test_causal_mask(self):
    # TODO: Investigate why it's less accurate with causal mask.
    with test_base.override_test_args(
        atol={1.0: 0.008, 0.99: 0.006}, atol_grads=0.025
    ):
      super().test_causal_mask()

  def test_causal_mask_cross_attention0(self):
    with test_base.override_test_args(
        atol={1.0: 0.008, 0.99: 0.006}, atol_grads={1.0: 0.02, 0.99: 0.012}
    ):
      super().test_causal_mask_cross_attention0()  # pytype: disable=attribute-error

  def test_causal_mask_cross_attention1(self):
    self.skipTest("TODO: Support k-sequence non-multiple of block_kv.")

  def test_padding_mask_with_nans(self):
    self.skipTest("TODO: Fix.")

  def test_normalize_output(self):
    with test_base.override_test_args(atol=0.02):
      super().test_normalize_output()

  @parameterized.product(
      use_stable_softmax=(True, False), rescale_threshold=(1.0, 0.5)
  )
  def test_op_parameters(self, use_stable_softmax, rescale_threshold):
    self._test_op_parameters(use_stable_softmax, rescale_threshold)

  def _test_op_parameters(self, use_stable_softmax, rescale_threshold):
    op_cls = type(self._attention_fn)
    kwargs = {}
    if hasattr(op_cls, "use_stable_softmax"):
      kwargs["use_stable_softmax"] = use_stable_softmax
    elif not use_stable_softmax:
      self.skipTest("`use_stable_softmax=False` unsupported.")

    if hasattr(op_cls, "rescale_threshold"):
      kwargs["rescale_threshold"] = rescale_threshold
    elif rescale_threshold != 1.0:
      self.skipTest("`rescale_threshold != 1.0` unsupported.")

    with mock.patch.object(self, "_attention_fn", op_cls(**kwargs)):  # type: ignore
      self.test_self_attention0()  # pytype: disable=attribute-error
      if use_stable_softmax:
        self.test_normalize_output()

  @override
  def _test_bench(self, spec):
    # TODO: Remove once fixed.
    if "B200" in jax.devices()[0].device_kind:
      self.skipTest("Skipping test on B200s")
    atol_grads = None if spec.get("bias") is None else 0.04
    try:
      with test_base.override_test_args(atol=0.02, atol_grads=atol_grads):
        super()._test_bench(spec)
    except ValueError as e:
      if "exceeds available shared memory" in str(e):
        self.skipTest(f"Test exceeds shared memory capacity: {e}")
      raise

  def test_autotune_configs(self):
    # Test that all autotuning configs yield reasonable results.
    assert isinstance(self._attention_fn, base.DotProductAttention)
    q, k, v, *_ = test_base._create_inputs(q_shape=(2, 384, 4, 128))
    bound_args = self._attention_fn.bind(q, k, v)
    configs = self._attention_fn._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)
    for config in configs:
      with self.subTest(f"{config=}"):
        impl = type(self._attention_fn)(config)
        self._run_test_with_inputs(q, k, v, impl=impl)

  def test_vjp_autotune_configs(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    assert isinstance(self._attention_fn, base.DotProductAttention)
    assert hasattr(self._attention_fn, "vjp")
    attn_fn = self._attention_fn
    vjp_fn = attn_fn.vjp

    q, k, v, *_ = test_base._create_inputs(q_shape=(2, 384, 4, 64))
    kwargs = dict(precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32)
    fwd_with_res = functools.partial(attn_fn, **kwargs, return_residuals=True)
    out, res = jax.eval_shape(fwd_with_res, q, k, v)
    ba = attn_fn.bind(q, k, v, **kwargs)
    ba = dataclasses.replace(
        ba, arguments=ba.arguments | dict(residuals=res, out=out, dout=out)
    )
    configs = vjp_fn._get_autotuning_configs(ba)
    self.assertNotEmpty(configs)
    for config in configs:
      with self.subTest(f"{config=}"):
        impl = type(attn_fn)(vjp=type(vjp_fn)(config=config))
        self._run_test_with_inputs(q, k, v, impl=impl)

  def test_split_k(self):
    assert hasattr(self._attention_fn, "config_cls")
    if not hasattr(self._attention_fn.config_cls, "split_k"):
      self.skipTest("split_k unsupported for this implementation.")
    op_cls = type(self._attention_fn)
    cfg_cls = op_cls.config_cls
    if get_origin(cfg_cls) in {Union, UnionType}:
      cfg_cls = fa._get_kernel_module(backend.get_default_device()).Config
    cfg_dict = dict(block_q=128, block_kv=64, split_k=2, collective=False)
    cfg_dict = {k: v for k, v in cfg_dict.items() if hasattr(cfg_cls, k)}
    self._run_test((2, 1024, 4, 64), impl=op_cls(config=cfg_cls(**cfg_dict)))

  @override
  def _test_small_sequences(self, seq_q, seq_kv):
    with test_base.override_test_args(atol=0.02, atol_grads=0.04):
      super()._test_small_sequences(seq_q, seq_kv)


# TODO: Add manual partitioning test.

if __name__ == "__main__":
  absltest.main()
