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
from typing import Union, cast, get_origin, override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
import pytest
from tokamax._src.ops.attention import arg_specs
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu as fa
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp as fa_vjp
from tokamax._src.ops.attention import test_base


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
    atol_grads = kwargs.get("atol_grads", 0.0) or 0.0
    kwargs["atol_grads"] = None if bias is None else max(atol_grads, 0.02)

    if not impl_kwargs.get("normalize_output", True):
      kwargs["test_vjp"] = False

    test_vjp = kwargs.get("test_vjp", self._supports_vjp)
    if gpu_utils.is_sm80():
      impl = kwargs.get("impl", self._attention_fn)
      if (
          not getattr(impl, "use_stable_softmax", True)
          or getattr(impl, "rescale_threshold", 1.0) != 1.0
      ):
        kwargs["expect_supported"] = False
      if (q.shape[-1] > 256 or v.shape[-1] > 256) and test_vjp:
        kwargs["expect_supported"] = False
    if gpu_utils.is_sm100():

      impl = kwargs.get("impl", self._attention_fn)
      if not getattr(impl, "use_stable_softmax", True):
        kwargs["expect_supported"] = False

      # SM100 requires block_q=128 and block_kv=128 for the VJP dual kernel.
      # For head_dim=256, this pushes SMEM > 227KB, causing a ValueError.
      if q.shape[-1] > 256 and test_vjp:
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
      super().test_causal_mask_cross_attention0()  # pyrefly: ignore[missing-attribute]

  def test_causal_mask_cross_attention1(self):
    if not gpu_utils.is_sm80():
      self.skipTest("TODO: Support k-sequence non-multiple of block_kv.")
    with test_base.override_test_args(
        atol={1.0: 0.008, 0.99: 0.006}, atol_grads={1.0: 0.02, 0.99: 0.012}
    ):
      super().test_causal_mask_cross_attention1()  # pyrefly: ignore[missing-attribute]

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
      self.test_self_attention0()  # pyrefly: ignore[missing-attribute]
      if use_stable_softmax:
        self.test_normalize_output()

  @override
  def _test_bench(self, spec):
    atol_grads = None if spec.get("bias") is None else {0.99995: 0.04, 1.0: 0.5}
    try:
      with test_base.override_test_args(
          atol={0.99: 0.02, 1.0: 0.05}, atol_grads=atol_grads
      ):
        super()._test_bench(spec)
    except ValueError as e:
      if "exceeds available shared memory" in str(e):
        self.skipTest(f"Test exceeds shared memory capacity: {e}")
      raise

  def test_autotune_configs(self):
    # Test that all autotuning configs yield reasonable results.
    assert isinstance(self._attention_fn, base.DotProductAttention)
    q, k, v, *_ = test_base._create_inputs(q_shape=(2, 384, 4, 128))
    precision = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    bound_args = self._attention_fn.bind(q, k, v, precision=precision)
    configs = self._attention_fn._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)
    for config in configs:
      with self.subTest(f"{config=}"):
        if (
            gpu_utils.is_sm100()
            and not config.collective
            and config.block_kv < 128
        ):
          self.skipTest(
              "Skipping test for config that is known to cause issues"
              " (b/517048781)."
          )
        impl = type(self._attention_fn)(config)
        self._run_test_with_inputs(q, k, v, impl=impl)

  def test_vjp_autotune_configs_smem_filtering(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    assert isinstance(self._attention_fn, base.DotProductAttention)
    assert hasattr(self._attention_fn, "vjp")
    attn_fn = self._attention_fn
    vjp_fn = cast(fa_vjp.PallasMosaicGpuFlashAttentionVjp, attn_fn.vjp)
    q = jax.ShapeDtypeStruct((2, 256, 4, 256), jnp.bfloat16)
    ba = attn_fn.bind(q, q, q)
    vjp_ba = vjp_fn.bind(**ba.vjp_arg_spec)
    configs = vjp_fn._get_autotuning_configs(vjp_ba)
    self.assertNotEmpty(configs)
    if gpu_utils.is_sm100():
      for cfg in configs:
        self.assertEqual(cfg.num_stages, 1)
    else:
      for cfg in configs:
        self.assertLessEqual(cfg.num_stages, 2)

  def test_vjp_autotune_configs(self):
    self.skipTest("TODO: Disable due to OOMs.")
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    assert isinstance(self._attention_fn, base.DotProductAttention)
    assert hasattr(self._attention_fn, "vjp")
    attn_fn = self._attention_fn
    vjp_fn = cast(fa_vjp.PallasMosaicGpuFlashAttentionVjp, attn_fn.vjp)

    q, k, v, *_ = test_base._create_inputs(q_shape=(2, 384, 4, 64))
    kwargs = dict(precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32)
    fwd_with_res = functools.partial(attn_fn, **kwargs, return_residuals=True)
    out, res = jax.eval_shape(fwd_with_res, q, k, v)
    ba = attn_fn.bind(q, k, v, **kwargs)
    ba = dataclasses.replace(
        ba, arguments=dict(**ba.arguments, residuals=res, out=out, dout=out)
    )
    configs = vjp_fn._get_autotuning_configs(ba)
    self.assertNotEmpty(configs)
    for config in configs:
      with self.subTest(f"{config=}"):
        impl = type(attn_fn)(vjp=type(vjp_fn)(config=config))
        self._run_test_with_inputs(q, k, v, impl=impl)
        jax.clear_caches()

  def test_split_k(self):
    assert hasattr(self._attention_fn, "config_cls")
    if not hasattr(self._attention_fn.config_cls, "split_k"):
      self.skipTest("split_k unsupported for this implementation.")
    op_cls = type(self._attention_fn)
    cfg_cls = op_cls.config_cls
    if get_origin(cfg_cls) in {Union, UnionType}:
      cfg_cls = fa._get_kernel_module().Config
    cfg_dict = dict(block_q=128, block_kv=64, split_k=2, collective=False)
    cfg_dict = {k: v for k, v in cfg_dict.items() if hasattr(cfg_cls, k)}
    self._run_test((2, 1024, 4, 64), impl=op_cls(config=cfg_cls(**cfg_dict)))

  @override
  def _test_small_sequences(self, seq_q, seq_kv):
    with test_base.override_test_args(atol=0.02, atol_grads=0.04):
      super()._test_small_sequences(seq_q, seq_kv)

  def test_unaligned_seq_kv_bias_and_mask_vjp(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    self._run_test(
        q_shape=(2, 129, 4, 64),
        kv_shape=(2, 129, 4, 64),
        bias_shape=(2, 4, 129, 129),
        mask_shape=(2, 4, 129, 129),
        dtype=jnp.bfloat16,
        atol=0.02,
        atol_grads=0.04,
        expect_supported=self._supports_bias and self._supports_mask,
    )

  def test_unaligned_seq_q_vjp(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    self._run_test(
        q_shape=(2, 63, 4, 64),
        kv_shape=(2, 128, 4, 64),
        dtype=jnp.bfloat16,
        atol=0.02,
        atol_grads=0.04,
    )
    self._run_test(
        q_shape=(2, 36, 4, 64),
        kv_shape=(2, 128, 4, 64),
        bias_shape=(2, 4, 36, 1),
        dtype=jnp.bfloat16,
        atol=0.02,
        atol_grads=0.04,
        expect_supported=self._supports_bias,
    )

  def test_unaligned_fp16_and_fully_masked_vjp(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    for seq_q in (16, 48, 72):
      self._run_test(
          q_shape=(1, seq_q, 4, 64),
          kv_shape=(1, 65, 4, 64),
          dtype=jnp.float16,
          atol=0.02,
          atol_grads=0.04,
      )

  def test_gqa_bcast_bias_and_mask_vjp(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    self._run_test(
        q_shape=(1, 128, 8, 64),
        kv_shape=(1, 128, 2, 64),
        bias_shape=(1, 8, 128, 1),
        mask_shape=(1, 8, 1, 128),
        dtype=jnp.bfloat16,
        ref_impl=base.DotProductAttention(),
        atol=0.02,
        atol_grads=0.04,
        expect_supported=self._supports_bias and self._supports_mask,
    )
    self._run_test(
        q_shape=(1, 128, 8, 64),
        kv_shape=(1, 128, 2, 64),
        mask_shape=(1, 1, 1, 1),
        dtype=jnp.bfloat16,
        ref_impl=base.DotProductAttention(),
        atol=0.02,
        atol_grads=0.04,
        expect_supported=self._supports_mask,
    )

  def test_causal_bias_and_cross_attention_vjp(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    with test_base.override_test_args(atol=0.02, atol_grads=0.04):
      self._run_test(
          q_shape=(1, 256, 4, 64),
          kv_shape=(1, 256, 4, 64),
          bias_shape=(1, 4, 256, 256),
          is_causal=True,
          dtype=jnp.bfloat16,
          ref_impl=base.DotProductAttention(),
          expect_supported=self._supports_bias and self._supports_is_causal,
      )
      self._run_test(
          q_shape=(1, 128, 4, 64),
          kv_shape=(1, 256, 4, 64),
          is_causal=True,
          dtype=jnp.bfloat16,
          ref_impl=base.DotProductAttention(),
          expect_supported=self._supports_is_causal,
      )

  def test_head_dim_256_vjp(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    for seq_len in (64, 256):
      for is_causal in (False, True):
        self._run_test(
            q_shape=(1, seq_len, 4, 256),
            kv_shape=(1, seq_len, 4, 256),
            dtype=jnp.bfloat16,
            is_causal=is_causal,
            ref_impl=base.DotProductAttention(),
            atol=0.02,
            atol_grads=0.04,
        )

  def test_fp32_inputs_vjp_grad_dtypes(self):
    if not self._supports_vjp:
      self.skipTest("VJP unsupported for this implementation.")
    q = jnp.ones((1, 128, 4, 64), dtype=jnp.float32)
    k = jnp.ones((1, 128, 4, 64), dtype=jnp.float32)
    v = jnp.ones((1, 128, 4, 64), dtype=jnp.float32)
    fn = functools.partial(
        self._attention_fn,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
    )
    out, vjp_fn = jax.vjp(fn, q, k, v)
    dq, dk, dv = vjp_fn(jnp.ones_like(out))
    self.assertEqual(out.dtype, jnp.float32)
    self.assertEqual(dq.dtype, jnp.float32)
    self.assertEqual(dk.dtype, jnp.float32)
    self.assertEqual(dv.dtype, jnp.float32)

  @override
  def _test_vmap(self, vmap_in_axes):
    if gpu_utils.is_sm80():
      with test_base.override_test_args(atol=0.01):
        super()._test_vmap(vmap_in_axes)
    else:
      super()._test_vmap(vmap_in_axes)

  def test_sm80_seq_lengths_decomposition_and_vjp(self):
    if not gpu_utils.is_sm80():
      self.skipTest("SM80-specific sequence length optimization test.")
    q, k, v, bias, _, _ = test_base._create_inputs(
        q_shape=(2, 512, 8, 128),
        kv_shape=(2, 512, 2, 128),
        bias_shape=(1, 8, 512, 512),
        dtype=jnp.bfloat16,
    )
    q_len = jnp.array([70, 390], dtype=jnp.int32)
    kv_len = jnp.array([135, 450], dtype=jnp.int32)
    mask = base.Mask(
        is_causal=True,
        q_end=q_len[:, None, None],
        k_end=kv_len[:, None, None],
    )
    # Verify `extract_q_range` + `decompose_mask` produces no dense bool mask.
    rem_mask, q_start, q_end = fa.common.extract_q_range(
        mask, q.ndim, None, None
    )
    bool_mask, _, _, k_end = fa.common.decompose_mask(
        rem_mask, q, k, None, None
    )
    self.assertIsNone(bool_mask)
    self.assertIsNone(q_start)
    self.assertIsNotNone(q_end)
    self.assertIsNotNone(k_end)

    ref_fn = lambda q_, k_, v_, b_: jax.nn.dot_product_attention(
        q_.astype(jnp.float32),
        jnp.repeat(k_.astype(jnp.float32), 4, axis=2),
        jnp.repeat(v_.astype(jnp.float32), 4, axis=2),
        bias=b_.astype(jnp.float32),
        is_causal=True,
        query_seq_lengths=q_len,
        key_value_seq_lengths=kv_len,
        implementation="xla",
    ).astype(jnp.bfloat16)
    ans_fn = lambda q_, k_, v_, b_: self._attention_fn(
        q_, k_, v_, bias=b_, mask=mask
    )
    dout = jnp.ones_like(q)
    out_ref, vjp_ref = jax.vjp(ref_fn, q, k, v, bias)
    out_ans, vjp_ans = jax.vjp(ans_fn, q, k, v, bias)
    dq_ref, dk_ref, dv_ref, db_ref = vjp_ref(dout)
    dq_ans, dk_ans, dv_ans, db_ans = vjp_ans(dout)

    test_base.test_utils.assert_trees_all_close(
        out_ans, out_ref, atol=0.02, rtol=0.02
    )
    test_base.test_utils.assert_trees_all_close(
        (dq_ans, dk_ans, dv_ans),
        (dq_ref, dk_ref, dv_ref),
        atol=0.04,
        rtol=0.04,
    )
    test_base.test_utils.assert_trees_all_close(
        db_ans, db_ref, atol=0.08, rtol=0.08
    )

  def test_sm80_alphafold_and_deepseek_shapes(self):
    if not gpu_utils.is_sm80():
      self.skipTest("SM80-specific shape coverage test.")
    # AlphaFold head_dim=16 and head_dim=32 (both forward + VJP).
    self._run_test(
        q_shape=(4, 256, 4, 16),
        kv_shape=(4, 256, 4, 16),
        bias_shape=(1, 4, 256, 256),
        mask_shape=(4, 1, 1, 256),
        dtype=jnp.bfloat16,
        atol=0.02,
        atol_grads=0.06,
    )
    self._run_test(
        q_shape=(4, 384, 4, 32),
        kv_shape=(4, 384, 4, 32),
        bias_shape=(1, 4, 384, 384),
        mask_shape=(4, 1, 1, 384),
        dtype=jnp.bfloat16,
        atol=0.02,
        atol_grads=0.06,
    )
    # DeepSeek-V2 asymmetric head_dim=(192, 128) causal forward + VJP.
    q, k, _, _, _, ss = test_base._create_inputs(
        q_shape=(2, 256, 4, 192), dtype=jnp.bfloat16
    )
    _, _, v, _, _, _ = test_base._create_inputs(
        q_shape=(2, 256, 4, 128), dtype=jnp.bfloat16
    )
    self._run_test_with_inputs(
        q,
        k,
        v,
        is_causal=True,
        ref_impl=base.DotProductAttention(),
        seed_seq=ss,
        atol=0.02,
        atol_grads=0.04,
    )

  def test_sm80_arg_specs_and_edge_cases(self):
    if not gpu_utils.is_sm80():
      self.skipTest("SM80-specific arg_specs and edge-case test.")

    # 1. Verify all specs in `arg_specs.ARG_SPECS` bind & fit SM80 SMEM.
    attn_fn = cast(fa.PallasMosaicGpuFlashAttention, self._attention_fn)
    vjp_fn = cast(fa_vjp.PallasMosaicGpuFlashAttentionVjp, attn_fn.vjp)
    self.assertNotEmpty(arg_specs.ARG_SPECS)
    for spec in arg_specs.ARG_SPECS:
      with self.subTest(spec=spec.full_name):
        ba = attn_fn.bind(**spec.args)
        cfg = attn_fn._get_heuristics_config(ba)
        self.assertIsInstance(cfg, fa.ConfigSM80)
        smem_bytes = fa.sm80._estimate_shared_mem_usage_bytes(ba, cfg)
        self.assertLessEqual(smem_bytes, fa.sm80._MAX_SMEM)
        if "forward_only" not in spec.tags:
          vjp_ba = vjp_fn.bind(**ba.vjp_arg_spec)
          vjp_cfg = vjp_fn._get_heuristics_config(vjp_ba)
          self.assertLessEqual(
              fa_vjp.sm80._estimate_dq_smem_bytes(
                  vjp_ba,
                  vjp_cfg.block_q_dq,
                  vjp_cfg.block_kv_dq,
                  vjp_cfg.num_stages,
                  None if spec.args.get("bias") is None else jnp.float32,
                  vjp_cfg.compute_wgs,
              ),
              fa_vjp.sm80._SMEM_SIZE_BYTES,
          )

    # 2. `jax.vmap` over non-leading axis (`in_axes=3`) forward + VJP.
    q5 = test_base.numerics.random_initialize(
        jax.ShapeDtypeStruct((2, 128, 4, 3, 64), jnp.bfloat16)
    )
    vmapped_ans = jax.vmap(
        lambda x: self._attention_fn(x, x, x, is_causal=True),
        in_axes=3,
        out_axes=3,
    )
    vmapped_ref = jax.vmap(
        lambda x: jax.nn.dot_product_attention(
            x.astype(jnp.float32),
            x.astype(jnp.float32),
            x.astype(jnp.float32),
            is_causal=True,
            implementation="xla",
        ).astype(jnp.bfloat16),
        in_axes=3,
        out_axes=3,
    )
    out_v_ans, vjp_v_ans = jax.vjp(vmapped_ans, q5)
    out_v_ref, vjp_v_ref = jax.vjp(vmapped_ref, q5)
    test_base.test_utils.assert_trees_all_close(
        out_v_ans, out_v_ref, atol=0.015, rtol=0.015
    )
    test_base.test_utils.assert_trees_all_close(
        vjp_v_ans(jnp.ones_like(q5)),
        vjp_v_ref(jnp.ones_like(q5)),
        atol=0.03,
        rtol=0.03,
    )

    # 3. Zero-length and single-token `q_end` / `k_end` boundary + VJP.
    q_b, k_b, v_b, bias_b, _, _ = test_base._create_inputs(
        q_shape=(2, 128, 4, 64),
        kv_shape=(2, 128, 4, 64),
        bias_shape=(1, 4, 128, 128),
        dtype=jnp.bfloat16,
    )
    q_len_b = jnp.array([0, 1], dtype=jnp.int32)
    kv_len_b = jnp.array([1, 65], dtype=jnp.int32)
    mask_b = base.Mask(
        is_causal=True,
        q_end=q_len_b[:, None, None],
        k_end=kv_len_b[:, None, None],
    )

    def _b_ans(q_, k_, v_, b_):
      out = self._attention_fn(q_, k_, v_, bias=b_, mask=mask_b)
      valid_q = (jnp.arange(128)[None, :] < q_len_b[:, None])[:, :, None, None]
      return out * valid_q

    def _b_ref(q_, k_, v_, b_):
      return jax.nn.dot_product_attention(
          q_.astype(jnp.float32),
          k_.astype(jnp.float32),
          v_.astype(jnp.float32),
          bias=b_.astype(jnp.float32),
          is_causal=True,
          query_seq_lengths=q_len_b,
          key_value_seq_lengths=kv_len_b,
          implementation="xla",
      ).astype(jnp.bfloat16)

    dout_b = jnp.ones_like(q_b)
    out_ba, vjp_ba = jax.vjp(_b_ans, q_b, k_b, v_b, bias_b)
    out_br, vjp_br = jax.vjp(_b_ref, q_b, k_b, v_b, bias_b)
    test_base.test_utils.assert_trees_all_close(
        out_ba, out_br, atol=0.02, rtol=0.02
    )
    test_base.test_utils.assert_trees_all_close(
        vjp_ba(dout_b), vjp_br(dout_b), atol=0.04, rtol=0.04
    )

    # 4. `fold_q_sequence_heads=True` forward + VJP.
    fold_impl = fa.PallasMosaicGpuFlashAttention(
        config=fa.ConfigSM80(
            block_q=64, block_kv=64, num_stages=2, num_wgs=1,
            fold_q_sequence_heads=True,
        ),
        vjp=vjp_fn,
    )
    self._run_test(
        q_shape=(2, 64, 8, 64),
        kv_shape=(2, 128, 2, 64),
        dtype=jnp.bfloat16,
        impl=fold_impl,
        ref_impl=base.DotProductAttention(),
        atol=0.02,
        atol_grads=0.04,
    )

    # 5. `head_dim > 256` VJP raises `NotImplementedError`.
    q_512 = jnp.ones((1, 64, 2, 512), jnp.bfloat16)
    with self.assertRaises(NotImplementedError):
      _, vjp_512 = jax.vjp(self._attention_fn, q_512, q_512, q_512)
      vjp_512(q_512)


# TODO: Add manual partitioning test.

if __name__ == "__main__":
  absltest.main()
