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


import functools
import typing
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.attention import base as fa_base
from tokamax._src.ops.attention import pallas_mosaic_tpu as fa
from tokamax._src.ops.attention import pallas_mosaic_tpu_vjp as fa_vjp


class PallasMosaicTpuFlashAttentionTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != 'tpu':
      self.skipTest('Only supported on TPUs.')
    super().setUp()

  def _test_attention(
      self, q, k, v, do, mask, is_causal, logits_soft_cap, config=None, vjp_config=None
  ):
    kwargs = dict(
        mask=mask,
        is_causal=is_causal,
        logits_soft_cap=logits_soft_cap,
        logits_scale=0.1,
    )
    vjp = (
        None
        if vjp_config is None
        else fa_vjp.PallasMosaicTpuFlashAttentionVjp(config=vjp_config)
    )
    attention_impl = fa.PallasMosaicTpuFlashAttention(config=config, vjp=vjp)

    @jax.jit
    def f_base(query, key, value, do):
      primals, f_vjp = jax.vjp(
          functools.partial(fa_base.DotProductAttention(), **kwargs),
          query,
          key,
          value,
      )
      return primals, f_vjp(do)

    @jax.jit
    def f(query, key, value, do):
      primals, f_vjp = jax.vjp(
          functools.partial(attention_impl, **kwargs),
          query,
          key,
          value,
      )
      return primals, f_vjp(do)

    out_base, (dq_base, dk_base, dv_base) = f_base(q, k, v, do)
    out, (dq, dk, dv) = f(q, k, v, do)

    atol = 0.035 if logits_soft_cap else 0.15
    with self.subTest('output'):
      chex.assert_trees_all_close(out, out_base, atol=atol)

    atol = 0.15 if logits_soft_cap else 1.5
    with self.subTest('dq'):
      chex.assert_trees_all_close(dq, dq_base, atol=atol)

    atol = 0.13 if logits_soft_cap else 1.2
    with self.subTest('dk'):
      chex.assert_trees_all_close(dk, dk_base, atol=atol)

    atol = 0.03 if logits_soft_cap else 0.15
    with self.subTest('dv'):
      chex.assert_trees_all_close(dv, dv_base, atol=atol)

  @parameterized.product(
      dtype=[jnp.float32],
      num_kv_heads=[1, 2, 4],
      masking=[None, 'causal', 'bool'],
      logits_soft_cap=[None, 3.4],
  )
  def test_simple(self, dtype, num_kv_heads, masking, logits_soft_cap):
    head_dim = 32
    q_seq_len = 128
    kv_seq_len = 128
    num_q_heads = 4
    batch_size = 2

    q = jax.ShapeDtypeStruct(
        (batch_size, q_seq_len, num_q_heads, head_dim), dtype
    )
    k = jax.ShapeDtypeStruct(
        (batch_size, kv_seq_len, num_kv_heads, head_dim), dtype
    )
    v = jax.ShapeDtypeStruct(
        (batch_size, kv_seq_len, num_kv_heads, head_dim), dtype
    )

    is_causal = masking == 'causal'
    mask = (
        jax.ShapeDtypeStruct((1, 1, q_seq_len, kv_seq_len), dtype=jnp.bool_)
        if masking == 'bool'
        else None
    )
    do = jax.ShapeDtypeStruct(
        (batch_size, q_seq_len, num_q_heads, head_dim), dtype=jnp.float32
    )
    q, k, v, mask, do = numerics.random_initialize((q, k, v, mask, do))
    self._test_attention(q, k, v, do, mask, is_causal, logits_soft_cap)

  def test_autotune_configs(self):
    head_dim = 32
    q_seq_len = 256
    kv_seq_len = 256
    num_q_heads = 4
    num_kv_heads = 4
    batch_size = 2
    dtype = jnp.float32

    q_shape = (batch_size, q_seq_len, num_q_heads, head_dim)
    k_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
    v_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)

    q = jax.ShapeDtypeStruct(q_shape, dtype)
    k = jax.ShapeDtypeStruct(k_shape, dtype)
    v = jax.ShapeDtypeStruct(v_shape, dtype)
    do = jax.ShapeDtypeStruct(q_shape, dtype=jnp.float32)
    q, k, v, do = numerics.random_initialize((q, k, v, do))

    attention_fn = fa.PallasMosaicTpuFlashAttention()
    bound_args = attention_fn.bind(q, k, v)
    configs = attention_fn._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)

    for config in configs:
      with self.subTest(f'{config=}'):
        self._test_attention(
            q,
            k,
            v,
            do,
            mask=None,
            is_causal=False,
            logits_soft_cap=None,
            config=config,
        )

  def test_autotune_vjp(self):
    head_dim = 128
    q_seq_len = 512
    kv_seq_len = 512
    num_q_heads = 4
    num_kv_heads = 4
    batch_size = 2
    dtype = jnp.float32

    q_shape = (batch_size, q_seq_len, num_q_heads, head_dim)
    k_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
    v_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)

    q = jax.ShapeDtypeStruct(q_shape, dtype)
    k = jax.ShapeDtypeStruct(k_shape, dtype)
    v = jax.ShapeDtypeStruct(v_shape, dtype)
    do = jax.ShapeDtypeStruct(q_shape, dtype=jnp.float32)
    q, k, v, do = numerics.random_initialize((q, k, v, do))

    residuals = (
        jnp.zeros((batch_size, num_q_heads, q_seq_len), dtype=jnp.float32),
        jnp.zeros((batch_size, num_q_heads, q_seq_len), dtype=jnp.float32),
    )
    out = jnp.zeros(q_shape, dtype=jnp.float32)

    attention_fn = fa.PallasMosaicTpuFlashAttention()
    vjp_op = typing.cast(
        fa_vjp.PallasMosaicTpuFlashAttentionVjp, attention_fn.vjp
    )
    bound_args = vjp_op.bind(
        residuals,
        out,
        do,
        q,
        k,
        v,
        bias=None,
        logits_scale=0.1,
        mask=fa_base.Mask(),
        precision=(
            jax.lax.DotAlgorithmPreset.DEFAULT,
            jax.lax.DotAlgorithmPreset.DEFAULT,
        ),
        logits_dtype=jnp.float32,
        logits_soft_cap=None,
        dropout_mask=None,
        dropout_rate=0.0,
        paging_info=None,
        q_indices=None,
        k_indices=None,
        normalize_output=True,
        return_residuals=False,
    )
    configs = vjp_op._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)

    for config in configs:
      with self.subTest(f'{config=}'):
        self._test_attention(
            q,
            k,
            v,
            do,
            mask=None,
            is_causal=False,
            logits_soft_cap=None,
            vjp_config=config,
        )

  @parameterized.named_parameters(
      ('seq_64', 64, 128, jnp.bfloat16, False, (128, 128, 128), 128),
      ('seq_256', 256, 128, jnp.bfloat16, False, (256, 256, 256), 256),
      ('non_pow2', 768, 128, jnp.bfloat16, False, (256, 256, 256), 256),
      ('bf16_2k', 2048, 128, jnp.bfloat16, False, (1024, 1024, 512), 512),
      ('fp32_2k', 2048, 128, jnp.float32, False, (512, 512, 512), 256),
      ('masked_2k', 2048, 128, jnp.bfloat16, True, (512, 512, 512), 256),
      ('hd256_2k', 2048, 256, jnp.bfloat16, False, (512, 512, 512), 256),
      ('hd64_bf16', 2048, 64, jnp.bfloat16, False, (1024, 1024, 512), 512),
      ('hd64_fp32', 2048, 64, jnp.float32, False, (512, 512, 512), 256),
      (
          'q_fp32_v_bf16',
          2048,
          128,
          jnp.float32,
          False,
          (512, 512, 512),
          256,
          jnp.bfloat16,
      ),
      (
          'q_bf16_v_fp32',
          2048,
          128,
          jnp.bfloat16,
          False,
          (512, 512, 512),
          256,
          jnp.float32,
      ),
  )
  def test_heuristics_config(
      self,
      seq_len,
      head_dim,
      dtype,
      has_mask,
      fwd_tiles,
      vjp_bkv_compute,
      v_dtype=None,
  ):
    q = jax.ShapeDtypeStruct((2, seq_len, 8, head_dim), dtype)
    v = jax.ShapeDtypeStruct(
        (2, seq_len, 8, head_dim), dtype if v_dtype is None else v_dtype
    )
    res = jax.ShapeDtypeStruct((2, 8, seq_len), jnp.float32)
    mask = (
        jax.ShapeDtypeStruct((1, 1, seq_len, seq_len), jnp.bool_)
        if has_mask
        else None
    )
    ba = fa.PallasMosaicTpuFlashAttention().bind(
        q, q, v, mask=mask, is_causal=not has_mask
    )
    cfg = ba.heuristics_config
    self.assertEqual(
        (cfg.block_q, cfg.block_kv, cfg.block_kv_compute), fwd_tiles
    )
    vjp_cfg = (
        fa_vjp.PallasMosaicTpuFlashAttentionVjp()
        .bind((res, res), v, v, *ba.args, **ba.kwargs)
        .heuristics_config
    )
    self.assertEqual(
        (
            vjp_cfg.block_q_dkv,
            vjp_cfg.block_kv_dkv,
            vjp_cfg.block_kv_dkv_compute,
        ),
        (*fwd_tiles[:2], vjp_bkv_compute),
    )

  @parameterized.named_parameters(
      ('bf16_gqa_1024', 1024, 2, jnp.bfloat16, True),
      ('bf16_non_pow2_768', 768, 8, jnp.bfloat16, True),
      ('fp32_mha_1024', 1024, 8, jnp.float32, False),
  )
  def test_long_seq_numerics(self, seq_len, num_kv_heads, dtype, is_causal):
    q = jax.ShapeDtypeStruct((2, seq_len, 8, 128), dtype)
    kv = jax.ShapeDtypeStruct((2, seq_len, num_kv_heads, 128), dtype)
    q, k, v, do = numerics.random_initialize((q, kv, kv, q))
    self._test_attention(
        q, k, v, do, mask=None, is_causal=is_causal, logits_soft_cap=None
    )

  @parameterized.parameters(64, 128)
  def test_seq_minor_layouts_with_default_vjp(self, head_dim):
    q_s = jax.ShapeDtypeStruct((2, 256, 4, head_dim), jnp.bfloat16)
    kv_s = jax.ShapeDtypeStruct((2, 256, 2, head_dim), jnp.bfloat16)
    q, k, v, do = numerics.random_initialize((q_s, kv_s, kv_s, q_s), seed=41)
    seq_minor = fa.splash.QKVLayout.SEQ_MINOR
    config = fa.Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_experimental_scheduler=True,
    )
    op_seq = fa.PallasMosaicTpuFlashAttention(config=config)
    assert isinstance(op_seq.vjp, fa_vjp.PallasMosaicTpuFlashAttentionVjp)
    self.assertEqual(op_seq.vjp.q_layout, seq_minor)
    self.assertEqual(op_seq.vjp.k_layout, seq_minor)
    self.assertEqual(op_seq.vjp.v_layout, seq_minor)
    self._test_attention(
        q,
        k,
        v,
        do,
        mask=None,
        is_causal=True,
        logits_soft_cap=None,
        config=config,
    )

  @parameterized.parameters('scalar', '1d', '2d', 'const')
  def test_max_logit_and_config_options(self, mode):
    b, t, n_q, n_kv, d = 2, 256, 4, 2, 64
    q_s = jax.ShapeDtypeStruct((b, t, n_q, d), jnp.float32)
    kv_s = jax.ShapeDtypeStruct((b, t, n_kv, d), jnp.float32)
    q, k, v, do = numerics.random_initialize((q_s, kv_s, kv_s, q_s), seed=53)

    max_logit_const = 10.0 if mode == 'const' else None
    if mode == 'scalar':
      max_logit_value = 10.0
    elif mode == '1d':
      max_logit_value = jnp.array([8.0, 9.0, 10.0, 11.0], dtype=jnp.float32)
    elif mode == '2d':
      max_logit_value = jnp.array(
          [[8.0, 9.0, 10.0, 11.0], [8.5, 9.5, 10.5, 11.5]], dtype=jnp.float32
      )
    else:
      max_logit_value = None

    head_minor = fa.splash.QKVLayout.HEAD_DIM_MINOR
    config = fa.Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=head_minor,
        k_layout=head_minor,
        v_layout=head_minor,
        use_experimental_scheduler=True,
        max_logit_const=max_logit_const,
        fuse_reciprocal=False,
        residual_checkpoint_name='context',
    )
    op_tpu = fa.PallasMosaicTpuFlashAttention(config=config)
    op_ref = fa_base.DotProductAttention()

    mask = fa_base.Mask(
        causal_mask=fa_base.CausalMask(offset=0, window_size=64)
    )
    _, res_default = fa.PallasMosaicTpuFlashAttention()(
        q, k, v, mask=mask, logits_scale=0.1, return_residuals=True
    )
    out_res, res_fixed = op_tpu(
        q,
        k,
        v,
        mask=mask,
        logits_scale=0.1,
        max_logit_value=max_logit_value,
        return_residuals=True,
    )
    expected_val = 10.0 if max_logit_const is not None else max_logit_value
    expected_max = jnp.broadcast_to(
        jnp.asarray(expected_val, dtype=jnp.float32)[..., None], (b, n_q, t)
    )
    chex.assert_trees_all_close(res_fixed[0], expected_max, atol=1e-4)
    self.assertFalse(jnp.allclose(res_default[0], res_fixed[0], atol=1e-2))

    out, vjp_fn = jax.vjp(
        lambda q_, k_, v_: op_tpu(
            q_,
            k_,
            v_,
            mask=mask,
            logits_scale=0.1,
            max_logit_value=max_logit_value,
        ),
        q,
        k,
        v,
    )
    out_base, vjp_base = jax.vjp(
        lambda q_, k_, v_: op_ref(q_, k_, v_, mask=mask, logits_scale=0.1),
        q,
        k,
        v,
    )
    dq, dk, dv = vjp_fn(do)
    dq_base, dk_base, dv_base = vjp_base(do)

    chex.assert_trees_all_close(out_res, out_base, atol=0.15)
    chex.assert_trees_all_close(out, out_base, atol=0.15)
    chex.assert_trees_all_close(dq, dq_base, atol=1.5)
    chex.assert_trees_all_close(dk, dk_base, atol=1.2)
    chex.assert_trees_all_close(dv, dv_base, atol=0.15)

  def test_unsupported_inputs_errors(self):
    q = jnp.ones((2, 128, 4, 64), dtype=jnp.bfloat16)
    head_minor = fa.splash.QKVLayout.HEAD_DIM_MINOR
    cfg_both = fa.Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=head_minor,
        k_layout=head_minor,
        v_layout=head_minor,
        use_experimental_scheduler=True,
        max_logit_const=10.0,
    )
    with self.assertRaisesRegex(ValueError, 'max_logit_const'):
      fa.PallasMosaicTpuFlashAttention(config=cfg_both)(
          q, q, q, max_logit_value=10.0
      )
    op_tpu = fa.PallasMosaicTpuFlashAttention()
    with self.assertRaisesRegex(NotImplementedError, 'normalize_output=False'):
      op_tpu(q, q, q, normalize_output=False)
    with self.assertRaisesRegex(NotImplementedError, 'q_indices'):
      op_tpu(q, q, q, q_indices=jnp.arange(128)[None, :])
    with self.assertRaisesRegex(NotImplementedError, 'k_indices'):
      op_tpu(q, q, q, k_indices=jnp.arange(128)[None, :])
    with self.assertRaisesRegex(NotImplementedError, 'non-negative'):
      op_tpu(
          q,
          q,
          q,
          mask=fa_base.Mask(causal_mask=fa_base.CausalMask(offset=-1)),
      )
    q_cross = jnp.ones((2, 256, 4, 16), dtype=jnp.bfloat16)
    k_cross = jnp.ones((2, 128, 4, 16), dtype=jnp.bfloat16)
    with self.assertRaisesRegex(NotImplementedError, 'all-masked query rows'):
      op_tpu(
          q_cross,
          k_cross,
          k_cross,
          mask=fa_base.Mask(
              causal_mask=fa_base.CausalMask(offset=0, window_size=65)
          ),
      )
    with self.assertRaisesRegex(NotImplementedError, 'all-masked query rows'):
      op_tpu(
          q,
          q,
          q,
          mask=fa_base.Mask(
              causal_mask=fa_base.CausalMask(offset=64, window_size=55)
          ),
      )

  @parameterized.parameters(
      (128, 127, None),
      (128, 128, None),
      (128, 128, 256),
      (256, 255, None),
  )
  def test_full_covering_analytic_masks(self, seq_len, offset, window_size):
    s = jax.ShapeDtypeStruct((1, seq_len, 4, 64), jnp.bfloat16)
    q, k, v, do = numerics.random_initialize((s, s, s, s), seed=61)
    mask = fa_base.Mask(
        causal_mask=fa_base.CausalMask(offset=offset, window_size=window_size)
    )
    self._test_attention(
        q, k, v, do, mask=mask, is_causal=False, logits_soft_cap=None
    )

  @parameterized.product(
      fuse_reciprocal=[True, False],
      use_base2_exp=[True, False],
  )
  def test_max_logit_zero_denom_and_underflow(
      self, fuse_reciprocal, use_base2_exp
  ):
    s = jax.ShapeDtypeStruct((1, 128, 4, 64), jnp.float32)
    q, k, v, do = numerics.random_initialize((s, s, s, s), seed=67)
    head_minor = fa.splash.QKVLayout.HEAD_DIM_MINOR
    cfg = fa.Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=head_minor,
        k_layout=head_minor,
        v_layout=head_minor,
        use_experimental_scheduler=True,
        use_base2_exp=use_base2_exp,
        fuse_reciprocal=fuse_reciprocal,
    )
    op_tpu = fa.PallasMosaicTpuFlashAttention(config=cfg)
    op_ref = fa_base.DotProductAttention()

    row0_mask = (
        jnp.ones((1, 1, 128, 128), dtype=jnp.bool_).at[:, :, 0, :].set(False)
    )
    with warnings.catch_warnings():
      warnings.simplefilter('error', UserWarning)
      out_res, (_, lse) = op_tpu(
          q,
          k,
          v,
          mask=row0_mask,
          logits_scale=0.1,
          max_logit_value=10,
          return_residuals=True,
      )
      out, vjp_fn = jax.vjp(
          lambda q_, k_, v_: op_tpu(
              q_, k_, v_, mask=row0_mask, logits_scale=0.1, max_logit_value=10
          ),
          q,
          k,
          v,
      )
    dq, dk, dv = vjp_fn(do)
    self.assertTrue(bool(jnp.all(jnp.isneginf(lse[:, :, 0]))))
    chex.assert_tree_all_finite((out_res, out, dq, dk, dv))
    chex.assert_trees_all_close(out[:, 0, :, :], 0.0, atol=1e-7)
    chex.assert_trees_all_close(dq[:, 0, :, :], 0.0, atol=1e-7)

    out_base, vjp_base = jax.vjp(
        lambda q_, k_, v_: op_ref(q_, k_, v_, mask=row0_mask, logits_scale=0.1),
        q,
        k,
        v,
    )
    dq_base, dk_base, dv_base = vjp_base(do)
    chex.assert_trees_all_close(
        out[:, 1:, :, :], out_base[:, 1:, :, :], atol=0.15
    )
    chex.assert_trees_all_close(dq[:, 1:, :, :], dq_base[:, 1:, :, :], atol=1.5)
    chex.assert_trees_all_close(dk, dk_base, atol=1.2)
    chex.assert_trees_all_close(dv, dv_base, atol=0.15)

    out_uf, vjp_uf = jax.vjp(
        lambda q_, k_, v_: op_tpu(
            q_, k_, v_, logits_scale=0.1, max_logit_value=120.0
        ),
        q,
        k,
        v,
    )
    dq_uf, dk_uf, dv_uf = vjp_uf(do)
    chex.assert_tree_all_finite((out_uf, dq_uf, dk_uf, dv_uf))
    chex.assert_trees_all_close(out_uf, 0.0, atol=1e-7)
    chex.assert_trees_all_close(dq_uf, 0.0, atol=1e-7)
    chex.assert_trees_all_close(dk_uf, 0.0, atol=1e-7)
    chex.assert_trees_all_close(dv_uf, 0.0, atol=1e-7)


if __name__ == '__main__':
  absltest.main()
