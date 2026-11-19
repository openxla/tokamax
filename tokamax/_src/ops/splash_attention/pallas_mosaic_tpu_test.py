# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Tests for Pallas TPU Splash Attention operator wrapper."""

import functools
import typing
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import pallas_mosaic_tpu
from tokamax._src.ops.splash_attention import pallas_mosaic_tpu_vjp as sp_vjp
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib

PallasTpuSplash = pallas_mosaic_tpu.PallasMosaicTpuSplashAttention
ReferenceSplash = base.SplashAttention


class PallasMosaicTpuSplashAttentionTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only tested on TPU.")
    super().setUp()

  def _test_attention(
      self,
      q,
      k,
      v,
      do,
      mask=base.CAUSAL_MASK,
      is_mqa=False,
      attn_logits_soft_cap=None,
      segment_ids=None,
      sinks=None,
      config=None,
      vjp_config=None,
  ):
    kwargs = dict(
        mask=mask,
        segment_ids=segment_ids,
        sinks=sinks,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    vjp = (
        None
        if vjp_config is None
        else sp_vjp.PallasMosaicTpuSplashAttentionVjp(config=vjp_config)
    )
    attention_impl = PallasTpuSplash(config=config, vjp=vjp)
    reference_impl = ReferenceSplash()

    @jax.jit
    def f_base(query, key, value, dout):
      primals, f_vjp = jax.vjp(
          functools.partial(reference_impl, **kwargs),
          query,
          key,
          value,
      )
      return primals, f_vjp(dout)

    @jax.jit
    def f(query, key, value, dout):
      primals, f_vjp = jax.vjp(
          functools.partial(attention_impl, **kwargs),
          query,
          key,
          value,
      )
      return primals, f_vjp(dout)

    out_base, (dq_base, dk_base, dv_base) = f_base(q, k, v, do)
    out, (dq, dk, dv) = f(q, k, v, do)

    atol = 0.05 if attn_logits_soft_cap else 0.1
    with self.subTest("output"):
      chex.assert_trees_all_close(out, out_base, atol=atol)

    atol = 0.35 if attn_logits_soft_cap else 1.5
    with self.subTest("dq"):
      chex.assert_trees_all_close(dq, dq_base, atol=atol)

    atol = 0.3 if attn_logits_soft_cap else 1.2
    with self.subTest("dk"):
      chex.assert_trees_all_close(dk, dk_base, atol=atol)

    atol = 0.05 if attn_logits_soft_cap else 0.15
    with self.subTest("dv"):
      chex.assert_trees_all_close(dv, dv_base, atol=atol)

  @parameterized.parameters(
      (4, 4, 256, 256, 64, False),
      (4, 2, 256, 256, 64, False),
      (8, 2, 128, 128, 64, False),
      (4, 1, 256, 256, 64, False),
      (4, 1, 256, 256, 64, True),
      (4, 1, 256, 256, 64, True, None, False),
      (2, 2, 128, 128, 64, False, 30.0),
      (4, 2, 128, 128, 64, False, 30.0),
      (4, 2, 128, 256, 64, False),
      (4, 2, 256, 128, 64, False),
  )
  def test_pallas_mosaic_tpu_vs_reference(
      self,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      kv_seq_len,
      head_dim,
      is_mqa,
      attn_logits_soft_cap=None,
      mqa_2d=True,
  ):
    q = jax.ShapeDtypeStruct(
        (num_q_heads, q_seq_len, head_dim), dtype=jnp.float32
    )

    if is_mqa and mqa_2d:
      k = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype=jnp.float32)
      v = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype=jnp.float32)
    else:
      k = jax.ShapeDtypeStruct(
          (num_kv_heads, kv_seq_len, head_dim), dtype=jnp.float32
      )
      v = jax.ShapeDtypeStruct(
          (num_kv_heads, kv_seq_len, head_dim), dtype=jnp.float32
      )

    mask = base.CAUSAL_MASK

    do = jax.ShapeDtypeStruct(
        (num_q_heads, q_seq_len, head_dim), dtype=jnp.float32
    )
    q, k, v, do = numerics.random_initialize((q, k, v, do))

    self._test_attention(
        q,
        k,
        v,
        do,
        mask=mask,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

  @parameterized.parameters(1, 2, 4)
  def test_mask(self, num_heads):
    head_dim = 32
    q_seq_len = 128
    kv_seq_len = 128

    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)

    mask = mask_lib.CausalMask(shape=(q_seq_len, kv_seq_len))

    q, k, v, do = numerics.random_initialize((q, k, v, do))

    self._test_attention(q, k, v, do, mask=mask)

  @parameterized.parameters(
      (0, False),
      (16, False),
      (128, False),
      (256, False),
      (16, True),
      (256, True),
  )
  def test_causal_mask_offset(self, offset, use_base_mask):
    head_dim = 64
    num_heads = 2
    q_seq_len = 256
    kv_seq_len = 256
    dtype = jnp.float32

    query = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)
    key = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    value = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    dout = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)

    if use_base_mask:
      mask = base.Mask(is_causal=True, offset=offset)
    else:
      mask = mask_lib.CausalMask(shape=(q_seq_len, kv_seq_len), offset=offset)

    query, key, value, dout = numerics.random_initialize(
        (query, key, value, dout)
    )

    self._test_attention(query, key, value, dout, mask=mask)

  def test_custom_config(self):
    q_seq_len, kv_seq_len, head_dim = 128, 128, 64
    num_heads = 2
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)

    mask = base.CAUSAL_MASK

    q, k, v, do = numerics.random_initialize((q, k, v, do))

    config = pallas_mosaic_tpu.Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        k_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        v_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
        qk_diag_skip=True,
        qk_diag_grid=2,
        sv_diag_skip=True,
    )
    vjp_config = sp_vjp.Config(
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        use_base2_exp=True,
    )
    self._test_attention(
        q, k, v, do, mask=mask, config=config, vjp_config=vjp_config
    )

  @parameterized.parameters(
      (512, 128, False, 1),
      (1024, 128, False, 3),
      (1024, 256, True, 3),
  )
  def test_custom_config_diag_band_skip(
      self, seq_len, bkv_compute, fwd_band_skip, dq_reduction_steps
  ):
    q_seq_len, kv_seq_len, head_dim = seq_len, seq_len, 64
    num_heads = 2
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim), dtype)

    mask = base.CAUSAL_MASK

    q, k, v, do = numerics.random_initialize((q, k, v, do))

    config = pallas_mosaic_tpu.Config(
        block_q=256,
        block_kv=256,
        block_kv_compute=bkv_compute,
        q_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        k_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        v_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
        qk_diag_skip=True,
        qk_diag_grid=2,
        sv_diag_skip=False,
        diag_band_skip=fwd_band_skip,
    )
    vjp_config = sp_vjp.Config(
        block_q_dkv=256,
        block_kv_dkv=256,
        block_kv_dkv_compute=256,
        use_base2_exp=True,
        dq_reduction_steps=dq_reduction_steps,
        qk_diag_skip=True,
        qk_diag_grid=2,
        diag_band_skip=True,
    )
    self._test_attention(
        q, k, v, do, mask=mask, config=config, vjp_config=vjp_config
    )

  def test_fwd_diag_band_skip_config_validation(self):
    base_kwargs = dict(
        block_q=256,
        block_kv=256,
        q_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        k_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        v_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
    )
    with self.assertRaisesRegex(ValueError, "qk_diag_skip=True"):
      pallas_mosaic_tpu.Config(
          **base_kwargs,
          block_kv_compute=256,
          qk_diag_skip=False,
          diag_band_skip=True,
      )
    with self.assertRaisesRegex(ValueError, "square forward blocks"):
      pallas_mosaic_tpu.Config(
          **base_kwargs,
          block_kv_compute=128,
          qk_diag_skip=True,
          diag_band_skip=True,
      )
    with self.assertRaisesRegex(ValueError, "multiple of NUM_LANES"):
      pallas_mosaic_tpu.Config(
          **base_kwargs,
          block_kv_compute=256,
          qk_diag_skip=True,
          qk_diag_grid=4,
          diag_band_skip=True,
      )
    with self.assertRaisesRegex(ValueError, "multiple of NUM_LANES"):
      sp_vjp.Config(
          block_q_dkv=256,
          block_kv_dkv=256,
          block_kv_dkv_compute=256,
          qk_diag_skip=True,
          qk_diag_grid=4,
          diag_band_skip=True,
      )

  def test_autotune_configs(self):
    head_dim = 64
    q_seq_len = 256
    kv_seq_len = 256
    num_q_heads = 4
    num_kv_heads = 4
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    q, k, v, do = numerics.random_initialize((q, k, v, do))
    mask = base.CAUSAL_MASK

    attention_fn = PallasTpuSplash()
    bound_args = attention_fn.bind(q, k, v, mask=mask)
    configs = attention_fn._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)

    for config in configs:
      with self.subTest(f"{config=}"):
        self._test_attention(
            q,
            k,
            v,
            do,
            mask=mask,
            config=config,
        )

  def test_autotune_vjp(self):
    head_dim = 64
    q_seq_len = 256
    kv_seq_len = 256
    num_q_heads = 4
    num_kv_heads = 4
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    q, k, v, do = numerics.random_initialize((q, k, v, do))
    mask = base.CAUSAL_MASK

    residuals = (
        jnp.zeros((num_q_heads, q_seq_len), dtype=jnp.float32),
        jnp.zeros((num_q_heads, q_seq_len), dtype=jnp.float32),
    )
    out = jnp.zeros((num_q_heads, q_seq_len, head_dim), dtype=jnp.float32)

    attention_fn = PallasTpuSplash()
    vjp_op = typing.cast(
        sp_vjp.PallasMosaicTpuSplashAttentionVjp, attention_fn.vjp
    )
    bound_args = vjp_op.bind(
        residuals,
        out,
        do,
        q,
        k,
        v,
        mask=mask,
    )
    configs = vjp_op._get_autotuning_configs(bound_args)
    self.assertNotEmpty(configs)

    for config in configs:
      with self.subTest(f"{config=}"):
        self._test_attention(
            q,
            k,
            v,
            do,
            mask=mask,
            vjp_config=config,
        )

  def test_dropout(self):
    seq_len = 256
    head_dim = seq_len
    num_heads = 4
    dropout_rate = 0.25
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), dtype)
    q, k = numerics.random_initialize((q, k))
    q = q * (head_dim**-0.5)

    eye = jnp.broadcast_to(
        jnp.eye(seq_len, dtype=dtype), (num_heads, seq_len, head_dim)
    )
    v = eye
    do = eye
    mask = base.FULL_MASK
    prng_key = jax.random.key(42)

    config = pallas_mosaic_tpu.Config(
        block_q=128,
        block_kv=256,
        block_kv_compute=128,
        dropout_block_q=128,
        dropout_block_kv=128,
        q_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        k_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        v_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
    )
    vjp_config = sp_vjp.Config(
        block_q_dkv=256,
        block_kv_dkv=256,
        block_kv_dkv_compute=256,
        dropout_block_q=128,
        dropout_block_kv=128,
        use_base2_exp=True,
    )
    vjp = sp_vjp.PallasMosaicTpuSplashAttentionVjp(config=vjp_config)
    attention_impl = PallasTpuSplash(config=config, vjp=vjp)
    reference_impl = ReferenceSplash()

    def make_step(impl):
      @jax.jit
      def step(query, key, value, dout, rng):
        primals, f_vjp = jax.vjp(
            functools.partial(
                impl,
                mask=mask,
                dropout_rate=dropout_rate,
                prng_key=rng,
            ),
            query,
            key,
            value,
        )
        return primals, f_vjp(dout)

      return step

    f_pallas = make_step(attention_impl)
    f_base = make_step(reference_impl)

    out_pallas, (dq_pallas, dk_pallas, dv_pallas) = f_pallas(
        q, k, v, do, prng_key
    )
    out_pallas2, (dq_pallas2, dk_pallas2, dv_pallas2) = f_pallas(
        q, k, v, do, prng_key
    )
    out_diff_key, _ = f_pallas(q, k, v, do, jax.random.key(99))
    out_base, (_, _, dv_base) = f_base(q, k, v, do, prng_key)

    chex.assert_trees_all_equal(
        (out_pallas, dq_pallas, dk_pallas, dv_pallas),
        (out_pallas2, dq_pallas2, dk_pallas2, dv_pallas2),
    )
    self.assertGreater(float(jnp.abs(out_pallas - out_diff_key).max()), 1e-2)

    dropped_pallas = out_pallas == 0.0
    dropped_base = out_base == 0.0
    self.assertAlmostEqual(
        float(jnp.mean(dropped_pallas)), dropout_rate, delta=1e-2
    )
    self.assertAlmostEqual(
        float(jnp.mean(dropped_base)), dropout_rate, delta=1e-2
    )

    chex.assert_trees_all_equal(
        dv_pallas == 0.0, jnp.swapaxes(dropped_pallas, -1, -2)
    )
    chex.assert_trees_all_equal(
        dv_base == 0.0, jnp.swapaxes(dropped_base, -1, -2)
    )

    both_kept = (~dropped_pallas) & (~dropped_base)
    both_kept_t = jnp.swapaxes(both_kept, -1, -2)
    chex.assert_trees_all_close(
        jnp.where(both_kept, out_pallas, 0.0),
        jnp.where(both_kept, out_base, 0.0),
        atol=1e-3,
    )
    chex.assert_trees_all_close(
        jnp.where(both_kept_t, dv_pallas, 0.0),
        jnp.where(both_kept_t, dv_base, 0.0),
        atol=1e-3,
    )

  @parameterized.parameters(
      (("x", None, None), False),
      ((None, "x", None), False),
      (("x", None, None), True),
      ((None, "x", None), True),
  )
  def test_sharded(self, spec_tuple, is_mqa):
    num_devices = min(jax.device_count(), 2)
    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ("x",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*spec_tuple)
    )

    num_q_heads, q_seq_len, head_dim = 4, 512, 64
    num_kv_heads, kv_seq_len = (1, 512) if is_mqa else (4, 512)
    dtype = jnp.float32

    q_s = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    k_s = (
        jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype)
        if is_mqa
        else jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    )
    v_s = k_s
    do_s = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    q, k, v, do = numerics.random_initialize((q_s, k_s, v_s, do_s), seed=21)

    kwargs = dict(
        mask=base.CAUSAL_MASK,
        is_mqa=is_mqa,
    )
    attention_impl = PallasTpuSplash()

    out_base, f_vjp_base = jax.vjp(
        functools.partial(attention_impl, **kwargs), q, k, v
    )
    dq_base, dk_base, dv_base = f_vjp_base(do)

    out, f_vjp = jax.vjp(
        functools.partial(attention_impl, q_sharding=sharding, **kwargs),
        q,
        k,
        v,
    )
    dq, dk, dv = f_vjp(do)

    chex.assert_trees_all_close(out, out_base, atol=1e-4)
    chex.assert_trees_all_close(dq, dq_base, atol=1e-3)
    chex.assert_trees_all_close(dk, dk_base, atol=1e-3)
    chex.assert_trees_all_close(dv, dv_base, atol=1e-4)

  @parameterized.parameters(64, 128)
  def test_seq_minor_layouts_with_default_vjp(self, head_dim):
    num_q_heads, num_kv_heads, seq_len = 4, 2, 256
    q_s = jax.ShapeDtypeStruct((num_q_heads, seq_len, head_dim), jnp.bfloat16)
    k_s = jax.ShapeDtypeStruct((num_kv_heads, seq_len, head_dim), jnp.bfloat16)
    q, k, v, do = numerics.random_initialize((q_s, k_s, k_s, q_s), seed=51)
    seq_minor = pallas_mosaic_tpu.QKVLayout.SEQ_MINOR
    config = pallas_mosaic_tpu.Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_experimental_scheduler=True,
    )
    op_seq = PallasTpuSplash(config=config)
    assert isinstance(op_seq.vjp, sp_vjp.PallasMosaicTpuSplashAttentionVjp)
    self.assertEqual(op_seq.vjp.q_layout, seq_minor)
    self.assertEqual(op_seq.vjp.k_layout, seq_minor)
    self.assertEqual(op_seq.vjp.v_layout, seq_minor)
    self._test_attention(q, k, v, do, config=config)

  @parameterized.parameters("scalar", "1d", "batched_2d", "const")
  def test_max_logit_and_config_options(self, mode):
    batch_shape = (2,) if mode == "batched_2d" else ()
    num_q_heads, num_kv_heads, seq_len, head_dim = 4, 2, 256, 64
    q_s = jax.ShapeDtypeStruct(
        (*batch_shape, num_q_heads, seq_len, head_dim), jnp.float32
    )
    k_s = jax.ShapeDtypeStruct(
        (*batch_shape, num_kv_heads, seq_len, head_dim), jnp.float32
    )
    q, k, v, do = numerics.random_initialize((q_s, k_s, k_s, q_s), seed=63)
    q = q * (head_dim**-0.5)

    max_logit_const = 10.0 if mode == "const" else None
    if mode == "scalar":
      max_logit_value = 10.0
    elif mode == "1d":
      max_logit_value = jnp.array([8.0, 9.0, 10.0, 11.0], dtype=jnp.float32)
    elif mode == "batched_2d":
      max_logit_value = jnp.array(
          [[8.0, 9.0, 10.0, 11.0], [8.5, 9.5, 10.5, 11.5]], dtype=jnp.float32
      )
    else:
      max_logit_value = None

    config = pallas_mosaic_tpu.Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        k_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        v_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
        max_logit_const=max_logit_const,
        fuse_reciprocal=False,
        residual_checkpoint_name="context",
    )
    op_pallas = PallasTpuSplash(config=config)
    op_ref = ReferenceSplash()

    out_fwd_only = op_pallas(
        q, k, v, mask=base.CAUSAL_MASK, max_logit_value=max_logit_value
    )
    _, res_default = PallasTpuSplash()(
        q, k, v, mask=base.CAUSAL_MASK, return_residuals=True
    )
    out_res, res_fixed = op_pallas(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        max_logit_value=max_logit_value,
        return_residuals=True,
    )
    expected_val = 10.0 if max_logit_const is not None else max_logit_value
    expected_max = jnp.broadcast_to(
        jnp.asarray(expected_val, dtype=jnp.float32)[..., None],
        (*batch_shape, num_q_heads, seq_len),
    )
    chex.assert_trees_all_close(res_fixed[0], expected_max, atol=1e-4)
    self.assertFalse(jnp.allclose(res_default[0], res_fixed[0], atol=1e-2))

    out, vjp_fn = jax.vjp(
        lambda q_, k_, v_: op_pallas(
            q_, k_, v_, mask=base.CAUSAL_MASK, max_logit_value=max_logit_value
        ),
        q,
        k,
        v,
    )
    out_base, vjp_base = jax.vjp(
        lambda q_, k_, v_: op_ref(q_, k_, v_, mask=base.CAUSAL_MASK), q, k, v
    )
    dq, dk, dv = vjp_fn(do)
    dq_base, dk_base, dv_base = vjp_base(do)

    chex.assert_trees_all_close(out_fwd_only, out_base, atol=0.05)
    chex.assert_trees_all_close(out_res, out_base, atol=0.05)
    chex.assert_trees_all_close(out, out_base, atol=0.05)
    chex.assert_trees_all_close(dq, dq_base, atol=0.2, rtol=0.02)
    chex.assert_trees_all_close(dk, dk_base, atol=0.2, rtol=0.02)
    chex.assert_trees_all_close(dv, dv_base, atol=0.1, rtol=0.02)

    if mode == "scalar":
      cfg_both = pallas_mosaic_tpu.Config(
          block_q=128,
          block_kv=128,
          block_kv_compute=128,
          q_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
          k_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
          v_layout=pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR,
          use_experimental_scheduler=True,
          max_logit_const=10.0,
      )
      with self.assertRaisesRegex(ValueError, "max_logit_const"):
        PallasTpuSplash(config=cfg_both)(
            q, k, v, mask=base.CAUSAL_MASK, max_logit_value=10.0
        )

  def test_full_covering_local_mask(self):
    s = jax.ShapeDtypeStruct((2, 128, 64), jnp.float32)
    q, k, v, do = numerics.random_initialize((s, s, s, s), seed=73)
    local_full = mask_lib.LocalMask(
        shape=(128, 128), window_size=(255, 255), offset=0
    )
    self._test_attention(q, k, v, do, mask=local_full)

  @parameterized.product(
      fuse_reciprocal=[True, False],
      use_base2_exp=[True, False],
  )
  def test_max_logit_zero_denom_and_underflow(
      self, fuse_reciprocal, use_base2_exp
  ):
    s = jax.ShapeDtypeStruct((4, 128, 64), jnp.float32)
    q, k, v, do = numerics.random_initialize((s, s, s, s), seed=79)
    q = q * 0.125
    head_minor = pallas_mosaic_tpu.QKVLayout.HEAD_DIM_MINOR
    cfg = pallas_mosaic_tpu.Config(
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
    op_pallas = PallasTpuSplash(config=cfg)
    op_ref = ReferenceSplash()

    row0_mask = jnp.ones((128, 128), dtype=jnp.bool_).at[0, :].set(False)
    with warnings.catch_warnings():
      warnings.simplefilter("error", UserWarning)
      out_res, (_, lse) = op_pallas(
          q, k, v, mask=row0_mask, max_logit_value=10, return_residuals=True
      )
      out, vjp_fn = jax.vjp(
          lambda q_, k_, v_: op_pallas(
              q_, k_, v_, mask=row0_mask, max_logit_value=10
          ),
          q,
          k,
          v,
      )
    dq, dk, dv = vjp_fn(do)
    self.assertTrue(bool(jnp.all(jnp.isneginf(lse[:, 0]))))
    chex.assert_tree_all_finite((out_res, out, dq, dk, dv))
    chex.assert_trees_all_close(out[:, 0, :], 0.0, atol=1e-7)
    chex.assert_trees_all_close(dq[:, 0, :], 0.0, atol=1e-7)

    out_base, vjp_base = jax.vjp(
        lambda q_, k_, v_: op_ref(
            q_, k_, v_, mask=row0_mask, max_logit_value=10.0
        ),
        q,
        k,
        v,
    )
    dq_base, dk_base, dv_base = vjp_base(do)
    chex.assert_trees_all_close(out, out_base, atol=0.05)
    chex.assert_trees_all_close(dq, dq_base, atol=0.2, rtol=0.02)
    chex.assert_trees_all_close(dk, dk_base, atol=0.2, rtol=0.02)
    chex.assert_trees_all_close(dv, dv_base, atol=0.1, rtol=0.02)

    sinks = jnp.full((4,), -200.0, dtype=jnp.float32)
    out_uf, vjp_uf = jax.vjp(
        lambda q_, k_, v_, s_: op_pallas(
            q_,
            k_,
            v_,
            mask=base.FULL_MASK,
            sinks=s_,
            max_logit_value=120.0,
        ),
        q,
        k,
        v,
        sinks,
    )
    dq_uf, dk_uf, dv_uf, dsinks_uf = vjp_uf(do)
    chex.assert_tree_all_finite((out_uf, dq_uf, dk_uf, dv_uf, dsinks_uf))
    chex.assert_trees_all_close(out_uf, 0.0, atol=1e-7)
    chex.assert_trees_all_close(dq_uf, 0.0, atol=1e-7)
    chex.assert_trees_all_close(dk_uf, 0.0, atol=1e-7)
    chex.assert_trees_all_close(dv_uf, 0.0, atol=1e-7)
    chex.assert_trees_all_close(dsinks_uf, 0.0, atol=1e-7)


if __name__ == "__main__":
  absltest.main()
