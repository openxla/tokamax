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
"""Comprehensive tests for Splash Attention API and argument specifications."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import hlo_utils
from tokamax._src import numerics
from tokamax._src.ops.experimental.tpu.splash_attention import api
from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import reference
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib


class SplashAttentionApiTest(parameterized.TestCase):

  def test_implementations_dict(self):
    self.assertIn("base", api.IMPLEMENTATIONS)
    self.assertIsInstance(api.IMPLEMENTATIONS["base"], base.SplashAttention)
    if jax.default_backend() == "tpu":
      self.assertIn("mosaic_tpu", api.IMPLEMENTATIONS)
      self.assertIsInstance(
          api.IMPLEMENTATIONS["mosaic_tpu"],
          base.SplashAttention,
      )
    elif "mosaic_tpu" in api.IMPLEMENTATIONS:
      self.assertIsInstance(
          api.IMPLEMENTATIONS["mosaic_tpu"],
          base.SplashAttention,
      )

  @parameterized.product(
      implementation=["base", ("base",), "mosaic_tpu", None],
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_splash_attention_basic(self, implementation, dtype):
    if implementation == "mosaic_tpu" and (
        "mosaic_tpu" not in api.IMPLEMENTATIONS
        or jax.default_backend() != "tpu"
    ):
      self.skipTest("mosaic_tpu only runs on TPU.")

    num_q_heads, q_seq_len, head_dim = 4, 128, 64
    num_kv_heads, kv_seq_len = 4, 128

    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)

    q, k, v = numerics.random_initialize((q, k, v), seed=0)

    out = api.splash_attention(
        q, k, v, mask=base.CAUSAL_MASK, implementation=implementation
    )
    ref_out = base.SplashAttention()(q, k, v, mask=base.CAUSAL_MASK)

    chex.assert_shape(out, (num_q_heads, q_seq_len, head_dim))
    expected_dtype = (
        dtype
        if implementation == "mosaic_tpu"
        or (implementation is None and jax.default_backend() == "tpu")
        else ref_out.dtype
    )
    chex.assert_type(out, expected_dtype)
    atol = (
        0.1
        if jax.default_backend() == "tpu" or implementation == "mosaic_tpu"
        else 1e-3
    )
    rtol = (
        0.1
        if jax.default_backend() == "tpu" or implementation == "mosaic_tpu"
        else 1e-3
    )
    chex.assert_trees_all_close(
        out.astype(jnp.float32),
        ref_out.astype(jnp.float32),
        atol=atol,
        rtol=rtol,
    )

  @parameterized.parameters(
      lambda q_len, kv_len: base.CAUSAL_MASK,
      lambda q_len, kv_len: base.FULL_MASK,
      lambda q_len, kv_len: None,
      lambda q_len, kv_len: jnp.tril(
          jnp.ones((q_len, kv_len), dtype=jnp.bool_)
      ),
      lambda q_len, kv_len: mask_lib.CausalMask(shape=(q_len, kv_len)),
      lambda q_len, kv_len: mask_lib.FullMask(_shape=(q_len, kv_len)),
      lambda q_len, kv_len: mask_lib.ChunkedCausalMask(
          shape=(q_len, kv_len), chunk_size=64
      ),
  )
  def test_masks(self, mask_fn):
    q_len, kv_len, head_dim = 128, 128, 64
    num_heads = 4
    mask = mask_fn(q_len, kv_len)

    q = jax.ShapeDtypeStruct((num_heads, q_len, head_dim), jnp.float32)
    k = jax.ShapeDtypeStruct((num_heads, kv_len, head_dim), jnp.float32)
    v = jax.ShapeDtypeStruct((num_heads, kv_len, head_dim), jnp.float32)
    q, k, v = numerics.random_initialize((q, k, v), seed=1)

    out = api.splash_attention(q, k, v, mask=mask, implementation="base")
    ref_out = base.SplashAttention()(q, k, v, mask=mask)

    chex.assert_shape(out, (num_heads, q_len, head_dim))
    chex.assert_trees_all_close(out, ref_out, atol=1e-4, rtol=1e-4)

  @parameterized.parameters(
      (4, 4, False, False),
      (4, 2, False, False),
      (4, 1, True, True),
      (4, 1, True, False),
  )
  def test_attention_variants(self, num_q_heads, num_kv_heads, is_mqa, mqa_2d):
    seq_len, head_dim = 128, 64
    q_struct = jax.ShapeDtypeStruct(
        (num_q_heads, seq_len, head_dim), jnp.float32
    )

    if mqa_2d:
      k_struct = jax.ShapeDtypeStruct((seq_len, head_dim), jnp.float32)
      v_struct = jax.ShapeDtypeStruct((seq_len, head_dim), jnp.float32)
    else:
      k_struct = jax.ShapeDtypeStruct(
          (num_kv_heads, seq_len, head_dim), jnp.float32
      )
      v_struct = jax.ShapeDtypeStruct(
          (num_kv_heads, seq_len, head_dim), jnp.float32
      )

    q, k, v = numerics.random_initialize((q_struct, k_struct, v_struct), seed=2)

    out = api.splash_attention(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        is_mqa=is_mqa,
        implementation="base",
    )
    ref_out = base.SplashAttention()(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        is_mqa=is_mqa,
    )

    chex.assert_shape(out, (num_q_heads, seq_len, head_dim))
    chex.assert_trees_all_close(out, ref_out, atol=1e-4, rtol=1e-4)

  @parameterized.parameters(
      (30.0, reference.DEFAULT_MASK_VALUE, False, False),
      (None, -1e4, False, False),
      (None, reference.DEFAULT_MASK_VALUE, True, False),
      (None, reference.DEFAULT_MASK_VALUE, False, True),
  )
  def test_optional_arguments(
      self,
      attn_logits_soft_cap,
      mask_value,
      use_sinks,
      use_segment_ids,
  ):
    num_heads, seq_len, head_dim = 4, 128, 64
    q_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    k_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    v_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    q, k, v = numerics.random_initialize((q_s, k_s, v_s), seed=3)

    sinks = None
    if use_sinks:
      sinks_s = jax.ShapeDtypeStruct((num_heads,), jnp.float32)
      sinks = numerics.random_initialize(sinks_s, seed=4)

    segment_ids = None
    if use_segment_ids:
      half_seq = seq_len // 2
      q_seg = jnp.array([0] * half_seq + [1] * (seq_len - half_seq), jnp.int32)
      segment_ids = base.SegmentIds(q=q_seg, kv=q_seg)

    out = api.splash_attention(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        segment_ids=segment_ids,
        sinks=sinks,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        implementation="base",
    )
    ref_out = base.SplashAttention()(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        segment_ids=segment_ids,
        sinks=sinks,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    chex.assert_shape(out, (num_heads, seq_len, head_dim))
    chex.assert_trees_all_close(out, ref_out, atol=1e-4, rtol=1e-4)

  def test_bound_args_and_hlo(self):
    num_heads, seq_len, head_dim = 2, 64, 32
    q_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    k_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    v_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    q, k, v = numerics.random_initialize((q_s, k_s, v_s), seed=7)

    fn = functools.partial(
        api.splash_attention, mask=base.CAUSAL_MASK, implementation="base"
    )
    args = hlo_utils.get_bound_args(fn, q, k, v)

    self.assertLen(args, 1)
    self.assertIsInstance(args[0].op, base.SplashAttention)
    self.assertEqual(
        args[0].arguments["q"].shape, (num_heads, seq_len, head_dim)
    )
    self.assertEqual(
        args[0].arguments["k"].shape, (num_heads, seq_len, head_dim)
    )
    self.assertEqual(
        args[0].arguments["v"].shape, (num_heads, seq_len, head_dim)
    )

  @parameterized.parameters(
      (4, 4, False, None),
      (4, 1, True, None),
      (2, 2, False, 30.0),
  )
  def test_mosaic_tpu_variants(
      self,
      num_q_heads,
      num_kv_heads,
      is_mqa,
      attn_logits_soft_cap,
  ):
    if (
        "mosaic_tpu" not in api.IMPLEMENTATIONS
        or jax.default_backend() != "tpu"
    ):
      self.skipTest("mosaic_tpu only runs on TPU.")

    q_seq_len, kv_seq_len, head_dim = 128, 128, 64
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    if is_mqa:
      k = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype)
      v = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype)
    else:
      k = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)
      v = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)

    q, k, v = numerics.random_initialize((q, k, v), seed=42)

    out = api.splash_attention(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
        implementation="mosaic_tpu",
    )
    ref_out = base.SplashAttention()(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    chex.assert_shape(out, (num_q_heads, q_seq_len, head_dim))
    chex.assert_trees_all_close(out, ref_out, atol=0.1, rtol=0.1)

  def test_correct_implementation_used(self):
    num_heads, seq_len, head_dim = 2, 128, 64
    q_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    k_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    v_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    q, k, v = numerics.random_initialize((q_s, k_s, v_s), seed=7)

    fn = functools.partial(api.splash_attention, mask=base.CAUSAL_MASK)
    args = hlo_utils.get_bound_args(fn, q, k, v)
    self.assertLen(args, 1)

    if jax.default_backend() == "tpu":
      mosaic_tpu_impl = type(api.IMPLEMENTATIONS["mosaic_tpu"])
      self.assertIsInstance(args[0].op, mosaic_tpu_impl)
    else:
      self.assertIsInstance(args[0].op, base.SplashAttention)

  def test_errors(self):
    num_heads, seq_len, head_dim = 2, 64, 32
    q_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    k_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    v_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    q, k, v = numerics.random_initialize((q_s, k_s, v_s), seed=10)

    # Empty sequence
    with self.assertRaisesRegex(ValueError, "empty sequence"):
      api.splash_attention(q, k, v, implementation=())

    # Unsupported string name
    with self.assertRaisesRegex(ValueError, "Unsupported implementation"):
      api.splash_attention(
          q,
          k,
          v,
          implementation="unsupported_backend",  # pyrefly: ignore[bad-argument-type]
      )

    # Wrong head counts (q heads not divisible by kv heads)
    q_inc = jax.ShapeDtypeStruct((3, seq_len, head_dim), jnp.float32)
    k_inc = jax.ShapeDtypeStruct((2, seq_len, head_dim), jnp.float32)
    v_inc = jax.ShapeDtypeStruct((2, seq_len, head_dim), jnp.float32)
    q_inc, k_inc, v_inc = numerics.random_initialize(
        (q_inc, k_inc, v_inc), seed=11
    )
    with self.assertRaises(ValueError):
      api.splash_attention(q_inc, k_inc, v_inc, implementation="base")

    # Invalid mask type
    with self.assertRaises(TypeError):
      api.splash_attention(
          q,
          k,
          v,
          mask="invalid_mask",  # pyrefly: ignore[bad-argument-type]
          implementation="base",
      )

    # When mosaic_tpu is not imported, passing it raises ValueError
    if "mosaic_tpu" not in api.IMPLEMENTATIONS:
      with self.assertRaisesRegex(
          ValueError, "Unsupported implementation: mosaic_tpu"
      ):
        api.splash_attention(q, k, v, implementation="mosaic_tpu")
    elif jax.default_backend() != "tpu":
      with self.assertRaises(NotImplementedError):
        api.splash_attention(q, k, v, implementation="mosaic_tpu")


if __name__ == "__main__":
  absltest.main()
