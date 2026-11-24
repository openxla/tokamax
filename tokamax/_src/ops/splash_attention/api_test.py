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

import contextlib
import dataclasses
import functools
from typing import Any, cast
from unittest import mock
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src import hlo_utils
from tokamax._src import numerics
from tokamax._src.ops.splash_attention import api
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import reference
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib

try:
  from tokamax._src import mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
  from tokamax._src.ops.splash_attention import pallas_mosaic_tpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
except ImportError:
  mosaic_tpu = None
  pallas_mosaic_tpu = None

try:
  from tokamax._src.ops.attention import api as attention_api  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import,missing-module-attribute]
  from tokamax._src.ops.attention import base as attention_base  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import,missing-module-attribute]
  from tokamax._src.ops.attention import pallas_mosaic_tpu_common as attention_tpu_common  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import,missing-module-attribute]
  from tokamax.ops import tpu as tpu_ops  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import,missing-module-attribute]
except ImportError:
  attention_api = None
  attention_base = None
  attention_tpu_common = None
  tpu_ops = None


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

  def test_is_causal_and_segment_ids(self):
    num_heads, seq_len, head_dim = 4, 128, 64
    q_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    k_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    v_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    q, k, v = numerics.random_initialize((q_s, k_s, v_s), seed=5)

    half_seq = seq_len // 2
    q_seg = jnp.array([0] * half_seq + [1] * (seq_len - half_seq), jnp.int32)

    out = api.splash_attention(
        q, k, v, segment_ids=q_seg, is_causal=True, implementation="base"
    )
    ref_out = base.SplashAttention()(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        segment_ids=base.SegmentIds(q=q_seg, kv=q_seg),
    )
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
      ((2,), 4, 4, False, False),
      ((2,), 4, 1, True, True),
      ((2,), 4, 1, True, False),
      ((2, 3), 4, 2, False, False),
  )
  def test_batched_splash_attention(
      self, batch_shape, num_q_heads, num_kv_heads, is_mqa, mqa_2d
  ):
    seq_len, head_dim = 64, 32
    dtype = jnp.float32

    q_s = jax.ShapeDtypeStruct(
        (*batch_shape, num_q_heads, seq_len, head_dim), dtype
    )
    if mqa_2d:
      k_s = jax.ShapeDtypeStruct((*batch_shape, seq_len, head_dim), dtype)
      v_s = jax.ShapeDtypeStruct((*batch_shape, seq_len, head_dim), dtype)
    else:
      k_s = jax.ShapeDtypeStruct(
          (*batch_shape, num_kv_heads, seq_len, head_dim), dtype
      )
      v_s = jax.ShapeDtypeStruct(
          (*batch_shape, num_kv_heads, seq_len, head_dim), dtype
      )

    q, k, v = numerics.random_initialize((q_s, k_s, v_s), seed=8)
    out = api.splash_attention(
        q, k, v, mask=base.CAUSAL_MASK, is_mqa=is_mqa, implementation="base"
    )
    chex.assert_shape(out, (*batch_shape, num_q_heads, seq_len, head_dim))

    ref_fn = functools.partial(
        reference.attention_reference,
        mask=base.CAUSAL_MASK.as_array(seq_len, seq_len),
        is_mqa=is_mqa,
    )
    for _ in batch_shape:
      ref_fn = jax.vmap(ref_fn)
    ref_out = ref_fn(
        q=q,
        k=k if not (is_mqa and not mqa_2d) else jnp.squeeze(k, axis=-3),
        v=v if not (is_mqa and not mqa_2d) else jnp.squeeze(v, axis=-3),
    )
    chex.assert_trees_all_close(out, ref_out, atol=1e-5)

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

  @parameterized.product(
      implementation=["base", "mosaic_tpu"],
      shard_axis=[0, 1],
  )
  def test_splash_attention_sharding(self, implementation, shard_axis):
    if implementation == "mosaic_tpu" and jax.default_backend() != "tpu":
      self.skipTest("mosaic_tpu only runs on TPU.")

    num_devices = min(jax.device_count(), 2)
    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ("x",))
    spec: list[str | None] = [None, None, None]
    spec[shard_axis] = "x"
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*spec)
    )

    num_heads = 4
    seq_len = 512
    head_dim = 64
    dtype = jnp.float32

    q_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), dtype)
    k_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), dtype)
    v_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), dtype)
    q, k, v = numerics.random_initialize((q_s, k_s, v_s), seed=99)

    ref_out = api.splash_attention(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        implementation=implementation,
    )
    out = api.splash_attention(
        q,
        k,
        v,
        mask=base.CAUSAL_MASK,
        implementation=implementation,
        q_sharding=sharding,
    )
    chex.assert_trees_all_close(out, ref_out, atol=1e-4, rtol=1e-4)

  @parameterized.product(
      mode=["scalar", "per_head"],
      implementation=["base", "mosaic_tpu"],
  )
  def test_max_logit_value(self, mode, implementation):
    if implementation == "mosaic_tpu" and (
        "mosaic_tpu" not in api.IMPLEMENTATIONS
        or jax.default_backend() != "tpu"
    ):
      self.skipTest("mosaic_tpu only runs on TPU.")

    num_heads, seq_len, head_dim = 4, 128, 64
    q_s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    q, k, v, do = numerics.random_initialize((q_s, q_s, q_s, q_s), seed=105)
    q = q * (head_dim**-0.5)
    max_logit_value = (
        4.0
        if mode == "scalar"
        else jnp.array([3.5, 4.0, 4.5, 5.0], dtype=jnp.float32)
    )

    fn = lambda q_, k_, v_: api.splash_attention(
        q_,
        k_,
        v_,
        mask=base.CAUSAL_MASK,
        max_logit_value=max_logit_value,
        implementation=implementation,
    )
    ref_fn = lambda q_, k_, v_: base.SplashAttention()(
        q_, k_, v_, mask=base.CAUSAL_MASK
    )
    # Default float32 matmuls use bf16 on TPU and tf32 on GPU, which amplifies
    # tiny log-sum-exp shift differences in the reference implementation.
    with contextlib.ExitStack() as stack:
      if implementation == "base":
        stack.enter_context(jax.default_matmul_precision("highest"))
      out, vjp_fn = jax.vjp(fn, q, k, v)
      out_ref, vjp_ref = jax.vjp(ref_fn, q, k, v)
      if jax.default_backend() == "cpu":
        out_tol = dict(atol=1e-5, rtol=1e-5)
        vjp_tol = dict(atol=2e-5, rtol=1e-5)
      elif implementation == "mosaic_tpu":
        out_tol = dict(atol=0.05, rtol=0.02)
        vjp_tol = dict(atol=0.2, rtol=0.02)
      else:
        # Non-CPU reference VJP recomputes exp(logits - lse) through hardware
        # exp/log approximations (~1.4e-3 diff observed on TPU v6e).
        out_tol = dict(atol=1e-4, rtol=1e-4)
        vjp_tol = dict(atol=2e-3, rtol=2e-3)
      chex.assert_trees_all_close(out, out_ref, **out_tol)
      chex.assert_trees_all_close(vjp_fn(do), vjp_ref(do), **vjp_tol)

      # Non-shift-invariant check: a shift far below actual logits overflows
      # float32 `exp(logits - max_logit_value)` only on the targeted head(s).
      if mode == "scalar":
        out_bad = api.splash_attention(
            q,
            k,
            v,
            mask=base.CAUSAL_MASK,
            max_logit_value=-100.0,
            implementation=implementation,
        )
        self.assertFalse(bool(jnp.all(jnp.isfinite(out_bad))))
      else:
        bad_mlv = jnp.array([3.5, -100.0, 4.5, 5.0], dtype=jnp.float32)
        devices = jax.devices()
        n_dev = min(len(devices), num_heads)
        while num_heads % n_dev != 0:
          n_dev -= 1
        mesh = jax.sharding.Mesh(devices[:n_dev], ("x",))
        q_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x", None, None)
        )
        mlv_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x")
        )
        out_bad = api.splash_attention(
            jax.device_put(q, q_sharding),
            jax.device_put(k, q_sharding),
            jax.device_put(v, q_sharding),
            mask=base.CAUSAL_MASK,
            max_logit_value=jax.device_put(bad_mlv, mlv_sharding),
            implementation=implementation,
            q_sharding=q_sharding,
        )
        self.assertFalse(bool(jnp.all(jnp.isfinite(out_bad[1]))))
        chex.assert_trees_all_close(
            out_bad[jnp.array([0, 2, 3])],
            out_ref[jnp.array([0, 2, 3])],
            **out_tol,
        )

  def test_full_covering_mask_fallback_and_int_max_logit(self):
    q_s = jax.ShapeDtypeStruct((4, 128, 64), jnp.float32)
    q, k, v, do = numerics.random_initialize((q_s, q_s, q_s, q_s), seed=107)
    q = q * 0.125
    full_causal = mask_lib.CausalMask(shape=(128, 128), offset=128)
    impl = (
        ("mosaic_tpu", "base")
        if "mosaic_tpu" in api.IMPLEMENTATIONS
        else "base"
    )
    with contextlib.ExitStack() as stack:
      if jax.default_backend() != "tpu":
        stack.enter_context(jax.default_matmul_precision("highest"))
      with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out, vjp_fn = jax.vjp(
            lambda q_, k_, v_: api.splash_attention(
                q_,
                k_,
                v_,
                mask=full_causal,
                max_logit_value=10,
                implementation=impl,
            ),
            q,
            k,
            v,
        )
        dq, dk, dv = vjp_fn(do)
      out_ref, vjp_ref = jax.vjp(
          lambda q_, k_, v_: api.splash_attention(
              q_, k_, v_, mask=full_causal, implementation="base"
          ),
          q,
          k,
          v,
      )
      dq_ref, dk_ref, dv_ref = vjp_ref(do)
    tol = (
        dict(atol=1e-4, rtol=1e-4)
        if jax.default_backend() != "tpu"
        else dict(atol=0.05, rtol=0.02)
    )
    vjp_tol = (
        dict(atol=1e-4, rtol=1e-4)
        if jax.default_backend() != "tpu"
        else dict(atol=0.2, rtol=0.02)
    )
    chex.assert_trees_all_close(out, out_ref, **tol)
    chex.assert_trees_all_close(dq, dq_ref, **vjp_tol)
    chex.assert_trees_all_close(dk, dk_ref, **vjp_tol)
    chex.assert_trees_all_close(dv, dv_ref, **vjp_tol)


class SplashAttentionVirtualTpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if pallas_mosaic_tpu is None or "mosaic_tpu" not in api.IMPLEMENTATIONS:
      self.skipTest("mosaic_tpu backend is not linked in this build.")

  def test_has_mosaic_tpu_support(self):
    assert mosaic_tpu is not None
    cpu_device = jax.devices("cpu")[0]
    tpu_v7 = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    tpu_v5 = jax.sharding.AbstractDevice(
        device_kind="TPU v5 lite", num_cores=1, platform="tpu"
    )
    tpu_v4 = jax.sharding.AbstractDevice(
        device_kind="TPU v4", num_cores=2, platform="tpu"
    )
    with mock.patch.object(
        backend, "get_default_device", return_value=cpu_device
    ):
      self.assertFalse(mosaic_tpu.has_mosaic_tpu_support(cpu_device))
      self.assertFalse(mosaic_tpu.has_mosaic_tpu_support(None))
      self.assertTrue(mosaic_tpu.has_mosaic_tpu_support(tpu_v7))
      self.assertTrue(mosaic_tpu.has_mosaic_tpu_support(tpu_v5))
      self.assertFalse(mosaic_tpu.has_mosaic_tpu_support(tpu_v4))

      tpu_v7_none_cores = jax.sharding.AbstractDevice(
          device_kind="TPU7x", num_cores=None, platform="tpu"
      )
      mesh_v7_none_cores = jax.sharding.AbstractMesh(
          (2,),
          ("x",),
          axis_types=(jax.sharding.AxisType.Auto,),
          abstract_device=tpu_v7_none_cores,
      )
      with jax.sharding.use_abstract_mesh(mesh_v7_none_cores):
        self.assertTrue(mosaic_tpu.has_mosaic_tpu_support(tpu_v7_none_cores))
        q_sds = jax.ShapeDtypeStruct((4, 128, 128), jnp.bfloat16)
        lowered = (
            jax.jit(
                lambda q, k, v: api.splash_attention(
                    q, k, v, is_causal=True, implementation="mosaic_tpu"
                )
            )
            .trace(q_sds, q_sds, q_sds)
            .lower(lowering_platforms=("tpu",))
        )
        self.assertIn("tpu_custom_call", lowered.as_text())

  @parameterized.parameters("mosaic_tpu", None, "callable")
  def test_virtual_tpu_lowering_on_cpu(self, impl_mode):
    assert pallas_mosaic_tpu is not None
    implementation = (
        pallas_mosaic_tpu.PallasMosaicTpuSplashAttention()
        if impl_mode == "callable"
        else impl_mode
    )
    cpu_device = jax.devices("cpu")[0]
    tpu_v7 = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    abstract_mesh = jax.sharding.AbstractMesh(
        (2,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=tpu_v7,
    )
    sharding = jax.sharding.NamedSharding(
        abstract_mesh, jax.sharding.PartitionSpec("x", None, None)
    )
    q_sds = jax.ShapeDtypeStruct((4, 128, 128), jnp.bfloat16, sharding=sharding)
    k_sds = jax.ShapeDtypeStruct((4, 128, 128), jnp.bfloat16, sharding=sharding)
    v_sds = jax.ShapeDtypeStruct((4, 128, 128), jnp.bfloat16, sharding=sharding)

    def loss_fn(q, k, v):
      return jnp.sum(
          api.splash_attention(
              q,
              k,
              v,
              is_causal=True,
              implementation=implementation,
              q_sharding=sharding,
          )
      )

    with mock.patch.object(
        backend, "get_default_device", return_value=cpu_device
    ):
      lowered = (
          jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))
          .trace(q_sds, k_sds, v_sds)
          .lower(lowering_platforms=("tpu",))
      )
      self.assertIn("tpu_custom_call", lowered.as_text())

  def test_virtual_tpu_v4_rejection_and_fallback(self):
    cpu_device = jax.devices("cpu")[0]
    tpu_v4 = jax.sharding.AbstractDevice(
        device_kind="TPU v4", num_cores=2, platform="tpu"
    )
    mesh_v4 = jax.sharding.AbstractMesh(
        (2,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=tpu_v4,
    )
    q_sds = jax.ShapeDtypeStruct((4, 128, 128), jnp.bfloat16)

    with mock.patch.object(
        backend, "get_default_device", return_value=cpu_device
    ):
      with jax.sharding.use_abstract_mesh(mesh_v4):
        with self.assertRaisesRegex(
            NotImplementedError, "Not supported on TPU v4"
        ):
          jax.jit(
              lambda q, k, v: api.splash_attention(
                  q, k, v, is_causal=True, implementation="mosaic_tpu"
              )
          ).trace(q_sds, q_sds, q_sds)

        lowered_default = (
            jax.jit(
                lambda q, k, v: api.splash_attention(
                    q, k, v, is_causal=True, implementation=None
                )
            )
            .trace(q_sds, q_sds, q_sds)
            .lower(lowering_platforms=("tpu",))
        )
        self.assertNotIn("tpu_custom_call", lowered_default.as_text())

  def test_tpu_exports_and_vjp_layout_propagation(self):
    assert tpu_ops is not None
    seq_minor = tpu_ops.QKVLayout.SEQ_MINOR
    head_minor = tpu_ops.QKVLayout.HEAD_DIM_MINOR

    splash_cfg = tpu_ops.SplashAttentionConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=head_minor,
        use_experimental_scheduler=True,
        use_base2_exp=False,
    )
    splash_op = tpu_ops.SplashAttention(config=splash_cfg)
    self.assertIsInstance(splash_op.vjp, tpu_ops.SplashAttentionVjp)
    assert isinstance(splash_op.vjp, tpu_ops.SplashAttentionVjp)
    self.assertEqual(splash_op.vjp.q_layout, seq_minor)
    self.assertEqual(splash_op.vjp.k_layout, seq_minor)
    self.assertEqual(splash_op.vjp.v_layout, head_minor)
    self.assertFalse(splash_op.vjp.use_base2_exp)

    splash_cfg2 = dataclasses.replace(splash_cfg, v_layout=seq_minor)
    splash_op2 = splash_op.replace(config=splash_cfg2)
    assert isinstance(splash_op2.vjp, tpu_ops.SplashAttentionVjp)
    self.assertEqual(splash_op2.vjp.v_layout, seq_minor)

    custom_vjp = tpu_ops.SplashAttentionVjp(q_layout=head_minor)
    splash_custom = tpu_ops.SplashAttention(config=splash_cfg, vjp=custom_vjp)
    splash_custom2 = splash_custom.replace(config=splash_cfg2)
    self.assertIs(splash_custom2.vjp, custom_vjp)

    flash_cfg = tpu_ops.FlashAttentionConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_experimental_scheduler=True,
        use_base2_exp=False,
    )
    flash_op = tpu_ops.FlashAttention(config=flash_cfg)
    self.assertIsInstance(flash_op.vjp, tpu_ops.FlashAttentionVjp)
    assert isinstance(flash_op.vjp, tpu_ops.FlashAttentionVjp)
    self.assertEqual(flash_op.vjp.q_layout, seq_minor)
    self.assertEqual(flash_op.vjp.k_layout, seq_minor)
    self.assertEqual(flash_op.vjp.v_layout, seq_minor)
    self.assertFalse(flash_op.vjp.use_base2_exp)

  def test_dot_product_attention_max_logit_value_and_analytic_local_mask(self):
    assert (
        attention_api is not None
        and attention_base is not None
        and attention_tpu_common is not None
        and pallas_mosaic_tpu is not None
        and tpu_ops is not None
    )

    q_s = jax.ShapeDtypeStruct((2, 32, 4, 16), jnp.float32)
    q, k, v, do = numerics.random_initialize((q_s, q_s, q_s, q_s), seed=19)
    mlv = jnp.array([8.0, 9.0, 10.0, 11.0], dtype=jnp.float32)

    with self.assertRaisesRegex(
        NotImplementedError, "`max_logit_value` is not supported"
    ):
      attention_api.dot_product_attention(
          q, k, v, max_logit_value=mlv, implementation="xla_chunked"
      )

    out_fb, vjp_fb = jax.vjp(
        lambda q_, k_, v_: attention_api.dot_product_attention(
            q_,
            k_,
            v_,
            is_causal=True,
            max_logit_value=mlv,
            precision=jax.lax.Precision.HIGHEST,
            implementation=("xla_chunked", "xla"),
        ),
        q,
        k,
        v,
    )
    out_ref, vjp_ref = jax.vjp(
        lambda q_, k_, v_: attention_api.dot_product_attention(
            q_,
            k_,
            v_,
            is_causal=True,
            precision=jax.lax.Precision.HIGHEST,
            implementation="xla",
        ),
        q,
        k,
        v,
    )
    vjp_tol = (
        dict(atol=1e-5, rtol=1e-5)
        if jax.default_backend() == "cpu"
        else dict(atol=1e-4, rtol=5e-4)
    )
    chex.assert_trees_all_close(out_fb, out_ref, atol=1e-5, rtol=1e-4)
    chex.assert_trees_all_close(vjp_fb(do), vjp_ref(do), **vjp_tol)

    dpa = attention_base.DotProductAttention()
    _, res_default = dpa(q, k, v, is_causal=True, return_residuals=True)
    _, res_fixed = dpa(
        q, k, v, is_causal=True, max_logit_value=mlv, return_residuals=True
    )
    chex.assert_trees_all_close(
        res_fixed[0],
        jnp.broadcast_to(mlv[None, :, None], (2, 4, 32)),
        atol=1e-6,
    )
    self.assertFalse(jnp.allclose(res_default[0], res_fixed[0], atol=1e-2))

    splash_cfg = pallas_mosaic_tpu.splash_attention_kernel.SplashConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
    )
    local_mask = attention_base.Mask(
        causal_mask=attention_base.CausalMask(offset=4, window_size=64)
    )
    splash_maker, splash_mask = attention_tpu_common.build_splash_kernel(
        mask=local_mask,
        splash_config=splash_cfg,
        q_seq_len=256,
        kv_seq_len=256,
        is_mqa=False,
        save_residuals=True,
    )
    self.assertIsInstance(splash_mask, mask_lib.LocalMask)
    kernel = splash_maker(splash_mask)
    self.assertIsNone(kernel.fwd_mask_info.partial_mask_blocks)
    assert kernel.dkv_mask_info is not None
    self.assertIsNone(kernel.dkv_mask_info.partial_mask_blocks)

    neg_offset_mask = attention_base.Mask(
        causal_mask=attention_base.CausalMask(offset=-1, window_size=64)
    )
    with self.assertRaisesRegex(NotImplementedError, "non-negative"):
      attention_tpu_common.check_inputs_support(
          q,
          k,
          v,
          bias=None,
          logits_dtype=jnp.float32,
          dropout_mask=None,
          dropout_rate=0.0,
          paging_info=None,
          mask=neg_offset_mask,
      )

    # Trailing query rows with empty sliding windows must reject/fallback.
    q_cross = jnp.ones((1, 256, 4, 16), dtype=jnp.bfloat16)
    k_cross = jnp.ones((1, 128, 4, 16), dtype=jnp.bfloat16)
    empty_row_mask = attention_base.Mask(
        causal_mask=attention_base.CausalMask(offset=0, window_size=65)
    )
    with self.assertRaisesRegex(NotImplementedError, "all-masked query rows"):
      attention_tpu_common.check_inputs_support(
          q_cross,
          k_cross,
          k_cross,
          bias=None,
          logits_dtype=jnp.float32,
          dropout_mask=None,
          dropout_rate=0.0,
          paging_info=None,
          mask=empty_row_mask,
      )
    out_cross = attention_api.dot_product_attention(
        q_cross,
        k_cross,
        k_cross,
        is_causal=True,
        local_window_size=(64, 0),
        implementation=("mosaic", "xla"),
    )
    self.assertTrue(bool(jnp.all(out_cross[:, 192:, :, :] == 0.0)))

    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(devices, ("x",))
    k_only_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", None, None)
    )
    with self.assertRaisesRegex(ValueError, "q_sharding must be specified"):
      api.splash_attention(
          jnp.ones((4, 128, 64), dtype=jnp.float32),
          jnp.ones((4, 128, 64), dtype=jnp.float32),
          jnp.ones((4, 128, 64), dtype=jnp.float32),
          k_sharding=k_only_sharding,
      )
    k_only_dpa_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", None, None, None)
    )
    with self.assertRaisesRegex(ValueError, "q_sharding.*must be specified"):
      attention_api.dot_product_attention(
          q, k, v, k_sharding=k_only_dpa_sharding
      )

  def test_vjp_autotuning_cache_filters_layouts_and_base2(self):
    assert (
        tpu_ops is not None
        and pallas_mosaic_tpu is not None
        and attention_base is not None
    )
    seq_minor = tpu_ops.QKVLayout.SEQ_MINOR
    head_minor = tpu_ops.QKVLayout.HEAD_DIM_MINOR
    cpu_device = jax.devices("cpu")[0]
    tpu_v7 = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh_v7 = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=tpu_v7,
    )

    sp_default = tpu_ops.SplashAttentionVjp()
    sp_seq = tpu_ops.SplashAttentionVjp(
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_base2_exp=False,
    )
    sp_default_cache = sp_default.get_autotuning_cache("TPU7x")
    self.assertNotEmpty(sp_default_cache)
    for data in sp_default_cache.values():
      for cfg in data:
        self.assertEqual(cfg.q_layout, head_minor)
        self.assertEqual(cfg.k_layout, head_minor)
        self.assertEqual(cfg.v_layout, head_minor)
        self.assertTrue(cfg.use_base2_exp)
    self.assertEmpty(sp_seq.get_autotuning_cache("TPU7x"))

    # Verify filtered view writes through without clobbering other configs.
    sample_key, sample_data = next(iter(sp_default_cache.items()))
    sample_cfg, sample_ev = next(iter(sample_data.items()))
    seq_cfg = dataclasses.replace(
        sample_cfg,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_base2_exp=False,
    )
    seq_data = type(sample_data)({seq_cfg: sample_ev})
    shared_raw_cache = {sample_key: sample_data}
    filter_fn = pallas_mosaic_tpu.pallas_mosaic_tpu_vjp.filter_autotuning_cache
    filtered_default = filter_fn(shared_raw_cache, sp_default._matches_config)
    filtered_seq = filter_fn(shared_raw_cache, sp_seq._matches_config)
    self.assertIsNone(filtered_seq.pop(sample_key, None))
    self.assertIn(sample_key, shared_raw_cache)

    filtered_seq[sample_key] = seq_data
    self.assertIn(sample_cfg, shared_raw_cache[sample_key])
    self.assertIn(seq_cfg, shared_raw_cache[sample_key])

    filtered_seq.clear()
    self.assertNotIn(sample_key, filtered_seq)
    self.assertEqual(set(shared_raw_cache[sample_key].keys()), {sample_cfg})

    filtered_seq.update({sample_key: seq_data})
    self.assertIn(seq_cfg, shared_raw_cache[sample_key])
    filtered_default.pop(sample_key)
    self.assertEqual(set(shared_raw_cache[sample_key].keys()), {seq_cfg})
    del filtered_seq[sample_key]
    self.assertNotIn(sample_key, shared_raw_cache)

    fa_default = tpu_ops.FlashAttentionVjp()
    fa_seq = tpu_ops.FlashAttentionVjp(
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_base2_exp=False,
    )
    fa_default_cache = fa_default.get_autotuning_cache("TPU7x")
    self.assertNotEmpty(fa_default_cache)
    for data in fa_default_cache.values():
      for cfg in data:
        self.assertEqual(cfg.q_layout, head_minor)
        self.assertTrue(cfg.use_base2_exp)
    self.assertEmpty(fa_seq.get_autotuning_cache("TPU7x"))

    # Verify cached TPU7x shapes use cache for default VJP, heuristics for seq.
    with (
        mock.patch.object(
            backend, "get_default_device", return_value=cpu_device
        ),
        jax.sharding.use_abstract_mesh(mesh_v7),
    ):
      sp_res = (
          jax.ShapeDtypeStruct((64, 8192), jnp.float32),
          jax.ShapeDtypeStruct((64, 8192), jnp.float32),
      )
      sp_q = jax.ShapeDtypeStruct((64, 8192, 128), jnp.bfloat16)
      sp_k = jax.ShapeDtypeStruct((8, 8192, 128), jnp.bfloat16)
      fa_res = (
          jax.ShapeDtypeStruct((32, 32, 4096), jnp.float32),
          jax.ShapeDtypeStruct((32, 32, 4096), jnp.float32),
      )
      fa_q = jax.ShapeDtypeStruct((32, 4096, 32, 128), jnp.bfloat16)
      fa_k = jax.ShapeDtypeStruct((32, 4096, 8, 128), jnp.bfloat16)
      resolved_configs = {}

      def resolve_configs(s_res, s_q, s_k, f_res, f_q, f_k):
        resolved_configs["sp_default"] = sp_default.bind(
            s_res,
            s_q,
            s_q,
            s_q,
            s_k,
            s_k,
            mask=base.CAUSAL_MASK,
            is_mqa=False,
            mask_value=base.DEFAULT_MASK_VALUE,
        ).default_config
        resolved_configs["sp_seq"] = sp_seq.bind(
            s_res,
            s_q,
            s_q,
            s_q,
            s_k,
            s_k,
            mask=base.CAUSAL_MASK,
            is_mqa=False,
            mask_value=base.DEFAULT_MASK_VALUE,
        ).default_config
        fa_kwargs = dict(
            precision=(
                jax.lax.DotAlgorithmPreset.DEFAULT,
                jax.lax.DotAlgorithmPreset.DEFAULT,
            ),
            logits_dtype=jnp.dtype(jnp.float32),
            logits_scale=128**-0.5,
            bias=None,
            logits_soft_cap=None,
            mask=attention_base.Mask(is_causal=True),
            dropout_mask=None,
            dropout_rate=0.0,
            paging_info=None,
            q_indices=None,
            k_indices=None,
            normalize_output=True,
        )
        resolved_configs["fa_default"] = fa_default.bind(
            f_res, f_q, f_q, f_q, f_k, f_k, **fa_kwargs
        ).default_config
        resolved_configs["fa_seq"] = fa_seq.bind(
            f_res, f_q, f_q, f_q, f_k, f_k, **fa_kwargs
        ).default_config
        return s_q

      jax.eval_shape(resolve_configs, sp_res, sp_q, sp_k, fa_res, fa_q, fa_k)

    sp_default_cfg = resolved_configs["sp_default"]
    self.assertEqual(sp_default_cfg.q_layout, head_minor)
    self.assertTrue(sp_default_cfg.use_base2_exp)

    sp_seq_cfg = resolved_configs["sp_seq"]
    self.assertEqual(sp_seq_cfg.q_layout, seq_minor)
    self.assertEqual(sp_seq_cfg.k_layout, seq_minor)
    self.assertEqual(sp_seq_cfg.v_layout, seq_minor)
    self.assertFalse(sp_seq_cfg.use_base2_exp)

    fa_default_cfg = resolved_configs["fa_default"]
    self.assertEqual(fa_default_cfg.q_layout, head_minor)
    self.assertTrue(fa_default_cfg.use_base2_exp)

    fa_seq_cfg = resolved_configs["fa_seq"]
    self.assertEqual(fa_seq_cfg.q_layout, seq_minor)
    self.assertEqual(fa_seq_cfg.k_layout, seq_minor)
    self.assertEqual(fa_seq_cfg.v_layout, seq_minor)
    self.assertFalse(fa_seq_cfg.use_base2_exp)

    self.assertNotIn(
        "__class__",
        sp_default.get_autotuning_cache.__code__.co_freevars,
    )
    self.assertNotIn(
        "__class__",
        fa_default.get_autotuning_cache.__code__.co_freevars,
    )
    self.assertNotIn(
        "__class__",
        pallas_mosaic_tpu.PallasMosaicTpuSplashAttention.replace.__code__.co_freevars,
    )
    with (
        warnings.catch_warnings(),
        mock.patch.object(
            backend, "get_default_device", return_value=cpu_device
        ),
        jax.sharding.use_abstract_mesh(mesh_v7),
    ):
      warnings.simplefilter("error", UserWarning)
      ba_int_box = {}

      def bind_int(s_q, s_k):
        ba_int_box[
            "ba"
        ] = pallas_mosaic_tpu.PallasMosaicTpuSplashAttention().bind(
            s_q, s_k, s_k, max_logit_value=10
        )
        return s_q

      jax.eval_shape(bind_int, sp_q, sp_k)
      self.assertEqual(ba_int_box["ba"].kwargs["max_logit_value"], 10.0)
      self.assertIsInstance(ba_int_box["ba"].kwargs["max_logit_value"], float)

  def test_fold_q_seq_heads_virtual_tpu_and_fallback(self):
    assert (
        attention_api is not None
        and attention_base is not None
        and attention_tpu_common is not None
        and pallas_mosaic_tpu is not None
        and tpu_ops is not None
    )

    seq_minor = tpu_ops.QKVLayout.SEQ_MINOR
    with self.assertRaisesRegex(ValueError, "fold_q_seq_heads"):
      tpu_ops.SplashAttentionConfig(
          block_q=128,
          block_kv=128,
          block_kv_compute=128,
          q_layout=seq_minor,
          k_layout=seq_minor,
          v_layout=seq_minor,
          use_experimental_scheduler=True,
          fold_q_seq_heads=True,
          qk_diag_skip=True,
      )
    with self.assertRaisesRegex(ValueError, "fold_q_seq_heads"):
      tpu_ops.SplashAttentionVjpConfig(
          block_q_dkv=128,
          block_kv_dkv=128,
          block_kv_dkv_compute=128,
          fold_q_seq_heads=True,
          qk_diag_skip=True,
      )

    folded_splash_cfg = tpu_ops.SplashAttentionConfig(
        block_q=256,
        block_kv=128,
        block_kv_compute=128,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_experimental_scheduler=True,
        fold_q_seq_heads=True,
    )
    unfolded_splash_cfg = dataclasses.replace(
        folded_splash_cfg, block_q=128, fold_q_seq_heads=False
    )
    folded_splash_op = tpu_ops.SplashAttention(config=folded_splash_cfg)
    assert isinstance(folded_splash_op.vjp, tpu_ops.SplashAttentionVjp)
    self.assertTrue(folded_splash_op.vjp.fold_q_seq_heads)

    folded_flash_cfg = tpu_ops.FlashAttentionConfig(
        block_q=256,
        block_kv=128,
        block_kv_compute=128,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_experimental_scheduler=True,
        fold_q_seq_heads=True,
    )
    unfolded_flash_cfg = dataclasses.replace(
        folded_flash_cfg, block_q=128, fold_q_seq_heads=False
    )
    folded_flash_op = tpu_ops.FlashAttention(config=folded_flash_cfg)
    assert isinstance(folded_flash_op.vjp, tpu_ops.FlashAttentionVjp)
    self.assertTrue(folded_flash_op.vjp.fold_q_seq_heads)

    # Verify VJP cache filtering rejects fold_q_seq_heads=False entries.
    sp_folded_vjp = tpu_ops.SplashAttentionVjp(fold_q_seq_heads=True)
    fa_folded_vjp = tpu_ops.FlashAttentionVjp(fold_q_seq_heads=True)
    self.assertEmpty(sp_folded_vjp.get_autotuning_cache("TPU7x"))
    self.assertEmpty(fa_folded_vjp.get_autotuning_cache("TPU7x"))
    cpu_dev = jax.devices("cpu")[0]
    tpu_dev7 = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh_v7 = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=tpu_dev7,
    )
    with (
        mock.patch.object(backend, "get_default_device", return_value=cpu_dev),
        jax.sharding.use_abstract_mesh(mesh_v7),
    ):
      folded_resolved = {}

      def resolve_folded_vjp(s_res, s_q, s_k, f_res, f_q, f_k):
        folded_resolved["sp"] = sp_folded_vjp.bind(
            s_res,
            s_q,
            s_q,
            s_q,
            s_k,
            s_k,
            mask=base.CAUSAL_MASK,
            is_mqa=False,
            mask_value=base.DEFAULT_MASK_VALUE,
        ).default_config
        folded_resolved["fa"] = fa_folded_vjp.bind(
            f_res,
            f_q,
            f_q,
            f_q,
            f_k,
            f_k,
            precision=(
                jax.lax.DotAlgorithmPreset.DEFAULT,
                jax.lax.DotAlgorithmPreset.DEFAULT,
            ),
            logits_dtype=jnp.dtype(jnp.float32),
            logits_scale=128**-0.5,
            bias=None,
            logits_soft_cap=None,
            mask=attention_base.Mask(is_causal=True),
            dropout_mask=None,
            dropout_rate=0.0,
            paging_info=None,
            q_indices=None,
            k_indices=None,
            normalize_output=True,
        ).default_config
        return s_q

      jax.eval_shape(
          resolve_folded_vjp,
          (
              jax.ShapeDtypeStruct((64, 8192), jnp.float32),
              jax.ShapeDtypeStruct((64, 8192), jnp.float32),
          ),
          jax.ShapeDtypeStruct((64, 8192, 128), jnp.bfloat16),
          jax.ShapeDtypeStruct((8, 8192, 128), jnp.bfloat16),
          (
              jax.ShapeDtypeStruct((32, 32, 4096), jnp.float32),
              jax.ShapeDtypeStruct((32, 32, 4096), jnp.float32),
          ),
          jax.ShapeDtypeStruct((32, 4096, 32, 128), jnp.bfloat16),
          jax.ShapeDtypeStruct((32, 4096, 8, 128), jnp.bfloat16),
      )
    self.assertTrue(folded_resolved["sp"].fold_q_seq_heads)
    self.assertTrue(folded_resolved["fa"].fold_q_seq_heads)

    # Verify analytic LocalMask + 2x GQA mask keeps partial_mask_blocks=None.
    splash_cfg = pallas_mosaic_tpu.splash_attention_kernel.SplashConfig(
        block_q=256,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=256,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
    )
    local_mask = attention_base.Mask(
        causal_mask=attention_base.CausalMask(offset=0, window_size=64)
    )
    splash_maker, splash_mask = attention_tpu_common.build_splash_kernel(
        mask=local_mask,
        splash_config=splash_cfg,
        q_seq_len=128,
        kv_seq_len=128,
        is_mqa=False,
        save_residuals=True,
        fold_q_seq_heads=True,
    )
    self.assertEqual(splash_mask.shape, (256, 128))
    kernel = splash_maker(splash_mask)
    self.assertIsNone(kernel.fwd_mask_info.partial_mask_blocks)
    assert kernel.dkv_mask_info is not None
    self.assertIsNone(kernel.dkv_mask_info.partial_mask_blocks)

    # Negative control: virtual-TPU HLO reflects folded [2,2,256,64] Q tensor.
    cpu_device = jax.devices("cpu")[0]
    tpu_v7 = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=tpu_v7,
    )
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", None, None, None)
    )
    q_sp = jax.ShapeDtypeStruct(
        (2, 4, 128, 64), jnp.bfloat16, sharding=sharding
    )
    k_sp = jax.ShapeDtypeStruct(
        (2, 2, 128, 64), jnp.bfloat16, sharding=sharding
    )
    q_fa = jax.ShapeDtypeStruct(
        (2, 128, 4, 64), jnp.bfloat16, sharding=sharding
    )
    k_fa = jax.ShapeDtypeStruct(
        (2, 128, 2, 64), jnp.bfloat16, sharding=sharding
    )
    local_sp_mask = mask_lib.LocalMask(
        (128, 128), window_size=(63, 0), offset=0
    )
    attn_api = attention_api

    def lower_splash(impl):
      fn = lambda q_, k_, v_: jnp.sum(
          api.splash_attention(
              q_,
              k_,
              v_,
              mask=local_sp_mask,
              max_logit_value=8.0,
              implementation=impl,
          )
      )
      return (
          jax.jit(jax.value_and_grad(fn, argnums=(0, 1, 2)))
          .trace(q_sp, k_sp, k_sp)
          .lower(lowering_platforms=("tpu",))
          .as_text()
      )

    def lower_flash(impl):
      fn = lambda q_, k_, v_: jnp.sum(
          attn_api.dot_product_attention(
              q_,
              k_,
              v_,
              is_causal=True,
              local_window_size=(64, 0),
              max_logit_value=8.0,
              implementation=impl,
          )
      )
      return (
          jax.jit(jax.value_and_grad(fn, argnums=(0, 1, 2)))
          .trace(q_fa, k_fa, k_fa)
          .lower(lowering_platforms=("tpu",))
          .as_text()
      )

    with mock.patch.object(
        backend, "get_default_device", return_value=cpu_device
    ):
      hlo_sp_folded = lower_splash(folded_splash_op)
      hlo_sp_unfolded = lower_splash(
          tpu_ops.SplashAttention(config=unfolded_splash_cfg)
      )
      self.assertIn("tpu_custom_call", hlo_sp_folded)
      self.assertIn("tensor<2x2x256x64xbf16>", hlo_sp_folded)
      self.assertNotIn("tensor<2x2x256x64xbf16>", hlo_sp_unfolded)

      hlo_fa_folded = lower_flash(folded_flash_op)
      hlo_fa_unfolded = lower_flash(
          tpu_ops.FlashAttention(config=unfolded_flash_cfg)
      )
      self.assertIn("tpu_custom_call", hlo_fa_folded)
      self.assertIn("tensor<2x2x256x64xbf16>", hlo_fa_folded)
      self.assertNotIn("tensor<2x2x256x64xbf16>", hlo_fa_unfolded)

    # Causal + window_size `base.Mask` must not emit pure-causal diagonal-skip
    # autotuning candidates on either forward or VJP SplashAttention.
    with (
        mock.patch.object(
            backend, "get_default_device", return_value=cpu_device
        ),
        jax.sharding.use_abstract_mesh(mesh),
    ):
      q_tune = cast(Any, jax.ShapeDtypeStruct((4, 1024, 128), jnp.bfloat16))
      splash_default = tpu_ops.SplashAttention()
      assert isinstance(splash_default.vjp, tpu_ops.SplashAttentionVjp)
      ba_pure_causal = splash_default.bind(
          q_tune, q_tune, q_tune, mask=base.Mask(is_causal=True)
      )
      ba_window_causal = splash_default.bind(
          q_tune,
          q_tune,
          q_tune,
          mask=base.Mask(is_causal=True, window_size=(64, 0)),
      )
      ba_local = splash_default.bind(
          q_tune,
          q_tune,
          q_tune,
          mask=mask_lib.LocalMask((1024, 1024), window_size=(64, 0), offset=0),
      )
      self.assertIsNone(ba_local.arguments["mask"].bool_mask)
      self.assertEqual(ba_local.arguments["mask"].window_size, (64, 0))
      fwd_pure = splash_default._get_autotuning_configs(  # pylint: disable=protected-access
          ba_pure_causal
      )
      fwd_window = splash_default._get_autotuning_configs(  # pylint: disable=protected-access
          ba_window_causal
      )
      self.assertTrue(any(c.qk_diag_skip for c in fwd_pure))
      self.assertFalse(
          any(
              c.qk_diag_skip or c.sv_diag_skip or c.diag_band_skip
              for c in fwd_window
          )
      )
      vjp_window = splash_default.vjp._get_autotuning_configs(  # pylint: disable=protected-access
          splash_default.vjp.bind(**ba_window_causal.vjp_arg_spec)
      )
      self.assertFalse(
          any(c.qk_diag_skip or c.diag_band_skip for c in vjp_window)
      )

    # Unsupported fold_q_seq_heads preconditions raise NotImplementedError on
    # both SplashAttention and FlashAttention, and fall back cleanly through
    # `api.splash_attention` and `dot_product_attention`.
    q_mha_s = jax.ShapeDtypeStruct((4, 128, 64), jnp.float32)
    q_mha, k_mha, v_mha = numerics.random_initialize(
        (q_mha_s, q_mha_s, q_mha_s), seed=51
    )
    k_gqa = k_mha[:2]
    v_gqa = v_mha[:2]
    q_odd_gqa = jnp.concatenate([q_mha, q_mha[:2]], axis=0)
    q_odd_mqa = q_mha[:3]
    k_mqa_2d = k_mha[0]
    v_mqa_2d = v_mha[0]
    bool_sp_mask = jnp.ones((128, 128), dtype=jnp.bool_)

    folded_splash_bypass = folded_splash_op.replace(bypass_device_check=True)
    folded_flash_bypass = folded_flash_op.replace(bypass_device_check=True)

    sp_err_cases: tuple[
        tuple[jax.Array, jax.Array, jax.Array, dict[str, Any], str], ...
    ] = (
        (q_mha, k_mha, v_mha, {}, "fold_q_seq_heads"),
        (q_odd_gqa, k_gqa, v_gqa, {}, "fold_q_seq_heads"),
        (
            q_odd_mqa,
            k_mqa_2d,
            v_mqa_2d,
            dict(is_mqa=True),
            "fold_q_seq_heads",
        ),
        (
            q_mha,
            k_gqa,
            v_gqa,
            dict(sinks=jnp.zeros((4,), dtype=jnp.float32)),
            "sinks",
        ),
        (
            q_mha,
            k_gqa,
            v_gqa,
            dict(dropout_rate=0.1, prng_key=jax.random.PRNGKey(0)),
            "dropout",
        ),
        (
            q_mha,
            k_gqa,
            v_gqa,
            dict(max_logit_value=jnp.arange(4, dtype=jnp.float32)),
            "max_logit_value",
        ),
        (
            q_mha,
            k_gqa,
            v_gqa,
            dict(mask=base.Mask(bool_mask=bool_sp_mask)),
            "boolean",
        ),
    )
    for sp_q, sp_k, sp_v, extra_kw, err_re in sp_err_cases:
      call_kw: dict[str, Any] = dict(mask=base.CAUSAL_MASK)
      call_kw.update(extra_kw)
      with self.assertRaisesRegex(NotImplementedError, err_re):
        folded_splash_bypass(sp_q, sp_k, sp_v, **call_kw)

    q_fa_mha = jnp.swapaxes(q_mha, 0, 1)[None, ...]
    k_fa_mha = jnp.swapaxes(k_mha, 0, 1)[None, ...]
    v_fa_mha = jnp.swapaxes(v_mha, 0, 1)[None, ...]
    q_fa_odd = jnp.swapaxes(q_odd_gqa, 0, 1)[None, ...]
    k_fa_gqa = jnp.swapaxes(k_gqa, 0, 1)[None, ...]
    v_fa_gqa = jnp.swapaxes(v_gqa, 0, 1)[None, ...]
    bool_fa_mask = jnp.ones((1, 1, 128, 128), dtype=jnp.bool_)

    fa_err_cases: tuple[
        tuple[jax.Array, jax.Array, jax.Array, dict[str, Any], str], ...
    ] = (
        (
            q_fa_mha,
            k_fa_mha,
            v_fa_mha,
            dict(is_causal=True),
            "fold_q_seq_heads",
        ),
        (
            q_fa_odd,
            k_fa_gqa,
            v_fa_gqa,
            dict(is_causal=True),
            "fold_q_seq_heads",
        ),
        (
            q_fa_mha,
            k_fa_gqa,
            v_fa_gqa,
            dict(
                is_causal=True,
                max_logit_value=jnp.arange(4, dtype=jnp.float32),
            ),
            "max_logit_value",
        ),
        (
            q_fa_mha,
            k_fa_gqa,
            v_fa_gqa,
            dict(mask=bool_fa_mask),
            "boolean",
        ),
    )
    for fa_q, fa_k, fa_v, fa_extra, err_re in fa_err_cases:
      with self.assertRaisesRegex(NotImplementedError, err_re):
        folded_flash_bypass(fa_q, fa_k, fa_v, **fa_extra)

    # End-to-end API tuple fallback on virtual TPU (AOT lowering) and CPU
    # execution (`jax.value_and_grad`).
    with (
        mock.patch.object(
            backend, "get_default_device", return_value=cpu_device
        ),
        jax.sharding.use_abstract_mesh(mesh),
    ):
      q_sp_mha = jax.ShapeDtypeStruct((2, 4, 128, 64), jnp.bfloat16)
      q_fa_mha_sds = jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16)
      hlo_sp_fb = (
          jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_: jnp.sum(
                      api.splash_attention(
                          q_,
                          k_,
                          v_,
                          is_causal=True,
                          implementation=(folded_splash_op, "base"),
                      )
                  ),
                  argnums=(0, 1, 2),
              )
          )
          .trace(q_sp_mha, q_sp_mha, q_sp_mha)
          .lower(lowering_platforms=("tpu",))
          .as_text()
      )
      self.assertNotIn("tpu_custom_call", hlo_sp_fb)
      hlo_fa_fb = (
          jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_: jnp.sum(
                      attn_api.dot_product_attention(
                          q_,
                          k_,
                          v_,
                          is_causal=True,
                          implementation=(folded_flash_op, "xla"),
                      )
                  ),
                  argnums=(0, 1, 2),
              )
          )
          .trace(q_fa_mha_sds, q_fa_mha_sds, q_fa_mha_sds)
          .lower(lowering_platforms=("tpu",))
          .as_text()
      )
      self.assertNotIn("tpu_custom_call", hlo_fa_fb)

    sinks_vec = jnp.array([0.5, -0.5, 0.25, 0.0], dtype=jnp.float32)
    per_head_mlv = jnp.array([8.0, 9.0, 10.0, 11.0], dtype=jnp.float32)

    sp_fb_cases: tuple[
        tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]], ...
    ] = (
        (q_mha, k_mha, v_mha, dict(is_causal=True)),
        (q_odd_gqa, k_gqa, v_gqa, dict(is_causal=True)),
        (q_mha, k_gqa, v_gqa, dict(is_causal=True, sinks=sinks_vec)),
        (
            q_mha,
            k_gqa,
            v_gqa,
            dict(is_causal=True, max_logit_value=per_head_mlv),
        ),
        (q_mha, k_gqa, v_gqa, dict(mask=bool_sp_mask)),
    )
    for sp_q, sp_k, sp_v, sp_kw in sp_fb_cases:
      val_fb, grads_fb = jax.value_and_grad(
          lambda qq, kk, vv, kw=sp_kw: jnp.sum(
              api.splash_attention(
                  qq,
                  kk,
                  vv,
                  implementation=(folded_splash_bypass, "base"),
                  **kw,
              )
          ),
          argnums=(0, 1, 2),
      )(sp_q, sp_k, sp_v)
      val_ref, grads_ref = jax.value_and_grad(
          lambda qq, kk, vv, kw=sp_kw: jnp.sum(
              api.splash_attention(qq, kk, vv, implementation="base", **kw)
          ),
          argnums=(0, 1, 2),
      )(sp_q, sp_k, sp_v)
      chex.assert_trees_all_close(val_fb, val_ref, atol=1e-5, rtol=1e-5)
      chex.assert_trees_all_close(grads_fb, grads_ref, atol=1e-5, rtol=1e-5)

    fa_fb_cases: tuple[
        tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]], ...
    ] = (
        (q_fa_mha, k_fa_mha, v_fa_mha, dict(is_causal=True)),
        (q_fa_odd, k_fa_gqa, v_fa_gqa, dict(is_causal=True)),
        (
            q_fa_mha,
            k_fa_gqa,
            v_fa_gqa,
            dict(is_causal=True, max_logit_value=per_head_mlv),
        ),
        (q_fa_mha, k_fa_gqa, v_fa_gqa, dict(mask=bool_fa_mask)),
    )
    for fa_q, fa_k, fa_v, fa_kw in fa_fb_cases:
      val_fb, grads_fb = jax.value_and_grad(
          lambda qq, kk, vv, kw=fa_kw: jnp.sum(
              attn_api.dot_product_attention(
                  qq,
                  kk,
                  vv,
                  implementation=(folded_flash_bypass, "xla"),
                  **kw,
              )
          ),
          argnums=(0, 1, 2),
      )(fa_q, fa_k, fa_v)
      val_ref, grads_ref = jax.value_and_grad(
          lambda qq, kk, vv, kw=fa_kw: jnp.sum(
              attn_api.dot_product_attention(
                  qq, kk, vv, implementation="xla", **kw
              )
          ),
          argnums=(0, 1, 2),
      )(fa_q, fa_k, fa_v)
      chex.assert_trees_all_close(val_fb, val_ref, atol=1e-5, rtol=1e-5)
      chex.assert_trees_all_close(grads_fb, grads_ref, atol=1e-5, rtol=1e-5)

  def test_fold_q_seq_heads_non_divisible_gqa_raises_value_error(self):
    assert pallas_mosaic_tpu is not None and tpu_ops is not None
    sp_vjp = pallas_mosaic_tpu.pallas_mosaic_tpu_vjp
    for bad_q, bad_kv in ((5, 2), (4, 0)):
      with self.assertRaisesRegex(
          ValueError, "must be divisible by num_kv_heads"
      ):
        sp_vjp.check_fold_q_seq_heads(bad_q, bad_kv, False)

    seq_minor = tpu_ops.QKVLayout.SEQ_MINOR
    fwd_cfg = tpu_ops.SplashAttentionConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_experimental_scheduler=True,
        fold_q_seq_heads=True,
    )
    vjp_cfg = tpu_ops.SplashAttentionVjpConfig(
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        fold_q_seq_heads=True,
    )
    q = jnp.zeros((5, 128, 64), dtype=jnp.bfloat16)
    k = jnp.zeros((2, 128, 64), dtype=jnp.bfloat16)
    v = jnp.zeros((2, 128, 64), dtype=jnp.bfloat16)
    res = (
        jnp.zeros((5, 128), dtype=jnp.float32),
        jnp.zeros((5, 128), dtype=jnp.float32),
    )

    with self.assertRaisesRegex(
        ValueError, "must be divisible by num_kv_heads"
    ):
      pallas_mosaic_tpu.PallasMosaicTpuSplashAttention()._fwd(  # pylint: disable=protected-access
          q, k, v, mask=base.CAUSAL_MASK, config=fwd_cfg
      )
    with self.assertRaisesRegex(
        ValueError, "must be divisible by num_kv_heads"
    ):
      sp_vjp.PallasMosaicTpuSplashAttentionVjp()._fwd(  # pylint: disable=protected-access
          res, q, q, q, k, v, mask=base.CAUSAL_MASK, config=vjp_cfg
      )


if __name__ == "__main__":
  absltest.main()
