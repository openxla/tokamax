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

"""Tests for base SplashAttention operator and reference implementation."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import reference
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib


class SplashAttentionBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.op = base.SplashAttention()

  @parameterized.parameters(
      (4, 4, 128, 128, 64, False, "causal", None),
      (2, 2, 128, 128, 64, False, "full", None),
      (4, 2, 128, 128, 64, False, "causal", 30.0),
      (4, 1, 128, 128, 64, True, "causal", None),
      (4, 4, 128, 128, 64, False, "legacy_causal", None),
  )
  def test_equivalence_with_reference(
      self,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      kv_seq_len,
      head_dim,
      is_mqa,
      mask_type,
      attn_logits_soft_cap,
  ):
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype=dtype)
    if is_mqa:
      k = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype=dtype)
      v = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype=dtype)
    else:
      k = jax.ShapeDtypeStruct(
          (num_kv_heads, kv_seq_len, head_dim), dtype=dtype
      )
      v = jax.ShapeDtypeStruct(
          (num_kv_heads, kv_seq_len, head_dim), dtype=dtype
      )

    q, k, v = numerics.random_initialize((q, k, v))

    if mask_type == "causal":
      mask = base.CAUSAL_MASK
      mask_array = mask.as_array(q_seq_len, kv_seq_len)
    elif mask_type == "legacy_causal":
      mask = mask_lib.CausalMask(shape=(q_seq_len, kv_seq_len))
      mask_array = jnp.asarray(mask[:, :])
    else:
      mask_array = jnp.ones((q_seq_len, kv_seq_len), dtype=jnp.bool_)
      mask = mask_array

    out_op = self.op(
        q,
        k,
        v,
        mask,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    out_ref = reference.attention_reference(
        q=q,
        k=k,
        v=v,
        mask=mask_array,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    chex.assert_trees_all_close(out_op, out_ref, atol=1e-5)

  def test_with_segment_ids(self):
    q_seq_len, kv_seq_len, head_dim = 128, 128, 64
    num_q_heads = 2

    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype=dtype)
    k = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype=dtype)
    v = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype=dtype)

    q, k, v = numerics.random_initialize((q, k, v))

    mask = base.CAUSAL_MASK
    q_seg = jnp.array([0] * 64 + [1] * 64, dtype=jnp.int32)
    kv_seg = jnp.array([0] * 64 + [1] * 64, dtype=jnp.int32)
    segment_ids = reference.SegmentIds(q=q_seg, kv=kv_seg)

    out_op = self.op(q, k, v, mask, segment_ids=segment_ids)
    out_ref = reference.attention_reference(
        q=q,
        k=k,
        v=v,
        mask=mask.as_array(q_seq_len, kv_seq_len),
        segment_ids=segment_ids,
        is_mqa=False,
    )

    chex.assert_trees_all_close(out_op, out_ref, atol=1e-5)

  def test_validation_errors(self):
    q = jnp.ones((4, 128, 64))
    k = jnp.ones((3, 128, 64))
    v = jnp.ones((3, 128, 64))
    mask = jnp.ones((128, 128), dtype=jnp.bool_)

    # Query heads q/k/v mismatch
    with self.assertRaisesRegex(ValueError, "divisible by num_kv_heads"):
      self.op(q, k, v, mask, is_mqa=False)

    # Dropout rate out of bounds
    with self.assertRaisesRegex(ValueError, "dropout_rate must be in"):
      self.op(q, q, q, mask, dropout_rate=1.0, is_mqa=False)

    with self.assertRaisesRegex(ValueError, "dropout_rate must be in"):
      self.op(q, q, q, mask, dropout_rate=-0.1, is_mqa=False)

    # Unsupported mask type
    with self.assertRaises(TypeError):
      self.op(q, q, q, mask="invalid_mask", is_mqa=False)

    # return_residuals success
    out, res = self.op(q, q, q, mask, return_residuals=True, is_mqa=False)
    self.assertEqual(out.shape, (4, 128, 64))
    self.assertLen(res, 2)
    self.assertEqual(res[0].shape, (4, 128))
    self.assertEqual(res[1].shape, (4, 128))

    # VJP return_residuals not supported
    assert self.op.vjp is not None
    with self.assertRaises(NotImplementedError):
      self.op.vjp(res, out, out, q, q, q, mask, return_residuals=True)

  def test_symbolic_mask_direct(self):
    q = jnp.ones((2, 128, 64))
    full_mask = mask_lib.FullMask((128, 128))
    ba_full = self.op.bind(q, q, q, mask=full_mask, is_mqa=False)
    self.assertIsInstance(ba_full.arguments["mask"], base.Mask)
    self.assertFalse(ba_full.arguments["mask"].is_causal)
    out_full = self.op(q, q, q, full_mask, is_mqa=False)
    self.assertEqual(out_full.shape, (2, 128, 64))

    causal_mask = mask_lib.CausalMask((128, 128))
    ba_causal = self.op.bind(q, q, q, mask=causal_mask, is_mqa=False)
    self.assertIsInstance(ba_causal.arguments["mask"], base.Mask)
    self.assertTrue(ba_causal.arguments["mask"].is_causal)
    out_causal = self.op(q, q, q, causal_mask, is_mqa=False)
    self.assertEqual(out_causal.shape, (2, 128, 64))

  def test_mask_as_array_composite(self):
    dense = jnp.ones((8, 8), dtype=jnp.bool_)
    m = base.Mask(bool_mask=dense, is_causal=True)
    arr = m.as_array(8, 8)
    expected = base.CAUSAL_MASK.as_array(8, 8)
    chex.assert_trees_all_close(arr, expected, atol=0)

  def test_vmap_support(self):
    batch_size = 3
    num_heads = 2
    seq_len = 32
    head_dim = 16

    dtype = jnp.float32

    q = jax.ShapeDtypeStruct(
        (batch_size, num_heads, seq_len, head_dim), dtype=dtype
    )
    k = jax.ShapeDtypeStruct(
        (batch_size, num_heads, seq_len, head_dim), dtype=dtype
    )
    v = jax.ShapeDtypeStruct(
        (batch_size, num_heads, seq_len, head_dim), dtype=dtype
    )

    q, k, v = numerics.random_initialize((q, k, v))

    mask = base.CAUSAL_MASK

    vmapped_op = jax.vmap(self.op, in_axes=(0, 0, 0, None))
    out = vmapped_op(q, k, v, mask)
    self.assertEqual(out.shape, (batch_size, num_heads, seq_len, head_dim))

  @parameterized.parameters(
      (4, 4, 128, 128, 64, False, None),
      (4, 1, 128, 128, 64, True, None),
      (2, 2, 128, 128, 64, False, 30.0),
  )
  def test_vjp_equivalence(
      self,
      num_q_heads,
      num_kv_heads,
      q_seq_len,
      kv_seq_len,
      head_dim,
      is_mqa,
      attn_logits_soft_cap,
  ):
    dtype = jnp.float32
    q = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype=dtype)
    if is_mqa:
      k = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype=dtype)
      v = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype=dtype)
    else:
      k = jax.ShapeDtypeStruct(
          (num_kv_heads, kv_seq_len, head_dim), dtype=dtype
      )
      v = jax.ShapeDtypeStruct(
          (num_kv_heads, kv_seq_len, head_dim), dtype=dtype
      )
    do = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype=dtype)
    q, k, v, do = numerics.random_initialize((q, k, v, do))

    mask = base.CAUSAL_MASK

    kwargs = dict(
        mask=mask,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    out, f_vjp = jax.vjp(
        functools.partial(self.op, **kwargs),
        q,
        k,
        v,
    )
    dq, dk, dv = f_vjp(do)

    out_ref, stats_ref = reference.attention_reference(
        q=q,
        k=k,
        v=v,
        mask=mask.as_array(q_seq_len, kv_seq_len),
        is_mqa=is_mqa,
        save_residuals=True,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    dq_ref, dk_ref, dv_ref, _ = reference.attention_reference_vjp(
        do=do,
        q=q,
        k=k if not (is_mqa and k.ndim == 3) else k[0],
        v=v if not (is_mqa and v.ndim == 3) else v[0],
        mask=mask.as_array(q_seq_len, kv_seq_len),
        segment_ids=None,
        sinks=None,
        o=out_ref,
        logsumexp=stats_ref["logsumexp"],
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    if is_mqa and k.ndim == 3:
      dk_ref = dk_ref.reshape(k.shape)
      dv_ref = dv_ref.reshape(v.shape)

    chex.assert_trees_all_close(out, out_ref, atol=1e-5)
    chex.assert_trees_all_close(dq, dq_ref, atol=1e-5)
    chex.assert_trees_all_close(dk, dk_ref, atol=1e-5)
    chex.assert_trees_all_close(dv, dv_ref, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
