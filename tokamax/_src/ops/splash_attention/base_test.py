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
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import reference
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib


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
      self.op(q, q, q, mask="invalid_mask", is_mqa=False)  # pyrefly: ignore[no-matching-overload]

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
    expected = jnp.stack(
        [self.op(q[i], k[i], v[i], mask) for i in range(batch_size)]
    )
    chex.assert_trees_all_close(out, expected, atol=1e-5)

  @parameterized.parameters(
      ((2,), 4, 4, False, False),
      ((2,), 4, 2, False, False),
      ((2,), 4, 1, True, True),
      ((2,), 4, 1, True, False),
      ((2, 3), 4, 2, False, False),
      ((2, 3), 4, 1, True, True),
  )
  def test_batch_dimension_support(
      self, batch_shape, num_q_heads, num_kv_heads, is_mqa, mqa_2d
  ):
    q_seq_len, kv_seq_len, head_dim = 64, 64, 32
    dtype = jnp.float32

    q_struct = jax.ShapeDtypeStruct(
        (*batch_shape, num_q_heads, q_seq_len, head_dim), dtype=dtype
    )
    if mqa_2d:
      k_struct = jax.ShapeDtypeStruct(
          (*batch_shape, kv_seq_len, head_dim), dtype=dtype
      )
      v_struct = jax.ShapeDtypeStruct(
          (*batch_shape, kv_seq_len, head_dim), dtype=dtype
      )
    else:
      k_struct = jax.ShapeDtypeStruct(
          (*batch_shape, num_kv_heads, kv_seq_len, head_dim), dtype=dtype
      )
      v_struct = jax.ShapeDtypeStruct(
          (*batch_shape, num_kv_heads, kv_seq_len, head_dim), dtype=dtype
      )

    q, k, v = numerics.random_initialize((q_struct, k_struct, v_struct))
    mask = base.CAUSAL_MASK

    out = self.op(q, k, v, mask=mask, is_mqa=is_mqa)
    chex.assert_shape(out, (*batch_shape, num_q_heads, q_seq_len, head_dim))

    ref_fn = functools.partial(
        reference.attention_reference,
        mask=mask.as_array(q_seq_len, kv_seq_len),
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

  @parameterized.parameters(False, True)
  def test_batched_vjp(self, batched_sinks):
    batch_size = 2
    num_heads = 4
    seq_len = 32
    head_dim = 16
    dtype = jnp.float32

    q = jax.ShapeDtypeStruct((batch_size, num_heads, seq_len, head_dim), dtype)
    k = jax.ShapeDtypeStruct((batch_size, num_heads, seq_len, head_dim), dtype)
    v = jax.ShapeDtypeStruct((batch_size, num_heads, seq_len, head_dim), dtype)
    do = jax.ShapeDtypeStruct((batch_size, num_heads, seq_len, head_dim), dtype)
    sinks_shape = (batch_size, num_heads) if batched_sinks else (num_heads,)
    sinks = jax.ShapeDtypeStruct(sinks_shape, dtype)

    q, k, v, do, sinks = numerics.random_initialize((q, k, v, do, sinks))
    mask = base.CAUSAL_MASK

    f = lambda q, k, v, s: self.op(q, k, v, mask=mask, sinks=s)
    out, vjp_fn = jax.vjp(f, q, k, v, sinks)
    dq, dk, dv, dsinks = vjp_fn(do)

    chex.assert_shape(out, (batch_size, num_heads, seq_len, head_dim))
    chex.assert_shape(dq, (batch_size, num_heads, seq_len, head_dim))
    chex.assert_shape(dk, (batch_size, num_heads, seq_len, head_dim))
    chex.assert_shape(dv, (batch_size, num_heads, seq_len, head_dim))
    chex.assert_shape(dsinks, sinks_shape)

    f_single = lambda q_i, k_i, v_i, s_i: self.op(
        q_i, k_i, v_i, mask=mask, sinks=s_i
    )
    out_refs, dq_refs, dk_refs, dv_refs, dsinks_refs = [], [], [], [], []
    for i in range(batch_size):
      s_i = sinks[i] if batched_sinks else sinks
      out_i, vjp_i = jax.vjp(f_single, q[i], k[i], v[i], s_i)
      dqi, dki, dvi, dsi = vjp_i(do[i])
      out_refs.append(out_i)
      dq_refs.append(dqi)
      dk_refs.append(dki)
      dv_refs.append(dvi)
      dsinks_refs.append(dsi)

    chex.assert_trees_all_close(out, jnp.stack(out_refs), atol=1e-3)
    chex.assert_trees_all_close(dq, jnp.stack(dq_refs), atol=1e-3)
    chex.assert_trees_all_close(dk, jnp.stack(dk_refs), atol=1e-3)
    chex.assert_trees_all_close(dv, jnp.stack(dv_refs), atol=1e-3)
    expected_dsinks = (
        jnp.stack(dsinks_refs) if batched_sinks else sum(dsinks_refs)
    )
    chex.assert_trees_all_close(dsinks, expected_dsinks, atol=1e-3)

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

  @parameterized.parameters(0, 4, -4)
  def test_canonicalize_causal_mask_preserves_offset(self, offset: int):
    causal_mask = mask_lib.CausalMask(shape=(16, 16), offset=offset)
    canonical_mask = base._canonicalize_mask(causal_mask)
    self.assertTrue(canonical_mask.is_causal)
    self.assertEqual(canonical_mask.offset, offset)
    self.assertIsNone(canonical_mask.bool_mask)
    materialized = canonical_mask.as_array(16, 16)
    expected = jnp.asarray(causal_mask[:, :])
    chex.assert_trees_all_close(materialized, expected, atol=0)

  @parameterized.parameters(4, -2)
  def test_mask_offset_requires_causal(self, offset: int):
    with self.assertRaises(ValueError):
      base.Mask(is_causal=False, offset=offset)

  @parameterized.parameters(0, 4)
  def test_splash_attention_op_causal_mask_offset_preservation(
      self, offset: int
  ):
    seq_len = 16
    head_dim = 16
    q_struct = jax.ShapeDtypeStruct((2, seq_len, head_dim), dtype=jnp.float32)
    k_struct = jax.ShapeDtypeStruct((2, seq_len, head_dim), dtype=jnp.float32)
    v_struct = jax.ShapeDtypeStruct((2, seq_len, head_dim), dtype=jnp.float32)
    q_tensor, k_tensor, v_tensor = numerics.random_initialize(
        (q_struct, k_struct, v_struct)
    )

    causal_mask = mask_lib.CausalMask(shape=(seq_len, seq_len), offset=offset)
    explicit_mask = jnp.asarray(causal_mask[:, :])

    out_symbolic = self.op(
        q_tensor, k_tensor, v_tensor, mask=causal_mask, is_mqa=False
    )
    out_explicit = self.op(
        q_tensor, k_tensor, v_tensor, mask=explicit_mask, is_mqa=False
    )

    chex.assert_trees_all_close(out_symbolic, out_explicit, atol=1e-5)

  def test_sharding_errors(self):
    q = jnp.ones((4, 128, 64))
    k = jnp.ones((4, 128, 64))
    v = jnp.ones((4, 128, 64))
    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(devices, ("x",))
    P = jax.sharding.PartitionSpec
    NamedSharding = jax.sharding.NamedSharding

    with self.assertRaisesRegex(
        NotImplementedError, "Sharding is not supported in `bind`"
    ):
      self.op.bind(q, k, v, q_sharding=NamedSharding(mesh, P("x", None, None)))

    with self.assertRaisesRegex(
        ValueError, "Too many dimensions in q_sharding's pspec"
    ):
      self.op(
          q,
          k,
          v,
          q_sharding=NamedSharding(mesh, P("x", None, None, None)),
      )

    mesh2 = jax.sharding.Mesh(devices, ("y",))
    with self.assertRaisesRegex(
        ValueError, "q_sharding and k_sharding must have the same mesh"
    ):
      self.op(
          q,
          k,
          v,
          q_sharding=NamedSharding(mesh, P("x", None, None)),
          k_sharding=NamedSharding(mesh2, P("y", None, None)),
      )

  @parameterized.parameters(
      (("x", None, None), False, False),
      ((None, "x", None), False, False),
      (("x", None, None), True, True),
      ((None, "x", None), True, True),
  )
  def test_sharding_execution(self, spec_tuple, is_mqa, use_extras):
    num_devices = min(jax.device_count(), 2)
    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ("x",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*spec_tuple)
    )

    num_q_heads, q_seq_len, head_dim = 4, 256, 64
    num_kv_heads, kv_seq_len = (1, 256) if is_mqa else (4, 256)
    dtype = jnp.float32

    q_s = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    if is_mqa:
      k_s = jax.ShapeDtypeStruct((kv_seq_len, head_dim), dtype)
    else:
      k_s = jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim), dtype)

    v_s = k_s
    do_s = jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), dtype)
    q, k, v, do = numerics.random_initialize((q_s, k_s, v_s, do_s), seed=12)

    segment_ids = None
    sinks = None
    if use_extras:
      # Simulate different sequences
      q_seg = jnp.array(
          [0] * (q_seq_len // 2) + [1] * (q_seq_len // 2), dtype=jnp.int32
      )
      segment_ids = reference.SegmentIds(q=q_seg, kv=q_seg)
      sinks = numerics.random_initialize(
          jax.ShapeDtypeStruct((num_q_heads,), dtype), seed=13
      )

    kwargs = dict(
        mask=base.CAUSAL_MASK,
        segment_ids=segment_ids,
        sinks=sinks,
        is_mqa=is_mqa,
    )

    out_ref, f_vjp_ref = jax.vjp(functools.partial(self.op, **kwargs), q, k, v)
    dq_ref, dk_ref, dv_ref = f_vjp_ref(do)

    out_sharded, f_vjp_sharded = jax.vjp(
        functools.partial(self.op, q_sharding=sharding, **kwargs), q, k, v
    )
    dq_sharded, dk_sharded, dv_sharded = f_vjp_sharded(do)

    chex.assert_trees_all_close(out_sharded, out_ref, atol=1e-2)
    chex.assert_trees_all_close(dq_sharded, dq_ref, atol=1e-2)
    chex.assert_trees_all_close(dk_sharded, dk_ref, atol=1e-2)
    chex.assert_trees_all_close(dv_sharded, dv_ref, atol=1e-2)


if __name__ == "__main__":
  absltest.main()
