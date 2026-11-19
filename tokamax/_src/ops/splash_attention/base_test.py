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

import dataclasses
import functools
import typing
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import reference
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib


@dataclasses.dataclass(frozen=True)
class _CustomSegmentIds:
  q: jax.Array
  kv: jax.Array


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

  @parameterized.parameters(
      lambda q, kv: None,
      lambda q, kv: reference.SegmentIds(q=q, kv=kv),
      lambda q, kv: (q, kv),
      lambda q, kv: (np.asarray(q), np.asarray(kv)),
      lambda q, _: q,
      lambda q, _: np.asarray(q),
      lambda q, kv: _CustomSegmentIds(q=q, kv=kv),
  )
  def test_with_segment_ids(self, segment_ids_fn):
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
    segment_ids = segment_ids_fn(q_seg, kv_seg)

    segment_ids_ref = None
    if segment_ids is not None:
      segment_ids_ref = reference.SegmentIds(q=q_seg, kv=kv_seg)

    ba = self.op.bind(q, k, v, mask, segment_ids=segment_ids)
    if segment_ids is None:
      self.assertIsNone(ba.arguments["segment_ids"])
    else:
      self.assertIsInstance(ba.arguments["segment_ids"], reference.SegmentIds)

    out_ref = reference.attention_reference(
        q=q,
        k=k,
        v=v,
        mask=mask.as_array(q_seq_len, kv_seq_len),
        segment_ids=segment_ids_ref,
        is_mqa=False,
    )

    out_op = self.op(q, k, v, mask, segment_ids=segment_ids)
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

    # Incorrect mask type
    with self.assertRaises(TypeError):
      self.op(q, q, q, mask="invalid_mask", is_mqa=False)  # pyrefly: ignore[no-matching-overload]

    # Both is_causal and mask specified
    with self.assertRaisesRegex(
        ValueError, "Cannot specify both a mask and the is_causal parameter"
    ):
      self.op(q, q, q, mask=mask, is_causal=True, is_mqa=False)

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

    # VJP validation errors for is_causal + mask
    with self.assertRaisesRegex(
        ValueError, "Cannot specify both a mask and the is_causal parameter"
    ):
      self.op.vjp(res, out, out, q, q, q, mask=mask, is_causal=True)

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

    ba_is_causal = self.op.bind(q, q, q, is_causal=True, is_mqa=False)
    self.assertEqual(ba_is_causal.arguments["mask"], base.CAUSAL_MASK)
    self.assertTrue(ba_is_causal.arguments["mask"].is_causal)
    out_is_causal = self.op(q, q, q, is_causal=True, is_mqa=False)
    self.assertEqual(out_is_causal.shape, (2, 128, 64))
    chex.assert_trees_all_close(out_is_causal, out_causal, atol=1e-5)

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

    with self.assertRaisesRegex(ValueError, "q_sharding must be specified"):
      self.op(
          q,
          k,
          v,
          k_sharding=NamedSharding(mesh, P("x", None, None)),
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

  @parameterized.parameters(
      (base.QuantizationStrategy.UNQUANTIZED, jnp.float8_e4m3fn, -1, False),
      (base.QuantizationStrategy.CAST, jnp.float8_e4m3fn, -1, True),
      (base.QuantizationStrategy.PER_TENSOR, jnp.float8_e5m2, -1, True),
      (base.QuantizationStrategy.PER_AXIS, jnp.float8_e4m3fn, -1, False),
  )
  def test_apply_quantization(
      self,
      strategy: base.QuantizationStrategy,
      quant_dtype: jax.typing.DTypeLike,
      contract_axis: int,
      apply_log2e_scaling: bool,
  ):
    x = jnp.linspace(-3.0, 3.0, 2 * 16 * 8, dtype=jnp.bfloat16).reshape(
        2, 16, 8
    )
    q_x, scale = base.apply_quantization(
        x,
        quant_dtype=quant_dtype,
        strategy=strategy,
        contract_axis=contract_axis,
        apply_log2e_scaling=apply_log2e_scaling,
    )
    if strategy == base.QuantizationStrategy.UNQUANTIZED:
      self.assertEqual(q_x.dtype, x.dtype)
      self.assertIsNone(scale)
    elif strategy == base.QuantizationStrategy.CAST:
      self.assertEqual(q_x.dtype, quant_dtype)
      self.assertIsNone(scale)
    elif strategy == base.QuantizationStrategy.PER_TENSOR:
      self.assertEqual(q_x.dtype, quant_dtype)
      self.assertIsNotNone(scale)
      self.assertEqual(scale.shape, (2, 1, 1))
      self.assertEqual(scale.dtype, x.dtype)
    elif strategy == base.QuantizationStrategy.PER_AXIS:
      self.assertEqual(q_x.dtype, quant_dtype)
      self.assertIsNotNone(scale)
      self.assertEqual(scale.shape, (2, 16, 1))
      self.assertEqual(scale.dtype, x.dtype)
      _, scale_axis_m2 = base.apply_quantization(
          x, quant_dtype=quant_dtype, strategy=strategy, contract_axis=-2
      )
      self.assertIsNotNone(scale_axis_m2)
      self.assertEqual(scale_axis_m2.shape, (2, 1, 8))

    dequant = q_x.astype(jnp.float32) * (
        scale.astype(jnp.float32) if scale is not None else 1.0
    )
    expected_x = x.astype(jnp.float32) * (
        reference.LOG2E if apply_log2e_scaling else 1.0
    )
    chex.assert_trees_all_close(dequant, expected_x, rtol=0.15, atol=0.15)
    if apply_log2e_scaling and scale is not None:
      _, scale_unscaled = base.apply_quantization(
          x,
          quant_dtype=quant_dtype,
          strategy=strategy,
          contract_axis=contract_axis,
      )
      assert scale_unscaled is not None
      expected_scale = (
          scale_unscaled.astype(jnp.float32) * reference.LOG2E
      ).astype(x.dtype)
      chex.assert_trees_all_close(scale, expected_scale, rtol=2e-2)

    q_zeros, scale_zeros = base.apply_quantization(
        jnp.zeros((2, 8), dtype=jnp.bfloat16),
        quant_dtype=quant_dtype,
        strategy=strategy,
        contract_axis=-1,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(q_zeros))))
    if scale_zeros is not None:
      self.assertFalse(bool(jnp.any(jnp.isnan(scale_zeros))))

    q_large, scale_large = base.apply_quantization(
        jnp.array([[1000.0, -1000.0]], dtype=jnp.bfloat16),
        quant_dtype=quant_dtype,
        strategy=strategy,
        contract_axis=-1,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(q_large))))
    if scale_large is not None:
      self.assertFalse(bool(jnp.any(jnp.isnan(scale_large))))
    if strategy == base.QuantizationStrategy.CAST:
      dtype_max = float(jnp.finfo(quant_dtype).max)
      np.testing.assert_array_equal(
          np.asarray(q_large.astype(jnp.float32)),
          np.array([[dtype_max, -dtype_max]], dtype=np.float32),
      )

    upstream_grad = jnp.arange(x.size, dtype=jnp.bfloat16).reshape(x.shape)

    @jax.custom_vjp
    def _mock_kernel(q_fwd, q_scales, q_dk, q_dk_scales):
      del q_scales, q_dk, q_dk_scales
      return q_fwd.astype(jnp.bfloat16)

    _mock_kernel.defvjp(
        lambda q_fwd, q_scales, q_dk, q_dk_scales: (
            _mock_kernel(q_fwd, q_scales, q_dk, q_dk_scales),
            None,
        ),
        lambda _, ct: (ct, None, None, None),
    )

    def loss_fn(inp):
      q_fwd, q_scales = base.apply_quantization(
          inp,
          quant_dtype=quant_dtype,
          strategy=strategy,
          contract_axis=contract_axis,
          apply_log2e_scaling=apply_log2e_scaling,
      )
      q_dk, q_dk_scales = base.apply_quantization(
          inp, quant_dtype=quant_dtype, strategy=strategy, contract_axis=-2
      )
      return jnp.sum(
          _mock_kernel(q_fwd, q_scales, q_dk, q_dk_scales) * upstream_grad
      )

    grad_x = jax.grad(loss_fn)(x)
    self.assertEqual(grad_x.dtype, x.dtype)
    expected_grad = (
        upstream_grad.astype(jnp.float32) * reference.LOG2E
        if apply_log2e_scaling
        else upstream_grad
    ).astype(x.dtype)
    chex.assert_trees_all_close(grad_x, expected_grad, rtol=2e-2, atol=1e-2)

  @parameterized.parameters("scalar", "0d", "1d", "2d", "sharded_heads")
  def test_max_logit_value_and_vjp(self, mode):
    batch_size, num_q_heads, num_kv_heads = 2, 4, 2
    seq_len, head_dim = 64, 32
    dtype = jnp.float32

    q_s = jax.ShapeDtypeStruct(
        (batch_size, num_q_heads, seq_len, head_dim), dtype
    )
    k_s = jax.ShapeDtypeStruct(
        (batch_size, num_kv_heads, seq_len, head_dim), dtype
    )
    do_s = q_s
    sinks_s = jax.ShapeDtypeStruct((num_q_heads,), dtype)
    q, k, v, do, sinks = numerics.random_initialize(
        (q_s, k_s, k_s, do_s, sinks_s), seed=31
    )
    q = q * (head_dim**-0.5)

    if mode == "scalar":
      max_logit_value = 10.0
    elif mode == "0d":
      max_logit_value = jnp.asarray(10.0, dtype=dtype)
    elif mode in ("1d", "sharded_heads"):
      max_logit_value = jnp.array([8.0, 9.0, 10.0, 11.0], dtype=dtype)
    else:
      max_logit_value = jnp.array(
          [[8.0, 9.0, 10.0, 11.0], [8.5, 9.5, 10.5, 11.5]], dtype=dtype
      )

    q_sharding = None
    if mode == "sharded_heads":
      mesh = jax.sharding.Mesh(
          jax.devices()[: min(jax.device_count(), 2)], ("x",)
      )
      q_sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(None, "x", None, None)
      )

    # Default float32 matmuls use bf16 on TPU and tf32 on GPU, which masks
    # shift-invariance with coarse p @ v rounding noise.
    with jax.default_matmul_precision("highest"):
      _, res_default = self.op(
          q, k, v, mask=base.CAUSAL_MASK, sinks=sinks, return_residuals=True
      )
      out_res, res_fixed = self.op(
          q,
          k,
          v,
          mask=base.CAUSAL_MASK,
          sinks=sinks,
          max_logit_value=max_logit_value,
          q_sharding=q_sharding,
          return_residuals=True,
      )
      expected_max = jnp.broadcast_to(
          jnp.asarray(max_logit_value, dtype=dtype)[..., None],
          (batch_size, num_q_heads, seq_len),
      )
      chex.assert_trees_all_close(res_fixed[0], expected_max, atol=1e-6)
      self.assertFalse(jnp.allclose(res_default[0], res_fixed[0], atol=1e-2))

      fn_ref = lambda q_, k_, v_, s_: self.op(
          q_, k_, v_, mask=base.CAUSAL_MASK, sinks=s_
      )
      fn_mlv = lambda q_, k_, v_, s_, mlv_: self.op(
          q_,
          k_,
          v_,
          mask=base.CAUSAL_MASK,
          sinks=s_,
          max_logit_value=mlv_,
          q_sharding=q_sharding,
      )
      out_ref, vjp_ref = jax.vjp(fn_ref, q, k, v, sinks)
      dq_ref, dk_ref, dv_ref, dsinks_ref = vjp_ref(do)
      if isinstance(max_logit_value, jax.Array):
        out, vjp_fn = jax.vjp(fn_mlv, q, k, v, sinks, max_logit_value)
        dq, dk, dv, dsinks, dmlv = vjp_fn(do)
        chex.assert_trees_all_close(dmlv, jnp.zeros_like(max_logit_value))
      else:
        out, vjp_fn = jax.vjp(
            lambda q_, k_, v_, s_: fn_mlv(q_, k_, v_, s_, max_logit_value),
            q,
            k,
            v,
            sinks,
        )
        dq, dk, dv, dsinks = vjp_fn(do)

    # Non-CPU VJP recomputes p = exp(logits - lse) through hardware exp/log
    # approximations (~1.7e-4 relative error on TPU v4 when shifted by 10.0).
    vjp_tol = (
        dict(atol=1e-5)
        if jax.default_backend() == "cpu"
        else dict(atol=5e-4, rtol=5e-4)
    )
    chex.assert_trees_all_close(out_res, out_ref, atol=1e-5)
    chex.assert_trees_all_close(out, out_ref, atol=1e-5)
    chex.assert_trees_all_close(dq, dq_ref, **vjp_tol)
    chex.assert_trees_all_close(dk, dk_ref, **vjp_tol)
    chex.assert_trees_all_close(dv, dv_ref, **vjp_tol)
    chex.assert_trees_all_close(dsinks, dsinks_ref, **vjp_tol)

  def test_subclass_without_max_logit_value(self):
    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class _VjpWithoutMaxLogit(base.SplashAttentionVjp[None]):

      @typing.override
      def _fwd(  # pyrefly: ignore[bad-override]
          self,
          residuals: base.Residuals,
          out: jax.Array,
          dout: jax.Array,
          q: jax.Array,
          k: jax.Array,
          v: jax.Array,
          *,
          mask: base.Mask,
          segment_ids: base.SegmentIds | None = None,
          sinks: jax.Array | None = None,
          is_mqa: bool = False,
          mask_value: float = base.DEFAULT_MASK_VALUE,
          attn_logits_soft_cap: float | None = None,
          dropout_rate: float = 0.0,
          prng_key: jax.Array | None = None,
          return_residuals: bool = False,
          config: None = None,
      ) -> tuple[base.SplashAttentionGrads, None]:
        return base.SplashAttentionVjp._fwd(
            self,
            residuals,
            out,
            dout,
            q,
            k,
            v,
            mask=mask,
            segment_ids=segment_ids,
            sinks=sinks,
            is_mqa=is_mqa,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            dropout_rate=dropout_rate,
            prng_key=prng_key,
            return_residuals=return_residuals,
            config=config,
        )

    @dataclasses.dataclass(frozen=True)
    class _OpWithoutMaxLogit(base.SplashAttention[None]):

      def __post_init__(self):
        if self.vjp is None:
          object.__setattr__(self, "vjp", _VjpWithoutMaxLogit())

      @typing.override
      def _fwd(  # pyrefly: ignore[bad-override]
          self,
          q: jax.Array,
          k: jax.Array,
          v: jax.Array,
          *,
          mask: base.Mask,
          segment_ids: base.SegmentIds | None = None,
          sinks: jax.Array | None = None,
          is_mqa: bool = False,
          mask_value: float = base.DEFAULT_MASK_VALUE,
          attn_logits_soft_cap: float | None = None,
          dropout_rate: float = 0.0,
          prng_key: jax.Array | None = None,
          return_residuals: bool = False,
          config: None = None,
      ) -> tuple[jax.Array, base.Residuals | None]:
        return base.SplashAttention._fwd(
            self,
            q,
            k,
            v,
            mask=mask,
            segment_ids=segment_ids,
            sinks=sinks,
            is_mqa=is_mqa,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            dropout_rate=dropout_rate,
            prng_key=prng_key,
            return_residuals=return_residuals,
            config=config,
        )

    self.assertNotIn("__class__", _VjpWithoutMaxLogit._fwd.__code__.co_freevars)
    vjp_no_mlv = _VjpWithoutMaxLogit()
    op_no_mlv = _OpWithoutMaxLogit(vjp=vjp_no_mlv)
    q_s = jax.ShapeDtypeStruct((1, 4, 64, 32), jnp.float32)
    q, k, v, do = numerics.random_initialize((q_s, q_s, q_s, q_s), seed=17)

    mesh = jax.sharding.Mesh(jax.devices()[:1], ("x",))
    q_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, "x", None, None)
    )
    with jax.default_matmul_precision("highest"):
      out, vjp_fn = jax.vjp(
          lambda q_, k_, v_: op_no_mlv(q_, k_, v_, mask=base.CAUSAL_MASK),
          q,
          k,
          v,
      )
      out_ref, vjp_ref = jax.vjp(
          lambda q_, k_, v_: self.op(q_, k_, v_, mask=base.CAUSAL_MASK),
          q,
          k,
          v,
      )
      chex.assert_trees_all_close(out, out_ref, atol=1e-5)
      chex.assert_trees_all_close(vjp_fn(do), vjp_ref(do), atol=1e-5)

      out_sharded, residuals = op_no_mlv(
          q,
          k,
          v,
          mask=base.CAUSAL_MASK,
          q_sharding=q_sharding,
          return_residuals=True,
      )
      chex.assert_trees_all_close(out_sharded, out_ref, atol=1e-5)
      grads_direct = vjp_no_mlv(
          residuals, out_sharded, do, q, k, v, mask=base.CAUSAL_MASK
      )
      chex.assert_trees_all_close(
          (grads_direct["q"], grads_direct["k"], grads_direct["v"]),
          vjp_ref(do),
          atol=1e-5,
      )

    with self.assertRaises(NotImplementedError):
      op_no_mlv(q, k, v, max_logit_value=1.0)
    with self.assertRaises(NotImplementedError):
      op_no_mlv(q, k, v, max_logit_value=1.0, q_sharding=q_sharding)
    with self.assertRaises(NotImplementedError):
      op_no_mlv.bind(q, k, v, max_logit_value=1.0)
    with self.assertRaises(NotImplementedError):
      vjp_no_mlv.bind(residuals, out, do, q, k, v, max_logit_value=1.0)

  def test_max_logit_value_masked_and_underflow_rows(self):
    num_heads, seq_len, head_dim = 2, 16, 32
    s = jax.ShapeDtypeStruct((num_heads, seq_len, head_dim), jnp.float32)
    q, k, v, do = numerics.random_initialize((s, s, s, s), seed=71)
    q = q * (head_dim**-0.5)

    row0_mask = (
        jnp.ones((seq_len, seq_len), dtype=jnp.bool_).at[0, :].set(False)
    )
    all_masked = jnp.zeros((seq_len, seq_len), dtype=jnp.bool_)

    with jax.default_matmul_precision("highest"):
      out_row0, vjp_row0 = jax.vjp(
          lambda q_, k_, v_: self.op(
              q_, k_, v_, mask=row0_mask, max_logit_value=10.0
          ),
          q,
          k,
          v,
      )
      dq_r0, dk_r0, dv_r0 = vjp_row0(do)
      chex.assert_tree_all_finite((out_row0, dq_r0, dk_r0, dv_r0))
      chex.assert_trees_all_close(out_row0[:, 0, :], 0.0, atol=1e-7)
      chex.assert_trees_all_close(dq_r0[:, 0, :], 0.0, atol=1e-7)

      out_ref_r0, vjp_ref_r0 = jax.vjp(
          lambda q_, k_, v_: self.op(q_, k_, v_, mask=row0_mask), q, k, v
      )
      dq_ref_r0, _, _ = vjp_ref_r0(do)
      vjp_tol = (
          dict(atol=1e-5)
          if jax.default_backend() == "cpu"
          else dict(atol=5e-4, rtol=5e-4)
      )
      chex.assert_trees_all_close(
          out_row0[:, 1:, :], out_ref_r0[:, 1:, :], atol=1e-5
      )
      chex.assert_trees_all_close(
          dq_r0[:, 1:, :], dq_ref_r0[:, 1:, :], **vjp_tol
      )

      out_all, vjp_all = jax.vjp(
          lambda q_, k_, v_: self.op(
              q_, k_, v_, mask=all_masked, max_logit_value=10.0
          ),
          q,
          k,
          v,
      )
      dq_all, dk_all, dv_all = vjp_all(do)
      chex.assert_tree_all_finite((out_all, dq_all, dk_all, dv_all))
      chex.assert_trees_all_close(out_all, 0.0, atol=1e-7)
      chex.assert_trees_all_close(dq_all, 0.0, atol=1e-7)
      chex.assert_trees_all_close(dk_all, 0.0, atol=1e-7)
      chex.assert_trees_all_close(dv_all, 0.0, atol=1e-7)

      sinks = jnp.full((num_heads,), -200.0, dtype=jnp.float32)
      out_uf, vjp_uf = jax.vjp(
          lambda q_, k_, v_, s_: self.op(
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

  def test_max_logit_value_int_and_numpy_sharding(self):
    q_s = jax.ShapeDtypeStruct((2, 4, 32, 16), jnp.float32)
    q, k, v, do = numerics.random_initialize((q_s, q_s, q_s, q_s), seed=83)
    q = q * 0.25

    with warnings.catch_warnings():
      warnings.simplefilter("error", UserWarning)
      out_int, vjp_int = jax.vjp(
          lambda q_, k_, v_: self.op(
              q_, k_, v_, mask=base.CAUSAL_MASK, max_logit_value=10
          ),
          q,
          k,
          v,
      )
      grads_int = vjp_int(do)
    out_flt, vjp_flt = jax.vjp(
        lambda q_, k_, v_: self.op(
            q_, k_, v_, mask=base.CAUSAL_MASK, max_logit_value=10.0
        ),
        q,
        k,
        v,
    )
    chex.assert_trees_all_close(out_int, out_flt, atol=1e-6)
    chex.assert_trees_all_close(grads_int, vjp_flt(do), atol=1e-6)

    np_mlv = np.array([8.0, 9.0, 10.0, 11.0], dtype=np.float32)
    np_sinks = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    np_mask = np.tril(np.ones((32, 32), dtype=np.bool_))
    mesh = jax.sharding.Mesh(
        jax.devices()[: min(jax.device_count(), 2)], ("x",)
    )
    q_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, "x", None, None)
    )
    out_unsharded = self.op(  # pyrefly: ignore[no-matching-overload]
        q, k, v, mask=np_mask, sinks=np_sinks, max_logit_value=np_mlv
    )
    out_sharded = self.op(  # pyrefly: ignore[no-matching-overload]
        q,
        k,
        v,
        mask=np_mask,
        sinks=np_sinks,
        max_logit_value=np_mlv,
        q_sharding=q_sharding,
    )
    chex.assert_trees_all_close(out_sharded, out_unsharded, atol=1e-5)

    jax_mask = jnp.asarray(np_mask)
    jax_sinks = jnp.asarray(np_sinks)
    jax_mlv = jnp.asarray(np_mlv)
    out_jax_sharded = self.op(
        q,
        k,
        v,
        mask=jax_mask,
        sinks=jax_sinks,
        max_logit_value=jax_mlv,
        q_sharding=q_sharding,
    )
    chex.assert_trees_all_close(out_jax_sharded, out_unsharded, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
