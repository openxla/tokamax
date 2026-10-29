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

import pytest
import random
from typing import cast, override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import test_base
from tokamax._src.ops.attention import xla_chunked

import pytest


class XlaChunkedAttentionTest(test_base.AttentionTestBase):

  def __init__(self, *args):
    op = xla_chunked.XlaChunkedDotProductAttention(chunk_size=128)
    super().__init__(*args, attention_fn=op)

  def _run_test(self, q_shape, *args, dtype=jnp.float32, **kwargs):
    if q_shape[1] >= 16384:
      self.skipTest("XLA seems to fail for so long sequences (b/384038935)")

    impl_kwargs = kwargs.pop("impl_kwargs", {})
    if dtype == jnp.float32:
      impl_kwargs["precision"] = jax.lax.Precision.HIGHEST

    super()._run_test(
        q_shape, *args, dtype=dtype, impl_kwargs=impl_kwargs, **kwargs
    )

  def _run_test_with_inputs(self, *args, **kwargs):
    if kwargs.get("impl_kwargs", {}).get("return_residuals", False):
      self.skipTest("Residuals unsupported with XLA chunked attention.")
    super()._run_test_with_inputs(*args, **kwargs)

  @parameterized.parameters(31, 123, 256, 2048, (64, 128), (128, 64))
  def test_chunk_sizes(self, *chunk_size):
    chunk_size = chunk_size[0] if len(chunk_size) == 1 else chunk_size
    op = cast(xla_chunked.XlaChunkedDotProductAttention, self._attention_fn)
    mocked_impl = dataclasses.replace(op, chunk_size=chunk_size)
    with mock.patch.object(self, "_attention_fn", mocked_impl):
      self._run_test((2, 1024, 4, 64))

  def test_normalize_output(self):
    with test_base.override_test_args(atol_grads=3e-5):
      super().test_normalize_output()

  @parameterized.product(
      num_heads_kv=(1, 2),
      dtype=(jnp.float32, jnp.bfloat16),
      normalize_output=(True, False),
  )
  def test_consistency_with_unchunked_mqa_gqa(
      self, num_heads_kv, dtype, normalize_output
  ):
    num_heads_q = 8
    # Non-multiples of chunk_size=(48, 80) exercise both main loops and
    # remainders on both Q and KV sequences.
    seq_q, seq_k = 140, 210
    chunked_op = xla_chunked.XlaChunkedDotProductAttention(chunk_size=(48, 80))
    unchunked_op = base.DotProductAttention()

    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    q = jax.random.normal(keys[0], (2, seq_q, num_heads_q, 32), dtype)
    k = jax.random.normal(keys[1], (2, seq_k, num_heads_kv, 32), dtype)
    v = jax.random.normal(keys[2], (2, seq_k, num_heads_kv, 24), dtype)
    bias = jax.random.normal(keys[3], (1, num_heads_q, 1, seq_k), dtype)
    dout = jax.random.normal(keys[4], (2, seq_q, num_heads_q, 24), dtype)
    dropout_mask = jax.random.bernoulli(
        keys[5], p=0.75, shape=(2, 1, seq_q, seq_k)
    )

    k_start = jnp.arange(num_heads_kv, dtype=jnp.int32)[:, None] * 3
    k_end = jnp.full((num_heads_kv, 1), seq_k - 5, dtype=jnp.int32)
    q_start = jnp.arange(num_heads_q, dtype=jnp.int32)[:, None]
    mask = base.Mask(
        is_causal=True, q_start=q_start, k_start=k_start, k_end=k_end
    )
    precision = (
        jax.lax.Precision.HIGHEST
        if dtype == jnp.float32
        else jax.lax.Precision.DEFAULT
    )
    common_kwargs = dict(
        mask=mask,
        logits_soft_cap=15.0,
        dropout_mask=dropout_mask,
        dropout_rate=0.25,
        normalize_output=normalize_output,
        precision=precision,
    )

    out_chunked, vjp_chunked = jax.vjp(
        lambda q, k, v, b: chunked_op(q, k, v, bias=b, **common_kwargs),
        q,
        k,
        v,
        bias,
    )
    out_unchunked, vjp_unchunked = jax.vjp(
        lambda q, k, v, b: unchunked_op(q, k, v, bias=b, **common_kwargs),
        q,
        k,
        v,
        bias,
    )
    grads_chunked = vjp_chunked(dout)
    grads_unchunked = vjp_unchunked(dout)

    if dtype == jnp.float32:
      atol = rtol = 1e-4
      atol_grads = rtol_grads = 2e-4
    elif normalize_output:
      atol = rtol = 2e-2
      atol_grads = rtol_grads = 5e-2
    else:
      atol = rtol = 4e-2
      atol_grads = rtol_grads = 1e-1
    chex.assert_trees_all_close(
        out_chunked, out_unchunked, atol=atol, rtol=rtol
    )
    chex.assert_trees_all_close(
        grads_chunked, grads_unchunked, atol=atol_grads, rtol=rtol_grads
    )

  @parameterized.parameters(1, 2)
  def test_mqa_gqa_no_logits_transpose_or_kv_repeat_in_hlo(self, num_heads_kv):
    q = jax.ShapeDtypeStruct((2, 64, 8, 32), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((2, 96, num_heads_kv, 32), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((2, 96, num_heads_kv, 32), jnp.bfloat16)
    for op in (
        self._attention_fn,
        xla_chunked.XlaChunkedDotProductAttention(chunk_size=(32, 48)),
    ):
      hlo_text = jax.jit(op).lower(q, k, v).as_text("hlo")
      for line in hlo_text.splitlines():
        if "transpose(" in line:
          has_q_dim = any(d in line for d in ("64", "256", "512"))
          self.assertFalse(
              has_q_dim and "96" in line,
              msg=f"Unexpected quadratic logits/weights transpose: {line}",
          )
        self.assertNotIn("bf16[2,96,8,32]", line)
        self.assertNotIn("bf16[2,48,8,32]", line)

  @parameterized.product(
      num_heads_kv=(1, 2, 8),
      logits_dtype=(base.AUTO, jnp.bfloat16),
      normalize_output=(True, False),
  )
  def test_no_bf16_reductions_in_bwd_hlo(
      self, num_heads_kv, logits_dtype, normalize_output
  ):
    bias_shape = {
        1: (1, 1, 1, 112),
        2: (1, 8, 80, 1),
        8: (1, 1, 112),
    }[num_heads_kv]
    chunked_op = xla_chunked.XlaChunkedDotProductAttention(chunk_size=(32, 48))
    q = jax.ShapeDtypeStruct((2, 80, 8, 32), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((2, 112, num_heads_kv, 32), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((2, 112, num_heads_kv, 32), jnp.bfloat16)
    bias = jax.ShapeDtypeStruct(bias_shape, jnp.bfloat16)
    dout = jax.ShapeDtypeStruct((2, 80, 8, 32), jnp.bfloat16)

    def bwd(q, k, v, bias, dout):
      fn = lambda q, k, v, b: chunked_op(
          q,
          k,
          v,
          bias=b,
          is_causal=True,
          logits_dtype=logits_dtype,
          normalize_output=normalize_output,
      )
      _, vjp_fn = jax.vjp(fn, q, k, v, bias)
      return vjp_fn(dout)

    lowered = jax.jit(bwd).lower(q, k, v, bias, dout)
    self.assertEmpty(
        test_base.find_low_precision_reductions(lowered.as_text("hlo"))
    )
    # XLA:TPU's BFloat16Propagation demotes f32 reductions between bf16 converts
    # because TPU hardware accumulates bf16 reductions in f32.
    if jax.default_backend() != "tpu":
      compiled_hlo = lowered.compile().as_text()
      if compiled_hlo is not None:
        self.assertEmpty(test_base.find_low_precision_reductions(compiled_hlo))

  @parameterized.parameters(
      ((0, None, None, None),),
      ((None, 0, 0, None),),
      ((None, None, None, 0),),
  )
  def test_no_bf16_reductions_in_vmap_bwd_hlo(self, in_axes):
    vmap_size = 3
    chunked_op = xla_chunked.XlaChunkedDotProductAttention(chunk_size=(32, 48))
    q_shape = (2, 80, 8, 32)
    kv_shape = (2, 112, 2, 32)
    bias_shape = (1, 8, 1, 112)
    q = jax.ShapeDtypeStruct(
        (vmap_size, *q_shape) if in_axes[0] == 0 else q_shape, jnp.bfloat16
    )
    k = jax.ShapeDtypeStruct(
        (vmap_size, *kv_shape) if in_axes[1] == 0 else kv_shape, jnp.bfloat16
    )
    v = jax.ShapeDtypeStruct(
        (vmap_size, *kv_shape) if in_axes[2] == 0 else kv_shape, jnp.bfloat16
    )
    bias = jax.ShapeDtypeStruct(
        (vmap_size, *bias_shape) if in_axes[3] == 0 else bias_shape,
        jnp.bfloat16,
    )
    dout = jax.ShapeDtypeStruct((vmap_size, *q_shape), jnp.bfloat16)

    def bwd(q, k, v, bias, dout):
      fn = jax.vmap(
          lambda q, k, v, b: chunked_op(
              q, k, v, bias=b, is_causal=True, logits_dtype=jnp.bfloat16
          ),
          in_axes=in_axes,
      )
      _, vjp_fn = jax.vjp(fn, q, k, v, bias)
      return vjp_fn(dout)

    lowered = jax.jit(bwd).lower(q, k, v, bias, dout)
    self.assertEmpty(
        test_base.find_low_precision_reductions(lowered.as_text("hlo"))
    )
    if jax.default_backend() != "tpu":
      compiled_hlo = lowered.compile().as_text()
      if compiled_hlo is not None:
        self.assertEmpty(test_base.find_low_precision_reductions(compiled_hlo))

  def test_bwd_broadcast_bias_cross_chunk_f32_accumulation(self):
    # Chunk 0 contributes ~1.0 to dbias while each of chunks 1..64 contributes
    # ~0.002 (~0.128 total). Accumulating across Q chunks in bfloat16 swamps
    # every 0.002 increment (< bf16 ULP 0.0078125 at 1.0), yielding 1.0 instead
    # of ~1.125.
    seq_q, seq_k, head_dim = 65 * 16, 16, 16
    chunked_op = xla_chunked.XlaChunkedDotProductAttention(chunk_size=16)
    unchunked_op = base.DotProductAttention()

    q = jnp.zeros((1, seq_q, 1, head_dim), jnp.bfloat16)
    k = jnp.zeros((1, seq_k, 1, head_dim), jnp.bfloat16)
    v = jnp.ones((1, seq_k, 1, head_dim), jnp.bfloat16) / head_dim
    bias = jnp.zeros((1, 1, 1, seq_k), jnp.bfloat16)
    dout = jnp.full((1, seq_q, 1, head_dim), 0.002, jnp.bfloat16)
    dout = dout.at[:, :16].set(1.0 / 16.0)

    def dbias_fn(op, q, k, v, b, dout):
      _, vjp_fn = jax.vjp(
          lambda b_: op(
              q,
              k,
              v,
              bias=b_,
              logits_dtype=jnp.bfloat16,
              normalize_output=False,
          ),
          b,
      )
      return vjp_fn(dout)[0]

    dbias_chunked = jax.jit(functools.partial(dbias_fn, chunked_op))(
        q, k, v, bias, dout
    )
    dbias_unchunked = jax.jit(functools.partial(dbias_fn, unchunked_op))(
        q, k, v, bias, dout
    )
    chex.assert_trees_all_close(
        dbias_chunked, dbias_unchunked, atol=1e-2, rtol=1e-2
    )


def _split_into_pages(unpadded_k, unpadded_v, max_num_pages, max_page_size):
  batch, unpadded_seq_kv, num_heads_kv, head_dim = unpadded_k.shape
  head_dim_out = unpadded_v.shape[-1]
  kv_dtype = unpadded_k.dtype

  kv_pages_shape = (max_num_pages, max_page_size, num_heads_kv)
  k_pages = jnp.zeros(kv_pages_shape + (head_dim,), kv_dtype)
  v_pages = jnp.zeros(kv_pages_shape + (head_dim_out,), kv_dtype)
  lengths = jnp.zeros((max_num_pages,), jnp.int32)
  num_active_pages = jnp.zeros((batch,), jnp.int32)
  active_page_indices = jnp.full((batch, max_num_pages), -1, jnp.int32)
  empty_indices = list(range(max_num_pages))
  random.shuffle(empty_indices)

  # Track the maximum number of active pages across batches since the
  # reference implementation doesn't support ragged tensors.
  max_num_active_pages = 0

  for b in range(batch):
    # Split the sequence into randomly-sized pages and assign them to
    # randomly-chosen empty indices.
    curr_seq = 0
    while curr_seq < unpadded_seq_kv:
      empty_idx = empty_indices.pop()

      remaining_seq = unpadded_seq_kv - curr_seq
      length = random.randint(1, min(max_page_size, remaining_seq))
      lengths = lengths.at[empty_idx].set(jnp.array(length))

      k_page = unpadded_k[b, curr_seq : curr_seq + length, :, :]
      v_page = unpadded_v[b, curr_seq : curr_seq + length, :, :]
      k_pages = k_pages.at[empty_idx, :length].set(k_page)
      v_pages = v_pages.at[empty_idx, :length].set(v_page)

      i = num_active_pages[b]
      active_page_indices = active_page_indices.at[b, i].set(empty_idx)
      num_active_pages = num_active_pages.at[b].set(i + 1)
      max_num_active_pages = max(max_num_active_pages, i + 1)

      curr_seq += length

  return (
      k_pages,
      v_pages,
      max_num_active_pages,
      num_active_pages,
      active_page_indices,
      lengths,
  )


def _concatenate_pages(
    k_pages,
    v_pages,
    max_num_active_pages,
    num_active_pages,
    active_page_indices,
    max_page_size,
):
  ref_k, ref_v, ref_mask = [], [], []

  for b in range(num_active_pages.shape[0]):
    curr_b_cache_k, curr_b_cache_v, curr_b_mask = [], [], []

    for i in range(num_active_pages[b]):
      page_idx = active_page_indices[b, i]
      curr_b_cache_k.append(k_pages[page_idx])
      curr_b_cache_v.append(v_pages[page_idx])
      curr_b_mask.append(jnp.full((max_page_size,), True))

    def pad_cache(x):
      pad_amount = max_num_active_pages * max_page_size - x.shape[0]
      return jnp.pad(x, ((0, pad_amount), (0, 0), (0, 0)))

    def pad_mask(x):
      pad_amount = max_num_active_pages * max_page_size - x.shape[0]
      return jnp.pad(x, (0, pad_amount), constant_values=False)

    ref_k.append(pad_cache(jnp.concatenate(curr_b_cache_k)))
    ref_v.append(pad_cache(jnp.concatenate(curr_b_cache_v)))
    ref_mask.append(pad_mask(jnp.concatenate(curr_b_mask)))

  return (
      jnp.stack(ref_k, axis=0),
      jnp.stack(ref_v, axis=0),
      jnp.stack(ref_mask, axis=0).reshape(num_active_pages.shape[0], 1, 1, -1),
  )


@pytest.mark.skip(reason="Too slow for OSS regression tests.")
class XlaPagedAttentionTest(test_base.AttentionTestBase):

  def __init__(self, *args):
    super().__init__(
        *args,
        attention_fn=xla_chunked.XlaChunkedDotProductAttention(chunk_size=128),
        supports_bias=False,
        supports_mask=False,
        supports_dropout=False,
        supports_indices=False,
        supports_vjp=False,
    )

  def _run_test_with_inputs(self, *args, **kwargs):
    """Initialize a random KV cache and split it into pages of variable lens."""
    # The rest of the values (per page) are padded up to `max_page_size`, and
    # correspond to masked values for the reference computation.
    # In order to recreate an equivalent KV cache for the reference, the padded
    # pages must be re-concatenated in order.

    q, unpadded_k, unpadded_v, *args = args
    q, unpadded_k, unpadded_v = jax.tree.map(
        lambda x: jax.lax.collapse(x, 0, -3), (q, unpadded_k, unpadded_v)
    )

    impl_kwargs = kwargs.pop("impl_kwargs", {})
    mask = impl_kwargs.get("mask", xla_chunked.Mask())
    if (
        impl_kwargs.get("is_causal", False)
        or (isinstance(mask, xla_chunked.Mask) and mask.is_causal)
        or kwargs.get("is_causal", False)
    ):
      self.skipTest("Causal mask unsupported with paged attention.")

    if impl_kwargs.get("return_residuals", False):
      self.skipTest("Residuals unsupported with XLA paged attention.")

    batch, unpadded_seq_kv, *_ = unpadded_k.shape
    max_num_pages = batch * unpadded_seq_kv  # Worst case: each page has 1 token
    max_page_size = 128  # Best case: each page has 128 tokens.

    (
        k_pages,
        v_pages,
        max_num_active_pages,
        num_active_pages,
        active_page_indices,
        lengths,
    ) = _split_into_pages(unpadded_k, unpadded_v, max_num_pages, max_page_size)

    ref_k, ref_v, ref_mask = _concatenate_pages(
        k_pages,
        v_pages,
        max_num_active_pages,
        num_active_pages,
        active_page_indices,
        max_page_size,
    )

    ref_impl_ = kwargs.pop("ref_impl", nn.dot_product_attention)

    def ref_impl(q, k_pages, v_pages, *args, **kwargs):
      del k_pages, v_pages
      if "mask" in kwargs:
        del kwargs["mask"]
      return ref_impl_(q, ref_k, ref_v, *args, mask=ref_mask, **kwargs)

    impl_kwargs["paging_info"] = xla_chunked.PagingInfo(
        num_active_pages,
        active_page_indices,
        lengths,
    )

    if q.shape[-3] % 128 != 0:
      kwargs["expect_supported"] = False

    super()._run_test_with_inputs(
        q,
        k_pages,
        v_pages,
        *args,
        impl_kwargs=impl_kwargs,
        ref_impl=ref_impl,
        **kwargs,
    )

  def test_normalize_output(self):
    self.skipTest("Reference implementation doesn't support masks.")

  @override
  def _test_invalid_shapes(self, **kwargs):
    self.skipTest("Flax impl doesn't support different batch sizes.")

  @override
  def _test_quantized_int8(self, tile_shape, quantize_q):
    self.skipTest("Unsupported.")

  @override
  def _test_quantized_int4(self, subchannel_size):
    self.skipTest("Unsupported.")


if __name__ == "__main__":
  absltest.main()
