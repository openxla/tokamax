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
import pytest
import random
from typing import cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
from tokamax._src import quantization
from tokamax._src.ops.attention import test_base
from tokamax._src.ops.attention import xla_chunked
from typing_extensions import override
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
      super().test_normalize_output()  # pytype: disable=attribute-error


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
    if (
        impl_kwargs.get("is_causal", False)
        or impl_kwargs.get("mask", xla_chunked.Mask()).is_causal
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
