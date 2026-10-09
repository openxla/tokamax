# Copyright 2026 Google LLC
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
"""Shared correctness tests for SparseMLA implementations."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.sparse_mla import reference
from tokamax._src.ops.experimental.tpu.compress_store import csa_cache_layout

ATOL = RTOL = 0.1


def _quantize_lane_periodic(
    x: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes x into FP8 (e4m3) values and lane-periodic e8m0 scales."""
  blocked = x.reshape(
      *x.shape[:-1],
      x.shape[-1] // reference.NOPE_SCALE_PERIOD,
      reference.NOPE_SCALE_PERIOD,
  )
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-2, keepdims=True), 1e-4, None)
  sf = jnp.power(2.0, jnp.ceil(jnp.log2(amax / fp8_max)))
  q = (blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
  return q, jnp.squeeze(sf, -2).astype(jnp.float8_e8m0fnu)


def make_inputs(
    *,
    batch_size: int = 12,
    num_heads: int = 64,
    page_size: int = 16,
    topk: int = 1024,
    pad_tokens: bool = False,
    seed: int = 0,
) -> tuple[tuple[jax.Array, ...], int]:
  """Generates deterministic SparseMLA inputs and valid token count."""
  rng = np.random.default_rng(seed)
  head_dim = reference.HEAD_DIM

  num_decode_seqs = batch_size // 2
  prefill_lens = rng.integers(8, 60, size=(batch_size - num_decode_seqs,))
  new_kv_lens = np.concatenate(
      [np.ones((num_decode_seqs,), dtype=np.int32), prefill_lens]
  )
  cu_q_lens = jnp.concatenate([
      jnp.array([0], dtype=jnp.int32),
      jnp.cumulative_sum(jnp.asarray(new_kv_lens, dtype=jnp.int32)),
  ])
  kv_lens = new_kv_lens + rng.integers(30, 200, size=(batch_size,))
  actual_tokens = int(np.sum(new_kv_lens))
  total_tokens = (
      ((actual_tokens + 32 + 63) // 64) * 64 if pad_tokens else actual_tokens
  )

  q = jnp.asarray(
      rng.random(size=(total_tokens, num_heads, head_dim), dtype=np.float32),
      dtype=jnp.bfloat16,
  )

  topk_indices_list = []
  for i in range(batch_size):
    kv_len_i = int(kv_lens[i])
    for _ in range(int(new_kv_lens[i])):
      perm = rng.permutation(kv_len_i)
      indices = list(perm[:topk])
      if len(indices) < topk:
        indices.extend([-1] * (topk - len(indices)))
      topk_indices_list.append(indices)
  for _ in range(total_tokens - actual_tokens):
    topk_indices_list.append([-1] * topk)
  topk_indices = jnp.asarray(topk_indices_list, dtype=jnp.int32)

  pages_per_seq = (500 + page_size - 1) // page_size
  total_pages = batch_size * pages_per_seq
  page_indices = jnp.arange(total_pages, dtype=jnp.int32)

  kv_c_flat = jnp.asarray(
      rng.random(size=(total_pages, page_size, head_dim), dtype=np.float32),
      dtype=jnp.bfloat16,
  )
  fp8_part = kv_c_flat[..., : reference.NOPE_DIM]
  bf16_part = kv_c_flat[..., reference.NOPE_DIM : head_dim]
  fp8_quant, scales_quant = _quantize_lane_periodic(fp8_part)
  fp8_uint8 = jax.lax.bitcast_convert_type(
      fp8_quant.reshape(total_pages, page_size, reference.NOPE_DIM), jnp.uint8
  )
  scales_uint8 = jax.lax.bitcast_convert_type(scales_quant, jnp.uint8)
  nope_slabs = jnp.concatenate([fp8_uint8, scales_uint8], axis=-1).reshape(
      total_pages, page_size, 4, 128
  )
  cache_kv_nope = csa_cache_layout.slabs_to_words(nope_slabs)
  cache_kv_rope = csa_cache_layout.encode_rope(
      bf16_part.reshape(-1, reference.ROPE_DIM)
  ).reshape(total_pages, page_size // 4, 128)

  distribution = jnp.array(
      [num_decode_seqs, num_decode_seqs, batch_size], dtype=jnp.int32
  )
  attention_sinks = jnp.asarray(rng.random(size=(num_heads,), dtype=np.float32))
  swa_accumution = jnp.asarray(
      rng.random(size=(total_tokens, num_heads, head_dim), dtype=np.float32),
      dtype=jnp.bfloat16,
  )
  swa_l = jnp.asarray(
      rng.random(size=(total_tokens, num_heads), dtype=np.float32)
  )
  swa_m = jnp.asarray(
      rng.random(size=(total_tokens, num_heads), dtype=np.float32)
  )

  inputs = (
      q,
      cache_kv_nope,
      cache_kv_rope,
      topk_indices,
      page_indices,
      cu_q_lens,
      distribution,
      attention_sinks,
      swa_accumution,
      swa_l,
      swa_m,
  )
  return inputs, actual_tokens


def assert_matches_reference(
    actual: jax.Array,
    inputs: tuple[jax.Array, ...],
    actual_tokens: int,
    *,
    sm_scale: float = 1.0,
) -> None:
  """Asserts that `actual` matches the reference output on valid tokens."""
  expected = reference.sparse_ragged_paged_attention(
      *inputs, sm_scale=sm_scale
  )
  assert actual.shape == expected.shape, (
      f"Expected shape {expected.shape}, got {actual.shape}."
  )
  np.testing.assert_allclose(
      actual[:actual_tokens],
      expected[:actual_tokens],
      rtol=RTOL,
      atol=ATOL,
  )


class SparseMlaTestBase(parameterized.TestCase):
  """Shared correctness suite for SparseMLA operator implementations."""

  def __init__(self, *args, sparse_mla_fn):
    super().__init__(*args)
    self._sparse_mla_fn = sparse_mla_fn

  @parameterized.named_parameters(
      dict(testcase_name="unpadded_topk1024", topk=1024, pad_tokens=False),
      dict(testcase_name="padded_topk1024", topk=1024, pad_tokens=True),
      dict(testcase_name="unpadded_topk512", topk=512, pad_tokens=False),
      dict(testcase_name="unpadded_topk128", topk=128, pad_tokens=False),
  )
  def test_correctness(self, topk: int, pad_tokens: bool):
    inputs, actual_tokens = make_inputs(topk=topk, pad_tokens=pad_tokens)
    actual = self._sparse_mla_fn(*inputs, sm_scale=1.0)
    assert_matches_reference(actual, inputs, actual_tokens)
