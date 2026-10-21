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
"""Shared correctness tests for MLA-SWA implementations."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.dsv4_compressor import csa_cache_layout
from tokamax._src.ops.experimental.mla_swa import reference

ATOL = RTOL = 0.1
HEAD_DIM = 512
PAGE_SIZE = 16
SLIDING_WINDOW = 16


def clone_inputs(inputs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
  """Returns `inputs` with a fresh copy of `cache_kv` (which the kernel donates)."""
  return (inputs[0], inputs[1], jnp.copy(inputs[2]), *inputs[3:])


def make_inputs(
    *,
    batch_size: int = 8,
    num_heads: int = 128,
    page_size: int = PAGE_SIZE,
    sliding_window: int = SLIDING_WINDOW,
    pad_tokens: bool = False,
    int32_cache: bool = False,
    non_causal_block: bool = False,
    seed: int = 0,
) -> tuple[tuple[jax.Array, ...], int]:
  """Generates deterministic MLA-SWA inputs and valid token count."""
  rng = np.random.default_rng(seed)

  if non_causal_block:
    num_decode_seqs = 0
    q_lens = np.array(
        [5 if i % 2 == 0 else 6 for i in range(batch_size)], dtype=np.int32
    )
    ctx_lens = np.array(
        [[1, 8, 40][i % 3] for i in range(batch_size)], dtype=np.int32
    )
  else:
    num_decode_seqs = batch_size // 2
    prefill_lens = rng.integers(
        sliding_window // 2,
        sliding_window * 2,
        size=(batch_size - num_decode_seqs,),
        dtype=np.int32,
    )
    q_lens = np.concatenate(
        [np.ones((num_decode_seqs,), dtype=np.int32), prefill_lens]
    )
    ctx_lens = rng.integers(
        sliding_window // 2,
        sliding_window * 2,
        size=(batch_size,),
        dtype=np.int32,
    )

  kv_lens_np = ctx_lens + q_lens
  kv_lens = jnp.asarray(kv_lens_np, dtype=jnp.int32)
  cu_q_lens = jnp.concatenate([
      jnp.array([0], dtype=jnp.int32),
      jnp.cumulative_sum(jnp.asarray(q_lens, dtype=jnp.int32)),
  ])

  actual_tokens = int(np.sum(q_lens))
  total_tokens = (
      ((actual_tokens + 16 + 31) // 32) * 32 if pad_tokens else actual_tokens
  )

  q = jnp.asarray(
      rng.random(size=(total_tokens, num_heads, HEAD_DIM), dtype=np.float32),
      dtype=jnp.bfloat16,
  )
  new_kv = jnp.asarray(
      rng.random(size=(total_tokens, HEAD_DIM), dtype=np.float32),
      dtype=jnp.bfloat16,
  )

  pages_per_seq = (sliding_window * 10 + page_size - 1) // page_size
  total_pages = batch_size * pages_per_seq
  page_indices = jnp.arange(total_pages, dtype=jnp.int32)

  physical_tokens_per_page = page_size + 4
  init_tokens = jnp.asarray(
      rng.random(
          size=(total_pages * physical_tokens_per_page, HEAD_DIM),
          dtype=np.float32,
      ),
      dtype=jnp.bfloat16,
  )
  cache_slabs = reference.encode_kv_tokens(init_tokens).reshape(
      total_pages, physical_tokens_per_page * 2, 4, 128
  )
  cache_kv = (
      csa_cache_layout.slabs_to_words(cache_slabs)
      if int32_cache
      else cache_slabs
  )

  distribution = jnp.array(
      [num_decode_seqs, num_decode_seqs, batch_size], dtype=jnp.int32
  )
  attention_sinks = jnp.asarray(
      rng.random(size=(num_heads,), dtype=np.float32), dtype=jnp.float32
  )

  inputs = (
      q,
      new_kv,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      attention_sinks,
  )
  return inputs, actual_tokens


def assert_matches_reference(
    actual: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    inputs: tuple[jax.Array, ...],
    actual_tokens: int,
    *,
    sliding_window: int = SLIDING_WINDOW,
    logical_page_size: int = PAGE_SIZE,
    sm_scale: float = 1.0,
    unnormalized_output: bool = False,
    non_causal_block: bool = False,
) -> None:
  """Asserts that `actual` matches the reference outputs on valid tokens/pages."""
  expected = reference.mla_sliding_window_ragged_paged_attention(
      *clone_inputs(inputs),
      sliding_window=sliding_window,
      logical_page_size=logical_page_size,
      sm_scale=sm_scale,
      unnormalized_output=unnormalized_output,
      non_causal_block=non_causal_block,
  )
  out_act, cache_act, l_act, m_act = actual
  out_exp, cache_exp, l_exp, m_exp = expected

  assert out_act.shape == out_exp.shape, (
      f"Expected out shape {out_exp.shape}, got {out_act.shape}."
  )
  assert cache_act.shape == cache_exp.shape, (
      f"Expected cache shape {cache_exp.shape}, got {cache_act.shape}."
  )
  assert l_act.shape == l_exp.shape, (
      f"Expected l shape {l_exp.shape}, got {l_act.shape}."
  )
  assert m_act.shape == m_exp.shape, (
      f"Expected m shape {m_exp.shape}, got {m_act.shape}."
  )

  np.testing.assert_allclose(
      out_act[:actual_tokens],
      out_exp[:actual_tokens],
      rtol=RTOL,
      atol=ATOL,
  )
  np.testing.assert_allclose(
      l_act[:actual_tokens],
      l_exp[:actual_tokens],
      rtol=RTOL,
      atol=ATOL,
  )
  np.testing.assert_allclose(
      m_act[:actual_tokens],
      m_exp[:actual_tokens],
      rtol=RTOL,
      atol=ATOL,
  )
  np.testing.assert_array_equal(cache_act, cache_exp)


class MlaSwaTestBase(parameterized.TestCase):
  """Shared correctness suite for MLA-SWA operator implementations."""

  def __init__(self, *args, mla_swa_fn):
    super().__init__(*args)
    self._mla_swa_fn = mla_swa_fn

  @parameterized.named_parameters(
      dict(
          testcase_name="unpadded_u8",
          pad_tokens=False,
          int32_cache=False,
          unnormalized_output=False,
          non_causal_block=False,
      ),
      dict(
          testcase_name="padded_u8",
          pad_tokens=True,
          int32_cache=False,
          unnormalized_output=False,
          non_causal_block=False,
      ),
      dict(
          testcase_name="unpadded_int32",
          pad_tokens=False,
          int32_cache=True,
          unnormalized_output=False,
          non_causal_block=False,
      ),
      dict(
          testcase_name="unnormalized_output",
          pad_tokens=False,
          int32_cache=False,
          unnormalized_output=True,
          non_causal_block=False,
      ),
      dict(
          testcase_name="non_causal_block",
          pad_tokens=False,
          int32_cache=False,
          unnormalized_output=False,
          non_causal_block=True,
      ),
  )
  def test_correctness(
      self,
      pad_tokens: bool,
      int32_cache: bool,
      unnormalized_output: bool,
      non_causal_block: bool,
  ):
    inputs, actual_tokens = make_inputs(
        pad_tokens=pad_tokens,
        int32_cache=int32_cache,
        non_causal_block=non_causal_block,
    )
    actual = self._mla_swa_fn(
        *clone_inputs(inputs),
        sliding_window=SLIDING_WINDOW,
        logical_page_size=PAGE_SIZE,
        sm_scale=1.0,
        unnormalized_output=unnormalized_output,
        non_causal_block=non_causal_block,
    )
    assert_matches_reference(
        actual,
        inputs,
        actual_tokens,
        sliding_window=SLIDING_WINDOW,
        logical_page_size=PAGE_SIZE,
        sm_scale=1.0,
        unnormalized_output=unnormalized_output,
        non_causal_block=non_causal_block,
    )
