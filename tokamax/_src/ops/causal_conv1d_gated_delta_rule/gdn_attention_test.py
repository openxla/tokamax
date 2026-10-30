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
"""Tests for GDN kernel synced from tpu-inference."""

import dataclasses
import functools
import math
import sys
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.causal_conv1d_gated_delta_rule import tiling
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper


_RACE_REPEATS = flags.DEFINE_integer(
    "race_repeats",
    2,
    "Number of repeated kernel launches in state-slot race parity tests.",
)
_DECODE_TILE = flags.DEFINE_integer(
    "gdn_decode_tile_size",
    0,
    "Optional decode_tile_size override (0 uses tiling.get_tile_sizes).",
)


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  x_f32 = x.astype(jnp.float32)
  norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
  return (x_f32 / norm).astype(x.dtype)


def gdn_attention_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray | None,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    read_state_indices: jnp.ndarray | None = None,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Runs reference GDN attention sequence-by-sequence in eager mode.

  Bypasses XLA ragged masking complexity required for jax.jit by slicing
  sequence chunks directly in Python loop using query_start_loc boundaries.
  """

  num_tokens = qkv.shape[0]
  num_valid_seqs = int(distribution[2])

  if read_state_indices is None:
    read_state_indices = state_indices

  out_mixed_qkv = jnp.zeros_like(qkv)
  new_conv_state = jnp.array(conv_state)
  new_recurrent_state = jnp.array(recurrent_state)
  output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)

  for req_idx in range(num_valid_seqs):
    s_read = int(read_state_indices[req_idx])
    s_write = int(state_indices[req_idx])
    start = int(query_start_loc[req_idx])
    end = int(query_start_loc[req_idx + 1])
    query_len = end - start
    if query_len <= 0:
      continue

    has_init = bool((seq_lens[req_idx] - query_len) > 0)
    if has_init:
      c_state = conv_state[s_read]
    else:
      c_state = jnp.zeros_like(conv_state[s_read])

    # Part 1: Conv1D
    X = qkv[start:end]
    x_full = jnp.concatenate([c_state, X], axis=0)
    acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
    for k in range(kernel_size):
      acc += (
          x_full[k : k + query_len].astype(jnp.float32)
          * conv_weight[:, 0, k].astype(jnp.float32)[None, :]
      )
    if conv_bias is not None:
      acc += conv_bias.astype(jnp.float32)[None, :]
    conv_out = acc.astype(qkv.dtype)
    new_conv_state = new_conv_state.at[s_write].set(
        x_full[-(kernel_size - 1) :]
    )
    out_mixed_qkv = out_mixed_qkv.at[start:end].set(conv_out)

  out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

  for req_idx in range(num_valid_seqs):
    s_read = int(read_state_indices[req_idx])
    s_write = int(state_indices[req_idx])
    start = int(query_start_loc[req_idx])
    end = int(query_start_loc[req_idx + 1])
    query_len = end - start
    if query_len <= 0:
      continue

    has_init = bool((seq_lens[req_idx] - query_len) > 0)
    if has_init:
      r_state = recurrent_state[s_read]
    else:
      r_state = jnp.zeros_like(recurrent_state[s_read])

    qkv_seq = out_mixed_qkv[start:end]
    key_dim = n_kq * d_k
    q_seq = qkv_seq[:, :key_dim].reshape(query_len, n_kq, d_k)
    k_seq = qkv_seq[:, key_dim : key_dim * 2].reshape(query_len, n_kq, d_k)
    v_seq = qkv_seq[:, key_dim * 2 :].reshape(query_len, n_v, d_v)

    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
      q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
      k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)

    q_seq = _l2_normalize(q_seq)
    k_seq = _l2_normalize(k_seq)
    v_seq = v_seq.astype(jnp.float32)

    b_seq = b[start:end]
    a_seq = a[start:end]

    beta_seq = jax.nn.sigmoid(b_seq.astype(jnp.float32))
    g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
        a_seq.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
    )

    def step_fn(carry_state, xs):
      q_t, k_t, v_t, beta_t, g_t = xs
      q_t = q_t * (d_k**-0.5)
      exp_g = jnp.exp(g_t)

      k_state = jnp.einsum("hd, hdm -> hm", k_t, carry_state)
      v_diff = v_t - exp_g[:, None] * k_state
      v_new = beta_t[:, None] * v_diff

      q_state = jnp.einsum("hd, hdm -> hm", q_t, carry_state)
      q_k = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
      out_step = exp_g[:, None] * q_state + q_k * v_new

      k_v_new = jnp.einsum("hd, hm -> hdm", k_t, v_new)
      new_state = carry_state * exp_g[:, None, None] + k_v_new

      return new_state.astype(recurrent_state.dtype), out_step.astype(qkv.dtype)

    final_r_state, out_seq = jax.lax.scan(
        step_fn, r_state, (q_seq, k_seq, v_seq, beta_seq, g_seq)
    )
    new_recurrent_state = new_recurrent_state.at[s_write].set(final_r_state)
    output = output.at[start:end].set(out_seq.reshape(query_len, n_v * d_v))

  return (new_conv_state, new_recurrent_state), output


def gdn_attention_spec_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray | None,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    read_offsets: jnp.ndarray,
    num_spec_seqs: int,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Reference for SPEC mode: per-token state checkpoints with read offsets.

  Sequence s reads its initial state from slot
  `state_indices[s] + read_offsets[s]` and writes the state after window
  position t to slot `state_indices[s] + t`.
  """
  num_tokens = qkv.shape[0]
  new_conv_state = jnp.array(conv_state)
  new_recurrent_state = jnp.array(recurrent_state)
  output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)

  for req_idx in range(num_spec_seqs):
    base = int(state_indices[req_idx])
    read_slot = base + int(read_offsets[req_idx])
    start = int(query_start_loc[req_idx])
    end = int(query_start_loc[req_idx + 1])
    query_len = end - start
    if query_len <= 0:
      continue

    has_init = bool((seq_lens[req_idx] - query_len) > 0)
    c_state = (
        conv_state[read_slot]
        if has_init
        else jnp.zeros_like(conv_state[read_slot])
    )
    r_state = (
        recurrent_state[read_slot]
        if has_init
        else jnp.zeros_like(recurrent_state[read_slot])
    )

    # Conv1D with per-token windows.
    X = qkv[start:end]
    x_full = jnp.concatenate([c_state, X], axis=0)
    acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
    for k in range(kernel_size):
      acc += (
          x_full[k : k + query_len].astype(jnp.float32)
          * conv_weight[:, 0, k].astype(jnp.float32)[None, :]
      )
    if conv_bias is not None:
      acc += conv_bias.astype(jnp.float32)[None, :]
    conv_out = jax.nn.silu(acc).astype(qkv.dtype)
    for t in range(query_len):
      new_conv_state = new_conv_state.at[base + t].set(
          x_full[t + 1 : t + kernel_size]
      )

    # Recurrent GDN with per-token states.
    key_dim = n_kq * d_k
    q_seq = conv_out[:, :key_dim].reshape(query_len, n_kq, d_k)
    k_seq = conv_out[:, key_dim : key_dim * 2].reshape(query_len, n_kq, d_k)
    v_seq = conv_out[:, key_dim * 2 :].reshape(query_len, n_v, d_v)

    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
      q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
      k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)

    q_seq = _l2_normalize(q_seq)
    k_seq = _l2_normalize(k_seq)
    v_seq = v_seq.astype(jnp.float32)

    beta_seq = jax.nn.sigmoid(b[start:end].astype(jnp.float32))
    g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
        a[start:end].astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
    )

    def step_fn(carry_state, xs):
      q_t, k_t, v_t, beta_t, g_t = xs
      q_t = q_t * (d_k**-0.5)
      exp_g = jnp.exp(g_t)

      k_state = jnp.einsum("hd, hdm -> hm", k_t, carry_state)
      v_diff = v_t - exp_g[:, None] * k_state
      v_new = beta_t[:, None] * v_diff

      q_state = jnp.einsum("hd, hdm -> hm", q_t, carry_state)
      q_k = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
      out_step = exp_g[:, None] * q_state + q_k * v_new

      k_v_new = jnp.einsum("hd, hm -> hdm", k_t, v_new)
      new_state = carry_state * exp_g[:, None, None] + k_v_new
      new_state = new_state.astype(recurrent_state.dtype)

      return new_state, (out_step.astype(qkv.dtype), new_state)

    _, (out_seq, states_seq) = jax.lax.scan(
        step_fn, r_state, (q_seq, k_seq, v_seq, beta_seq, g_seq)
    )
    for t in range(query_len):
      new_recurrent_state = new_recurrent_state.at[base + t].set(states_seq[t])
    output = output.at[start:end].set(out_seq.reshape(query_len, n_v * d_v))

  return (new_conv_state, new_recurrent_state), output


class GDNAttentionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 6:
        self.skipTest("Pallas TPU kernel requires TPU v6 or newer.")
    except Exception:
      self.skipTest("Failed to get TPU info.")

  @parameterized.named_parameters(
      dict(
          testcase_name="prefill",
          max_reqs=1,
          lengths=[8192],
          q_loc=[0, 8192],
          distribution=[0, 0, 3],
      ),
      dict(
          testcase_name="mixed",
          max_reqs=3,
          lengths=[256, 128, 128],
          q_loc=[0, 256, 384, 512],
          distribution=[0, 3, 3],
      ),
      dict(
          testcase_name="decode_only",
          max_reqs=64,
          lengths=[1] * 64,
          q_loc=list(range(65)),
          distribution=[64, 64, 64],
      ),
      dict(
          testcase_name="mixed_prefill_decode",
          max_reqs=11,
          lengths=[1] * 8 + [128, 128, 256],
          q_loc=[0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520],
          distribution=[8, 11, 11],
      ),
      dict(
          testcase_name="padded_mixed_prefill",
          max_reqs=16,
          lengths=[128, 64, 32, 16, 8],
          q_loc=[0, 128, 192, 224, 240, 248] + [1] * 11,
          distribution=[0, 5, 5],
      ),
      dict(
          testcase_name="padded_decode_only",
          max_reqs=512,
          lengths=[1] * 64,
          q_loc=list(range(65)) + [1] * 448,
          distribution=[64, 64, 64],
      ),
      dict(
          testcase_name="prefill_fused",
          max_reqs=1,
          lengths=[8192],
          q_loc=[0, 8192],
          distribution=[0, 0, 3],
      ),
      dict(
          testcase_name="mixed_fused",
          max_reqs=3,
          lengths=[256, 128, 128],
          q_loc=[0, 256, 384, 512],
          distribution=[0, 3, 3],
      ),
  )
  def test_run_jax_gdn_attention_local(
      self, max_reqs, lengths, q_loc, distribution
  ):
    kq_head_dim = 128
    v_head_dim = 128
    n_kq = 2
    n_v = 8
    kernel_size = 4

    num_tokens = sum(lengths)

    q_loc = jnp.array(q_loc)
    distribution = jnp.array(distribution, dtype=jnp.int32)

    # recurrent_state[0] and conv_state[0] are reserved for null blocks
    # (invalid / padded tokens). so start with index 1
    state_indices = jnp.arange(1, max_reqs + 1)
    num_blocks = max_reqs + 1

    rngs = iter(jax.random.split(jax.random.key(0), 12))

    query = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
    key = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
    value = jax.random.normal(next(rngs), (num_tokens, n_v * v_head_dim))
    b = jax.random.normal(next(rngs), (num_tokens, n_v))
    a = jax.random.normal(next(rngs), (num_tokens, n_v))

    conv_state_q = jnp.zeros((num_blocks, kernel_size - 1, n_kq * kq_head_dim))
    conv_state_k = jnp.zeros((num_blocks, kernel_size - 1, n_kq * kq_head_dim))
    conv_state_v = jnp.zeros((num_blocks, kernel_size - 1, n_v * v_head_dim))
    recurrent_state = jnp.zeros((num_blocks, n_v, kq_head_dim, v_head_dim))

    conv_weight_q = jax.random.normal(
        next(rngs), (n_kq * kq_head_dim, 1, kernel_size)
    )
    conv_weight_k = jax.random.normal(
        next(rngs), (n_kq * kq_head_dim, 1, kernel_size)
    )
    conv_weight_v = jax.random.normal(
        next(rngs), (n_v * v_head_dim, 1, kernel_size)
    )

    conv_bias_q = jax.random.normal(next(rngs), (n_kq * kq_head_dim,))
    conv_bias_k = jax.random.normal(next(rngs), (n_kq * kq_head_dim,))
    conv_bias_v = jax.random.normal(next(rngs), (n_v * v_head_dim,))

    A_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(jax.random.key(0), (n_v,))

    mixed_qkv = jnp.concatenate([query, key, value], axis=-1)
    conv_state = jnp.concatenate(
        [conv_state_q, conv_state_k, conv_state_v], axis=-1
    )
    conv_weight = jnp.concatenate(
        [conv_weight_q, conv_weight_k, conv_weight_v], axis=0
    )
    conv_bias = jnp.concatenate(
        [conv_bias_q, conv_bias_k, conv_bias_v], axis=-1
    )

    gdn_attention_jitted = jax.jit(
        wrapper.fused_conv1d_gdn,
        static_argnames=[
            "n_kq",
            "n_v",
            "d_k",
            "d_v",
            "kernel_size",
            "num_spec_tokens",
        ],
    )

    # All sequences in this test start from a fresh slot; the existing
    # parametrizations don't exercise prefix-cache-hit / chunked-prefill
    # continuation. ``seq_lens == query_lens`` (context_len = 0)
    # reproduces the prior behavior (zero initial state regardless of
    # slot contents).
    seq_lens = jnp.asarray(
        q_loc[1 : max_reqs + 1] - q_loc[:max_reqs], dtype=jnp.int32
    )

    common_kwargs = dict(
        qkv=mixed_qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        read_state_indices=state_indices,
        n_kq=n_kq,
        n_v=n_v,
        d_k=kq_head_dim,
        d_v=v_head_dim,
        kernel_size=kernel_size,
    )

    # Run ref
    new_states_ref, output_ref = gdn_attention_ref(
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        **common_kwargs,
    )

    # Run chunked
    new_states_chunked, output_chunked = gdn_attention_jitted(
        states=(conv_state, recurrent_state),
        **common_kwargs,
    )

    # Compare results
    np.testing.assert_allclose(output_chunked, output_ref, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(
        new_states_chunked[0], new_states_ref[0], rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        new_states_chunked[1], new_states_ref[1], rtol=2e-2, atol=2e-2
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="spec_windows",
          # 3 verify windows: full (5), partial drafts (3), no drafts (1).
          spec_lengths=[5, 3, 1],
          read_offsets=[2, 0, 4],
          prefill_lengths=[],
      ),
      dict(
          testcase_name="spec_windows_padded",
          # More windows than one tile (decode_tile_size=4).
          spec_lengths=[5, 1, 3, 2, 4],
          read_offsets=[0, 1, 2, 3, 4],
          prefill_lengths=[],
      ),
      dict(
          testcase_name="spec_and_prefill",
          spec_lengths=[5, 2],
          read_offsets=[3, 1],
          prefill_lengths=[64],
      ),
  )
  def test_spec_mode_checkpoints(
      self, spec_lengths, read_offsets, prefill_lengths
  ):
    """SPEC mode: initial state comes from `base + read_offset` and one

    state checkpoint per window position lands at `base + t`, matching a
    token-by-token eager reference. Prefill sequences in the same batch
    keep the original PER_SEQ behavior (read/write the base slot).
    """
    kq_head_dim = 128
    v_head_dim = 128
    n_kq = 2
    n_v = 8
    kernel_size = 4
    num_spec_tokens = 4
    window = num_spec_tokens + 1

    lengths = list(spec_lengths) + list(prefill_lengths)
    num_seqs = len(lengths)
    num_spec_seqs = len(spec_lengths)
    num_tokens = sum(lengths)
    q_loc = jnp.array(
        np.concatenate([[0], np.cumsum(lengths)]), dtype=jnp.int32
    )
    distribution = jnp.array(
        [num_spec_seqs, num_spec_seqs, num_seqs], dtype=jnp.int32
    )

    # Slot groups of `window` consecutive slots per sequence; slot 0 is
    # the null block.
    state_indices = jnp.array(
        [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
    )
    num_blocks = 1 + num_seqs * window
    read_offsets_arr = jnp.array(
        list(read_offsets) + [0] * len(prefill_lengths), dtype=jnp.int32
    )

    rngs = iter(jax.random.split(jax.random.key(3), 12))
    conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim

    mixed_qkv = jax.random.normal(next(rngs), (num_tokens, conv_dim))
    b = jax.random.normal(next(rngs), (num_tokens, n_v))
    a = jax.random.normal(next(rngs), (num_tokens, n_v))

    # Every slot holds a distinct random state so a read from the wrong
    # slot (or a write to the wrong slot) is caught.
    conv_state = jax.random.normal(
        next(rngs), (num_blocks, kernel_size - 1, conv_dim)
    )
    recurrent_state = jax.random.normal(
        next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim)
    )

    conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
    conv_bias = jax.random.normal(next(rngs), (conv_dim,))
    A_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v,))

    # All sequences continue an existing context (has_initial=True) for
    # the spec windows; prefills start fresh (context_len = 0).
    seq_lens = jnp.array(
        [16 + length for length in spec_lengths] + list(prefill_lengths),
        dtype=jnp.int32,
    )

    run_jitted = jax.jit(
        wrapper.fused_conv1d_gdn,
        static_argnames=[
            "n_kq",
            "n_v",
            "d_k",
            "d_v",
            "kernel_size",
            "num_spec_tokens",
        ],
    )

    (new_conv, new_rec), output = run_jitted(
        mixed_qkv,
        b,
        a,
        (conv_state, recurrent_state),
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        q_loc,
        state_indices,
        distribution,
        seq_lens,
        state_indices,
        read_offsets_arr,
        n_kq=n_kq,
        n_v=n_v,
        d_k=kq_head_dim,
        d_v=v_head_dim,
        kernel_size=kernel_size,
        num_spec_tokens=num_spec_tokens,
    )

    # Reference: spec windows token-by-token with checkpoints.
    (ref_conv, ref_rec), ref_output = gdn_attention_spec_ref(
        qkv=mixed_qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        read_offsets=read_offsets_arr,
        num_spec_seqs=num_spec_seqs,
        seq_lens=seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=kq_head_dim,
        d_v=v_head_dim,
        kernel_size=kernel_size,
    )
    # Reference for the prefill tail: original eager reference over the
    # same buffers (spec sequences excluded via the distribution).
    if prefill_lengths:
      spec_token_end = int(q_loc[num_spec_seqs])
      (ref_conv, ref_rec), prefill_output = gdn_attention_ref(
          qkv=mixed_qkv[spec_token_end:],
          b=b[spec_token_end:],
          a=a[spec_token_end:],
          conv_state=ref_conv,
          recurrent_state=ref_rec,
          conv_weight=conv_weight,
          conv_bias=conv_bias,
          a_log=A_log,
          dt_bias=dt_bias,
          query_start_loc=q_loc[num_spec_seqs:] - q_loc[num_spec_seqs],
          state_indices=state_indices[num_spec_seqs:],
          distribution=jnp.array(
              [0, 0, num_seqs - num_spec_seqs], dtype=jnp.int32
          ),
          seq_lens=seq_lens[num_spec_seqs:],
          n_kq=n_kq,
          n_v=n_v,
          d_k=kq_head_dim,
          d_v=v_head_dim,
          kernel_size=kernel_size,
      )
      ref_output = ref_output.at[spec_token_end:].set(prefill_output)

    np.testing.assert_allclose(output, ref_output, rtol=2e-2, atol=2e-2)

    # Verify every written checkpoint slot; untouched slots must keep
    # their original contents.
    touched = set()
    for i, length in enumerate(spec_lengths):
      base = int(state_indices[i])
      touched.update(range(base, base + length))
    for i in range(len(prefill_lengths)):
      touched.add(int(state_indices[num_spec_seqs + i]))
    for slot in range(num_blocks):
      expected_conv = ref_conv[slot] if slot in touched else conv_state[slot]
      expected_rec = (
          ref_rec[slot] if slot in touched else (recurrent_state[slot])
      )
      np.testing.assert_allclose(
          new_conv[slot],
          expected_conv,
          rtol=2e-2,
          atol=2e-2,
          err_msg=f"conv checkpoint mismatch at slot {slot}",
      )
      np.testing.assert_allclose(
          new_rec[slot],
          expected_rec,
          rtol=2e-2,
          atol=2e-2,
          err_msg=f"recurrent checkpoint mismatch at slot {slot}",
      )

  def test_has_initial_state_zeros_stale_slot(self):
    """Ensure stale states are ignore by new request.

    A new prefill landing on a slot whose previous tenant left
    non-zero state must produce the same output and final state as it
    would on a fresh-zero slot. This exercises the production bug fixed
    by the `has_initial_state` plumbing: vLLM's mamba pool reuses
    freed slots without clearing them, and previously the TPU GDN
    kernel consumed the stale state, silently corrupting the new
    request's recurrent trajectory.
    """
    kq_head_dim = 128
    v_head_dim = 128
    n_kq = 2
    n_v = 8
    kernel_size = 4

    # Two requests, one prefill of 64 tokens each.
    max_reqs = 2
    lengths = [64, 64]
    q_loc = jnp.array([0, 64, 128])
    distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
    num_tokens = sum(lengths)

    state_indices = jnp.arange(1, max_reqs + 1)
    num_blocks = max_reqs + 1

    rngs = iter(jax.random.split(jax.random.key(7), 12))
    query = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
    key = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
    value = jax.random.normal(next(rngs), (num_tokens, n_v * v_head_dim))
    b = jax.random.normal(next(rngs), (num_tokens, n_v))
    a = jax.random.normal(next(rngs), (num_tokens, n_v))

    conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
    conv_state_fresh = jnp.zeros((num_blocks, kernel_size - 1, conv_dim))
    recurrent_state_fresh = jnp.zeros(
        (num_blocks, n_v, kq_head_dim, v_head_dim)
    )

    # Build a "stale" pair where the slots that the two new requests
    # land on are filled with arbitrary nonzero values (simulating a
    # prior request that finished without the pool clearing the slot).
    stale_conv = jax.random.normal(
        next(rngs), (num_blocks, kernel_size - 1, conv_dim)
    )
    stale_recurrent = jax.random.normal(
        next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim)
    )
    # Slot 0 is the null block; leave it zero.
    conv_state_stale = conv_state_fresh.at[1:].set(stale_conv[1:])
    recurrent_state_stale = recurrent_state_fresh.at[1:].set(
        stale_recurrent[1:]
    )

    conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
    conv_bias = jax.random.normal(next(rngs), (conv_dim,))
    A_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v,))

    mixed_qkv = jnp.concatenate([query, key, value], axis=-1)

    run_jitted = jax.jit(
        wrapper.fused_conv1d_gdn,
        static_argnames=[
            "n_kq",
            "n_v",
            "d_k",
            "d_v",
            "kernel_size",
            "num_spec_tokens",
        ],
    )

    # Both requests are brand new — no prior context. seq_lens equals
    # query_lens so context_len = 0 → has_initial_state = False.
    seq_lens_new = jnp.asarray(lengths, dtype=jnp.int32)

    common_kwargs = dict(
        qkv=mixed_qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens_new,
        read_state_indices=state_indices,
        n_kq=n_kq,
        n_v=n_v,
        d_k=kq_head_dim,
        d_v=v_head_dim,
        kernel_size=kernel_size,
    )

    # Reference run: fresh-zero slots.
    (new_conv_fresh, new_rec_fresh), output_fresh = run_jitted(
        states=(conv_state_fresh, recurrent_state_fresh),
        **common_kwargs,
    )
    # Stale-slot run: same inputs, but the slots already contain a
    # prior request's state. With the fix, has_initial_state=False
    # masks that out — outputs and the writeback at the active slots
    # must match the fresh-zero run.
    (new_conv_stale, new_rec_stale), output_stale = run_jitted(
        states=(conv_state_stale, recurrent_state_stale),
        **common_kwargs,
    )

    np.testing.assert_allclose(output_fresh, output_stale, rtol=1e-5, atol=1e-5)
    # Compare the active slots (1..max_reqs+1); slot 0 (null) is
    # untouched in both runs and inactive slots are unused.
    np.testing.assert_allclose(
        new_conv_fresh[1 : max_reqs + 1],
        new_conv_stale[1 : max_reqs + 1],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        new_rec_fresh[1 : max_reqs + 1],
        new_rec_stale[1 : max_reqs + 1],
        rtol=1e-5,
        atol=1e-5,
    )

  def test_has_initial_state_preserves_continuation(self):
    """Ensure existing slots are read during multi-step execution.

    When `has_initial_state[i]` is True, the kernel must use the
    slot's existing state as the prefill's initial state. This is the
    chunked-prefill / prefix-cache continuation path: a prior step
    wrote a recurrent/conv state to the slot, and the next prefill
    chunk for the same request must continue from that state — not
    from zero. Compares against running the equivalent single-shot
    prefill on the concatenated token stream from a zero state.
    """

    kq_head_dim = 128
    v_head_dim = 128
    n_kq = 2
    n_v = 8
    kernel_size = 4

    # One request, prefill split into two halves of 32 tokens each.
    # Step A: tokens [0, 32) starting from zero state.
    # Step B: tokens [32, 64) starting from Step A's final state.
    # Reference: tokens [0, 64) in a single shot from zero state.
    half = 32
    full = 64
    state_indices = jnp.array([1])
    num_blocks = 2

    rngs = iter(jax.random.split(jax.random.key(11), 12))
    query = jax.random.normal(next(rngs), (full, n_kq * kq_head_dim))
    key = jax.random.normal(next(rngs), (full, n_kq * kq_head_dim))
    value = jax.random.normal(next(rngs), (full, n_v * v_head_dim))
    b = jax.random.normal(next(rngs), (full, n_v))
    a = jax.random.normal(next(rngs), (full, n_v))

    conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
    conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
    conv_bias = jax.random.normal(next(rngs), (conv_dim,))
    A_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v,))

    mixed_qkv_full = jnp.concatenate([query, key, value], axis=-1)
    mixed_qkv_a = mixed_qkv_full[:half]
    mixed_qkv_b = mixed_qkv_full[half:]

    conv_state_zero = jnp.zeros((num_blocks, kernel_size - 1, conv_dim))
    recurrent_state_zero = jnp.zeros((num_blocks, n_v, kq_head_dim, v_head_dim))

    run_jitted = jax.jit(
        wrapper.fused_conv1d_gdn,
        static_argnames=[
            "n_kq",
            "n_v",
            "d_k",
            "d_v",
            "kernel_size",
            "num_spec_tokens",
        ],
    )
    common_static = dict(
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=A_log,
        dt_bias=dt_bias,
        state_indices=state_indices,
        read_state_indices=state_indices,
        n_kq=n_kq,
        n_v=n_v,
        d_k=kq_head_dim,
        d_v=v_head_dim,
        kernel_size=kernel_size,
    )

    # Single-shot reference (all 64 tokens, zero state, has_initial=False
    # encoded as seq_lens == query_lens == [full]).
    (_, _), output_ref = run_jitted(
        qkv=mixed_qkv_full,
        b=b,
        a=a,
        states=(conv_state_zero, recurrent_state_zero),
        query_start_loc=jnp.array([0, full]),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([full], dtype=jnp.int32),
        **common_static,
    )

    # Step A: first 32 tokens, zero state, has_initial=False.
    (conv_after_a, rec_after_a), output_a = run_jitted(
        qkv=mixed_qkv_a,
        b=b[:half],
        a=a[:half],
        states=(conv_state_zero, recurrent_state_zero),
        query_start_loc=jnp.array([0, half]),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([half], dtype=jnp.int32),
        **common_static,
    )

    # Step B: next 32 tokens, slot now holds Step A's state.
    # seq_lens=[full] with query_lens=[half] gives context_len=half>0,
    # i.e., has_initial=True so the kernel continues from that state.
    (_, _), output_b = run_jitted(
        qkv=mixed_qkv_b,
        b=b[half:],
        a=a[half:],
        states=(conv_after_a, rec_after_a),
        query_start_loc=jnp.array([0, half]),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([full], dtype=jnp.int32),
        **common_static,
    )

    # Step A's output must match the first half of the single-shot
    # reference; Step B (continuation) must match the second half.
    np.testing.assert_allclose(
        output_a, output_ref[:half], rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        output_b, output_ref[half:], rtol=2e-2, atol=2e-2
    )

  def test_split_read_write_state_slots(self):
    """Resume from one state slot while checkpointing into another.

    This is the mamba prefix-caching path (`mamba_cache_mode="align"`):
    a request continues from the state block cached at the last block
    boundary and writes its new checkpoint into the block covering the
    current position. The continuation must be numerically identical to
    reading and writing one slot, and the slot that was read must be
    left untouched so other requests can still hit it in the cache.
    """

    kq_head_dim = 128
    v_head_dim = 128
    n_kq = 2
    n_v = 8
    kernel_size = 4

    half = 32
    full = 64
    # Slot 0 is the null block; step A writes slot 1, step B reads slot 1
    # and writes slot 2.
    read_slot, write_slot = 1, 2
    num_blocks = 3

    rngs = iter(jax.random.split(jax.random.key(23), 12))
    query = jax.random.normal(next(rngs), (full, n_kq * kq_head_dim))
    key = jax.random.normal(next(rngs), (full, n_kq * kq_head_dim))
    value = jax.random.normal(next(rngs), (full, n_v * v_head_dim))
    b = jax.random.normal(next(rngs), (full, n_v))
    a = jax.random.normal(next(rngs), (full, n_v))

    conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
    conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
    conv_bias = jax.random.normal(next(rngs), (conv_dim,))
    A_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v,))

    mixed_qkv_full = jnp.concatenate([query, key, value], axis=-1)

    conv_state_zero = jnp.zeros((num_blocks, kernel_size - 1, conv_dim))
    recurrent_state_zero = jnp.zeros((num_blocks, n_v, kq_head_dim, v_head_dim))

    run_jitted = jax.jit(
        wrapper.fused_conv1d_gdn,
        static_argnames=[
            "n_kq",
            "n_v",
            "d_k",
            "d_v",
            "kernel_size",
            "num_spec_tokens",
        ],
    )
    common_static = dict(
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=A_log,
        dt_bias=dt_bias,
        n_kq=n_kq,
        n_v=n_v,
        d_k=kq_head_dim,
        d_v=v_head_dim,
        kernel_size=kernel_size,
    )

    # Single-shot reference over all 64 tokens from a zero state.
    (_, _), output_ref = run_jitted(
        qkv=mixed_qkv_full,
        b=b,
        a=a,
        states=(conv_state_zero, recurrent_state_zero),
        query_start_loc=jnp.array([0, full]),
        state_indices=jnp.array([read_slot]),
        read_state_indices=jnp.array([read_slot]),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([full], dtype=jnp.int32),
        **common_static,
    )

    # Step A: first half from a zero state, checkpointing into read_slot.
    (conv_after_a, rec_after_a), _ = run_jitted(
        qkv=mixed_qkv_full[:half],
        b=b[:half],
        a=a[:half],
        states=(conv_state_zero, recurrent_state_zero),
        query_start_loc=jnp.array([0, half]),
        state_indices=jnp.array([read_slot]),
        read_state_indices=jnp.array([read_slot]),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([half], dtype=jnp.int32),
        **common_static,
    )

    # Step B: second half, resuming from read_slot but checkpointing into
    # write_slot — the block-boundary crossing that align mode performs.
    (conv_after_b, rec_after_b), output_b = run_jitted(
        qkv=mixed_qkv_full[half:],
        b=b[half:],
        a=a[half:],
        states=(conv_after_a, rec_after_a),
        query_start_loc=jnp.array([0, half]),
        state_indices=jnp.array([write_slot]),
        read_state_indices=jnp.array([read_slot]),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([full], dtype=jnp.int32),
        **common_static,
    )

    # Splitting the slots must not change the numerics.
    np.testing.assert_allclose(
        output_b, output_ref[half:], rtol=2e-2, atol=2e-2
    )

    # The slot that was read is the cached prefix state; leaving it intact
    # is what lets a later request hit it.
    np.testing.assert_array_equal(
        conv_after_b[read_slot], conv_after_a[read_slot]
    )
    np.testing.assert_array_equal(
        rec_after_b[read_slot], rec_after_a[read_slot]
    )

    # The new checkpoint landed in write_slot, which started out zeroed.
    self.assertTrue(
        np.any(np.asarray(rec_after_b[write_slot]) != 0.0),
        "write_slot should hold the step's final recurrent state",
    )

  def test_per_seq_rolls_read_offsets_with_start_seq(self):
    """PER_SEQ continuation behind a decode sequence honors read_offsets."""
    kq_head_dim = 128
    v_head_dim = 128
    n_kq = 2
    n_v = 8
    kernel_size = 4
    conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim

    # Seq 0: 1-token decode at slot 1 with read_offset=0.
    # Seq 1: 32-token continuation prefill with base state_indices=10,
    # read_offsets=2 (initial state lives at slot 12), writing final state to
    # base slot 10.
    num_blocks = 16
    rngs = iter(jax.random.split(jax.random.key(31), 10))
    qkv = jax.random.normal(next(rngs), (33, conv_dim))
    b = jax.random.normal(next(rngs), (33, n_v))
    a = jax.random.normal(next(rngs), (33, n_v))
    conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
    conv_bias = jax.random.normal(next(rngs), (conv_dim,))
    a_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v,))

    conv_state = jax.random.normal(
        next(rngs), (num_blocks, kernel_size - 1, conv_dim)
    )
    recurrent_state = jax.random.normal(
        next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim)
    )

    common_static = dict(
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        n_kq=n_kq,
        n_v=n_v,
        d_k=kq_head_dim,
        d_v=v_head_dim,
        kernel_size=kernel_size,
        # Same chunk schedule in both calls, so 1e-5 compares like for like.
        mixed_tile_size=32,
    )

    (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
        qkv=qkv,
        b=b,
        a=a,
        states=(jnp.array(conv_state), jnp.array(recurrent_state)),
        query_start_loc=jnp.array([0, 1, 33], dtype=jnp.int32),
        state_indices=jnp.array([1, 10], dtype=jnp.int32),
        read_state_indices=jnp.array([1, 10], dtype=jnp.int32),
        read_offsets=jnp.array([0, 2], dtype=jnp.int32),
        distribution=jnp.array([1, 2, 2], dtype=jnp.int32),
        seq_lens=jnp.array([10, 64], dtype=jnp.int32),
        **common_static,
    )

    # Reference: run Seq 1 directly reading from slot 12 (10 + 2) and writing
    # to slot 10 with read_offsets=0.
    (conv_ref, rec_ref), out_ref = wrapper.fused_conv1d_gdn(
        qkv=qkv[1:],
        b=b[1:],
        a=a[1:],
        states=(jnp.array(conv_state), jnp.array(recurrent_state)),
        query_start_loc=jnp.array([0, 32], dtype=jnp.int32),
        state_indices=jnp.array([10], dtype=jnp.int32),
        read_state_indices=jnp.array([12], dtype=jnp.int32),
        read_offsets=jnp.array([0], dtype=jnp.int32),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([64], dtype=jnp.int32),
        **common_static,
    )

    np.testing.assert_allclose(out[1:], out_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv_out[10], conv_ref[10], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rec_out[10], rec_ref[10], rtol=1e-5, atol=1e-5)

  def test_inner_kernel_explicit_p_id_matches_default(self):
    """Explicit p_id in inner_kernel overrides step.index[0]."""
    metadata = wrapper.metadata
    pl = wrapper.pl
    kq_head_dim = 128
    v_head_dim = 128
    n_kq = 2
    n_v = 8
    kernel_size = 4
    batch_size = 32
    dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
    num_blocks = 4

    rngs = iter(jax.random.split(jax.random.key(41), 10))
    qkv = jax.random.normal(next(rngs), (batch_size, dim))
    b = jax.random.normal(next(rngs), (batch_size, n_v))
    a = jax.random.normal(next(rngs), (batch_size, n_v))
    conv_weight = jax.random.normal(next(rngs), (dim, 1, kernel_size))
    conv_bias = jax.random.normal(next(rngs), (dim,))
    a_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v,))
    conv_state = jax.random.normal(
        next(rngs), (num_blocks, kernel_size - 1, dim)
    )
    recurrent_state = jax.random.normal(
        next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim)
    )

    query_start_loc = jnp.array([0, batch_size], dtype=jnp.int32)
    state_indices = jnp.array([1], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
    seq_lens = jnp.array([batch_size], dtype=jnp.int32)

    (conv_ref, rec_ref), out_ref = wrapper.fused_conv1d_gdn(
        qkv=qkv,
        b=b,
        a=a,
        states=(jnp.array(conv_state), jnp.array(recurrent_state)),
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=kq_head_dim,
        d_v=v_head_dim,
        kernel_size=kernel_size,
        mixed_tile_size=batch_size,
    )

    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_num_v_heads = tiling.align_to(n_v, num_lanes)
    num_v_pad = aligned_num_v_heads - n_v
    qkv_3d = qkv.reshape(batch_size, 1, -1)
    b_3d = jnp.pad(b, ((0, 0), (0, num_v_pad))).reshape(batch_size, 1, -1)
    a_3d = jnp.pad(a, ((0, 0), (0, num_v_pad))).reshape(batch_size, 1, -1)
    in_conv = jnp.array(conv_state).reshape(-1, kernel_size - 1, 1, dim)
    in_rec = jnp.array(recurrent_state)
    weights = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(
            weight=conv_weight.swapaxes(0, 2).astype(jnp.float32),
            bias=conv_bias.astype(jnp.float32),
        ),
        gdn=memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias),
    )

    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        batch_size=batch_size,
        kernel_size=kernel_size,
        tile_size=batch_size,
        window_size=1,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=kq_head_dim,
        v_head_dim=v_head_dim,
        dtypes=config.Dtypes(
            act_in=qkv.dtype,
            act_out=qkv.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=in_rec.dtype,
            conv_state=in_conv.dtype,
        ),
    )
    dma_meta = metadata.compute_per_seq_metadata(
        cfg=cfg,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        start_seq=distribution[0],
        end_seq=distribution[-1],
        read_indices=state_indices,
        read_offsets=jnp.zeros_like(state_indices),
    )
    # Place the valid tile record at index 1 and zero out index 0 so a kernel
    # that ignores p_id=1 and reads step.index[0]==0 sees r_size=0.
    s = memory_ref.PackedPIdRecord.STRUCT_SIZE
    shifted_records = (
        jnp.zeros((2 * s,), dtype=dma_meta.records.dtype)
        .at[s : 2 * s]
        .set(dma_meta.records[:s])
    )
    inner_meta = dataclasses.replace(dma_meta, records=shifted_records)

    def custom_outer(
        dma_meta_ref,
        inner_meta_ref,
        qkv_ref,
        b_ref,
        a_ref,
        conv_state_ref,
        recurrent_state_ref,
        _,
        weights_ref,
        out_ref,
        conv_state_out_ref,
        recurrent_state_out_ref,
        carry_conv_scratch_ref,
        carry_recurrent_scratch_ref,
    ):
      allocs = memory_ref.create_allocs(
          metadata_ref=dma_meta_ref,
          qkv_ref=qkv_ref,
          b_ref=b_ref,
          a_ref=a_ref,
          out_ref=out_ref,
          cfg=cfg,
      )
      (
          qkv_a,
          b_a,
          a_a,
          conv_in_a,
          rec_in_a,
          out_a,
          conv_out_a,
          rec_out_a,
      ) = allocs
      pipeline_fn = pltpu.emit_pipeline(
          body=functools.partial(
              wrapper.inner_kernel, cfg=cfg, p_id=jnp.int32(1)
          ),
          grid=(dma_meta_ref.num_tiles[...],),
          in_specs=(
              qkv_a.spec,
              b_a.spec,
              a_a.spec,
              conv_in_a.spec,
              rec_in_a.spec,
          ),
          out_specs=(
              out_a.spec,
              conv_out_a.spec,
              rec_out_a.spec,
          ),
          _explicit_indices=True,
      )

      @pl.with_scoped(allocations=allocs)
      def _run(allocations):
        pipeline_fn(
            qkv_ref,
            b_ref,
            a_ref,
            conv_state_ref,
            recurrent_state_ref,
            out_ref,
            conv_state_out_ref,
            recurrent_state_out_ref,
            scratches=(
                inner_meta_ref,
                weights_ref,
                carry_conv_scratch_ref,
                carry_recurrent_scratch_ref,
            ),
            allocations=allocations,
        )

      _run()

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    meta_spec = jax.tree.map(lambda _: smem_spec, dma_meta)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)
    out_init = jnp.zeros_like(cfg.get_out_shape())
    n_meta = 2 * len(dma_meta)
    aliases = {n_meta + 3: 1, n_meta + 4: 2, n_meta + 5: 0}

    out_act, conv_out, rec_out = pl.pallas_call(
        custom_outer,
        out_shape=(out_init, in_conv, in_rec),
        in_specs=(
            meta_spec,
            meta_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            weights_spec,
        ),
        out_specs=(hbm_spec, hbm_spec, hbm_spec),
        scratch_shapes=cfg.get_scratch_shape_dict(),
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
        ),
    )(
        dma_meta,
        inner_meta,
        qkv_3d,
        b_3d,
        a_3d,
        in_conv,
        in_rec,
        out_init,
        weights,
    )

    out_act = out_act.reshape(batch_size, -1)
    conv_out = conv_out.reshape(conv_state.shape)
    np.testing.assert_array_equal(out_act, out_ref)
    np.testing.assert_array_equal(conv_out[1], conv_ref[1])
    np.testing.assert_array_equal(rec_out[1], rec_ref[1])

  def test_state_pipeline_has_no_dma_race(self):
    """Pallas interpret-mode vector-clock detector finds zero DMA races."""
    if jax.__version_info__ < (0, 11, 2):
      self.skipTest(
          "Pallas TPU interpret mode requires jax >= 0.11.2 to unwrap"
          " TransformedRef views."
      )
    n_kq, n_v, d_k, d_v, kernel_size = 1, 2, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    # 6 decode sequences, then 3 one-token prefill sequences (one PER_SEQ tile
    # each), so both kernels reuse a slot two steps later.
    num_decode = 6
    num_seqs = 9
    decode_tile = 2  # 3 pipeline steps: step 2 reuses step 0's slot.

    rngs = iter(jax.random.split(jax.random.key(0), 9))
    qkv = jax.random.normal(next(rngs), (num_seqs, dim), dtype=jnp.bfloat16)
    b = jax.random.normal(next(rngs), (num_seqs, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(next(rngs), (num_seqs, n_v), dtype=jnp.bfloat16)
    conv_w = jax.random.normal(
        next(rngs), (dim, 1, kernel_size), dtype=jnp.bfloat16
    )
    conv_b = jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16)
    a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
    dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
    init_conv = jax.random.normal(
        next(rngs), (num_seqs, kernel_size - 1, dim), dtype=jnp.float32
    )
    init_rec = jax.random.normal(
        next(rngs), (num_seqs, n_v, d_k, d_v), dtype=jnp.float32
    )

    q_loc = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
    distribution = jnp.array([num_decode, num_seqs, num_seqs], dtype=jnp.int32)
    seq_lens = jnp.full((num_seqs,), 11, dtype=jnp.int32)

    # OSS jax 0.11.x has no InterpretParams.on_race, so read the detector's
    # verdict before each pallas_call clears the shared state. Also resolve
    # granule_view's BitcastTransform and ReshapeTransform in CPU interpret
    # mode, which otherwise assumes NDIndexer-only DMA transforms. Delete the
    # shim once JAX interpret mode resolves bitcast and reshape refs.
    mosaic_interpret = sys.modules[
        pltpu.reset_tpu_interpret_mode_state.__module__
    ]
    interpret_utils = mosaic_interpret.interpret_utils
    indexing = interpret_utils.indexing
    state_types = interpret_utils.state_types
    shared_memory = sys.modules[f"{mosaic_interpret.__package__}.shared_memory"]
    races_per_call: list[bool] = []
    orig_clear = getattr(mosaic_interpret, "_clear_shared_memory")
    orig_store = shared_memory.SharedMemory.store_buffer_content

    def _record_and_clear():
      races = getattr(mosaic_interpret, "races", None)
      if races is not None:
        races_per_call.append(bool(races.races_found))
      orig_clear()

    # granule_view = ref.bitcast(uint32).reshape(*lead, -1, 1, 128) over the
    # 3 state dims. Resolve ref ranges on the lead axes only: drop the state
    # dims at the first view and the granule dims after it, requiring a
    # full-state block; Buffer._normalize_range re-expands the state dims.
    def _resolve_to_range(transforms):
      ret, viewed = (), False
      for t in transforms:
        if isinstance(t, indexing.NDIndexer):
          idx = tuple(
              interpret_utils._transform_slice_or_index(i) for i in t.indices
          )
          if viewed:
            full = tuple(slice(0, d, 1) for d in t.shape[-3:])
            assert idx[-3:] == full, f"partial-state ref: {idx}"
            idx = idx[:-3]
          ret = interpret_utils._compose_slice_or_index(ret, idx)
        elif isinstance(
            t, (state_types.BitcastTransform, state_types.ReshapeTransform)
        ):
          if not viewed:
            ret, viewed = ret[:-3], True
        else:
          raise NotImplementedError(f"unsupported ref transform: {t}")
      return ret

    def _store_buffer_content(buf_self, key, rnge, value, *args, **kwargs):
      # Bitcast DMAs move raw bits between equal-width dtypes only.
      buff_dtype = buf_self.mem[key].dtype
      assert value.itemsize == buff_dtype.itemsize, (value.dtype, buff_dtype)
      return orig_store(
          buf_self, key, rnge, value.view(buff_dtype), *args, **kwargs
      )

    pltpu.reset_tpu_interpret_mode_state()
    setattr(mosaic_interpret, "_clear_shared_memory", _record_and_clear)
    try:
      with (
          mock.patch.object(interpret_utils, "to_range", _resolve_to_range),
          mock.patch.object(
              shared_memory.SharedMemory,
              "store_buffer_content",
              _store_buffer_content,
          ),
          pltpu.force_tpu_interpret_mode(
              pltpu.InterpretParams(
                  detect_races=True,
                  dma_execution_mode="eager",
                  skip_floating_point_ops=False,
                  vector_clock_size=128,
              )
          ),
      ):
        (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
            qkv=qkv,
            b=b,
            a=a,
            states=(init_conv, init_rec),
            conv_weight=conv_w,
            conv_bias=conv_b,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            decode_tile_size=decode_tile,
            # CPU interpret mode does not support ref.reshape on the C > 1
            # chunked prefill path; each 1-token prefill is one tile at C = 1.
            mixed_tile_size=1,
        )
        jax.block_until_ready((conv_out, rec_out, out))
      self.assertTrue(races_per_call)
      self.assertFalse(
          any(races_per_call),
          f"DMA race detected across pallas_call phases: {races_per_call}",
      )
    finally:
      setattr(mosaic_interpret, "_clear_shared_memory", orig_clear)
      pltpu.reset_tpu_interpret_mode_state()

  def _run_batched_decode_initial_state_parity(self, state_dtype: jnp.dtype):
    """Multi-tile batched decode preserves per-sequence initial states."""
    n_kq, n_v, d_k, d_v, kernel_size = 16, 64, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    num_seqs = 256
    if _DECODE_TILE.value > 0:
      decode_tile = _DECODE_TILE.value
    else:
      decode_tile, _ = tiling.get_tile_sizes(
          batch_size=num_seqs,
          num_seqs=num_seqs,
          padded_batch_size=num_seqs,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          kernel_size=kernel_size,
          conv_state_dim_size=dim,
          act_out_dtype=jnp.dtype(jnp.bfloat16),
          recurrent_state_dtype=state_dtype,
          num_lanes=pltpu.get_tpu_info().num_lanes,
          decode_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(1),
          mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
      )
    steps = num_seqs // decode_tile
    self.assertGreaterEqual(steps, 4)

    block_ids = jnp.arange(1, num_seqs + 1, dtype=jnp.float32)
    init_conv = jnp.broadcast_to(
        block_ids[:, None, None] * jnp.float32(1e-2),
        (num_seqs, kernel_size - 1, dim),
    ).astype(jnp.float32)
    init_rec = jnp.broadcast_to(
        block_ids[:, None, None, None] * jnp.float32(1e-2),
        (num_seqs, n_v, d_k, d_v),
    ).astype(state_dtype)

    qkv = jnp.zeros((num_seqs, dim), dtype=jnp.bfloat16)
    b = jnp.full((num_seqs, n_v), -20.0, dtype=jnp.bfloat16)
    a = jnp.full((num_seqs, n_v), -20.0, dtype=jnp.bfloat16)
    conv_weight = jnp.zeros((dim, 1, kernel_size), dtype=jnp.bfloat16)
    conv_bias = jnp.zeros((dim,), dtype=jnp.bfloat16)
    a_log = jnp.full((n_v,), -20.0, dtype=jnp.float32)
    dt_bias = jnp.full((n_v,), -20.0, dtype=jnp.float32)

    q_loc = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    seq_lens = jnp.full((num_seqs,), 11, dtype=jnp.int32)

    (conv_ref, rec_ref), out_ref = gdn_attention_ref(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=init_conv,
        recurrent_state=init_rec,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    for rep in range(_RACE_REPEATS.value):
      (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
          qkv=qkv,
          b=b,
          a=a,
          states=(init_conv.copy(), init_rec.copy()),
          conv_weight=conv_weight,
          conv_bias=conv_bias,
          a_log=a_log,
          dt_bias=dt_bias,
          query_start_loc=q_loc,
          state_indices=state_indices,
          distribution=distribution,
          seq_lens=seq_lens,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          kernel_size=kernel_size,
          decode_tile_size=decode_tile,
      )
      block_err = jnp.max(
          jnp.abs(rec_out.astype(jnp.float32) - rec_ref.astype(jnp.float32)),
          axis=(1, 2, 3),
      )
      bad_blocks = np.where(np.asarray(block_err) > 1e-5)[0]
      if bad_blocks.size > 0:
        blk = int(bad_blocks[0])
        src_blk = blk + 2 * decode_tile
        rec_blk = np.asarray(rec_out[blk], dtype=np.float32)
        ref_blk = np.asarray(rec_ref[blk], dtype=np.float32)
        corrupted_mask = np.abs(rec_blk - ref_blk) > 1e-5
        if src_blk < num_seqs and np.any(corrupted_mask):
          hit = float(
              np.mean(
                  rec_blk[corrupted_mask]
                  == np.asarray(init_rec[src_blk], np.float32)[corrupted_mask]
              )
          )
        else:
          hit = 0.0
        self.fail(
            f"rep={rep} state_dtype={state_dtype} tile={decode_tile}: "
            f"{bad_blocks.size}/{num_seqs} blocks corrupted "
            f"(blocks={bad_blocks.tolist()}); block {blk} matches "
            f"init_rec[{src_blk}] on {hit:.1%} of corrupted elements "
            f"({int(np.sum(corrupted_mask))} elements)."
        )
      np.testing.assert_allclose(conv_out, conv_ref, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(out, out_ref, rtol=1e-5, atol=1e-5)

  def test_batched_decode_initial_state_parity_fp32(self):
    self._run_batched_decode_initial_state_parity(jnp.dtype(jnp.float32))

  def test_batched_decode_initial_state_parity_bf16(self):
    self._run_batched_decode_initial_state_parity(jnp.dtype(jnp.bfloat16))

  def test_spec_window_checkpoints_multi_tile(self):
    """Multi-tile spec verify (W=4) matches reference across repeats."""
    n_kq, n_v, d_k, d_v, kernel_size = 4, 8, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    window = 4
    num_seqs = 12
    batch_size = num_seqs * window
    num_blocks = num_seqs * window + 8

    decode_tile, _ = tiling.get_tile_sizes(
        batch_size=batch_size,
        num_seqs=num_seqs,
        padded_batch_size=batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        conv_state_dim_size=dim,
        act_out_dtype=jnp.dtype(jnp.bfloat16),
        recurrent_state_dtype=jnp.dtype(jnp.float32),
        num_lanes=pltpu.get_tpu_info().num_lanes,
        decode_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(window),
        mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
        window_size=window,
    )
    self.assertGreaterEqual(decode_tile, 2)
    self.assertGreaterEqual((num_seqs + decode_tile - 1) // decode_tile, 3)

    rngs = iter(jax.random.split(jax.random.key(314), 10))
    qkv = jax.random.normal(next(rngs), (batch_size, dim), dtype=jnp.bfloat16)
    b = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
    conv_w = jax.random.normal(
        next(rngs), (dim, 1, kernel_size), dtype=jnp.bfloat16
    )
    conv_b = jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16)
    a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
    dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
    init_conv = jax.random.normal(
        next(rngs), (num_blocks, kernel_size - 1, dim), dtype=jnp.float32
    )
    init_rec = jax.random.normal(
        next(rngs), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
    )

    q_loc = jnp.arange(0, batch_size + 1, window, dtype=jnp.int32)
    state_indices = jnp.arange(num_seqs, dtype=jnp.int32) * window
    read_offsets = jnp.array(
        [s % window for s in range(num_seqs)], dtype=jnp.int32
    )
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    seq_lens = jnp.full((num_seqs,), window + 10, dtype=jnp.int32)

    (conv_ref, rec_ref), out_ref = gdn_attention_spec_ref(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=init_conv,
        recurrent_state=init_rec,
        conv_weight=conv_w,
        conv_bias=conv_b,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        read_offsets=read_offsets,
        num_spec_seqs=num_seqs,
        seq_lens=seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    for _ in range(3):
      (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
          qkv=qkv,
          b=b,
          a=a,
          states=(init_conv.copy(), init_rec.copy()),
          conv_weight=conv_w,
          conv_bias=conv_b,
          a_log=a_log,
          dt_bias=dt_bias,
          query_start_loc=q_loc,
          state_indices=state_indices,
          distribution=distribution,
          seq_lens=seq_lens,
          read_offsets=read_offsets,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          kernel_size=kernel_size,
          num_spec_tokens=window - 1,
      )
      np.testing.assert_allclose(out, out_ref, rtol=1e-2, atol=1e-2)
      for s in range(num_seqs):
        base = int(state_indices[s])
        np.testing.assert_allclose(
            conv_out[base : base + window],
            conv_ref[base : base + window],
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            rec_out[base : base + window],
            rec_ref[base : base + window],
            rtol=1e-2,
            atol=1e-2,
        )

  def test_per_seq_boundary_read_after_write(self):
    """PER_SEQ multi-tile sequences preserve initial state and final writeback."""
    n_kq, n_v, d_k, d_v, kernel_size = 4, 8, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    # One-tile sequences between two-tile ones put a store at tile p and a
    # load at tile p + 2 on the same slot of the old shared ring.
    lengths = [64, 32, 64, 32, 64, 32]
    num_seqs = len(lengths)
    batch_size = sum(lengths)

    rngs = iter(jax.random.split(jax.random.key(2718), 10))
    qkv = jax.random.normal(next(rngs), (batch_size, dim), dtype=jnp.bfloat16)
    b = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
    conv_w = jax.random.normal(
        next(rngs), (dim, 1, kernel_size), dtype=jnp.bfloat16
    )
    conv_b = jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16)
    a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
    dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
    init_conv = jax.random.normal(
        next(rngs), (num_seqs, kernel_size - 1, dim), dtype=jnp.float32
    )
    init_rec = jax.random.normal(
        next(rngs), (num_seqs, n_v, d_k, d_v), dtype=jnp.float32
    )

    q_loc = jnp.array([0, *np.cumsum(lengths)], dtype=jnp.int32)
    state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
    distribution = jnp.array([0, num_seqs, num_seqs], dtype=jnp.int32)
    seq_lens = jnp.array(lengths, dtype=jnp.int32) + 10

    (conv_ref, rec_ref), out_ref = gdn_attention_ref(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=init_conv,
        recurrent_state=init_rec,
        conv_weight=conv_w,
        conv_bias=conv_b,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    for _ in range(3):
      (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
          qkv=qkv,
          b=b,
          a=a,
          states=(init_conv.copy(), init_rec.copy()),
          conv_weight=conv_w,
          conv_bias=conv_b,
          a_log=a_log,
          dt_bias=dt_bias,
          query_start_loc=q_loc,
          state_indices=state_indices,
          distribution=distribution,
          seq_lens=seq_lens,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          kernel_size=kernel_size,
          mixed_tile_size=32,
      )
      np.testing.assert_allclose(out, out_ref, rtol=1e-2, atol=1e-2)
      np.testing.assert_allclose(conv_out, conv_ref, rtol=1e-2, atol=1e-2)
      np.testing.assert_allclose(rec_out, rec_ref, rtol=1e-2, atol=1e-2)

  def test_qwen3_5_397b_parallelism_tile_selection(self):
    """Selects optimal (decode_tile, mixed_tile) across Qwen3.5-397B parallelisms."""
    d_k, d_v, kernel_size = 128, 128, 4
    f32 = jnp.dtype(jnp.float32)
    bf16 = jnp.dtype(jnp.bfloat16)
    # Use the 64 MiB TPU v7x VMEM budget (0.8 default, 0.9 windowed)
    # so the expected tile table is identical across TPU test targets.
    vmem_64m = 64 * 1024 * 1024
    limit_w1 = int(config.GDNConfig.DEFAULT_VMEM_FRACTION * vmem_64m)
    limit_w4 = int(config.GDNConfig.WINDOWED_VMEM_FRACTION * vmem_64m)

    # (label, n_kq, n_v, batch_size, num_seqs, window_size, state_dtype, expected)
    cases = [
        # TP8 + DCP=4 (n_kq=2, n_v=8, W=4 for MTP=3): warmup & verify buckets
        ("tp8_dcp4_c16_b4", 2, 8, 4, 16, 4, f32, (2, 4)),
        ("tp8_dcp4_c16_b8", 2, 8, 8, 16, 4, f32, (4, 8)),
        ("tp8_dcp4_c16_b16", 2, 8, 16, 16, 4, f32, (4, 16)),
        ("tp8_dcp4_c16_b32", 2, 8, 32, 16, 4, f32, (4, 32)),
        ("tp8_dcp4_c16_b64", 2, 8, 64, 16, 4, f32, (4, 64)),
        ("tp8_dcp4_c32_b128", 2, 8, 128, 32, 4, f32, (8, 128)),
        ("tp8_dcp4_c64_b256", 2, 8, 256, 64, 4, f32, (8, 128)),
        # TP8 + DCP=1 (n_kq=2, n_v=8, W=1 and W=4) and padded slot table
        ("tp8_dcp1_c8_mtp3_b32", 2, 8, 32, 8, 4, f32, (4, 32)),
        ("tp8_dcp1_c32_b16", 2, 8, 16, 32, 1, f32, (4, 16)),
        ("tp8_dcp1_c64_b64", 2, 8, 64, 64, 1, f32, (8, 64)),
        ("tp8_padded_n512_b64", 2, 8, 64, 512, 1, bf16, (8, 64)),
        # PCP8 + TP1 disagg prefill (n_kq=2, n_v=8, W=1)
        ("pcp8_prefill_b4096", 2, 8, 4096, 16, 1, f32, (4, 128)),
        # DP4 + TP2 + EP8 (n_kq=8, n_v=32, W=1 and W=4)
        ("dp4tp2_c64_b8", 8, 32, 8, 32, 1, f32, (2, 8)),
        ("dp4tp2_c64_b32", 8, 32, 32, 32, 1, f32, (4, 32)),
        ("dp4tp2_c64_b2048", 8, 32, 2048, 32, 1, f32, (4, 128)),
        ("dp4tp2_mtp3_c16_b64_f32", 8, 32, 64, 16, 4, f32, (2, 64)),
        ("dp4tp2_mtp3_c32_b128_f32", 8, 32, 128, 32, 4, f32, (2, 128)),
        ("dp4tp2_mtp3_c32_b128_bf16", 8, 32, 128, 32, 4, bf16, (4, 128)),
        # DP8 + TP1 + EP8 (n_kq=16, n_v=64, W=1 and W=4)
        ("dp8_c128_b8", 16, 64, 8, 16, 1, f32, (1, 8)),
        ("dp8_c128_b16", 16, 64, 16, 16, 1, f32, (2, 16)),
        ("dp8_c256_b32_f32", 16, 64, 32, 32, 1, f32, (2, 32)),
        ("dp8_c256_b32_bf16", 16, 64, 32, 32, 1, bf16, (2, 32)),
        ("dp8_c256_b320_f32", 16, 64, 320, 32, 1, f32, (2, 64)),
        ("dp8_mtp3_c32_b128_f32", 16, 64, 128, 32, 4, f32, (1, 64)),
        ("dp8_mtp3_c32_b128_bf16", 16, 64, 128, 32, 4, bf16, (2, 64)),
    ]
    for label, n_kq, n_v, b, n, w, state_dtype, expected in cases:
      with self.subTest(label=label):
        dim = 2 * n_kq * d_k + n_v * d_v
        got = tiling.get_tile_sizes(
            batch_size=b,
            num_seqs=n,
            padded_batch_size=b,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim,
            act_out_dtype=bf16,
            recurrent_state_dtype=state_dtype,
            num_lanes=128,
            decode_vmem_limit_bytes=limit_w4 if w > 1 else limit_w1,
            mixed_vmem_limit_bytes=limit_w1,
            window_size=w,
        )
        self.assertEqual(got, expected, msg=label)

  def test_qwen3_5_397b_parallelism_execution(self):
    """Executes default-tiled GDN across Qwen3.5-397B head splits and DCP=4."""
    d_k, d_v, kernel_size = 128, 128, 4
    rng = jax.random.key(42)

    for n_kq, n_v, num_seqs, num_spec_tokens, prefill_len, num_decodes in [
        (2, 8, 16, 3, 16, 0),  # TP8 DCP=4 c=16 MTP=3: 16-tok prefill in B=16
        (2, 8, 16, 3, 0, 16),  # TP8 DCP=4 c=16 MTP=3 warmup: 16 decodes, B=16
        (8, 32, 16, 0, 12, 4),  # DP4TP2: 4 decodes + 12-tok prefill in B=16
        (16, 64, 16, 0, 12, 4),  # DP8: 4 decodes + 12-tok prefill in B=16
    ]:
      with self.subTest(
          n_kq=n_kq,
          n_v=n_v,
          num_spec_tokens=num_spec_tokens,
          prefill_len=prefill_len,
          num_decodes=num_decodes,
      ):
        dim = 2 * n_kq * d_k + n_v * d_v
        window = num_spec_tokens + 1
        batch_size = num_decodes + prefill_len
        num_blocks = num_seqs * window + 4
        k1, k2, k3, k4, k5, k6, k7, k8, k9, rng = jax.random.split(rng, 10)
        init_conv = jax.random.normal(
            k8, (num_blocks, kernel_size - 1, dim), dtype=jnp.float32
        )
        init_rec = jax.random.normal(
            k9, (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
        )
        active_seqs = num_decodes + (1 if prefill_len > 0 else 0)
        state_indices = jnp.arange(num_seqs, dtype=jnp.int32) * window
        common_kwargs = dict(
            qkv=jax.random.normal(k1, (batch_size, dim), dtype=jnp.bfloat16),
            b=jax.random.normal(k2, (batch_size, n_v), dtype=jnp.bfloat16),
            a=jax.random.normal(k3, (batch_size, n_v), dtype=jnp.bfloat16),
            conv_weight=jax.random.normal(
                k4, (dim, 1, kernel_size), dtype=jnp.bfloat16
            ),
            conv_bias=jax.random.normal(k5, (dim,), dtype=jnp.bfloat16),
            a_log=jax.random.normal(k6, (n_v,), dtype=jnp.float32),
            dt_bias=jax.random.normal(k7, (n_v,), dtype=jnp.float32),
            query_start_loc=jnp.array(
                list(range(num_decodes + 1))
                + [batch_size] * (num_seqs - num_decodes),
                dtype=jnp.int32,
            ),
            state_indices=state_indices,
            distribution=jnp.array(
                [num_decodes, num_decodes, active_seqs], dtype=jnp.int32
            ),
            seq_lens=jnp.array(
                [10] * num_decodes
                + ([prefill_len + 10] if prefill_len > 0 else [])
                + [0] * (num_seqs - active_seqs),
                dtype=jnp.int32,
            ),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )
        (conv_ref, rec_ref), out_ref = gdn_attention_ref(
            conv_state=init_conv,
            recurrent_state=init_rec,
            **common_kwargs,
        )
        (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
            states=(init_conv.copy(), init_rec.copy()),
            read_offsets=(
                jnp.zeros((num_seqs,), dtype=jnp.int32)
                if num_spec_tokens > 0
                else None
            ),
            num_spec_tokens=num_spec_tokens,
            **common_kwargs,
        )
        np.testing.assert_allclose(out, out_ref, rtol=2e-2, atol=2e-2)
        for s in range(active_seqs):
          slot = int(state_indices[s])
          np.testing.assert_allclose(
              conv_out[slot], conv_ref[slot], rtol=2e-2, atol=2e-2
          )
          np.testing.assert_allclose(
              rec_out[slot], rec_ref[slot], rtol=3e-2, atol=3e-2
          )


# ----------------------------------------------------------------------------
# Host-side unified-pool state layout helpers and seam tests (ported from
# vllm-torchtpu PR #1620: gdn_pool_state_utils.py, test_gdn_pool_state_utils.py,
# and test_v3_pool_state_seam.py).
# ----------------------------------------------------------------------------

_GRANULE = 512
_WORDS_PER_GRANULE = _GRANULE // 4

_SEAM_N_KQ, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V = 2, 8, 128, 128
_SEAM_KERNEL_SIZE = 4
_SEAM_DIM = _SEAM_N_KQ * _SEAM_D_K * 2 + _SEAM_N_V * _SEAM_D_V  # 1536
_SEAM_SPLIT, _SEAM_KBS, _SEAM_LANES = 3, 256, 256
_SEAM_PAYLOAD = (1, 2)
_SEAM_TOK_BYTES = 2 * _SEAM_LANES * 2  # bf16 pool
_SEAM_CONV_GRANULES = (
    (_SEAM_KERNEL_SIZE - 1) * _SEAM_DIM * 4 // _GRANULE
)  # 36
_SEAM_CONV_END_TOK = (
    _SEAM_CONV_GRANULES * _GRANULE // _SEAM_TOK_BYTES
)  # 18
_SEAM_NUM_MGR = 5  # null block + 4 slots
_SEAM_STATE_SHAPES = dict(
    recurrent_shape=(_SEAM_N_V, _SEAM_D_K, _SEAM_D_V),
    conv_shape=(_SEAM_KERNEL_SIZE - 1, _SEAM_DIM),
)


def _on_cpu(fn, *args):
  """Run a pure bit-shuffle on CPU, then hand the result back to the TPU.

  TPU canonicalises NaN payloads whenever float bytes pass through a vector
  register, so bitcasting or transposing a bf16/fp8 pool on device silently
  rewrites about a percent of its words -- measured on v7/jax 0.10.2: 71 of
  4096 uint32 words through bfloat16, 77 through float8_e4m3fn, 0 through
  int8. The kernel never meets this, because its granule DMA moves the pool
  as uint32 and no float register ever sees it; a host-side harness that
  builds or reads the pool *through the pool's own dtype* does. CPU bitcasts
  are exact, so the byte layout these tests assert on is the real one.

  Args:
    fn: Pure bit-shuffle function to execute on CPU.
    *args: Arguments to pass to `fn`.

  Returns:
    Result of `fn(*args)` placed back on the default TPU device.
  """
  cpu = jax.devices("cpu")[0]
  with jax.default_device(cpu):
    out = jax.jit(fn)(*jax.device_put(args, cpu))
  return jax.device_put(out, jax.devices()[0])


def _state_words(pool, split):
  """The pool the way the kernel addresses it: one uint32 row per state.

  A ref's `bitcast(uint32)` packs along the *second-minor* (sublane) axis,
  which for a pool row `(..., packing, lanes)` means one word per lane, not
  one word per adjacent pair. Moving that axis minormost first is what makes
  the host view agree with the kernel's granule stream.

  Args:
    pool: Unified state pool tensor.
    split: Number of kernel blocks per manager block.

  Returns:
    A 2D uint32 view with one row per manager state block.
  """
  words = jnp.moveaxis(pool, -2, -1).view(jnp.uint32)
  return words.reshape(pool.shape[0] // split, -1)


def _words_to_pool(words, pool):
  """Inverse of `_state_words`, back to `pool`'s shape and dtype."""
  moved = words.reshape(*pool.shape[:-2], pool.shape[-1], 1).view(pool.dtype)
  return jnp.moveaxis(moved, -2, -1).reshape(pool.shape)


def _put_words(words, idx, granule0, payload):
  """Drop `payload`'s raw bytes at a granule offset of each named state."""
  body = payload.reshape(idx.shape[0], -1).view(jnp.uint32)
  start = granule0 * _WORDS_PER_GRANULE
  return words.at[idx, start : start + body.shape[1]].set(body)


def _take_words(words, idx, granule0, shape, dtype):
  """Read `shape`-shaped `dtype` values at a granule offset of each state."""
  nwords = math.prod(shape) * jnp.dtype(dtype).itemsize // 4
  start = granule0 * _WORDS_PER_GRANULE
  return (
      words[idx, start : start + nwords]
      .view(dtype)
      .reshape(idx.shape[0], *shape)
  )


def _pack_bf16_rows(values):
  """bf16 [..., D_K, D_V] -> uint32 [..., D_K // 2, D_V] matching ref bitcast."""
  pairs = values.astype(jnp.bfloat16).reshape(
      *values.shape[:-2], values.shape[-2] // 2, 2, values.shape[-1]
  )
  return jax.lax.bitcast_convert_type(jnp.moveaxis(pairs, -2, -1), jnp.uint32)


def _unpack_bf16_rows(words):
  """Inverse of `_pack_bf16_rows`."""
  pairs = jax.lax.bitcast_convert_type(words, jnp.bfloat16)
  rows = jnp.moveaxis(pairs, -1, -2)
  return rows.reshape(*words.shape[:-2], words.shape[-2] * 2, words.shape[-1])


def _recurrent_granules(recurrent_shape, recurrent_dtype=jnp.float32) -> int:
  """Granules taken by one recurrent state, i.e. where the conv state starts."""
  nbytes = math.prod(recurrent_shape) * jnp.dtype(recurrent_dtype).itemsize
  assert nbytes % _GRANULE == 0, (recurrent_shape, recurrent_dtype)
  return nbytes // _GRANULE


def _recurrent_words(recurrent, recurrent_dtype):
  if jnp.dtype(recurrent_dtype) == jnp.bfloat16:
    return _pack_bf16_rows(recurrent)
  assert jnp.dtype(recurrent_dtype) == jnp.float32, recurrent_dtype
  return jnp.asarray(recurrent, jnp.float32).view(jnp.uint32)


def _write_pool_states(
    pool, ssm, conv, idx, *, split, recurrent_dtype=jnp.float32
):
  """Place recurrent and conv states of the states named by `idx`."""

  def build(pool, ssm, conv, idx):
    words = _state_words(pool, split)
    words = _put_words(words, idx, 0, _recurrent_words(ssm, recurrent_dtype))
    words = _put_words(
        words,
        idx,
        _recurrent_granules(ssm.shape[1:], recurrent_dtype),
        jnp.asarray(conv, jnp.float32),
    )
    return _words_to_pool(words, pool)

  return _on_cpu(build, pool, ssm, conv, idx)


def _read_pool_states(
    pool,
    idx,
    *,
    split,
    recurrent_shape,
    conv_shape,
    recurrent_dtype=jnp.float32,
):
  """Read back `(ssm, conv)` of the states named by `idx`."""

  def take(pool, idx):
    words = _state_words(pool, split)
    if jnp.dtype(recurrent_dtype) == jnp.bfloat16:
      num_v_heads, d_k, d_v = recurrent_shape
      packed = _take_words(
          words, idx, 0, (num_v_heads, d_k // 2, d_v), jnp.uint32
      )
      ssm = _unpack_bf16_rows(packed)
    else:
      ssm = _take_words(words, idx, 0, recurrent_shape, recurrent_dtype)
    conv = _take_words(
        words,
        idx,
        _recurrent_granules(recurrent_shape, recurrent_dtype),
        conv_shape,
        jnp.float32,
    )
    return ssm, conv

  return _on_cpu(take, pool, idx)


def _garbage_pool(seed, num_mgr: int = _SEAM_NUM_MGR):
  rng = np.random.default_rng(seed)
  shape = (num_mgr * _SEAM_SPLIT, _SEAM_KBS) + _SEAM_PAYLOAD + (_SEAM_LANES,)
  return jnp.asarray(rng.standard_normal(shape), dtype=jnp.bfloat16)


def _write_states(
    pool, ssm, conv, idx, split=_SEAM_SPLIT, ssm_dtype=jnp.float32
):
  return _write_pool_states(
      pool, ssm, conv, idx, split=split, recurrent_dtype=ssm_dtype
  )


def _read_states(pool, idx, split=_SEAM_SPLIT, ssm_dtype=jnp.float32):
  return _read_pool_states(
      pool, idx, split=split, recurrent_dtype=ssm_dtype, **_SEAM_STATE_SHAPES
  )


_write_states_bf16 = functools.partial(_write_states, ssm_dtype=jnp.bfloat16)
_read_states_bf16 = functools.partial(_read_states, ssm_dtype=jnp.bfloat16)


def _rand_inputs(seed, num_tokens):
  rngs = iter(jax.random.split(jax.random.key(seed), 8))
  return dict(
      qkv=jax.random.normal(next(rngs), (num_tokens, _SEAM_DIM)),
      b=jax.random.normal(next(rngs), (num_tokens, _SEAM_N_V)),
      a=jax.random.normal(next(rngs), (num_tokens, _SEAM_N_V)),
      conv_weight=jax.random.normal(
          next(rngs), (_SEAM_DIM, 1, _SEAM_KERNEL_SIZE)
      ),
      conv_bias=jax.random.normal(next(rngs), (_SEAM_DIM,)),
      a_log=jax.random.normal(next(rngs), (_SEAM_N_V,)),
      dt_bias=jax.random.normal(next(rngs), (_SEAM_N_V,)),
      n_kq=_SEAM_N_KQ,
      n_v=_SEAM_N_V,
      d_k=_SEAM_D_K,
      d_v=_SEAM_D_V,
      kernel_size=_SEAM_KERNEL_SIZE,
  )


def _run_roundtrip(pool, idx, kwargs):
  """The pre-seam pooled path (gather -> tuple kernel -> scatter)."""
  ssm, conv = _read_states(pool, idx)
  identity = jnp.arange(idx.shape[0], dtype=jnp.int32)
  (new_conv, new_ssm), out = wrapper.fused_conv1d_gdn(
      states=(conv, ssm), state_indices=identity, **kwargs
  )
  return _write_states(pool, new_ssm, new_conv, idx), out


def _merge(pool, split=_SEAM_SPLIT):
  """Kernel blocks -> manager blocks, as the pooled caller does."""
  return pool.reshape(
      pool.shape[0] // split, split * pool.shape[1], *pool.shape[2:]
  )


def _split(pool, shape):
  return pool.reshape(shape)


def _run_seam(pool, idx, kwargs, split=_SEAM_SPLIT):
  out_pool, out = wrapper.fused_conv1d_gdn(
      states=_merge(pool, split),
      state_indices=idx,
      conv_dtype=jnp.float32,
      recurrent_dtype=jnp.float32,
      **kwargs,
  )
  return _split(out_pool, pool.shape), out


def _bits(x):
  return np.asarray(jax.lax.bitcast_convert_type(x, jnp.int16))


class GdnPoolStateUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 6:
        self.skipTest("Pallas TPU kernel requires TPU v6 or newer.")
    except Exception:  # pylint: disable=broad-except
      self.skipTest("Failed to get TPU info.")

  @parameterized.product(
      num_v_heads=[1, 4, 8],
      recurrent_dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_conv_offset_matches_kernel_config(
      self, num_v_heads, recurrent_dtype
  ):
    """The helper must place conv where the kernel reads it."""
    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.bfloat16.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.dtype(recurrent_dtype),
            conv_state=jnp.float32.dtype,
        ),
        batch_size=1,
        dim_size=(2 + num_v_heads) * 128,
        kernel_size=4,
        tile_size=64,
        num_kq_heads=1,
        num_v_heads=num_v_heads,
        kq_head_dim=128,
        v_head_dim=128,
        is_shared=True,
    )

    self.assertEqual(cfg.recurrent_hbm_offset, 0)
    self.assertEqual(
        cfg.conv_hbm_offset,
        _recurrent_granules((num_v_heads, 128, 128), recurrent_dtype),
    )

  @parameterized.product(
      pool_dtype=[jnp.float8_e4m3fn, jnp.bfloat16],
      recurrent_dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_write_read_roundtrip_leaves_other_states(
      self, pool_dtype, recurrent_dtype
  ):
    split, block_tokens, lanes = 2, 256, 128
    packing = 4 // jnp.dtype(pool_dtype).itemsize
    num_states, num_v_heads, conv_shape = 4, 2, (3, 512)
    pool_shape = (num_states * split, block_tokens, 1, packing, lanes)
    rng = np.random.default_rng(0)
    pool_bytes = rng.integers(
        0,
        256,
        size=(*pool_shape[:-1], lanes * jnp.dtype(pool_dtype).itemsize),
        dtype=np.uint8,
    )
    pool = (
        np.asarray(pool_bytes).view(jnp.dtype(pool_dtype)).reshape(pool_shape)
    )
    idx = jnp.asarray([2, 1], dtype=jnp.int32)
    keys = jax.random.split(jax.random.key(0), 2)
    ssm = jax.random.normal(keys[0], (2, num_v_heads, 128, 128)).astype(
        recurrent_dtype
    )
    conv = jax.random.normal(keys[1], (2, *conv_shape), dtype=jnp.float32)

    new_pool = _write_pool_states(
        pool, ssm, conv, idx, split=split, recurrent_dtype=recurrent_dtype
    )
    got_ssm, got_conv = _read_pool_states(
        new_pool,
        idx,
        split=split,
        recurrent_shape=(num_v_heads, 128, 128),
        conv_shape=conv_shape,
        recurrent_dtype=recurrent_dtype,
    )

    np.testing.assert_array_equal(np.asarray(got_ssm), np.asarray(ssm))
    np.testing.assert_array_equal(np.asarray(got_conv), np.asarray(conv))
    new_bytes = np.asarray(jax.device_get(new_pool)).view(np.uint8)
    old_bytes = pool.view(np.uint8)
    for state in (0, 3):
      rows = slice(state * split, (state + 1) * split)
      np.testing.assert_array_equal(new_bytes[rows], old_bytes[rows])


class GdnSeamMatchesRoundTripTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 6:
        self.skipTest("Pallas TPU kernel requires TPU v6 or newer.")
    except Exception:  # pylint: disable=broad-except
      self.skipTest("Failed to get TPU info.")

  def _compare(self, pool, idx, kwargs):
    pool_rt, out_rt = _run_roundtrip(jnp.copy(pool), idx, kwargs)
    pool_seam, out_seam = _run_seam(jnp.copy(pool), idx, kwargs)

    tol = dict(rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(np.asarray(out_seam), np.asarray(out_rt), **tol)
    ssm_rt, conv_rt = _read_states(pool_rt, idx)
    ssm_seam, conv_seam = _read_states(pool_seam, idx)
    np.testing.assert_allclose(np.asarray(ssm_seam), np.asarray(ssm_rt), **tol)
    np.testing.assert_allclose(
        np.asarray(conv_seam, dtype=np.float32),
        np.asarray(conv_rt, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    touched = set()
    for mgr in np.asarray(idx):
      touched.update(
          range(int(mgr) * _SEAM_SPLIT, int(mgr) * _SEAM_SPLIT + _SEAM_SPLIT)
      )
    for kb in range(pool_seam.shape[0]):
      if kb not in touched:
        self.assertTrue(
            bool(
                jnp.array_equal(
                    pool_seam[kb].view(jnp.int16), pool_rt[kb].view(jnp.int16)
                )
            ),
            msg=f"kb={kb}",
        )

  def test_decode_continuation(self):
    n = 4
    idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    keys = iter(jax.random.split(jax.random.key(10), 2))
    pool = _write_states(
        _garbage_pool(0),
        jax.random.normal(next(keys), (n, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V)),
        jax.random.normal(
            next(keys), (n, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM)
        ).astype(jnp.bfloat16),
        idx,
    )
    kwargs = _rand_inputs(1, n) | dict(
        query_start_loc=jnp.arange(n + 1),
        distribution=jnp.array([n, n, n], dtype=jnp.int32),
        seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
    )
    self._compare(pool, idx, kwargs)

  def test_per_ckpt_read_selects_the_named_block(self):
    """Rollback must resume from `ckpt_indices[s, read_offsets[s]]`."""
    n, num_spec = 2, 3
    window = num_spec + 1
    ckpt_blocks = jnp.asarray(
        np.array([[1, 3, 5, 7], [8, 6, 4, 2]], dtype=np.int32)
    )
    num_mgr = 9
    keys = iter(jax.random.split(jax.random.key(77), 2 * window))
    states = []
    pool = _garbage_pool(11, num_mgr)
    for t in range(window):
      ssm_t = jax.random.normal(
          next(keys), (n, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V)
      )
      conv_t = jax.random.normal(
          next(keys), (n, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM)
      ).astype(jnp.bfloat16)
      states.append((ssm_t, conv_t))
      pool = _write_states(pool, ssm_t, conv_t, ckpt_blocks[:, t])

    kwargs = _rand_inputs(12, n) | dict(
        query_start_loc=jnp.arange(n + 1),
        distribution=jnp.array([n, n, n], dtype=jnp.int32),
        seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
    )

    def _run(src, read_ckpt):
      return wrapper.fused_conv1d_gdn(
          states=_merge(jnp.copy(src)),
          state_indices=jnp.zeros((n,), dtype=jnp.int32),
          conv_dtype=jnp.float32,
          recurrent_dtype=jnp.float32,
          read_offsets=jnp.full((n,), read_ckpt, dtype=jnp.int32),
          ckpt_indices=ckpt_blocks,
          num_spec_tokens=num_spec,
          **kwargs,
      )[1]

    for r in range(window):
      ssm_r, conv_r = states[r]
      at_col0 = _write_states(
          _garbage_pool(11, num_mgr), ssm_r, conv_r, ckpt_blocks[:, 0]
      )
      np.testing.assert_allclose(
          np.asarray(_run(pool, r)),
          np.asarray(_run(at_col0, 0)),
          rtol=1e-6,
          atol=1e-6,
          err_msg=f"checkpoint {r}",
      )

  def test_per_ckpt_writes_only_touch_named_blocks(self):
    """Checkpoint `t` is written to `ckpt_indices[:, t]`; nothing else changes."""
    n, num_spec = 2, 3
    window = num_spec + 1
    ckpt_blocks = np.array([[1, 3, 5, 7], [8, 6, 4, 2]], dtype=np.int32)
    num_mgr = 9
    pool = _garbage_pool(13, num_mgr)
    kwargs = _rand_inputs(14, n * window) | dict(
        query_start_loc=jnp.arange(0, n * window + 1, window),
        distribution=jnp.array([n, n, n], dtype=jnp.int32),
        seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
    )
    before = np.asarray(pool.view(jnp.int16))
    out_pool, _ = wrapper.fused_conv1d_gdn(
        states=_merge(jnp.copy(pool)),
        state_indices=jnp.zeros((n,), dtype=jnp.int32),
        conv_dtype=jnp.float32,
        recurrent_dtype=jnp.float32,
        read_offsets=jnp.zeros((n,), dtype=jnp.int32),
        ckpt_indices=jnp.asarray(ckpt_blocks),
        num_spec_tokens=num_spec,
        **kwargs,
    )
    out_pool = _split(out_pool, pool.shape)

    named = set(int(m) for m in ckpt_blocks.reshape(-1))
    touched_kb = set()
    for m in named:
      touched_kb.update(range(m * _SEAM_SPLIT, m * _SEAM_SPLIT + _SEAM_SPLIT))
    after = np.asarray(out_pool.view(jnp.int16))
    changed = {
        kb
        for kb in range(after.shape[0])
        if not np.array_equal(after[kb], before[kb])
    }
    self.assertTrue(changed, "the run wrote no state at all")
    self.assertLessEqual(changed, touched_kb, msg=sorted(changed - touched_kb))

  def test_fresh_prefill_on_garbage_pool(self):
    idx = jnp.array([2, 4], dtype=jnp.int32)
    kwargs = _rand_inputs(2, 128) | dict(
        query_start_loc=jnp.array([0, 64, 128]),
        distribution=jnp.array([0, 2, 2], dtype=jnp.int32),
        seq_lens=jnp.array([64, 64], dtype=jnp.int32),
    )
    self._compare(_garbage_pool(3), idx, kwargs)

  def test_chunked_prefill_continuation(self):
    idx = jnp.array([1, 3], dtype=jnp.int32)
    keys = iter(jax.random.split(jax.random.key(11), 2))
    pool = _write_states(
        _garbage_pool(4),
        jax.random.normal(next(keys), (2, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V)),
        jax.random.normal(
            next(keys), (2, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM)
        ).astype(jnp.bfloat16),
        idx,
    )
    kwargs = _rand_inputs(5, 112) | dict(
        query_start_loc=jnp.array([0, 64, 112]),
        distribution=jnp.array([0, 2, 2], dtype=jnp.int32),
        seq_lens=jnp.array([96, 80], dtype=jnp.int32),
    )
    self._compare(pool, idx, kwargs)

  def test_mixed_decode_and_prefill(self):
    idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    keys = iter(jax.random.split(jax.random.key(12), 2))
    decode_idx = idx[:2]
    pool = _write_states(
        _garbage_pool(6),
        jax.random.normal(next(keys), (2, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V)),
        jax.random.normal(
            next(keys), (2, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM)
        ).astype(jnp.bfloat16),
        decode_idx,
    )
    kwargs = _rand_inputs(7, 130) | dict(
        query_start_loc=jnp.array([0, 1, 2, 66, 130]),
        distribution=jnp.array([2, 4, 4], dtype=jnp.int32),
        seq_lens=jnp.array([7, 9, 64, 64], dtype=jnp.int32),
    )
    self._compare(pool, idx, kwargs)


class GdnSeamSkipsInvalidSlotsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 6:
        self.skipTest("Pallas TPU kernel requires TPU v6 or newer.")
    except Exception:  # pylint: disable=broad-except
      self.skipTest("Failed to get TPU info.")

  def test_padded_slots_and_null_block_untouched(self):
    idx = jnp.array([2, 4, 0, 0], dtype=jnp.int32)
    keys = iter(jax.random.split(jax.random.key(13), 2))
    pool = _write_states(
        _garbage_pool(8),
        jax.random.normal(next(keys), (2, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V)),
        jax.random.normal(
            next(keys), (2, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM)
        ).astype(jnp.bfloat16),
        idx[:2],
    )
    before = jnp.copy(pool)
    kwargs = _rand_inputs(9, 2) | dict(
        query_start_loc=jnp.array([0, 1, 2, 2, 2]),
        distribution=jnp.array([2, 2, 2], dtype=jnp.int32),
        seq_lens=jnp.array([5, 5, 0, 0], dtype=jnp.int32),
    )
    after, _ = _run_seam(pool, idx, kwargs)

    def b16(x):
      return x.view(jnp.int16)

    for kb in (0, 1, 2, 3, 4, 5, 9, 10, 11):
      self.assertTrue(
          bool(jnp.array_equal(b16(after[kb]), b16(before[kb]))), msg=f"kb={kb}"
      )
    for mgr in (2, 4):
      self.assertFalse(
          bool(
              jnp.array_equal(
                  b16(after[mgr * _SEAM_SPLIT]), b16(before[mgr * _SEAM_SPLIT])
              )
          ),
          msg=f"mgr={mgr}",
      )
      conv_kb = mgr * _SEAM_SPLIT + 2
      self.assertTrue(
          bool(
              jnp.array_equal(
                  b16(after[conv_kb, _SEAM_CONV_END_TOK:]),
                  b16(before[conv_kb, _SEAM_CONV_END_TOK:]),
              )
          ),
          msg=f"mgr={mgr}",
      )


class GdnPooledSpecWindowsTest(parameterized.TestCase):
  """Speculative verify windows on the pooled path."""

  NUM_SPEC = 1
  WINDOW = NUM_SPEC + 1
  N = 3

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 6:
        self.skipTest("Pallas TPU kernel requires TPU v6 or newer.")
    except Exception:  # pylint: disable=broad-except
      self.skipTest("Failed to get TPU info.")

  def _pool(self, split):
    if split > 1:
      prefix = (_SEAM_NUM_MGR * self.WINDOW * split, _SEAM_KBS)
    else:
      prefix = (_SEAM_NUM_MGR * self.WINDOW, 1280)
    shape = prefix + _SEAM_PAYLOAD + (_SEAM_LANES,)
    rng = np.random.default_rng(20)
    return jnp.asarray(rng.standard_normal(shape), dtype=jnp.bfloat16)

  def _steps(self):
    n = self.N
    return [
        dict(
            num_tokens=5,
            query_start_loc=jnp.array([0, 2, 3, 5]),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.array([10, 7, 9], dtype=jnp.int32),
            read_offsets=jnp.zeros((n,), dtype=jnp.int32),
        ),
        dict(
            num_tokens=5,
            query_start_loc=jnp.array([0, 2, 4, 5]),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.array([12, 9, 10], dtype=jnp.int32),
            read_offsets=jnp.array([1, 0, 1], dtype=jnp.int32),
        ),
        dict(
            num_tokens=6,
            query_start_loc=jnp.array([0, 1, 6, 6]),
            distribution=jnp.array([1, 1, n], dtype=jnp.int32),
            seq_lens=jnp.array([14, 14, 0], dtype=jnp.int32),
            read_offsets=jnp.array([0, 1, 0], dtype=jnp.int32),
        ),
    ]

  @parameterized.named_parameters(
      dict(testcase_name="kernel_grained", split=6),
      dict(testcase_name="manager", split=1),
  )
  def test_pooled_spec_matches_dense(self, split):
    n, window = self.N, self.WINDOW
    pool = self._pool(split)
    pool_idx = jnp.array([4, 1, 3], dtype=jnp.int32)
    ckpt_indices = jnp.array([[4, 7], [1, 9], [3, 5]], dtype=jnp.int32)

    keys = iter(jax.random.split(jax.random.key(21), 2))
    ssm0 = jax.random.normal(next(keys), (n, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V))
    conv0 = jax.random.normal(
        next(keys), (n, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM)
    )
    pool = _write_states(pool, ssm0, conv0, pool_idx, split=split)

    rng = np.random.default_rng(22)
    num_slots = 1 + n * window
    dense_base = jnp.array(
        [1 + i * window for i in range(n)], dtype=jnp.int32
    )
    dense_conv = jnp.asarray(
        rng.standard_normal((num_slots, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM)),
        dtype=jnp.float32,
    )
    dense_rec = jnp.asarray(
        rng.standard_normal((num_slots, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V)),
        dtype=jnp.float32,
    )
    dense_conv = dense_conv.at[dense_base].set(conv0)
    dense_rec = dense_rec.at[dense_base].set(ssm0.astype(jnp.float32))

    for step_num, step in enumerate(self._steps()):
      kwargs = _rand_inputs(30 + step_num, step["num_tokens"]) | dict(
          query_start_loc=step["query_start_loc"],
          distribution=step["distribution"],
          seq_lens=step["seq_lens"],
          num_spec_tokens=self.NUM_SPEC,
          read_offsets=step["read_offsets"],
      )
      (dense_conv, dense_rec), out_dense = wrapper.fused_conv1d_gdn(
          states=(dense_conv, dense_rec), state_indices=dense_base, **kwargs
      )
      pool_shape = pool.shape
      pool, out_pooled = wrapper.fused_conv1d_gdn(
          states=_merge(pool, split),
          state_indices=pool_idx,
          ckpt_indices=ckpt_indices,
          conv_dtype=jnp.float32,
          recurrent_dtype=jnp.float32,
          **kwargs,
      )
      pool = _split(pool, pool_shape)
      np.testing.assert_allclose(
          np.asarray(out_pooled),
          np.asarray(out_dense),
          rtol=2e-2,
          atol=2e-2,
          err_msg=f"step {step_num}",
      )


class GdnPooledBf16RecurrentTest(parameterized.TestCase):
  """bf16 recurrent state with an fp32 conv state."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 6:
        self.skipTest("Pallas TPU kernel requires TPU v6 or newer.")
    except Exception:  # pylint: disable=broad-except
      self.skipTest("Failed to get TPU info.")

  def _states(self, seed, n):
    keys = iter(jax.random.split(jax.random.key(seed), 2))
    ssm = jax.random.normal(
        next(keys), (n, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V)
    ).astype(jnp.bfloat16)
    conv = jax.random.normal(next(keys), (n, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM))
    return ssm, conv

  def _tuple(self, ssm, conv, kwargs):
    ident = jnp.arange(ssm.shape[0], dtype=jnp.int32)
    (new_conv, new_ssm), out = wrapper.fused_conv1d_gdn(
        states=(conv, ssm), state_indices=ident, **kwargs
    )
    return new_ssm, new_conv, out

  def _pooled(self, ssm, conv, idx, kwargs, write=_write_states_bf16):
    pool = write(_garbage_pool(0), ssm, conv, idx)
    out_pool, out = wrapper.fused_conv1d_gdn(
        states=_merge(pool),
        state_indices=idx,
        conv_dtype=jnp.float32,
        recurrent_dtype=jnp.bfloat16,
        **kwargs,
    )
    new_ssm, new_conv = _read_states_bf16(_split(out_pool, pool.shape), idx)
    return new_ssm, new_conv, out

  def _assert_same(self, got, want, ssm_ulps=0):
    diff = np.abs(
        _bits(got[0]).astype(np.int32) - _bits(want[0]).astype(np.int32)
    )
    self.assertLessEqual(int(diff.max()), ssm_ulps)
    np.testing.assert_allclose(
        np.asarray(got[1]), np.asarray(want[1]), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        np.asarray(got[2]), np.asarray(want[2]), rtol=2e-2, atol=2e-2
    )

  def _mixed_kwargs(self, seed):
    return _rand_inputs(seed, 130) | dict(
        query_start_loc=jnp.array([0, 1, 2, 66, 130]),
        distribution=jnp.array([2, 4, 4], dtype=jnp.int32),
        seq_lens=jnp.array([7, 9, 96, 80], dtype=jnp.int32),
    )

  def test_pooled_bf16_recurrent_matches_tuple_path(self):
    idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    ssm, conv = self._states(40, 4)
    kwargs = self._mixed_kwargs(41)
    self._assert_same(
        self._pooled(ssm, conv, idx, kwargs),
        self._tuple(ssm, conv, kwargs),
        ssm_ulps=32,
    )

  def test_bf16_recurrent_matches_fp32_on_bf16_values(self):
    ssm, conv = self._states(42, 4)
    kwargs = self._mixed_kwargs(43)
    ssm32, conv32 = ssm.astype(jnp.float32), conv.copy()
    got = self._tuple(ssm, conv, kwargs)
    new_ssm, new_conv, out = self._tuple(ssm32, conv32, kwargs)
    self._assert_same(got, (new_ssm.astype(jnp.bfloat16), new_conv, out))

  def test_multi_tile_prefill_with_bf16_recurrent_keeps_fp32_carry(self):
    ssm, conv = self._states(44, 1)
    kwargs = _rand_inputs(45, 200) | dict(
        query_start_loc=jnp.array([0, 200]),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([230], dtype=jnp.int32),
    )
    ssm32, conv32 = ssm.astype(jnp.float32), conv.copy()
    got = self._tuple(ssm, conv, kwargs)
    new_ssm, new_conv, out = self._tuple(ssm32, conv32, kwargs)
    self._assert_same(got, (new_ssm.astype(jnp.bfloat16), new_conv, out))

  def test_spec_windows_with_bf16_recurrent_match_fp32_on_bf16_values(self):
    spec = GdnPooledSpecWindowsTest()
    n, window = spec.N, spec.WINDOW
    base = jnp.array([1 + i * window for i in range(n)], dtype=jnp.int32)
    rng = np.random.default_rng(46)
    num_slots = 1 + n * window
    conv = jnp.asarray(
        rng.standard_normal((num_slots, _SEAM_KERNEL_SIZE - 1, _SEAM_DIM)),
        jnp.float32,
    )
    rec = jnp.asarray(
        rng.standard_normal((num_slots, _SEAM_N_V, _SEAM_D_K, _SEAM_D_V)),
        jnp.bfloat16,
    )
    conv16, rec16 = conv, rec
    conv32, rec32 = conv.copy(), rec.astype(jnp.float32)
    for step_num, step in enumerate(spec._steps()):
      kwargs = _rand_inputs(47 + step_num, step["num_tokens"]) | dict(
          query_start_loc=step["query_start_loc"],
          distribution=step["distribution"],
          seq_lens=step["seq_lens"],
          num_spec_tokens=spec.NUM_SPEC,
          read_offsets=step["read_offsets"],
      )
      (conv16, rec16), out16 = wrapper.fused_conv1d_gdn(
          states=(conv16, rec16), state_indices=base, **kwargs
      )
      (conv32, rec32), out32 = wrapper.fused_conv1d_gdn(
          states=(conv32, rec32), state_indices=base, **kwargs
      )
      rec32 = rec32.astype(jnp.bfloat16)
      self._assert_same((rec16, conv16, out16), (rec32, conv32, out32))
      rec32 = rec32.astype(jnp.float32)

  def test_conv_first_pool_layout_is_detected(self):
    """Negative control: regions written in the other order must not match."""

    def write_conv_first(pool, ssm, conv, idx, split=_SEAM_SPLIT):
      def build(pool, ssm, conv, idx):
        words = _state_words(pool, split)
        words = _put_words(words, idx, 0, jnp.asarray(conv, jnp.float32))
        body = _pack_bf16_rows(ssm).reshape(idx.shape[0], -1)
        start = _SEAM_CONV_GRANULES * _WORDS_PER_GRANULE
        words = words.at[idx, start : start + body.shape[1]].set(body)
        return _words_to_pool(words, pool)

      return _on_cpu(build, pool, ssm, conv, idx)

    idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    ssm, conv = self._states(48, 4)
    kwargs = _rand_inputs(49, 4) | dict(
        query_start_loc=jnp.arange(5),
        distribution=jnp.array([4, 4, 4], dtype=jnp.int32),
        seq_lens=jnp.full((4,), 9, dtype=jnp.int32),
    )
    got = self._pooled(ssm, conv, idx, kwargs, write=write_conv_first)
    want = self._tuple(ssm, conv, kwargs)
    self.assertFalse(np.array_equal(np.asarray(got[2]), np.asarray(want[2])))

  def test_bf16_conv_state_is_rejected(self):
    ssm, conv = self._states(50, 1)
    kwargs = _rand_inputs(51, 1) | dict(
        query_start_loc=jnp.arange(2),
        distribution=jnp.array([1, 1, 1], dtype=jnp.int32),
        seq_lens=jnp.array([3], dtype=jnp.int32),
    )
    with self.assertRaises(NotImplementedError):
      self._tuple(ssm, conv.astype(jnp.bfloat16), kwargs)


if __name__ == "__main__":
  absltest.main()
