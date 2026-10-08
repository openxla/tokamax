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
"""Correctness tests for the KDA path of the fused conv1d + GDN kernel.

What these tests exist for is the wiring: the per-channel gate is `kq_head_dim`
times wider than GDN's per-head scalar, so it travels through a different
BlockSpec and a different VMEM load on its way into the kernel.

The oracle is a float64 token-at-a-time recurrence, preceded by the conv1d
and silu that the fused kernel folds in. Comparing against a recurrence rather
than against another Pallas kernel keeps this independent of the implementation
it is meant to check.

Both gate forms are covered. `gate_lower_bound=None` gives
`-exp(A_log) * softplus(g + dt_bias)`, which Kimi-Linear-48B uses; a float
gives `gate_lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, which bounded
configurations use with -5.0. They are different functions of the same
weights, so each needs its own oracle.
"""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import test_base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper


def _l2_normalize(x: np.ndarray) -> np.ndarray:
  """Mirror the kernel's `_l2_norm_f32`, which accumulates in float32."""
  x_f32 = x.astype(np.float32)
  inv_norm = 1.0 / np.sqrt(np.sum(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6)
  return (x_f32 * inv_norm).astype(np.float64)


def _recurrence(
    q: np.ndarray,  # [T, H, K], normalised
    k: np.ndarray,  # [T, H, K], normalised
    v: np.ndarray,  # [T, H, V]
    gate: np.ndarray,  # [T, H, K], activated (natural log)
    beta: np.ndarray,  # [T, H], through the sigmoid
    state: np.ndarray,  # [H, K, V]
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
  """One token at a time, in float64."""
  running = state.astype(np.float64).copy()
  output = np.zeros((q.shape[0], q.shape[1], v.shape[-1]), np.float64)
  for token in range(q.shape[0]):
    # The per-channel gate decays along K, hence the trailing axis.
    running *= np.exp(gate[token])[:, :, None]
    prediction = np.einsum("hk,hkv->hv", k[token], running)
    delta = beta[token][:, None] * (v[token] - prediction)
    running += k[token][:, :, None] * delta[:, None, :]
    output[token] = scale * np.einsum("hk,hkv->hv", q[token], running)
  return output, running


def kda_attention_ref(
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
    attention_mode: config.AttentionMode = config.AttentionMode.KDA,
    gate_lower_bound: float | None = None,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], np.ndarray]:
  """Reference conv1d + silu + KDA, sequence by sequence in eager mode."""
  del attention_mode  # Only the KDA form is implemented here.

  num_tokens = qkv.shape[0]
  num_valid_seqs = int(distribution[2])

  out_mixed_qkv = jnp.zeros_like(qkv)
  new_conv_state = jnp.array(conv_state)
  new_recurrent_state = jnp.array(recurrent_state)
  output = np.zeros((num_tokens, n_v * d_v), np.float64)

  # Part 1: Conv1D, over the raw activations.
  for req_idx in range(num_valid_seqs):
    s = int(state_indices[req_idx])
    start = int(query_start_loc[req_idx])
    end = int(query_start_loc[req_idx + 1])
    query_len = end - start
    if query_len <= 0:
      continue

    has_init = bool((seq_lens[req_idx] - query_len) > 0)
    if has_init:
      c_state = new_conv_state[s]
    else:
      c_state = jnp.zeros_like(new_conv_state[s])
      new_recurrent_state = new_recurrent_state.at[s].set(
          jnp.zeros_like(new_recurrent_state[s])
      )

    x_full = jnp.concatenate([c_state, qkv[start:end]], axis=0)
    acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
    for k_idx in range(kernel_size):
      acc += (
          x_full[k_idx : k_idx + query_len].astype(jnp.float32)
          * conv_weight[:, 0, k_idx].astype(jnp.float32)[None, :]
      )
    if conv_bias is not None:
      acc += conv_bias.astype(jnp.float32)[None, :]
    new_conv_state = new_conv_state.at[s].set(x_full[-(kernel_size - 1) :])
    out_mixed_qkv = out_mixed_qkv.at[start:end].set(acc.astype(qkv.dtype))

  out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

  # Part 2: KDA, in float64.
  a_log_np = np.asarray(a_log, np.float64).reshape(n_v, 1)
  dt_bias_np = np.asarray(dt_bias, np.float64).reshape(n_v, d_k)
  scale = float(d_k**-0.5)

  for req_idx in range(num_valid_seqs):
    s = int(state_indices[req_idx])
    start = int(query_start_loc[req_idx])
    end = int(query_start_loc[req_idx + 1])
    query_len = end - start
    if query_len <= 0:
      continue

    qkv_seq = np.asarray(out_mixed_qkv[start:end], np.float64)
    key_dim = n_kq * d_k
    q_seq = qkv_seq[:, :key_dim].reshape(query_len, n_kq, d_k)
    k_seq = qkv_seq[:, key_dim : 2 * key_dim].reshape(query_len, n_kq, d_k)
    v_seq = qkv_seq[:, 2 * key_dim :].reshape(query_len, n_v, d_v)

    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
      q_seq = np.repeat(q_seq, repeat_factor, axis=1)
      k_seq = np.repeat(k_seq, repeat_factor, axis=1)

    q_seq = _l2_normalize(q_seq)
    k_seq = _l2_normalize(k_seq)

    beta_seq = 1.0 / (1.0 + np.exp(-np.asarray(b[start:end], np.float64)))

    # The gate is per (head, channel): [T, H, K].
    a_seq = np.asarray(a[start:end], np.float64).reshape(query_len, n_v, d_k)
    if gate_lower_bound is None:
      gate_seq = -np.exp(a_log_np) * np.logaddexp(0.0, a_seq + dt_bias_np)
    else:
      # sigmoid(x) written as exp(-logaddexp(0, -x)) to avoid overflow
      # in the exponential for large negative x.
      gate_seq = gate_lower_bound * np.exp(
          -np.logaddexp(0.0, -(np.exp(a_log_np) * (a_seq + dt_bias_np)))
      )

    out_seq, final_state = _recurrence(
        q_seq,
        k_seq,
        v_seq,
        gate_seq,
        beta_seq,
        np.asarray(new_recurrent_state[s], np.float64),
        scale,
    )
    new_recurrent_state = new_recurrent_state.at[s].set(
        jnp.asarray(final_state, new_recurrent_state.dtype)
    )
    output[start:end] = out_seq.reshape(query_len, n_v * d_v)

  return (new_conv_state, new_recurrent_state), output


def kda_attention_spec_ref(
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
    read_offsets: jnp.ndarray,
    num_spec_seqs: int,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    gate_lower_bound: float | None,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Token reference for KDA verify windows and their checkpoints."""
  del distribution
  new_conv_state = jnp.array(conv_state)
  new_recurrent_state = jnp.array(recurrent_state)
  output = np.zeros((qkv.shape[0], n_v * d_v), np.float64)
  a_log_np = np.asarray(a_log, np.float64).reshape(n_v, 1)
  dt_bias_np = np.asarray(dt_bias, np.float64).reshape(n_v, d_k)

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
    running = np.asarray(
        recurrent_state[read_slot]
        if has_init
        else jnp.zeros_like(recurrent_state[read_slot]),
        np.float64,
    ).copy()

    x_full = jnp.concatenate([c_state, qkv[start:end]], axis=0)
    acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
    for k_idx in range(kernel_size):
      acc += (
          x_full[k_idx : k_idx + query_len].astype(jnp.float32)
          * conv_weight[:, 0, k_idx].astype(jnp.float32)[None, :]
      )
    if conv_bias is not None:
      acc += conv_bias.astype(jnp.float32)[None, :]
    conv_out = jax.nn.silu(acc).astype(qkv.dtype)
    for t in range(query_len):
      new_conv_state = new_conv_state.at[base + t].set(
          x_full[t + 1 : t + kernel_size]
      )

    conv_out_np = np.asarray(conv_out, np.float64)
    key_dim = n_kq * d_k
    q_seq = conv_out_np[:, :key_dim].reshape(query_len, n_kq, d_k)
    k_seq = conv_out_np[:, key_dim : 2 * key_dim].reshape(query_len, n_kq, d_k)
    v_seq = conv_out_np[:, 2 * key_dim :].reshape(query_len, n_v, d_v)
    if n_v != n_kq:
      repeat_factor = n_v // n_kq
      q_seq = np.repeat(q_seq, repeat_factor, axis=1)
      k_seq = np.repeat(k_seq, repeat_factor, axis=1)
    q_seq = _l2_normalize(q_seq) * float(d_k**-0.5)
    k_seq = _l2_normalize(k_seq)
    # Production activates beta in fp32, then stores it in the model
    # activation dtype before the recurrent update.
    beta_seq = np.asarray(
        jax.nn.sigmoid(b[start:end].astype(jnp.float32)).astype(b.dtype),
        np.float64,
    )
    a_seq = np.asarray(a[start:end], np.float64).reshape(query_len, n_v, d_k)
    if gate_lower_bound is None:
      gate_seq = -np.exp(a_log_np) * np.logaddexp(0.0, a_seq + dt_bias_np)
    else:
      gate_seq = gate_lower_bound * np.exp(
          -np.logaddexp(0.0, -(np.exp(a_log_np) * (a_seq + dt_bias_np)))
      )

    for t in range(query_len):
      running *= np.exp(gate_seq[t])[:, :, None]
      prediction = np.einsum("hk,hkv->hv", k_seq[t], running)
      delta = beta_seq[t][:, None] * (v_seq[t] - prediction)
      running += k_seq[t][:, :, None] * delta[:, None, :]
      output[start + t] = np.einsum("hk,hkv->hv", q_seq[t], running).reshape(-1)
      new_recurrent_state = new_recurrent_state.at[base + t].set(
          jnp.asarray(running, recurrent_state.dtype)
      )

  return (new_conv_state, new_recurrent_state), jnp.asarray(output, qkv.dtype)


_SHAPES = (
    dict(
        name="decode_only",
        max_reqs=8,
        q_loc=list(range(9)),
        distribution=[8, 8, 8],
    ),
    dict(
        name="single_prefill",
        max_reqs=1,
        q_loc=[0, 256],
        distribution=[0, 1, 1],
    ),
    dict(
        name="mixed_prefill",
        max_reqs=3,
        q_loc=[0, 192, 320, 384],
        distribution=[0, 3, 3],
    ),
    dict(
        name="ragged_prefill",
        max_reqs=4,
        q_loc=[0, 65, 130, 131, 195],
        distribution=[0, 4, 4],
    ),
    # Kimi-Linear-48B's head count. `triangular_block_size` is a step
    # function of `num_v_heads`, so this covers a different sub-block
    # size (8) than the cases above (16). Sized to chunk_size=64 to fit the
    # 4096-channel gate (n_v * d_k) within the 51.20 MiB scoped VMEM limit.
    dict(
        name="kimi_linear_heads",
        max_reqs=2,
        q_loc=[0, 192, 256],
        distribution=[0, 2, 2],
        n_v=32,
        mixed_tile_size=64,
    ),
)

# Kimi-Linear-48B sets no bound; bounded configurations set -5.0.
_GATES = (("softplus", None), ("bounded", -5.0))


def _cases():
  """Every shape under both gate forms."""
  for shape in _SHAPES:
    for gate_name, lower_bound in _GATES:
      case: dict[str, Any] = dict(shape)
      case["testcase_name"] = f"{case.pop('name')}_{gate_name}"
      case["gate_lower_bound"] = lower_bound
      yield case


class KdaAttentionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_base.skip_if_unsupported(self)

  @parameterized.named_parameters(*_cases())
  def test_kda_matches_the_recurrence(
      self,
      max_reqs,
      q_loc,
      distribution,
      gate_lower_bound,
      n_v=4,
      mixed_tile_size=None,
  ):
    n_kq = n_v
    d_k = d_v = 128
    kernel_size = 4

    q_loc = jnp.array(q_loc, dtype=jnp.int32)
    distribution = jnp.array(distribution, dtype=jnp.int32)
    num_tokens = int(q_loc[max_reqs])

    # Slot 0 is the reserved null block for invalid / padded tokens.
    state_indices = jnp.arange(1, max_reqs + 1)
    num_blocks = max_reqs + 1
    dim_size = 2 * n_kq * d_k + n_v * d_v

    rngs = iter(jax.random.split(jax.random.key(0), 8))
    qkv = jax.random.normal(next(rngs), (num_tokens, dim_size))
    b = jax.random.normal(next(rngs), (num_tokens, n_v))
    # The KDA gate is per-channel: n_v * d_k wide, not n_v.
    a = jax.random.normal(next(rngs), (num_tokens, n_v * d_k))

    conv_state = jnp.zeros((num_blocks, kernel_size - 1, dim_size))
    recurrent_state = jnp.zeros((num_blocks, n_v, d_k, d_v))
    conv_weight = jax.random.normal(next(rngs), (dim_size, 1, kernel_size))
    conv_bias = jax.random.normal(next(rngs), (dim_size,))

    a_log = jax.random.normal(next(rngs), (n_v,))
    # Per-channel, and kept 2D: the kernel reshapes it to broadcast over
    # (head, channel), which Mosaic will not do from a flat vector.
    dt_bias = jax.random.normal(next(rngs), (n_v, d_k))

    seq_lens = jnp.asarray(
        q_loc[1 : max_reqs + 1] - q_loc[:max_reqs], dtype=jnp.int32
    )

    common_kwargs = dict(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
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
        attention_mode=config.AttentionMode.KDA,
        gate_lower_bound=gate_lower_bound,
    )

    (_, ref_state), ref_out = kda_attention_ref(**common_kwargs)
    (_, out_state), out = wrapper.fused_conv1d_gdn(
        **common_kwargs, mixed_tile_size=mixed_tile_size
    )

    np.testing.assert_allclose(
        np.asarray(out, np.float64), ref_out, rtol=2e-2, atol=5e-3
    )
    np.testing.assert_allclose(
        np.asarray(out_state, np.float64),
        np.asarray(ref_state, np.float64),
        rtol=2e-2,
        atol=1e-2,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="tp32_multi_tile_recurrent_window",
          num_heads=3,
          spec_lengths=[1, 6, 3, 8, 5, 2, 7, 4] * 4,
          read_offsets=[0, 2, 4, 6, 0, 2, 4, 6] * 4,
      ),
      dict(
          testcase_name="tp8_large_recurrent_window",
          num_heads=12,
          spec_lengths=[1, 6, 3, 8, 5, 2, 7, 4],
          read_offsets=[0, 2, 4, 6, 0, 2, 4, 6],
      ),
  )
  def test_speculative_window_checkpoints(
      self, num_heads, spec_lengths, read_offsets
  ):
    """An 8-token verify window stays in one kernel invocation."""
    n_kq = n_v = num_heads
    d_k = d_v = 128
    kernel_size = 4
    num_spec_tokens = 7
    window = num_spec_tokens + 1
    num_seqs = len(spec_lengths)
    q_loc = jnp.array(
        np.concatenate([[0], np.cumsum(spec_lengths)]), dtype=jnp.int32
    )
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    state_indices = jnp.array(
        [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
    )
    num_blocks = 1 + num_seqs * window
    read_offsets_arr = jnp.array(read_offsets, dtype=jnp.int32)
    seq_lens = jnp.array(
        [32 + length for length in spec_lengths], dtype=jnp.int32
    )
    dim_size = 2 * n_kq * d_k + n_v * d_v

    rngs = iter(jax.random.split(jax.random.key(11), 8))
    qkv = jax.random.normal(
        next(rngs), (sum(spec_lengths), dim_size), dtype=jnp.bfloat16
    )
    b = jax.random.normal(
        next(rngs), (sum(spec_lengths), n_v), dtype=jnp.bfloat16
    )
    a = jax.random.normal(
        next(rngs), (sum(spec_lengths), n_v * d_k), dtype=jnp.bfloat16
    )
    conv_state = jax.random.normal(
        next(rngs),
        (num_blocks, kernel_size - 1, dim_size),
        dtype=jnp.bfloat16,
    )
    recurrent_state = jax.random.normal(next(rngs), (num_blocks, n_v, d_k, d_v))
    conv_weight = jax.random.normal(
        next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
    )
    a_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v, d_k))

    common_kwargs = dict(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        conv_weight=conv_weight,
        conv_bias=None,
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
        gate_lower_bound=-5.0,
    )

    (ref_conv, ref_rec), ref_out = kda_attention_spec_ref(
        **common_kwargs,
        read_offsets=read_offsets_arr,
        num_spec_seqs=num_seqs,
    )
    (new_conv, new_rec), out = wrapper.fused_conv1d_gdn(
        **common_kwargs,
        read_offsets=read_offsets_arr,
        num_spec_tokens=num_spec_tokens,
        attention_mode=config.AttentionMode.KDA,
    )

    np.testing.assert_allclose(out, ref_out, rtol=2e-2, atol=5e-3)
    for slot in range(num_blocks):
      np.testing.assert_allclose(
          new_conv[slot],
          ref_conv[slot],
          rtol=2e-2,
          atol=5e-3,
          err_msg=f"conv checkpoint mismatch at slot {slot}",
      )
      np.testing.assert_allclose(
          new_rec[slot],
          ref_rec[slot],
          rtol=2e-2,
          atol=1e-2,
          err_msg=f"recurrent checkpoint mismatch at slot {slot}",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="bound_under_gdn",
          overrides=dict(
              attention_mode=config.AttentionMode.GDN, gate_lower_bound=-5.0
          ),
          error=ValueError,
      ),
      dict(
          testcase_name="non_negative_bound",
          overrides=dict(gate_lower_bound=0.0),
          error=ValueError,
      ),
  )
  def test_rejects(self, overrides, error):
    """The entry point's guards, which all fail silently if dropped."""
    n_kq = n_v = 4
    d_k = d_v = 128
    kernel_size = 4
    dim_size = 2 * n_kq * d_k + n_v * d_v

    kwargs = dict(
        qkv=jnp.zeros((8, dim_size)),
        b=jnp.zeros((8, n_v)),
        a=jnp.zeros((8, n_v * d_k)),
        conv_state=jnp.zeros((9, kernel_size - 1, dim_size)),
        recurrent_state=jnp.zeros((9, n_v, d_k, d_v)),
        conv_weight=jnp.zeros((dim_size, 1, kernel_size)),
        conv_bias=None,
        a_log=jnp.zeros((n_v,)),
        dt_bias=jnp.zeros((n_v, d_k)),
        query_start_loc=jnp.arange(9, dtype=jnp.int32),
        state_indices=jnp.arange(1, 9),
        distribution=jnp.array([8, 8, 8], dtype=jnp.int32),
        seq_lens=jnp.ones((8,), dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        attention_mode=config.AttentionMode.KDA,
    )
    kwargs.update(overrides)
    with self.assertRaises(error):
      wrapper.fused_conv1d_gdn(**kwargs)


if __name__ == "__main__":
  absltest.main()
