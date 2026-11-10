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
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_kda
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.causal_conv1d_gated_delta_rule import test_base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import tiling
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper


def _to_words(x: jax.Array) -> jax.Array:
  num_lanes = pltpu.get_tpu_info().num_lanes
  packing = 4 // x.dtype.itemsize
  x = x.reshape(x.shape[0], -1, packing, num_lanes)
  return jax.lax.bitcast_convert_type(
      jnp.moveaxis(x, -2, -1), jnp.uint32
  ).reshape(x.shape[0], -1, 1, num_lanes)


def _from_words(
    words: jax.Array, shape: tuple[int, ...], dtype: jax.typing.DTypeLike
) -> jax.Array:
  dtype = jnp.dtype(dtype)
  if dtype.itemsize < 4:
    x = jax.lax.bitcast_convert_type(words.squeeze(-2), dtype)
    x = jnp.moveaxis(x, -1, -2)
  else:
    x = jax.lax.bitcast_convert_type(words, dtype)
  return x.reshape(shape)


def _pack_state_pool(
    conv_state: jax.Array, recurrent_state: jax.Array
) -> jax.Array:
  rec_words = _to_words(recurrent_state)
  conv_words = _to_words(conv_state)
  return jnp.concatenate([rec_words, conv_words], axis=1)


def _unpack_state_pool(
    pool: jax.Array,
    conv_shape: tuple[int, ...],
    conv_dtype: jax.typing.DTypeLike,
    rec_shape: tuple[int, ...],
    rec_dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
  num_lanes = pltpu.get_tpu_info().num_lanes
  rec_packing = 4 // jnp.dtype(rec_dtype).itemsize
  rec_granules = int(np.prod(rec_shape[1:])) // (num_lanes * rec_packing)
  rec_words = pool[:, :rec_granules]
  conv_words = pool[:, rec_granules:]
  return (
      _from_words(conv_words, conv_shape, conv_dtype),
      _from_words(rec_words, rec_shape, rec_dtype),
  )


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
    # size (8) than the cases above (16). Sized automatically by VMEM
    # heuristic (chunk_size=64 on 64 MiB TPUs, chunk_size=128 on 128 MiB TPUs).
    dict(
        name="kimi_linear_heads",
        max_reqs=2,
        q_loc=[0, 192, 256],
        distribution=[0, 2, 2],
        n_v=32,
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


def _unified_pool_cases():
  """Every shape and gate form under float32 and bfloat16 recurrent state."""
  for case in _cases():
    for rec_name, rec_dtype in (
        ("f32_rec", jnp.float32),
        ("bf16_rec", jnp.bfloat16),
    ):
      yield dict(
          case,
          testcase_name=f"{case['testcase_name']}_{rec_name}",
          recurrent_dtype=rec_dtype,
      )


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
  ):
    self._run_kda_matches_the_recurrence(
        max_reqs, q_loc, distribution, gate_lower_bound, n_v=n_v
    )

  def _run_kda_matches_the_recurrence(
      self,
      max_reqs,
      q_loc,
      distribution,
      gate_lower_bound,
      n_v=4,
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

    common_kwargs: dict[str, Any] = dict(
        qkv=qkv,
        b=b,
        a=a,
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

    (_, ref_state), ref_out = kda_attention_ref(
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        **common_kwargs,
    )
    (_, out_state), out = wrapper.fused_conv1d_gdn(
        states=(conv_state, recurrent_state),
        **common_kwargs,
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

  @parameterized.parameters(*test_base.SERVED_KDA_HEADS)
  def test_every_reachable_tile_matches_recurrence(self, n_kq, n_v):
    for batch in test_base.reachable_tile_batches(
        n_kq, n_v, jnp.float32, 1, config.AttentionMode.KDA
    ):
      with self.subTest(max_reqs=batch["max_reqs"], tokens=batch["q_loc"][-1]):
        self._run_kda_matches_the_recurrence(
            batch["max_reqs"],
            batch["q_loc"],
            batch["distribution"],
            gate_lower_bound=None,
            n_v=n_v,
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
        dtype=jnp.float32,
    )
    recurrent_state = jax.random.normal(next(rngs), (num_blocks, n_v, d_k, d_v))
    conv_weight = jax.random.normal(
        next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
    )
    a_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v, d_k))

    common_kwargs: dict[str, Any] = dict(
        qkv=qkv,
        b=b,
        a=a,
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
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        **common_kwargs,
        read_offsets=read_offsets_arr,
        num_spec_seqs=num_seqs,
    )
    (new_conv, new_rec), out = wrapper.fused_conv1d_gdn(
        states=(conv_state, recurrent_state),
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
      dict(
          testcase_name="bf16_conv_state",
          overrides=dict(),
          conv_dtype=jnp.bfloat16,
          error=NotImplementedError,
      ),
  )
  def test_rejects(self, overrides, error, conv_dtype=jnp.float32):
    """The entry point's guards, which all fail silently if dropped."""
    n_kq = n_v = 4
    d_k = d_v = 128
    kernel_size = 4
    dim_size = 2 * n_kq * d_k + n_v * d_v

    kwargs: dict[str, Any] = dict(
        qkv=jnp.zeros((8, dim_size)),
        b=jnp.zeros((8, n_v)),
        a=jnp.zeros((8, n_v * d_k)),
        states=(
            jnp.zeros((9, kernel_size - 1, dim_size), dtype=conv_dtype),
            jnp.zeros((9, n_v, d_k, d_v)),
        ),
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

  def test_kda_spec_window_multi_sequence_tile(self):
    """KDA verify window (W=4, n_v=3) runs at multi-sequence tile T >= 2."""
    n_kq = n_v = 3
    d_k = d_v = 128
    kernel_size = 4
    num_spec_tokens = 3
    window = num_spec_tokens + 1
    num_seqs = 24
    batch_size = num_seqs * window
    dim_size = 2 * n_kq * d_k + n_v * d_v
    num_blocks = 1 + num_seqs * window

    decode_tile, _ = tiling.get_tile_sizes(
        batch_size=batch_size,
        num_seqs=num_seqs,
        padded_batch_size=batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        conv_state_dim_size=dim_size,
        act_out_dtype=jnp.dtype(jnp.bfloat16),
        recurrent_state_dtype=jnp.dtype(jnp.float32),
        num_lanes=pltpu.get_tpu_info().num_lanes,
        decode_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(window),
        mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
        window_size=window,
        attention_mode=config.AttentionMode.KDA,
    )
    self.assertGreaterEqual(decode_tile, 2)
    self.assertGreaterEqual((num_seqs + decode_tile - 1) // decode_tile, 3)

    q_loc = jnp.arange(0, batch_size + 1, window, dtype=jnp.int32)
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    state_indices = jnp.array(
        [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
    )
    read_offsets_arr = jnp.array(
        [i % window for i in range(num_seqs)], dtype=jnp.int32
    )
    seq_lens = jnp.full((num_seqs,), window + 10, dtype=jnp.int32)

    rngs = iter(jax.random.split(jax.random.key(29), 8))
    qkv = jax.random.normal(
        next(rngs), (batch_size, dim_size), dtype=jnp.bfloat16
    )
    b = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(
        next(rngs), (batch_size, n_v * d_k), dtype=jnp.bfloat16
    )
    conv_state = jax.random.normal(
        next(rngs),
        (num_blocks, kernel_size - 1, dim_size),
        dtype=jnp.float32,
    )
    recurrent_state = jax.random.normal(
        next(rngs), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
    )
    a_log = jax.random.normal(next(rngs), (n_v,))
    dt_bias = jax.random.normal(next(rngs), (n_v, d_k))

    common_kwargs: dict[str, Any] = dict(
        qkv=qkv,
        b=b,
        a=a,
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
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        **common_kwargs,
        read_offsets=read_offsets_arr,
        num_spec_seqs=num_seqs,
    )
    for _ in range(3):
      (new_conv, new_rec), out = wrapper.fused_conv1d_gdn(
          states=(conv_state.copy(), recurrent_state.copy()),
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

  @parameterized.named_parameters(*_unified_pool_cases())
  def test_kda_unified_pool_matches_the_recurrence(
      self,
      max_reqs,
      q_loc,
      distribution,
      gate_lower_bound,
      recurrent_dtype,
      n_v=4,
  ):
    """KDA unified granule pool (`is_shared=True`) matches the oracle."""
    n_kq = n_v
    d_k = d_v = 128
    kernel_size = 4

    q_loc = jnp.array(q_loc, dtype=jnp.int32)
    distribution = jnp.array(distribution, dtype=jnp.int32)
    num_tokens = int(q_loc[max_reqs])

    state_indices = jnp.arange(1, max_reqs + 1)
    num_blocks = max_reqs + 1
    dim_size = 2 * n_kq * d_k + n_v * d_v

    rngs = iter(jax.random.split(jax.random.key(3), 10))
    qkv = jax.random.normal(
        next(rngs), (num_tokens, dim_size), dtype=jnp.bfloat16
    )
    b = jax.random.normal(next(rngs), (num_tokens, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(
        next(rngs), (num_tokens, n_v * d_k), dtype=jnp.bfloat16
    )

    conv_state = jax.random.normal(
        next(rngs),
        (num_blocks, kernel_size - 1, dim_size),
        dtype=jnp.float32,
    )
    recurrent_state = jax.random.normal(
        next(rngs),
        (num_blocks, n_v, d_k, d_v),
        dtype=recurrent_dtype,
    )
    conv_weight = jax.random.normal(
        next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
    )
    conv_bias = jax.random.normal(next(rngs), (dim_size,), dtype=jnp.bfloat16)
    a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
    dt_bias = jax.random.normal(next(rngs), (n_v, d_k), dtype=jnp.float32)

    # Give sequences non-zero history so initial conv_state and recurrent_state
    # reads from the shared pool are exercised.
    seq_lens = (
        jnp.asarray(q_loc[1 : max_reqs + 1] - q_loc[:max_reqs], dtype=jnp.int32)
        + 10
    )

    ref_kwargs = dict(
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
    (ref_conv, ref_rec), ref_out = kda_attention_ref(**ref_kwargs)

    pool = _pack_state_pool(conv_state, recurrent_state)
    new_pool, out = wrapper.fused_conv1d_gdn(
        qkv=qkv,
        b=b,
        a=a,
        states=pool,
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
        conv_dtype=jnp.float32,
        recurrent_dtype=recurrent_dtype,
    )
    new_conv, new_rec = _unpack_state_pool(
        new_pool,
        conv_state.shape,
        conv_state.dtype,
        recurrent_state.shape,
        recurrent_dtype,
    )

    np.testing.assert_allclose(
        np.asarray(out, np.float64), ref_out, rtol=3e-2, atol=3e-2
    )
    np.testing.assert_allclose(
        np.asarray(new_conv, np.float64),
        np.asarray(ref_conv, np.float64),
        rtol=2e-2,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(new_rec, np.float64),
        np.asarray(ref_rec, np.float64),
        rtol=3e-2,
        atol=3e-2,
    )

  def test_kda_vmem_and_tile_sizes_across_parallelism(self):
    """KDA VMEM accounting and tile picks across TP/DP/DCP/PCP/MTP regimes."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    d_k = d_v = 128
    kernel_size = 4
    vmem_64m = 64 * 1024 * 1024
    limit_w1 = config.GDNConfig.get_vmem_limit_bytes(
        1, vmem_capacity_bytes=vmem_64m
    )
    limit_w4 = config.GDNConfig.get_vmem_limit_bytes(
        4, vmem_capacity_bytes=vmem_64m
    )

    dim_32 = 2 * 32 * d_k + 32 * d_v
    est_kda_c128 = tiling.get_vmem_estimate_bytes(
        tile_b=1,
        chunk_sz=128,
        n_kq=32,
        n_v=32,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        act_out_bytes=2,
        rec_state_bytes=4,
        num_lanes=num_lanes,
        conv_state_dim_size=dim_32,
        is_decode=False,
        window_size=1,
        attention_mode=config.AttentionMode.KDA,
    )
    self.assertEqual(est_kda_c128, 59_884_032)
    self.assertGreater(est_kda_c128, limit_w1)

    dp8_dec, dp8_mix = tiling.get_tile_sizes(
        batch_size=128,
        num_seqs=64,
        padded_batch_size=128,
        n_kq=32,
        n_v=32,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        conv_state_dim_size=dim_32,
        act_out_dtype=jnp.dtype(jnp.bfloat16),
        recurrent_state_dtype=jnp.dtype(jnp.float32),
        num_lanes=num_lanes,
        decode_vmem_limit_bytes=limit_w1,
        mixed_vmem_limit_bytes=limit_w1,
        window_size=1,
        attention_mode=config.AttentionMode.KDA,
    )
    self.assertEqual((dp8_dec, dp8_mix), (4, 64))

    for n_kq, n_v, w_sz, expected in (
        (3, 3, 1, (16, 128)),
        (3, 3, 4, (16, 128)),
        (12, 12, 1, (8, 128)),
        (12, 12, 4, (4, 128)),
        (4, 4, 1, (16, 128)),
        (4, 4, 4, (16, 128)),
        (32, 32, 1, (4, 64)),
        (32, 32, 4, (2, 64)),
    ):
      dim = 2 * n_kq * d_k + n_v * d_v
      got = tiling.get_tile_sizes(
          batch_size=512,
          num_seqs=64,
          padded_batch_size=512,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          kernel_size=kernel_size,
          conv_state_dim_size=dim,
          act_out_dtype=jnp.dtype(jnp.bfloat16),
          recurrent_state_dtype=jnp.dtype(jnp.float32),
          num_lanes=num_lanes,
          decode_vmem_limit_bytes=limit_w4 if w_sz > 1 else limit_w1,
          mixed_vmem_limit_bytes=limit_w1,
          window_size=w_sz,
          attention_mode=config.AttentionMode.KDA,
      )
      self.assertEqual(got, expected)

    dummy_arr_dt: Any = jnp.dtype(jnp.float32)
    dummy_out_dt: Any = jnp.dtype(jnp.bfloat16)
    for n_kq, n_v, tile_sz, is_decode, w_sz in (
        (3, 3, 16, True, 4),
        (12, 12, 4, True, 4),
        (4, 4, 16, True, 4),
        (32, 32, 2, True, 4),
        (32, 32, 64, False, 1),
    ):
      dim = 2 * n_kq * d_k + n_v * d_v
      aligned_dim = tiling.align_to(dim, num_lanes)
      cfg = config.GDNConfig(
          mode=config.GDNMode.BATCHED if is_decode else config.GDNMode.PER_SEQ,
          dtypes=config.Dtypes(
              act_in=jnp.dtype(jnp.float32),
              act_out=jnp.dtype(jnp.bfloat16),
              compute=jnp.dtype(jnp.float32),
              recurrent_state=jnp.dtype(jnp.float32),
              conv_state=jnp.dtype(jnp.float32),
          ),
          batch_size=tile_sz,
          dim_size=aligned_dim,
          kernel_size=kernel_size,
          tile_size=tile_sz,
          num_kq_heads=n_kq,
          num_v_heads=n_v,
          kq_head_dim=d_k,
          v_head_dim=d_v,
          num_buffers=2,
          window_size=w_sz,
          attention_mode=config.AttentionMode.KDA,
      )
      dummy_meta = memory_ref.MetadataRef(
          num_tiles=None,
          records=None,
          s_idx_has_initial_state=None,
          s_idx_to_state_indices=None,
          s_idx_to_read_offset=None,
          s_idx_to_read_indices=None,
          s_idx_to_ckpt_indices=None,
          shape=(1, cfg.seq_tile_size),
      )
      allocs = memory_ref.create_allocs(
          metadata_ref=dummy_meta,
          qkv_ref=dummy_arr_dt,
          b_ref=dummy_arr_dt,
          a_ref=dummy_arr_dt,
          out_ref=dummy_out_dt,
          cfg=cfg,
      )
      declared_allocs_bytes = sum(
          int(np.prod(a.window_ref.shape))
          * jnp.dtype(a.window_ref.dtype).itemsize
          for a in allocs
          if a.window_ref is not None
      )
      scratch_shapes = cfg.get_scratch_shape_dict()
      declared_scratch_bytes = sum(
          int(np.prod(s.shape)) * 4
          for s in scratch_shapes.values()
          if s is not None
      )
      weights_bytes = (kernel_size + 1) * aligned_dim * 4 + (
          tiling.align_to(n_v, num_lanes)
          + tiling.align_to(n_v * d_k, num_lanes)
      ) * 4
      total_declared = (
          declared_allocs_bytes + declared_scratch_bytes + weights_bytes
      )

      tile_b = cfg.seq_tile_size
      chunk_sz = cfg.chunk_size
      vmem_declared = tiling.get_vmem_declared_bytes(
          tile_b=tile_b,
          chunk_sz=chunk_sz,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          kernel_size=kernel_size,
          act_out_bytes=2,
          rec_state_bytes=4,
          num_lanes=num_lanes,
          conv_state_dim_size=aligned_dim,
          is_decode=is_decode,
          window_size=w_sz,
          attention_mode=config.AttentionMode.KDA,
      )
      self.assertEqual(vmem_declared, total_declared)

  def test_recurrent_kda_batched_multi_token_window_with_ragged_masks(self):
    """Batched recurrent_kda matches explicit per-sequence 2-dot reference across W=1, 4 and ragged real_sizes."""
    for w_sz in (1, 4):
      for act_dt in (jnp.float32, jnp.bfloat16):
        cfg = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=config.Dtypes(
                act_in=jnp.dtype(act_dt),
                act_out=jnp.dtype(act_dt),
                compute=jnp.dtype(jnp.float32),
                recurrent_state=jnp.dtype(jnp.float32),
                conv_state=jnp.dtype(jnp.float32),
            ),
            batch_size=8 * w_sz,
            dim_size=2048,
            kernel_size=4,
            tile_size=8,
            num_kq_heads=4,
            num_v_heads=8,
            kq_head_dim=128,
            v_head_dim=128,
            num_buffers=2,
            window_size=w_sz,
            attention_mode=config.AttentionMode.KDA,
        )
        real_sizes = jnp.array(
            [w_sz, max(1, w_sz - 1), w_sz, 1, w_sz, w_sz, 0, w_sz],
            dtype=jnp.int32,
        )
        q_c = jax.random.normal(
            jax.random.key(42), (8, 4, w_sz, 1, 128), dtype=act_dt
        )
        k_c = jax.random.normal(
            jax.random.key(43), (8, 4, w_sz, 1, 128), dtype=act_dt
        )
        v_c = jax.random.normal(
            jax.random.key(44), (8, 8, w_sz, 1, 128), dtype=act_dt
        )
        b_c = jax.random.normal(
            jax.random.key(45), (8, 1, w_sz, 1, 128), dtype=act_dt
        )
        a_c = jax.random.normal(
            jax.random.key(46), (8, 8, w_sz, 1, 128), dtype=act_dt
        )
        s_prev = jax.random.normal(
            jax.random.key(47), (8, 8, 128, 128), dtype=jnp.float32
        )
        a_log = jax.random.normal(jax.random.key(48), (8,), dtype=jnp.float32)
        dt_bias = jax.random.normal(
            jax.random.key(49), (8, 128), dtype=jnp.float32
        )

        with jax.default_matmul_precision("highest"):
          out, state = compute_kda.recurrent_kda(
              real_sizes, q_c, k_c, v_c, b_c, a_c, s_prev, a_log, dt_bias, cfg
          )
          # Build explicit per-sequence 2-dot reference across all 8 sequence slots.
          iota = jax.lax.broadcasted_iota(jnp.int32, (8, 1, w_sz, 1, 1), 2)
          mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1)
          q_m = compute_kda._l2_norm_f32(
              jnp.where(mask, q_c.astype(jnp.float32), 0.0)
          ) * (128**-0.5)
          k_m = compute_kda._l2_norm_f32(
              jnp.where(mask, k_c.astype(jnp.float32), 0.0)
          )
          v_m = jnp.where(mask, v_c.astype(jnp.float32), 0.0)
          beta_m = jnp.where(mask, jax.nn.sigmoid(b_c.astype(jnp.float32)), 0.0)
          if w_sz > 1:
            beta_m = beta_m.astype(act_dt).astype(jnp.float32)
          beta_m = jnp.swapaxes(beta_m, 1, 4)[:, :8, :, 0, :]
          g_log = jnp.where(
              mask,
              compute_kda.activate_gate(
                  a_c,
                  a_log.reshape(1, 8, 1, 1, 1),
                  dt_bias.reshape(1, 8, 1, 1, 128),
                  cfg,
              ),
              0.0,
          )
          g_dec = jnp.exp(g_log)[:, :, :, 0, :]
          q_m, k_m, v_m = (
              q_m[:, :, :, 0, :],
              k_m[:, :, :, 0, :],
              v_m[:, :, :, 0, :],
          )
          ref_outs, ref_states = [], []
          for s_i in range(8):
            st = s_prev[s_i]
            o_seq, s_seq = [], []
            for c_i in range(w_sz):
              qh = jnp.repeat(q_m[s_i, :, c_i], 2, axis=0)[:, None, :]
              kh = jnp.repeat(k_m[s_i, :, c_i], 2, axis=0)[:, None, :]
              kht = jnp.swapaxes(kh, 1, 2)
              st_u = st * g_dec[s_i, :, c_i, :, None]
              vu = jax.lax.dot(
                  kh,
                  st_u,
                  dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                  preferred_element_type=jnp.float32,
              )
              vn = beta_m[s_i, :, c_i : c_i + 1] * (
                  v_m[s_i, :, c_i : c_i + 1] - vu
              )
              st = st_u + kht * vn
              ot = jax.lax.dot(
                  qh,
                  st,
                  dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                  preferred_element_type=jnp.float32,
              )
              o_seq.append(ot[:, 0, :])
              s_seq.append(st)
            ref_outs.append(jnp.stack(o_seq, axis=0))
            ref_states.append(jnp.stack(s_seq, axis=0))
          ref_out = jnp.stack(ref_outs, axis=0)
          ref_state = jnp.stack(ref_states, axis=0)

        np.testing.assert_allclose(
            np.asarray(out, np.float32),
            np.asarray(ref_out, np.float32),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(state, np.float32),
            np.asarray(ref_state, np.float32),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(out[6], np.float32), 0.0, atol=1e-5
        )
        if w_sz == 4:
          np.testing.assert_allclose(
              np.asarray(state[1, 3], np.float32),
              np.asarray(state[1, 2], np.float32),
              rtol=1e-5,
              atol=1e-5,
          )
          np.testing.assert_allclose(
              np.asarray(state[3, 3], np.float32),
              np.asarray(state[3, 0], np.float32),
              rtol=1e-5,
              atol=1e-5,
          )


if __name__ == "__main__":
  absltest.main()
