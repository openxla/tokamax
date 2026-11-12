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
"""Unit tests for Mamba-2 and Mamba-3 Selective State Space (SSD) scan."""

# pylint: disable=invalid-name

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import export
import jax.numpy as jnp
import numpy as np
from tokamax._src import jaxtyping as tokamax_jaxtyping
from tokamax._src import mosaic_tpu
from tokamax._src.ops.experimental import ssd_scan
from tokamax._src.ops.experimental.ssd_scan import ssd_scan as ssd_scan_lib


def _ssd_sequential_reference(
    x_c: jax.Array,
    a_c: jax.Array,
    b_c: jax.Array,
    c_c: jax.Array,
    last_chunk_indices: jax.Array,
    initial_states: jax.Array | None = None,
    reset_mask: jax.Array | None = None,
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
) -> tuple[jax.Array, jax.Array]:
  """Independent per-token sequential SSM recurrence oracle in float64 NumPy."""
  x_np = np.asarray(x_c.astype(jnp.float32), dtype=np.float64)
  a_np = np.asarray(a_c.astype(jnp.float32), dtype=np.float64)
  b_np = np.asarray(b_c.astype(jnp.float32), dtype=np.float64)
  c_np = np.asarray(c_c.astype(jnp.float32), dtype=np.float64)

  C, H, L, P = x_np.shape
  _, G, _, N = b_np.shape
  Q = L // mimo_rank
  if G < H:
    repeats = H // G
    b_np = np.repeat(b_np, repeats, axis=1)
    c_np = np.repeat(c_np, repeats, axis=1)

  init_np = (
      None
      if initial_states is None
      else np.asarray(initial_states.astype(jnp.float32), dtype=np.float64)
  )

  if reset_mask is None:
    rst_np = np.zeros((C,), dtype=bool)
    rst_np[0] = True
  else:
    rst_np = np.asarray(reset_mask, dtype=bool)

  y_out = np.zeros_like(x_np, dtype=np.float64)
  states_outgoing = np.zeros((C, H, P, N), dtype=np.float64)
  h = np.zeros((H, P, N), dtype=np.float64)
  _, _, a_len = a_np.shape
  seq_idx = 0

  for c_idx in range(C):
    if rst_np[c_idx]:
      if init_np is None:
        h = np.zeros((H, P, N), dtype=np.float64)
      else:
        S_init, _, _, _ = init_np.shape
        safe_seq = min(seq_idx, S_init - 1)
        h = init_np[safe_seq].copy()
      seq_idx += 1

    for tok_idx in range(Q):
      r_start = tok_idx * mimo_rank
      r_end = r_start + mimo_rank
      if a_len == L:
        a_tok = np.sum(a_np[c_idx, :, r_start:r_end], axis=-1)
      else:
        a_tok = a_np[c_idx, :, tok_idx]
      decay = np.exp(a_tok)[:, None, None]
      h = decay * h

      if strict_causal:
        for r_idx in range(r_start, r_end):
          y_out[c_idx, :, r_idx, :] = np.einsum(
              "hpn,hn->hp", h, c_np[c_idx, :, r_idx, :]
          )

      for r_idx in range(r_start, r_end):
        h = h + x_np[c_idx, :, r_idx, :, None] * b_np[c_idx, :, r_idx, None, :]

      if not strict_causal:
        for r_idx in range(r_start, r_end):
          y_out[c_idx, :, r_idx, :] = np.einsum(
              "hpn,hn->hp", h, c_np[c_idx, :, r_idx, :]
          )

    states_outgoing[c_idx] = h.copy()

  safe_idx = np.maximum(np.asarray(last_chunk_indices, dtype=np.int64), 0)
  y_jax = jnp.asarray(y_out, dtype=jnp.float32).astype(x_c.dtype)
  final_states_jax = jnp.asarray(states_outgoing[safe_idx], dtype=jnp.float32)
  return y_jax, final_states_jax


def _make_ssd_inputs(
    batch: int,
    seqlen: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    *,
    n_groups: int | None = None,
    dtype: jax.typing.DTypeLike = jnp.float32,
):
  groups = n_heads if n_groups is None else n_groups
  n_chunks_per_seq = seqlen // block_len
  n_chunks = batch * n_chunks_per_seq

  rng = jax.random.key(42 + n_heads + groups + block_len)
  k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

  x_c = jax.random.normal(
      k1, (n_chunks, n_heads, block_len, d_head), dtype=dtype
  )
  a_c = (
      -jnp.exp(
          jax.random.normal(
              k2, (n_chunks, n_heads, block_len), dtype=jnp.float32
          )
      )
      * 0.25
  )
  b_c = jax.random.normal(
      k3, (n_chunks, groups, block_len, d_state), dtype=dtype
  )
  c_c = jax.random.normal(
      k4, (n_chunks, groups, block_len, d_state), dtype=dtype
  )
  initial_states = jax.random.normal(
      k5, (batch, n_heads, d_head, d_state), dtype=dtype
  )

  reset_mask = np.zeros((n_chunks,), dtype=bool)
  for b_idx in range(batch):
    reset_mask[b_idx * n_chunks_per_seq] = True

  last_chunk_indices = jnp.array(
      [
          b_idx * n_chunks_per_seq + n_chunks_per_seq - 1
          for b_idx in range(batch)
      ],
      dtype=jnp.int32,
  )
  return (
      x_c,
      a_c,
      b_c,
      c_c,
      last_chunk_indices,
      initial_states,
      jnp.array(reset_mask),
  )


class SsdScanTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "gpu":
      self.skipTest("Pallas Mosaic TPU kernels are not supported on GPU.")

  def test_largest_divisor_and_vmem_head_block_selection(self):
    self.assertEqual(ssd_scan_lib._largest_divisor(1, 8), 1)
    self.assertEqual(ssd_scan_lib._largest_divisor(7, 5), 1)
    self.assertEqual(ssd_scan_lib._largest_divisor(3, 8), 3)
    self.assertEqual(ssd_scan_lib._largest_divisor(6, 8), 6)
    self.assertEqual(ssd_scan_lib._largest_divisor(8, 8), 8)
    self.assertEqual(ssd_scan_lib._largest_divisor(12, 8), 6)
    self.assertEqual(ssd_scan_lib._largest_divisor(16, 8), 8)
    self.assertEqual(ssd_scan_lib._largest_divisor(24, 8), 8)
    with self.assertRaisesRegex(ValueError, "Expected n > 0"):
      ssd_scan_lib._largest_divisor(0, 8)

    self.assertEqual(
        ssd_scan_lib._aligned_vmem_tile_bytes(
            lead_dim=2,
            rows=8,
            cols=128,
            elem_bytes=4,
            num_sublanes=8,
            num_lanes=128,
        ),
        2 * 8 * 128 * 4,
    )
    self.assertEqual(
        ssd_scan_lib._aligned_vmem_tile_bytes(
            lead_dim=3,
            rows=9,
            cols=129,
            elem_bytes=4,
            num_sublanes=8,
            num_lanes=128,
        ),
        3 * 16 * 256 * 4,
    )

    class _FakeTpuDevice:

      def __init__(self, device_kind: str):
        self.device_kind = device_kind
        self.platform = "tpu"
        self.num_cores = 1

    mesh_16m = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=_FakeTpuDevice("TPU v4")
    )
    mesh_128m = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=_FakeTpuDevice("TPU v6 lite")
    )

    with jax.sharding.use_abstract_mesh(mesh_16m):
      hb_16m = ssd_scan_lib._select_head_block_size(
          64, 64, 512, 128, 256, include_decay_tile=True, interpret=False
      )
      hb_grouped = ssd_scan_lib._select_head_block_size(
          12, 2, 64, 64, 128, include_decay_tile=True, interpret=False
      )
    with jax.sharding.use_abstract_mesh(mesh_128m):
      hb_128m = ssd_scan_lib._select_head_block_size(
          64, 64, 512, 128, 256, include_decay_tile=True, interpret=False
      )

    self.assertEqual(hb_16m, 4)
    self.assertEqual(hb_grouped, 6)
    self.assertGreater(hb_128m, hb_16m)

    with jax.sharding.use_abstract_mesh(mesh_128m):
      jitted = jax.jit(
          functools.partial(
              ssd_scan.ssd_minimal_discrete_hybrid, interpret=False
          )
      )
      exported = export.export(jitted, platforms=("tpu",))(
          jax.ShapeDtypeStruct((3, 12, 64, 64), jnp.bfloat16),
          jax.ShapeDtypeStruct((3, 12, 64), jnp.float32),
          jax.ShapeDtypeStruct((3, 2, 64, 128), jnp.bfloat16),
          jax.ShapeDtypeStruct((3, 2, 64, 128), jnp.bfloat16),
          jax.ShapeDtypeStruct((2,), jnp.int32),
          jax.ShapeDtypeStruct((2, 12, 64, 128), jnp.float32),
          jax.ShapeDtypeStruct((3,), jnp.bool_),
      )
      self.assertIn("ssd_chunk_state_pallas", exported.mlir_module())
      self.assertIn("ssd_chunk_scan_pallas", exported.mlir_module())

  @parameterized.product(
      strict_causal=[False, True],
      mimo_rank=[1, 4],
  )
  def test_segsum_jax_against_naive_loop(
      self, strict_causal: bool, mimo_rank: int
  ):
    x = jnp.array(
        [
            [-0.5, -0.25, -0.75, -0.125, -0.2, -0.4, -0.1, -0.3],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        ],
        dtype=jnp.float32,
    )
    actual = np.asarray(
        ssd_scan_lib._segsum_jax(
            x, strict_causal=strict_causal, mimo_rank=mimo_rank
        )
    )
    x_np = np.asarray(x)
    batch, total_len = x_np.shape
    n_tok = total_len // mimo_rank
    x_tok = x_np.reshape(batch, n_tok, mimo_rank).sum(axis=-1)

    expected = np.full((batch, total_len, total_len), -np.inf, dtype=np.float32)
    for b_idx in range(batch):
      for i in range(total_len):
        tok_i = i // mimo_rank
        for j in range(total_len):
          tok_j = j // mimo_rank
          if tok_i == tok_j and not strict_causal:
            expected[b_idx, i, j] = 0.0
          elif tok_i > tok_j:
            expected[b_idx, i, j] = np.sum(x_tok[b_idx, tok_j + 1 : tok_i + 1])

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(8, 2)
  def test_ssd_scan_jax_against_float64_oracle(self, n_groups: int):
    x_c, a_c, b_c, c_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            2, 64, 16, 8, 16, 8, n_groups=n_groups, dtype=jnp.float32
        )
    )
    y_f32, state_f32 = ssd_scan.ssd_scan_jax(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=True,
        mimo_rank=4,
    )
    with jax.enable_x64():
      y_f64, state_f64 = ssd_scan.ssd_scan_jax(
          x_c.astype(jnp.float64),
          a_c.astype(jnp.float64),
          b_c.astype(jnp.float64),
          c_c.astype(jnp.float64),
          last_chunk_indices,
          initial_states.astype(jnp.float64),
          reset_mask,
          strict_causal=True,
          mimo_rank=4,
      )
    y_seq, state_seq = _ssd_sequential_reference(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=True,
        mimo_rank=4,
    )
    np.testing.assert_allclose(y_f32, y_f64, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_f32, state_f64, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_f32, y_seq, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_f32, state_seq, rtol=1e-5, atol=1e-5)

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
      n_groups=[6, 2],
  )
  def test_chunk_state_and_chunk_scan_pallas_subops(self, dtype, n_groups: int):
    x_c, a_c, b_c, c_c, _, _, _ = _make_ssd_inputs(
        2, 64, 16, 6, 16, 8, n_groups=n_groups, dtype=dtype
    )
    a_cumsum = jnp.cumsum(a_c, axis=-1)
    decay_states = jnp.exp(a_cumsum[..., -1:] - a_cumsum)[:, :, None, :]

    states_actual = ssd_scan.chunk_state_pallas(b_c, decay_states, x_c)
    self.assertEqual(states_actual.dtype, jnp.float32)
    b_rep = jnp.repeat(b_c, 6 // n_groups, axis=1)
    c_rep = jnp.repeat(c_c, 6 // n_groups, axis=1)
    states_expected = jnp.einsum(
        "chln,chlp,chl->chpn",
        b_rep.astype(jnp.float32),
        x_c.astype(jnp.float32),
        decay_states[:, :, 0, :].astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    tol_state = 1e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(
        states_actual, states_expected, rtol=tol_state, atol=tol_state
    )

    y_off_default = ssd_scan.chunk_scan_pallas(c_c, states_actual)
    self.assertEqual(y_off_default.dtype, jnp.dtype(dtype))

    y_off_f32 = ssd_scan.chunk_scan_pallas(
        c_c, states_actual, out_dtype=jnp.float32
    )
    self.assertEqual(y_off_f32.dtype, jnp.float32)
    y_off_expected = jnp.einsum(
        "chln,chpn->chlp",
        c_rep.astype(jnp.float32),
        states_actual,
        precision=jax.lax.Precision.HIGHEST,
    )
    np.testing.assert_allclose(
        y_off_f32, y_off_expected, rtol=tol_state, atol=tol_state
    )

  @parameterized.product(
      strict_causal=[False, True],
      shape=[
          (2, 16, 4, 4, 8, 8),
          (2, 128, 32, 4, 16, 8),
          (2, 64, 16, 6, 16, 8),
          (2, 64, 16, 12, 16, 8),
          (2, 32, 16, 1, 16, 8),
      ],
  )
  def test_ssd_minimal_discrete_hybrid_siso(
      self, strict_causal: bool, shape: tuple[int, int, int, int, int, int]
  ):
    batch, seqlen, block_len, n_heads, d_head, d_state = shape
    x_c, a_c, b_c, c_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            batch,
            seqlen,
            block_len,
            n_heads,
            d_head,
            d_state,
            dtype=jnp.float32,
        )
    )
    y_ref, final_state_ref = ssd_scan.ssd_scan_jax(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=strict_causal,
    )
    y_actual, final_state_actual = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=strict_causal,
        interpret=True,
    )
    self.assertEqual(final_state_actual.dtype, jnp.float32)
    np.testing.assert_allclose(y_actual, y_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        final_state_actual, final_state_ref, rtol=1e-4, atol=1e-4
    )

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
      heads_and_groups=[(8, 2), (12, 2), (6, 1)],
  )
  def test_ssd_scan_grouped_bc_projections(
      self, dtype, heads_and_groups: tuple[int, int]
  ):
    n_heads, n_groups = heads_and_groups
    (
        x_c,
        a_c,
        b_grouped,
        c_grouped,
        last_chunk_indices,
        initial_states,
        reset_mask,
    ) = _make_ssd_inputs(
        2, 64, 16, n_heads, 16, 8, n_groups=n_groups, dtype=dtype
    )
    repeats = n_heads // n_groups
    b_rep = jnp.repeat(b_grouped, repeats, axis=1)
    c_rep = jnp.repeat(c_grouped, repeats, axis=1)

    y_grouped, state_grouped = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_grouped,
        c_grouped,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )
    y_rep, state_rep = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_rep,
        c_rep,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )
    y_ref, state_ref = ssd_scan.ssd_scan_jax(
        x_c,
        a_c,
        b_grouped,
        c_grouped,
        last_chunk_indices,
        initial_states,
        reset_mask,
    )

    np.testing.assert_allclose(state_grouped, state_rep, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        y_grouped.astype(jnp.float32),
        y_rep.astype(jnp.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(state_grouped, state_ref, rtol=1e-4, atol=1e-4)
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(
        y_grouped.astype(jnp.float32),
        y_ref.astype(jnp.float32),
        rtol=tol,
        atol=tol,
    )

  @parameterized.product(
      strict_causal=[True, False],
      per_token_a=[False, True],
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_ssd_minimal_discrete_hybrid_mimo(
      self, strict_causal: bool, per_token_a: bool, dtype
  ):
    batch, n_chunks_per_seq, chunk_tokens, mimo_rank = 2, 2, 8, 4
    n_chunks = batch * n_chunks_per_seq
    block_len = chunk_tokens * mimo_rank
    n_heads, n_groups, d_head, d_state = 6, 2, 16, 8

    rng = jax.random.key(99 + int(strict_causal) + 2 * int(per_token_a))
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    x_c = jax.random.normal(
        k1, (n_chunks, n_heads, block_len, d_head), dtype=dtype
    )
    a_tok = (
        -jax.nn.softplus(
            jax.random.normal(
                k2, (n_chunks, n_heads, chunk_tokens), dtype=jnp.float32
            )
        )
        * 0.1
    )
    if per_token_a:
      a_c = a_tok
    else:
      a_zeros = jnp.zeros(
          (n_chunks, n_heads, chunk_tokens, mimo_rank - 1), dtype=jnp.float32
      )
      a_c = jnp.concatenate([a_tok[..., None], a_zeros], axis=-1).reshape(
          n_chunks, n_heads, block_len
      )
    b_c = jax.random.normal(
        k3, (n_chunks, n_groups, block_len, d_state), dtype=dtype
    )
    c_c = jax.random.normal(
        k4, (n_chunks, n_groups, block_len, d_state), dtype=dtype
    )
    initial_states = jax.random.normal(
        k5, (batch, n_heads, d_head, d_state), dtype=dtype
    )
    reset_mask = np.zeros((n_chunks,), dtype=bool)
    for b_idx in range(batch):
      reset_mask[b_idx * n_chunks_per_seq] = True
    reset_mask_jax = jnp.array(reset_mask)
    last_chunk_indices = jnp.array(
        [
            b_idx * n_chunks_per_seq + n_chunks_per_seq - 1
            for b_idx in range(batch)
        ],
        dtype=jnp.int32,
    )

    y_ref, final_state_ref = ssd_scan.ssd_scan_jax(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask_jax,
        strict_causal=strict_causal,
        mimo_rank=mimo_rank,
    )
    y_actual, final_state_actual = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask_jax,
        strict_causal=strict_causal,
        mimo_rank=mimo_rank,
        interpret=True,
    )

    self.assertEqual(y_actual.dtype, jnp.dtype(dtype))
    self.assertEqual(final_state_actual.dtype, jnp.float32)
    np.testing.assert_allclose(
        final_state_actual, final_state_ref, rtol=1e-4, atol=1e-4
    )
    tol_y = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(
        y_actual.astype(jnp.float32),
        y_ref.astype(jnp.float32),
        rtol=tol_y,
        atol=tol_y,
    )

  def test_ragged_tail_padding_sentinels_and_checkpoints(self):
    block_len, n_heads, n_groups, d_head, d_state = 8, 6, 2, 8, 8
    seq_token_lens = [5, 19, 16]
    seq_chunk_counts = [
        (seq_len + block_len - 1) // block_len for seq_len in seq_token_lens
    ]
    n_chunks = sum(seq_chunk_counts)

    rng = jax.random.key(123)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    x_raw = np.array(
        jax.random.normal(
            k1, (n_chunks, n_heads, block_len, d_head), dtype=jnp.float32
        ),
        copy=True,
    )
    a_raw = np.array(
        -jnp.exp(
            jax.random.normal(
                k2, (n_chunks, n_heads, block_len), dtype=jnp.float32
            )
        ),
        copy=True,
    )
    b_raw = np.array(
        jax.random.normal(
            k3, (n_chunks, n_groups, block_len, d_state), dtype=jnp.float32
        ),
        copy=True,
    )
    c_raw = np.array(
        jax.random.normal(
            k4, (n_chunks, n_groups, block_len, d_state), dtype=jnp.float32
        ),
        copy=True,
    )
    init_states = jax.random.normal(
        k5, (len(seq_token_lens), n_heads, d_head, d_state), dtype=jnp.float32
    )

    reset_mask = np.zeros((n_chunks,), dtype=bool)
    seq_last_chunks: list[int] = []
    chunk_cursor = 0
    for seq_len, n_seq_chunks in zip(seq_token_lens, seq_chunk_counts):
      reset_mask[chunk_cursor] = True
      last_c = chunk_cursor + n_seq_chunks - 1
      seq_last_chunks.append(last_c)
      rem = seq_len % block_len
      if rem != 0:
        x_raw[last_c, :, rem:, :] = 0.0
        a_raw[last_c, :, rem:] = 0.0
      chunk_cursor += n_seq_chunks

    x_c = jnp.asarray(x_raw)
    a_c = jnp.asarray(a_raw)
    b_c = jnp.asarray(b_raw)
    c_c = jnp.asarray(c_raw)
    reset_mask_jax = jnp.asarray(reset_mask)

    checkpoint_and_final_indices = jnp.array(
        [0, 1, 2, 3, 5, -1, -1], dtype=jnp.int32
    )

    y_init, states_init = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_c,
        c_c,
        checkpoint_and_final_indices,
        initial_states=init_states,
        reset_mask=reset_mask_jax,
        interpret=True,
    )
    y_init_ref, states_init_ref = _ssd_sequential_reference(
        x_c,
        a_c,
        b_c,
        c_c,
        checkpoint_and_final_indices,
        initial_states=init_states,
        reset_mask=reset_mask_jax,
    )
    np.testing.assert_allclose(y_init, y_init_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        states_init, states_init_ref, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        states_init[-1], states_init[0], rtol=1e-6, atol=1e-6
    )

    init_states_padded = jnp.pad(
        init_states,
        ((0, n_chunks - len(seq_token_lens)), (0, 0), (0, 0), (0, 0)),
        constant_values=99.0,
    )
    y_pad_init, states_pad_init = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_c,
        c_c,
        checkpoint_and_final_indices,
        initial_states=init_states_padded,
        reset_mask=reset_mask_jax,
        interpret=True,
    )
    np.testing.assert_allclose(y_pad_init, y_init_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        states_pad_init, states_init_ref, rtol=1e-4, atol=1e-4
    )

    b_rep = np.repeat(b_raw, n_heads // n_groups, axis=1)
    h_seq0 = np.asarray(init_states[0], dtype=np.float32).copy()
    for t_idx in range(seq_token_lens[0]):
      h_seq0 = (
          np.exp(a_raw[0, :, t_idx])[:, None, None] * h_seq0
          + x_raw[0, :, t_idx, :, None] * b_rep[0, :, t_idx, None, :]
      )
    np.testing.assert_allclose(states_init[0], h_seq0, rtol=1e-4, atol=1e-4)

    last_only = jnp.asarray(seq_last_chunks, dtype=jnp.int32)
    y_zero, states_zero = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_c,
        c_c,
        last_only,
        initial_states=None,
        reset_mask=reset_mask_jax,
        interpret=True,
    )
    y_zero_ref, states_zero_ref = _ssd_sequential_reference(
        x_c,
        a_c,
        b_c,
        c_c,
        last_only,
        initial_states=None,
        reset_mask=reset_mask_jax,
    )
    np.testing.assert_allclose(y_zero, y_zero_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        states_zero, states_zero_ref, rtol=1e-4, atol=1e-4
    )

  def test_bfloat16_carry_precision_and_ragged_reset_mask(self):
    n_chunks, n_heads, block_len, d_head, d_state = 6, 8, 16, 32, 64
    rng = jax.random.key(777)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    x_c = jax.random.normal(
        k1, (n_chunks, n_heads, block_len, d_head), dtype=jnp.bfloat16
    )
    a_c = (
        -jnp.exp(
            jax.random.normal(
                k2, (n_chunks, n_heads, block_len), dtype=jnp.float32
            )
        )
        * 0.2
    )
    b_c = jax.random.normal(
        k3, (n_chunks, n_heads, block_len, d_state), dtype=jnp.bfloat16
    )
    c_c = jax.random.normal(
        k4, (n_chunks, n_heads, block_len, d_state), dtype=jnp.bfloat16
    )
    initial_states = jax.random.normal(
        k5, (3, n_heads, d_head, d_state), dtype=jnp.bfloat16
    )
    reset_mask = jnp.array([True, True, False, False, True, False], dtype=bool)
    last_chunk_indices = jnp.array([0, 3, 5, -1], dtype=jnp.int32)

    y_ref, final_state_ref = ssd_scan.ssd_scan_jax(
        x_c, a_c, b_c, c_c, last_chunk_indices, initial_states, reset_mask
    )
    y_actual, final_state_actual = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        interpret=True,
    )
    self.assertEqual(y_actual.dtype, jnp.bfloat16)
    self.assertEqual(final_state_actual.dtype, jnp.float32)
    np.testing.assert_allclose(
        final_state_actual, final_state_ref, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        y_actual.astype(jnp.float32),
        y_ref.astype(jnp.float32),
        rtol=2e-2,
        atol=2e-2,
    )

  def test_default_none_arguments_and_jit_with_default_interpret(self):
    x_c, a_c, b_c, c_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(2, 64, 16, 4, 16, 8, dtype=jnp.bfloat16)
    )
    y_ref_zero, state_ref_zero = ssd_scan.ssd_scan_jax(
        x_c, a_c, b_c, c_c, last_chunk_indices
    )
    y_eager, state_eager = ssd_scan_lib.ssd_scan(
        x_c, a_c, b_c, c_c, last_chunk_indices
    )
    np.testing.assert_allclose(
        y_eager.astype(jnp.float32),
        y_ref_zero.astype(jnp.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        state_eager, state_ref_zero, rtol=1e-3, atol=1e-3
    )

    jitted_fn = jax.jit(
        functools.partial(
            ssd_scan.ssd_minimal_discrete_hybrid,
            strict_causal=True,
            mimo_rank=4,
        )
    )
    y_jit, state_jit = jitted_fn(
        x_c, a_c, b_c, c_c, last_chunk_indices, initial_states, reset_mask
    )
    y_ref_mimo, state_ref_mimo = ssd_scan.ssd_scan_jax(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=True,
        mimo_rank=4,
    )
    np.testing.assert_allclose(
        y_jit.astype(jnp.float32),
        y_ref_mimo.astype(jnp.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(state_jit, state_ref_mimo, rtol=1e-3, atol=1e-3)

  def test_shape_validation_raises_value_error(self):
    n_chunks, n_heads, block_len, d_head, d_state = 2, 4, 8, 16, 8
    x_c = jnp.zeros((n_chunks, n_heads, block_len, d_head), dtype=jnp.float32)
    a_c = jnp.zeros((n_chunks, n_heads, block_len), dtype=jnp.float32)
    b_c = jnp.zeros((n_chunks, n_heads, block_len, d_state), dtype=jnp.float32)
    c_c = jnp.zeros((n_chunks, n_heads, block_len, d_state), dtype=jnp.float32)
    decay_states = jnp.zeros(
        (n_chunks, n_heads, 1, block_len), dtype=jnp.float32
    )
    states = jnp.zeros((n_chunks, n_heads, d_head, d_state), dtype=jnp.float32)
    last_chunk_indices = jnp.array([0, 1], dtype=jnp.int32)

    # 1. Verify `@jaxtyping.jaxtyped` catches rank/dimension mismatches at API
    # boundary (`jaxtyping.TypeCheckError` subclasses `TypeError`).
    with self.assertRaises(TypeError):
      ssd_scan.chunk_state_pallas(b_c, decay_states, x_c[0], interpret=True)
    with self.assertRaises(TypeError):
      ssd_scan.ssd_minimal_discrete_hybrid(
          x_c, a_c, b_c[:, :2], c_c, last_chunk_indices, interpret=True
      )

    # 2. Verify explicit in-body shape checks when jaxtyping is disabled.
    with tokamax_jaxtyping.disable_jaxtyping():
      with self.assertRaisesRegex(ValueError, "Expected mimo_rank >= 1"):
        ssd_scan_lib._segsum_jax(a_c, mimo_rank=0)
      with self.assertRaisesRegex(ValueError, "dividing L="):
        ssd_scan_lib._segsum_jax(a_c, mimo_rank=3)

      with self.assertRaisesRegex(ValueError, "Expected positive 4D X_c"):
        ssd_scan.chunk_state_pallas(b_c, decay_states, x_c[0], interpret=True)
      with self.assertRaisesRegex(ValueError, "Expected positive 4D X_c"):
        ssd_scan.chunk_state_pallas(
            b_c[:0], decay_states[:0], x_c[:0], interpret=True
        )
      with self.assertRaisesRegex(ValueError, "Expected positive 4D B_c"):
        ssd_scan.chunk_state_pallas(
            b_c[:, :3], decay_states, x_c, interpret=True
        )
      with self.assertRaisesRegex(ValueError, "Expected decay_states shape"):
        ssd_scan.chunk_state_pallas(
            b_c, jnp.swapaxes(decay_states, -1, -2), x_c, interpret=True
        )

      with self.assertRaisesRegex(ValueError, "Expected positive 4D C_c"):
        ssd_scan.chunk_scan_pallas(c_c[0], states, interpret=True)
      with self.assertRaisesRegex(ValueError, "Expected positive 4D C_c"):
        ssd_scan.chunk_scan_pallas(c_c[:0], states[:0], interpret=True)
      with self.assertRaisesRegex(ValueError, "Expected positive 4D states"):
        ssd_scan.chunk_scan_pallas(c_c[:, :3], states, interpret=True)

      with self.assertRaisesRegex(ValueError, "Expected positive 4D X_c"):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c[:0],
            a_c[:0],
            b_c[:0],
            c_c[:0],
            last_chunk_indices,
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "dividing block_len="):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c, a_c, b_c, c_c, last_chunk_indices, mimo_rank=3, interpret=True
        )
      with self.assertRaisesRegex(ValueError, "Expected A_c shape"):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c, a_c[:, :2], b_c, c_c, last_chunk_indices, interpret=True
        )
      with self.assertRaisesRegex(ValueError, "Expected A_c shape in"):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c,
            a_c[..., :3],
            b_c,
            c_c,
            last_chunk_indices,
            mimo_rank=4,
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected positive 4D B_c"):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c, a_c, b_c[:, :3], c_c[:, :3], last_chunk_indices, interpret=True
        )
      with self.assertRaisesRegex(ValueError, "Expected C_c shape"):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c, a_c, b_c, c_c[:, :2], last_chunk_indices, interpret=True
        )
      with self.assertRaisesRegex(
          ValueError, "Expected non-empty 1D last_chunk_indices"
      ):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c, a_c, b_c, c_c, last_chunk_indices[:, None], interpret=True
        )
      with self.assertRaisesRegex(
          ValueError, "Expected non-empty 1D last_chunk_indices"
      ):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c,
            a_c,
            b_c,
            c_c,
            jnp.zeros((0,), dtype=jnp.int32),
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected initial_states shape"):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c,
            a_c,
            b_c,
            c_c,
            last_chunk_indices,
            initial_states=states[:, :2],
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected reset_mask shape"):
        ssd_scan.ssd_minimal_discrete_hybrid(
            x_c,
            a_c,
            b_c,
            c_c,
            last_chunk_indices,
            reset_mask=jnp.ones((n_chunks + 1,), dtype=bool),
            interpret=True,
        )

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
      strict_causal=[False, True],
      mimo_rank=[1, 4],
      heads_and_groups=[(6, 6), (12, 2)],
  )
  def test_ssd_minimal_discrete_hybrid_tpu(
      self,
      dtype,
      strict_causal: bool,
      mimo_rank: int,
      heads_and_groups: tuple[int, int],
  ):
    if jax.default_backend() != "tpu":
      self.skipTest("TPU hardware execution requires TPU backend.")

    n_heads, n_groups = heads_and_groups
    x_c, a_c, b_c, c_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(
            2, 128, 64, n_heads, 64, 128, n_groups=n_groups, dtype=dtype
        )
    )
    if mimo_rank > 1:
      a_c = a_c.reshape(4, n_heads, 64 // mimo_rank, mimo_rank).sum(axis=-1)

    y_ref, final_state_ref = ssd_scan.ssd_scan_jax(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=strict_causal,
        mimo_rank=mimo_rank,
    )
    y_actual, final_state_actual = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_c,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=strict_causal,
        mimo_rank=mimo_rank,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(y_actual))))
    self.assertFalse(bool(jnp.any(jnp.isnan(final_state_actual))))
    self.assertEqual(y_actual.dtype, jnp.dtype(dtype))
    self.assertEqual(final_state_actual.dtype, jnp.float32)
    np.testing.assert_allclose(
        final_state_actual, final_state_ref, rtol=1e-3, atol=1e-3
    )
    tol_y = 2e-2 if dtype == jnp.bfloat16 else 1e-3
    np.testing.assert_allclose(
        y_actual.astype(jnp.float32),
        y_ref.astype(jnp.float32),
        rtol=tol_y,
        atol=tol_y,
    )

  def test_uninitialized_vmem_nan_robustness_tpu(self):
    if jax.default_backend() != "tpu":
      self.skipTest("TPU hardware execution requires TPU backend.")

    mosaic_tpu.poison_tpu_memory()
    x_c, a_c, b_c, c_c, last_chunk_indices, initial_states, reset_mask = (
        _make_ssd_inputs(2, 128, 64, 6, 64, 128, dtype=jnp.bfloat16)
    )
    a_tok = a_c.reshape(4, 6, 16, 4).sum(axis=-1)
    y_ref, state_ref = ssd_scan.ssd_scan_jax(
        x_c,
        a_tok,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=True,
        mimo_rank=4,
    )
    y_actual, state_actual = ssd_scan.ssd_minimal_discrete_hybrid(
        x_c,
        a_tok,
        b_c,
        c_c,
        last_chunk_indices,
        initial_states,
        reset_mask,
        strict_causal=True,
        mimo_rank=4,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(y_actual))))
    self.assertFalse(bool(jnp.any(jnp.isnan(state_actual))))
    np.testing.assert_allclose(state_actual, state_ref, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        y_actual.astype(jnp.float32),
        y_ref.astype(jnp.float32),
        rtol=2e-2,
        atol=2e-2,
    )


if __name__ == "__main__":
  absltest.main()
