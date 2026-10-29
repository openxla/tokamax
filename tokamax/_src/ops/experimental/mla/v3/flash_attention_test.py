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
"""Unit tests for MLA Flash Attention pure math engine."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mla import reference
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import flash_attention


def _ref_mla_attention(
    q_nope: jax.Array,  # [n_q, d_nope]
    q_pe: jax.Array,  # [n_q, d_pe]
    k_nope: jax.Array,  # [d_nope, kv_len]
    k_pe: jax.Array,  # [d_pe, kv_len]
    sm_scale: float,
    *,
    num_q_heads: int = 4,
    causal: bool = False,
    scale_q: float | None = None,
    scale_k: float | None = None,
    scale_v: float | None = None,
) -> jax.Array:
  """Delegates single-sequence MLA attention to `reference.mla_attention`."""
  n_q, d_nope = q_nope.shape
  d_pe = q_pe.shape[-1]
  n_tokens = n_q // num_q_heads
  kv_len = k_nope.shape[-1]

  ql_nope_3d = q_nope.reshape(n_tokens, num_q_heads, d_nope).astype(jnp.float32)
  q_pe_3d = q_pe.reshape(n_tokens, num_q_heads, d_pe).astype(jnp.float32)

  r_dim_padded = reference.align_to(d_pe, 128)
  k_pe_padded = jnp.pad(k_pe.T, ((0, 0), (0, r_dim_padded - d_pe)))
  cache_kv = jnp.concatenate([k_nope.T, k_pe_padded], axis=-1)[
      None, :, None, :
  ].astype(jnp.float32)

  new_kv_c = k_nope.T[-n_tokens:].astype(jnp.float32)
  new_k_pe = k_pe.T[-n_tokens:].astype(jnp.float32)

  out, _ = reference.mla_attention(
      ql_nope_3d,
      q_pe_3d,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens=jnp.array([kv_len], dtype=jnp.int32),
      page_indices=jnp.array([0], dtype=jnp.int32),
      cu_q_lens=jnp.array([0, n_tokens if causal else 0], dtype=jnp.int32),
      distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
      sm_scale=sm_scale,
      q_scale=scale_q,
      k_scale=scale_k,
      v_scale=scale_v,
      s_dtype=jnp.float32,
  )
  return out.reshape(n_q, d_nope)


class FlashAttentionMathTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()
    # Pin matmul precision so float32 online-softmax tests can use tight
    # tolerances (1e-4) against the single-pass JAX reference on TPU.
    self.enterContext(jax.default_matmul_precision("highest"))

    self.num_q_heads = 4
    self.lkv_dim = 512
    self.r_dim = 64
    self.sm_scale = 1.0 / np.sqrt(self.lkv_dim + self.r_dim)

    self.model_cfg = configs.MlaModelConfigs(
        num_q_heads=self.num_q_heads,
        lkv_dim=self.lkv_dim,
        r_dim=self.r_dim,
        sm_scale=self.sm_scale,
        mask_value=-1e9,
    )

  def _make_cfg(
      self,
      *,
      batch_size: int,
      bq_sz: int,
      q_split: int = 1,
      bkv_sz: int = 128,
      mode: configs.MlaCase = configs.MlaCase.DECODE,
      dtype: jax.typing.DTypeLike = jnp.float32,
      scale_q: float | None = None,
      scale_k: float | None = None,
      scale_v: float | None = None,
  ) -> configs.MlaConfigs:
    bq_c_sz = bq_sz // q_split
    return configs.MlaConfigs(
        block=configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=2,
        ),
        model=self.model_cfg,
        serve=configs.ServingConfigs(
            num_seqs=batch_size,
            page_size=bkv_sz,
            total_q_tokens=batch_size * bq_sz,
            num_page_indices=2,
            dtype_q=dtype,
            dtype_kv=dtype,
            # Output stays f32 so the exact-divide path and tight tolerances
            # apply regardless of the input dtype.
            dtype_out=jnp.float32,
            scale_q=scale_q,
            scale_k=scale_k,
            scale_v=scale_v,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        ),
        mode=mode,
    )

  def _run_kernel_steps(
      self,
      steps: list[tuple[Any, ...]],
      cfgs: configs.MlaConfigs,
  ) -> jax.Array:
    """Runs one or more pipeline steps of `chunked_flash_attention`.

    Initializes VMEM scratch refs (`m`, `l`, `acc`) once at the start of the
    kernel and threads them across `steps`, returning the final `o_vref` of
    shape `[batch_size, bq_sz, num_q_heads, lkv_dim]`.

    Args:
      steps: Per pipeline step, either `(q, k, is_last_k)` for unmasked
        attention or `(q, k, is_last_k, q_abs_start, kv_abs_start)` to enable
        causal masking, where the last two are `[batch_size]` int32 arrays.
      cfgs: MLA configuration.

    Returns:
      Output tensor `o_vref` after all steps complete.
    """
    n_q = cfgs.bq_sz * self.num_q_heads
    m_init = jnp.full((n_q, 128), -jnp.inf, dtype=jnp.float32)
    l_init = jnp.zeros((n_q, 128), dtype=jnp.float32)
    acc_init = jnp.zeros((n_q, self.lkv_dim), dtype=jnp.float32)
    o_init = jnp.zeros(
        (cfgs.batch_size, cfgs.bq_sz, self.num_q_heads, self.lkv_dim),
        dtype=jnp.float32,
    )

    def _body(refs):
      m_ref, l_ref, acc_ref, o_ref = refs
      for step in steps:
        q_step, k_step, is_last_k_step, *mask_offsets = step
        q_abs_start, kv_abs_start = mask_offsets or (None, None)
        flash_attention.chunked_flash_attention(
            q=q_step,
            k=k_step,
            m_scratch_ref=m_ref,
            l_scratch_ref=l_ref,
            acc_scratch_ref=acc_ref,
            o_vref=o_ref,
            is_last_k=is_last_k_step,
            q_abs_start=q_abs_start,
            kv_abs_start=kv_abs_start,
            cfgs=cfgs,
        )

    _, _, _, o_out = pl.run_state(_body)((m_init, l_init, acc_init, o_init))
    return o_out

  @parameterized.named_parameters(
      ("f32", jnp.float32),
      ("f8_e4m3", jnp.float8_e4m3),
  )
  def test_decode_two_sequences_in_one_step(self, dtype: jax.typing.DTypeLike):
    """Decode path: 2 independent 1-block sequences packed into 1 step.

    Lane 0 runs Seq 0 (`is_last_k[0]=True`), resets the running max to `-inf`,
    and Lane 1 runs Seq 1 (`is_last_k[1]=True`). Both lanes' outputs in `o_vref`
    must match their respective independent references.

    Parameterized over the Q/KV input dtype; the f8 variant checks that the
    QK and PV dots accept narrow operands while accumulating in f32.
    """
    b, bq_sz, s = 2, 1, 128
    cfgs = self._make_cfg(
        batch_size=b,
        bq_sz=bq_sz,
        q_split=1,
        bkv_sz=s,
        mode=configs.MlaCase.DECODE,
        dtype=dtype,
    )
    n_q = bq_sz * self.num_q_heads

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    q_nope = jax.random.normal(k1, (b, n_q, self.lkv_dim), dtype=dtype)
    q_pe = jax.random.normal(k2, (b, n_q, self.r_dim), dtype=dtype)
    k_nope = jax.random.normal(k3, (b, self.lkv_dim, s), dtype=dtype)
    k_pe = jax.random.normal(k4, (b, self.r_dim, s), dtype=dtype)

    q = jnp.concatenate([q_nope, q_pe], axis=-1)
    k = jnp.concatenate([k_nope, k_pe], axis=1)
    is_last_k = jnp.array([True, True])

    o_vref = self._run_kernel_steps([(q, k, is_last_k)], cfgs)

    for lane in range(b):
      expected = _ref_mla_attention(
          q_nope[lane], q_pe[lane], k_nope[lane], k_pe[lane], self.sm_scale
      )
      np.testing.assert_allclose(
          o_vref[lane].reshape(n_q, self.lkv_dim),
          expected,
          rtol=1e-4,
          atol=1e-4,
      )

  def test_decode_single_sequence_across_two_lanes(self):
    """Decode path: 1 sequence with 2 K-blocks packed across lanes 0 and 1.

    Lane 0 processes K-block 0 (`is_last_k[0]=False`) and passes its running
    `(m, l, acc)` in registers to Lane 1, which processes K-block 1
    (`is_last_k[1]=True`) and writes the combined output to `o_vref[1]`.
    """
    b, bq_sz, s = 2, 1, 128
    cfgs = self._make_cfg(
        batch_size=b,
        bq_sz=bq_sz,
        q_split=1,
        bkv_sz=s,
        mode=configs.MlaCase.DECODE,
    )
    n_q = bq_sz * self.num_q_heads

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(7), 4)
    # Same query on both lanes (since both lanes serve the same sequence).
    seq_q_nope = jax.random.normal(k1, (n_q, self.lkv_dim), dtype=jnp.float32)
    seq_q_pe = jax.random.normal(k2, (n_q, self.r_dim), dtype=jnp.float32)
    q_nope = jnp.stack([seq_q_nope, seq_q_nope], axis=0)
    q_pe = jnp.stack([seq_q_pe, seq_q_pe], axis=0)

    # Lane 0 holds K-block 0; Lane 1 holds K-block 1 of the same sequence.
    k_nope = jax.random.normal(k3, (b, self.lkv_dim, s), dtype=jnp.float32)
    k_pe = jax.random.normal(k4, (b, self.r_dim, s), dtype=jnp.float32)

    q = jnp.concatenate([q_nope, q_pe], axis=-1)
    k = jnp.concatenate([k_nope, k_pe], axis=1)
    is_last_k = jnp.array([False, True])

    o_vref = self._run_kernel_steps([(q, k, is_last_k)], cfgs)

    # Reference attends over the concatenated [K_block_0, K_block_1] of length
    # 2*s.
    expected = _ref_mla_attention(
        seq_q_nope,
        seq_q_pe,
        jnp.concatenate([k_nope[0], k_nope[1]], axis=-1),
        jnp.concatenate([k_pe[0], k_pe[1]], axis=-1),
        self.sm_scale,
    )
    np.testing.assert_allclose(
        o_vref[1].reshape(n_q, self.lkv_dim), expected, rtol=1e-4, atol=1e-4
    )

  @parameterized.named_parameters(
      ("prefill_q_split_1", configs.MlaCase.PREFILL, 1),
      ("prefill_q_split_2", configs.MlaCase.PREFILL, 2),
      ("prefill_q_split_4", configs.MlaCase.PREFILL, 4),
      ("mixed_q_split_2", configs.MlaCase.MIXED, 2),
  )
  def test_prefill_multi_step_with_q_split(
      self, mode: configs.MlaCase, q_split: int
  ):
    """Prefill/Mixed path (`batch_size=1`): 8 query tokens over 2 K-block steps.

    Step 0 processes K-block 0 (`is_last_k=[False]`), accumulating into
    `m_scratch_ref`, `l_scratch_ref`, and `acc_scratch_ref` across `q_split`
    query sub-chunks without normalizing. Step 1 processes K-block 1
    (`is_last_k=[True]`), accumulating onto the step-0 state and normalizing
    the full output into `o_vref[0]`.
    """
    b, bq_sz, s = 1, 8, 64
    cfgs = self._make_cfg(
        batch_size=b,
        bq_sz=bq_sz,
        q_split=q_split,
        bkv_sz=s,
        mode=mode,
    )
    n_q = bq_sz * self.num_q_heads

    keys = jax.random.split(jax.random.PRNGKey(101), 6)
    q_nope = jax.random.normal(
        keys[0], (b, n_q, self.lkv_dim), dtype=jnp.float32
    )
    q_pe = jax.random.normal(keys[1], (b, n_q, self.r_dim), dtype=jnp.float32)
    k0_nope = jax.random.normal(
        keys[2], (b, self.lkv_dim, s), dtype=jnp.float32
    )
    k0_pe = jax.random.normal(keys[3], (b, self.r_dim, s), dtype=jnp.float32)
    k1_nope = jax.random.normal(
        keys[4], (b, self.lkv_dim, s), dtype=jnp.float32
    )
    k1_pe = jax.random.normal(keys[5], (b, self.r_dim, s), dtype=jnp.float32)

    q = jnp.concatenate([q_nope, q_pe], axis=-1)
    k0 = jnp.concatenate([k0_nope, k0_pe], axis=1)
    k1 = jnp.concatenate([k1_nope, k1_pe], axis=1)

    o_vref = self._run_kernel_steps(
        [
            (q, k0, jnp.array([False])),  # Step 0: K-block 0 (intermediate)
            (q, k1, jnp.array([True])),  # Step 1: K-block 1 (final)
        ],
        cfgs,
    )

    expected = _ref_mla_attention(
        q_nope[0],
        q_pe[0],
        jnp.concatenate([k0_nope[0], k1_nope[0]], axis=-1),
        jnp.concatenate([k0_pe[0], k1_pe[0]], axis=-1),
        self.sm_scale,
    )
    np.testing.assert_allclose(
        o_vref[0].reshape(n_q, self.lkv_dim), expected, rtol=1e-4, atol=1e-4
    )

  def test_fallback_batched_with_q_split(self):
    """Fallback path: `batch_size > 1` combined with `q_split > 1`."""
    b, bq_sz, q_split, s = 2, 4, 2, 64
    cfgs = self._make_cfg(
        batch_size=b,
        bq_sz=bq_sz,
        q_split=q_split,
        bkv_sz=s,
        mode=configs.MlaCase.PREFILL,
    )
    n_q = bq_sz * self.num_q_heads

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(999), 4)
    q_nope = jax.random.normal(k1, (b, n_q, self.lkv_dim), dtype=jnp.float32)
    q_pe = jax.random.normal(k2, (b, n_q, self.r_dim), dtype=jnp.float32)
    k_nope = jax.random.normal(k3, (b, self.lkv_dim, s), dtype=jnp.float32)
    k_pe = jax.random.normal(k4, (b, self.r_dim, s), dtype=jnp.float32)

    q = jnp.concatenate([q_nope, q_pe], axis=-1)
    k = jnp.concatenate([k_nope, k_pe], axis=1)
    is_last_k = jnp.array([True, True])

    o_vref = self._run_kernel_steps([(q, k, is_last_k)], cfgs)

    for lane in range(b):
      expected = _ref_mla_attention(
          q_nope[lane], q_pe[lane], k_nope[lane], k_pe[lane], self.sm_scale
      )
      np.testing.assert_allclose(
          o_vref[lane].reshape(n_q, self.lkv_dim),
          expected,
          rtol=1e-4,
          atol=1e-4,
      )

  @parameterized.named_parameters(
      ("q_split_1", 1),
      ("q_split_2", 2),
  )
  def test_prefill_causal_mask_with_scales(self, q_split: int):
    """General mask path (`n_tokens > 1`) with `historical_kv_len > 0`.

    One sequence, `q_len = 8` new tokens on top of a 120-token cache, so
    `q_abs_start = 120` and the queries sit at absolute positions 120..127.
    Two K-blocks of 64: block 0 (keys 0..63) is fully visible; block 1 (keys
    64..127) is cut diagonally -- query `i` sees keys up to `120 + i`. With
    `q_split = 2` the diagonal crosses the chunk boundary, so `bq_start` is
    exercised as well. `scale_q/k/v` are set to distinct values so a missing or
    misplaced multiply shows up.
    """
    b, bq_sz, s = 1, 8, 64
    historical_kv_len = 120
    kv_len = historical_kv_len + bq_sz
    scale_q, scale_k, scale_v = 0.5, 1.5, 2.0
    cfgs = self._make_cfg(
        batch_size=b,
        bq_sz=bq_sz,
        q_split=q_split,
        bkv_sz=s,
        mode=configs.MlaCase.PREFILL,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )
    n_q = bq_sz * self.num_q_heads

    keys = jax.random.split(jax.random.PRNGKey(2024), 4)
    q_nope = jax.random.normal(keys[0], (b, n_q, self.lkv_dim), jnp.float32)
    q_pe = jax.random.normal(keys[1], (b, n_q, self.r_dim), jnp.float32)
    k_nope = jax.random.normal(keys[2], (b, self.lkv_dim, kv_len), jnp.float32)
    k_pe = jax.random.normal(keys[3], (b, self.r_dim, kv_len), jnp.float32)

    q = jnp.concatenate([q_nope, q_pe], axis=-1)
    k_full = jnp.concatenate([k_nope, k_pe], axis=1)
    k0, k1 = k_full[..., :s], k_full[..., s:]

    q_abs_start = jnp.array([historical_kv_len], jnp.int32)
    o_vref = self._run_kernel_steps(
        [
            (q, k0, jnp.array([False]), q_abs_start, jnp.array([0], jnp.int32)),
            (q, k1, jnp.array([True]), q_abs_start, jnp.array([s], jnp.int32)),
        ],
        cfgs,
    )

    expected = _ref_mla_attention(
        q_nope[0],
        q_pe[0],
        k_nope[0],
        k_pe[0],
        self.sm_scale,
        num_q_heads=self.num_q_heads,
        causal=True,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )
    np.testing.assert_allclose(
        o_vref[0].reshape(n_q, self.lkv_dim), expected, rtol=1e-4, atol=1e-4
    )

  def test_decode_causal_mask_fast_path_with_scales(self):
    """`n_tokens == 1` mask fast path with two lanes of different visibility.

    Each lane is a separate decode sequence (`q_len = 1`) over one 128-key
    block. Lane 0 has `kv_len = 100` and lane 1 `kv_len = 37`, so
    `q_abs_start = kv_len - 1` and keys past it must be masked out. The K
    block is filled with random data past `kv_len` to make sure masked keys
    really are ignored rather than merely zero.
    """
    b, bq_sz, s = 2, 1, 128
    kv_lens = [100, 37]
    scale_q, scale_k, scale_v = 0.5, 1.5, 2.0
    cfgs = self._make_cfg(
        batch_size=b,
        bq_sz=bq_sz,
        q_split=1,
        bkv_sz=s,
        mode=configs.MlaCase.DECODE,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )
    n_q = bq_sz * self.num_q_heads

    keys = jax.random.split(jax.random.PRNGKey(31), 4)
    q_nope = jax.random.normal(keys[0], (b, n_q, self.lkv_dim), jnp.float32)
    q_pe = jax.random.normal(keys[1], (b, n_q, self.r_dim), jnp.float32)
    k_nope = jax.random.normal(keys[2], (b, self.lkv_dim, s), jnp.float32)
    k_pe = jax.random.normal(keys[3], (b, self.r_dim, s), jnp.float32)

    q = jnp.concatenate([q_nope, q_pe], axis=-1)
    k = jnp.concatenate([k_nope, k_pe], axis=1)

    q_abs_start = jnp.array([kv - 1 for kv in kv_lens], jnp.int32)
    kv_abs_start = jnp.zeros((b,), jnp.int32)
    o_vref = self._run_kernel_steps(
        [(q, k, jnp.array([True, True]), q_abs_start, kv_abs_start)], cfgs
    )

    for lane, kv_len in enumerate(kv_lens):
      expected = _ref_mla_attention(
          q_nope[lane],
          q_pe[lane],
          k_nope[lane][:, :kv_len],
          k_pe[lane][:, :kv_len],
          self.sm_scale,
          scale_q=scale_q,
          scale_k=scale_k,
          scale_v=scale_v,
      )
      np.testing.assert_allclose(
          o_vref[lane].reshape(n_q, self.lkv_dim),
          expected,
          rtol=1e-4,
          atol=1e-4,
      )


if __name__ == "__main__":
  absltest.main()
