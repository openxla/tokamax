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
"""Tests for MultiHeadLatentAttention Pallas/Mosaic kernel correctness."""

from tokamax._src.mosaic_tpu import pl_align_to, pl_cdiv
from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mla import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.mla import reference
from tokamax._src.ops.experimental.mla import utils

jax.config.parse_flags_with_absl()

hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=10,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile(name="deterministic")


class MlaKernelTest(parameterized.TestCase):

  @hp.given(hps.data())
  def test_mla_output_shapes(self, data):
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

    page_size = 16
    num_heads = data.draw(hps.sampled_from([4, 8]))
    lkv_dim = data.draw(hps.sampled_from([128, 256]))
    r_dim = data.draw(hps.sampled_from([64, 128]))
    dtype = jnp.bfloat16

    seq_lens_list = []
    for _ in range(data.draw(hps.integers(1, 4))):
      q_len = data.draw(hps.sampled_from([1, 4096, 8192]))
      kv_len = data.draw(hps.sampled_from([4096, 8192, 9216]))
      hp.assume(kv_len >= q_len)
      seq_lens_list.append((q_len, kv_len))

    total_q_len = sum(s[0] for s in seq_lens_list)
    total_kv_tokens = sum(s[1] for s in seq_lens_list)
    num_pages = pl_cdiv(total_kv_tokens, page_size) + len(seq_lens_list)

    inputs = utils.generate_mla_inputs(
        seq_lens_list,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        rng=np.random.default_rng(data.draw(hps.integers(0, 1000))),
    )
    (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ) = inputs

    assert cache_kv is not None

    out, updated_kv = pallas_mosaic_tpu_kernel.mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        num_kv_pages_per_block=4,
        num_queries_per_block=4,
        vmem_limit_bytes=64 * 1024 * 1024,
        s_dtype=jnp.bfloat16,
    )

    self.assertEqual(out.shape, (total_q_len, num_heads, lkv_dim))
    self.assertEqual(updated_kv.shape, cache_kv.shape)

    packing = utils.get_dtype_packing(dtype)
    padded_lkv_dim = pl_align_to(lkv_dim, 128)
    padded_r_dim = pl_align_to(r_dim, 128)
    padded_kv_dim = padded_lkv_dim + padded_r_dim
    page_count = sum(pl_cdiv(s[1], page_size) for s in seq_lens_list)
    num_pages_arg = pl_cdiv(total_kv_tokens, page_size) + len(seq_lens_list)
    total_num_pages = max(num_pages_arg, page_count)

    self.assertEqual(updated_kv.shape[0], total_num_pages)
    self.assertEqual(updated_kv.shape[1], page_size // packing)
    self.assertEqual(updated_kv.shape[2], packing)
    self.assertEqual(updated_kv.shape[3], padded_kv_dim)

  @hp.given(hps.data())
  def test_mla_correctness(self, data):
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

    page_size = data.draw(hps.sampled_from([16, 256, 1024]))
    num_heads = data.draw(hps.sampled_from([4, 8, 64, 128, 256]))
    lkv_dim = data.draw(hps.sampled_from([128, 256]))
    r_dim = data.draw(hps.sampled_from([64, 128]))
    dtype = jnp.bfloat16

    seq_lens = []
    q_len = data.draw(hps.sampled_from([1, 4096 // 2, 8192 // 2]))
    kv_len = data.draw(hps.sampled_from([4096 // 2, 8192 // 2, 9216 // 2]))
    batch_size = data.draw(hps.sampled_from([2, 4]))
    hp.assume(kv_len >= q_len)
    seq_lens.extend([(q_len, kv_len)] * batch_size)

    total_kv_tokens = sum(s[1] for s in seq_lens)
    num_pages = pl_cdiv(total_kv_tokens, page_size) + len(seq_lens)

    inputs = utils.generate_mla_inputs(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        rng=np.random.default_rng(data.draw(hps.integers(0, 1000))),
    )
    (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ) = inputs

    out_ref, _ = reference.mla_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    out, _ = pallas_mosaic_tpu_kernel.mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        num_kv_pages_per_block=4,
        num_queries_per_block=4,
        vmem_limit_bytes=64 * 1024 * 1024,
        s_dtype=jnp.bfloat16,
    )

    np.testing.assert_allclose(out, out_ref, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
  absltest.main()
