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
"""Tests for MultiHeadLatentAttention Pallas/Mosaic kernel."""

from tokamax._src.mosaic_tpu import pl_align_to, pl_cdiv
import csv
import os
from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tensorboardX import writer
import tokamax
from tokamax._src.autotuning import api as autotuning
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla import pallas_mosaic_tpu
from tokamax._src.ops.experimental.mla import utils

PallasTpuMhla = pallas_mosaic_tpu.PallasTpuMultiHeadLatentAttention
ReferenceMhla = base.MultiHeadLatentAttention


def default_batched_decode_benchmark_params():
  params = {
      "batch_size": 128,
      "q_len": 1,
      "kv_len_val": 8192,
      "num_q_heads": 128,
      "lkv_dim": 512,
      "r_dim": 64,
      "page_size": 256,
      "q_dtype": jnp.float8_e4m3fn,
      "kv_dtype": jnp.float8_e4m3fn,
  }
  return params


def _generate_mla_params():
  params = []
  for bs in [128, 160, 192]:
    for kv_len in [1024, 9216]:
      params.append((bs, kv_len))
  return params


class MultiHeadLatentAttentionTest(parameterized.TestCase):
  def test_mla_benchmark_correctness(self):
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")
    (
        batch_size,
        q_len,
        _,
        num_q_heads,
        lkv_dim,
        r_dim,
        page_size,
        _,
        _,
    ) = default_batched_decode_benchmark_params().values()
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.bfloat16
    kv_len_val = 1024
    total_kv_tokens = batch_size * kv_len_val
    num_pages = pl_cdiv(total_kv_tokens, page_size) + batch_size
    seq_lens = [(q_len, kv_len_val)] * batch_size

    key = jax.random.PRNGKey(0)
    seed = int(np.array(key).flatten()[0])
    rng = np.random.default_rng(seed)

    inputs = utils.generate_mla_inputs(
        seq_lens,
        num_q_heads,
        lkv_dim,
        r_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        rng=rng,
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

    mla_op = PallasTpuMhla()

    assert cache_kv is not None
    cache_kv_copy = cache_kv.copy()
    op_out, op_updated_kv = mla_op(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv_copy,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        s_dtype=jnp.float32,
    )
    baseline_op = ReferenceMhla()
    baseline_out, baseline_updated_kv = baseline_op(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        s_dtype=jnp.float32,
    )
    # TODO: Improve precision.
    np.testing.assert_allclose(op_out, baseline_out, atol=0.01, rtol=0.01)
    np.testing.assert_allclose(
        op_updated_kv,
        baseline_updated_kv,
        atol=0.01,
        rtol=0.01,
    )


if __name__ == "__main__":
  absltest.main()
