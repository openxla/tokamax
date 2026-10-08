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
"""Tests for Lightning Indexer API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.lightning_indexer import api
from tokamax._src.ops.experimental.lightning_indexer import base
from tokamax._src.ops.experimental.lightning_indexer import test_base


class ApiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_base.skip_if_unsupported(self)

  def test_api_xla(self):
    inputs = test_base.make_test_inputs([1, 2], [128, 256], seed=99)

    @jax.jit
    def f(
        q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens, dist
    ):
      return api.lightning_indexer(
          q=q,
          indexer_weights=indexer_weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=dist,
          k=16,
          implementation="xla",
      )

    args = (
        inputs["q"],
        inputs["indexer_weights"],
        inputs["cache_kv"],
        inputs["seq_lens"],
        inputs["page_indices"],
        inputs["cu_q_lens"],
        inputs["distribution"],
    )
    out_api = np.asarray(f(*args))
    out_base = np.asarray(base.LightningIndexer()(**inputs, k=16))
    np.testing.assert_array_equal(out_api, out_base)

    opspecs = hlo_utils.get_opspecs(f.lower(*args), include_xla_kernels=False)
    self.assertEmpty(opspecs)

  def test_api_mosaic_tpu(self):
    inputs = test_base.make_test_inputs([1, 4], [256, 256], seed=123)

    @jax.jit
    def f(
        q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens, dist
    ):
      return api.lightning_indexer(
          q=q,
          indexer_weights=indexer_weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=dist,
          k=16,
          implementation="mosaic",
      )

    args = (
        inputs["q"],
        inputs["indexer_weights"],
        inputs["cache_kv"],
        inputs["seq_lens"],
        inputs["page_indices"],
        inputs["cu_q_lens"],
        inputs["distribution"],
    )
    out_tpu = np.asarray(f(*args))
    out_ref = np.asarray(
        api.lightning_indexer(**inputs, k=16, implementation="xla")
    )
    np.testing.assert_array_equal(
        np.sort(out_tpu, axis=-1),
        np.sort(out_ref, axis=-1),
    )

    opspecs = hlo_utils.get_opspecs(f.lower(*args), include_xla_kernels=False)
    self.assertNotEmpty(opspecs)
    self.assertIsInstance(opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic"]))


if __name__ == "__main__":
  absltest.main()
