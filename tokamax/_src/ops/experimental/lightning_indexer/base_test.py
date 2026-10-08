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
"""Tests for base Lightning Indexer operator."""

from typing import Any
from absl.testing import absltest
import jax.numpy as jnp
from tokamax._src.ops.experimental.lightning_indexer import base
from tokamax._src.ops.experimental.lightning_indexer import test_base


class LightningIndexerBaseTest(test_base.LightningIndexerTestBase):

  def __init__(self, *args):
    super().__init__(*args, topk_fn=base.LightningIndexer())

  def test_validation_errors(self):
    op = base.LightningIndexer()
    inputs = test_base.make_test_inputs([1, 2], [128, 256], seed=0)

    with self.assertRaisesRegex(ValueError, "k must be positive"):
      op.bind(**inputs, k=0)

    with self.assertRaisesRegex(ValueError, "compression_ratio must be a"):
      op.bind(**inputs, k=16, compression_ratio=3)

    bad_cu_q: dict[str, Any] = {
        **inputs,
        "cu_q_lens": jnp.zeros((5,), dtype=jnp.int32),
    }
    with self.assertRaisesRegex(ValueError, "cu_q_lens length"):
      op.bind(**bad_cu_q, k=16)

    bad_pages: dict[str, Any] = {
        **inputs,
        "page_indices": jnp.zeros((5,), dtype=jnp.int32),
    }
    with self.assertRaisesRegex(ValueError, "page_indices length"):
      op.bind(**bad_pages, k=16)


if __name__ == "__main__":
  absltest.main()
