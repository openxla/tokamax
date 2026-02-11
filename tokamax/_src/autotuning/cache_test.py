# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
from importlib import resources
import os
import re
from typing import Any, Final
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
import jax
from tokamax._src.autotuning import cache
from tokamax._src.ops.attention import base as attention_base
from tokamax._src.ops.normalization import pallas_triton

_CACHE_PATHS: Final[immutabledict.immutabledict[str, str]] = (
    immutabledict.immutabledict({
        "external": "data/autotuning",
    })
)


class CacheTest(parameterized.TestCase):

  @parameterized.parameters(
      ("NVIDIA H100 80GB HBM3", pallas_triton.PallasTritonNormalization),
      ("TPU7x", attention_base.DotProductAttention),
      ("not_a_real_device", pallas_triton.PallasTritonNormalization),
  )
  def test_load_cache(self, device: str, op_cls: Any):
    c = cache.AutotuningCache(op_cls())

    self.assertIsInstance(c._load_cache(device), dict)
    if device == "not_a_real_device":
      self.assertEmpty(c._load_cache(device))
    else:
      self.assertNotEmpty(c._load_cache(device))


if __name__ == "__main__":
  absltest.main()
