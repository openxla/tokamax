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
import json
import os
import re
from typing import Any, Final

from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
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
    device_cache = cache.AutotuningCache(op_cls())._load_cache(device)
    self.assertIsInstance(device_cache, dict)
    if device == "not_a_real_device":
      self.assertEmpty(device_cache)
    else:
      self.assertNotEmpty(device_cache)

  def test_validate_json_cache_files(self):
    """Checks that all cache files are valid JSON."""
    tokamax_files = resources.files("tokamax")
    for cache_path in _CACHE_PATHS.values():
      for device_dir in (tokamax_files / cache_path).iterdir():
        for cache_file in device_dir.iterdir():
          with self.subTest(str(cache_file)):
            self.assertTrue(cache_file.is_file())
            self.assertTrue(cache_file.name.endswith(".json"))
            json.loads(cache_file.read_text())


if __name__ == "__main__":
  absltest.main()
