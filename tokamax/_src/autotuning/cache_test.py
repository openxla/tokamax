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
from typing import Final
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

  def test_load_cache(self):
    device_kind = jax.devices()[0].device_kind
    if device_kind != "NVIDIA H100 80GB HBM3":
      self.skipTest("Only NVIDIA H100 80GB HBM3 is supported.")

    c = cache.AutotuningCache(pallas_triton.PallasTritonNormalization())

    self.assertIsInstance(c._load_cache("not_a_real_device"), dict)
    self.assertEmpty(c._load_cache("not_a_real_device"))

    self.assertNotEmpty(c._load_cache(device_kind))

  def test_caches_exist(self):
    """Checks that the cache files exist for the given device kind."""
    # TODO: Right now we only have H100 pallas triton normalization cache.
    # Add more caches for other devices and operations.
    tokamax_files = resources.files("tokamax")
    path = tokamax_files.joinpath(
        _CACHE_PATHS["external"],
        "nvidia_h100_80gb_hbm3",
        "pallas_triton_normalization.json",
    )
    self.assertTrue(path.is_file())

  def test_default_cache(self):
    device_kind = jax.devices()[0].device_kind
    flex_attention_cache = cache.AutotuningCache(
        attention_base.DotProductAttention()
    )._load_cache(device_kind)
    self.assertEmpty(flex_attention_cache)


if __name__ == "__main__":
  absltest.main()
