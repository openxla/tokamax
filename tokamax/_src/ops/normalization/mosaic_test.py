# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Numerics for the shapes this kernel exists for: the benchmark arg specs.

Same shapes as `bench_lowering_test`, which pins that they lower; this checks
that what they lower to is right. `test_base` supplies the reference impl and
the tolerances; its own shape battery is skipped below, as those shapes are the
shared cross-impl set and this kernel is only aimed at the benchmark ones.
"""

import inspect

from absl.testing import absltest
import jax
from tokamax._src.ops.normalization import mosaic
from tokamax._src.ops.normalization import test_base


class PallasMosaicGpuNormalizationTest(test_base.NormalizationTestBase):
  """Runs `test_base`'s benchmark-shape tests, and only those."""

  # Everything the base class parameterizes over shapes of its own choosing.
  # `test_bench` is what we keep.
  _OUT_OF_SCOPE = ('test_layer_norm', 'test_rms_norm')

  def __init__(self, *args):
    # No backward kernel yet, and Mosaic GPU Pallas calls have no AD rule.
    super().__init__(
        *args,
        norm_fn=mosaic.PallasMosaicGpuNormalization(),
        supports_vjp=False,
    )

  def setUp(self):
    if self._testMethodName.startswith(self._OUT_OF_SCOPE):
      self.skipTest('Only the benchmark shapes are in scope for this kernel.')
    if not self._norm_fn.supported_on(jax.devices()[0]):
      self.skipTest('Mosaic GPU normalization not supported on this device.')
    super().setUp()


if __name__ == '__main__':
  absltest.main()
