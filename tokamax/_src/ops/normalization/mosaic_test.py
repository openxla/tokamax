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
that what they lower to is right. The reference and the tolerances are borrowed
from `test_base`, but not its shape battery -- those shapes are the shared
cross-impl set, and this kernel is only aimed at the benchmark ones.
"""

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import mosaic
from tokamax._src.ops.normalization import test_base


class PallasMosaicGpuNormalizationTest(parameterized.TestCase):

  _supports_vjp = True
  # The reference impl, the vmap plumbing and the tolerances, without the shape
  # battery the base class's own test methods would drag in.
  _run_test = test_base.NormalizationTestBase._run_test

  def setUp(self):
    self._norm_fn = mosaic.PallasMosaicGpuNormalization()
    if not self._norm_fn.supported_on(jax.devices()[0]):
      self.skipTest('Mosaic GPU normalization not supported on this device.')
    super().setUp()

  @parameterized.named_parameters(test_base.NAMED_ARG_SPECS.items())
  def test_bench(self, kwargs):
    ba = inspect.signature(base.Normalization.__call__).bind(None, **kwargs)
    ba.apply_defaults()
    ba.arguments.pop('return_residuals')
    self._run_test(*ba.args[1:], **ba.kwargs)


if __name__ == '__main__':
  absltest.main()
