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
"""Checks the Mosaic normalization kernel lowers for the benchmark shapes.

Ahead-of-time lowering, so no GPU is needed: the shapes come from `arg_specs`
(what `bench.py` runs), the device is a stand-in for an A100, and the target is
pinned to CUDA. A lowering failure here is a bug in the kernel; a shape the
kernel declines with `NotImplementedError` is a benchmark that silently falls
back to another impl, so those are reported too rather than passing quietly.
"""

import contextlib
import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
from tokamax._src import config as config_lib
from tokamax._src.ops.normalization import arg_specs
from tokamax._src.ops.normalization import mosaic


class _AmpereDevice:
  """Just enough of a `jax.Device` for the support checks and heuristics."""

  platform = 'gpu'
  device_kind = 'NVIDIA A100-SXM4-80GB'
  compute_capability = '8.0'
  core_count = 108  # A100 SMs.


@contextlib.contextmanager
def _pretending_to_be_ampere():
  device = _AmpereDevice()
  with (
      config_lib.cross_compile(True),
      mock.patch.object(jax, 'devices', lambda *_, **__: [device]),
      mock.patch.object(backend, 'get_default_device', lambda: device),
  ):
    yield


class BenchLoweringTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (spec.full_name, spec) for spec in arg_specs.ARG_SPECS
  )
  def test_lowers(self, spec):
    args = dict(spec.args)
    x, scale, offset = (args.pop(k) for k in ('x', 'scale', 'offset'))
    norm = mosaic.PallasMosaicGpuNormalization(input_output_alias=False)

    with _pretending_to_be_ampere():
      f = jax.jit(functools.partial(norm, **args))
      lowered = f.trace(x, scale, offset).lower(lowering_platforms=('cuda',))

    self.assertIn('mosaic_gpu', lowered.as_text())


if __name__ == '__main__':
  absltest.main()
