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

"""Benchmarks for Sparse Multi-Head Latent Attention (SparseMLA)."""

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
import jax
import tokamax
from tokamax._src.ops.sparse_mla import api as sparse_mla_api
from tokamax._src.ops.sparse_mla import arg_specs as sparse_mla_specs
from tokamax._src.ops.sparse_mla import pallas_mosaic_tpu
from tokamax.benchmarks import common

_TENSORBOARD_OUTPUT_ENV_VAR = flags.DEFINE_string(
    'tensorboard_output_env_var',
    'TENSORBOARD_OUTPUT_DIR',
    'Environment variable to use to retrieve TensorBoard output directory.',
)
_SKIP_IMPLEMENTATIONS = flags.DEFINE_list(
    'skip_implementations',
    [],
    'A comma-separated list of implementations to skip.',
)

EXAMPLES = immutabledict.immutabledict({
    spec.name: spec.args
    for spec in sparse_mla_specs.ARG_SPECS
    if 'primary' in spec.tags
})


class SparseMlaBenchmark(parameterized.TestCase):
  """Performance benchmarks for Sparse Multi-Head Latent Attention (SparseMLA)."""

  def tearDown(self):
    """Frees any resources that may be held by the runner."""
    super().tearDown()
    for arr in jax.live_arrays():
      arr.delete()
    jax.clear_caches()

  @parameterized.product(
      implementation=(None, 'mosaic_tpu', 'xla'),
      benchmark_mode=('forward',),
      args_spec_name=tuple(EXAMPLES.keys()),
  )
  def test_sparse_mla(self, implementation, benchmark_mode, args_spec_name):
    """Benchmarks SparseMLA forward pass for a given configuration."""
    logging.info('device_kind=%s', jax.devices()[0].device_kind)

    if str(implementation) in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          ' --skip_implementations flag.'
      )

    if (
        implementation == 'mosaic_tpu'
        and not pallas_mosaic_tpu.PallasTpuSparseMla().supported_on(
            jax.devices()[0]
        )
    ):
      self.skipTest(
          'PallasTpuSparseMla is only supported on TPU v7+ with SparseCore.'
      )

    example = EXAMPLES[args_spec_name] | {'implementation': implementation}
    fn, args = tokamax.standardize_function(
        sparse_mla_api.sparse_mla,
        kwargs=example,
        mode=benchmark_mode,
    )
    fn = jax.jit(fn)
    res = tokamax.benchmark(fn, args)
    res_wallclock = tokamax.benchmark(fn, args, method='wallclock')

    logging.info(
        'wallclock_median_time_ms: %s', res_wallclock.median_evaluation_time_ms
    )

    common.write_tensorboard_logs(
        tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
        value=res.evaluation_times_ms,
        metric_tag=(
            f"sparse_mla/{args_spec_name}/{implementation or 'default'}/{benchmark_mode}"
        ),
    )


if __name__ == '__main__':
  absltest.main()
