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

"""Benchmarks for Multi-Head Latent Attention (MLA)."""

import json
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
import jax
import tokamax
from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import arg_specs as mla_specs
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
    for spec in mla_specs.ARG_SPECS
    if 'primary' in spec.tags
})


def setUpModule():  # pylint: disable=invalid-name
  """Runs once before any tests in this module start."""
  metadata_dir = os.environ.get('WORKLOAD_METADATA_DIR')
  if not metadata_dir:
    return

  metadata: dict[str, str] = {}
  if jax.default_backend() == 'gpu':
    cuda_versions = getattr(jax._src.lib, 'cuda_versions', None)  # pylint: disable=protected-access
    if cuda_versions is not None:
      try:
        cudnn_version = cuda_versions.cudnn_get_version()
      except AttributeError:
        pass
      else:
        metadata['cudnn_version'] = str(cudnn_version)

  if metadata:
    with open(os.path.join(metadata_dir, 'workload_info.json'), 'w') as f:
      json.dump(metadata, f)


class MlaBenchmark(parameterized.TestCase):
  """Performance benchmarks for Multi-Head Latent Attention (MLA) kernels."""

  @parameterized.product(
      implementation=(None, 'mosaic', 'xla'),
      benchmark_mode=('forward',),
      args_spec_name=tuple(EXAMPLES.keys()),
  )
  def test_mla(self, implementation, benchmark_mode, args_spec_name):
    """Benchmarks MLA forward pass for a given configuration."""
    logging.info('device_kind=%s', jax.devices()[0].device_kind)

    if str(implementation) in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          ' --skip_implementations flag.'
      )

    if implementation == 'mosaic':
      if jax.default_backend() != 'tpu':
        self.skipTest('Mosaic TPU implementation is only supported on TPU.')
      mosaic_impl = mla_api.IMPLEMENTATIONS.get('mosaic_tpu')
      if mosaic_impl is None or not mosaic_impl.supported_on(jax.devices()[0]):
        self.skipTest('Mosaic TPU implementation is only supported on TPU v7+.')

    example = EXAMPLES[args_spec_name] | {'implementation': implementation}
    fn, args = tokamax.standardize_function(
        mla_api.multi_head_latent_attention,
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
            f"mla/{args_spec_name}/{implementation or 'default'}/{benchmark_mode}"
        ),
    )


if __name__ == '__main__':
  absltest.main()
