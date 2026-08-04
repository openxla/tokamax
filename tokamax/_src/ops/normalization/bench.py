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
"""Benchmarks for layer norm ops."""

import functools
from absl import app
from absl import flags
import google_benchmark
from tokamax._src import batching
from tokamax._src import benchmarking
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import jax_triton as jt_norm
from tokamax._src.ops.normalization import pallas_triton as pl_norm
from tokamax._src.ops.normalization import arg_specs

# Batch size for the batched (`vmap`) benchmarks.
_VMAP_BATCH = 8


_IMPLS = dict(
    pallas=pl_norm.PallasTritonNormalization(input_output_alias=False),
    jax_triton=jt_norm.JaxTritonNormalization(input_output_alias=False),
    xla=base.Normalization(),
)
_BENCHMARK_IMPLS_FWD = flags.DEFINE_list(
    'benchmark_impls_fwd',
    ','.join(_IMPLS),
    'List of implementations to benchmark forward only.',
)
_BENCHMARK_IMPLS_FWD_BWD = flags.DEFINE_list(
    'benchmark_impls_fwd_bwd',
    _BENCHMARK_IMPLS_FWD.default,
    'List of implementations to benchmark forward and backward.',
)


def _batched_args(args):
  """Returns `args` with x batched over a new leading axis (`scale`/`offset`

  shared). All array args become `batching.BatchedShapeDtype`, which the
  benchmark harness turns into a `jax.vmap` over the op — exercising the
  batching (fold-into-M) path.
  """
  def batched(sds, vmapped):
    vmap_axes = ((0, _VMAP_BATCH),) if vmapped else (None,)
    return batching.BatchedShapeDtype(sds.shape, sds.dtype, vmap_axes)

  out = dict(args)
  out['x'] = batched(args['x'], True)
  for k in ('scale', 'offset'):
    if args[k] is not None:
      out[k] = batched(args[k], False)
  return out


def _register_benchmarks():
  """Registers benchmarks."""
  register_benchmark = functools.partial(
      benchmarking.register_benchmark, iterations=10
  )

  # Non-batched and batched (`vmap`) variants of each arg spec.
  specs = [(s.full_name, s.args) for s in arg_specs.ARG_SPECS]
  specs += [(f'{s.full_name}_vmap', _batched_args(s.args)) for s in
            arg_specs.ARG_SPECS]

  for name, kwargs in specs:
    for impl_name in _BENCHMARK_IMPLS_FWD.value:
      impl = _IMPLS[impl_name]
      register_benchmark(name, impl_name, impl, kwargs)

  for name, kwargs in specs:
    for impl_name in _BENCHMARK_IMPLS_FWD_BWD.value:
      impl = _IMPLS[impl_name]
      register_benchmark(name, impl_name, impl, kwargs, mode='forward_and_vjp')


if __name__ == '__main__':
  app.call_after_init(_register_benchmarks)
  app.run(lambda _: google_benchmark.main())
