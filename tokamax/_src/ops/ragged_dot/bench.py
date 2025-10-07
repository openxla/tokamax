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
"""Benchmarks for ragged dot implementations."""

import functools

from absl import app
from absl import flags
import google_benchmark
import jax
from jax.experimental import layout
import jax.numpy as jnp
from tokamax._src import benchmarking
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu as pl_mgpu
from tokamax._src.ops.ragged_dot import pallas_triton
from tokamax._src.ops.ragged_dot import bench_arg_specs

if jax.__version_info__ >= (0, 6, 3):
  DLL = layout.Layout
else:
  DLL = layout.DeviceLocalLayout  # type: ignore


# This provides an upper-bound for performance for compute-bound workloads.
def _xla_only_group0(lhs, rhs, *, group_sizes):
  """XLA baseline that ignores group sizes and only uses group 0 weights."""
  del group_sizes
  if isinstance(lhs, quantization.QuantizedArray):
    lhs = lhs.recompose()
  if isinstance(rhs, quantization.QuantizedArray):
    rhs = rhs.recompose()
  return jnp.matmul(lhs, rhs[0])


# This provides an upper-bound for performance for memory-bound workloads.
def _xla_even_groups(lhs, rhs, *, group_sizes):
  """XLA baseline that ignores group sizes and splits evenly between groups."""
  del group_sizes
  if isinstance(lhs, quantization.QuantizedArray):
    lhs = lhs.recompose()
  if isinstance(rhs, quantization.QuantizedArray):
    rhs = rhs.recompose()
  num_groups = rhs.shape[0]
  lhs = lhs.reshape(num_groups, lhs.shape[0] // num_groups, -1)
  return jnp.matmul(lhs, rhs)


def _flops(lhs, rhs, *, group_sizes) -> int:
  """Returns the floating-point operations."""
  del lhs  # Unused.
  _, k, n = rhs.shape
  return 2 * sum(group_sizes.value) * k * n


_IMPLS = dict(
    triton=pallas_triton.PallasTritonRaggedDot(),
    mosaic=pl_mgpu.PallasMosaicGpuRaggedDot(),
    xla=base.RaggedDot(),
    xla_only_group0=_xla_only_group0,
    xla_even_groups=_xla_even_groups,
)
_BENCHMARK_IMPLS = flags.DEFINE_list(
    'benchmark_impls', ','.join(_IMPLS), 'List of implementations to benchmark.'
)
_register_benchmark = functools.partial(
    benchmarking.get_benchmark_registrar(_IMPLS),
    iterations=3,
    items_processed_fn=_flops,
)


def _transpose_rhs(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
  dev_layout = DLL((0, 2, 1), ())
  no_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  dll_layout = layout.Format(dev_layout, no_sharding)
  return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=dll_layout)


def _register_benchmarks():
  """Registers benchmarks."""
  for arg_spec in bench_arg_specs.ARG_SPECS:
    name = arg_spec.full_name
    spec = arg_spec.args
    for impl_name in _BENCHMARK_IMPLS.value:
      if 'mosaic' in impl_name and (
          not mgpu_lib.has_mosaic_gpu_support()
          or isinstance(spec['lhs'], quantization.QuantizedArray)
          or not isinstance(spec['rhs'], quantization.QuantizedArray)
      ):
        continue
      if impl_name == 'xla_only_group0' and name != 'compute_bound':
        continue
      if impl_name == 'xla_even_groups' and name != 'memory_bound':
        continue
      _register_benchmark(name, impl_name, spec)

      # The MGPU implementation is optimized for transposed RHS.
      if 'mosaic' in impl_name:
        spec = spec | dict(rhs=jax.tree.map(_transpose_rhs, spec['rhs']))
        _register_benchmark(name + '_transposed_rhs', impl_name, spec)


if __name__ == '__main__':
  app.call_after_init(_register_benchmarks)
  flags.FLAGS.set_default('tokamax_autotuning_cache_miss_fallback', 'error')
  google_benchmark.main()
