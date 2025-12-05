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
"""Benchmarks for attention implementations."""

import functools

from absl import app
from absl import flags
import google_benchmark
from tokamax._src import benchmarking
from tokamax._src.ops.attention import arg_specs
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import jax_nn
from tokamax._src.ops.attention import pallas_mosaic_gpu as mgpu_attn
from tokamax._src.ops.attention import pallas_triton as triton_attn
from tokamax._src.ops.flex_attention import pallas_triton as triton_flex
from tokamax._src.ops.flex_attention import wrapper
ARG_SPECS = arg_specs.ARG_SPECS


_IMPLS = dict(
    triton=triton_attn.PallasTritonFlashAttention(),
    triton_flex=wrapper.WrappedFlexAttention(
        impl=triton_flex.PallasTritonFlexAttention()
    ),
    mosaic=(mgpu_attn.PallasMosaicGpuFlashAttention()),
    cudnn=jax_nn.JaxNnDotProductAttention(implementation='cudnn'),
    xla=base.DotProductAttention(),
)

_BENCHMARK_IMPLS_FWD = flags.DEFINE_list(
    'benchmark_impls_fwd',
    ','.join(_IMPLS),
    'List of implementations to benchmark forward only.',
)
_BENCHMARK_IMPLS_FWD_BWD = flags.DEFINE_list(
    'benchmark_impls_fwd_bwd',
    'triton,cudnn,xla',
    'List of implementations to benchmark forward and backward.',
)
_register_benchmark = functools.partial(
    benchmarking.get_benchmark_registrar(_IMPLS),
    iterations=3,
    raise_on_error=False,
)


def _register_benchmarks():
  """Registers benchmarks."""
  for arg_spec in ARG_SPECS:
    for impl_name in _BENCHMARK_IMPLS_FWD.value:
      _register_benchmark(arg_spec.full_name, impl_name, arg_spec.args)

  for arg_spec in ARG_SPECS:
    name = arg_spec.full_name
    kwargs = arg_spec.args
    for impl_name in _BENCHMARK_IMPLS_FWD_BWD.value:
      _register_benchmark(name, impl_name, kwargs, mode='forward_and_vjp')


if __name__ == '__main__':
  app.call_after_init(_register_benchmarks)
  google_benchmark.main()
