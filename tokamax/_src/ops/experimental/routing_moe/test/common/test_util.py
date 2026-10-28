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
"""Shared setup, profiling, and timing helpers for the routing_moe tests."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Self

from absl import logging
from absl.testing import absltest
import jax
from jax.experimental.pallas import tpu as pltpu
from tokamax._src import benchmarking
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases


def skip_unless_supported(
    test_case: absltest.TestCase, *, min_devices: int = 1
) -> None:
  """Skips `test_case` unless platform, JAX version, and device count match."""
  if jax.default_backend() != "tpu":
    test_case.skipTest("Only supported on TPUs.")
  if pltpu.get_tpu_info().generation < 7:
    test_case.skipTest("Only tested on TPUv7x for now.")
  if jax.__version_info__ < (0, 11, 2):
    test_case.skipTest("routing_moe requires JAX >= 0.11.2.")
  if jax.device_count() < min_devices:
    test_case.skipTest(f"Requires at least {min_devices} devices.")


def check_gmm_supported(
    case: test_cases.Case, *, live_group_walk: bool = False
) -> None:
  """Skips the test if `case` uses features unsupported by `routing_gmm`."""
  if case.shape.has_bias:
    raise absltest.SkipTest("TODO: Support bias in routing_moe.")
  if case.shape.quant_block_size is not None:
    raise absltest.SkipTest(
        "TODO: Support sub-channel quantization in routing_moe."
    )
  if case.local_intermediate_size % 256 != 0:
    raise absltest.SkipTest(
        "TODO: Fix gmm/plan.py raising error if intermediate is not"
        " multiple of 256 even when up & down can be fused."
    )
  if live_group_walk and case.num_tokens >= 65536:
    raise absltest.SkipTest(
        "TODO: live_group_walk places row_group in SMEM, which overflows"
        " at 64k tokens."
    )


def check_output_reduce_supported(case: test_cases.Case) -> None:
  """Skips the test if `case` exceeds `output_reduce` SMEM limits."""
  if case.num_tokens >= 65536:
    raise absltest.SkipTest(
        "TODO: Support SMEM streaming in output_reduce for 64k tokens."
    )


def check_moe_supported(
    case: test_cases.Case, *, remote_scatter: bool = False
) -> None:
  """Skips the test if `case` or the mesh layout is unsupported."""
  check_gmm_supported(case)
  check_output_reduce_supported(case)
  if remote_scatter and case.sharding.ep > 1 and case.sharding.tp > 1:
    raise absltest.SkipTest(
        "TODO: Support remote_scatter with hybrid parallelism."
    )


class XprofTrace:
  """Context manager that captures an XProf session and logs trace links.

  Attributes:
    links: Markdown links to XProf and Percale populated on context exit.
  """

  links: str = ""

  def __enter__(self) -> Self:
    return self

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    del exc_type, exc_value, traceback


@dataclasses.dataclass(frozen=True)
class Timed:
  """Standardized function and arguments for XProf benchmarking.

  Attributes:
    fn: Standardized callable taking `args`.
    args: Positional device array arguments for `fn`.
  """

  fn: Callable[[Sequence[jax.Array]], Any]
  args: Sequence[jax.Array]

  @classmethod
  def create(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Self:
    """Standardizes `fn(*args, **kwargs)` into a `Timed` instance."""
    return cls(*benchmarking.standardize_function(fn, *args, kwargs=kwargs))

  def latency_us(self) -> float:
    """Benchmarks `fn(args)` and returns the median latency in microseconds."""
    result = benchmarking.benchmark(self.fn, self.args, method="hermetic_xprof")
    return 1e3 * result.median_evaluation_time_ms

  def trace(self) -> str:
    """Runs `fn(args)` once under XProf and returns the trace links."""
    with XprofTrace() as xprof:
      jax.block_until_ready(jax.jit(self.fn)(self.args))
    return xprof.links
