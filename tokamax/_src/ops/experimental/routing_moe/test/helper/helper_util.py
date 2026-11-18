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
import numpy as np
from tokamax._src import benchmarking
from tokamax._src import block_scaled
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_cases


def skip_unless_supported(
    test_case: absltest.TestCase, *, min_devices: int = 1
) -> None:
  """Skips `test_case` unless platform, TPU generation, and devices match."""
  if jax.default_backend() != "tpu":
    test_case.skipTest("Only supported on TPUs.")
  if pltpu.get_tpu_info().generation < 7:
    test_case.skipTest("Requires TPU generation >= 7.")
  if jax.device_count() < min_devices:
    test_case.skipTest(f"Requires at least {min_devices} devices.")


def skip_unless_block_scaled_supported(
    test_case: absltest.TestCase, *, min_devices: int = 1
) -> None:
  """Skips `test_case` unless block-scaled matmul and TPU_8T+ are available."""
  if not block_scaled.is_available():
    test_case.skipTest("Block-scaled matmul is not available in OSS.")
  skip_unless_supported(test_case, min_devices=min_devices)
  if not gmm_plan.is_block_scaled_tpu(pltpu.get_tpu_info()):
    test_case.skipTest("Block-scaled matmul requires TPU_8T or newer.")


def check_gmm_supported(
    case: helper_cases.Case, *, fused_up_and_down: bool = False
) -> None:
  """Skips the test if `case` uses features unsupported by `routing_gmm`.

  Args:
    case: The test case.
    fused_up_and_down: Whether the test only runs gate/up and down fused, which
      supports intermediate sizes that aren't a multiple of 256.
  """
  if case.local_intermediate_size % 256 != 0 and not fused_up_and_down:
    raise absltest.SkipTest(
        "TODO: Fix gmm/plan.py raising error if intermediate is not"
        " multiple of 256 even when up & down can be fused."
    )
  if case.shape.w_block_scaled:
    if not block_scaled.is_available():
      raise absltest.SkipTest("Block-scaled matmul is not available in OSS.")
    if not gmm_plan.is_block_scaled_tpu(pltpu.get_tpu_info()):
      raise absltest.SkipTest("Block-scaled matmul requires TPU_8T or newer.")


def check_moe_supported(
    case: helper_cases.Case, *, remote_scatter: bool = False
) -> None:
  """Skips the test if `case` or the mesh layout is unsupported."""
  check_gmm_supported(case)
  if case.shape.has_bias:
    raise absltest.SkipTest("TODO: Support bias in routing_moe.")
  if remote_scatter and case.sharding.ep > 1 and case.sharding.tp > 1:
    raise absltest.SkipTest(
        "TODO: Support remote_scatter with hybrid parallelism."
    )


def log_expert_load(
    name: str, group_sizes: jax.Array | np.ndarray, *, ep: int = 1
) -> None:
  """Logs the rows each expert receives, overall and per EP shard.

  Logs the max-to-mean rows per expert, then per shard its total rows and the
  min, 10th percentile, median, and max rows per expert.

  Args:
    name: Name of the test case.
    group_sizes: Rows routed to each global expert.
    ep: Number of EP shards, each serving a contiguous range of experts.
  """
  rows = np.asarray(group_sizes)
  parts = [f"max/mean rows per expert {rows.max() / rows.mean():.2f}x"]
  for shard, local in enumerate(np.split(rows, ep)):
    p10, p50 = np.percentile(local, [10, 50])
    parts.append(
        f"shard {shard}: {local.sum()} rows, per expert min {local.min()}"
        f" p10 {p10:.0f} p50 {p50:.0f} max {local.max()}"
    )
  logging.info("%s expert load: %s", name, "; ".join(parts))


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
