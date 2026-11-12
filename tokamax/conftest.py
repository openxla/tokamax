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

"""Pytest configuration for Tokamax."""

import inspect
import os
import sys
from absl import flags
import jax
import pytest

# Backward-compatibility smoke mode, set by the `compat-tests` CI job. A JAX
# compat break -- a renamed symbol, a Mosaic lowering that changed -- fails
# every case of the kernel it touches, so one case per test function finds it,
# and the rest of a parameter grid only re-proves it. Each test function runs
# until one of its cases has actually run; the remaining cases are skipped.
#
# Decided at run time rather than by deselecting at collection, because a
# case can skip itself (a GPU-only dtype on TPU): that one does not count, and
# the next case gets its turn.
_COMPAT_SMOKE = os.environ.get('TOKAMAX_COMPAT_SMOKE') == '1'

# Test function -> node ID of the case that ran it.
_smoke_ran: dict[tuple[object, object], str] = {}


def _smoke_key(item: pytest.Item) -> tuple[object, object]:
  """Identifies the test function a collected case was generated from.

  absl's `parameterized` wraps every generated case with `functools.wraps`, so
  unwrapping leads each case back to the one function it came from; pytest's
  own `parametrize` reuses that function directly. Keyed by class too, because
  a method inherited from a `*TestBase` is the same function in every subclass,
  and each subclass tests a different implementation.

  Args:
    item: A collected test case.

  Returns:
    A key equal for every case of the same test function in the same class.
  """
  if not isinstance(item, pytest.Function):
    # Not a Python test function (a doctest, say): nothing to group it with.
    return (item.nodeid, None)
  return (item.cls or item.module, inspect.unwrap(item.function))


def pytest_runtest_setup(item: pytest.Item) -> None:
  if _COMPAT_SMOKE and (ran := _smoke_ran.get(_smoke_key(item))):
    pytest.skip(f'compat smoke: {ran} already ran this test function')


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
  # pytest calls this once per phase of each test case:
  #   - after setup (`call.when == 'setup'`),
  #   - after the test function runs (`call.when == 'call'`),
  #   - after teardown (`call.when == 'teardown'`).
  # Only the `call` phase can mark a test function as run: a case skipped in
  # `pytest_runtest_setup` above, or one that fails in setup, never reaches it,
  # and a case that skips itself reports `call` as skipped.
  report = yield
  # A failure counts as having run: the case is reported, and the rest of the
  # grid would only fail the same way.
  if _COMPAT_SMOKE and call.when == 'call' and not report.skipped:
    _smoke_ran.setdefault(_smoke_key(item), item.nodeid)
  return report


@pytest.fixture(scope='session', autouse=True)
def tokamax_testing_setup():
  """Parse absl flags and disable x64 for JAX."""
  # TODO: Remove this once x64 is supported.
  jax.config.update('jax_enable_x64', False)
  flags.FLAGS(sys.argv[:1])
  os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + (
      ' --xla_gpu_deterministic_ops=True '
      ' --xla_gpu_exclude_nondeterministic_ops=True'
      ' --xla_gpu_enable_triton_gemm=False'
  )
  yield
