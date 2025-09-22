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

import os
import sys
from absl import flags
import jax
import pytest


@pytest.fixture(scope='session', autouse=True)
def tokamax_testing_setup():
  """Parse absl flags and disable x64 for JAX."""
  # TODO: Remove this once x64 is supported.
  jax.config.update('jax_enable_x64', False)
  flags.FLAGS(sys.argv[:1])
  os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + (
      ' --xla_gpu_deterministic_ops=True '
      ' --xla_gpu_exclude_nondeterministic_ops=True'
  )
  yield
