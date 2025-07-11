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
"""Triton utils."""

import jax
from tokamax._src import config as config_lib


# TODO: this should be maintained inside jax-triton.
def has_triton_support() -> bool:
  """Returns True if Triton is supported by the default JAX device."""
  if config_lib.cross_compile.value:
    return True

  if jax.default_backend() != 'gpu':
    return False

  # TODO: A temporary workaround for Pathways arrays not having a
  # compute_capability attribute.
  if jax.default_backend() == 'gpu':
    if not hasattr(jax.devices()[0], 'compute_capability'):
      return True

  # Only currently supported for Ampere and above.
  return float(jax.devices()[0].compute_capability) >= 8.0
