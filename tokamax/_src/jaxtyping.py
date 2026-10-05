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
"""jaxtyping utilities."""

import contextlib
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Array, Int  # pylint: disable=g-importing-member,g-multiple-import
import numpy as np
import pydantic


ScalarInt = Int[Array, ""] | Int[np.generic, ""] | Int[jnp.generic, ""]
# pydantic (already a dependency) is used as the runtime typechecker rather
# than typeguard: typeguard>=3 is incompatible with jaxtyping (its AST
# instrumentation parses shape strings like "A B" as forward references), and
# pinning typeguard==2.13.3 conflicts with packages requiring typeguard>=4
# (see https://github.com/openxla/tokamax/issues/240).
_typechecker = pydantic.validate_call(
    config=pydantic.ConfigDict(arbitrary_types_allowed=True),
    validate_return=True,
)
jaxtyped = jaxtyping.jaxtyped(typechecker=_typechecker)


@contextlib.contextmanager
def disable_jaxtyping(disable: bool = True):
  """Disables jaxtyping for the duration of the context."""

  current_context = jaxtyping.config.jaxtyping_disable
  try:
    jaxtyping.config.update("jaxtyping_disable", disable)
    yield
  finally:
    jaxtyping.config.update("jaxtyping_disable", current_context)
