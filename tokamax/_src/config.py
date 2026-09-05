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
"""Configuration options."""

import contextlib
import dataclasses
import sys
from typing import Any

from absl import flags
import jax

_DEFAULT = object()


@dataclasses.dataclass(frozen=True, slots=True)
class _ConfigOption[T]:
  """A configuration option."""

  flag: flags.FlagHolder[T]
  config: Any = None

  def __post_init__(self):
    if self.config is None:
      object.__setattr__(self, "config", jax.make_user_context(_DEFAULT))

  def __call__(self, value: T | str) -> contextlib.AbstractContextManager[None]:
    name = self.flag.name
    flag = self.flag._flagvalues[name]

    try:
      value_ = flag.parser.parse(value) if isinstance(value, str) else value
    except ValueError as e:
      raise ValueError(f"Invalid value for config `{name}`: {value}") from e

    return self.config(value_)

  @property
  def value(self) -> T:
    if not flags.FLAGS.is_parsed():
      # `known_only=True` parses the flags absl actually defines (so any
      # `--tokamax_*` flags on the command line still take effect) and ignores
      # the rest instead of raising `UnrecognizedFlagError`. Tokamax is a
      # library, so `sys.argv` may carry flags owned by the host program
      # (pytest's `-s`, vLLM's CLI args, etc.) that absl does not recognize.
      flags.FLAGS(sys.argv, known_only=True)
    return self.flag.value if (v := self.config.value) is _DEFAULT else v


autotuning_cache_miss_fallback = _ConfigOption(
    flags.DEFINE_enum(
        "tokamax_autotuning_cache_miss_fallback",
        "heuristics",
        ("heuristics", "autotune", "error"),
        "Fallback when no config is found in the autotuning cache by"
        " `BoundArguments.default_config` ('heuristics' - use heuristics to"
        " create a config; 'autotune' - autotune over the default autotuning"
        " configs and use the fastest; 'error' - raise an error)",
    )
)


cross_compile = _ConfigOption(
    flags.DEFINE_bool(
        "tokamax_cross_compile",
        False,
        "With this option the user can disable checks like"
        " `has_triton_support()` and `has_mosaic_gpu_support()` that check that"
        " the correct hardware is present. It is possible that the machine"
        " where kernels are lowered and compiled is not the same as the machine"
        " where they are run. `has_mosaic_gpu_support()` that check that the"
        " correct hardware is present.",
    )
)

ignore_autotuning_cache = _ConfigOption(
    flags.DEFINE_bool(
        "tokamax_ignore_autotuning_cache",
        False,
        "If true, ignore the autotuning cache when looking for configs and"
        " autotuning.",
    )
)

disable_multi_core_mode = _ConfigOption(
    flags.DEFINE_bool(
        "tokamax_disable_multi_core_mode",
        False,
        "Temporary flag to disable multi-core execution in GMM/TGMM v2 kernels"
        " and fall back to the legacy pl.pallas_call path to unblock jax.vmap."
        " TODO: Revert this flag once Tokamax updates its JAX"
        " version.",
    )
)
