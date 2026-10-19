# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tokamax experimental operations public namespace."""

import importlib
from typing import Any

# pylint: disable=g-importing-member,useless-import-alias
from tokamax._src.ops.experimental.mla_swa.api import mla_swa as mla_swa
from tokamax._src.ops.experimental.tpu.splash_attention.api import splash_attention as splash_attention

# pylint: enable=g-importing-member,useless-import-alias

# pylint: disable=undefined-all-variable
__all__ = (
    "mla_swa",
    "splash_attention",
    "tpu",
)
# pylint: enable=undefined-all-variable


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
  if name == "tpu":
    return importlib.import_module(f"{__name__}.{name}")
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
