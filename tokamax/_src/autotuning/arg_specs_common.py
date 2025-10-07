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

"""Common functionality for autotuning argument specs."""

import dataclasses
from typing import Any, Literal, TypeAlias

ProjectName: TypeAlias = Literal[
    'alphafold'
    'deepseek2',
    'mixtral',
]

# Tags are used to quickly identify different workloads for the same op.
# forward_only models are models that only require forward passes - meaning no
# vjp tuning is required.
# mlcompass models are models that are tracked by mlcompass. These are all
# internal models.
# primary models are models that are considered the "primary" models for the
# PA.
Tag: TypeAlias = Literal['primary']


@dataclasses.dataclass(frozen=True)
class ArgSpec:
  """Argument specification for an op with metadata.

  Attributes:
    args: The argument specification.
    project: The project the argument specification comes from.
    name: The name of the argument specification.
    tags: Tags for the argument specification.
  """

  args: dict[str, Any]
  project: ProjectName | None = None
  name: str | None = None
  tags: tuple[Tag, ...] = tuple()

  @property
  def full_name(self) -> str:
    """The full name (including project name)."""
    return '_'.join(filter(None, (self.project, self.name)))
