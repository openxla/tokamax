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

"""Common Flash Attention Mosaic GPU utilities."""

import pydantic


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  # TODO: Relax constraints to multiple of 32.
  block_q: pydantic.conint(multiple_of=64, gt=0) = 64
  block_kv: pydantic.conint(multiple_of=64, gt=0) = 64
  num_stages: pydantic.conint(gt=1) = 2
