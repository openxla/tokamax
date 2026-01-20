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
import pydantic


@pydantic.dataclasses.dataclass(frozen=True, slots=True)
class Config:
  num_ab_stages: pydantic.conint(ge=3)
  num_acc_stages: pydantic.PositiveInt
  block_m: pydantic.conint(multiple_of=8, ge=8, le=256)
  block_k: pydantic.conint(multiple_of=16, ge=16)
