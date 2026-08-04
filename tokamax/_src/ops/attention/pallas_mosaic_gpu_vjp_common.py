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

"""Common Flash Attention Mosaic GPU VJP utilities."""

from typing import Annotated

import jax
import pydantic


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_q_dkv: Annotated[int, pydantic.Field(multiple_of=64, gt=0)]
  block_kv_dkv: Annotated[int, pydantic.Field(multiple_of=64, gt=0)]
  block_q_dq: pydantic.PositiveInt
  block_kv_dq: pydantic.PositiveInt
  num_stages: pydantic.PositiveInt = 2
  compute_wgs: pydantic.PositiveInt = 2


def get_ds_dtype(
    q: jax.Array,
    k: jax.Array,
    bias: jax.Array | None,
    dbias_intermediate_dtype: jax.typing.DTypeLike | None,
) -> jax.typing.DTypeLike | None:
  """Returns the dtype for ds output."""
  if bias is None:
    return None
  if dbias_intermediate_dtype is None:
    return bias.dtype
  if bias.shape == (*q.shape[:-3], q.shape[-2], q.shape[-3], k.shape[-3]):
    return bias.dtype
  return dbias_intermediate_dtype
