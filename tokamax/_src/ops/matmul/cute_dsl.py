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
import dataclasses

import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op
from tokamax._src.ops.matmul import base
import tokamax._src.ops.matmul.cute_dsl_common as common
import tokamax._src.ops.matmul.cute_dsl_kernel_sm100 as sm100
from typing_extensions import override

Config = common.Config


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CuteDslMatmul(base.Matmul[Config, None]):

  @override
  def _fwd(
      self,
      a: jax.Array,
      b: jax.Array,
      *,
      precision: base.CanonicalPrecision,
      preferred_element_type: jnp.dtype | None,
      return_residuals: bool,
      config: Config,
  ) -> tuple[jax.Array, base.Residuals]:

    if not gpu_utils.is_sm100():
      raise NotImplementedError("Unsupported GPU architecture.")
    if not precision_lib.is_default(a.dtype, b.dtype, precision):
      raise NotImplementedError(f"{precision=} not supported.")
    fn = sm100.matmul_kernel
    if preferred_element_type is None:
      preferred_element_type = jnp.promote_types(a.dtype, b.dtype)

    matmul_out = fn(
        a,
        b,
        preferred_element_type,
        config,
    )
    return matmul_out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    if gpu_utils.is_sm100():
      return Config(
          num_ab_stages=4,
          num_acc_stages=2,
          block_m=256,
          block_k=64,
      )
    else:
      raise NotImplementedError("Unsupported GPU architecture.")

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    configs = set()
    for num_ab_stages in (3, 4):
      for num_acc_stages in range(1, 2+1):
        for block_m in (128, 256):
          for block_k in (16, 32, 64, 128):
            configs.add(
              Config(
                num_ab_stages=num_ab_stages,
                num_acc_stages=num_acc_stages,
                block_m=block_m,
                block_k=block_k,
              )
            )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.is_sm100(device)
