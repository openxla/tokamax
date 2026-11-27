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
"""FlexAttention Pallas-Mosaic-GPU implementation."""

import dataclasses
from typing import Any, ClassVar, override

import immutabledict
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.experimental.flex_attention import base
from tokamax._src.ops.experimental.flex_attention import pallas_mosaic_gpu_common as common
from tokamax._src.ops.experimental.flex_attention import pallas_mosaic_gpu_kernel_sm90 as sm90

Config = sm90.Config
type Key = immutabledict.immutabledict[str, Any]
MaskMod = base.MaskMod
QArray = base.QArray
Residuals = base.Residuals
ScoreMod = base.ScoreMod


def _get_kernel_module():
  if not gpu_utils.has_mosaic_gpu_support():
    raise NotImplementedError("Mosaic GPU not supported on this platform.")
  if gpu_utils.is_sm90():
    return sm90
  raise NotImplementedError("Only supported for sm90 GPUs.")


@dataclasses.dataclass(frozen=True)
class PallasMosaicGpuFlexAttention(base.FlexAttention[Config, Key]):
  """FlexAttention with Mosaic GPU.

  Attributes:
    use_base2: If `True`, use base-2 exponential and logarithms.
    use_stable_softmax: If `True`, use stable softmax, where the maximum value
      will be subtracted from the logits before applying the softmax function,
      improving numerical stability.
    rescale_threshold: The threshold for rescaling the accumulator when using
      stable softmax.
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  use_base2: bool = True
  use_stable_softmax: bool = True
  rescale_threshold: float = 1.0

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array | QArray, "*B T H D"],
      k: Float[Array | QArray, "*B t h D"],
      v: Float[Array | QArray, "*B t h d"],
      *,
      precision: tuple[base.CanonicalPrecision, base.CanonicalPrecision],
      score_mod: ScoreMod | None,
      mask_mod: MaskMod | None,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
    """Performs FlexAttention, optionally returning softmax residuals."""
    kernel_module = _get_kernel_module()

    q, k, v = map(quantization.as_array, (q, k, v))
    supported_dtypes = (jnp.float32, jnp.float16, jnp.bfloat16)
    if any(x.dtype not in supported_dtypes for x in (q, k, v)):
      raise NotImplementedError("Only f32, f16 and bf16 inputs are supported.")

    out_dtype = q.dtype
    q, k, v = common.prepare_qkv(q, k, v, precision)

    return kernel_module.flex_attention_kernel(
        q,
        k,
        v,
        score_mod=score_mod,
        mask_mod=mask_mod,
        dropout_mask=dropout_mask,
        dropout_rate=dropout_rate,
        out_dtype=out_dtype,
        normalize_output=normalize_output,
        return_residuals=return_residuals,
        use_base2=self.use_base2,
        use_stable_softmax=self.use_stable_softmax,
        rescale_threshold=self.rescale_threshold,
        config=config,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return _get_kernel_module().get_heuristics_config(ba)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return _get_kernel_module().get_autotuning_configs(ba)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device) and gpu_utils.is_sm90(
        device
    )
