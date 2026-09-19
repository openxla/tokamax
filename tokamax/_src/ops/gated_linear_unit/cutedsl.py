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
"""CuTeDSL SM100 implementation of the gated linear unit op."""

from collections.abc import Callable
from typing import Annotated, ClassVar, override

import cudnn
import cutlass
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
import pydantic
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Configuration for the CuTeDSL SM100 Gated Linear Unit.

  Attributes:
    mma_tiler_m: MMA tile size in the M dimension (128 or 256).
    mma_tiler_n: MMA tile size in the N dimension (multiple of 32 between 32 and
      256).
    cluster_shape_m: Cluster shape in the M dimension (positive power of 2).
    cluster_shape_n: Cluster shape in the N dimension (positive power of 2).
  """

  mma_tiler_m: Annotated[
      int, pydantic.Field(ge=128, le=256, multiple_of=128)
  ] = 128
  mma_tiler_n: Annotated[int, pydantic.Field(ge=32, le=256, multiple_of=32)] = (
      128
  )
  cluster_shape_m: pydantic.PositiveInt = 1
  cluster_shape_n: pydantic.PositiveInt = 1


class CuteDslGatedLinearUnit(base.GatedLinearUnit[Config, None]):
  """CuTeDSL SM100 Gated Linear Unit."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.is_sm100(device)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba  # Unused.
    return Config()

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba  # Unused.
    configs = set()
    for n in (64, 128, 192, 256):
      configs.add(
          Config(
              mma_tiler_m=128,
              mma_tiler_n=n,
              cluster_shape_m=1,
              cluster_shape_n=1,
          )
      )
      configs.add(
          Config(
              mma_tiler_m=256,
              mma_tiler_n=n,
              cluster_shape_m=2,
              cluster_shape_n=1,
          )
      )
      configs.add(
          Config(
              mma_tiler_m=256,
              mma_tiler_n=n,
              cluster_shape_m=2,
              cluster_shape_n=2,
          )
      )
    return configs

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      x: Float[Array, "*B M K"],
      weights: base.FusedWeights | base.UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      precision: base.CanonicalPrecision,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B M N"], base.Residuals | None]:
    if not gpu_utils.is_sm100():
      raise NotImplementedError("CuteDSL GLU is only supported on SM100+ GPUs.")

    if activation not in (jax.nn.swish, jax.nn.silu) and getattr(
        activation, "__name__", ""
    ) not in ("swish", "silu"):
      raise NotImplementedError(
          f"CuteDSL GLU only supports swish/silu activation, got {activation}."
      )

    weight_dtype = (
        weights[0].dtype if isinstance(weights, tuple) else weights.dtype
    )
    if not precision_lib.is_default(x.dtype, weight_dtype, precision):
      raise NotImplementedError(
          f"CuteDSL GLU only supports default precision, got {precision=}."
      )

    supported_dtypes = {jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)}
    if jnp.dtype(x.dtype) not in supported_dtypes:
      raise NotImplementedError(
          f"CuteDSL GLU only supports {supported_dtypes}, got {x.dtype}."
      )
    if isinstance(weights, tuple):
      if jnp.dtype(weights[0].dtype) != jnp.dtype(x.dtype) or jnp.dtype(
          weights[1].dtype
      ) != jnp.dtype(x.dtype):
        raise NotImplementedError(
            f"Weight dtypes ({weights[0].dtype}, {weights[1].dtype}) must match"
            f" input dtype ({x.dtype})."
        )
    else:
      if jnp.dtype(weights.dtype) != jnp.dtype(x.dtype):
        raise NotImplementedError(
            f"Weight dtype ({weights.dtype}) must match input dtype"
            f" ({x.dtype})."
        )

    k = x.shape[-1]
    if k % 8 != 0:
      raise NotImplementedError(
          f"CuteDSL GLU requires K ({k}) to be a multiple of 8."
      )

    n_out = (
        weights[0].shape[-1]
        if isinstance(weights, tuple)
        else weights.shape[-1]
    )
    if n_out % 32 != 0:
      raise NotImplementedError(
          f"CuteDSL GLU requires N ({n_out}) to be a multiple of 32."
      )

    weights = (
        jnp.stack(weights, axis=1) if isinstance(weights, tuple) else weights
    )

    def fn(x, weights):
      out_shape = x.shape[:-1] + (weights.shape[-1],)
      x_collapsed = jax.lax.collapse(x, 0, -1)
      m_total, k_dim = x_collapsed.shape
      n_dim = weights.shape[-1]
      num_blocks = n_dim // 32

      w_gate = weights[:, 0, :]
      w_up = weights[:, 1, :]
      w_up_blocks = w_up.T.reshape(num_blocks, 32, k_dim)
      w_gate_blocks = w_gate.T.reshape(num_blocks, 32, k_dim)
      b = jnp.stack([w_up_blocks, w_gate_blocks], axis=1).reshape(
          2 * n_dim, k_dim, 1
      )

      a = jnp.expand_dims(x_collapsed, axis=-1)

      ab12, c = cudnn.gemm_swiglu_jax_sm100(  # pyrefly: ignore[missing-attribute]
          a,
          b,
          alpha=1.0,
          ab12_dtype=x.dtype,
          c_dtype=x.dtype,
          acc_dtype=cutlass.Float32,
          mma_tiler_mn=(config.mma_tiler_m, config.mma_tiler_n),
          cluster_shape_mn=(config.cluster_shape_m, config.cluster_shape_n),
      )

      out = c[:, :, 0].reshape(out_shape)

      if return_residuals:
        ab12_squeezed = ab12[:, :, 0]
        blocks = ab12_squeezed.reshape(m_total, num_blocks, 2, 32)
        proj = blocks[:, :, 0, :].reshape(m_total, n_dim)
        gate = blocks[:, :, 1, :].reshape(m_total, n_dim)
        residuals = jnp.stack([gate, proj], axis=-2).reshape(
            out_shape[:-1] + (2, n_dim)
        )
      else:
        residuals = None

      return out, residuals

    fn = self._with_vmap(fn, fallback_to_sequential=False)
    return fn(x, weights)
