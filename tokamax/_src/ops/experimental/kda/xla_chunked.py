# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Pure-JAX/XLA chunked implementation of Kimi Delta Attention."""

import dataclasses
from typing import Annotated, Any, ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.kda import base
from tokamax._src.ops.experimental.kda.cp_utils import ContextParallelMetadataArg
from tokamax._src.ops.experimental.kda.xla_chunked_bwd_kernel import (
    chunk_kda_bwd,
)
from tokamax._src.ops.experimental.kda.xla_chunked_fwd_kernel import (
    chunk_kda_fwd,
)
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Execution configuration for pure-JAX chunked KDA."""

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 64
  safe_gate: bool | None = None


def _resolve_safe_gate(
    config: Config,
    *,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
) -> bool:
  if config.safe_gate is not None:
    return config.safe_gate
  return not use_gate_in_kernel or lower_bound is not None


def _check_supported(
    query: jax.Array,
    *,
    segment_ids: jax.Array | None,
    context_parallel_metadata: ContextParallelMetadataArg,
    chunk_size: int,
) -> None:
  if chunk_size != 64:
    raise NotImplementedError("`xla_chunked` currently supports chunk_size=64.")
  if segment_ids is None and query.shape[2] % chunk_size:
    raise NotImplementedError(
        "`xla_chunked` requires fixed-length T to be divisible by "
        f"chunk_size; got T={query.shape[2]}, chunk_size={chunk_size}."
    )
  if context_parallel_metadata is not None and context_parallel_metadata.is_cp_enabled:
    raise NotImplementedError(
        "`xla_chunked` does not currently support context parallelism."
    )


@dataclasses.dataclass(frozen=True)
class XlaChunkedKimiDeltaAttention(base.KimiDeltaAttention[Config, Any]):
  """Pure-JAX chunked KDA backend lowered by XLA on the active device."""

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", XlaChunkedKimiDeltaAttentionVjp())

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba
    return {Config(chunk_size=64)}

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      query: Float[Array, "H B T K"],
      key: Float[Array, "H B T K"],
      value: Float[Array, "H B T V"],
      gate: Float[Array, "H B T K"],
      beta: Float[Array, "H B T"],
      *,
      a_log: Float[Array, "H"] | None,
      delta_time_bias: Float[Array, "H*K"] | None,
      scale: float,
      initial_state: Float[Array, "B N H K V"] | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      segment_ids: Int[Array, "B T"] | None,
      lower_bound: float | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      max_num_segments: int | None,
      return_residuals: bool,
      config: Config,
  ) -> tuple[base.Output, None]:
    del return_residuals
    _check_supported(
        query,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
        chunk_size=config.chunk_size,
    )
    base._validate_beta(beta)  # pylint: disable=protected-access
    output = chunk_kda_fwd(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        lower_bound=lower_bound,
        max_num_segments=max_num_segments,
        chunk_size=config.chunk_size,
        safe_gate=_resolve_safe_gate(
            config,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
        ),
    )
    return output, None


@dataclasses.dataclass(frozen=True, kw_only=True)
class XlaChunkedKimiDeltaAttentionVjp(op.Op[Any, dict[str, Any], None, Config, Any]):
  """VJP adapter that lowers the pure-JAX chunked backward separately."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba
    return {Config(chunk_size=64)}

  def _fwd(
      self,
      residuals: None,
      out: base.Output,
      dout: base.Output,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      gate: jax.Array,
      beta: jax.Array,
      *,
      a_log: jax.Array | None,
      delta_time_bias: jax.Array | None,
      scale: float,
      initial_state: jax.Array | None,
      output_final_state: bool,
      use_qk_l2norm: bool,
      use_gate_in_kernel: bool,
      segment_ids: jax.Array | None,
      lower_bound: float | None,
      context_parallel_metadata: ContextParallelMetadataArg,
      max_num_segments: int | None,
      return_residuals: bool,
      config: Config,
  ) -> tuple[dict[str, jax.Array], None]:
    del residuals, out, return_residuals
    _check_supported(
        query,
        segment_ids=segment_ids,
        context_parallel_metadata=context_parallel_metadata,
        chunk_size=config.chunk_size,
    )
    safe_gate = _resolve_safe_gate(
        config,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )
    dq, dk, dv, dg, db, da, dbias, dh0 = chunk_kda_bwd(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        lower_bound=lower_bound,
        max_num_segments=max_num_segments,
        chunk_size=config.chunk_size,
        safe_gate=safe_gate,
        output_cotangent=dout,
    )
    grads = {
        "query": dq,
        "key": dk,
        "value": dv,
        "gate": dg,
        "beta": db,
    }
    if a_log is not None:
      grads["a_log"] = da if da is not None else jnp.zeros_like(a_log)
    if delta_time_bias is not None:
      grads["delta_time_bias"] = (
          dbias if dbias is not None else jnp.zeros_like(delta_time_bias)
      )
    if initial_state is not None:
      grads["initial_state"] = dh0 if dh0 is not None else jnp.zeros_like(initial_state)
    if segment_ids is not None:
      grads["segment_ids"] = jnp.zeros_like(segment_ids)
    return grads, None


__all__ = ["Config", "XlaChunkedKimiDeltaAttention"]
