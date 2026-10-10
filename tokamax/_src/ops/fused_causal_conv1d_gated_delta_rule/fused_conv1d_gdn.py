# Copyright 2026 Google LLC
# Copyright 2026 Rabdos AI
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
"""Fused Causal Conv1D Gated Delta Rule Tokamax Op for TPU."""

import dataclasses
from typing import ClassVar, Optional

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.ops import op
from tokamax._src.ops.causal_conv1d_gated_delta_rule import base
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import prefill_decode
from typing_extensions import override

_MIN_TPU_GENERATION = 6


@dataclasses.dataclass(frozen=True)
class Config:
  """Fused schedule options; see prefill_decode.fused_conv1d_gdn.

  zero_initialize_out affects fallback only; the fused path fills every row.
  compute_precision uses Tokamax's dtype serializer for cache/HLO metadata.
  """

  zero_initialize_out: bool = True
  compute_precision: pydantic_lib.NumpyDtype = jnp.dtype(jnp.float32)
  decode_tile_size: Optional[int] = None
  mixed_tile_size: Optional[int] = None

  def __post_init__(self):
    # Equal dtype classes, instances and strings must share a cache-key hash.
    """Canonicalize the compute dtype for stable serialized configuration."""
    object.__setattr__(
        self, "compute_precision", jnp.dtype(self.compute_precision)
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FusedCausalConv1dGatedDeltaRule(base.CausalConv1dGatedDeltaRule[Config]):
  """Tokamax adapter for fused GDN; unsupported layouts use the upstream
  wrapper.

  See prefill_decode.fused_conv1d_gdn for numerical and schedule contracts.
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def _fwd(
      self,
      qkv: jax.Array,
      b: jax.Array,
      a: jax.Array,
      conv_state: jax.Array,
      recurrent_state: jax.Array,
      conv_weight: jax.Array,
      conv_bias: Optional[jax.Array],
      a_log: jax.Array,
      dt_bias: jax.Array,
      query_start_loc: jax.Array,
      state_indices: jax.Array,
      distribution: jax.Array,
      seq_lens: jax.Array,
      read_state_indices: Optional[jax.Array] = None,
      read_offsets: Optional[jax.Array] = None,
      *,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
      num_spec_tokens: int = 0,
      config: Config = Config(),
      return_residuals: bool = False,
  ) -> tuple[tuple[tuple[jax.Array, jax.Array], jax.Array], None]:
    """Forward the call to the fused entry point, with no backward residual.

    Args:
      qkv: [tokens, 2 * n_kq * d_k + n_v * d_v] packed activations.
      b: [tokens, n_v] beta gate inputs; same dtype as qkv.
      a: [tokens, n_v] decay gate inputs; same dtype as qkv.
      conv_state: [slots, kernel_size - 1, QKV width] convolution cache.
      recurrent_state: [slots, n_v, d_k, d_v] recurrent cache.
      conv_weight: [QKV width, 1, kernel_size] convolution weights.
      conv_bias: Optional [QKV width] convolution bias.
      a_log: [n_v] log decay weights.
      dt_bias: [n_v] decay biases.
      query_start_loc: [requests + 1] int32 cumulative token offsets, starting
        at zero.
      state_indices: [requests] int32 write-cache slots; active slots must be
        distinct, nonzero, and in bounds.
      distribution: [3] int32 cumulative endpoints [decode_end, prefill_end,
        mixed_end].
      seq_lens: [requests] int32 sequence lengths including history and
        submitted tokens.
      read_state_indices: Optional [requests] initial-state slots; defaults to
        state_indices.
      read_offsets: Optional [requests] read-slot offsets; decode applies them
        and prefill uses base slots.
      n_kq: Number of query/key heads.
      n_v: Number of value heads; supported layouts group them evenly over Q/K
        heads.
      d_k: Key/query head dimension; the fused path requires 128.
      d_v: Value head dimension; the fused path requires 128.
      kernel_size: Positive static convolution window size.
      num_spec_tokens: Speculative draft-token count; nonzero values delegate to
        the upstream wrapper.
      config: Tokamax configuration carrying precision and tile hints.
      return_residuals: Interface flag; this forward-only adapter always returns
        no residual.

    Returns:
      (((new_conv, new_recurrent), output), None); the final None is the unused
      backward residual.
    """
    del return_residuals
    output = prefill_decode.fused_conv1d_gdn(
        qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        read_state_indices,
        read_offsets,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        num_spec_tokens=num_spec_tokens,
        zero_initialize_out=config.zero_initialize_out,
        compute_precision=config.compute_precision,
        decode_tile_size=config.decode_tile_size,
        mixed_tile_size=config.mixed_tile_size,
    )
    return output, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    """Return default config; lower-level planning selects shapes and tile
    sizes.

    Args:
      ba: Bound operation arguments; unused by the default configuration rule.

    Returns:
      Default Config; lower-level planning chooses eligible tiles.
    """
    del ba  # Tile selection uses input shapes and dtypes.
    return Config()

  @override
  def supported_on(self, device: jax.Device) -> bool:
    """Check the device platform and TPU generation in the current context.

    Args:
      device: Device whose platform is checked; generation comes from the
        current default or abstract device.

    Returns:
      True when device is a TPU and the current context reports generation 6
      or later; False for other platforms or failed hardware queries.
    """
    if device.platform != "tpu":
      return False
    try:
      return pltpu.get_tpu_info().generation >= _MIN_TPU_GENERATION
    except Exception:  # pylint: disable=broad-except
      return False
