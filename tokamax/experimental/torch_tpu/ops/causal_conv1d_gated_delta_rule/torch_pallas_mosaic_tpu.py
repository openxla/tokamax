# Copyright 2026 Google LLC
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
"""Tokamax operator wrapper for Pallas Mosaic TPU Causal Conv1D Gated Delta Rule."""

import dataclasses
from typing import Any
import jax
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.causal_conv1d_gated_delta_rule import torch_base
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuCausalConv1dGatedDeltaRule(
    torch_base._CausalConv1dGatedDeltaRule[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Causal Conv1D Gated Delta Rule Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule()
    )
    self.jax_op_name = "pallas_mosaic_tpu_causal_conv1d_gated_delta_rule"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def get_bound_args(self, *args: Any, **kwargs: Any) -> Any:
    norm_kwargs = dict(kwargs)
    if "compute_precision" in norm_kwargs:
      norm_kwargs["compute_precision"] = torch_utils.str_to_jax_dtype(
          norm_kwargs["compute_precision"]
      )
    return super().get_bound_args(*args, **norm_kwargs)

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, int] | None:
    """Converts Config into a 2-int tuple for jax_op."""
    if config is None:
      return None
    decode_tile_size, mixed_tile_size = (
        tuple(config)
        if isinstance(config, (tuple, list))
        else dataclasses.astuple(config)
    )
    if decode_tile_size is None or mixed_tile_size is None:
      return None
    return decode_tile_size, mixed_tile_size

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed tuple back into a Config."""
    config = config_parts[0]
    assert config is not None, "Config not set."
    decode_tile_size, mixed_tile_size = config
    return Config(
        decode_tile_size=decode_tile_size,
        mixed_tile_size=mixed_tile_size,
    )

  @override
  def _fake_impl(
      self,
      qkv: torch.Tensor,
      b: torch.Tensor,
      a: torch.Tensor,
      conv_state: torch.Tensor,
      recurrent_state: torch.Tensor,
      conv_weight: torch.Tensor,
      conv_bias: torch.Tensor | None,
      a_log: torch.Tensor,
      dt_bias: torch.Tensor,
      query_start_loc: torch.Tensor,
      state_indices: torch.Tensor,
      distribution: torch.Tensor,
      seq_lens: torch.Tensor,
      read_state_indices: torch.Tensor | None = None,
      read_offsets: torch.Tensor | None = None,
      *,
      n_kq: int = 0,
      n_v: int = 0,
      d_k: int = 0,
      d_v: int = 0,
      kernel_size: int = 0,
      num_spec_tokens: int = 0,
      return_residuals: bool = False,
      zero_initialize_out: bool = True,
      compute_precision: str | None = "float32",
      decode_tile_size: int | None = None,
      mixed_tile_size: int | None = None,
      config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del zero_initialize_out, compute_precision, decode_tile_size
    del mixed_tile_size, config
    return super()._fake_impl(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        read_state_indices=read_state_indices,
        read_offsets=read_offsets,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        num_spec_tokens=num_spec_tokens,
        return_residuals=return_residuals,
    )

  @override
  def __call__(
      self,
      qkv: torch.Tensor,
      b: torch.Tensor,
      a: torch.Tensor,
      conv_state: torch.Tensor,
      recurrent_state: torch.Tensor,
      conv_weight: torch.Tensor,
      conv_bias: torch.Tensor | None,
      a_log: torch.Tensor,
      dt_bias: torch.Tensor,
      query_start_loc: torch.Tensor,
      state_indices: torch.Tensor,
      distribution: torch.Tensor,
      seq_lens: torch.Tensor,
      read_state_indices: torch.Tensor | None = None,
      read_offsets: torch.Tensor | None = None,
      *,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
      num_spec_tokens: int = 0,
      zero_initialize_out: bool = True,
      compute_precision: Any = jnp.float32.dtype,
      decode_tile_size: int | None = None,
      mixed_tile_size: int | None = None,
      config: Config | None = None,
      return_residuals: bool = False,
  ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax was not"
        " set in the constructor."
    )
    new_conv_state, new_recurrent_state, out = self._torch_tokamax_op(
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
        return_residuals=return_residuals,
        zero_initialize_out=zero_initialize_out,
        compute_precision=torch_utils.dtype_to_str(compute_precision),
        decode_tile_size=decode_tile_size,
        mixed_tile_size=mixed_tile_size,
        config=self.deconstruct_config(config),
    )
    return (new_conv_state, new_recurrent_state), out

  @override
  def op_impl_call(
      self,
      qkv: jax.Array,
      b: jax.Array,
      a: jax.Array,
      conv_state: jax.Array,
      recurrent_state: jax.Array,
      conv_weight: jax.Array,
      conv_bias: jax.Array | None,
      a_log: jax.Array,
      dt_bias: jax.Array,
      query_start_loc: jax.Array,
      state_indices: jax.Array,
      distribution: jax.Array,
      seq_lens: jax.Array,
      read_state_indices: jax.Array | None = None,
      read_offsets: jax.Array | None = None,
      *,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
      num_spec_tokens: int = 0,
      return_residuals: bool = False,
      zero_initialize_out: bool = True,
      compute_precision: str | None = "float32",
      decode_tile_size: int | None = None,
      mixed_tile_size: int | None = None,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax was not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    precision = (
        torch_utils.str_to_jax_dtype(compute_precision) or jnp.float32.dtype
    )
    ((new_conv_state, new_recurrent_state), out), _ = (
        self.op_impl_jax._fwd(  # pylint: disable=protected-access
            qkv=qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            read_state_indices=read_state_indices,
            read_offsets=read_offsets,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
            zero_initialize_out=zero_initialize_out,
            compute_precision=precision,
            decode_tile_size=decode_tile_size,
            mixed_tile_size=mixed_tile_size,
            config=kernel_config,
            return_residuals=return_residuals,
        )
    )
    return new_conv_state, new_recurrent_state, out


# Singleton instance of Pallas Mosaic TPU Causal Conv1D Gated Delta Rule.
PallasMosaicTpuCausalConv1dGatedDeltaRule = (  # pylint: disable=invalid-name
    _PallasMosaicTpuCausalConv1dGatedDeltaRule()
)
