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
"""Causal Conv1D Gated Delta Rule PyTorch Op API using reference impl."""

from typing import Any, TypeVar
import jax
from tokamax._src.ops.causal_conv1d_gated_delta_rule import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
from typing_extensions import override

_Config = TypeVar("_Config")


class _CausalConv1dGatedDeltaRule(torch_op.TorchOp[_Config]):
  """Causal Conv1D Gated Delta Rule PyTorch Op API using reference impl."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.CausalConv1dGatedDeltaRule()
    self.jax_op_name = "base_causal_conv1d_gated_delta_rule"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

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
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del b, a, conv_weight, conv_bias, a_log, dt_bias
    del query_start_loc, state_indices, distribution, seq_lens
    del read_state_indices, read_offsets, n_kq, d_k, kernel_size
    del num_spec_tokens, return_residuals
    new_conv_state = torch.empty_like(conv_state)
    new_recurrent_state = torch.empty_like(recurrent_state)
    out = qkv.new_empty((qkv.shape[0], n_v * d_v))
    return new_conv_state, new_recurrent_state, out

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
      config: _Config | None = None,
      return_residuals: bool = False,
  ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    del config
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
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax was not set"
        " in the constructor."
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
            return_residuals=return_residuals,
        )
    )
    return new_conv_state, new_recurrent_state, out


# Singleton instance of CausalConv1dGatedDeltaRule.
CausalConv1dGatedDeltaRule = (  # pylint: disable=invalid-name
    _CausalConv1dGatedDeltaRule[Any]()
)
