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
"""TopK PyTorch Op API using reference implementation."""

from typing import Any, TypeVar
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.topk import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
import torch
from typing_extensions import override

_Config = TypeVar("_Config")


class _TopK(torch_op.TorchOp[_Config]):
  """TopK PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.TopK()
    self.jax_op_name = "base_topk"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      scores: torch.Tensor,
      k: int,
      row_lengths: torch.Tensor | None = None,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del row_lengths, return_residuals, config
    out_shape = (scores.shape[0], k)
    indices = scores.new_empty(out_shape, dtype=torch.int32)
    scores_bits = (
        scores.new_empty(out_shape, dtype=torch.int32)
        if return_scores
        else scores.new_empty((), dtype=torch.int32)
    )
    return indices, scores_bits

  @override
  def __call__(
      self,
      scores: torch.Tensor,
      k: int,
      row_lengths: torch.Tensor | None = None,
      return_scores: bool = False,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    indices, scores_bits = self._torch_tokamax_op(
        scores,
        k,
        row_lengths,
        return_scores=return_scores,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )
    if return_scores:
      return indices, scores_bits
    return indices, None

  @override
  def op_impl_call(
      self,
      scores: jax.Array,
      k: int,
      row_lengths: jax.Array | None = None,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        scores,
        k,
        row_lengths,
        return_scores=return_scores,
        return_residuals=return_residuals,
        config=None,
    )
    if isinstance(out, tuple):
      return out
    return out, jnp.empty((), dtype=jnp.int32)


# Singleton instance of TopK.
TopK = _TopK[Any]()
