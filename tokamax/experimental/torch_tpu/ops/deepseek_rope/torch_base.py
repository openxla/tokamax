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
"""Base class for PyTorch interfaces to Tokamax DeepSeek-V4 RoPE operators."""

from collections.abc import Sequence
from typing import Any, TypeVar, cast, override

import jax
import jax.numpy as jnp
from tokamax._src.ops import op as jax_tokamax_op
from tokamax._src.ops.deepseek_rope import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch

_Config = TypeVar("_Config")


def _to_torch_quant_dtype(
    quant_dtype: str | None,
) -> torch.dtype:
  """Resolves `quant_dtype` to a `torch.dtype` (default `torch.float8_e4m3fn`)."""
  return torch_utils.to_torch_dtype(quant_dtype) or torch.float8_e4m3fn


def canonicalize_jax_quant_dtype(
    mode: str,
    quant_dtype: torch.dtype | str | None,
) -> jnp.dtype | None:
  """Canonicalizes `quant_dtype` to match `jax_base.DeepseekRope.bind`."""
  if mode != "rope_quant":
    return None
  if quant_dtype is None:
    return jnp.dtype(jnp.float8_e4m3fn)
  return torch_utils.to_jax_dtype(quant_dtype)


class _DeepseekRope(torch_op.TorchOp[_Config]):
  """DeepSeek-V4 RoPE PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.DeepseekRope()
    self.jax_op_name = "base_deepseek_rope"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def get_bound_args(
      self,
      *args: Any,
      **kwargs: Any,
  ) -> jax_tokamax_op.BoundArguments:
    # `TorchOp.get_bound_args` bypasses `jax_base.DeepseekRope.bind`, leaving
    # `quant_dtype` as a raw `torch.dtype`, `str`, or `None` instead of a
    # canonical `jnp.dtype`. We normalize it here so autotuning and heuristics
    # config lookups see the same argument types as the JAX op.
    ba = super().get_bound_args(*args, **kwargs)
    mode = ba.arguments.get("mode", "rope")
    arguments = dict(ba.arguments)
    arguments["quant_dtype"] = canonicalize_jax_quant_dtype(
        mode, arguments.get("quant_dtype")
    )
    return jax_tokamax_op.BoundArguments(ba.op, arguments)

  def _fake_impl(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      mode: str = "rope",
      inverse: bool = False,
      eps: float = 1e-6,
      quant_dtype: str | None = None,
      return_residuals: bool = False,
      config: Sequence[int] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del positions, cos_sin_cache, inverse, eps, return_residuals, config
    if mode == "rope_quant":
      q_dtype = _to_torch_quant_dtype(quant_dtype)
      return (
          x.new_empty(x.shape, dtype=q_dtype),
          x.new_empty(x.shape[:-1], dtype=torch.float32),
      )
    return (
        torch.empty_like(x),
        x.new_empty((), dtype=torch.float32),
    )

  @override
  def __call__(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      mode: str = "rope",
      inverse: bool = False,
      eps: float = 1e-6,
      quant_dtype: torch.dtype | str | None = None,
      return_residuals: bool = False,
      config: _Config | None = None,
      configs: tuple[Any, Any] | None = None,
  ) -> Any:
    del config, configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, scales = self._torch_tokamax_op(
        x,
        positions,
        cos_sin_cache,
        mode=mode,
        inverse=inverse,
        eps=eps,
        quant_dtype=torch_utils.dtype_to_str(quant_dtype),
        return_residuals=return_residuals,
        config=None,
    )
    return (out, scales) if mode == "rope_quant" else out

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      positions: jax.Array,
      cos_sin_cache: jax.Array,
      mode: str = "rope",
      inverse: bool = False,
      eps: float = 1e-6,
      quant_dtype: str | None = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        positions,
        cos_sin_cache,
        mode=cast(jax_base.Mode, mode),
        inverse=inverse,
        eps=eps,
        quant_dtype=canonicalize_jax_quant_dtype(mode, quant_dtype),
        return_residuals=return_residuals,
        config=None,
    )
    if mode == "rope_quant":
      assert isinstance(out, tuple)
      return out
    assert isinstance(out, jax.Array)
    return out, jnp.empty((), dtype=jnp.float32)


# Singleton instance of DeepseekRope.
DeepseekRope = _DeepseekRope[Any]()  # pylint: disable=invalid-name
