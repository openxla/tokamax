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
"""Tokamax operator wrapper for Pallas Mosaic TPU SparseMLA."""

import dataclasses
from typing import Any, override

import jax
from tokamax._src.ops.sparse_mla import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.sparse_mla import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuSparseMla(
    torch_base._SparseMla[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax SparseMLA Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuSparseMla()
    self.jax_op_name = "pallas_mosaic_tpu_sparse_mla"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, int, int] | None:
    """Deconstructs a Config object into a 3-int tuple for jax_op."""
    if config is None:
      return None
    assert len(dataclasses.fields(config)) == 3, (
        f"Expected 3 fields in Config, got {len(dataclasses.fields(config))}."
    )
    return (
        config.gather_and_attention_chunk_size,
        config.attention_kernel_batch_size,
        config.vmem_limit_bytes,
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed 3-int tuple back into a Config."""
    if not config_parts or config_parts[0] is None:
      raise ValueError("Forward config cannot be None.")
    flat = config_parts[0]
    if isinstance(flat, Config):
      return flat
    assert len(flat) == 3, f"Invalid config: {flat!r}"
    return Config(
        gather_and_attention_chunk_size=int(flat[0]),
        attention_kernel_batch_size=int(flat[1]),
        vmem_limit_bytes=int(flat[2]),
    )

  @override
  def __call__(
      self,
      q: torch.Tensor,
      cache_kv_nope: torch.Tensor,
      cache_kv_rope: torch.Tensor,
      topk_indices: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      attention_sinks: torch.Tensor,
      swa_accumution: torch.Tensor,
      swa_l: torch.Tensor,
      swa_m: torch.Tensor,
      sm_scale: float = 1.0,
      return_residuals: bool = False,
      configs: tuple[Config | None, Any] | None = None,
  ) -> torch.Tensor:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        q,
        cache_kv_nope,
        cache_kv_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        sm_scale=sm_scale,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      cache_kv_nope: jax.Array,
      cache_kv_rope: jax.Array,
      topk_indices: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      attention_sinks: jax.Array,
      swa_accumution: jax.Array,
      swa_l: jax.Array,
      swa_m: jax.Array,
      sm_scale: float = 1.0,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_cfg = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        cache_kv_nope,
        cache_kv_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        sm_scale=sm_scale,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    return out


# Singleton instance of Pallas Mosaic TPU SparseMla.
PallasMosaicTpuSparseMla = _PallasMosaicTpuSparseMla()  # pylint: disable=invalid-name
