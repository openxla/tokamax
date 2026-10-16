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
"""Tokamax operator wrapper for Pallas Mosaic TPU MaskedDenseMLA."""

from collections.abc import Sequence
import dataclasses
from typing import Any, override

import jax
from tokamax._src.ops.masked_dense_mla import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.masked_dense_mla import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


def _to_triple(val: Sequence[int] | int) -> tuple[int, int, int]:
  """Normalizes an int or 3-element sequence into a 3-tuple of ints."""
  if isinstance(val, int) and not isinstance(val, bool):
    return (int(val), int(val), int(val))
  if isinstance(val, Sequence):
    vals = tuple(int(x) for x in val)
    if len(vals) == 3:
      return (vals[0], vals[1], vals[2])
  raise ValueError(f"Expected 3-tuple or int, got {val!r}.")


class _PallasMosaicTpuMaskedDenseMla(
    torch_base._MaskedDenseMla[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax MaskedDenseMLA Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuMaskedDenseMla()
    self.jax_op_name = "pallas_mosaic_tpu_masked_dense_mla"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    """Deconstructs a Config object into a flat 7-int tuple for jax_op."""
    if config is None:
      return None
    assert (
        len(dataclasses.fields(config)) == 3
    ), f"Expected 3 fields in Config, got {len(dataclasses.fields(config))}."
    kv_pages = _to_triple(config.num_kv_pages_per_block)
    queries = _to_triple(config.num_queries_per_block)
    vmem_limit = int(config.vmem_limit_bytes)
    return (*kv_pages, *queries, vmem_limit)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed 7-int tuple back into a Config."""
    if not config_parts or config_parts[0] is None:
      raise ValueError("Forward config cannot be None.")
    flat = config_parts[0]
    if isinstance(flat, Config):
      return flat
    assert len(flat) == 7, f"Invalid config: {flat!r}"
    return Config(
        num_kv_pages_per_block=(int(flat[0]), int(flat[1]), int(flat[2])),
        num_queries_per_block=(int(flat[3]), int(flat[4]), int(flat[5])),
        vmem_limit_bytes=int(flat[6]),
    )

  @override
  def __call__(
      self,
      q: torch.Tensor,
      cache_kv_nope: torch.Tensor,
      cache_kv_rope: torch.Tensor,
      kv_lens: torch.Tensor,
      topk_indices: torch.Tensor,
      page_indices: torch.Tensor,
      cu_q_lens: torch.Tensor,
      distribution: torch.Tensor,
      sm_scale: float = 1.0,
      k_scale: float = 1.0,
      mask_value: float | None = None,
      max_kv_len: int | None = None,
      chunk_prefill_size: int | None = None,
      sequence_start: torch.Tensor | None = None,
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
        kv_lens,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        max_kv_len=max_kv_len,
        chunk_prefill_size=chunk_prefill_size,
        sequence_start=sequence_start,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      cache_kv_nope: jax.Array,
      cache_kv_rope: jax.Array,
      kv_lens: jax.Array,
      topk_indices: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      sm_scale: float = 1.0,
      k_scale: float = 1.0,
      mask_value: float | None = None,
      max_kv_len: int | None = None,
      chunk_prefill_size: int | None = None,
      sequence_start: jax.Array | None = None,
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
        kv_lens,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        max_kv_len=max_kv_len,
        chunk_prefill_size=chunk_prefill_size,
        sequence_start=sequence_start,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    return out


# Singleton instance of Pallas Mosaic TPU MaskedDenseMla.
PallasMosaicTpuMaskedDenseMla = _PallasMosaicTpuMaskedDenseMla()  # pylint: disable=invalid-name
