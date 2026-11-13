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
"""Tokamax operator wrapper for Pallas Mosaic TPU Lightning Indexer."""

from collections.abc import Sequence
import dataclasses
from typing import Any, override

from tokamax._src.ops.lightning_indexer import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.lightning_indexer import torch_base

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


class _PallasMosaicTpuLightningIndexer(
    torch_base._LightningIndexer[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax Lightning Indexer Op wrapped for PyTorch."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuLightningIndexer()
    self.jax_op_name = "pallas_mosaic_tpu_lightning_indexer"

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, ...] | None:
    """Deconstructs a Config object into a flat 13-int tuple for jax_op."""
    if config is None:
      return None
    if isinstance(config, (tuple, list)):
      return tuple(int(x) for x in config)
    assert len(dataclasses.fields(config)) == 7, (
        f"Expected 7 fields in Config, got {len(dataclasses.fields(config))}."
    )
    kv_pages = _to_triple(config.num_kv_pages_per_block)
    queries = _to_triple(config.num_queries_per_block)
    buf_cnt = _to_triple(config.buffer_count)
    vmem_limit = int(config.vmem_limit_bytes)
    decode_bs = int(config.decode_req_batch_size)
    early_exit = int(bool(config.enable_early_exit))
    chunk_tokens = (
        -1 if config.chunk_tokens is None else int(config.chunk_tokens)
    )
    return (
        *kv_pages,
        *queries,
        *buf_cnt,
        vmem_limit,
        decode_bs,
        early_exit,
        chunk_tokens,
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed 13-int tuple back into a Config."""
    if not config_parts or config_parts[0] is None:
      raise ValueError("Forward config cannot be None.")
    flat = config_parts[0]
    if isinstance(flat, Config):
      return flat
    assert len(flat) == 13, f"Invalid config: {flat!r}"
    return Config(
        num_kv_pages_per_block=(flat[0], flat[1], flat[2]),
        num_queries_per_block=(flat[3], flat[4], flat[5]),
        buffer_count=(flat[6], flat[7], flat[8]),
        vmem_limit_bytes=int(flat[9]),
        decode_req_batch_size=int(flat[10]),
        enable_early_exit=bool(flat[11]),
        chunk_tokens=None if flat[12] <= 0 else int(flat[12]),
    )


# Singleton instance of Pallas Mosaic TPU LightningIndexer.
PallasMosaicTpuLightningIndexer = _PallasMosaicTpuLightningIndexer()  # pylint: disable=invalid-name
PallasTpuLightningIndexer = PallasMosaicTpuLightningIndexer  # pylint: disable=invalid-name
