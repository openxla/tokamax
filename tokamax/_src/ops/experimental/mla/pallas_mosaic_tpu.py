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
"""Pallas/Mosaic TPU Multi-Head Latent Attention."""

from collections.abc import Mapping
import dataclasses
import itertools
from typing import Annotated, Any, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla import utils
from tokamax._src.ops.experimental.mla.kernel import bref_override
from tokamax._src.ops.experimental.mla.kernel import configs as mla_configs
from tokamax._src.ops.experimental.mla.kernel import mla_wrapper
from tokamax._src.ops.experimental.mla.kernel import schedule as mla_schedule

# Heuristic KV block sizes, in tokens. With the 1024-token pages used in serving
# these are 2 pages per decode block and 1 per prefill/mixed block.
_DECODE_KV_TOKENS_PER_BLOCK = 2048
_PREFILL_KV_TOKENS_PER_BLOCK = 1024
# At or below this many tokens per device the decode pass packs 2 lanes per
# step rather than 4, as there are too few sequences to fill 4.
_SMALL_DECODE_BUCKET_TOKENS = 4

# Autotuning search space. The decode values bracket a sweep that found nothing
# above the noise floor around the heuristics; the prefill/mixed ones bracket
# the 16-query blocks used in serving. Buffering depth is only swept for
# decode; the prefill/mixed sweep keeps the heuristic's.
_DECODE_KV_PAGES_PER_BLOCK = (1, 2, 3, 4)
_DECODE_BATCH_SIZES = (1, 2, 4, 8)
_DECODE_N_BUFFERS = (2, 3, 4)
_PREFILL_KV_PAGES_PER_BLOCK = (1, 2, 3, 4)
_PREFILL_QUERIES_PER_BLOCK = (8, 16, 32)
_PREFILL_Q_SPLITS = (1, 2, 4, 8, 16)
# Fraction of VMEM a candidate's estimated footprint may use.
_VMEM_HEADROOM = 0.85
# The key's stand-in page count; see `_get_autotuning_cache_key`.
_CACHE_KEY_NUM_PAGES = 1

Schedules = Mapping[mla_configs.MlaCase, mla_schedule.MlaSchedule]


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Tunable parameters of the MLA kernel.

  The decode pass (`bq_sz == 1`) and the prefill/mixed passes are tuned
  separately; the prefill and mixed passes share their block sizes.

  Attributes:
    decode_num_kv_pages_per_block: KV pages per decode block.
    decode_batch_size: Decode lanes packed into one kernel step.
    decode_n_buffer: Pipelining depth of the decode Q and KV buffers.
    prefill_num_kv_pages_per_block: KV pages per prefill/mixed block.
    prefill_num_queries_per_block: Query tokens per prefill/mixed block.
    prefill_q_split: Sub-chunks each prefill/mixed query block is split into;
      must divide `prefill_num_queries_per_block`.
    prefill_n_buffer: Pipelining depth of the prefill/mixed Q and KV buffers.
    vmem_limit_bytes: Scoped VMEM limit, or `None` for all of VMEM.
  """

  decode_num_kv_pages_per_block: Annotated[int, pydantic.Field(gt=0)] = 2
  decode_batch_size: Annotated[int, pydantic.Field(gt=0)] = 4
  decode_n_buffer: Annotated[int, pydantic.Field(ge=2)] = 2
  prefill_num_kv_pages_per_block: Annotated[int, pydantic.Field(gt=0)] = 1
  prefill_num_queries_per_block: Annotated[int, pydantic.Field(gt=0)] = 16
  prefill_q_split: Annotated[int, pydantic.Field(gt=0)] = 1
  prefill_n_buffer: Annotated[int, pydantic.Field(ge=2)] = 2
  vmem_limit_bytes: Annotated[int, pydantic.Field(gt=0)] | None = None

  def __post_init__(self):
    if self.prefill_num_queries_per_block % self.prefill_q_split:
      raise ValueError(
          f"{self.prefill_q_split=} must divide"
          f" {self.prefill_num_queries_per_block=}."
      )

  def block_sizes(
      self, page_size: int
  ) -> tuple[mla_configs.BlockSizes, mla_configs.BlockSizes]:
    """Returns the (decode, prefill/mixed) kernel block sizes."""
    decode = mla_configs.BlockSizes(
        bq_sz=1,
        bq_c_sz=1,
        bkv_sz=self.decode_num_kv_pages_per_block * page_size,
        batch_size=self.decode_batch_size,
        n_buffer=self.decode_n_buffer,
    )
    prefill = mla_configs.BlockSizes(
        bq_sz=self.prefill_num_queries_per_block,
        bq_c_sz=self.prefill_num_queries_per_block // self.prefill_q_split,
        bkv_sz=self.prefill_num_kv_pages_per_block * page_size,
        batch_size=1,
        n_buffer=self.prefill_n_buffer,
    )
    return decode, prefill


def _estimated_vmem_bytes(
    blocks: mla_configs.BlockSizes,
    *,
    num_heads: int,
    lkv_dim: int,
    r_dim: int,
    q_itemsize: int,
    kv_itemsize: int,
) -> int:
  """A rough upper estimate of one pass's VMEM footprint.

  Counts the buffers `create_allocs` makes, the m/l/acc scratch and the f32
  score and probability tiles of one query chunk. It only has to be good enough
  to drop candidates that cannot fit, sparing the autotuner their compiles.
  """
  lanes = 128
  heads = utils.align_to(num_heads, 4 // q_itemsize)  # Word packing.
  lkv = utils.align_to(lkv_dim, lanes)
  kv_dim = lkv + utils.align_to(r_dim, lanes)
  rows = blocks.batch_size * blocks.bq_sz * heads
  q = blocks.n_buffer * rows * kv_dim * q_itemsize
  o = 2 * rows * lkv * q_itemsize
  kv_slots = blocks.n_buffer + 1  # `MlaConfigs.kv_num_slots`.
  kv = kv_slots * blocks.batch_size * (blocks.bkv_sz + 2 * lanes) * kv_dim
  kv *= kv_itemsize
  scratch = blocks.bq_sz * heads * (lkv + 2 * lanes) * 4
  chunk_rows = blocks.batch_size * blocks.bq_c_sz * heads
  scores = 2 * chunk_rows * blocks.bkv_sz * 4
  return q + o + kv + scratch + scores


class PallasTpuMultiHeadLatentAttention(base.MultiHeadLatentAttention[Config]):
  """MLA kernel: KV cache with tokens along lanes, head-major queries.

  Logits are always computed in float32, so `s_dtype` is ignored.

  The kernel runs one pass per `distribution` band, each with a DMA schedule
  built by a separate kernel. A model can build the schedules once per step and
  pass them to every layer:

  ```python
  schedules = op.build_schedules(*args, **kwargs)
  for layer in layers:
    out, cache = op(*args, **kwargs, schedules=schedules)
  ```
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      ql_nope: Float[Array, "num_q_heads max_num_tokens actual_lkv_dim"],
      q_pe: Float[Array, "max_num_tokens num_q_heads actual_r_dim"],
      new_kv_c: Float[Array, "max_num_tokens actual_lkv_dim"],
      new_k_pe: Float[Array, "max_num_tokens actual_r_dim"],
      cache_kv: Float[Array, "total_num_pages aligned_kv_dim page_size"],
      kv_lens: Int[Array, "max_num_seqs"],
      page_indices: Int[Array, "num_page_indices"],
      cu_q_lens: Int[Array, "max_num_seqs_plus_1"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      s_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      schedules: base.OpaqueMetadata = None,
      debug_mode: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    del s_dtype, return_residuals  # Unused.
    assert config is not None, "Config must be specified."
    decode_blocks, prefill_blocks = config.block_sizes(cache_kv.shape[-1])
    out = mla_wrapper.mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=config.vmem_limit_bytes,
        schedules=schedules,
        debug_mode=debug_mode,
    )
    return out, None

  def build_schedules(self, *args, **kwargs) -> Schedules:
    """Builds the per-pass schedules for the given arguments.

    Takes the same arguments as the op. The schedules depend on the shapes,
    the config and the *values* of `cu_q_lens`, `kv_lens`, `page_indices` and
    `distribution` -- not on the activations or the cache contents. Build them
    once per step, pass them as `schedules=` to every layer of that step, and
    rebuild them whenever those values change.

    The config is resolved exactly as a call with the same arguments would
    resolve it, so the schedules match it. Only the configs (shapes, dtypes,
    block sizes) are checked when schedules are passed in: schedules built from
    different lengths or page indices with the same shapes are not detected and
    produce wrong results.

    Args:
      *args: Positional arguments, as for the op.
      **kwargs: Keyword arguments, as for the op.

    Returns:
      The schedules, keyed by pass.
    """
    ba = self.bind(*args, **kwargs)
    a = ba.arguments
    config = ba.default_config
    decode_blocks, prefill_blocks = config.block_sizes(a["cache_kv"].shape[-1])
    cfgs_by_mode = mla_wrapper.mode_configs(
        a["ql_nope"],
        a["q_pe"],
        a["cache_kv"],
        a["kv_lens"],
        a["page_indices"],
        sm_scale=a["sm_scale"],
        sliding_window=a["sliding_window"],
        soft_cap=a["soft_cap"],
        mask_value=a["mask_value"],
        q_scale=a["q_scale"],
        k_scale=a["k_scale"],
        v_scale=a["v_scale"],
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=config.vmem_limit_bytes,
    )
    return mla_wrapper.build_schedules(
        a["cu_q_lens"],
        a["kv_lens"],
        a["page_indices"],
        a["distribution"],
        cfgs_by_mode,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    """The configuration the end-to-end DeepSeek serving runs used."""
    page_size = ba.arguments["cache_kv"].shape[-1]
    num_tokens = ba.arguments["ql_nope"].shape[1]
    small_bucket = num_tokens <= _SMALL_DECODE_BUCKET_TOKENS
    return Config(
        decode_num_kv_pages_per_block=max(
            1, _DECODE_KV_TOKENS_PER_BLOCK // page_size
        ),
        decode_batch_size=2 if small_bucket else 4,
        prefill_num_kv_pages_per_block=max(
            1, _PREFILL_KV_TOKENS_PER_BLOCK // page_size
        ),
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return (
        device.platform == "tpu"
        and pltpu.get_tpu_info().generation >= 7
        and bref_override.SUPPORTS_IN_OUT_BUFFER_COUNT
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Any:
    """Normalizes away arguments that do not affect the best config.

    - The cache's page count: serving allocates the whole per-device HBM pool,
      whose size no arg spec can match, and the kernel only touches the pages
      `page_indices` names.
    - The float scales: they multiply the logits/output and do not change the
      kernel's cost.

    Note that array values (`kv_lens`, ...) never reach the key, only shapes.
    """
    arguments = dict(ba.arguments)
    cache_kv = arguments["cache_kv"]
    arguments["cache_kv"] = jax.ShapeDtypeStruct(
        (_CACHE_KEY_NUM_PAGES, *cache_kv.shape[1:]), cache_kv.dtype
    )
    arguments["sm_scale"] = 1.0
    for name in ("q_scale", "k_scale", "v_scale"):
      arguments[name] = None
    ba = dataclasses.replace(ba, arguments=arguments)
    return super()._get_autotuning_cache_key(ba)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    """Varies the decode and the prefill/mixed knobs, each around the heuristic.

    The two passes are tuned separately rather than jointly: one sweep varies
    the decode knobs with the prefill ones at the heuristic, the other varies
    the prefill knobs with the decode ones at the heuristic, and the space is
    their union -- tens of configs rather than the product. Both sweeps run for
    every shape: like the cache key, the space depends only on shapes, never on
    the `distribution` values a production call would carry.

    Candidates are pruned to blocks whose estimated VMEM footprint fits. The
    manual autotune target then also times the merge of the two sweeps'
    winners: the passes share no knobs, so the best of each makes the best
    config.
    """
    a = ba.arguments
    heuristic = self._get_heuristics_config(ba)
    ql_nope, cache_kv = a["ql_nope"], a["cache_kv"]
    page_size = cache_kv.shape[-1]
    vmem_capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    dims = dict(
        num_heads=ql_nope.shape[0],
        lkv_dim=ql_nope.shape[-1],
        r_dim=a["q_pe"].shape[-1],
        q_itemsize=jnp.dtype(ql_nope.dtype).itemsize,
        kv_itemsize=jnp.dtype(cache_kv.dtype).itemsize,
    )

    def fits(config: Config) -> bool:
      decode, prefill = config.block_sizes(page_size)
      limit = config.vmem_limit_bytes or vmem_capacity
      return all(
          _estimated_vmem_bytes(blocks, **dims) <= _VMEM_HEADROOM * limit
          for blocks in (decode, prefill)
      )

    configs = set()
    for pages, batch, n_buffer in itertools.product(
        _DECODE_KV_PAGES_PER_BLOCK,
        _DECODE_BATCH_SIZES,
        _DECODE_N_BUFFERS,
    ):
      configs.add(
          dataclasses.replace(
              heuristic,
              decode_num_kv_pages_per_block=pages,
              decode_batch_size=batch,
              decode_n_buffer=n_buffer,
          )
      )
    for pages, queries, q_split in itertools.product(
        _PREFILL_KV_PAGES_PER_BLOCK,
        _PREFILL_QUERIES_PER_BLOCK,
        _PREFILL_Q_SPLITS,
    ):
      if queries % q_split:
        continue
      configs.add(
          dataclasses.replace(
              heuristic,
              prefill_num_kv_pages_per_block=pages,
              prefill_num_queries_per_block=queries,
              prefill_q_split=q_split,
          )
      )
    return {config for config in configs if fits(config)}
