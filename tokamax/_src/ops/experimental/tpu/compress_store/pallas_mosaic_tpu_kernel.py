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
"""TensorCore Pallas/Mosaic kernel for the DeepSeek-V4 compress-and-store.

For each boundary token, the kernel DMAs the state-cache pages covering its
window into VMEM, softmax-pools the window, applies RMSNorm and interleaved
RoPE, packs the record (FP8 for CSA and the indexer, bf16 for HCA) and stores
it in place into the compressed KV cache (and, for CSA, the RoPE cache). See
`reference` for the modes and cache layouts, and `config` for the
geometry.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.compress_store import buffered_ref
from tokamax._src.ops.experimental.tpu.compress_store import compute
from tokamax._src.ops.experimental.tpu.compress_store import config

# The token tile upstream vllm-torchtpu hardcodes.
DEFAULT_TILE_N = 4


def inner_kernel(
    cos_sin_vmem,
    out_vmem,
    page_buffer_vmem,
    rope_out_vmem,
    positions_ref,
    rms_weight_vmem_ref,
    window_vmem,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
    *,
    cfgs: config.Configs,
):
  """Pipeline body: compresses, normalizes and packs one tile of tokens."""
  tile_n = cfgs.tile_sizes.tile_n
  window = cfgs.window
  head_tiles = cfgs.head_tiles

  pid = pl.program_id(0)
  global_idx = pid * tile_n

  # Per-token window start = position - window + 1
  start_list = []
  for i in range(tile_n):
    idx = global_idx + i
    safe_idx = jax.lax.select(idx < cfgs.dims.size_n, idx, 0)
    start_list.append(positions_ref[safe_idx] - window + 1)
  start = jnp.stack(start_list)  # (tile_n,)

  window_u8 = window_vmem.bitcast(jnp.uint8).reshape(
      2, tile_n, window, head_tiles, 4, 128
  )
  kv_window_u8 = window_u8.at[0]
  score_window_u8 = window_u8.at[1]

  compute.gather_from_page_buffer(
      page_buffer=page_buffer_vmem,
      positions_ref=positions_ref,
      kv_window_u8=kv_window_u8,
      score_window_u8=score_window_u8,
      global_idx=global_idx,
      num_tokens=cfgs.dims.size_n,
      window=window,
      block_size=cfgs.state_block_size,
      pages_to_buffer_per_token=cfgs.pages_to_buffer_per_token,
      field_rows=cfgs.field_rows,
      state_rows_per_token=cfgs.state_rows_per_token,
      overlap=cfgs.dims.overlap,
      is_indexer=cfgs.dims.mode == config.Mode.CSA_INDEXER,
  )

  kv_val = window_vmem.at[0][...]  # (tile_n, window, head_tiles, 128)
  scores_val = window_vmem.at[1][...]  # (tile_n, window, head_tiles, 128)
  rms_weight_tiled = rms_weight_vmem_ref[...].astype(
      jnp.float32
  )  # (head_tiles, 128)

  # --- windowed softmax ---
  curr_pos = start[:, None] + jnp.arange(window)[None, :]  # (tile_n, window)
  mask = curr_pos >= 0  # (tile_n, window)
  mask_float = mask.astype(scores_val.dtype)
  mask_float_reshaped = mask_float[:, :, None, None]  # (tile_n, window, 1, 1)
  neg_inf = jnp.array(-jnp.inf, dtype=scores_val.dtype)
  masked_scores = jnp.where(mask_float_reshaped > 0.5, scores_val, neg_inf)
  weights = jax.nn.softmax(masked_scores, axis=1)
  kv_val = jnp.where(mask_float_reshaped > 0.5, kv_val, 0.0)
  compressed = jnp.sum(weights * kv_val, axis=1)  # (tile_n, head_tiles, 128)

  # --- rms norm ---
  variance = jnp.mean(jnp.square(compressed), axis=(1, 2), keepdims=True)
  normed = (
      compressed
      * jax.lax.rsqrt(variance + cfgs.dims.rms_eps)
      * rms_weight_tiled[None, :, :]
  )  # (tile_n, head_tiles, 128)

  # --- rope ---
  rope_ropped = None
  if cfgs.dims.has_rope:
    rope_slot = cfgs.rope_slot
    rope_val = normed[:, rope_slot : rope_slot + 1]

    cos_sin = cos_sin_vmem[...][:, None, :]
    cos_val = cos_sin[:, :, : cfgs.half_rope]
    sin_val = cos_sin[:, :, cfgs.half_rope :]

    rope_ropped = compute.interleaved_rope_vector(rope_val, cos_val, sin_val)
    if head_tiles > 1:
      normed = jnp.concatenate([normed[:, :rope_slot], rope_ropped], axis=1)
    else:
      normed = rope_ropped

  # --- pack + store ---
  if cfgs.dims.is_quantized:
    if cfgs.dims.mode == config.Mode.CSA:
      # CSA's reader wants the scales per lane, not per block.
      q, scale = compute.quantize_fp8_lane_periodic(
          normed, cfgs.dims.quant_block, cfgs.nope_store_dim
      )
      nope_val_padded = compute.pack_nope_lane_periodic(
          q,
          scale,
          cfgs.nope_store_dim,
          cfgs.record_bytes,
          cfgs.last_dim_size,
      )
    else:
      q, scale = compute.quantize_fp8_tiled(normed, cfgs.dims.quant_block)
      nope_val_padded = compute.pack_nope_tiled(
          q,
          scale,
          cfgs.nope_store_dim,
          cfgs.dims.quant_block,
          nope_width_bytes=cfgs.record_bytes,
          last_dim_size=cfgs.last_dim_size,
      )
    if cfgs.dims.mode == config.Mode.CSA_INDEXER:
      kv_slots = []
      for i in range(tile_n):
        kv_slots.append(kv_slot_mapping_ref[global_idx + i])
      kv_slots = jnp.stack(kv_slots)

      for n in range(tile_n):
        is_first = is_first_mask_ref[global_idx + n]

        @pl.when(is_first)
        def _merge_nope():
          # only applicable to csa-indexer
          # out_vmem shape: (tile_n, 1, 4, 256)
          slots_val = out_vmem[n, 0]
          out_vmem[n, 0] = compute.merge_slot_updates(
              slots_val,
              kv_slots,
              nope_val_padded,
              n,
          )

    else:
      out_vmem[:, 0] = nope_val_padded
    if cfgs.dims.has_rope_cache:
      # RoPE row of 4 tokens, token j in words [32 j, 32 j + 32) (see
      # `csa_cache_layout`); rope_out_vmem holds each row-writer's row.
      words = compute.rope_words_tiled(rope_ropped, cfgs.dims.rope_head_dim)
      rope_lane = config.LANE - cfgs.dims.rope_head_dim
      kv_slots = []
      for i in range(tile_n):
        kv_slots.append(kv_slot_mapping_ref[global_idx + i])
      kv_slots = jnp.stack(kv_slots)
      for n in range(tile_n):
        is_first = is_first_mask_rope_ref[global_idx + n]

        @pl.when(is_first)
        def _merge_rope():
          row = pltpu.bitcast(rope_out_vmem[n], jnp.int32)  # (1, 128)
          rope_out_vmem[n] = pltpu.bitcast(
              compute.merge_rope_row(row, kv_slots, words, rope_lane, n),
              jnp.uint8,
          )

  else:
    # hca: bitcast bf16 -> uint8 and match the output block shape.
    out_vmem[...] = pltpu.bitcast(
        normed.astype(cfgs.dims.nope_dtype), jnp.uint8
    ).reshape(tile_n, cfgs.record_rows, cfgs.hbm_pack, 128)


def kernel_fn(
    # prefetched inputs (scalar)
    block_table_ref,
    positions_ref,
    token_to_req_indices_ref,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
    grid_size_ref,
    rms_weight_ref,
    # HBM inputs (dynamic access)
    cos_sin_cache_ref,
    cache_ref,
    rope_cache_ref,
    state_cache_ref,
    # outputs (aliased)
    _out_cache_ref,
    _out_rope_cache_ref,
    window_scratch_ref,
    *,
    cfgs: config.Configs,
    block_table_stride: int,
):
  """Pallas kernel entry point."""
  grid_size = grid_size_ref[...]
  allocs, in_specs, pipeline_args = buffered_ref.create_allocs_and_specs(
      cfgs=cfgs,
      cache_ref=cache_ref,
      rope_cache_ref=rope_cache_ref,
      state_cache_ref=state_cache_ref,
      cos_sin_cache_ref=cos_sin_cache_ref,
      positions_ref=positions_ref,
      block_table_ref=block_table_ref,
      block_table_stride=block_table_stride,
      token_to_req_indices_ref=token_to_req_indices_ref,
      kv_slot_mapping_ref=kv_slot_mapping_ref,
      is_first_mask_ref=is_first_mask_ref,
      is_first_mask_rope_ref=is_first_mask_rope_ref,
  )

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(inner_kernel, cfgs=cfgs),
      grid=(grid_size,),
      in_specs=in_specs,
      out_specs=[],
  )

  @pl.with_scoped(allocations=tuple(allocs))
  def _run(allocations):
    pipeline_func(
        *pipeline_args,
        scratches=(
            positions_ref,
            rms_weight_ref,
            window_scratch_ref,
            kv_slot_mapping_ref,
            is_first_mask_ref,
            is_first_mask_rope_ref,
        ),
        allocations=allocations,
    )

  _run()


def _select_mode(head_dim: int, overlap: bool) -> config.Mode:
  return config.select_mode(head_dim, overlap)


def derive_aliases(
    has_rope: bool, has_rope_cache: bool, num_scalar_prefetch: int
) -> dict[int, int]:
  """Returns the `pallas_call` input-output aliases of the caches."""
  cache_index = num_scalar_prefetch + 1 + int(has_rope)
  aliases = {cache_index: 0}
  if has_rope_cache:
    aliases[cache_index + 1] = 1
  return aliases


def compute_is_first_mask(kv_slot_mapping, tile_n, pack_factor=4):
  """Marks the first token of each tile to map to each physical HBM row.

  If multiple tokens in the same tile map to the same row, only the first one
  is responsible for writing the merged VMEM row buffer back to HBM. The
  subsequent tokens in the same tile that conflict will skip the HBM write to
  prevent overwriting each other's data and reduce memory traffic.

  Args:
    kv_slot_mapping: `[num_tokens]` compressed-KV slots, -1 for no slot.
    tile_n: Tokens per tile.
    pack_factor: Slots per physical HBM row.

  Returns:
    The `[num_tokens]` bool mask.
  """
  num_tokens = kv_slot_mapping.shape[0]
  pad_len = (tile_n - (num_tokens % tile_n)) % tile_n
  if pad_len > 0:
    kv_slots_padded = jnp.pad(kv_slot_mapping, (0, pad_len), constant_values=-1)
  else:
    kv_slots_padded = kv_slot_mapping

  total_tokens = kv_slots_padded.shape[0]
  num_tiles = total_tokens // tile_n
  kv_slots_tiled = kv_slots_padded.reshape(num_tiles, tile_n)

  row_idxs = kv_slots_tiled // pack_factor
  valid = kv_slots_tiled >= 0

  eq = row_idxs[:, None, :] == row_idxs[:, :, None]
  tril = jnp.tril(jnp.ones((tile_n, tile_n), dtype=bool), k=-1)
  conflict = eq & tril[None, :, :] & valid[:, None, :]
  has_conflict = jnp.any(conflict, axis=-1)
  is_first = valid & ~has_conflict

  return is_first.flatten()[:num_tokens]


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_table_stride",
        "state_block_size",
        "compress_ratio",
        "overlap",
        "quant_block",
        "rms_eps",
        "tile_n",
        "interpret",
        "name",
    ),
    donate_argnames=("cache", "rope_cache"),
)
def compress_norm_rope_store(
    cache: jax.Array,
    positions: jax.Array,
    block_table: jax.Array,  # [num_reqs * block_table_stride] int
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    rms_weight: jax.Array,
    *,
    block_table_stride: int,
    state_block_size: int,
    state_cache: jax.Array | None = None,
    rope_cache: jax.Array | None = None,
    cos_sin_cache: jax.Array | None = None,
    compress_ratio: int,
    overlap: bool,
    quant_block: int = 64,
    rms_eps: float = 1e-6,
    tile_n: int = DEFAULT_TILE_N,
    interpret: bool = False,
    name: str = "compress_norm_rope_store",
) -> tuple[jax.Array, jax.Array | None]:
  """Compresses, normalizes, applies RoPE and stores to cache.

  `cache` and `rope_cache` are donated and updated in place through
  `input_output_aliases`. Bounds checks are disabled, so every index below must
  be in bounds for every token of every tile the grid visits: the grid covers
  the tiles up to the last token with a non-negative `kv_slot_mapping` entry,
  and reads `positions`, `token_to_req_indices` and `kv_slot_mapping` for every
  token of those tiles.

  Args:
    cache: The compressed KV cache (see `reference`). Also hosts the state
      unless `state_cache` is given.
    positions: `[num_tokens]` int32 token positions.
    block_table: `[num_reqs * block_table_stride]` int32 state pages.
    token_to_req_indices: `[num_tokens]` int32 request of each token.
    kv_slot_mapping: `[num_tokens]` int32 compressed-KV slot of each token, -1
      to skip it. Only boundary tokens may have a slot.
    rms_weight: `[head_dim]` f32 RMSNorm weight.
    block_table_stride: Row stride of `block_table`.
    state_block_size: Token states per state page.
    state_cache: The separate f32 state array (HCA), or `None`.
    rope_cache: The CSA RoPE cache; required in CSA mode.
    cos_sin_cache: `[max_pos, rope_head_dim]` f32 RoPE `[cos | sin]` table.
    compress_ratio: Tokens compressed into one record.
    overlap: Whether windows overlap (CSA / indexer).
    quant_block: FP8 quantization block.
    rms_eps: RMSNorm epsilon.
    tile_n: Tokens per grid step; a multiple of 4 for CSA and the indexer.
    interpret: Whether to run the kernel in interpret mode.
    name: The kernel name.

  Returns:
    The updated `(cache, rope_cache)`; `rope_cache` is `None` outside CSA mode.
  """
  assert block_table.ndim == 1
  assert block_table.shape[0] % block_table_stride == 0

  num_tokens = positions.shape[0]
  head_dim = rms_weight.shape[0]
  rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0

  state_operand = (
      None if state_cache is None or state_cache is cache else state_cache
  )
  state_source = cache if state_operand is None else state_operand

  cfgs = config.Configs.make(
      _select_mode(head_dim, overlap),
      size_n=num_tokens,
      physical_page_size=cache.shape[1],
      state_physical_page_size=state_source.shape[1],
      state_block_size=state_block_size,
      rms_eps=rms_eps,
      tile_n=tile_n,
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      compress_ratio=compress_ratio,
      quant_block=quant_block,
  )

  if cfgs.dims.mode in (config.Mode.CSA, config.Mode.CSA_INDEXER):
    assert cfgs.tile_sizes.tile_n % 4 == 0, (
        f"tile_n must be a multiple of 4 for {cfgs.dims.mode.value}, "
        f"got {cfgs.tile_sizes.tile_n}"
    )

  if cfgs.dims.has_rope_cache and rope_cache is None:
    raise ValueError("rope_cache must be provided when has_rope_cache is True")

  rms_weight_reshaped = rms_weight.reshape(cfgs.head_tiles, 128)

  # Compute grid size dynamically based on the maximum index in kv_slot_mapping.
  valid = kv_slot_mapping >= 0
  indices = jnp.arange(kv_slot_mapping.shape[0])
  max_idx = jnp.max(jnp.where(valid, indices, -1))
  grid_size = jnp.where(
      max_idx >= 0, pl.cdiv(max_idx + 1, cfgs.tile_sizes.tile_n), 0
  )

  is_first_mask = compute_is_first_mask(
      kv_slot_mapping,
      cfgs.tile_sizes.tile_n,
      pack_factor=cfgs.tokens_in_second_minor,
  )
  is_first_mask_rope = compute_is_first_mask(
      kv_slot_mapping,
      cfgs.tile_sizes.tile_n,
      # `hbm_pack` (4) doubles as the number of RoPE tokens per cache row.
      pack_factor=cfgs.hbm_pack,
  )
  # Outer pallas_call operands, in call order. Optional operands are passed as
  # None to keep kernel_fn's argument positions fixed; pallas drops the Nones
  # when indexing, so alias indices count only the operands actually present.
  scalar_prefetch = (
      block_table,
      positions,
      token_to_req_indices,
      kv_slot_mapping,
      is_first_mask,
      is_first_mask_rope,
      grid_size,
  )
  cos_sin_operand = cos_sin_cache if cfgs.dims.has_rope else None
  rope_operand = rope_cache if cfgs.dims.has_rope_cache else None

  in_specs = (
      pl.BlockSpec(memory_space=pltpu.VMEM),  # rms_weight
      (
          pl.BlockSpec(memory_space=pltpu.HBM) if cfgs.dims.has_rope else None
      ),  # cos_sin
      pl.BlockSpec(memory_space=pltpu.HBM),  # cache
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if cfgs.dims.has_rope_cache
          else None
      ),  # rope_cache
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if state_operand is not None
          else None
      ),  # state_cache
  )
  out_specs = (
      pl.BlockSpec(memory_space=pltpu.HBM),  # cache
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if cfgs.dims.has_rope_cache
          else None
      ),  # rope_cache
  )
  out_shapes = (
      jax.ShapeDtypeStruct(cache.shape, cache.dtype),
      jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype)  # pyrefly: ignore[missing-attribute]
      if cfgs.dims.has_rope_cache
      else None,
  )

  aliases = derive_aliases(
      cfgs.dims.has_rope, cfgs.dims.has_rope_cache, len(scalar_prefetch)
  )

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=len(scalar_prefetch),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=[pltpu.VMEM(cfgs.window_shape(), jnp.float32)],
  )

  out_cache, out_rope_cache = pl.pallas_call(
      functools.partial(
          kernel_fn, cfgs=cfgs, block_table_stride=block_table_stride
      ),
      out_shape=out_shapes,
      grid_spec=grid_spec,
      input_output_aliases=aliases,
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
      interpret=interpret,
      name=name,
  )(
      *scalar_prefetch,
      rms_weight_reshaped,
      cos_sin_operand,
      cache,
      rope_operand,
      state_operand,
  )

  return out_cache, out_rope_cache
