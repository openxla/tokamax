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
"""SparseCore Pallas/Mosaic kernel for the DeepSeek-V4 CSA cache gather."""

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp

# DeepSeek-V4 CSA compressed-KV cache layout. Both caches are int32 arrays with
# 128 lanes: the NoPE cache holds one 128-word row per token, and the RoPE cache
# packs 4 tokens per row, 32 words per token.
ROW_WORDS = 128
ROPE_WORDS = 32
# Upper bound on `num_streams`. Larger values hang the kernel on TPU7x.
MAX_NUM_STREAMS = 4


def main_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    valid_indices_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    nope_sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    num_streams: int,
    top_k: int,
):
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
  row_subchunk_size = num_simd_lanes
  row_chunk_size = row_subchunk_size * num_row_subchunks
  block_size = row_chunk_size * num_cores
  num_blocks = pl.cdiv(indices_hbm_ref.shape[0], block_size)

  core_index = lax.axis_index((core_axis_name, subcore_axis_name))

  nope_in_cols = nope_in_hbm_ref.shape[1]
  rope_in_cols = rope_in_hbm_ref.shape[1]
  # A `bf16[n, 64]` output is tiled to 128 lanes, so XLA pads it and it
  # costs twice its own bytes -- on this kernel's write and on the
  # consumer's read. The output is 128 lanes wide instead, pairing entry `i`
  # of each `top_k`-entry period with entry `i + top_k // 2`, which the
  # consumer splits apart with a lane slice and a row concatenate.
  rope_out_cols = rope_out_hbm_ref.shape[1] // 2

  def process_rope(gather_ref, out_ref, out_row_base=0):
    # bf16 output rows 2t and 2t + 1 are the low and high halves of int32
    # row t, so each output word pairs one rope value of an even entry
    # with the same value of the next entry. A cache rope word k holds
    # values k (low) and k + 32 (high).
    for t in range(num_simd_lanes // 2):
      even = gather_ref[pl.ds(2 * t, 1), :]
      odd = gather_ref[pl.ds(2 * t + 1, 1), :]
      row = pl.ds(out_row_base + t, 1)
      out_ref[row, pl.ds(0, rope_in_cols)] = jnp.bitwise_or(
          jnp.bitwise_and(even, 0xFFFF), jnp.left_shift(odd, 16)
      )
      out_ref[row, pl.ds(rope_in_cols, rope_in_cols)] = jnp.bitwise_or(
          lax.shift_right_logical(even, 16),
          jnp.bitwise_and(odd, jnp.int32(-65536)),  # 0xFFFF0000
      )

  def outer_pipeline(idx_ref, valid_ref):
    valid_indices = valid_ref[pl.ds(0, row_subchunk_size)][0]
    b = pl.program_id(0)
    out_row_base = (b * num_cores + core_index) * num_row_subchunks

    # Subchunk handled by stream `s` at inner step `r`. `num_streams`
    # independent `pl.Indirect` gathers run concurrently per step, keeping
    # several gather DMAs in flight to raise effective read bandwidth. A
    # step's subchunks are consecutive, so they lie in one half of a
    # period.
    def subchunk(r, s):
      return r * num_streams + s

    def idx_window(r, s):
      return idx_ref[
          pl.ds(subchunk(r, s) * row_subchunk_size, row_subchunk_size)
      ]

    # int32 rows produced per stream in the rope output (2 tokens per row).
    rope_rows_per_stream = row_subchunk_size // 2

    def _body(*refs):
      r = pl.program_id(0)
      nope_g = refs[0 * num_streams : 1 * num_streams]
      rope_g = refs[1 * num_streams : 2 * num_streams]
      rope_o = refs[2 * num_streams]

      # nope needs no vector work at all. The nope output keeps the
      # cache's raw per-token layout. The gathered row *is* the output
      # row.
      #
      # DMA the gather buffer straight to HBM rather than routing it
      # through a pipeline output buffer: staging it there costs a
      # VMEM->VMEM copy.
      nope_copies = []
      for s in range(num_streams):
        out_row = (out_row_base + subchunk(r, s)) * row_subchunk_size
        copy = pltpu.make_async_copy(
            nope_g[s],
            nope_out_hbm_ref.at[pl.ds(out_row, row_subchunk_size)],
            nope_sem.at[s],
        )
        copy.start()
        nope_copies.append(copy)

      for s in range(num_streams):
        process_rope(
            gather_ref=rope_g[s],
            out_ref=rope_o,
            out_row_base=s * rope_rows_per_stream,
        )

      # Wait for all nope DMAs to complete.
      for copy in nope_copies:
        copy.wait()

    # Multiple parallel `pl.Indirect` gathers hide random-access read
    # latency. The writes are contiguous and need no extra parallelism:
    # rope writes one block per step, and nope issues one DMA per stream
    # only because each stream has its own gather buffer.

    # nope: gather int32 row == index (1 int32 row per entry).
    nope_in_specs = tuple(
        pl.BlockSpec(
            (pl.Indirect(row_subchunk_size), nope_in_cols),
            lambda r, s=s: (idx_window(r, s), 0),
        )
        for s in range(num_streams)
    )
    # rope: gather the entry's own 32-word quarter row == index.
    rope_in_specs = tuple(
        pl.BlockSpec(
            (pl.Indirect(row_subchunk_size), rope_in_cols),
            lambda r, s=s: (idx_window(r, s), 0),
        )
        for s in range(num_streams)
    )

    # One rope output block per step, covering all `num_streams`
    # subchunks: a step in the first half of a period fills int32 lanes
    # [0, rope_out_cols) of its rows, a step in the second half lanes
    # [rope_out_cols, 2 * rope_out_cols) of the same rows.
    steps_per_half = top_k // 2 // (num_streams * row_subchunk_size)

    def rope_out_block(r):
      half, step = divmod(out_row_base // num_streams + r, steps_per_half)
      return (half // 2) * steps_per_half + step, half % 2

    rope_out_spec = pl.BlockSpec(
        (num_streams * rope_rows_per_stream, rope_out_cols), rope_out_block
    )
    core_start_idx = out_row_base * row_subchunk_size

    @pl.when(core_start_idx < valid_indices)
    def _run_gather():
      pltpu.emit_pipeline(
          _body,
          grid=(num_row_subchunks // num_streams,),
          in_specs=nope_in_specs + rope_in_specs,
          out_specs=(rope_out_spec,),
      )(
          *([nope_in_hbm_ref] * num_streams),
          *([rope_in_hbm_ref] * num_streams),
          rope_out_hbm_ref,
      )

  pltpu.emit_pipeline(
      outer_pipeline,
      grid=(num_blocks,),
      in_specs=(
          pl.BlockSpec(
              (row_chunk_size,),
              lambda b: (b * num_cores + core_index,),
          ),
          pl.BlockSpec(
              (row_subchunk_size,),
              lambda b: (0,),
          ),
      ),
  )(indices_hbm_ref, valid_indices_ref)


@functools.partial(jax.jit, static_argnames=("top_k", "num_streams"))
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    num_valid_indices: jax.Array | None = None,
    *,
    top_k: int = 1024,
    num_streams: int = 4,
) -> tuple[jax.Array, jax.Array]:
  """Fused SparseCore gather of the nope and rope caches.

  Args:
   nope_cache: (total_pages, page_size, 128) int32, one row per token: the (4,
     128) uint8 nope slab (see `reference`).
   rope_cache: (total_pages, page_size // 4, 128) int32, 32 words per token (see
     `reference`).
   indices: (N,) int32. Token indices into the caches; N a multiple of top_k.
   num_valid_indices: Optional (1,) or scalar int32. Number of valid indices to
     gather, a multiple of top_k. A subcore skips every top_k period that starts
     at or past this count, leaving those output rows unwritten.
   top_k: the consumer's row block (the attention kernel's top-k), a multiple of
     128. `rope_out` pairs entry i of a period with entry i + top_k // 2.
   num_streams: Number of independent `pl.Indirect` gathers issued per pipeline
     step, at most `MAX_NUM_STREAMS`. `top_k // 2` must be a multiple of
     `num_streams` times the number of SparseCore lanes.

  Returns:
   Both outputs are int32 with 128 lanes, which XLA lays out byte-for-byte
   like the uint8 / bf16 arrays they encode, so the consumer views them with
   a free `ref.bitcast`.
   nope_out: (N, 128) int32. Row i is entry i's (4, 128) uint8 nope slab.
   rope_out: (N // 4, 128) int32, i.e. (N // 2, 128) bf16 once bitcast.
    For i < top_k // 2, bf16 row `period * (top_k // 2) + i` holds entry
    i of that period in lanes 0:64 and entry i + top_k // 2 in lanes
    64:128 -- 128 lanes so XLA does not pad the buffer to twice its size.
    The consumer restores a period's (top_k, 64) from its
    (top_k // 2, 128) block `blk` with
    `jnp.concatenate([blk[:, :64], blk[:, 64:]], axis=0)`.
  """
  assert indices.ndim == 1, "Indices must be 1D."
  assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
  assert nope_cache.dtype == jnp.int32, "Caches must be int32."
  assert nope_cache.shape[2] == ROW_WORDS
  assert rope_cache.shape[2] == ROW_WORDS

  # Free reshapes: a 128-lane int32 array's T(8,128) tiling is byte-identical
  # to row-major, and the kernel takes its operands untiled
  # (`use_tc_tiling_on_sc=False`), so token t's rope is words
  # [32 t, 32 t + 32).
  nope_cache = nope_cache.reshape(-1, ROW_WORDS)
  rope_cache = rope_cache.reshape(-1, ROPE_WORDS)
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "SparseCore info is missing."
  out_size = indices.size
  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores
  row_subchunk_size = num_simd_lanes

  if num_valid_indices is None:
    valid_indices = jnp.full((row_subchunk_size,), out_size, dtype=jnp.int32)
  else:
    valid_indices = jnp.full(
        (row_subchunk_size,), num_valid_indices, dtype=jnp.int32
    )

  # `num_streams` independent `pl.Indirect` gathers are issued per
  # pipeline step to keep multiple gather DMAs in flight.
  # See `outer_pipeline` for details.
  # More than `MAX_NUM_STREAMS` streams hangs the kernel on TPU7x.
  if not 0 < num_streams <= MAX_NUM_STREAMS:
    raise ValueError(
        f"num_streams must be in [1, {MAX_NUM_STREAMS}], got {num_streams}."
    )
  if (top_k // 2) % (num_streams * row_subchunk_size):
    raise ValueError(
        f"top_k // 2 ({top_k // 2}) must be a multiple of num_streams"
        f" ({num_streams}) times the SparseCore lane count"
        f" ({row_subchunk_size})."
    )
  assert out_size % top_k == 0
  num_row_subchunks = top_k // row_subchunk_size
  row_chunk_size = row_subchunk_size * num_row_subchunks
  block_size = row_chunk_size * num_cores
  out_pad_size = (
      (out_size + block_size - 1) // block_size
  ) * block_size - out_size
  if out_pad_size:
    # spread the padding to avoid hotspots
    num_tokens = nope_cache.shape[0]
    pad = (jnp.arange(out_pad_size, dtype=indices.dtype) * 104729) % num_tokens
    indices = jnp.concatenate([indices, pad])
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  nope_out, rope_out = pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          num_row_subchunks=num_row_subchunks,
          num_streams=num_streams,
          top_k=top_k,
      ),
      out_type=(
          jax.ShapeDtypeStruct((out_size + out_pad_size, ROW_WORDS), jnp.int32),
          jax.ShapeDtypeStruct(
              ((out_size + out_pad_size) // 4, ROW_WORDS),
              jnp.int32,
          ),
      ),
      # One DMA semaphore per stream for the direct nope gather-buffer -> HBM
      # copies issued in `main_kernel`.
      scratch_types=(pltpu.SemaphoreType.DMA((num_streams,)),),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=False,
          needs_layout_passes=True,
          disable_bounds_checks=True,
      ),
      mesh=vector_mesh,
      name="sc_csa_gather",
  )(nope_cache, rope_cache, indices, valid_indices)
  return nope_out[:out_size], rope_out[: out_size // 4]
