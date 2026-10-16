# Copyright 2026 Google LLC
# Copyright 2026 Rabdos AI
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
"""Fused mixed prefill/decode dispatch for Causal Conv1D Gated Delta Rule."""

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import compute as fused_compute
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import decode as fused_decode
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import memory as fused_memory
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import metadata as fused_metadata
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import prefill as fused_prefill
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import tiling as fused_tiling

_STATIC_OPTIONS = (
    "n_kq",
    "n_v",
    "d_k",
    "d_v",
    "kernel_size",
    "zero_initialize_out",
    "compute_precision",
    "decode_tile_size",
    "mixed_tile_size",
)


def _mixed_outer(
    query_start_ref: Any,
    state_indices_ref: Any,
    read_state_indices_ref: Any,
    read_offsets_ref: Any,
    distribution_ref: Any,
    seq_lens_ref: Any,
    qkv_ref: Any,
    b_ref: Any,
    a_ref: Any,
    conv_state_ref: Any,
    recurrent_state_ref: Any,
    weights_ref: Any,
    dense_conv_ref: tuple[Any, Any | None],
    out_ref: Any,
    conv_state_out_ref: Any,
    recurrent_state_out_ref: Any,
    output_tail_ref: Any | None,
    *,
    decode_cfg: config.GDNConfig,
    prefill_cfg: config.GDNConfig,
    **scratch: Any,
) -> None:
  """Run validated decode and prefill within one Pallas dispatch.

  Both caches alias their inputs. Runtime validation gates cache updates, and
  scratch must match _fused_conv1d_gdn_fast's allocations.

  Args:
    query_start_ref: SMEM int32 request token boundaries, with requests + 1
      entries.
    state_indices_ref: SMEM per-request write-cache slots; active owners must be
      distinct.
    read_state_indices_ref: Per-request initial-state cache slots, separate from
      write ownership.
    read_offsets_ref: Per-request int32 read-slot offsets; public dispatch zeros
      prefill offsets.
    distribution_ref: SMEM endpoints for decode, prefill, and active request
      prefixes.
    seq_lens_ref: SMEM total sequence lengths, including previously consumed
      history.
    qkv_ref: HBM packed activations [tokens, 2 * n_kq * d_k + n_v * d_v].
    b_ref: HBM beta-gate inputs [tokens, n_v], with the activation dtype.
    a_ref: HBM decay-gate inputs [tokens, n_v], with the activation dtype.
    conv_state_ref: HBM convolution cache; reads use prefix slots and writes use
      owned slots.
    recurrent_state_ref: HBM recurrent cache [slots, n_v, d_k, d_v].
    weights_ref: VMEM refs for log decay weights and timestep biases.
    dense_conv_ref: FP32 convolution tap and optional bias refs in head-major
      layout.
    out_ref: HBM output array [tokens, n_v * d_v], written by the pipeline.
    conv_state_out_ref: Convolution-cache output alias; updates use the input
      reference.
    recurrent_state_out_ref: HBM recurrent-cache destination alias, written only
      at owned slots.
    output_tail_ref: Optional HBM 16-row tail block stitched into the final
      output later.
    decode_cfg: Single-token configuration for the decode prefix.
    prefill_cfg: Single-sequence prefill configuration with the selected tile
      size.
    **scratch: Scratch refs matching the fast path's allocated buffer
      dictionary.
  """
  del conv_state_out_ref
  recurrent_state_out_ref = recurrent_state_ref
  metadata_ref = scratch["metadata_ref"]
  output_refs = fused_memory.OutputRefs(
      scratch=scratch["output_scratch_ref"],
      carry=scratch["output_carry_ref"],
      sem=scratch["output_sem_ref"],
      tail=output_tail_ref,
  )

  schedule_valid = fused_metadata.build_smem_schedule(
      query_start_ref,
      state_indices_ref,
      distribution_ref,
      seq_lens_ref,
      metadata_ref,
      scratch["request_prefix_ref"],
      scratch["occupancy_ref"],
      cfg=prefill_cfg,
      read_state_indices_ref=read_state_indices_ref,
      read_offsets_ref=read_offsets_ref,
  )
  output_refs.carry[...] = jnp.zeros(
      output_refs.carry.shape, dtype=output_refs.carry.dtype
  )

  def run_decode_prefix() -> None:
    """Finish decode before any prefill history read."""

    @pl.when(schedule_valid)
    def run_decodes() -> None:
      """Assemble grouped decode references and call the decode pipeline."""
      fused_decode.run_grouped_decode(
          state_indices_ref,
          read_state_indices_ref,
          read_offsets_ref,
          distribution_ref,
          seq_lens_ref,
          qkv_ref,
          b_ref,
          a_ref,
          conv_state_ref,
          recurrent_state_ref,
          recurrent_state_out_ref,
          weights_ref,
          dense_conv_ref,
          out_ref,
          scratch["qkv_head_major_scratch_ref"],
          scratch["grouped_state_scratch_ref"],
          output_refs.carry,
          output_refs.sem,
          fused_decode.DecodeGroupRefs(
              qkv=scratch["decode_group_qkv_ref"],
              b=scratch["decode_group_b_ref"],
              a=scratch["decode_group_a_ref"],
              input_sem=scratch["decode_group_input_sem_ref"],
              conv=scratch["decode_group_conv_ref"],
              conv_load_sem=scratch["decode_group_conv_load_sem_ref"],
              conv_store_sem=scratch["decode_group_conv_store_sem_ref"],
              extra_state=scratch["decode_extra_state_refs"],
              member0_state=scratch.get("decode_member0_state_ref"),
              state_load_sem=scratch["decode_state_load_sem_refs"],
              state_store_sem=scratch["decode_state_store_sem_refs"],
          ),
          cfg=decode_cfg,
      )

  @pl.when(metadata_ref.num_tiles[...] == 0)
  def run_decode_only() -> None:
    """Run decode directly when no prefill tiles exist."""
    run_decode_prefix()

  def run_prefill_tile(
      p_id: jax.Array, qkv_slot: Any, b_slot: Any, a_slot: Any
  ) -> None:
    """Forward one prefill tile to the state/output pipeline.

    Args:
      p_id: Zero-based packed prefill tile index.
      qkv_slot: Private QKV input window for the current prefill tile.
      b_slot: Private beta-input window for one prefill tile, including
        alignment rows.
      a_slot: Private decay-input window for one prefill tile, including
        alignment rows.
    """
    fused_prefill.prefill_pipeline_inner(
        p_id,
        qkv_slot,
        b_slot,
        a_slot,
        recurrent_state_ref,
        recurrent_state_out_ref,
        scratch["prefill_recurrent_scratch_ref"],
        scratch["prefill_recurrent_load_sem_ref"],
        scratch["prefill_recurrent_store_sem_ref"],
        conv_state_ref,
        scratch["conv_dma_scratch_ref"],
        scratch["conv_state_slot_ref"],
        scratch["conv_dma_sem_ref"],
        scratch["conv_store_scratch_ref"],
        scratch["conv_store_sem_ref"],
        metadata_ref,
        weights_ref,
        dense_conv_ref,
        scratch["carry_conv_scratch_ref"],
        scratch.get("carry_recurrent_scratch_ref"),
        scratch["qkv_head_major_scratch_ref"],
        scratch["grouped_state_scratch_ref"],
        out_ref,
        output_refs.scratch,
        output_refs.carry,
        output_refs.sem,
        distribution_ref,
        read_state_indices_ref,
        read_offsets_ref,
        cfg=prefill_cfg,
    )

  @pl.when(metadata_ref.num_tiles[...] > 0)
  def run_prefills() -> None:
    """Run the prefill pipeline and drain its final cache stores."""
    first_owner = metadata_ref.get_record(0, 0).s_idx

    def initialize_prefill() -> None:
      # Input DMA is already in flight. Keep history reads after decode so
      # requests sharing prefix-cache slots retain their original ordering.
      """Finish decode, then start the first prefill request's state loads."""
      run_decode_prefix()

      @pl.when(
          fused_metadata.metadata_storage(metadata_ref.s_idx_has_initial_state)[
              first_owner
          ]
      )
      def prime_states() -> None:
        # Overlap the first state and activation loads. Later state loads start
        # on the preceding request's last tile.
        """Start initial prefill state loads only for a resumed request."""
        fused_memory.recurrent_state_transfer(
            metadata_ref,
            read_state_indices_ref,
            read_offsets_ref,
            recurrent_state_ref,
            scratch["prefill_recurrent_scratch_ref"],
            scratch["prefill_recurrent_load_sem_ref"],
            scratch["prefill_recurrent_store_sem_ref"],
            first_owner,
            to_hbm=False,
            wait=False,
        )
        fused_memory.conv_state_transfer(
            metadata_ref,
            read_state_indices_ref,
            read_offsets_ref,
            conv_state_ref,
            scratch["conv_dma_scratch_ref"],
            scratch["conv_dma_sem_ref"],
            first_owner,
            to_hbm=False,
            wait=False,
        )

    fused_prefill.run_prefill_pipeline(
        run_prefill_tile,
        qkv_ref,
        b_ref,
        a_ref,
        metadata_ref,
        chunk_size=prefill_cfg.chunk_size,
        before_first_tile=initialize_prefill,
    )
    last_owner = metadata_ref.get_record(
        metadata_ref.num_tiles[...] - 1, 0
    ).s_idx
    wait_for_conv_store = functools.partial(
        fused_memory.conv_state_transfer,
        metadata_ref,
        read_state_indices_ref,
        read_offsets_ref,
        conv_state_ref,
        scratch["conv_store_scratch_ref"],
        scratch["conv_store_sem_ref"],
        to_hbm=True,
        wait=True,
    )
    wait_for_conv_store(last_owner)
    wait_for_state_store = functools.partial(
        fused_memory.recurrent_state_transfer,
        metadata_ref,
        read_state_indices_ref,
        read_offsets_ref,
        recurrent_state_out_ref,
        scratch["prefill_recurrent_scratch_ref"],
        scratch["prefill_recurrent_load_sem_ref"],
        scratch["prefill_recurrent_store_sem_ref"],
        to_hbm=True,
        wait=True,
    )
    wait_for_state_store(last_owner)

    @pl.when(last_owner > distribution_ref[0])
    def drain_other_state_stores() -> None:
      """Drain the other prefill state slot when more than one prefill request
      was processed.
      """
      wait_for_conv_store(last_owner - 1)
      wait_for_state_store(last_owner - 1)

  fused_prefill.finish_bucket_output(
      query_start_ref,
      state_indices_ref,
      distribution_ref,
      out_ref,
      output_refs,
      schedule_valid=schedule_valid,
  )


@functools.partial(jax.jit, static_argnames=_STATIC_OPTIONS + ("interpret",))
def _fused_conv1d_gdn_fast(
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
    read_state_indices: jax.Array,
    read_offsets: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32,
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None,
    interpret: bool = False,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
  """Allocate and dispatch a statically eligible fused_conv1d_gdn call.

  Read indices/offsets must be resolved [requests] int32 arrays. Runtime
  validation precedes both pipelines; rejected schedules return zero output
  and unchanged caches. interpret selects Pallas interpret mode.

  Args:
    qkv: [tokens, 2 * n_kq * d_k + n_v * d_v] packed activations.
    b: [tokens, n_v] beta gate inputs; same dtype as qkv.
    a: [tokens, n_v] decay gate inputs; same dtype as qkv.
    conv_state: [slots, kernel_size - 1, QKV width] convolution cache.
    recurrent_state: [slots, n_v, d_k, d_v] recurrent cache.
    conv_weight: [QKV width, 1, kernel_size] convolution weights.
    conv_bias: Optional [QKV width] convolution bias.
    a_log: [n_v] log decay weights.
    dt_bias: [n_v] decay biases.
    query_start_loc: [requests + 1] int32 cumulative token offsets, starting at
      zero.
    state_indices: [requests] int32 write-cache slots; active slots must be
      distinct, nonzero, and in bounds.
    distribution: [3] int32 cumulative endpoints [decode_end, prefill_end,
      mixed_end].
    seq_lens: [requests] int32 sequence lengths including history and submitted
      tokens.
    read_state_indices: Resolved [requests] int32 initial-state slots.
    read_offsets: Resolved [requests] int32 offsets; prefill offsets must
      already be zero.
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.
    d_k: Key/query head dimension; the fused path requires 128.
    d_v: Value head dimension; the fused path requires 128.
    kernel_size: Positive static convolution window size.
    zero_initialize_out: Unused compatibility option; the kernel writes every
      output row.
    compute_precision: Requested accumulation dtype; fused execution requires
      FP32.
    decode_tile_size: Optional requested number of decode members processed per
      group.
    mixed_tile_size: Optional requested prefill token rows per tile.
    interpret: Whether Pallas executes in interpreter mode.

  Returns:
    ((new_conv, new_recurrent), output), with cache shapes/dtypes and activation
    output dtype preserved.
  """
  del zero_initialize_out
  tokens, width = qkv.shape
  slots = conv_state.shape[0]
  requests = state_indices.size
  activation_dtype = qkv.dtype
  conv_dtype = conv_state.dtype
  recurrent_dtype = recurrent_state.dtype
  conv_shape = conv_state.shape
  # One-tap convolutions need nonempty scratch but no history transfers.
  history_rows = max(1, kernel_size - 1)

  packed_heads = 2 * n_kq + n_v
  # Narrow layouts fit three state slots, overlapping group g stores with
  # g+1 loads/compute; wider layouts use two.
  state_ring_depth = fused_tiling.state_ring_depth(n_kq, n_v)
  # Preserve [tokens, width] through Pallas and DMA to avoid boundary copies
  # introduced by a [tokens, heads, 128] input layout.
  # Pad buckets shorter than the eight-row compact BF16 tile.
  if tokens < 8:
    qkv = jnp.pad(qkv, ((0, 8 - tokens), (0, 0)))
    b = jnp.pad(b, ((0, 8 - tokens), (0, 0)))
    a = jnp.pad(a, ((0, 8 - tokens), (0, 0)))
  # Keep gates [tokens, heads] to avoid boundary copies and per-token padding.
  # Select and clear aligned windows only in private scratch.

  # Taps run oldest to newest; the last tap multiplies the current token.
  conv_weight_f32 = conv_weight.swapaxes(0, 2).astype(jnp.float32)
  conv_bias_f32 = None if conv_bias is None else conv_bias.astype(jnp.float32)
  # Stage convolution weights only in the shared dense layout.
  weights = memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
  dense_conv = (
      conv_weight_f32.reshape(kernel_size, packed_heads, d_k),
      None
      if conv_bias_f32 is None
      else conv_bias_f32.reshape(packed_heads, d_k),
  )

  shared = dict(
      mode=config.GDNMode.PER_SEQ,
      batch_size=tokens,
      kernel_size=kernel_size,
      dim_size=width,
      num_kq_heads=n_kq,
      num_v_heads=n_v,
      kq_head_dim=d_k,
      v_head_dim=d_v,
      num_buffers=2,
      dtypes=config.Dtypes(
          act_in=jnp.float32,
          act_out=activation_dtype,
          compute=compute_precision,
          recurrent_state=recurrent_dtype,
          conv_state=conv_dtype,
      ),
  )
  decode_cfg = config.GDNConfig(tile_size=1, **shared)
  tile_rows = fused_tiling.prefill_tile_rows(width, mixed_tile_size)
  prefill_cfg = config.GDNConfig(tile_size=tile_rows, **shared)
  capacity = fused_metadata.prefill_schedule_capacity(
      requests, tokens, prefill_cfg.chunk_size, n_kq, n_v
  )
  metadata_template = fused_metadata.metadata_template(
      prefill_cfg, requests, capacity
  )
  metadata_scratch = jax.tree.map(
      lambda value: pltpu.SMEM(value.shape, value.dtype), metadata_template
  )

  smem_inputs = (
      query_start_loc,
      state_indices,
      read_state_indices,
      read_offsets,
      distribution,
      seq_lens,
  )
  hbm_inputs = (qkv, b, a, conv_state, recurrent_state)
  vmem_inputs = (weights, dense_conv)

  def input_specs(inputs: Any, space: Any) -> Any:
    """Map a pytree to whole-array BlockSpecs in the given memory space."""
    spec = pl.BlockSpec(memory_space=space)
    return jax.tree.map(lambda _: spec, inputs)

  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  conv_state_input_index = len(jax.tree.leaves(smem_inputs + hbm_inputs[:3]))
  # Outputs 1 and 2 are the convolution and recurrent caches. Reuse their input
  # buffers so inactive slots stay in place.
  cache_output_aliases = {
      conv_state_input_index: 1,
      conv_state_input_index + 1: 2,
  }

  tail_rows = tokens % fused_compute.OUTPUT_ALIGNMENT
  output_width = n_v * d_v
  tail_shape = (
      jax.ShapeDtypeStruct(
          (fused_compute.OUTPUT_ALIGNMENT, output_width), activation_dtype
      )
      if tail_rows
      else None
  )
  group_width = fused_tiling.decode_group_width(requests, n_v, decode_tile_size)
  vmem_limit = config.get_vmem_limit_bytes()
  dma = pltpu.SemaphoreType.DMA

  def group_buffer(
      shape: tuple[int, ...], dtype: jnp.dtype = activation_dtype
  ) -> Any:
    """Return a two-parity VMEM descriptor: [2, group_width, *shape]."""
    return pltpu.VMEM((2, group_width, *shape), dtype)

  scratch_shapes = {
      "carry_conv_scratch_ref": pltpu.VMEM(
          (1, history_rows, 1, width), jnp.float32
      ),
      # BF16 caches need a separate FP32 carry between tiles.
      **(
          {}
          if recurrent_dtype == jnp.float32
          else {
              "carry_recurrent_scratch_ref": pltpu.VMEM(
                  (1, n_v, d_k, d_v), jnp.float32
              )
          }
      ),
      "prefill_recurrent_scratch_ref": pltpu.VMEM(
          (2, n_v, d_k, d_v), recurrent_dtype
      ),
      "prefill_recurrent_load_sem_ref": dma((2,)),
      "prefill_recurrent_store_sem_ref": dma((2,)),
      "conv_dma_scratch_ref": pltpu.VMEM((2, history_rows, width), conv_dtype),
      "conv_state_slot_ref": pltpu.VMEM(
          (1, history_rows, 1, width), conv_dtype
      ),
      "conv_dma_sem_ref": dma((2,)),
      "conv_store_scratch_ref": pltpu.VMEM(
          (2, history_rows, width), conv_dtype
      ),
      "conv_store_sem_ref": dma((2,)),
      "qkv_head_major_scratch_ref": pltpu.VMEM(
          (max(packed_heads, state_ring_depth * n_v), max(tile_rows, d_k), d_k),
          jnp.float32,
      ),
      "grouped_state_scratch_ref": pltpu.VMEM(
          (n_kq, d_k, (n_v // n_kq) * d_v), jnp.float32
      ),
      "output_scratch_ref": pltpu.VMEM(
          (2, tile_rows + fused_compute.OUTPUT_ALIGNMENT, output_width),
          activation_dtype,
      ),
      "output_carry_ref": pltpu.VMEM(
          (fused_compute.OUTPUT_ALIGNMENT, output_width), activation_dtype
      ),
      "output_sem_ref": dma((2,)),
      "decode_group_qkv_ref": pltpu.VMEM(
          (2, ((group_width + 14) // 8) * 8, width), activation_dtype
      ),
      "decode_group_b_ref": pltpu.VMEM(
          (2, ((group_width + 14) // 8) * 8, n_v), activation_dtype
      ),
      "decode_group_a_ref": pltpu.VMEM(
          (2, ((group_width + 14) // 8) * 8, n_v), activation_dtype
      ),
      "decode_group_input_sem_ref": dma((2, 3)),
      "decode_group_conv_ref": group_buffer((history_rows, width), conv_dtype),
      "decode_group_conv_load_sem_ref": dma((2, group_width)),
      "decode_group_conv_store_sem_ref": dma((2, group_width)),
  }
  state_ring_depths = fused_tiling.decode_state_ring_depths(
      group_width,
      scratch_shapes,
      weights,
      dense_conv,
      vmem_limit,
      member0_buffered=recurrent_dtype != jnp.float32,
  )
  if recurrent_dtype != jnp.float32:
    scratch_shapes["decode_member0_state_ref"] = pltpu.VMEM(
        (state_ring_depths[0], n_v, d_k, d_v), recurrent_dtype
    )
  scratch_shapes.update(
      {
          "decode_extra_state_refs": tuple(
              pltpu.VMEM((depth, n_v, d_k, d_v), recurrent_dtype)
              for depth in state_ring_depths[1:]
          ),
          "decode_state_load_sem_refs": tuple(
              dma((depth,)) for depth in state_ring_depths
          ),
          "decode_state_store_sem_refs": tuple(
              dma((depth,)) for depth in state_ring_depths
          ),
          "metadata_ref": metadata_scratch,
          "request_prefix_ref": pltpu.SMEM((requests + 1,), jnp.int32),
          "occupancy_ref": pltpu.SMEM((slots,), jnp.int32),
      }
  )

  output, new_conv, new_recurrent, output_tail = pl.pallas_call(
      functools.partial(
          _mixed_outer,
          decode_cfg=decode_cfg,
          prefill_cfg=prefill_cfg,
      ),
      out_shape=(
          jax.ShapeDtypeStruct((tokens, output_width), activation_dtype),
          conv_state,
          recurrent_state,
          tail_shape,
      ),
      in_specs=(
          *input_specs(smem_inputs, pltpu.SMEM),
          *input_specs(hbm_inputs, pltpu.HBM),
          *input_specs(vmem_inputs, pltpu.VMEM),
      ),
      out_specs=(hbm, hbm, hbm, hbm if tail_rows else None),
      scratch_shapes=scratch_shapes,
      input_output_aliases=cache_output_aliases,
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
      # Cost hints omit gates, weights, metadata, repeated loads and some
      # recurrence work; they are not measurements.
      cost_estimate=pl.CostEstimate(
          flops=int(
              4 * tokens * n_v * d_k * d_v + 2 * tokens * width * kernel_size
          ),
          transcendentals=int(2 * tokens * n_v),
          bytes_accessed=int(
              2 * slots * n_v * d_k * d_v * recurrent_state.dtype.itemsize
              + 2 * slots * history_rows * width * conv_state.dtype.itemsize
              + tokens * width * qkv.dtype.itemsize
              + tokens * output_width * activation_dtype.itemsize
          ),
          remote_bytes_transferred=0,
      ),
      name="fused_conv1d_gdn",
      interpret=interpret,
  )(*smem_inputs, *hbm_inputs, *vmem_inputs)

  if tail_rows:
    output = jax.lax.dynamic_update_slice(
        output, output_tail[:tail_rows, :], (tokens - tail_rows, 0)
    )
  return ((new_conv.reshape(conv_shape), new_recurrent), output)


@functools.partial(
    jax.jit,
    donate_argnums=(3, 4),
    # num_spec_tokens controls Python dispatch, so keep it static. The fast
    # entry point has no such parameter; exclude it from _STATIC_OPTIONS.
    static_argnames=_STATIC_OPTIONS + ("num_spec_tokens",),
)
def fused_conv1d_gdn(
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
    num_spec_tokens: int = 0,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
  """Fuse mixed prefill/decode with donated convolution and recurrent caches.

  Supported layouts use one dispatch; others use the upstream wrapper. The
  fused path accepts BF16/FP32 activations and caches. Weights may be BF16,
  FP32, FP16 or int32; accumulation uses FP32. Active read/write addresses
  must be nonzero and in bounds, with distinct writes. Only active token
  offsets must be monotone; inactive padding may contain negative lengths.
  Invalid fused schedules return zero output and unchanged caches. Use
  returned caches after donation.

  Args:
    qkv: [tokens, 2 * n_kq * d_k + n_v * d_v] packed activations.
    b: [tokens, n_v] beta gate inputs; same dtype as qkv.
    a: [tokens, n_v] decay gate inputs; same dtype as qkv.
    conv_state: [slots, kernel_size - 1, QKV width] convolution cache.
    recurrent_state: [slots, n_v, d_k, d_v] recurrent cache.
    conv_weight: [QKV width, 1, kernel_size] convolution weights.
    conv_bias: Optional [QKV width] convolution bias.
    a_log: [n_v] log decay weights.
    dt_bias: [n_v] decay biases.
    query_start_loc: [requests + 1] int32 cumulative token offsets, starting at
      zero.
    state_indices: [requests] int32 cache slots; active requests must use
      distinct nonzero slots.
    distribution: [3] int32 cumulative endpoints [decode_end, prefill_end,
      mixed_end]. The active count is clip(mixed_end, 0, requests), which must
      be positive. decode_end must be nonnegative and no greater than prefill_end
      or the active count. Active requests form a prefix ending at the active
      count; decode requests form its one-token prefix and prefills have
      positive token counts. mixed_end may exceed the request count, as in
      upstream's [0, 0, 3] single-request case.
    seq_lens: [requests] int32 total sequence lengths, including history; each
      active length must cover its submitted tokens.
    read_state_indices: Optional [requests] int32 initial-state cache slots;
      defaults to state_indices. Multiple requests may share an unwritten
      checkpoint. Cross-request read/write overlap has no snapshot guarantee.
    read_offsets: Optional [requests] int32 offsets added to decode or verify
      read slots; defaults to zero. The fused path applies offsets only to
      decodes. Prefills resume from read_state_indices without an offset.
    num_spec_tokens: Number of speculative draft tokens. Nonzero values use
      the upstream wrapper; positive values require read_offsets.
    n_kq: Number of key/query heads; the fused path requires a positive count.
    n_v: Number of value heads; the fused path requires a positive count
      divisible by n_kq.
    d_k: Key/query head dimension; the fused path requires 128.
    d_v: Value head dimension; the fused path requires 128.
    kernel_size: Positive static convolution window size.
    zero_initialize_out: Wrapper initialization option; the fused path writes
      every output row regardless of this flag.
    compute_precision: Compute dtype; the fused path requires float32.
    decode_tile_size: Requested decode group width, up to MAX_DECODE_GROUP;
      unsupported hints use the default. None tries the default, then smaller
      groups if no prefill tile fits the memory budget.
    mixed_tile_size: Requested prefill rows, or None to select by packed width.
      Tiles must fit the budget, be multiples of 64 rows, and contain whole
      recurrent subchunks. Unsupported hints use the default or fallback tiles.

  Returns:
    ((convolution cache, recurrent cache), output). Cache shapes and dtypes are
    preserved; output has shape [tokens, n_v * d_v] and the activation dtype.
  """
  # Match the wrapper's address defaults and casts before selecting a path.
  if num_spec_tokens > 0 and read_offsets is None:
    raise ValueError("read_offsets is required when num_spec_tokens > 0")
  requested_read_indices, requested_read_offsets = (
      read_state_indices,
      read_offsets,
  )
  if read_state_indices is None:
    read_state_indices = state_indices
  if read_offsets is None:
    read_offsets = jnp.zeros_like(state_indices)
  if read_offsets.shape != state_indices.shape:
    raise ValueError(
        f"read_offsets must have shape {state_indices.shape},"
        f" got {read_offsets.shape}"
    )
  if read_state_indices.shape != state_indices.shape:
    raise ValueError(
        f"read_state_indices must have shape {state_indices.shape},"
        f" got {read_state_indices.shape}"
    )
  read_offsets = read_offsets.astype(jnp.int32)
  read_state_indices = read_state_indices.astype(state_indices.dtype)
  args = (
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
  )
  options = dict(
      n_kq=n_kq,
      n_v=n_v,
      d_k=d_k,
      d_v=d_v,
      kernel_size=kernel_size,
      zero_initialize_out=zero_initialize_out,
      compute_precision=compute_precision,
      decode_tile_size=decode_tile_size,
      mixed_tile_size=mixed_tile_size,
  )

  fused_options = dict(options)
  if decode_tile_size is not None and not (
      1 <= decode_tile_size <= fused_tiling.MAX_DECODE_GROUP
  ):
    fused_options["decode_tile_size"] = None
  tile_options = {
      k: v for k, v in fused_options.items() if k != "mixed_tile_size"
  }
  (
      eligible,
      fused_options["decode_tile_size"],
      fused_options["mixed_tile_size"],
  ) = fused_tiling.select_tiles(args, mixed_tile_size, **tile_options)
  if num_spec_tokens or not eligible:
    # Fallback keeps caller options; replacement tiles apply only to fused calls.
    return wrapper.fused_conv1d_gdn(
        *args,
        requested_read_indices,
        requested_read_offsets,
        num_spec_tokens=num_spec_tokens,
        **options,
    )

  # Match upstream PER_SEQ: prefill resumes at the base prefix-cache slot.
  read_offsets = jnp.where(
      jnp.arange(state_indices.size) < distribution[0], read_offsets, 0
  )
  return _fused_conv1d_gdn_fast(
      *args, read_state_indices, read_offsets, **fused_options
  )
