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

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_conv1d
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_gdn
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_kda
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.causal_conv1d_gated_delta_rule import metadata
from tokamax._src.ops.causal_conv1d_gated_delta_rule import tiling
from tokamax._src.ops.causal_conv1d_gated_delta_rule import vmem_ldst


def inner_kernel(
    # Inputs.
    qkv_slot_ref: Any,  # [seq, chunk, 1, dim_size]
    b_slot_ref: Any,  # [seq, chunk, 1, num_v_heads]
    a_slot_ref: Any,  # [seq, chunk, 1, num_v_heads]
    conv_state_slot_ref: Any,  # [seq, prev_kernel_size, 1, dim_size]
    recurrent_slot_ref: Any,  # [seq, num_v_heads, kq_head, v_head]
    # Outputs.
    out_slot_ref: Any,  # [seq * chunk, num_v_heads, v_head]
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: Any = None,
    carry_recurrent_scratch_ref: Any = None,
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array | None = None,
):
  """Orchestrates computation of Conv1D and GDN for a single tile.

  This kernel acts as a facade adhering to strict separation of concerns. It
  operates VMEM reference without knowledge on DMA logic. Furthermore, the
  kernel invokes vmem_ldst to pre-processes data needed for compute and
  invokes compute_conv1d and compute_gdn for actual compute.

  Args:
      qkv_slot_ref: qkv VMEM ref that stores data loaded from HBM.
      b_slot_ref: b VMEM ref that stores data loaded from HBM.
      a_slot_ref: a VMEM ref that stores data loaded from HBM.
      conv_state_slot_ref: Convolution state VMEM ref that stores data loaded
        from HBM. Data written into this VMEM ref will be used for VMEM to HBM
        write.
      recurrent_slot_ref: Recurrent state VMEM ref that stores data loaded from
        HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
      out_slot_ref: Output VMEM ref that will be used for VMEM to HBM write.
      metadata_ref: Metadata reference containing grid and sequence mappings.
      weights_ref: Weight references for Conv1D and GDN in VMEM.
      carry_conv_scratch_ref: Optional VMEM scratch reference for inter-tile
        convolution carry.
      carry_recurrent_scratch_ref: Optional VMEM scratch reference for
        inter-tile recurrent state carry.
      cfg: GDN configuration object.
      p_id: Optional explicit tile index. Communication-fused outer kernels pass
        their global compute-tile index because their surrounding pipeline owns
        the Pallas program id.
  """

  if p_id is None:
    p_id = pl.program_id(0)

  # Prepare states.
  real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
      metadata_ref=metadata_ref,
      p_id=p_id,
      conv_state_slot_ref=conv_state_slot_ref,
      recurrent_slot_ref=recurrent_slot_ref,
      carry_conv_scratch_ref=carry_conv_scratch_ref,
      carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
      cfg=cfg,
  )

  # Step 1: Conv1D.
  # NOTE: Conv1D requires performing sliding window where inputs are slided
  # across rows. If typical 2D layout was used, multiple rows are stored in a
  # single register which necessitate costly shuffling for every sliding.
  # Therefore, it is extremely important to leverage compact layout that
  # ensures 1 register only stores data from 1 row.
  qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
  qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

  # Prepare conv1d weights.
  conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
  conv_bias = None
  if weights_ref.conv.bias is not None:
    conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

  qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
      real_sizes=real_sizes,
      lhs=qkv_in_compact,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      cfg=cfg,
  )

  if cfg.state_plan is None:
    conv_state_slot_ref[...] = new_conv_state
  else:
    # External source tiles: encode each window position's checkpoint
    # through the region's typed view; copy_out pushes the raw bytes
    # of the valid checkpoints back to the source.
    for idx in range(cfg.seq_tile_size):

      for t in range(cfg.window_size):
        args = (
            conv_state_slot_ref.at[idx, t],
            cfg.state_plan.conv,
            new_conv_state[idx, t],
        )
        if carry_conv_scratch_ref is None:
          vmem_ldst.store_state_region(*args)
        else:
          # copy_out only DMAs the tile that ends a sequence.
          pl.when(metadata_ref.p_id_is_last_tile[p_id, idx])(
              functools.partial(vmem_ldst.store_state_region, *args)
          )
  if carry_conv_scratch_ref is not None:
    # The next tile resumes from the state after this tile's last token,
    # which is the final checkpoint.
    carry_conv_scratch_ref[...] = new_conv_state[:, -1]

  # Apply activation function.
  qkv_out_compact = jax.nn.silu(qkv_out_compact)

  # Step 2: GDN.

  # Prepare gdn weights.
  if cfg.is_kda:
    # KDA reshapes these per (head, channel) rather than padding out to
    # a lane multiple over heads.
    a_log = weights_ref.gdn.a_log[...]
    dt_bias = weights_ref.gdn.dt_bias[...]
  else:
    padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
    a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

  # NOTE: Ideally, we want to move this branching logic into gdn.py. However,
  # load_activation_as_compact and load_activation_as_large leverages vmem ldst.
  # Passing refs into gdn.py breaks strict separation of concerns.
  if cfg.use_recurrent:
    recurrent_fn = (
        compute_kda.recurrent_kda if cfg.is_kda else compute_gdn.recurrent_gdn
    )
    q_compact, k_compact, v_compact, b_compact, a_compact = (
        vmem_ldst.load_activation_as_compact(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )

    out, new_recurrent_state = recurrent_fn(
        q_compact=q_compact,
        k_compact=k_compact,
        v_compact=v_compact,
        b_compact=b_compact,
        a_compact=a_compact,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
    )

  else:
    chunked_fn = (
        compute_kda.chunked_kda if cfg.is_kda else compute_gdn.chunked_gdn
    )
    q_large, k_large, v_large, b_large, a_large = (
        vmem_ldst.load_activation_as_large(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )

    out, new_recurrent_state = chunked_fn(
        q_large=q_large,
        k_large=k_large,
        v_large=v_large,
        b_large=b_large,
        a_large=a_large,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
    )

  # Store output and recurrent to vmem.
  out_slot_ref[...] = out.astype(out_slot_ref.dtype)
  if cfg.state_plan is None:
    recurrent_slot_ref[...] = new_recurrent_state.astype(
        recurrent_slot_ref.dtype
    )
  else:
    for idx in range(cfg.seq_tile_size):

      for t in range(cfg.window_size):
        args = (
            recurrent_slot_ref.at[idx, t],
            cfg.state_plan.recurrent,
            new_recurrent_state[idx, t],
        )
        if carry_recurrent_scratch_ref is None:
          vmem_ldst.store_state_region(*args)
        else:
          pl.when(metadata_ref.p_id_is_last_tile[p_id, idx])(
              functools.partial(vmem_ldst.store_state_region, *args)
          )

  if carry_recurrent_scratch_ref is not None:
    carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
):
  """Setup memory allocations and emit pipeline for running inner_kernel."""
  del conv_state_out_ref, recurrent_state_out_ref

  if cfg.state_plan is not None:
    # Both state regions stream from the single external source ref,
    # which rides the conv-state input slot.
    recurrent_state_ref = conv_state_ref

  qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = (
      memory_ref.create_allocs(
          metadata_ref=metadata_ref,
          qkv_ref=qkv_ref,
          b_ref=b_ref,
          a_ref=a_ref,
          out_ref=out_ref,
          conv_state_ref=conv_state_ref,
          recurrent_state_ref=recurrent_state_ref,
          cfg=cfg,
      )
  )

  num_tiles = metadata_ref.num_tiles[...]

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(
          inner_kernel,
          cfg=cfg,
      ),
      grid=(num_tiles,),
      in_specs=(
          qkv_alloc.spec,
          b_alloc.spec,
          a_alloc.spec,
          conv_alloc.spec,
          recurrent_alloc.spec,
      ),
      out_specs=(out_alloc.spec,),
  )

  @pl.with_scoped(
      allocations=(
          qkv_alloc,
          b_alloc,
          a_alloc,
          conv_alloc,
          recurrent_alloc,
          out_alloc,
      ),
  )
  def _run(allocations):
    pipeline_func(
        qkv_ref,
        b_ref,
        a_ref,
        conv_state_ref,
        recurrent_state_ref,
        out_ref,
        scratches=(
            metadata_ref,
            weights_ref,
            carry_conv_scratch_ref,
            carry_recurrent_scratch_ref,
        ),
        allocations=allocations,
    )

  _run()


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state", "state_source"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "num_spec_tokens",
        "decode_tile_size",
        "mixed_tile_size",
        "zero_initialize_out",
        "compute_precision",
        "state_plan",
        "attention_mode",
        "gate_lower_bound",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
    b: jax.Array,  # [batch_size, n_v]
    a: jax.Array,  # [batch_size, n_v]; [batch_size, n_v * d_k] for KDA
    conv_state: jax.Array | None,  # [num_seqs + 1, kernel_size - 1, dim_size]
    recurrent_state: jax.Array | None,  # [num_seqs + 1, nv, dk, dv]
    conv_weight: jax.Array,  # [kernel_size - 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v] for GDN, [n_v, d_k] for KDA
    query_start_loc: jax.Array,  # [num_seqs + 1]
    state_indices: jax.Array,  # [num_seqs]
    distribution: jax.Array,  # [3]
    seq_lens: jax.Array,  # [num_seqs]
    read_offsets: jax.Array | None = None,  # [num_seqs]
    ckpt_indices: jax.Array | None = None,  # [num_seqs, num_spec_tokens + 1]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    gate_lower_bound: float | None = None,
    state_source: jax.Array | None = None,
    state_plan: config.StateSourcePlan | None = None,
    num_spec_tokens: int = 0,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    # TODO: Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array] | tuple[jax.Array, jax.Array]:
  """Perform conv1d and gdn in a single fused kernel.

  Args:
      qkv: Mixed query, key, value input tensor of shape [batch_size, dim_size],
        where `dim_size = n_kq * d_k * 2 + n_v * d_v`.
      b: b tensor (for beta) of shape [batch_size, n_v].
      a: a tensor (for g) of shape [batch_size, n_v].
      conv_state: Convolution state cache tensor of shape [num_seqs + 1,
        kernel_size - 1, dim_size] containing the last (kernel_size - 1) tokens
        from the last sequence invocation. The first slot is a null block used
        for padded or invalid tokens. It may contain garbage data if it is a
        first invocation of a sequence. None when `state_source` carries the
        states.
      recurrent_state: Recurrent state cache tensor of shape [num_seqs + 1, n_v,
        d_k, d_v]. The first slot is a null block used for padded or invalid
        tokens. It may contain garbage data if it is a first invocation of a
        sequence. None when `state_source` carries the states.
      conv_weight: Convolution weight tensor of shape [kernel_size - 1,
        dim_size].
      conv_bias: Optional convolution bias tensor of shape [dim_size].
      a_log: a_log tensor of shape [n_v].
      dt_bias: dt_bias tensor of shape [n_v]; [n_v, d_k] under KDA, whose gate
        is per-channel. Kept 2D rather than flattened because the KDA path
        reshapes it to broadcast over (head, channel), and Mosaic rejects that
        shape cast from a flat vector ("infer-vector-layout: unsupported shape
        cast").
      query_start_loc: Start locations of sequences of shape [num_seqs + 1].
      state_indices: Indices mapping sequences to state cache slots of shape
        [num_seqs]. With speculative decoding these are the *base* slots of
        per-request groups of `num_spec_tokens + 1` consecutive slots.
      distribution: Tensor of shape [3] int32 — [decode_end, prefill_end,
        mixed_end]. With `num_spec_tokens > 0`, the first segment holds
        speculative verify windows of up to `num_spec_tokens + 1` tokens rather
        than 1-token decodes.
      seq_lens: Sequence lengths for each sequence of shape [num_seqs].
      read_offsets: Optional [num_seqs] int32 — per-sequence state read offset
        (num_accepted - 1 from the last verify step). Windowed sequences read
        their initial state from `state_indices[s] + read_offsets[s]` and write
        one checkpoint per window position to `state_indices[s] + t`. Required
        when `num_spec_tokens > 0`.
      ckpt_indices: Optional [num_seqs, num_spec_tokens + 1] int32 — per-request
        state cache slots for speculative decoding.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key/query dimension.
      d_v: Value dimension.
      kernel_size: Convolution kernel size.
      attention_mode: `GDN` for the per-head scalar gate, `KDA` for the
        per-channel one. Selects which compute path runs.
      gate_lower_bound: KDA only, and must be negative. `None` gives the
        unbounded gate `-exp(A_log) * softplus(g + dt_bias)` (Kimi-Linear-48B);
        a float gives the bounded `gate_lower_bound * sigmoid(exp(A_log) * (g +
        dt_bias))` (every K3 variant, at -5.0). Different functions of the same
        weights, so a mismatch with the model's KDA config silently changes the
        decay.
      state_source: Optional indexed external state source (e.g. the unified KV
        pool) replacing the dense state tensors: both state regions are streamed
        between it and the kernel pipeline per `state_plan`, with `state_indices`
        addressing source block windows. Donanted and returned in place of the
        dense states.
      state_plan: Static copy-plan describing the source's state regions. Must be
        set iff `state_source` is set.
      num_spec_tokens: Number of speculative draft tokens (0 gives the plain
        1-token-per-sequence decode path).
      zero_initialize_out: Whether to zero-initialize the output buffer before
        executing non-batched sequences.
      compute_precision: Computation precision dtype.
      decode_tile_size: Tile size along sequence dimension for decode sequences.
      mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
        sequences.

  Returns:
      (new_conv_state, new_recurrent_state): Updated convolution state cache and
          recurrent state cache tensors. With a state source: the updated
          source tensor and output activations.
      out_act: Output activation tensor of shape [batch_size, n_v, d_v].
  """

  # Precondition checks.
  assert (conv_state is not None) == (recurrent_state is not None)
  pooled = state_source is not None
  conv_out_dtype = None
  recurrent_out_dtype = None
  conv_state_shape = None
  if pooled:
    assert conv_state is None and recurrent_state is None
    assert state_plan is not None, "state_source requires state_plan"
    assert state_source.ndim == 3, state_source.shape
    # Checkpoints are individually addressable source blocks; without
    # the index array they would all alias onto the slot's own block.
    assert num_spec_tokens == 0 or ckpt_indices is not None, (
        "num_spec_tokens > 0 requires ckpt_indices"
    )
  else:
    assert state_source is None
    assert conv_state is not None and recurrent_state is not None
    conv_out_dtype = conv_state.dtype
    recurrent_out_dtype = recurrent_state.dtype
    conv_state = conv_state.astype(jnp.float32)

  qkv = qkv.astype(jnp.float32)
  b = b.astype(jnp.float32)
  a = a.astype(jnp.float32)

  # Step 1: Validate inputs.
  is_kda = attention_mode == config.AttentionMode.KDA
  if is_kda and num_spec_tokens > 0:
    # The KDA compute path returns a single end-of-tile state, not one
    # checkpoint per window position, so it cannot roll back rejected
    # draft tokens.
    raise NotImplementedError(
        "KDA does not support speculative decoding yet "
        f"(num_spec_tokens={num_spec_tokens})."
    )
  if gate_lower_bound is not None:
    if not is_kda:
      raise ValueError(
          "gate_lower_bound only applies to KDA; got"
          f" attention_mode={attention_mode}."
      )
    # The chunked solve needs every causal exponent g_r - g_t to stay
    # non-positive, which `gate_lower_bound * sigmoid(.)` only satisfies
    # for a negative bound.
    if gate_lower_bound >= 0:
      raise ValueError(
          "gate_lower_bound must be negative (it bounds a log-decay); got"
          f" {gate_lower_bound}. A non-negative bound makes the gate increase"
          " along the chunk, which this kernel does not support."
      )

  num_seqs = state_indices.size
  batch_size, dim = qkv.shape
  assert conv_weight.shape == (dim, 1, kernel_size)
  if conv_bias is not None:
    assert conv_bias.shape == (dim,)
  assert query_start_loc.shape == (num_seqs + 1,)
  assert state_indices.shape == (num_seqs,)
  assert distribution.shape == (3,)
  if num_spec_tokens > 0:
    assert read_offsets is not None, (
        "read_offsets is required when num_spec_tokens > 0"
    )
  if read_offsets is None:
    read_offsets = jnp.zeros((num_seqs,), dtype=jnp.int32)
  assert read_offsets.shape == (num_seqs,)
  read_offsets = read_offsets.astype(jnp.int32)
  if ckpt_indices is not None:
    # One source block per checkpoint (vLLM's ssm_state_indices scheme):
    # checkpoint t of sequence s lives at block ckpt_indices[s, t], so
    # the pool block only ever has to hold a single state.
    assert ckpt_indices.shape == (
        num_seqs,
        num_spec_tokens + 1,
    ), (ckpt_indices.shape, num_seqs, num_spec_tokens)
    ckpt_indices = ckpt_indices.astype(jnp.int32)
  act_in_dtype = qkv.dtype
  assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

  num_lanes = pltpu.get_tpu_info().num_lanes
  packing = 4 // act_in_dtype.itemsize
  padded_batch_size = pl.cdiv(batch_size, packing) * packing
  aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

  # In standard GDN mode without external pooled states and without speculative
  # decoding, leverage the dynamic VMEM tiling heuristics.
  if not pooled and num_spec_tokens == 0 and not is_kda:
    assert conv_state is not None and recurrent_state is not None
    conv_state_dim_size = conv_state.shape[-1]
    decode_tile_size, mixed_tile_size = tiling.get_tile_sizes(
        batch_size=batch_size,
        num_seqs=num_seqs,
        padded_batch_size=padded_batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        conv_state_dim_size=conv_state_dim_size,
        act_in_dtype=act_in_dtype,
        act_out_dtype=act_in_dtype,
        conv_state_dtype=conv_state.dtype,
        recurrent_state_dtype=recurrent_state.dtype,
        num_lanes=num_lanes,
        decode_tile_size=decode_tile_size,
        mixed_tile_size=mixed_tile_size,
    )
  else:
    decode_tile_size = min(decode_tile_size, batch_size)
    mixed_tile_size = min(mixed_tile_size, batch_size)

  if num_spec_tokens > 0:
    # A verify window holds one state checkpoint per window position per
    # sequence in VMEM, which multiplies the per-sequence footprint by
    # the window size. Shrink the tile so the double-buffered windows
    # fit in roughly half the scoped-VMEM budget (the rest goes to
    # weights, activations scratch and compiler temporaries).
    window = num_spec_tokens + 1
    num_buffers = config.GDNConfig.__dataclass_fields__["num_buffers"].default
    bytes_per_seq = window * (
        # Recurrent checkpoints (fp32) — the dominant term.
        n_v * d_k * d_v * 4
        # Conv checkpoints (fp32).
        + (kernel_size - 1) * dim * 4
        # qkv (fp32), b/a (fp32), out (act_out).
        + dim * 4
        + 2 * aligned_num_v_heads * 4
        + n_v * d_v * 2
    )
    vmem_budget = int(
        config.GDNConfig.WINDOWED_VMEM_FRACTION
        * pltpu.get_tpu_info().vmem_capacity_bytes
    )
    spec_tile_budget = (vmem_budget // 2) // num_buffers
    decode_tile_size = max(
        1, min(decode_tile_size, spec_tile_budget // bytes_per_seq)
    )

  batch_padding_size = padded_batch_size - batch_size
  num_v_padding_size = aligned_num_v_heads - n_v
  # KDA's gate carries d_k channels per value head, so it pads out to a
  # lane multiple of n_v * d_k rather than of n_v.
  gate_dim = n_v * d_k if is_kda else n_v
  gate_padding_size = (pl.cdiv(gate_dim, num_lanes) * num_lanes) - gate_dim
  qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
  b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
  a = jnp.pad(a, ((0, batch_padding_size), (0, gate_padding_size)))

  qkv = qkv.reshape(padded_batch_size, 1, -1)
  b = b.reshape(padded_batch_size, 1, -1)
  a = a.reshape(padded_batch_size, 1, -1)

  # Step 3: States and weights pre-processing.
  # TODO: To eliminate runtime cost, move this logic into model
  # loading stage.
  if not pooled:
    assert conv_state is not None
    conv_state_shape = conv_state.shape
    conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
  conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
  conv_bias = conv_bias.astype(jnp.float32) if conv_bias is not None else None

  # Step 4: Wrap inputs for the kernel.
  conv_weights = memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias)
  gdn_weights = memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
  weights = memory_ref.WeightRefs(conv=conv_weights, gdn=gdn_weights)

  # Step 5: Create specs.
  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

  def call_kernel(
      in_conv_state: jax.Array | None,
      in_recurrent_state: jax.Array | None,
      in_act: jax.Array | None,
      mode: config.GDNMode,
  ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    # With a state source, it rides the conv-state slot and the dense
    # recurrent-state input is absent.
    if mode == config.GDNMode.PER_SEQ:
      tile_size = mixed_tile_size
      # Prefill / mixed sequences keep a single state checkpoint.
      window_size = 1
    else:
      tile_size = decode_tile_size
      window_size = num_spec_tokens + 1

    if pooled:
      recurrent_dtype = jnp.float32.dtype
      conv_dtype = jnp.float32.dtype
    else:
      assert in_recurrent_state is not None
      assert in_conv_state is not None
      recurrent_dtype = in_recurrent_state.dtype
      conv_dtype = in_conv_state.dtype

    cfg = config.GDNConfig(
        mode=mode,
        batch_size=padded_batch_size,
        kernel_size=kernel_size,
        tile_size=tile_size,
        window_size=window_size,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        attention_mode=attention_mode,
        gate_lower_bound=gate_lower_bound,
        dtypes=config.Dtypes(
            act_in=act_in_dtype,
            act_out=act_in_dtype,
            compute=compute_precision,
            recurrent_state=recurrent_dtype,
            conv_state=conv_dtype,
        ),
        state_plan=state_plan,
    )

    # Step 6: Metadata preprocessing. Will be executed multiple times per-layer
    # but will be CSEed by compiler.
    if mode != config.GDNMode.PER_SEQ:
      metadata_obj = metadata.compute_batched_seq_metadata(
          cfg=cfg,
          seq_lens=seq_lens,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          read_offsets=read_offsets,
          end_seq=distribution[0],
          ckpt_indices=ckpt_indices,
      )
    else:
      metadata_obj = metadata.compute_per_seq_metadata(
          cfg=cfg,
          seq_lens=seq_lens,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          read_offsets=read_offsets,
          start_seq=distribution[0],
          end_seq=distribution[-1],
          ckpt_indices=ckpt_indices,
      )

    metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)

    # Step 7: Handle case where write needs to be done in existing out.
    # Flat aliasing indices account for the absent recurrent-state
    # input in source mode (None contributes no leaves).
    in_out_spec = None
    num_state_inputs = 1 if pooled else 2
    input_output_aliases = {len(metadata_obj) + 3: 1}
    if not pooled:
      input_output_aliases[len(metadata_obj) + 4] = 2
    out_shape = cfg.get_out_shape()

    if in_act is None and zero_initialize_out:
      in_act = jnp.zeros_like(out_shape)
    if in_act is not None:
      out_shape = in_act
      in_out_spec = hbm_spec
      input_output_aliases[len(metadata_obj) + 3 + num_state_inputs] = 0

    return pl.pallas_call(
        functools.partial(outer_kernel, cfg=cfg),
        out_shape=(out_shape, in_conv_state, in_recurrent_state),
        in_specs=(
            metadata_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            None if pooled else hbm_spec,
            in_out_spec,
            weights_spec,
        ),
        out_specs=(hbm_spec, hbm_spec, None if pooled else hbm_spec),
        scratch_shapes=cfg.get_scratch_shape_dict(),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=cfg.get_vmem_limit_bytes(),
        ),
        name=cfg.get_kernel_name(),
        metadata=cfg.get_metadata(),  # pyrefly: ignore[bad-argument-type]
    )(
        metadata_obj,
        qkv,
        b,
        a,
        in_conv_state,
        in_recurrent_state,
        in_act,
        weights,
    )

  if pooled:
    out_act, out_source, _ = call_kernel(
        state_source, None, None, config.GDNMode.BATCHED
    )
    out_act, out_source, _ = call_kernel(
        out_source, None, out_act, config.GDNMode.PER_SEQ
    )
    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    return out_source, out_act

  assert conv_state is not None and recurrent_state is not None
  assert conv_out_dtype is not None and recurrent_out_dtype is not None and conv_state_shape is not None
  # The first segment holds verify windows of up to `num_spec_tokens + 1`
  # tokens, or plain 1-token decodes without speculative decoding.
  out_act, out_conv_state, out_recurrent_state = call_kernel(
      conv_state, recurrent_state, None, config.GDNMode.BATCHED
  )
  out_act, out_conv_state, out_recurrent_state = call_kernel(
      out_conv_state, out_recurrent_state, out_act, config.GDNMode.PER_SEQ
  )

  assert out_recurrent_state is not None
  out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
  out_conv_state = out_conv_state.astype(conv_out_dtype)
  out_conv_state = out_conv_state.reshape(conv_state_shape)
  out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

  return (out_conv_state, out_recurrent_state), out_act
