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
"""Grouped decode pipelines for fused GDN."""

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import compute_gdn
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import compute as fused_compute
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import memory as fused_memory

# Minimum total bucket tokens for the wide-layout decode specializations.
_MIN_SPECIALIZED_BATCH_TOKENS = 64


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DecodeGroupRefs:
  """Decode-group VMEM staging buffers and DMA semaphores.

  Inputs use two parity slots: qkv [2, aligned_group_rows, width] and b/a [2,
  aligned_group_rows, n_v], in the activation dtype; input_sem is [2, 3].
  conv is [2, group, max(1, kernel_size - 1), width] in the cache dtype; its
  load/store semaphores are [2, group]. The one-tap history row is unused.
  Each member owns one to three state slots [depth, n_v, 128, 128] in the
  cache dtype, with [depth] load/store semaphores. extra_state holds members
  after zero; member0_state is None only when borrowing FP32 head-major QKV
  scratch. BF16 caches need separate storage because DMA cannot cast.
  """

  qkv: Any
  b: Any
  a: Any
  input_sem: Any
  conv: Any
  conv_load_sem: Any
  conv_store_sem: Any
  extra_state: Any
  state_load_sem: Any
  state_store_sem: Any
  member0_state: Any = None


def _fresh_or_resumed(uniform_fresh, fresh, resumed):
  """Specialize only when every active decode owner is fresh.

  None disables specialization; False allows mixed fresh/resumed owners.

  Args:
    uniform_fresh: Whether every decode owner is fresh; None disables this
      specialization.
    fresh: Zero-argument callback for the all-fresh path.
    resumed: Zero-argument callback for the normal path, which may mask fresh
      owners.

  Returns:
    The selected callback's result; both callbacks must return matching
    structures.
  """
  if uniform_fresh is None:
    return resumed()
  return jax.lax.cond(uniform_fresh, fresh, resumed)


def _decode_inner(
    qkv_member_ref: Any,
    b_member_ref: Any,
    a_member_ref: Any,
    conv_member_ref: Any,
    recurrent_slot_ref: Any,
    output_carry_ref: Any,
    weights_ref: Any,
    dense_conv_ref: tuple[Any, Any | None],
    grouped_state_scratch_ref: Any,
    *,
    cfg: config.GDNConfig,
    has_initial_state: jax.Array,
    output_offset: jax.Array,
    input_offset: jax.Array,
    uniform_fresh: jax.Array | None = None,
) -> None:
  """Advance one decode member after its input/cache DMAs complete.

  Uses DecodeGroupRefs layouts, a single-token cfg, and available output
  carry. False has_initial_state discards loaded history. uniform_fresh=True
  skips history reads for the whole prefix; None disables that
  specialization. grouped_state_scratch_ref is [n_kq, 128, v_per_kq * 128]
  FP32.

  Args:
    qkv_member_ref: Private aligned packed-QKV window containing this decode
      member.
    b_member_ref: Private aligned beta-input window containing this decode
      member.
    a_member_ref: Private aligned decay-input window containing this decode
      member.
    conv_member_ref: Mutable convolution-history buffer for one decode group
      member.
    recurrent_slot_ref: Mutable recurrent-state slice for the current
      request/member.
    output_carry_ref: Mutable [16, n_v * d_v] partial output block in activation
      dtype.
    weights_ref: VMEM refs for log decay weights and timestep biases.
    dense_conv_ref: FP32 convolution tap and optional bias refs in head-major
      layout.
    grouped_state_scratch_ref: Mutable FP32 scratch [n_kq, 128, (n_v // n_kq) *
      128].
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    has_initial_state: Whether this request resumes history instead of starting
      from zero.
    output_offset: Decode row position or prefill starting offset within the
      16-row carry.
    input_offset: Logical token offset within the private eight-row-aligned
      input window.
    uniform_fresh: Whether every decode owner is fresh; None disables this
      specialization.
  """

  def load_previous():
    """Read the previous recurrent state while substituting zero for a fresh
    request.

    Returns:
      FP32 recurrent state, or zeros of the same shape for a fresh request.
    """
    return jnp.where(
        has_initial_state,
        recurrent_slot_ref[...].astype(jnp.float32),
        jnp.float32(0),
    )

  specialize_fresh_tail = (
      uniform_fresh is not None
      and qkv_member_ref.dtype == jnp.bfloat16
      and conv_member_ref.dtype == jnp.bfloat16
      and recurrent_slot_ref.dtype in (jnp.bfloat16, jnp.float32)
  )
  if not specialize_fresh_tail:
    prev_recurrent = _fresh_or_resumed(
        uniform_fresh,
        lambda: jnp.zeros(recurrent_slot_ref.shape, dtype=jnp.float32),
        load_previous,
    )
  weight_ref, bias_ref = dense_conv_ref
  half = jnp.asarray(0.5, dtype=jnp.float32)
  activation_slabs = []

  packed_heads = 2 * cfg.num_kq_heads + cfg.num_v_heads
  head_dim = cfg.kq_head_dim
  history_size = cfg.kernel_size - 1
  for head_start in range(0, packed_heads, 16):
    head_end = min(head_start + 16, packed_heads)

    inputs = []
    if history_size:

      def load_history():
        """Read a head slab's convolution history, masking fresh requests to
        zero.

        Returns:
          FP32 history [history_rows, slab_heads, head_dim], zeroed for a fresh
          request.
        """
        return jnp.where(
            has_initial_state,
            conv_member_ref[
                :, head_start * head_dim : head_end * head_dim
            ].astype(jnp.float32),
            jnp.float32(0),
        ).reshape(history_size, head_end - head_start, head_dim)

      history = _fresh_or_resumed(
          uniform_fresh,
          lambda: jnp.zeros(
              (history_size, head_end - head_start, head_dim), jnp.float32
          ),
          load_history,
      )
      inputs = [history[tap] for tap in range(history_size)]
    # Select the token in native layout; TPU roll needs a nonnegative shift.
    values = jnp.concatenate(
        [
            pltpu.roll(
                qkv_member_ref[
                    :, head * head_dim : (head + 1) * head_dim
                ].astype(jnp.float32),
                shift=(-input_offset) % qkv_member_ref.shape[0],
                axis=0,
            )[:1, :]
            for head in range(head_start, head_end)
        ],
        axis=0,
    )

    inputs.append(values)
    result = inputs[0] * weight_ref[0, head_start:head_end, :]
    for tap in range(1, cfg.kernel_size):
      result = result + inputs[tap] * weight_ref[tap, head_start:head_end, :]
    if bias_ref is not None:
      result = result + bias_ref[head_start:head_end, :]
    activated = fused_compute.conv_silu_activation(
        result,
        half,
        cast_before_norm=False,
        head_start=head_start,
        n_kq=cfg.num_kq_heads,
    )
    activation_slabs.append(activated)

    if history_size:
      conv_member_ref[:, head_start * head_dim : head_end * head_dim] = (
          jnp.stack(inputs[1:], axis=0)
          .reshape(history_size, (head_end - head_start) * head_dim)
          .astype(conv_member_ref.dtype)
      )

  if cfg.num_kq_heads == 16 and cfg.num_v_heads == 32:
    # Q and K each occupy one complete slab in the 16/32-head layout.
    q = activation_slabs[0].reshape(cfg.num_kq_heads, 1, cfg.kq_head_dim)
    k = activation_slabs[1].reshape(cfg.num_kq_heads, 1, cfg.kq_head_dim)
    v = jnp.concatenate(activation_slabs[2:], axis=0).reshape(
        cfg.num_v_heads, 1, cfg.v_head_dim
    )
  else:
    activated = jnp.concatenate(activation_slabs, axis=0)
    q = activated[: cfg.num_kq_heads].reshape(cfg.num_kq_heads, 1, head_dim)
    k = activated[cfg.num_kq_heads : 2 * cfg.num_kq_heads].reshape(
        cfg.num_kq_heads, 1, head_dim
    )
    v = activated[2 * cfg.num_kq_heads :].reshape(cfg.num_v_heads, 1, head_dim)

  b_values = fused_compute.native_gate_values(b_member_ref, input_offset, 1)
  a_values = fused_compute.native_gate_values(a_member_ref, input_offset, 1)
  a_log = weights_ref.a_log[...].astype(jnp.float32).reshape(1, cfg.num_v_heads)
  dt_bias = (
      weights_ref.dt_bias[...].astype(jnp.float32).reshape(1, cfg.num_v_heads)
  )
  beta = jax.nn.sigmoid(b_values)
  gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_values + dt_bias)
  gating = jnp.exp(gating_log)
  beta = beta.reshape(cfg.num_v_heads, 1, 1)
  gating = gating.reshape(cfg.num_v_heads, 1, 1)

  def project_fresh():
    # Skip the zero-state dot, but preserve its NaNs for nonfinite Q/K.
    """Construct projections for a zero initial state without reading the cache.

    Returns:
      Zero-state K and Q projections, each [n_v, 1, d_v], preserving nonfinite
      propagation.
    """

    def zero_projection(vector):
      """Return zero for finite Q/K projections while preserving nonfinite
      propagation.

      Args:
        vector: Normalized query or key used to form a zero-state projection.

      Returns:
        Projection [n_v, 1, d_v], zero for finite inputs and NaN for nonfinite
        inputs.
      """
      values = jnp.where(
          jnp.all(jnp.isfinite(vector), axis=-1, keepdims=True),
          jnp.float32(0),
          jnp.float32(jnp.nan),
      )
      values = jnp.repeat(values, cfg.v_per_kq_head, axis=0)
      return jnp.broadcast_to(values, (cfg.num_v_heads, 1, cfg.v_head_dim))

    return zero_projection(k), zero_projection(q)

  def project_resumed(state_prev):
    # DEFAULT and HIGHEST differ in product rounding; both accumulate in FP32.
    """Project resumed state onto K and Q with the selected product precision.

    Args:
      state_prev: FP32 prior recurrent state [n_v, d_k, d_v].

    Returns:
      K and Q projections of the previous state, each [n_v, 1, d_v].
    """
    use_default_projection = (
        cfg.kq_head_dim == 128
        and cfg.v_head_dim == 128
        and cfg.kernel_size == 4
        and qkv_member_ref.dtype == jnp.bfloat16
        and conv_member_ref.dtype == jnp.bfloat16
        and recurrent_slot_ref.dtype == jnp.bfloat16
    )
    if (
        use_default_projection
        and cfg.batch_size >= _MIN_SPECIALIZED_BATCH_TOKENS
        and cfg.num_kq_heads == 16
        and cfg.num_v_heads == 64
    ):
      grouped_state = fused_compute.pack_value_heads(
          state_prev, cfg.v_per_kq_head
      )
    else:
      grouped_state_scratch_ref[...] = fused_compute.pack_value_heads(
          state_prev, cfg.v_per_kq_head
      )
      grouped_state = grouped_state_scratch_ref[...]
    projections = jax.lax.dot(
        jnp.concatenate([k, q], axis=1),
        grouped_state,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=(
            jax.lax.Precision.DEFAULT
            if use_default_projection
            else jax.lax.Precision.HIGHEST
        ),
        preferred_element_type=jnp.float32,
    )
    projections = fused_compute.unpack_value_heads(
        projections, cfg.v_per_kq_head
    )
    return tuple(jnp.split(projections, 2, axis=1))

  def finish(state_prev, k_S, q_S):
    """Compute the gated delta output and updated recurrent state.

    Args:
      state_prev: Prior FP32 state, or broadcastable zeros for the fresh
        specialization.
      k_S: Projection of the previous state onto the normalized key.
      q_S: Projection of the previous state onto the normalized, scaled query.
    """
    v_new = beta * (v - gating * k_S)
    q_dot_k = jnp.sum(q * k, axis=-1, keepdims=True, dtype=jnp.float32)
    q_dot_k = jnp.repeat(q_dot_k, cfg.v_per_kq_head, axis=0)
    out = gating * q_S + q_dot_k * v_new

    k_t = compute_gdn.fused_transpose_broadcast(k, src_dim=2, dst_dim=1)
    k_t = jnp.repeat(k_t, cfg.v_per_kq_head, axis=0)
    new_recurrent = state_prev * gating + k_t * v_new
    recurrent_slot_ref[...] = new_recurrent[None, ...].astype(
        recurrent_slot_ref.dtype
    )
    fused_compute.store_decode_output(out, output_carry_ref, output_offset)

  if specialize_fresh_tail:

    def advance_fresh():
      # Keep 0 * gating and its NaN/Inf behavior without a full zero-state matrix.
      """Build the new recurrent state from the fresh-request update alone."""
      finish(jnp.zeros_like(gating), *project_fresh())

    def advance_resumed():
      """Add the update to the decayed resumed state."""
      state_prev = load_previous()[0].astype(jnp.float32)
      finish(state_prev, *project_resumed(state_prev))

    jax.lax.cond(uniform_fresh, advance_fresh, advance_resumed)
  else:
    state_prev = prev_recurrent[0].astype(jnp.float32)
    k_S, q_S = _fresh_or_resumed(
        uniform_fresh, project_fresh, lambda: project_resumed(state_prev)
    )
    finish(state_prev, k_S, q_S)


def _decode_input_transfer(
    qkv_ref: Any,
    b_ref: Any,
    a_ref: Any,
    group_qkv_ref: Any,
    group_b_ref: Any,
    group_a_ref: Any,
    input_sem_ref: Any,
    group: jax.Array,
    decode_count: jax.Array,
    *,
    group_width: int,
    wait: bool,
) -> None:
  """Start or wait for contiguous decode-input DMA, including a partial group.

  Use DecodeGroupRefs layouts. The group must have an active request; start
  only into a free parity slot and match each wait to its start.

  Args:
    qkv_ref: HBM packed activations [tokens, 2 * n_kq * d_k + n_v * d_v].
    b_ref: HBM beta-gate inputs [tokens, n_v], with the activation dtype.
    a_ref: HBM decay-gate inputs [tokens, n_v], with the activation dtype.
    group_qkv_ref: Parity-buffered private packed-QKV windows.
    group_b_ref: Parity-buffered private beta-input windows.
    group_a_ref: Parity-buffered private decay-input windows.
    input_sem_ref: DMA semaphores for each parity slot's QKV and two gate
      transfers.
    group: Zero-based decode group index.
    decode_count: Number of active one-token requests in the decode prefix.
    group_width: Number of decode members per group, between one and eight.
    wait: True waits for the matching transfer; False starts it.
  """
  base = group * group_width
  rows = jnp.minimum(jnp.int32(group_width), decode_count - base)
  parity = group % 2

  def branch(row_count: int) -> Any:
    """Build a statically sized transfer branch for a possible active row count.

    Args:
      row_count: Static number of active rows in the decode transfer branch.

    Returns:
      Zero-argument callback starting or waiting for that branch's transfers.
    """

    def transfer() -> None:
      # Load the aligned window once; select each member in registers.
      """Start or wait for the aligned QKV and gate transfers."""
      aligned_base = pl.multiple_of((base // 8) * 8, 8)
      native_rows = pl.multiple_of(
          ((base - aligned_base + row_count + 7) // 8) * 8, 8
      )
      fused_memory.dma(
          qkv_ref.at[pl.ds(aligned_base, native_rows), :],
          group_qkv_ref.at[parity, pl.ds(0, native_rows), :],
          input_sem_ref.at[parity, 0],
          wait=wait,
      )
      for operand, (source, destination) in enumerate(
          (
              (b_ref, group_b_ref),
              (a_ref, group_a_ref),
          ),
          start=1,
      ):
        fused_memory.dma(
            source.at[pl.ds(aligned_base, native_rows), :],
            destination.at[parity, pl.ds(0, native_rows), :],
            input_sem_ref.at[parity, operand],
            wait=wait,
        )

    return transfer

  jax.lax.switch(
      rows - 1,
      tuple(branch(row_count) for row_count in range(1, group_width + 1)),
  )


def run_grouped_decode(
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
    recurrent_state_out_ref: Any,
    weights_ref: Any,
    dense_conv_ref: tuple[Any, Any | None],
    out_ref: Any,
    qkv_head_major_scratch_ref: Any,
    grouped_state_scratch_ref: Any,
    output_carry_ref: Any,
    output_sem_ref: Any,
    decode_refs: DecodeGroupRefs,
    *,
    cfg: config.GDNConfig,
) -> None:
  """Run grouped decode, specializing eligible all-fresh prefixes.

  The scan stops at the first resumed owner; the general path masks each
  request's history independently.

  Args:
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
    recurrent_state_out_ref: HBM recurrent-cache destination alias, written only
      at owned slots.
    weights_ref: VMEM refs for log decay weights and timestep biases.
    dense_conv_ref: FP32 convolution tap and optional bias refs in head-major
      layout.
    out_ref: HBM output array [tokens, n_v * d_v], written by the pipeline.
    qkv_head_major_scratch_ref: Shared FP32 head-major scratch; decode may
      borrow it before prefill.
    grouped_state_scratch_ref: Mutable FP32 scratch [n_kq, 128, (n_v // n_kq) *
      128].
    output_carry_ref: Mutable [16, n_v * d_v] partial output block in activation
      dtype.
    output_sem_ref: DMA semaphores for the output scratch slots.
    decode_refs: Grouped activation/state buffers and semaphores described by
      DecodeGroupRefs.
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
  """
  args = (
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
      qkv_head_major_scratch_ref,
      grouped_state_scratch_ref,
      output_carry_ref,
      output_sem_ref,
      decode_refs,
  )
  if (
      cfg.batch_size >= _MIN_SPECIALIZED_BATCH_TOKENS
      and cfg.kq_head_dim == 128
      and cfg.v_head_dim == 128
      and cfg.kernel_size == 4
      and (
          recurrent_state_ref.dtype == jnp.float32
          or (
              qkv_ref.dtype == jnp.bfloat16
              and conv_state_ref.dtype == jnp.bfloat16
              and recurrent_state_ref.dtype == jnp.bfloat16
          )
      )
      and cfg.num_kq_heads == 16
      and cfg.num_v_heads == 64
  ):

    def more(carry):
      """Return whether the all-fresh scan should continue.

      Args:
        carry: Scan carry (next request index, whether all inspected owners are
          fresh).

      Returns:
        Whether another request remains and all previously inspected requests
        are fresh.
      """
      owner, all_fresh = carry
      return (owner < distribution_ref[0]) & all_fresh

    def inspect(carry):
      """Check the next request's history length during that scan.

      Args:
        carry: Scan carry (next request index, whether all inspected owners are
          fresh).

      Returns:
        Next request index and whether the inspected request has no prior
        history.
      """
      owner, _ = carry
      return owner + 1, seq_lens_ref[owner] <= 1

    _, all_fresh = jax.lax.while_loop(
        more, inspect, (jnp.int32(0), jnp.bool_(True))
    )

    _run_grouped_decode(*args, cfg=cfg, uniform_fresh=all_fresh)
  else:
    _run_grouped_decode(*args, cfg=cfg)


def _run_grouped_decode(
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
    recurrent_state_out_ref: Any,
    weights_ref: Any,
    dense_conv_ref: tuple[Any, Any | None],
    out_ref: Any,
    qkv_head_major_scratch_ref: Any,
    grouped_state_scratch_ref: Any,
    output_carry_ref: Any,
    output_sem_ref: Any,
    decode_refs: DecodeGroupRefs,
    *,
    cfg: config.GDNConfig,
    uniform_fresh: jax.Array | None = None,
) -> None:
  """Run a validated, contiguous one-token prefix in groups of up to eight.

  Cache write slots must be distinct; scratch/semaphores available and output
  carry initialized. decode_refs follows DecodeGroupRefs. FP32 member zero
  may borrow [depth, n_v, 128, 128] from head-major prefill scratch because
  their lifetimes are disjoint; BF16 states cannot use this DMA destination.

  Args:
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
    recurrent_state_out_ref: HBM recurrent-cache destination alias, written only
      at owned slots.
    weights_ref: VMEM refs for log decay weights and timestep biases.
    dense_conv_ref: FP32 convolution tap and optional bias refs in head-major
      layout.
    out_ref: HBM output array [tokens, n_v * d_v], written by the pipeline.
    qkv_head_major_scratch_ref: Shared FP32 head-major scratch; decode may
      borrow it before prefill.
    grouped_state_scratch_ref: Mutable FP32 scratch [n_kq, 128, (n_v // n_kq) *
      128].
    output_carry_ref: Mutable [16, n_v * d_v] partial output block in activation
      dtype.
    output_sem_ref: DMA semaphores for the output scratch slots.
    decode_refs: Grouped activation/state buffers and semaphores described by
      DecodeGroupRefs.
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    uniform_fresh: Whether every decode owner is fresh; None disables this
      specialization.
  """
  decode_count = distribution_ref[0]
  # Semaphore count bounds ring depth; extra scratch panels have no semaphore.
  state_buffers = decode_refs.state_load_sem[0].shape[0]
  if decode_refs.member0_state is not None:
    resident_state = decode_refs.member0_state
  elif (
      cfg.num_kq_heads == 16
      and cfg.num_v_heads == 32
      and qkv_head_major_scratch_ref.shape == (64, 128, 128)
  ):
    resident_state = qkv_head_major_scratch_ref.reshape((2, 32, 128, 128))
  elif qkv_head_major_scratch_ref.shape[1] == cfg.kq_head_dim:
    resident_state = qkv_head_major_scratch_ref.at[
        : state_buffers * cfg.num_v_heads, :, :
    ].reshape((state_buffers, cfg.num_v_heads, cfg.kq_head_dim, cfg.v_head_dim))
  else:
    # Decode borrows prefill scratch. Flatten it to avoid gaps between panels.
    resident_state = (
        qkv_head_major_scratch_ref.reshape((-1, cfg.v_head_dim))
        .at[: state_buffers * cfg.num_v_heads * cfg.kq_head_dim, :]
        .reshape(
            (state_buffers, cfg.num_v_heads, cfg.kq_head_dim, cfg.v_head_dim)
        )
    )
  state_refs = (
      resident_state,
      *decode_refs.extra_state,
  )
  group_width = len(state_refs)
  defer_fresh_cache_waits = (
      uniform_fresh is not None
      and qkv_ref.dtype == jnp.bfloat16
      and conv_state_ref.dtype == jnp.bfloat16
      and recurrent_state_ref.dtype == jnp.float32
      and all(ref.shape[0] <= 2 for ref in state_refs)
  )
  can_flush_output = out_ref.shape[0] >= fused_compute.OUTPUT_ALIGNMENT
  groups = (decode_count + group_width - 1) // group_width

  def state_transfer(
      group: jax.Array,
      member: int,
      *,
      to_hbm: bool,
      wait: bool,
  ) -> None:
    """Map one member's recurrent read/write slot and ring index to a transfer.

    Args:
      group: Zero-based decode group index.
      member: Static member index within a decode group.
      to_hbm: True stores VMEM state to HBM; False loads HBM state into VMEM.
      wait: True waits for the matching transfer; False starts it.
    """
    owner = group * group_width + member
    state_slot = state_indices_ref[owner]
    read_slot = jnp.where(
        state_slot > 0,
        read_state_indices_ref[owner] + read_offsets_ref[owner],
        jnp.int32(0),
    )
    read_slot = jnp.clip(read_slot, 0, recurrent_state_ref.shape[0] - 1)
    buffers = state_refs[member].shape[0]
    buffer = group % buffers
    hbm_state = recurrent_state_out_ref if to_hbm else recurrent_state_ref
    hbm_slot = hbm_state.at[state_slot if to_hbm else read_slot]
    vmem_slot = state_refs[member].at[buffer]

    def transfer_state() -> None:
      """Issue or wait for the selected recurrent load/store."""
      fused_memory.bidirectional_dma(
          hbm_slot,
          vmem_slot,
          decode_refs.state_load_sem[member].at[buffer],
          decode_refs.state_store_sem[member].at[buffer],
          to_hbm=to_hbm,
          wait=wait,
      )

    if uniform_fresh is not None and not to_hbm:

      @pl.when(~uniform_fresh)
      def transfer_resumed_prefix_state() -> None:
        """Transfer recurrent history unless the entire decode prefix is fresh."""
        transfer_state()

    else:
      transfer_state()

  def conv_transfer(
      group: jax.Array,
      member: int,
      *,
      to_hbm: bool,
      wait: bool,
  ) -> None:
    """Map one member's convolution cache to its parity buffer; skip one-tap
    history.

    Args:
      group: Zero-based decode group index.
      member: Static member index within a decode group.
      to_hbm: True stores VMEM state to HBM; False loads HBM state into VMEM.
      wait: True waits for the matching transfer; False starts it.
    """
    if cfg.kernel_size == 1:
      return
    owner = group * group_width + member
    state_slot = state_indices_ref[owner]
    read_slot = jnp.where(
        state_slot > 0,
        read_state_indices_ref[owner] + read_offsets_ref[owner],
        jnp.int32(0),
    )
    read_slot = jnp.clip(read_slot, 0, conv_state_ref.shape[0] - 1)
    parity = group % 2
    hbm_slot = conv_state_ref.at[state_slot if to_hbm else read_slot]
    vmem_slot = decode_refs.conv.at[parity, member]

    def transfer_conv() -> None:
      """Issue or wait for the selected convolution transfer."""
      fused_memory.bidirectional_dma(
          hbm_slot,
          vmem_slot,
          decode_refs.conv_load_sem.at[parity, member],
          decode_refs.conv_store_sem.at[parity, member],
          to_hbm=to_hbm,
          wait=wait,
      )

    if uniform_fresh is not None and not to_hbm:

      @pl.when(~uniform_fresh)
      def transfer_resumed_prefix_conv() -> None:
        """Transfer convolution history unless the entire decode prefix is fresh."""
        transfer_conv()

    else:
      transfer_conv()

  def input_transfer(group: jax.Array, *, wait: bool) -> None:
    """Delegate the group's activation-window transfer.

    Args:
      group: Zero-based decode group index.
      wait: True waits for the matching transfer; False starts it.
    """
    _decode_input_transfer(
        qkv_ref,
        b_ref,
        a_ref,
        decode_refs.qkv,
        decode_refs.b,
        decode_refs.a,
        decode_refs.input_sem,
        group,
        decode_count,
        group_width=group_width,
        wait=wait,
    )

  def start_group(group: jax.Array, *, include_single: bool) -> None:
    """Start loads for an in-range, nonempty group using free ring slots.

    include_single permits single-buffered members only after their previous
    stores have completed.

    Args:
      group: Zero-based decode group index.
      include_single: Whether previous stores are drained enough to load
        single-slot members.
    """
    input_transfer(group, wait=False)
    for member in range(group_width):

      @pl.when(group * group_width + member < decode_count)
      def start_member() -> None:
        """Start cache loads for an active member of the group."""
        conv_transfer(group, member, to_hbm=False, wait=False)
        # Single-slot members must drain their stores before the next load.
        if include_single or state_refs[member].shape[0] > 1:
          state_transfer(group, member, to_hbm=False, wait=False)

  def wait_stores(group: jax.Array) -> None:
    """Drain stores belonging to one completed group.

    Args:
      group: Zero-based decode group index.
    """
    for member in range(group_width):

      @pl.when(group * group_width + member < decode_count)
      def wait_member() -> None:
        """Wait for both cache stores of an active member."""
        state_transfer(group, member, to_hbm=True, wait=True)
        conv_transfer(group, member, to_hbm=True, wait=True)

  def step(group: jax.Array, unused: None) -> None:
    """Release buffers, prefetch inputs, compute the group, and launch stores.

    Args:
      group: Zero-based decode group index.
      unused: Unused loop carry, always None.
    """

    @pl.when(group > 0)
    def release_previous() -> None:
      """Release slots needed by the next prefetch.

      Drain the preceding convolution store before reusing its two-slot ring.
      For recurrent state, drain the group owning the next slot at its ring
      depth.
      """
      previous = group - 1
      for member in range(group_width):

        @pl.when(
            (previous * group_width + member < decode_count)
            & (~uniform_fresh if defer_fresh_cache_waits else jnp.bool_(True))
        )
        def release_conv() -> None:
          """Wait for the previous convolution store."""
          conv_transfer(previous, member, to_hbm=True, wait=True)

        def release_state_for_prefetch() -> None:
          """Choose the recurrent store owner that must finish before the next
          load.
          """
          buffers = state_refs[member].shape[0]
          if buffers == 1:

            @pl.when(previous * group_width + member < decode_count)
            def release_single_state() -> None:
              """Drain a single-slot member's previous state store."""
              state_transfer(previous, member, to_hbm=True, wait=True)

          else:
            state_group = group + 1 - buffers

            @pl.when(
                (state_group >= 0)
                & (state_group * group_width + member < decode_count)
            )
            def release_ring_state() -> None:
              """Drain the owner of the recurrent ring slot needed for prefetch."""
              state_transfer(state_group, member, to_hbm=True, wait=True)

        if defer_fresh_cache_waits:
          # Fresh owners issue no loads, so defer the store wait until compute
          # would overwrite its ring slot.
          @pl.when(~uniform_fresh)
          def release_resumed_state() -> None:
            """Drain resumed-state stores before prefetch reuses their slots."""
            release_state_for_prefetch()

        else:
          release_state_for_prefetch()

      for member in range(group_width):
        if state_refs[member].shape[0] != 1:
          continue

        @pl.when(group * group_width + member < decode_count)
        def load_single_buffer_member() -> None:
          """Load a single-buffered member after its previous store finishes."""
          state_transfer(group, member, to_hbm=False, wait=False)

    @pl.when(group + 1 < groups)
    def prefetch_next() -> None:
      """Start the next group's loads when a next group exists."""
      start_group(group + 1, include_single=False)

    input_transfer(group, wait=True)
    parity = group % 2

    for member in range(group_width):
      owner = group * group_width + member

      if defer_fresh_cache_waits:
        conv_group = group - 2

        @pl.when(
            uniform_fresh
            & (conv_group >= 0)
            & (conv_group * group_width + member < decode_count)
        )
        def release_fresh_conv_for_compute() -> None:
          """Drain the fresh convolution store before compute reuses its buffer."""
          conv_transfer(conv_group, member, to_hbm=True, wait=True)

        state_group = group - state_refs[member].shape[0]

        @pl.when(
            uniform_fresh
            & (state_group >= 0)
            & (state_group * group_width + member < decode_count)
        )
        def release_fresh_state_for_compute() -> None:
          # A partial final group must still drain the old active owner's store.
          """Drain the prior fresh state owner, including a partial-group tail."""
          state_transfer(state_group, member, to_hbm=True, wait=True)

      @pl.when(owner < decode_count)
      def compute_member() -> None:
        """Wait for inputs/state, compute one active token, and start its cache
        stores.
        """
        state_transfer(group, member, to_hbm=False, wait=True)
        conv_transfer(group, member, to_hbm=False, wait=True)

        buffer = group % state_refs[member].shape[0]
        _decode_inner(
            decode_refs.qkv.at[parity],
            decode_refs.b.at[parity],
            decode_refs.a.at[parity],
            decode_refs.conv.at[parity, member],
            state_refs[member].at[pl.ds(buffer, 1)],
            output_carry_ref,
            weights_ref,
            dense_conv_ref,
            grouped_state_scratch_ref,
            cfg=cfg,
            has_initial_state=seq_lens_ref[owner] > 1,
            output_offset=owner % fused_compute.OUTPUT_ALIGNMENT,
            input_offset=(group * group_width) % 8 + member,
            uniform_fresh=uniform_fresh,
        )

        state_transfer(group, member, to_hbm=True, wait=False)
        conv_transfer(group, member, to_hbm=True, wait=False)

        if can_flush_output:

          @pl.when((owner + 1) % fused_compute.OUTPUT_ALIGNMENT == 0)
          def flush_output() -> None:
            """Write a completed 16-row output carry block synchronously."""
            lo = pl.multiple_of(
                owner + 1 - fused_compute.OUTPUT_ALIGNMENT,
                fused_compute.OUTPUT_ALIGNMENT,
            )
            destination = out_ref.at[
                pl.ds(lo, fused_compute.OUTPUT_ALIGNMENT), :
            ]
            fused_memory.dma_store_and_wait(
                output_carry_ref,
                destination,
                output_sem_ref.at[0],
            )

    return unused

  @pl.when(groups > 0)
  def run() -> None:
    """Prime the first group, run the group loop, and drain outstanding stores."""
    start_group(jnp.int32(0), include_single=True)
    jax.lax.fori_loop(0, groups, step, None)
    wait_stores(groups - 1)

    if defer_fresh_cache_waits:
      previous = groups - 2
      for member in range(group_width):

        @pl.when(
            uniform_fresh
            & (previous >= 0)
            & (previous * group_width + member < decode_count)
        )
        def drain_fresh_penultimate_cache() -> None:
          """Drain delayed fresh-path stores from the penultimate group."""
          if state_refs[member].shape[0] == 2:
            state_transfer(previous, member, to_hbm=True, wait=True)
          conv_transfer(previous, member, to_hbm=True, wait=True)

    # Three-slot rings can retain the penultimate group's store; drain it too.
    last = groups - 1
    for member in range(group_width):
      if state_refs[member].shape[0] < 3:
        continue

      @pl.when(
          (last - 1 >= 0) & ((last - 1) * group_width + member < decode_count)
      )
      def drain_intervening_state() -> None:
        """Drain the penultimate recurrent store left outstanding in a
        three-slot ring.
        """
        state_transfer(last - 1, member, to_hbm=True, wait=True)
