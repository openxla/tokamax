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
"""Staging references and asynchronous state transfers for fused GDN."""

import dataclasses
from typing import Any

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import metadata as fused_metadata


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OutputRefs:
  """Output staging: two VMEM tiles, a partial-block carry, and an HBM tail.

  scratch is [2, chunk_size + 16, output_width]; carry and optional tail are
  [16, output_width], all in the activation dtype. sem holds two DMA
  semaphores, one per scratch slot.
  """

  scratch: Any
  carry: Any
  sem: Any
  tail: Any


def dma(src_ref: Any, dst_ref: Any, sem_ref: Any, *, wait: bool) -> None:
  """Start or wait for DMA between matching-shape, matching-dtype refs.

  wait=True must match a start using the same scalar semaphore and
  descriptor. Keep both refs live and free of conflicting access until
  completion.

  Args:
    src_ref: Ready source ref; keep it unchanged until the transfer completes.
    dst_ref: Destination ref; its shape and dtype must match the source.
    sem_ref: Scalar DMA semaphore paired with this transfer.
    wait: True waits for the matching transfer; False starts it.
  """
  descriptor = pltpu.make_async_copy(src_ref, dst_ref, sem_ref)
  if wait:
    descriptor.wait()
  else:
    descriptor.start()


def bidirectional_dma(
    hbm_ref: Any,
    vmem_ref: Any,
    load_sem_ref: Any,
    store_sem_ref: Any,
    *,
    to_hbm: bool,
    wait: bool,
) -> None:
  """Load/store BF16/FP32 cache state between matching HBM and VMEM refs.

  to_hbm selects store_sem_ref; loads use load_sem_ref. Match each wait to
  its start and observe dma's ref lifetime requirements.

  Args:
    hbm_ref: Cache-state ref in HBM, used as source or destination by direction.
    vmem_ref: Cache-state staging ref matching the HBM ref's shape and dtype.
    load_sem_ref: DMA semaphore ref used for cache loads.
    store_sem_ref: DMA semaphore ref used for cache stores.
    to_hbm: True stores VMEM state to HBM; False loads HBM state into VMEM.
    wait: True waits for the matching transfer; False starts it.
  """
  if to_hbm:
    dma(vmem_ref, hbm_ref, store_sem_ref, wait=wait)
  else:
    dma(hbm_ref, vmem_ref, load_sem_ref, wait=wait)


def dma_store_and_wait(src_ref: Any, dst_ref: Any, sem_ref: Any) -> None:
  """Store and drain a ready [16, out_width] output block.

  VMEM source and exclusively owned HBM destination must have no conflicting
  transfers; sem_ref is a scalar DMA semaphore.

  Args:
    src_ref: Ready source ref; keep it unchanged until the transfer completes.
    dst_ref: Destination ref; its shape and dtype must match the source.
    sem_ref: Scalar DMA semaphore paired with this transfer.
  """
  dma(src_ref, dst_ref, sem_ref, wait=False)
  dma(src_ref, dst_ref, sem_ref, wait=True)


def conv_state_transfer(
    metadata_ref: memory_ref.MetadataRef,
    read_state_indices_ref: Any,
    read_offsets_ref: Any,
    conv_state_ref: Any,
    dma_scratch_ref: Any,
    dma_sem_ref: Any,
    owner: jax.Array,
    *,
    to_hbm: bool,
    wait: bool,
) -> None:
  """Transfer history through the owner's dedicated load/store parity slot.

  Owner cache metadata must be installed. Scratch is [2, max(1, kernel_size -
  1), width] in the cache dtype; semaphores are [2]. Match waits to starts;
  omit history transfers for a one-tap convolution.

  Args:
    metadata_ref: Packed tile/request metadata refs holding ownership, flags,
      and cache slots.
    read_state_indices_ref: Per-request initial-state cache slots, separate from
      write ownership.
    read_offsets_ref: Per-request int32 read-slot offsets; public dispatch zeros
      prefill offsets.
    conv_state_ref: HBM convolution cache; reads use prefix slots and writes use
      owned slots.
    dma_scratch_ref: Two convolution-history transfer slots in the cache dtype.
    dma_sem_ref: DMA semaphores paired with the convolution-history transfer
      slots.
    owner: Request index owning the cache transfer and its parity slot.
    to_hbm: True stores VMEM state to HBM; False loads HBM state into VMEM.
    wait: True waits for the matching transfer; False starts it.
  """
  if conv_state_ref.shape[1] == 0:
    return
  state_slot = fused_metadata.metadata_storage(
      metadata_ref.s_idx_to_state_indices
  )[owner]
  # Validation checks ownership; this clamp only bounds the DMA address.
  read_slot = jnp.where(
      state_slot > 0,
      read_state_indices_ref[owner] + read_offsets_ref[owner],
      jnp.int32(0),
  )
  read_slot = jnp.clip(read_slot, 0, conv_state_ref.shape[0] - 1)
  parity = owner % 2
  hbm_slot_ref = conv_state_ref.at[state_slot if to_hbm else read_slot]
  dma_slot_ref = dma_scratch_ref.at[parity]
  bidirectional_dma(
      hbm_slot_ref,
      dma_slot_ref,
      dma_sem_ref.at[parity],
      dma_sem_ref.at[parity],
      to_hbm=to_hbm,
      wait=wait,
  )


def recurrent_state_transfer(
    metadata_ref: memory_ref.MetadataRef,
    read_state_indices_ref: Any,
    read_offsets_ref: Any,
    recurrent_state_ref: Any,
    recurrent_scratch_ref: Any,
    load_sem_ref: Any,
    store_sem_ref: Any,
    owner: jax.Array,
    *,
    to_hbm: bool,
    wait: bool,
) -> None:
  """Transfer state through its owner's parity buffer, matching waits to starts.

  HBM [slots, heads, 128, 128] and scratch [2, heads, 128, 128] share the
  cache dtype; both semaphore arrays are [2]. Load only continuing requests,
  store every request, and drain transfers before buffer reuse.

  Args:
    metadata_ref: Packed tile/request metadata refs holding ownership, flags,
      and cache slots.
    read_state_indices_ref: Per-request initial-state cache slots, separate from
      write ownership.
    read_offsets_ref: Per-request int32 read-slot offsets; public dispatch zeros
      prefill offsets.
    recurrent_state_ref: HBM recurrent cache [slots, n_v, d_k, d_v].
    recurrent_scratch_ref: Two prefill recurrent-state parity slots in the cache
      dtype.
    load_sem_ref: DMA semaphore ref used for cache loads.
    store_sem_ref: DMA semaphore ref used for cache stores.
    owner: Request index owning the cache transfer and its parity slot.
    to_hbm: True stores VMEM state to HBM; False loads HBM state into VMEM.
    wait: True waits for the matching transfer; False starts it.
  """
  slot = fused_metadata.metadata_storage(metadata_ref.s_idx_to_state_indices)[
      owner
  ]
  read_slot = jnp.where(
      slot > 0,
      read_state_indices_ref[owner] + read_offsets_ref[owner],
      jnp.int32(0),
  )
  read_slot = jnp.clip(read_slot, 0, recurrent_state_ref.shape[0] - 1)
  parity = owner % 2
  bidirectional_dma(
      recurrent_state_ref.at[slot if to_hbm else read_slot],
      recurrent_scratch_ref.at[parity],
      load_sem_ref.at[parity],
      store_sem_ref.at[parity],
      to_hbm=to_hbm,
      wait=wait,
  )
