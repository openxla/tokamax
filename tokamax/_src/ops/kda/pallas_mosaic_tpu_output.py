# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Sequence-wise Pallas compaction of aligned KDA outputs."""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.kda import common
from tokamax._src.ops.kda import utils


def _compact_output_kernel(
    original_ref, aligned_ref, source_ref, output_ref, packed_ref, *, chunk_size
):
  """One program owns a head group and batch, including all sequence tails."""
  batch = pl.program_id(1)
  tokens = output_ref.shape[2]
  sequences = original_ref.shape[1] - 1
  packed_ref[...] = jnp.zeros(packed_ref.shape, packed_ref.dtype)

  @pl.loop(0, sequences)
  def copy_sequence(seq):
    start = original_ref[batch, seq]
    length = original_ref[batch, seq + 1] - start
    aligned_start = aligned_ref[batch, seq]

    @pl.loop(0, (length + chunk_size - 1) // chunk_size)
    def copy_chunk(chunk):
      values = source_ref[
          :,
          0,
          pl.ds(
              pl.multiple_of(aligned_start + chunk * chunk_size, chunk_size),
              chunk_size,
          ),
          :,
      ]
      token = jax.lax.broadcasted_iota(
          jnp.int32, values.shape, dimension=1
      )
      values = jnp.where(
          chunk * chunk_size + token < length, values, 0
      )
      # Token-major scratch makes arbitrary packed starts a major-axis slice.
      # Later sequences overwrite the preceding sequence's copied tail.
      packed_ref[pl.ds(start + chunk * chunk_size, chunk_size), :, :] = (
          values.transpose(1, 0, 2)
      )

  values = packed_ref[:tokens].transpose(1, 0, 2)
  output_ref[:, 0] = values


@functools.partial(jax.jit, static_argnames=("tokens", "chunk_size"))
def compact_output(output, original_cu, aligned_cu, tokens, *, chunk_size=64):
  """Compact [H,B,T_aligned,V] while preserving dtype and zero padding.

  The aligned input remains an HBM buffer. Each program compacts complete
  sequences in VMEM, then writes disjoint head/batch output regions. Large
  allocations or unsupported widths retain the XLA gather path.
  """
  heads, batch, aligned_tokens, width = output.shape
  if original_cu.ndim == 1:
    original_cu = jnp.broadcast_to(original_cu[None], (batch, original_cu.size))
  if aligned_cu.ndim == 1:
    aligned_cu = jnp.broadcast_to(aligned_cu[None], (batch, aligned_cu.size))
  limits = common.get_tpu_limits()
  mini_batch = limits.block_align_minor
  padded_tokens = (tokens + mini_batch - 1) // mini_batch * mini_batch
  # Include source, token-major scratch and output buffers with I/O headroom.
  per_head = (
      (2 * aligned_tokens + 3 * padded_tokens + chunk_size)
      * width
      * output.dtype.itemsize
  )
  if (
      chunk_size != 64
      or width % 128
      or heads % mini_batch
      or per_head * mini_batch > limits.vmem_limit_bytes
  ):
    result = utils._unalign_output(output, original_cu, aligned_cu, tokens)
    valid = jnp.arange(tokens)[None, :] < original_cu[:, -1:]
    return jnp.where(valid[None, :, :, None], result, 0)
  result = pl.pallas_call(
      functools.partial(_compact_output_kernel, chunk_size=chunk_size),
      out_shape=jax.ShapeDtypeStruct(
          (heads, batch, padded_tokens, width), output.dtype
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          grid=(heads // mini_batch, batch),
          in_specs=[
              pl.BlockSpec(
                  (mini_batch, 1, aligned_tokens, width),
                  lambda h, b, *_: (h, b, 0, 0),
              )
          ],
          out_specs=pl.BlockSpec(
              (mini_batch, 1, padded_tokens, width),
              lambda h, b, *_: (h, b, 0, 0),
          ),
          scratch_shapes=[
              pltpu.VMEM(
                  (padded_tokens + chunk_size, mini_batch, width), output.dtype
              )
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel")
      ),
      interpret=utils.get_interpret(),
  )(original_cu.astype(jnp.int32), aligned_cu.astype(jnp.int32), output)
  return result[:, :, :tokens]
