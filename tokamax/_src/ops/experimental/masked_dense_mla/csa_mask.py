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
"""SparseCore expansion of `topk_indices` into a dense per-token bitmap.

The masked-dense MLA kernel streams the whole KV cache and masks it, instead
of gathering the selected rows. It therefore needs the indexer's selection as
a dense bitmap rather than a list of indices: row `t` of the output has a 1 in
column `j` iff query token `t` selected KV position `j`.

Scattering `topk` ones per row is the irregular, indexed write that SparseCore
exists for, so the expansion runs there and leaves the TensorCore free.
"""

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


def mask_kernel(
    topk_indices_hbm_ref: Any,  # i32[padded_num_tokens, topk]
    mask_out_hbm_ref: Any,  # [padded_num_tokens, max_kv_len]
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    topk: int,
    max_kv_len: int,
):
  """Writes one row of the bitmap per grid step, striped across subcores."""
  core_index = jax.lax.axis_index((core_axis_name, subcore_axis_name))
  num_cores = jax.lax.psum(1, (core_axis_name, subcore_axis_name))

  padded_chunk_size = topk_indices_hbm_ref.shape[0]
  rows_per_core = padded_chunk_size // num_cores

  def _body(topk_indices_vmem_ref, mask_out_vmem_ref):
    mask_out_vmem_ref[pl.ds(0, 1), pl.ds(0, max_kv_len)] = jnp.zeros(
        (1, max_kv_len), dtype=jnp.int32
    )

    if topk % 8 == 0:
      # Read 8 indices per SIMD load; `unroll=8` keeps the scatter
      # pipeline full.
      def vec_k_loop(k_chunk, _):
        vec_idx = topk_indices_vmem_ref[0, pl.ds(k_chunk * 8, 8)]
        for lane in range(8):
          idx = vec_idx[lane]

          def write_fn(i=idx):
            mask_out_vmem_ref[0, pl.ds(i, 1)] = jnp.ones((1,), dtype=jnp.int32)

          jax.lax.cond((idx >= 0) & (idx < max_kv_len), write_fn, lambda: None)

      jax.lax.fori_loop(0, topk // 8, vec_k_loop, None, unroll=8)
    else:

      def k_loop(k, _):
        idx = topk_indices_vmem_ref[0, pl.ds(k, 1)][0]

        def write_fn(i=idx):
          mask_out_vmem_ref[0, pl.ds(i, 1)] = jnp.ones((1,), dtype=jnp.int32)

        jax.lax.cond((idx >= 0) & (idx < max_kv_len), write_fn, lambda: None)

      jax.lax.fori_loop(0, topk, k_loop, None, unroll=16)

  pltpu.emit_pipeline(
      _body,
      grid=(rows_per_core,),
      in_specs=pl.BlockSpec(
          (1, topk),
          lambda r: (r * num_cores + core_index, 0),
      ),
      out_specs=pl.BlockSpec(
          (1, max_kv_len),
          lambda r: (r * num_cores + core_index, 0),
      ),
  )(topk_indices_hbm_ref, mask_out_hbm_ref)


@functools.partial(jax.jit, static_argnames=("max_kv_len",))
def generate_mask_sc(
    topk_indices: jax.Array,  # i32[num_tokens, topk]
    max_kv_len: int,
) -> jax.Array:
  """Expands `topk_indices` into a dense `[num_tokens, max_kv_len]` bitmap.

  Args:
    topk_indices: per-token KV positions selected by the indexer, `-1` padded.
      Entries outside `[0, max_kv_len)` are dropped.
    max_kv_len: bitmap width. Must cover every position the kernel can read,
      i.e. `num_kv_blocks * bkv_sz`.

  Returns:
    `int32[num_tokens, max_kv_len]`, 1 where selected and 0 elsewhere.
  """
  chunk_size, topk = topk_indices.shape
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "generate_mask_sc requires a TPU with SparseCore"
  num_cores = sc_info.num_cores * sc_info.num_subcores

  # The grid stripes rows across subcores, so the row count must divide
  # evenly; `-1` rows produce an all-zero mask row and are then dropped.
  pad_size = (num_cores - (chunk_size % num_cores)) % num_cores
  padded_topk_indices = jnp.pad(
      topk_indices, ((0, pad_size), (0, 0)), constant_values=-1
  )
  padded_chunk_size = chunk_size + pad_size

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  out = pl.kernel(
      functools.partial(
          mask_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          topk=topk,
          max_kv_len=max_kv_len,
      ),
      out_type=jax.ShapeDtypeStruct((padded_chunk_size, max_kv_len), jnp.int32),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      mesh=vector_mesh,
      name="sc_mask_generate",
  )(padded_topk_indices)

  return out[:chunk_size, :]
