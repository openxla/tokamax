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
"""Public host entry point and HBM tensor packing for Batched RPA."""

from collections.abc import Sequence

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import kernel
from tokamax._src.ops.batched_rpa.kernel import scheduler as scheduler_lib
from tokamax._src.ops.batched_rpa.kernel import tuning
from tokamax._src.ops.batched_rpa.kernel import utils


@jax.jit(
    static_argnames=(
        "scheduler",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "use_per_token_scale",
        "per_token_scale_dtype",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "skip_kv_update",
        "update_kv_cache",
        "kv_layout",
        "decode_query_size",
        "return_lse",
        "quantize_p",
    ),
    # Donation of transient inputs can fail for some runtime buffer layouts in
    # the experimental tuning path. Keep donation only for kv_cache, which is
    # the intended long-lived mutable state.
    donate_argnames=("kv_cache",),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None = None,
    *,
    scheduler: scheduler_lib.BaseScheduler | None = None,
    extra_scalars: Sequence[jax.Array] = (),
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    use_per_token_scale: bool = False,
    per_token_scale_dtype: jnp.dtype | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    skip_kv_update: bool = False,
    update_kv_cache: bool | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    return_lse: bool = False,
    quantize_p: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
  """Perform batched ragged paged attention.

  Args:
    queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
    keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
    values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
    kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
      kv_packing, head_dim]. Stores existing kv cache data where k & vs are
      concatenated along num kv heads dim.
    kv_lens: [max_num_seqs]. Total KV length of each sequence.
    page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
      sequence.
    cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
      length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
      b=cu_q_lens[i+1] represents q/k/v of sequence i.
    distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
      sequences (cumulative end indices for decode, prefill-only, and mixed).
      The prefill-only segment is handled by the mixed kernel call.
      distribution[2] represents total number of sequences.
    attention_sink: [num_q_heads]. Per-head logit that joins the softmax
      denominator without contributing to the output.
    scheduler: Static schedule strategy instance (`BaseScheduler` or subclass).
      Defaults to `BaseScheduler()` when None.
    extra_scalars: Dynamic scalar arrays (e.g. `(cp_rank,)`) forwarded to
      `scheduler` on host and prefetched into SMEM in the kernels.
    sm_scale: Softmax scale value.
    sliding_window: Size of sliding window (also known as local attention). kvs
      outside of the window is not fetched from hbm and masked out during
      computation.
    soft_cap: Cap values of softmax inputs.
    mask_value: Value to use for causal masking. Defaults to smallest
      representable value of the activation dtype.
    use_per_token_scale: Whether to use per-token quantization.
    per_token_scale_dtype: Dtype of per-token scale. Defaults to None.
    q_scale: Quantization scale value of queries.
    k_scale: Per-tensor quantization scale value of keys.
    v_scale: Per-tensor quantization scale value of values.
    dynamic_k_scale: Per-token quantization scale value of keys.
    dynamic_v_scale: Per-token quantization scale value of values.
    chunk_prefill_size: Not used.
    decode_block_sizes: Kernel block size to use during decode.
    prefill_block_sizes: Kernel block size to use during prefill.
    vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
      size of the hardware.
    debug_mode: Not used.
    out_dtype: Dtype of output. Defaults to dtype of queries.
    use_causal_mask: Not used.
    skip_kv_update: If True, don't write keys/values back to kv_cache
      (KV-share). All kv, including this step's tokens, is read from kv_cache.
    update_kv_cache: If set, overrides skip_kv_update (update_kv_cache=False is
      the same as skip_kv_update=True). None (default) uses skip_kv_update.
    kv_layout: Physical layout of `kv_cache` and prepared `new_kv_hbm`.
    decode_query_size: Number of query tokens in decode (1 by default, can be
      higher in case of speculative decoding).
    return_lse: If True, return log-sum-exp (lse) values along with the output.
      Defaults to False.
    quantize_p: If True, cast the softmax probabilities down to the kv cache
      dtype before the pv matmul. Required to get a native fp8 pv matmul when
      the kv cache is fp8, at the cost of pv precision. Defaults to False.

  Returns:
    out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
    new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
      kv_packing, head_dim]. Result of new kv cache where k & vs are
      concatenated along num kv heads dim.
    lse (only when return_lse=True): [max_num_tokens, num_q_heads].
      Log-sum-exp values (m + log(l)) for each query token and head,
      needed for merging partial attention results in CP.
  """
  if not use_causal_mask:
    raise ValueError("Only causal attention is supported.")
  if chunk_prefill_size is not None:
    raise ValueError("Specifying chunk prefill size is not supported.")
  if debug_mode:
    raise ValueError("Debug mode is not supported.")

  # update_kv_cache (as passed by tpu-inference) overrides skip_kv_update when
  # set; update_kv_cache=False (KV-share) is the same as skip_kv_update=True.
  if update_kv_cache is not None:
    skip_kv_update = not update_kv_cache

  if scheduler is None:
    scheduler = scheduler_lib.BaseScheduler()

  if out_dtype is None:
    out_dtype = queries.dtype
  if mask_value is None:
    mask_value = jnp.finfo(out_dtype).min
  if vmem_limit_bytes is None:
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

  max_num_seqs = kv_lens.shape[0]
  if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
    page_size = kv_cache.shape[4]
  else:
    page_size = kv_cache.shape[1]

  num_q_heads = queries.shape[1]
  head_dim = queries.shape[2]
  num_kv_heads = keys.shape[1]
  num_page_indices = page_indices.shape[0]

  model_cfgs = configs.ModelConfigs(
      num_q_heads=num_q_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      sliding_window=sliding_window,
      sm_scale=sm_scale,
      soft_cap=soft_cap,
      mask_value=mask_value,
  )

  if k_scale is not None and dynamic_k_scale is not None:
    raise ValueError(
        "Only one of k_scale or dynamic_k_scale can be set. Got"
        f" {k_scale=} and {dynamic_k_scale=}"
    )
  if v_scale is not None and dynamic_v_scale is not None:
    raise ValueError(
        "Only one of v_scale or dynamic_v_scale can be set. Got"
        f" {v_scale=} and {dynamic_v_scale=}"
    )

  serve_cfgs = configs.ServingConfigs(
      num_seqs=max_num_seqs,
      num_page_indices=num_page_indices,
      total_q_tokens=queries.shape[0],
      dtype_q=queries.dtype,
      dtype_kv=kv_cache.dtype,
      dtype_out=out_dtype,
      page_size=page_size,
      per_token_scale=use_per_token_scale,
      per_token_scale_dtype=per_token_scale_dtype,
      scale_q=q_scale,
      scale_k=k_scale,
      scale_v=v_scale,
      kv_layout=kv_layout,
      decode_query_size=decode_query_size,
      cp_group_size=scheduler.get_cp_group_size(),
      attention_scope=scheduler.scope,
      return_lse=return_lse,
      skip_kv_update=skip_kv_update,
      quantize_p=quantize_p,
  )
  layout = configs.LayoutConfig.from_configs(model_cfgs, serve_cfgs)

  q_hbm, new_kv_hbm, attention_sink_hbm = prepare_inputs(
      queries,
      keys,
      values,
      queries.dtype,
      kv_cache.dtype,
      dynamic_k_scale,
      dynamic_v_scale,
      kv_layout=kv_layout,
      page_size=page_size,
      attention_sink=attention_sink,
  )

  default_decode, default_prefill = tuning.calculate_block_sizes(
      model_cfgs,
      serve_cfgs,
      vmem_limit_bytes,
      decode_query_size=decode_query_size,
  )

  lse_hbm_init: jax.Array | None = None
  if return_lse:
    lse_hbm_init = jnp.full(
        layout.lse_hbm_shape(queries.shape[0]),
        -jnp.inf,
        dtype=layout.softmax_dtype,
    )

  q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
  kv_cache_lens, kv_new_lens, q_offsets = scheduler.get_kv_lens(
      kv_lens, q_lens, page_size, extra_scalars=extra_scalars
  )

  def run_rpa_kernel(
      mode: configs.RpaCase,
      o_hbm_alias_q_hbm: jax.Array,
      kv_cache: jax.Array,
      lse_hbm_in: jax.Array | None,
  ):
    if mode == configs.RpaCase.DECODE:
      effective_blocks = decode_block_sizes or default_decode
    else:
      effective_blocks = prefill_block_sizes or default_prefill

    cfgs = configs.RpaConfigs(
        block=effective_blocks,
        model=model_cfgs,
        serve=serve_cfgs,
        vmem_limit_bytes=vmem_limit_bytes,
        mode=mode,
    )
    cfgs.validate_inputs(
        q=queries,
        k=keys,
        v=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=attention_sink,
    )

    schedule_hbm = scheduler_lib.generate_rpa_metadata(
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
        distribution,
        page_indices,
        cfgs=cfgs,
        scheduler=scheduler,
        extra_scalars=extra_scalars,
    )
    result = kernel.rpa_kernel(
        schedule_hbm,
        o_hbm_alias_q_hbm,
        new_kv_hbm,
        kv_cache,
        lse_hbm_in,
        attention_sink_hbm,
        cfgs=cfgs,
    )
    if return_lse:
      o_out, kv_out, lse_out = result
    else:
      o_out, kv_out, _ = result
      lse_out = None
    return o_out, kv_out, lse_out

  o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
      configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm_init
  )
  o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
      configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache, lse_hbm
  )

  # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
  o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
  # after: [kv_heads, max_tokens, q_per_kv, d]

  o_hbm = o_hbm[:, :, : layout.num_q_heads_per_kv_head, :head_dim]
  o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

  if not return_lse:
    return o_hbm, kv_cache

  max_tokens = queries.shape[0]
  lse = lse_hbm.reshape(
      num_kv_heads,
      max_tokens,
      layout.aligned_num_q_heads_per_kv_head,
      layout.num_lanes,
  )[:, :max_tokens, : layout.num_q_heads_per_kv_head, 0]
  lse = lse.swapaxes(0, 1).reshape(max_tokens, num_q_heads)

  return o_hbm, kv_cache, lse


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    kv_dtype: jax.typing.DTypeLike,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    chip_version: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X,
    use_per_token_scale: bool = False,
    scale_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[int, ...]:
  """Return the HBM KV-cache shape for the given model and chip."""
  chip_info = pltpu.get_tpu_info_for_chip(chip_version, 1)
  # For packed fp4 (`uint8`), callers pass the already-halved `k.shape[2]`
  # dimension; `LayoutConfig.head_dim` expects the un-halved query head dim.
  kv_dtype_resolved = jnp.dtype(kv_dtype)
  logical_head_dim = (
      actual_head_dim * 2 if kv_dtype_resolved == jnp.uint8 else actual_head_dim
  )
  layout = configs.LayoutConfig(
      num_q_heads=actual_num_kv_heads,
      num_kv_heads=actual_num_kv_heads,
      head_dim=logical_head_dim,
      page_size=page_size,
      dtype_q=jnp.dtype(jnp.bfloat16),
      dtype_kv=kv_dtype_resolved,
      dtype_out=jnp.dtype(jnp.bfloat16),
      kv_layout=kv_layout,
      per_token_scale=use_per_token_scale,
      per_token_scale_dtype=(
          None if scale_dtype is None else jnp.dtype(scale_dtype)
      ),
      _num_lanes=chip_info.num_lanes,
      _num_sublanes=chip_info.num_sublanes,
  )
  return layout.kv_cache_shape(total_num_pages)


def fold_kv_cache_head_dim(
    kv_cache: jax.Array,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    chip_version: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X,
) -> jax.Array:
  """Converts an unfolded `HEAD_ALONG_SUBLANE` cache to the folded layout."""
  num_lanes = pltpu.get_tpu_info_for_chip(chip_version, 1).num_lanes
  kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE
  fold = configs.num_lane_folds_in_head_dim(
      actual_head_dim, kv_cache.dtype, kv_layout, num_lanes
  )
  if fold == 1:
    return kv_cache
  num_pages, page_size = kv_cache.shape[:2]
  num_kv_heads_x2 = actual_num_kv_heads * 2
  folded_head_dim = actual_head_dim // fold
  folded_shape = get_kv_cache_shape(
      num_pages,
      page_size,
      actual_num_kv_heads,
      actual_head_dim,
      kv_cache.dtype,
      kv_layout=kv_layout,
      chip_version=chip_version,
  )
  kv = kv_cache.reshape(num_pages, page_size, -1, kv_cache.shape[-1])
  kv = kv[:, :, :num_kv_heads_x2, :actual_head_dim]
  kv = (
      kv.reshape(num_pages, page_size, num_kv_heads_x2, fold, folded_head_dim)
      .transpose(0, 1, 3, 2, 4)
      .reshape(num_pages, page_size, num_kv_heads_x2 * fold, folded_head_dim)
  )
  kv = jnp.pad(
      kv,
      (
          (0, 0),
          (0, 0),
          (0, folded_shape[2] * folded_shape[3] - num_kv_heads_x2 * fold),
          (0, folded_shape[4] - folded_head_dim),
      ),
  )
  return kv.reshape(folded_shape)


def unfold_kv_cache_head_dim(
    kv_cache: jax.Array,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    chip_version: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X,
) -> jax.Array:
  """Inverse of `fold_kv_cache_head_dim`."""
  num_lanes = pltpu.get_tpu_info_for_chip(chip_version, 1).num_lanes
  kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE
  fold = configs.num_lane_folds_in_head_dim(
      actual_head_dim, kv_cache.dtype, kv_layout, num_lanes
  )
  if fold == 1:
    return kv_cache
  num_pages, page_size, _, kv_packing, _ = kv_cache.shape
  num_kv_heads_x2 = actual_num_kv_heads * 2
  folded_head_dim = actual_head_dim // fold
  kv = kv_cache.reshape(num_pages, page_size, -1, kv_cache.shape[-1])
  kv = kv[:, :, : num_kv_heads_x2 * fold, :folded_head_dim]
  kv = (
      kv.reshape(num_pages, page_size, fold, num_kv_heads_x2, folded_head_dim)
      .transpose(0, 1, 3, 2, 4)
      .reshape(num_pages, page_size, num_kv_heads_x2, actual_head_dim)
  )
  num_kv_heads_x2_aligned = utils.align_to(num_kv_heads_x2, kv_packing)
  aligned_head_dim = utils.align_to(actual_head_dim, num_lanes)
  kv = jnp.pad(
      kv,
      (
          (0, 0),
          (0, 0),
          (0, num_kv_heads_x2_aligned - num_kv_heads_x2),
          (0, aligned_head_dim - actual_head_dim),
      ),
  )
  return kv.reshape(
      num_pages,
      page_size,
      num_kv_heads_x2_aligned // kv_packing,
      kv_packing,
      aligned_head_dim,
  )


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    k_scale: jax.Array | None = None,
    v_scale: jax.Array | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int = 128,
    attention_sink: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Pad and pack Q, K, V (+ optional scales/sink) into kernel HBM layouts."""
  total_q_tokens, actual_num_q_heads, actual_q_head_dim = q.shape
  _, actual_num_kv_heads, actual_kv_head_dim = k.shape
  has_per_token_scale = k_scale is not None and v_scale is not None

  layout = configs.LayoutConfig(
      num_q_heads=actual_num_q_heads,
      num_kv_heads=actual_num_kv_heads,
      head_dim=actual_q_head_dim,
      page_size=page_size,
      dtype_q=q_dtype,
      dtype_kv=kv_dtype,
      dtype_out=q_dtype,
      kv_layout=kv_layout,
      per_token_scale=has_per_token_scale,
      per_token_scale_dtype=k_scale.dtype if has_per_token_scale else None,
  )

  num_q_heads_per_kv_head = layout.num_q_heads_per_kv_head
  q_packing = layout.packing_q
  kv_packing = layout.packing_kv
  aligned_num_q_heads_per_kv_head = layout.aligned_num_q_heads_per_kv_head
  aligned_q_head_dim = layout.aligned_q_head_dim
  aligned_kv_head_dim = layout.storage_kv_head_dim
  num_lanes = layout.num_lanes

  # queries: (T, H, D) -> (H_kv, T, G // q_packing, q_packing, D)
  o_hbm_alias_q_hbm = (
      jnp.pad(
          q.reshape(
              total_q_tokens,
              actual_num_kv_heads,
              num_q_heads_per_kv_head,
              actual_q_head_dim,
          ),
          (
              (0, 0),
              (0, 0),
              (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
              (0, aligned_q_head_dim - actual_q_head_dim),
          ),
          constant_values=0,
      )
      .reshape(
          total_q_tokens,
          actual_num_kv_heads,
          aligned_num_q_heads_per_kv_head // q_packing,
          q_packing,
          aligned_q_head_dim,
      )
      .swapaxes(0, 1)
  )

  actual_num_kv_heads_x2 = actual_num_kv_heads * 2
  num_kv_heads_x2_aligned = layout.aligned_num_kv_heads_x2

  if has_per_token_scale:
    k_scale = k_scale.reshape(total_q_tokens, actual_num_kv_heads)
    v_scale = v_scale.reshape(total_q_tokens, actual_num_kv_heads)
    num_scale_channels = layout.scale_channels

    k_scale_split = jax.lax.bitcast_convert_type(k_scale, k.dtype).reshape(
        total_q_tokens, actual_num_kv_heads, num_scale_channels
    )
    v_scale_split = jax.lax.bitcast_convert_type(v_scale, v.dtype).reshape(
        total_q_tokens, actual_num_kv_heads, num_scale_channels
    )

    k = jnp.concatenate([k, k_scale_split], axis=-1)
    v = jnp.concatenate([v, v_scale_split], axis=-1)
    actual_kv_head_dim += num_scale_channels

  if layout.is_seq_along_lane:
    align_tokens = max(num_lanes, page_size)
    padded_total_tokens = utils.align_to(total_q_tokens, align_tokens)
    new_kv_hbm = (
        jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(
                total_q_tokens, actual_num_kv_heads_x2, actual_kv_head_dim
            ),
            (
                (0, padded_total_tokens - total_q_tokens),
                (0, 0),
                (0, aligned_kv_head_dim - actual_kv_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            padded_total_tokens,
            actual_num_kv_heads_x2,
            aligned_kv_head_dim // kv_packing,
            kv_packing,
        )
        .transpose(1, 2, 3, 0)
    )
  else:
    kv = jnp.concatenate([k, v], axis=-1).reshape(
        total_q_tokens, actual_num_kv_heads_x2, actual_kv_head_dim
    )
    num_folds = layout.num_lane_folds_in_head_dim
    if num_folds > 1:
      actual_kv_head_dim //= num_folds
      kv = (
          kv.reshape(
              total_q_tokens,
              actual_num_kv_heads_x2,
              num_folds,
              actual_kv_head_dim,
          )
          .transpose(0, 2, 1, 3)
          .reshape(total_q_tokens, -1, actual_kv_head_dim)
      )
      actual_num_kv_heads_x2 *= num_folds
    new_kv_hbm = jnp.pad(
        kv,
        (
            (0, 0),
            (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
            (0, aligned_kv_head_dim - actual_kv_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        num_kv_heads_x2_aligned // kv_packing,
        kv_packing,
        aligned_kv_head_dim,
    )

  if attention_sink is not None:
    attention_sink = jnp.pad(
        attention_sink.reshape(actual_num_kv_heads, num_q_heads_per_kv_head),
        (
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
        ),
        constant_values=-jnp.inf,
    )
    attention_sink = jnp.broadcast_to(
        attention_sink[..., None], (*attention_sink.shape, num_lanes)
    )

  return o_hbm_alias_q_hbm, new_kv_hbm, attention_sink


def prepare_outputs(out: jax.Array) -> jax.Array:
  kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
  return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)
