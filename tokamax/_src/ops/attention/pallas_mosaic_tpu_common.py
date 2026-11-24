# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Common utilities for Pallas Mosaic TPU attention."""

from collections.abc import Callable
import functools
import math
from typing import Any, Final

import jax
import jax.numpy as jnp
from tokamax._src.ops.attention import base
from tokamax._src.ops.splash_attention import pallas_mosaic_tpu_vjp as sp_vjp
from tokamax._src.ops.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib

NUM_LANES: Final[int] = 128


def largest_divisor_tile(seq_len: int, max_tile: int) -> int:
  """Returns the largest power-of-two tile <= `max_tile` dividing `seq_len`."""
  for tile in (1024, 512, 256):
    if tile <= max_tile and seq_len % tile == 0:
      return tile
  return NUM_LANES


def _padded_row_bytes(x: Any) -> int:
  padded_dim = math.ceil(x.shape[-1] / NUM_LANES) * NUM_LANES
  return padded_dim * jnp.dtype(x.dtype).itemsize


def max_vmem_tile(q: Any, v: Any, mask: base.Mask) -> int:
  """Returns the max tile size fitting inputs within 16 MiB TPU VMEM."""
  row_bytes = max(_padded_row_bytes(q), _padded_row_bytes(v))
  return 1024 if mask.bool_mask is None and row_bytes <= 256 else 512


def check_inputs_support(
    *x,
    bias,
    logits_dtype,
    dropout_mask,
    dropout_rate,
    paging_info,
    mask,
    q_indices=None,
    k_indices=None,
):
  """Checks that Mosaic TPU supports the given inputs."""
  supported_dtypes = (jnp.bfloat16, jnp.float32)
  if any(arg.dtype not in supported_dtypes for arg in x):
    raise NotImplementedError("Only bfloat16 and float32 inputs are supported.")

  if bias is not None:
    raise NotImplementedError("Bias is not supported.")
  if logits_dtype != jnp.float32:
    raise NotImplementedError("`logits_dtype` must be float32.")
  if dropout_mask is not None:
    raise NotImplementedError("dropout is not supported.")
  if paging_info is not None:
    raise NotImplementedError("Paged attention not supported.")
  if dropout_rate != 0.0:
    raise NotImplementedError("Dropout is not supported.")
  if k_indices is not None:
    raise NotImplementedError("`k_indices` is not supported.")
  if q_indices is not None and not mask.has_auto_q_indices:
    raise NotImplementedError("`q_indices` is not supported.")

  if mask.q_start is not None:
    raise NotImplementedError("mask.q_start is not supported.")
  if mask.q_end is not None:
    raise NotImplementedError("mask.q_end is not supported.")
  if mask.k_start is not None:
    raise NotImplementedError("mask.k_start is not supported.")
  if mask.k_end is not None:
    raise NotImplementedError("mask.k_end is not supported.")
  if (causal := mask.causal_mask) is not None:
    if not isinstance(causal.offset, int) or causal.offset < 0:
      raise NotImplementedError(
          "Only non-negative static causal offset is supported."
      )
    if causal.window_size is not None:
      if not isinstance(causal.window_size, int) or causal.window_size <= 0:
        raise NotImplementedError(
            "Only positive static window_size is supported."
        )
      q_seq_len = x[0].shape[-3]
      kv_seq_len = x[1].shape[-3]
      shard_count, _ = mask.get_q_seq_shard_info()
      if q_seq_len > 0 and (
          kv_seq_len == 0
          or q_seq_len * shard_count + causal.offset - causal.window_size
          >= kv_seq_len
      ):
        raise NotImplementedError(
            "Local sliding window masks with all-masked query rows are not"
            " supported on Mosaic TPU."
        )


def build_splash_kernel(
    mask: base.Mask,
    splash_config: splash.SplashConfig,
    q_seq_len: int,
    kv_seq_len: int,
    is_mqa: bool,
    save_residuals: bool,
    *,
    fold_q_seq_heads: bool = False,
) -> tuple[Callable[..., Any], Any]:
  """Returns the SplashAttention kernel maker and splash mask."""

  shard_count, q_shard_idx = mask.get_q_seq_shard_info()
  mask_shape = (q_seq_len * shard_count, kv_seq_len)
  if mask.bool_mask is not None:
    if fold_q_seq_heads:
      raise NotImplementedError(
          "`fold_q_seq_heads` does not support boolean masks."
      )
    q_idx = (
        q_shard_idx * q_seq_len + jnp.arange(q_seq_len)
        if shard_count > 1
        else q_seq_len
    )
    assert (mask_ := mask.as_array(q_idx, kv_seq_len)) is not None
    splash_mask = jax.lax.collapse(jax.lax.broadcast_to_rank(mask_, 4), 0, -3)
    mask_batch_size, num_mask_heads, _, _ = splash_mask.shape
    # TODO: Support boolean masks differing across heads.
    if num_mask_heads != 1:
      raise NotImplementedError(
          "Only num_mask_heads=1 is supported with a boolean mask."
      )
    if mask_batch_size == 1:
      splash_mask = jnp.squeeze(splash_mask, axis=(0, 1))  # (seq_q, seq_kv)
    else:
      splash_mask = jnp.squeeze(splash_mask, axis=1)  # (batch, seq_q, seq_kv)
  elif (causal := mask.causal_mask) is not None:
    assert isinstance(causal.offset, int)
    if causal.window_size is None:
      splash_mask = mask_lib.CausalMask(
          shape=mask_shape, offset=causal.offset, shard_count=shard_count
      )
    else:
      assert isinstance(causal.window_size, int)
      splash_mask = mask_lib.LocalMask(
          shape=mask_shape,
          window_size=(causal.window_size - 1, 0),
          offset=causal.offset,
          shard_count=shard_count,
      )
  else:
    splash_mask = mask_lib.FullMask(mask_shape)

  if fold_q_seq_heads:
    assert isinstance(splash_mask, mask_lib.Mask)
    splash_mask = mask_lib.make_stacked_gqa_mask(
        splash_mask, shard_count=shard_count
    )

  is_dynamic_mask = isinstance(splash_mask, jax.Array)

  if is_dynamic_mask:
    splash_maker = (
        splash.make_dynamic_splash_mqa
        if is_mqa
        else splash.make_dynamic_splash_mha
    )
    splash_maker = functools.partial(
        splash_maker,
        config=splash_config,
        save_residuals=save_residuals,
        mask_value=float(jnp.finfo(jnp.float32).min),
        downcast_smem_data=False,
    )
  else:
    make_splash = splash.make_splash_mqa if is_mqa else splash.make_splash_mha
    attn_fn = make_splash(
        splash_mask,
        config=splash_config,
        save_residuals=save_residuals,
        mask_value=float(jnp.finfo(jnp.float32).min),
        q_seq_shards=shard_count,
        downcast_smem_data=False,
    )
    if shard_count > 1:
      fwd_mask_info = sp_vjp.slice_mask_info(
          attn_fn.fwd_mask_info, q_shard_idx, shard_count
      )
      dkv_mask_info = None
      if attn_fn.dkv_mask_info is not None:
        dkv_mask_info = sp_vjp.slice_mask_info(
            attn_fn.dkv_mask_info, q_shard_idx, shard_count
        )
      attn_fn = splash.SplashAttentionKernel(
          fwd_mask_info,
          dkv_mask_info,
          **attn_fn.kwargs,
      )
    splash_maker = lambda mask: attn_fn

  return splash_maker, splash_mask
