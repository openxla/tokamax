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

import dataclasses
import functools
from typing import Any, Final

import jax
import jax.numpy as jnp
from tokamax._src.ops.attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

NUM_LANES: Final[int] = 128


def check_inputs_support(
    *x,
    bias,
    logits_dtype,
    dropout_mask,
    dropout_rate,
    paging_info,
    mask,
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

  if mask.q_start is not None:
    raise NotImplementedError("mask.q_start is not supported.")
  if mask.q_end is not None:
    raise NotImplementedError("mask.q_end is not supported.")
  if mask.k_start is not None:
    raise NotImplementedError("mask.k_start is not supported.")
  if mask.k_end is not None:
    raise NotImplementedError("mask.k_end is not supported.")


def build_splash_kernel(
    mask: base.Mask,
    splash_config: splash.SplashConfig,
    q_seq_len: int,
    kv_seq_len: int,
    is_mqa: bool,
    save_residuals: bool,
) -> tuple[splash.SplashAttentionKernel, splash.SplashConfig]:
  """Returns the SplashAttention kernel and config."""

  # TODO: support multiple shards.
  shard_count = 1
  mask_shape = (q_seq_len, kv_seq_len)
  if mask.bool_mask is not None:
    splash_mask = jax.lax.collapse(
        jax.lax.broadcast_to_rank(mask.as_array(q_seq_len, kv_seq_len), 4),
        0,
        -3,
    )
    mask_batch_size, num_mask_heads, _, _ = splash_mask.shape
    # TODO: Support boolean masks differing across heads.
    if num_mask_heads != 1:
      raise NotImplementedError(
          "Only num_mask_heads=1 is supported with a boolean mask."
      )
    # TODO: Support batched boolean masks.
    if mask_batch_size != 1:
      raise NotImplementedError("Only unbatched boolean masks are supported.")
    splash_mask = jnp.squeeze(splash_mask, axis=(0, 1))  # (seq_q, seq_kv)
  elif mask.is_causal:
    splash_mask = mask_lib.CausalMask(shape=mask_shape, shard_count=shard_count)
  else:
    splash_mask = mask_lib.FullMask(mask_shape)
  is_dynamic_mask = isinstance(splash_mask, jax.Array)

  if is_dynamic_mask:
    splash_maker = (
        splash.make_dynamic_splash_mqa
        if is_mqa
        else splash.make_dynamic_splash_mha
    )
  else:
    splash_maker = (
        splash.make_splash_mqa if is_mqa else splash.make_splash_mha
    )
    splash_maker = functools.partial(splash_maker, q_seq_shards=shard_count)
  splash_fn = splash_maker(
      mask=splash_mask,
      config=splash_config,
      save_residuals=save_residuals,
      mask_value=float(jnp.finfo(jnp.float32).min),
      downcast_smem_data=False,
  )
  return splash_fn, splash_config



