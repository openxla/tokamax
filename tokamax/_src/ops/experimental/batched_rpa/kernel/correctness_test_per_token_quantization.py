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
"""Correctness tests for batched ragged paged attention with quantization."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax

from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.experimental.batched_rpa.kernel import configs
from tokamax._src.ops.experimental.batched_rpa.kernel.utils import (
    align_to, get_dtype_packing
)
from tokamax._src.ops.experimental.batched_rpa.kernel.wrapper import (
    ragged_paged_attention
)

jax.config.parse_flags_with_absl()


def pack_fp4_to_uint8(x):
  assert x.dtype == jnp.float4_e2m1fn
  assert x.shape[-1] % 2 == 0
  x_reshaped = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
  return jax.lax.bitcast_convert_type(x_reshaped, jnp.uint8)


def pack_kv_cache_fp4_to_uint8(kv_cache):
  assert kv_cache.dtype == jnp.float4_e2m1fn
  # kv_cache shape: (num_pages, num_kv_heads * 2, aligned_kv_head_dim // 8, 8, page_size)
  # Transpose dim 3 (kv_packing=8) and 4 (page_size)
  kv_cache = kv_cache.transpose(0, 1, 2, 4, 3)
  # Reshape (..., 8) to (..., 4, 2)
  kv_cache = kv_cache.reshape(*kv_cache.shape[:-1], 4, 2)
  # Bitcast to uint8
  kv_cache = jax.lax.bitcast_convert_type(kv_cache, jnp.uint8)
  # Transpose back
  kv_cache = kv_cache.transpose(0, 1, 2, 4, 3)
  return kv_cache


def unpack_kv_cache_uint8_to_fp4(kv_cache):
  assert kv_cache.dtype == jnp.uint8
  # kv_cache shape: (num_pages, num_kv_heads * 2, aligned_kv_head_dim // 4, 4, page_size)
  # Transpose dim 3 (kv_packing=4) and 4 (page_size)
  kv_cache = kv_cache.transpose(0, 1, 2, 4, 3)
  # Bitcast to float4
  kv_cache = jax.lax.bitcast_convert_type(kv_cache, jnp.float4_e2m1fn)
  # Reshape (..., 4, 2) to (..., 8)
  kv_cache = kv_cache.reshape(*kv_cache.shape[:-2],
                               kv_cache.shape[-2] * kv_cache.shape[-1])
  # Transpose back
  kv_cache = kv_cache.transpose(0, 1, 2, 4, 3)
  return kv_cache



def cdiv(a, b):
  return (a + b - 1) // b


FP4_VALUES = [
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


def quantize_kv_per_token(k, v, scale_dtype=jnp.bfloat16):
  """Quantizes K and V to FP4 E2M1 with per-token scaling."""
  # k, v shape: [T, H, D]
  fp4_vals_jax = jnp.array(FP4_VALUES, dtype=k.dtype)
  fp4_vals_jax_4d = fp4_vals_jax.reshape(1, 1, 1, -1)

  max_k = jnp.max(jnp.abs(k), axis=-1, keepdims=True)
  scale_k = max_k / 6.0
  scale_k = jnp.where(scale_k == 0, 1.0, scale_k)
  k_quant = fp4_vals_jax[
      jnp.argmin(jnp.abs((k / scale_k)[..., None] - fp4_vals_jax_4d), axis=-1)
  ]

  max_v = jnp.max(jnp.abs(v), axis=-1, keepdims=True)
  scale_v = max_v / 6.0
  scale_v = jnp.where(scale_v == 0, 1.0, scale_v)
  v_quant = fp4_vals_jax[
      jnp.argmin(jnp.abs((v / scale_v)[..., None] - fp4_vals_jax_4d), axis=-1)
  ]

  return (
      k_quant.astype(k.dtype),
      v_quant.astype(v.dtype),
      scale_k.squeeze(axis=-1).astype(scale_dtype),  # [T, H]
      scale_v.squeeze(axis=-1).astype(scale_dtype),  # [T, H]
  )


def quantize_kv_fp8_per_token(k, v, scale_dtype=jnp.bfloat16):
  """Quantizes K and V to actual FP8 E4M3FN with per-token scaling."""
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  max_k = jnp.max(jnp.abs(k), axis=-1, keepdims=True)
  scale_k = max_k / fp8_max
  scale_k_safe = jnp.where(scale_k == 0, 1.0, scale_k)
  k_quant = jnp.clip(jnp.round(k / scale_k_safe), -fp8_max, fp8_max).astype(
      jnp.float8_e4m3fn
  )

  max_v = jnp.max(jnp.abs(v), axis=-1, keepdims=True)
  scale_v = max_v / fp8_max
  scale_v_safe = jnp.where(scale_v == 0, 1.0, scale_v)
  v_quant = jnp.clip(jnp.round(v / scale_v_safe), -fp8_max, fp8_max).astype(
      jnp.float8_e4m3fn
  )

  return (
      k_quant,
      v_quant,
      scale_k.squeeze(axis=-1).astype(scale_dtype),
      scale_v.squeeze(axis=-1).astype(scale_dtype),
  )


def quantize_kv_per_tensor_with_scale(k, v, scale_k, scale_v):
  """Fake quantizes K and V to FP4 E2M1 using fixed scalar scales."""
  fp4_vals_jax = jnp.array(FP4_VALUES, dtype=k.dtype)
  fp4_vals_jax_4d = fp4_vals_jax.reshape(1, 1, 1, -1)

  scale_k_safe = jnp.where(scale_k == 0, 1.0, scale_k)
  k_quant = fp4_vals_jax[
      jnp.argmin(
          jnp.abs((k / scale_k_safe)[..., None] - fp4_vals_jax_4d), axis=-1
      )
  ]

  scale_v_safe = jnp.where(scale_v == 0, 1.0, scale_v)
  v_quant = fp4_vals_jax[
      jnp.argmin(
          jnp.abs((v / scale_v_safe)[..., None] - fp4_vals_jax_4d), axis=-1
      )
  ]
  return k_quant.astype(k.dtype), v_quant.astype(v.dtype)


def quantize_kv_per_tensor(k, v):
  """Fake quantizes K and V to FP4 E2M1 with single per-tensor scalar 
  scaling."""
  max_k = jnp.max(jnp.abs(k))
  scale_k = max_k / 6.0
  scale_k = jnp.where(scale_k == 0, 1.0, scale_k)

  max_v = jnp.max(jnp.abs(v))
  scale_v = max_v / 6.0
  scale_v = jnp.where(scale_v == 0, 1.0, scale_v)

  k_quant, v_quant = quantize_kv_per_tensor_with_scale(k, v, scale_k, scale_v)
  return k_quant, v_quant, float(scale_k), float(scale_v)


def merge_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
  assert k.shape == v.shape
  assert k.dtype == v.dtype
  max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
  kv_packing = get_dtype_packing(k.dtype)
  actual_num_kv_heads_x2 = actual_num_kv_heads * 2
  num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)

  head_dim = align_to(actual_head_dim, 128)
  # If head_dim is 129 (due to co-located scales), align_to(129, 8) in caller or
  # 136 in sublane
  if actual_head_dim > 128 and actual_head_dim <= 136:
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    head_dim = align_to(actual_head_dim, num_sublanes * kv_packing)

  kv = jnp.pad(
      jnp.concat([k, v], axis=-1).reshape(
          max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
      ),
      (
          (0, 0),
          (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
          (0, head_dim - actual_head_dim),
      ),
      constant_values=0,
  ).reshape(
      max_num_tokens,
      num_kv_heads_x2 // kv_packing,
      kv_packing,
      head_dim,
  )
  return kv


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # [total_num_pages, page_size,
                          # num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | jax.Array | None = None,
    v_scale: float | jax.Array | None = None,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
    use_per_token_scale: bool = False,
    per_token_scale_dtype: Any = jnp.bfloat16,
    kv_cache_scale: jax.Array | None = None,
):
  if dynamic_k_scale is not None:
    assert k_scale is None
    k_scale = dynamic_k_scale
  if dynamic_v_scale is not None:
    assert v_scale is None
    v_scale = dynamic_v_scale

  if out_dtype is None:
    out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16

  if mask_value is None:
    # We do not set to -inf directly because (-inf) - (-inf) is nan.
    mask_value = -float(jnp.finfo(out_dtype).max)

  actual_head_dim = queries.shape[2]
  actual_num_q_heads = queries.shape[1]
  actual_num_kv_heads = keys.shape[1]
  merged_kv = merge_kv(keys, values)

  _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
      kv_cache.shape
  )
  num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
  assert num_kv_heads_x2 % 2 == 0
  assert actual_num_q_heads % actual_num_kv_heads == 0
  assert get_dtype_packing(kv_cache.dtype) == kv_packing
  assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs
  outputs = []

  if use_per_token_scale:
    assert kv_cache_scale is not None
    assert isinstance(k_scale, (jax.Array, np.ndarray))
    assert isinstance(v_scale, (jax.Array, np.ndarray))

  for i in range(distribution[-1]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start

    kv_len = kv_lens[i]
    indices_start = i * pages_per_seq
    indices_end = indices_start + cdiv(kv_len, page_size)
    indices = page_indices[indices_start:indices_end]
    q = queries[q_start:q_end, :, :actual_head_dim]

    # Update the kv cache.
    assert kv_len - q_len >= 0
    gathered_kv = kv_cache[indices]
    gathered_shape = gathered_kv.shape
    gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
    gathered_kv = gathered_kv.at[kv_len - q_len : kv_len].set(
        merged_kv[q_start:q_end]
    )
    kv_cache = kv_cache.at[indices].set(gathered_kv.reshape(gathered_shape))

    kv = gathered_kv.reshape(-1, num_kv_heads_x2, head_dim)[
        :, : actual_num_kv_heads * 2, :
    ].reshape(-1, actual_num_kv_heads, head_dim * 2)
    k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
    v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]

    # Update the scale cache and dequantize K and V if using per-token scale
    if use_per_token_scale:
      assert kv_cache_scale is not None
      assert isinstance(k_scale, (jax.Array, np.ndarray))
      assert isinstance(v_scale, (jax.Array, np.ndarray))
      gathered_scale = kv_cache_scale[indices]
      gathered_scale_shape = gathered_scale.shape
      gathered_scale_2d = (
          gathered_scale.swapaxes(0, 1)
          .reshape(num_kv_heads_x2, -1)
          .swapaxes(0, 1)
      )

      new_k_scale = k_scale[q_start:q_end]
      new_v_scale = v_scale[q_start:q_end]
      new_scales = jnp.stack([new_k_scale, new_v_scale], axis=-1).reshape(
          -1, num_kv_heads_x2
      )

      gathered_scale_2d = gathered_scale_2d.at[kv_len - q_len : kv_len].set(
          new_scales
      )
      kv_cache_scale = kv_cache_scale.at[indices].set(
          gathered_scale_2d.swapaxes(0, 1)
          .reshape(num_kv_heads_x2, -1, page_size)
          .swapaxes(0, 1)
      )

      # Dequantize K and V using scales from cache
      seq_scales = gathered_scale_2d[:kv_len]
      seq_k_scale = seq_scales[:, 0::2, jnp.newaxis]  # [kv_len, H, 1]
      seq_v_scale = seq_scales[:, 1::2, jnp.newaxis]  # [kv_len, H, 1]
      k = (k.astype(jnp.float32) * seq_k_scale.astype(jnp.float32)).astype(queries.dtype)
      v = (v.astype(jnp.float32) * seq_v_scale.astype(jnp.float32)).astype(queries.dtype)

    k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
    v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

    if q_scale is not None:
      q = q / q_scale
      if jnp.issubdtype(k.dtype, jnp.floating):
        dtype_info = jnp.finfo(k.dtype)
        minval = float(dtype_info.min)
        maxval = float(dtype_info.max)
        q = jnp.clip(q, min=minval, max=maxval)
      q = q.astype(k.dtype)

    attn = jnp.einsum(
        "qhd,khd->hqk", q, k, preferred_element_type=jnp.float32
    ).astype(out_dtype)
    attn *= sm_scale

    if not use_per_token_scale:
      if k_scale is not None:
        attn *= k_scale
      if q_scale is not None:
        attn *= q_scale

    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)

    if use_causal_mask:
      q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
          jnp.int32, attn.shape, 1
      )
      kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
      mask = q_span >= kv_span
      if sliding_window is not None:
        mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
      attn = jnp.where(mask, attn, mask_value)

    attn = jax.nn.softmax(attn, axis=-1).astype(out_dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v,
                     preferred_element_type=jnp.float32)

    if not use_per_token_scale and v_scale is not None:
      out *= v_scale

    outputs.append(out.astype(out_dtype))

  return (
      jnp.concat(outputs, axis=0),
      kv_cache,
      kv_cache_scale,
  )


class RaggedPagedAttentionKernelTest(parameterized.TestCase):
  """Correctness tests for quantized batched ragged paged attention.

  This suite has been pruned to the minimum set of tests required to validate:
  1. Numerical error bounds of fake FP4 quantization (vs BF16 reference).
  2. Robustness to outliers (per-token vs per-tensor quantization).
  3. Scale channel packing layouts (1, 2, and 4 channels):
     - 1 channel: BF16 KV + BF16 scale (quantized_basic)
     - 2 channels: BF16 KV + FP32 scale (quantized_fp32_scale_basic)
                   or FP8 KV + BF16 scale (fp8_basic)
     - 4 channels: FP8 KV + FP32 scale (fp8_fp32_scale_basic)
  4. Mixed prefill/decode scheduling with quantization (quantized_mixed).
  5. Layout/padding edge cases with scale channels (quantized_complex).
  """

  def assertAllClose(self, x, y, atol=None, rtol=None):
    kwargs = {}
    if atol is not None:
      kwargs["atol"] = atol
    if rtol is not None:
      kwargs["rtol"] = rtol
    np.testing.assert_allclose(x, y, **kwargs)

  def assertArraysEqual(self, x, y):
    np.testing.assert_array_equal(x, y)

  def _test_ragged_paged_attention(
      self,
      seq_lens: list[tuple[int, int]],
      num_heads: tuple[int, int],
      head_dim: int,
      page_size: int,
      q_dtype: Any,
      kv_dtype: Any,
      num_pages: int,
      *,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      out_dtype: Any = None,
      bq_sz: int = 64,
      bkv_sz: int = 128,
      bq_csz: int = 64,
      bkv_csz: int = 128,
      vmem_limit_bytes: int | None = None,
      q_scale: float | None = None,
      k_scale: float | jax.Array | None = None,
      v_scale: float | jax.Array | None = None,
      kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
      use_per_token_scale: bool = False,
      per_token_scale_dtype: Any = jnp.bfloat16,
      use_per_tensor_scale: bool = False,
      inject_outliers: bool = False,
      quantize_fn: Any = quantize_kv_per_token,
      pack_inputs_to_uint8: bool = False,
      pack_cache_to_uint8: bool = False,
  ):
    rng = np.random.default_rng(1234)

    def gen_random(shape, dtype):
      return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)

    if use_per_token_scale:
      kv_layout = configs.KVLayout.SEQ_ALONG_LANE

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
      page_size = 128

    if jax.default_backend() != "tpu":
      self.skipTest("Expect TPU")
    try:
      if pltpu.get_tpu_info().generation < 4:
        self.skipTest("Expect TPUv4+")
    except (ValueError, RuntimeError, AttributeError):
      self.skipTest("Failed to get TPU info.")
    cu_q_lens = [0]
    kv_lens = []
    for q_len, kv_len in seq_lens:
      assert q_len <= kv_len
      cu_q_lens.append(cu_q_lens[-1] + q_len)
      kv_lens.append(kv_len)

    max_num_batched_tokens = max(
        align_to(cu_q_lens[-1], 128), 128
    )
    max_num_seq = max(align_to(len(seq_lens), 8), 128)
    max_kv_len = max(kv_lens)
    pages_per_seq = cdiv(max_kv_len, page_size)
    num_q_heads, num_kv_heads = num_heads

    q = gen_random((max_num_batched_tokens, num_q_heads, head_dim), q_dtype)
    k = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)
    v = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)

    if inject_outliers:
      k = k.at[::64, :, :4].set(50.0)
      v = v.at[::64, :, :4].set(50.0)

    if use_per_token_scale:
      assert k_scale is None and v_scale is None
      k_quant, v_quant, k_scale, v_scale = quantize_fn(
          k, v, scale_dtype=per_token_scale_dtype
      )
      k = k_quant
      v = v_quant
    elif use_per_tensor_scale:
      assert k_scale is None and v_scale is None
      k_quant, v_quant, k_scale, v_scale = quantize_kv_per_tensor(k, v)
      k = k_quant
      v = v_quant

    page_cnt = 0
    page_indices_list = []
    kv_pages_list = []
    kv_pages_rpa_list = []
    kv_scale_pages_list = []
    kv_packing = get_dtype_packing(kv_dtype)
    padded_head_dim = align_to(head_dim, 128)
    num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)

    for kv_len in kv_lens:
      k_cache_i = gen_random((kv_len, num_kv_heads, head_dim), kv_dtype)
      v_cache_i = gen_random((kv_len, num_kv_heads, head_dim), kv_dtype)
      if inject_outliers:
        k_cache_i = k_cache_i.at[::64, :, :4].set(50.0)
        v_cache_i = v_cache_i.at[::64, :, :4].set(50.0)

      if use_per_token_scale:
        k_cache_quant, v_cache_quant, scale_k_cache, scale_v_cache = (
            quantize_fn(
                k_cache_i, v_cache_i, scale_dtype=per_token_scale_dtype
            )
        )
        # Prepare reference scale pages for ref_ragged_paged_attention
        scales = (
            jnp.stack([scale_k_cache, scale_v_cache], axis=-1)
            .reshape(kv_len, num_kv_heads * 2)
            .swapaxes(0, 1)
        )
        scales_padded = jnp.pad(
            scales,
            (
                (0, 0),
                (0, cdiv(kv_len, page_size) * page_size - kv_len),
            ),
            constant_values=1.0,
        ).reshape(
            num_kv_heads * 2,
            -1,
            page_size,
        ).swapaxes(0, 1)
        kv_scale_pages_list.append(scales_padded)

        # 1. Reference token cache (no scale channels)
        kv_ref = merge_kv(k_cache_quant, v_cache_quant)

        kv_bits = jax.dtypes.itemsize_bits(k_cache_quant.dtype)
        scale_bits = jax.dtypes.itemsize_bits(per_token_scale_dtype)
        num_scale_channels = max(1, scale_bits // kv_bits)
        if kv_bits < scale_bits:
          scale_k_split = jax.lax.bitcast_convert_type(
              scale_k_cache, k_cache_quant.dtype
          ).reshape(*scale_k_cache.shape, num_scale_channels)
          scale_v_split = jax.lax.bitcast_convert_type(
              scale_v_cache, v_cache_quant.dtype
          ).reshape(*scale_v_cache.shape, num_scale_channels)
        else:
          scale_k_split = scale_k_cache[..., None]
          scale_v_split = scale_v_cache[..., None]
        k_with_sc = jnp.concatenate([k_cache_quant, scale_k_split], axis=-1)
        v_with_sc = jnp.concatenate([v_cache_quant, scale_v_split], axis=-1)
        kv_rpa = merge_kv(k_with_sc, v_with_sc)
      elif use_per_tensor_scale:
        k_cache_quant, v_cache_quant = quantize_kv_per_tensor_with_scale(
            k_cache_i, v_cache_i, k_scale, v_scale
        )
        kv_ref = merge_kv(k_cache_quant, v_cache_quant)
        kv_rpa = kv_ref
      else:
        kv_ref = merge_kv(k_cache_i, v_cache_i)
        kv_rpa = kv_ref

      # Pad and append reference page slices
      kv_padded_ref = jnp.pad(
          kv_ref,
          (
              (0, cdiv(kv_len, page_size) * page_size - kv_len),
              (0, 0),
              (0, 0),
              (0, 0),
          ),
          constant_values=jnp.nan,
      ).reshape(
          -1,
          page_size,
          num_kv_heads_x2 // kv_packing,
          kv_packing,
          kv_ref.shape[-1],
      )
      # Pad and append RPA co-located page slices
      kv_padded_rpa = jnp.pad(
          kv_rpa,
          (
              (0, cdiv(kv_len, page_size) * page_size - kv_len),
              (0, 0),
              (0, 0),
              (0, 0),
          ),
          constant_values=jnp.nan,
      ).reshape(
          -1,
          page_size,
          num_kv_heads_x2 // kv_packing,
          kv_packing,
          kv_rpa.shape[-1],
      )

      indices = page_cnt + jnp.arange(kv_padded_ref.shape[0], dtype=jnp.int32)
      indices = jnp.pad(
          indices,
          ((0, pages_per_seq - indices.shape[0]),),
          constant_values=jnp.nan,
      )
      page_indices_list.append(indices)
      page_cnt += kv_padded_ref.shape[0]
      kv_pages_list.append(kv_padded_ref)
      kv_pages_rpa_list.append(kv_padded_rpa)

    kv_cache_ref = jnp.concatenate(kv_pages_list, axis=0)
    kv_cache_ref = jnp.pad(
        kv_cache_ref,
        ((0, num_pages - kv_cache_ref.shape[0]),
         (0, 0), (0, 0), (0, 0), (0, 0)),
        constant_values=jnp.nan,
    )

    rpa_kv_cache = jnp.concatenate(kv_pages_rpa_list, axis=0)
    rpa_kv_cache = jnp.pad(
        rpa_kv_cache,
        ((0, num_pages - rpa_kv_cache.shape[0]),
         (0, 0), (0, 0), (0, 0), (0, 0)),
        constant_values=jnp.nan,
    )

    kv_cache_scale = None
    if use_per_token_scale:
      kv_cache_scale = jnp.concatenate(kv_scale_pages_list, axis=0)
      kv_cache_scale = jnp.pad(
          kv_cache_scale,
          ((0, num_pages - kv_cache_scale.shape[0]), (0, 0), (0, 0)),
          constant_values=1.0,
      )

    page_indices = jnp.stack(page_indices_list, axis=0)
    page_indices = jnp.pad(
        page_indices,
        ((0, max_num_seq - page_indices.shape[0]), (0, 0)),
        constant_values=jnp.nan,
    )
    page_indices = page_indices.reshape(-1)

    cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
    cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seq + 1 - cu_q_lens.shape[0]))
    kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
    kv_lens = jnp.pad(kv_lens, (0, max_num_seq - kv_lens.shape[0]))
    distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)

    args_ref = (
        q,
        k,
        v,
        kv_cache_ref,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    kwargs = {
        "sliding_window": sliding_window,
        "soft_cap": soft_cap,
        "q_scale": q_scale,
        "use_per_token_scale": use_per_token_scale,
        "per_token_scale_dtype": per_token_scale_dtype,
    }
    if use_per_token_scale:
      kwargs["dynamic_k_scale"] = k_scale
      kwargs["dynamic_v_scale"] = v_scale
    else:
      kwargs["k_scale"] = k_scale
      kwargs["v_scale"] = v_scale

    expected, expected_kv_cache, expected_kv_scale = ref_ragged_paged_attention(
        *args_ref,
        **kwargs,
        kv_cache_scale=kv_cache_scale,
    )

    num_sublanes = pltpu.get_tpu_info().num_sublanes

    base_dim = head_dim
    if use_per_token_scale:
      kv_bits = jax.dtypes.itemsize_bits(jnp.dtype(kv_dtype))
      scale_bits = jax.dtypes.itemsize_bits(per_token_scale_dtype)
      num_scale_channels = max(1, scale_bits // kv_bits)
      base_dim += num_scale_channels

    aligned_head_dim_rpa = align_to(
        base_dim,
        num_sublanes * kv_packing
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE
        else 128,
    )

    if aligned_head_dim_rpa != rpa_kv_cache.shape[-1]:
      rpa_kv_cache = rpa_kv_cache[..., :aligned_head_dim_rpa]

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
      rpa_kv_cache = rpa_kv_cache.reshape(
          num_pages,
          page_size,
          num_kv_heads_x2,
          aligned_head_dim_rpa // kv_packing,
          kv_packing,
      ).transpose(0, 2, 3, 4, 1)[:, :num_kv_heads * 2]

    if pack_inputs_to_uint8:
      k = pack_fp4_to_uint8(k)
      v = pack_fp4_to_uint8(v)

    if pack_cache_to_uint8:
      rpa_kv_cache = pack_kv_cache_fp4_to_uint8(rpa_kv_cache)

    rpa_args = (
        q,
        k,
        v,
        rpa_kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    rpa_bkv_sz = bkv_sz
    rpa_n_buffer = 3
    rpa_batch_size = 2
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
      rpa_bkv_sz = max(bkv_sz, 128)
      rpa_n_buffer = 2
      rpa_batch_size = 1

    decode_block_sizes = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_csz,
        bkv_sz=rpa_bkv_sz,
        batch_size=rpa_batch_size,
        n_buffer=rpa_n_buffer,
    )
    prefill_block_sizes = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_csz,
        bkv_sz=rpa_bkv_sz,
        batch_size=rpa_batch_size,
        n_buffer=rpa_n_buffer,
    )

    output, updated_kv_cache = ragged_paged_attention(
        *rpa_args,
        **kwargs,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        kv_layout=kv_layout,
    )
    if pack_cache_to_uint8:
      updated_kv_cache = unpack_kv_cache_uint8_to_fp4(updated_kv_cache)
    output = output[: cu_q_lens[distribution[-1]]]

    dtype_bits = jax.dtypes.itemsize_bits(jnp.dtype(kv_dtype))
    tols = {
        32: 0.15,
        16: 0.2,
        8: 0.2,
        4: 0.2,
    }
    tol = tols[dtype_bits]
    if inject_outliers:
      tol = max(tol, 0.25)
    self.assertAllClose(output, expected, atol=tol, rtol=tol)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
      tmp = updated_kv_cache.transpose(0, 4, 1, 2, 3)
      if num_kv_heads * 2 != num_kv_heads_x2:
        tmp = jnp.pad(
            tmp,
            ((0, 0), (0, 0), (0, num_kv_heads_x2 - num_kv_heads * 2),
             (0, 0), (0, 0)),
            constant_values=jnp.nan,
        )
      updated_kv_cache = tmp.reshape(
          num_pages,
          page_size,
          num_kv_heads_x2 // kv_packing,
          kv_packing,
          aligned_head_dim_rpa,
      )

    updated_kv_scale = None
    if aligned_head_dim_rpa != head_dim:
      if use_per_token_scale:
        kv_bits = jax.dtypes.itemsize_bits(jnp.dtype(kv_dtype))
        scale_bits = jax.dtypes.itemsize_bits(per_token_scale_dtype)
        num_scale_channels = max(1, scale_bits // kv_bits)
        unpadded_cache = updated_kv_cache[:, :, : num_kv_heads * 2]
        if kv_bits < scale_bits:
          scale_slice = unpadded_cache[
              ..., head_dim : head_dim + num_scale_channels
          ]
          if kv_bits == 4 and scale_bits == 16:
            scale_slice_reshaped = scale_slice.reshape(*scale_slice.shape[:-1], 2, 2)
            packed_uint8 = jax.lax.bitcast_convert_type(scale_slice_reshaped, jnp.uint8)
            lo = packed_uint8[..., 0].astype(jnp.uint16)
            hi = packed_uint8[..., 1].astype(jnp.uint16)
            updated_kv_scale = jax.lax.bitcast_convert_type(lo | (hi << 8), per_token_scale_dtype)
          elif kv_bits == 4 and scale_bits == 32:
            scale_slice_reshaped = scale_slice.reshape(*scale_slice.shape[:-1], 4, 2)
            packed_uint8 = jax.lax.bitcast_convert_type(scale_slice_reshaped, jnp.uint8)
            b0 = packed_uint8[..., 0].astype(jnp.uint32)
            b1 = packed_uint8[..., 1].astype(jnp.uint32)
            b2 = packed_uint8[..., 2].astype(jnp.uint32)
            b3 = packed_uint8[..., 3].astype(jnp.uint32)
            updated_kv_scale = jax.lax.bitcast_convert_type(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24), per_token_scale_dtype)
          elif kv_bits == 8 and scale_bits == 16:
            lo = jax.lax.bitcast_convert_type(scale_slice[..., 0], jnp.uint8).astype(jnp.uint16)
            hi = jax.lax.bitcast_convert_type(scale_slice[..., 1], jnp.uint8).astype(jnp.uint16)
            updated_kv_scale = jax.lax.bitcast_convert_type(lo | (hi << 8), per_token_scale_dtype)
          elif kv_bits == 8 and scale_bits == 32:
            b0 = jax.lax.bitcast_convert_type(scale_slice[..., 0], jnp.uint8).astype(jnp.uint32)
            b1 = jax.lax.bitcast_convert_type(scale_slice[..., 1], jnp.uint8).astype(jnp.uint32)
            b2 = jax.lax.bitcast_convert_type(scale_slice[..., 2], jnp.uint8).astype(jnp.uint32)
            b3 = jax.lax.bitcast_convert_type(scale_slice[..., 3], jnp.uint8).astype(jnp.uint32)
            updated_kv_scale = jax.lax.bitcast_convert_type(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24), per_token_scale_dtype)
          elif kv_bits == 16 and scale_bits == 32:
            w0 = jax.lax.bitcast_convert_type(scale_slice[..., 0], jnp.uint16).astype(jnp.uint32)
            w1 = jax.lax.bitcast_convert_type(scale_slice[..., 1], jnp.uint16).astype(jnp.uint32)
            updated_kv_scale = jax.lax.bitcast_convert_type(w0 | (w1 << 16), per_token_scale_dtype)
          else:
            updated_kv_scale = jax.lax.bitcast_convert_type(
                scale_slice, per_token_scale_dtype
            )
        else:
          updated_kv_scale = unpadded_cache[..., head_dim].astype(
              per_token_scale_dtype
          )
      updated_kv_cache = updated_kv_cache[..., :head_dim]
      expected_kv_cache = expected_kv_cache[..., :head_dim]

    kv_bits = jax.dtypes.itemsize_bits(jnp.dtype(kv_dtype))
    mask = None
    if kv_bits == 4:
      cache_mask = np.zeros((num_pages, page_size), dtype=bool)
      for seq_idx, (q_len, kv_len) in enumerate(seq_lens):
        pages_needed = cdiv(kv_len, page_size)
        for p_idx in range(pages_needed):
          page_num = page_indices_list[seq_idx][p_idx]
          if page_num >= 0:
            if p_idx == pages_needed - 1:
              valid_in_page = kv_len - p_idx * page_size
              cache_mask[page_num, :valid_in_page] = True
            else:
              cache_mask[page_num, :] = True
      mask = jnp.array(cache_mask)
      self.assertArraysEqual(updated_kv_cache[mask], expected_kv_cache[mask])
    else:
      self.assertTrue(
          jnp.all(
              jnp.logical_or(
                  jnp.isnan(expected_kv_cache),
                  updated_kv_cache == expected_kv_cache,
              )
          ).item()
      )

    if use_per_token_scale and updated_kv_scale is not None:
      if kv_bits == 4 and mask is not None:
        mask_scale = mask
      else:
        mask_scale = ~jnp.isnan(expected_kv_cache).any(axis=(2, 3, 4))
      self.assertAllClose(
          updated_kv_scale.reshape(num_pages, page_size, -1)[mask_scale],
          expected_kv_scale.swapaxes(1, 2)[mask_scale],
          atol=1e-5,
          rtol=1e-5,
      )

    self.assertEqual(output.shape[-1], head_dim)
    return output

  @parameterized.product(
      block_sizes=[
          (64, 256, 32, 128),
      ],
  )
  def test_ragged_paged_attention_quantized_vs_no_quant(self, block_sizes):
    dtype = jnp.bfloat16
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes

    out_no_quant = self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=False,
        use_per_tensor_scale=False,
    )

    out_quant_pt = self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.bfloat16,
        use_per_tensor_scale=False,
    )

    out_quant_std = self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=False,
        use_per_tensor_scale=True,
    )

    def calc_metrics(ref, target):
      abs_diff = jnp.abs(ref - target)
      mae = jnp.mean(abs_diff)
      max_ae = jnp.max(abs_diff)
      epsilon = 1e-9
      rel_diff = abs_diff / (jnp.abs(ref) + epsilon)
      mre = jnp.mean(rel_diff)
      max_re = jnp.max(rel_diff)
      mse = jnp.mean(jnp.square(ref - target))
      flat_ref = ref.reshape(-1)
      flat_tgt = target.reshape(-1)
      cos_sim = jnp.dot(flat_ref, flat_tgt) / (
          jnp.linalg.norm(flat_ref) * jnp.linalg.norm(flat_tgt) + epsilon
      )
      return (
          float(mae),
          float(max_ae),
          float(mre),
          float(max_re),
          float(mse),
          float(cos_sim),
      )

    mae_pt, max_pt, mre_pt, max_re_pt, mse_pt, cos_pt = calc_metrics(
        out_no_quant, out_quant_pt
    )
    mae_std, max_std, mre_std, max_re_std, mse_std, cos_std = calc_metrics(
        out_no_quant, out_quant_std
    )

    print(f"\n============================================================")
    print(f"--- Quantization Error Comparison (Fake FP4 vs BF16 Ref) ---")
    print(f"============================================================")
    print(f"Metric                    | Per-Token FP4  | Per-Tensor FP4")
    print(f"--------------------------|----------------|----------------")
    print(f"Mean Absolute Error (MAE) | {mae_pt:14.6f} | {mae_std:14.6f}")
    print(f"Max Absolute Error        | {max_pt:14.6f} | {max_std:14.6f}")
    print(f"Mean Relative Error       | {mre_pt:14.6f} | {mre_std:14.6f}")
    print(f"Max Relative Error        | {max_re_pt:14.6f} | {max_re_std:14.6f}")
    print(f"Mean Squared Error (MSE)  | {mse_pt:14.6f} | {mse_std:14.6f}")
    print(f"Cosine Similarity         | {cos_pt:14.6f} | {cos_std:14.6f}")
    print(f"============================================================\n")

    self.assertAllClose(mae_pt, mae_std, atol=0.01)

    with self.assertRaises(AssertionError):
      self.assertAllClose(out_no_quant, out_quant_pt, atol=1e-5)
    with self.assertRaises(AssertionError):
      self.assertAllClose(out_no_quant, out_quant_std, atol=1e-5)

    self.assertAllClose(out_no_quant, out_quant_pt, atol=0.8, rtol=0.8)
    self.assertAllClose(out_no_quant, out_quant_std, atol=0.8, rtol=0.8)

  @parameterized.product(
      block_sizes=[
          (64, 256, 32, 128),
      ],
  )
  def test_ragged_paged_attention_quantized_outliers(self, block_sizes):
    dtype = jnp.bfloat16
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes

    out_no_quant = self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=False,
        use_per_tensor_scale=False,
        inject_outliers=True,
    )

    out_quant_pt = self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.bfloat16,
        use_per_tensor_scale=False,
        inject_outliers=True,
    )

    out_quant_std = self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=False,
        use_per_tensor_scale=True,
        inject_outliers=True,
    )

    def calc_metrics(ref, target):
      abs_diff = jnp.abs(ref - target)
      mae = jnp.mean(abs_diff)
      max_ae = jnp.max(abs_diff)
      epsilon = 1e-9
      rel_diff = abs_diff / (jnp.abs(ref) + epsilon)
      mre = jnp.mean(rel_diff)
      max_re = jnp.max(rel_diff)
      mse = jnp.mean(jnp.square(ref - target))
      return float(mae), float(max_ae), float(mre), float(max_re), float(mse)

    mae_pt, max_pt, mre_pt, max_re_pt, mse_pt = calc_metrics(
        out_no_quant, out_quant_pt
    )
    mae_std, max_std, mre_std, max_re_std, mse_std = calc_metrics(
        out_no_quant, out_quant_std
    )

    print(f"\n============================================================")
    print(f"--- Quantization Error with LLM Outliers (+50.0 Spikes) ----")
    print(f"============================================================")
    print(f"Metric                    | Per-Token FP4  | Per-Tensor FP4")
    print(f"--------------------------|----------------|----------------")
    print(f"Mean Absolute Error (MAE) | {mae_pt:14.6f} | {mae_std:14.6f}")
    print(f"Max Absolute Error        | {max_pt:14.6f} | {max_std:14.6f}")
    print(f"Mean Relative Error       | {mre_pt:14.6f} | {mre_std:14.6f}")
    print(f"Max Relative Error        | {max_re_pt:14.6f} | {max_re_std:14.6f}")
    print(f"============================================================\n")

  @parameterized.product(
      block_sizes_and_page_size=[
          ((64, 256, 32, 128), 64),
          ((60, 48, 30, 48), 16),
      ],
      kv_layout=[
          configs.KVLayout.SEQ_ALONG_LANE,
      ],
  )
  def test_ragged_paged_attention_quantized_basic(
      self, block_sizes_and_page_size, kv_layout
  ):
    dtype = jnp.bfloat16
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    num_pages = 1000

    block_sizes, page_size = block_sizes_and_page_size
    bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.bfloat16,
        kv_layout=kv_layout,
    )

  @parameterized.product(
      block_sizes_and_page_size=[
          ((64, 256, 32, 128), 64),
      ],
      kv_layout=[
          configs.KVLayout.SEQ_ALONG_LANE,
      ],
  )
  def test_ragged_paged_attention_fp8_basic(
      self, block_sizes_and_page_size, kv_layout
  ):
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.float8_e4m3fn
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    num_pages = 1000

    block_sizes, page_size = block_sizes_and_page_size
    bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.bfloat16,
        kv_layout=kv_layout,
    )





  def test_ragged_paged_attention_quantized_mixed(self):
    dtype = jnp.bfloat16
    seq_lens = [
        (5, 18),
        (1, 129),
        (120, 597),
        (1, 122),
        (1, 64),
        (32, 322),
        (251, 463),
        (1, 181),
        (1, 1107),
        (99, 123),
        (1, 31),
        (5, 18),
        (3, 1229),
        (117, 229),
        (1, 87),
        (1, 1328),
    ]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.bfloat16,
    )

  @parameterized.product(
      num_seqs=[1, 17],
      num_heads=[(32, 8), (12, 2), (5, 1), (3, 3)],
      head_dim=[80, 240],
  )
  def test_ragged_paged_attention_quantized_complex(
      self,
      num_seqs,
      num_heads,
      head_dim,
  ):
    dtype = jnp.bfloat16
    rng = np.random.default_rng(1234)
    q_lens = rng.integers(1, 100, num_seqs)
    kv_lens = q_lens + rng.integers(0, 50, num_seqs)
    seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.bfloat16,
    )





  @parameterized.product(
      block_sizes_and_page_size=[
          ((64, 256, 32, 128), 64),
      ],
      kv_layout=[
          configs.KVLayout.SEQ_ALONG_LANE,
      ],
  )
  def test_ragged_paged_attention_quantized_fp32_scale_basic(
      self, block_sizes_and_page_size, kv_layout
  ):
    dtype = jnp.bfloat16
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    num_pages = 1000

    block_sizes, page_size = block_sizes_and_page_size
    bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.float32,
        kv_layout=kv_layout,
        quantize_fn=quantize_kv_per_token,
    )

  @parameterized.product(
      block_sizes_and_page_size=[
          ((64, 256, 32, 128), 64),
      ],
      kv_layout=[
          configs.KVLayout.SEQ_ALONG_LANE,
      ],
  )
  def test_ragged_paged_attention_fp8_fp32_scale_basic(
      self, block_sizes_and_page_size, kv_layout
  ):
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.float8_e4m3fn
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    num_pages = 1000

    block_sizes, page_size = block_sizes_and_page_size
    bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        bkv_csz=bkv_csz,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.float32,
        kv_layout=kv_layout,
        quantize_fn=quantize_kv_fp8_per_token,
    )



  def test_ragged_paged_attention_fp4_packed_per_token_scale(self):
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.float4_e2m1fn
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        use_per_token_scale=True,
        per_token_scale_dtype=jnp.bfloat16,
        pack_inputs_to_uint8=True,
        pack_cache_to_uint8=True,
    )


if __name__ == "__main__":
  absltest.main()