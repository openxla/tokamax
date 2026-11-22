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
"""Sparse MLA benchmark argument specifications."""

from typing import Final, Literal

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct

_HEAD_DIM = 512
_COMPRESSION_RATIO = 4
_DEVICE_INIT_THRESHOLD = 1024 * 1024

# Bit masks `(raw & and_mask) | or_mask` on uniform uint32 words so every
# packed sub-word is independently random, zero-mean, and finite:
# - NoPE (`cache_kv_nope`, `int32[..., 128]`): bytes 0..2 (all lanes) and byte 3
#   (lanes 0..63) hold FP8 E4M3 values `(b & 0x8F) | 0x40` (+-[2.0, 7.5]); byte
#   3 of lanes 64..127 holds E8M0 scales `(b & 0x03) | 0x78` ({2^-7..2^-4}).
#   All values exclude 0x7F/0xFF, so they stay finite in both formats.
# - RoPE (`cache_kv_rope`, `int32[..., 128]`): both bfloat16 halves use
#   `(h & 0x80FF) | 0x3E00` (+-[0.125, 0.498]).
_NOPE_FP8_AND = 0x8F8F8F8F
_NOPE_FP8_OR = 0x40404040
_NOPE_SCALE_AND = 0x038F8F8F
_NOPE_SCALE_OR = 0x78404040
_ROPE_AND = 0x80FF80FF
_ROPE_OR = 0x3E003E00


class _PackedCacheInitializer(jax.ShapeDtypeStruct, numerics.ArrayInitializer):
  """Initializes packed int32 CSA KV cache words with finite random sub-words."""

  def __init__(self, shape: tuple[int, ...], kind: Literal["nope", "rope"]):
    jax.ShapeDtypeStruct.__init__(self, shape, jnp.int32)
    self.kind = kind

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _PackedCacheInitializer):
      return False
    return self.kind == other.kind and jax.ShapeDtypeStruct.__eq__(self, other)

  def __hash__(self) -> int:
    return hash((jax.ShapeDtypeStruct.__hash__(self), self.kind))

  def __call__(self, rng: np.random.Generator) -> np.ndarray | jax.Array:
    if self.size >= _DEVICE_INIT_THRESHOLD:
      seed = jnp.int32(rng.integers(0, 2**31))
      shape = self.shape
      is_nope = self.kind == "nope"

      def _init_dev(s: jax.Array) -> jax.Array:
        raw = jax.random.bits(jax.random.key(s), shape, dtype=jnp.uint32)
        if is_nope:
          scale_lane = jnp.arange(shape[-1], dtype=jnp.uint32) >= 64
          and_mask = jnp.where(
              scale_lane,
              jnp.uint32(_NOPE_SCALE_AND),
              jnp.uint32(_NOPE_FP8_AND),
          )
          or_mask = jnp.where(
              scale_lane, jnp.uint32(_NOPE_SCALE_OR), jnp.uint32(_NOPE_FP8_OR)
          )
        else:
          and_mask = jnp.uint32(_ROPE_AND)
          or_mask = jnp.uint32(_ROPE_OR)
        return lax.bitcast_convert_type((raw & and_mask) | or_mask, jnp.int32)

      return jax.jit(_init_dev)(seed)

    raw = rng.integers(0, 1 << 32, size=self.shape, dtype=np.uint32)
    if self.kind == "nope":
      scale_lane = np.arange(self.shape[-1]) >= 64
      and_mask = np.where(
          scale_lane, np.uint32(_NOPE_SCALE_AND), np.uint32(_NOPE_FP8_AND)
      )
      or_mask = np.where(
          scale_lane, np.uint32(_NOPE_SCALE_OR), np.uint32(_NOPE_FP8_OR)
      )
    else:
      and_mask = np.uint32(_ROPE_AND)
      or_mask = np.uint32(_ROPE_OR)
    return ((raw & and_mask) | or_mask).view(np.int32)


class _HashableNPArray(np.ndarray):
  """Hashable numpy array for use as an ArgSpec argument."""

  def __new__(cls, input_array):
    return np.asarray(input_array).view(cls)

  def __hash__(self):
    return hash((self.tobytes(), self.shape, self.dtype))


def _int32_array(x) -> _HashableNPArray:
  return _HashableNPArray(np.asarray(x, dtype=np.int32))


def _cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def _make_argspec(
    *,
    kv_len_val: int,
    batch_size: int = 1,
    q_len: int = 1,
    page_size: int = 1024,
    csa_topk: int = 2048,
    num_q_heads: int = 64,
    actual_batch_size: int | None = None,
    head_dim: int = _HEAD_DIM,
    q_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    primary: bool = False,
    name: str | None = None,
) -> arg_spec.ArgSpec:
  """Makes a single argspec for Sparse MLA."""
  if name is None:
    mode = "decode" if q_len == 1 else "prefill"
    act_str = (
        f"_act{actual_batch_size}" if actual_batch_size is not None else ""
    )
    page_str = f"_p{page_size}" if page_size != 1024 else ""
    name = (
        f"{mode}_b{batch_size}{act_str}_q{q_len}_kv{kv_len_val}"
        f"{page_str}_topk{csa_topk}_h{num_q_heads}"
    )
  tags: tuple[arg_spec.Tag, ...] = (
      ("primary", "forward_only") if primary else ("forward_only",)
  )
  compressed_page_size = page_size // _COMPRESSION_RATIO
  kv_len_after_compression = kv_len_val // _COMPRESSION_RATIO
  num_tokens = batch_size * q_len
  cu_q_lens = np.arange(0, num_tokens + 1, q_len, dtype=np.int32)

  pages_per_seq = _cdiv(kv_len_after_compression, compressed_page_size) + 2
  total_pages = batch_size * pages_per_seq
  page_indices = np.arange(total_pages, dtype=np.int32)

  effective_batch_size = (
      batch_size if actual_batch_size is None else actual_batch_size
  )
  num_decode_seqs = effective_batch_size if q_len == 1 else 0
  distribution = np.array(
      [num_decode_seqs, num_decode_seqs, effective_batch_size], dtype=np.int32
  )

  args = {
      "q": ShapeDtype((num_tokens, num_q_heads, head_dim), q_dtype),
      "cache_kv_nope": _PackedCacheInitializer(
          (total_pages, compressed_page_size, 128), "nope"
      ),
      "cache_kv_rope": _PackedCacheInitializer(
          (total_pages, compressed_page_size // 4, 128), "rope"
      ),
      "topk_indices": numerics.RangedArrayInitializer(
          (num_tokens, csa_topk), jnp.int32, 0, kv_len_after_compression
      ),
      "page_indices": _int32_array(page_indices),
      "cu_q_lens": _int32_array(cu_q_lens),
      "distribution": _int32_array(distribution),
      "attention_sinks": ShapeDtype((num_q_heads,), jnp.float32),
      "swa_accumution": ShapeDtype(
          (num_tokens, num_q_heads, head_dim), q_dtype
      ),
      "swa_l": numerics.FloatRangedArrayInitializer(
          (num_tokens, num_q_heads), jnp.float32, 0.5, 2.0
      ),
      "swa_m": ShapeDtype((num_tokens, num_q_heads), jnp.float32),
      "sm_scale": 1.0,
  }

  return arg_spec.ArgSpec(
      args=args,
      project="inference",
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    # DeepSeek-V4 shapes
    _make_argspec(
        batch_size=256,
        kv_len_val=9216,
        csa_topk=1024,
        num_q_heads=128,
        primary=True,
    ),
    _make_argspec(
        q_len=1024, kv_len_val=1024, csa_topk=1024, num_q_heads=128
    ),
    _make_argspec(
        q_len=1024,
        kv_len_val=8192,
        csa_topk=1024,
        num_q_heads=128,
        primary=True,
    ),
    _make_argspec(
        batch_size=64, kv_len_val=1048576, csa_topk=1024, num_q_heads=128
    ),
    _make_argspec(
        q_len=1024, kv_len_val=1048576, csa_topk=1024, num_q_heads=128
    ),
    _make_argspec(
        batch_size=256, kv_len_val=9216, csa_topk=512, primary=True
    ),
    _make_argspec(q_len=1024, kv_len_val=1024, csa_topk=512),
    _make_argspec(q_len=1024, kv_len_val=8192, csa_topk=512, primary=True),
    # GLM-5.2 decode shapes
    _make_argspec(batch_size=64, kv_len_val=8192),
    _make_argspec(batch_size=128, kv_len_val=8192),
    _make_argspec(batch_size=64, kv_len_val=9216),
    _make_argspec(batch_size=128, kv_len_val=9216, primary=True),
    _make_argspec(batch_size=1, kv_len_val=262144),
    _make_argspec(batch_size=16, kv_len_val=262144),
    _make_argspec(batch_size=32, kv_len_val=262144),
    _make_argspec(batch_size=16, actual_batch_size=8, kv_len_val=131072),
    _make_argspec(batch_size=64, actual_batch_size=40, kv_len_val=131072),
    # GLM-5.2 prefill shapes
    _make_argspec(q_len=1024, kv_len_val=1024),
    _make_argspec(q_len=1024, kv_len_val=8192, primary=True),
    _make_argspec(q_len=2048, kv_len_val=2048),
    _make_argspec(q_len=2048, kv_len_val=8192),
    _make_argspec(q_len=1024, kv_len_val=1024, page_size=256),
    _make_argspec(q_len=1024, kv_len_val=8192, page_size=256),
    _make_argspec(q_len=1024, kv_len_val=131072),
    _make_argspec(q_len=4096, kv_len_val=131072),
    _make_argspec(q_len=1024, kv_len_val=262144),
)
