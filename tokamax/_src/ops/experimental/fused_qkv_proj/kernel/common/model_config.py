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
"""Per-chip model shapes and options, and the weights the kernels take.

Shape dims, used with these names across the package:
  T: tokens.
  tile: tokens per tile (one kernel grid step).
  rows: rows of a helper's block (a group of rows within a tile).
  K: hidden size, the projection's contraction dim.
  N: projection columns (all heads' blocks side by side).
  Hq: query heads.
  Hkv: KV heads.
  Hv: GDN value heads.
  D: head dim.
  n_pairs: RoPE rotation pairs per head.
  block: columns per fp8 activation quant block (act_quant_block).
  lanes: TPU vector lanes (128).
"""

import dataclasses
import enum
from typing import Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jnp

# The Pinned* weights' leaves: jax.Array outside the kernel, jax.Ref inside.
Leaf = TypeVar("Leaf")


class Norm(enum.StrEnum):
  """A norm: the decoder input norm, or the per-head norm of q, k or v.

  A string enum, so the kernels' entry points also accept the plain strings,
  e.g. input_norm="rms_one_plus_w".
  """

  NONE = "none"
  RMS = "rms"  # RMSNorm: f32 normalize -> bf16 -> * w in bf16.
  # vLLM GemmaRMSNorm (also Qwen3.5's, as Qwen3NextRMSNorm): w is stored as an
  # offset from 1, and f32 normalize * (1 + w) -> bf16.
  RMS_ONE_PLUS_W = "rms_one_plus_w"
  RMS_NOSCALE = "rms_noscale"  # RMSNorm(has_weight=False).


class RopePrecision(enum.Enum):
  """The precision RoPE rotates in; heads are rounded to bf16 once after."""

  # vLLM TPU's: bf16 q/k and cos/sin upcast to f32 (vllm-torchtpu
  # layers/adapter/rotary_embedding.py, matching the CUDA kernel's numerics).
  FP32 = "fp32"
  BF16 = "bf16"  # Every product and sum rounded to bf16.


@dataclasses.dataclass(frozen=True)
class Rope:
  """Which dims of a head NeoX RoPE rotates, as pairs (i, offset + i).

  NeoX style (GPT-NeoX's convention, vLLM's is_neox_style=True) rotates dim i
  with dim i + n_pairs of the same head, rather than with its neighbor i + 1
  (GPT-J style). For i < n_pairs, x[i] pairs with x[offset + i]; an offset
  larger than n_pairs leaves the dims in between unrotated (Gemma 4's partial
  RoPE).
  """

  n_pairs: int
  offset: int

  @classmethod
  def neox(cls, rotary_dim: int) -> "Rope":
    return cls(rotary_dim // 2, rotary_dim // 2)


@dataclasses.dataclass(frozen=True)
class AttentionConfig:
  """Attention shapes and options for one chip (head counts after TP)."""

  num_q_heads: int  # Hq.
  num_kv_heads: int  # Hkv.
  head_dim: int  # D.
  gate: bool = False  # Qwen3.5: q_proj also outputs a per-head output gate.
  v_from_k: bool = False  # Gemma 4 full attn: no v_proj; v = raw k projection.
  fp8: bool = True
  # fp8 only: columns per dynamic activation quant block (sub-channel). Must
  # divide the hidden size; larger blocks mean less dequant work in the kernel.
  act_quant_block: int = 1024
  bias: bool = False
  # Decoder input norm; with `residual`, the residual add is fused before it.
  input_norm: Norm = Norm.RMS
  residual: bool = False
  q_norm: Norm = Norm.NONE
  k_norm: Norm = Norm.NONE
  v_norm: Norm = Norm.NONE
  rope: Rope | None = None
  rope_precision: RopePrecision = RopePrecision.FP32
  eps: float = 1e-6
  q_dtype: jax.typing.DTypeLike = jnp.bfloat16
  kv_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn
  k_scale: float = 1.0
  v_scale: float = 1.0

  @property
  def q_packing(self) -> int:  # Q values per 32-bit word.
    return 32 // jax.dtypes.itemsize_bits(self.q_dtype)

  @property
  def kv_packing(self) -> int:  # KV values per 32-bit word.
    return 32 // jax.dtypes.itemsize_bits(self.kv_dtype)

  @property
  def q_dim(self) -> int:  # Hq * D (o_proj's contraction dim).
    return self.num_q_heads * self.head_dim

  @property
  def q_per_kv(self) -> int:  # Query heads per KV head.
    return self.num_q_heads // self.num_kv_heads

  @property
  def has_input_norm(self) -> bool:
    return self.input_norm != Norm.NONE

  # W is q_proj, k_proj and v_proj side by side; each block is one head:
  #   [q_0 | q_1 | ... | k_0 | k_1 | ... | v_0 | v_1 | ...]
  # With `gate`, each q head is followed by its gate: [q_0 | g_0 | q_1 | ...].
  # With `v_from_k`, there are no v blocks; v reads k's columns.
  # The *_col methods return where a head's block starts.

  @property
  def kv_col(self) -> int:
    return (2 if self.gate else 1) * self.num_q_heads * self.head_dim

  @property
  def num_cols(self) -> int:  # N.
    num_kv = (1 if self.v_from_k else 2) * self.num_kv_heads
    return self.kv_col + num_kv * self.head_dim

  def q_col(self, q_head: int) -> int:
    return (2 * q_head if self.gate else q_head) * self.head_dim

  def gate_col(self, q_head: int) -> int:
    return (2 * q_head + 1) * self.head_dim

  def k_col(self, kv_head: int) -> int:
    return self.kv_col + kv_head * self.head_dim

  def v_col(self, kv_head: int) -> int:
    if self.v_from_k:
      return self.k_col(kv_head)
    return self.kv_col + (self.num_kv_heads + kv_head) * self.head_dim


@dataclasses.dataclass(frozen=True)
class GdnConfig:
  """Qwen3-Next / Qwen3.5 Gated DeltaNet prologue shapes for one chip.

  Head counts are after TP. in_proj_qkvz gives mixed_qkv ([q | k | v]) and z;
  in_proj_ba gives b and a.
  """

  num_k_heads: int  # Heads of q and k.
  num_v_heads: int  # Hv.
  head_dim: int = 128  # D, shared by key and value heads (128 in both models).
  fp8: bool = True
  act_quant_block: int = 1024  # As AttentionConfig.act_quant_block.
  input_norm: Norm = Norm.RMS_ONE_PLUS_W
  residual: bool = True
  eps: float = 1e-6
  # Precision of the input norm's x * w. vLLM uses f32; bf16 is the default
  # because it is ~3-5% faster at T >= 1024.
  norm_dtype: jax.typing.DTypeLike = jnp.bfloat16

  @property
  def has_input_norm(self) -> bool:
    return self.input_norm != Norm.NONE

  # W (in_proj_qkvz) is [q | k | v | z] (vLLM gqa_interleaved_layout=False):
  # q and k are num_k_heads heads each, v and z num_v_heads heads each.
  # side_w (in_proj_ba) is [b | a], num_v_heads columns each.

  @property
  def key_dim(self) -> int:
    return self.num_k_heads * self.head_dim

  @property
  def value_dim(self) -> int:
    return self.num_v_heads * self.head_dim

  @property
  def qkv_cols(self) -> int:  # mixed_qkv width.
    return 2 * self.key_dim + self.value_dim

  @property
  def z_col(self) -> int:
    return self.qkv_cols

  @property
  def num_cols(self) -> int:  # N.
    return self.qkv_cols + self.value_dim

  @property
  def side_cols(self) -> int:
    return 2 * self.num_v_heads


# Weights as the kernels take them. Each kernel streams its large projection
# weights from HBM itself; the small ones are grouped in a Pinned* tuple, which
# XLA places in VMEM before the kernel starts.


class PinnedGqaWeights(NamedTuple, Generic[Leaf]):
  """fused_qkv's small weights, read in place from VMEM."""

  w_scale: Leaf | None  # (1, N) f32, fp8 only.
  bias: Leaf | None  # (1, N).
  q_norm_w: Leaf | None  # (1, D).
  k_norm_w: Leaf | None  # (1, D).
  # fp8: (K / block, 1, block) u32, the bf16 weight in both halves of each u32
  # so one stride-0 load broadcasts it over a bf16 vreg. bf16: (1, K) as loaded.
  input_norm_w: Leaf | None


class GqaWeights(NamedTuple):
  """fused_qkv's weight arguments, in its argument order."""

  w: jax.Array  # (K, N) fp8 or bf16. Streamed from HBM by the kernel.
  w_scale: jax.Array | None
  bias: jax.Array | None
  q_norm_w: jax.Array | None
  k_norm_w: jax.Array | None
  input_norm_w: jax.Array | None  # From gqa.weights.prepare_input_norm_w.

  @property
  def pinned(self) -> PinnedGqaWeights[jax.Array]:
    return PinnedGqaWeights(
        self.w_scale, self.bias, self.q_norm_w, self.k_norm_w, self.input_norm_w
    )


class PinnedGdnWeights(NamedTuple, Generic[Leaf]):
  """fused_gdn's small weights, read in place from VMEM."""

  w_scale: Leaf | None  # (1, N) f32; None only in layout tests.
  # Input norm weight, pre-chopped into quant blocks: (K / block, 1, block)
  # u32 packed bf16, or f32 if norm_dtype is f32.
  norm_w: Leaf | None


class GdnWeights(NamedTuple):
  """fused_gdn's weight arguments, in its argument order."""

  w: jax.Array  # in_proj_qkvz (K, N) fp8. Streamed from HBM by the kernel.
  w_scale: jax.Array | None
  # in_proj_ba from gdn.weights.prepare_weights. Streamed from HBM by the
  # kernel.
  ba_w: jax.Array
  norm_w: jax.Array | None  # From gdn.weights.prepare_weights.

  @property
  def pinned(self) -> PinnedGdnWeights[jax.Array]:
    return PinnedGdnWeights(self.w_scale, self.norm_w)


class PinnedOutProjWeights(NamedTuple, Generic[Leaf]):
  """fused_out_proj's small weights, read in place from VMEM."""

  w_scale: Leaf | None  # (1, K) f32, fp8 only.
  bias: Leaf | None  # (1, K).


class OutProjWeights(NamedTuple):
  """fused_out_proj's weight arguments (o_proj, as loaded), in its order."""

  w: jax.Array  # (Hq*D, K) fp8 or bf16.
  w_scale: jax.Array | None = None  # (1, K) f32, fp8 only.
  bias: jax.Array | None = None  # (1, K).
