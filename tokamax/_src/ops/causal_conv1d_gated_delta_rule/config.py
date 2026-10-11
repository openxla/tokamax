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

import dataclasses
import enum
from typing import Any, ClassVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


class AttentionMode(enum.StrEnum):
  # Gated Delta Net: one scalar gate per value head.
  GDN = enum.auto()
  # Kimi Delta Attention: the gate is per-channel, so it carries
  # `kq_head_dim` values per value head instead of one.
  KDA = enum.auto()


class GDNMode(enum.StrEnum):
  BATCHED = enum.auto()
  PER_SEQ = enum.auto()

  def get_seq_tile_size(self, tile_size: int) -> int:
    if self == GDNMode.BATCHED:
      return tile_size
    return 1

  def get_chunk_size(self, tile_size: int) -> int:
    if self == GDNMode.BATCHED:
      return 1
    return tile_size


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
  act_in: jnp.dtype
  act_out: jnp.dtype
  compute: jnp.dtype
  recurrent_state: jnp.dtype
  conv_state: jnp.dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
  mode: GDNMode
  dtypes: Dtypes
  batch_size: int
  dim_size: int
  kernel_size: int
  tile_size: int
  num_kq_heads: int
  num_v_heads: int
  kq_head_dim: int
  v_head_dim: int
  attention_mode: AttentionMode = AttentionMode.GDN
  # KDA only, and must be negative. Selects the bounded gate form over the
  # unbounded one; see `fused_conv1d_gdn` for the two and their validation.
  gate_lower_bound: float | None = None
  num_buffers: int = 2
  # Max tokens per speculative verify window (= num_speculative_tokens + 1),
  # which is also the number of state checkpoints kept per sequence. The
  # kernel reads a sequence's initial state from
  # `state_indices[s] + read_offset[s]` and writes one checkpoint per window
  # position to `state_indices[s] + t`, which is how rejected draft tokens
  # are rolled back (by checkpoint selection). It is 1 without speculative
  # decoding, where only the state after the last real token is kept, so
  # shapes and loops can be sized off it unconditionally and the extra axis
  # / iterations fold away in the non-speculative paths.
  window_size: int = 1

  @property
  def chunk_size(self) -> int:
    if self.mode == GDNMode.PER_SEQ:
      return self.tile_size
    # One tile per sequence, holding its whole verify window. BATCHED is
    # the single-token case of that (window_size == 1).
    return self.window_size

  @property
  def seq_tile_size(self) -> int:
    if self.mode == GDNMode.PER_SEQ:
      return 1
    return self.tile_size

  @property
  def prev_kernel_size(self) -> int:
    return self.kernel_size - 1

  @property
  def use_recurrent(self) -> bool:
    """Whether GDN runs the token-recurrent scan instead of the chunked one.

    Keeping more than one state checkpoint mandates the recurrent scan,
    since the chunked path only ever produces the final state.
    """
    return self.chunk_size == 1 or self.window_size > 1

  @property
  def v_dim_size(self) -> int:
    return self.num_v_heads * self.v_head_dim

  @property
  def kq_dim_size(self) -> int:
    return self.num_kq_heads * self.kq_head_dim

  @property
  def v_per_kq_head(self) -> int:
    return self.num_v_heads // self.num_kq_heads

  @property
  def aligned_num_v_heads(self) -> int:
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes

  @property
  def is_kda(self) -> bool:
    return self.attention_mode == AttentionMode.KDA

  @property
  def gate_dim(self) -> int:
    """Width of the gate activation `a`, in elements per token."""
    if self.is_kda:
      return self.num_v_heads * self.kq_head_dim
    return self.num_v_heads

  @property
  def aligned_gate_dim(self) -> int:
    num_lanes = pltpu.get_tpu_info().num_lanes
    return pl.cdiv(self.gate_dim, num_lanes) * num_lanes

  @property
  def triangular_block_size(self) -> int:
    """Diagonal block size B for `invert_triangular_matrix`."""
    # Within-block forward substitution issues head_groups * B / 2 broadcasts
    # per token. At 1 head group (<= 8 heads), B = 16 is optimal.
    # - GDN issues O(1 / B) between-block solver matmuls per token, so scale
    #   B inversely with head_groups (head_groups * B <= 16).
    # - KDA also tiles Aqk and L into O(1 / B^2) sub-block matmuls per token,
    #   so scale B^2 inversely with head_groups (head_groups * B**2 <= 16**2).
    head_groups = pl.cdiv(self.num_v_heads, 8)
    # KDA
    if self.is_kda:
      for b in (16, 8, 4):
        if head_groups * b * b <= 256:
          return b
      return 2
    # GDN
    return max(2, min(16, 16 // head_groups))

  def get_kernel_name(self) -> str:
    # Windows of different sizes compile to different kernels; keep them
    # distinguishable in profiles.
    suffix = f"_w{self.window_size}" if self.window_size > 1 else ""
    if self.is_kda and self.gate_lower_bound is not None:
      suffix += "_bounded"
    return (
        f"fused_conv1d_{self.attention_mode.value}_{self.mode.value}"
        f"_b{self.seq_tile_size}_c{self.chunk_size}{suffix}"
    )

  def get_metadata(self) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(self)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
      key = jax.tree_util.keystr(path, simple=True, separator=".")
      if not isinstance(val, str | int | float):
        val = str(val)
      ret[key] = val
    return ret

  def get_out_shape(self) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        (self.batch_size, self.num_v_heads, self.v_head_dim),
        self.dtypes.act_out,
    )

  # Fraction of VMEM the kernel may use. A multi-token window holds one
  # state checkpoint per position in VMEM, so for large-head models the
  # default 0.80 budget is not enough; those get a higher limit and the
  # wrapper sizes the tile against the same factor.
  DEFAULT_VMEM_FRACTION: ClassVar[float] = 0.80
  WINDOWED_VMEM_FRACTION: ClassVar[float] = 0.90
  # Cap decode tiles at 16 sequences: larger tiles increase per-slot state DMA
  # descriptors and register spills without reducing per-slot compute.
  DECODE_TILE_SIZES: ClassVar[tuple[int, ...]] = (16, 8, 4, 2, 1)
  MIXED_TILE_SIZES: ClassVar[tuple[int, ...]] = (128, 64, 32, 16, 8, 4, 2, 1)

  @classmethod
  def get_vmem_limit_bytes(cls, window_size: int = 1) -> int:
    """Returns the VMEM limit for a kernel keeping window_size checkpoints."""
    fraction = (
        cls.WINDOWED_VMEM_FRACTION
        if window_size > 1
        else cls.DEFAULT_VMEM_FRACTION
    )
    return int(fraction * pltpu.get_tpu_info().vmem_capacity_bytes)

  def get_scratch_shape_dict(self) -> dict[str, Any]:
    conv_shape = (self.seq_tile_size, self.prev_kernel_size, 1, self.dim_size)
    recurrent_shape = (
        self.seq_tile_size,
        self.num_v_heads,
        self.kq_head_dim,
        self.v_head_dim,
    )

    carry_conv_scratch = carry_recurrent_scratch = None
    # NOTE: In batched mode 1 seq = 1 tile, so inter-tile carry is not
    # needed.
    if self.mode == GDNMode.PER_SEQ:
      carry_conv_scratch = pltpu.VMEM(conv_shape, jnp.float32)
      carry_recurrent_scratch = pltpu.VMEM(recurrent_shape, jnp.float32)

    return dict(
        carry_conv_scratch_ref=carry_conv_scratch,
        carry_recurrent_scratch_ref=carry_recurrent_scratch,
    )
