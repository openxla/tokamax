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
"""Shared data structures, tiling specs, and cost models for mHC-lite Pallas kernels."""

import contextlib
import dataclasses
import functools
import math
from typing import Any
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


@contextlib.contextmanager
def tpu_mesh_context():
  """Ensures a valid TPU AbstractMesh is available when running on CPU / interpret mode."""
  needs_abstract_mesh = False
  try:
    if not pltpu.is_tpu_device():
      needs_abstract_mesh = True
  except (RuntimeError, ValueError, AttributeError, TypeError):
    needs_abstract_mesh = True

  if needs_abstract_mesh:
    abstract_mesh = jax.sharding.AbstractMesh(
        (),
        (),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      yield
  else:
    yield


def get_tpu_info() -> pltpu.TpuInfo:
  """Returns the active TPU hardware info, using an abstract TPU mesh on CPU."""
  with tpu_mesh_context():
    return pltpu.get_tpu_info()


HBM_SPEC = pl.BlockSpec(memory_space=pltpu.HBM)


class UnsupportedInputError(ValueError):
  """Known Mosaic shape, dtype, or tiling restriction."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MHCContext:
  """Opaque token-local context passed from `pre` to `post`."""

  x: jax.Array
  h_post: jax.Array
  residual: jax.Array
  implementation: str = dataclasses.field(metadata={"static": True})


FWD_TILING_FIELDS = (
    "block_size",
    "fwd_pipeline_blocks",
    "fwd_sub_tokens",
    "fwd_strip_tokens",
    "fwd_band_width",
)
BWD_TILING_FIELDS = ("bwd_block_size", "bwd_num_chunks", "bwd_sub_tokens")
_POSITIVE_CONFIG_FIELDS = (
    "block_size",
    "fwd_pipeline_blocks",
    "fwd_sub_tokens",
    "fwd_strip_tokens",
    "fwd_band_width",
    "bwd_block_size",
    "bwd_num_chunks",
    "bwd_sub_tokens",
    "vmem_limit_bytes",
)


@dataclasses.dataclass(frozen=True)
class MhcKernelConfig:
  """Compiler and tiling configuration for mHC Pallas kernels."""

  block_size: int | None = None
  fwd_pipeline_blocks: int | None = None
  fwd_sub_tokens: int | None = None
  fwd_strip_tokens: int | None = None
  fwd_band_width: int | None = None
  bwd_block_size: int | None = None
  bwd_num_chunks: int | None = None
  bwd_sub_tokens: int | None = None
  vmem_limit_bytes: int | None = None
  rms_epsilon: float = 1e-5
  pre_mapping_epsilon: float = 1e-6
  interpret: bool = False

  def has_fwd_tiling_overrides(self) -> bool:
    """Returns True if any forward tiling field is explicitly set."""
    return any(getattr(self, name) is not None for name in FWD_TILING_FIELDS)

  def has_bwd_tiling_overrides(self) -> bool:
    """Returns True if any backward tiling field is explicitly set."""
    return any(getattr(self, name) is not None for name in BWD_TILING_FIELDS)

  def has_tiling_overrides(self) -> bool:
    """Returns True if any explicit tiling or VMEM override is set."""
    return (
        self.has_fwd_tiling_overrides()
        or self.has_bwd_tiling_overrides()
        or self.vmem_limit_bytes is not None
    )


def _default(val: int | None, fallback: int) -> int:
  """Returns `fallback` when `val` is `None`, otherwise `val`."""
  return fallback if val is None else val


def chunk_tokens(tokens: int, block_size: int, fwd_pipeline_blocks: int) -> int:
  """Returns `block_size * fwd_pipeline_blocks` when dividing `tokens`."""
  chunk = block_size * fwd_pipeline_blocks
  if fwd_pipeline_blocks > 1 and tokens % chunk == 0:
    return chunk
  return block_size


def validate_config_fields(
    config: object,
    num_lanes: int,
    num_sublanes: int,
    *,
    check_fwd: bool = True,
    check_bwd: bool = True,
) -> None:
  """Validates positive and alignment constraints on config fields."""
  for name in _POSITIVE_CONFIG_FIELDS:
    if (val := getattr(config, name, None)) is not None and val < 1:
      raise UnsupportedInputError(
          f"Expected positive {name}; got {val}."
      )
  aligned_fields: list[tuple[str, int]] = []
  if check_fwd:
    aligned_fields.extend((
        ("fwd_sub_tokens", num_sublanes),
        ("fwd_strip_tokens", num_sublanes),
        ("fwd_band_width", num_lanes),
    ))
  if check_bwd:
    aligned_fields.append(("bwd_sub_tokens", num_sublanes))
  for name, align in aligned_fields:
    if (val := getattr(config, name, None)) is not None and val % align:
      raise UnsupportedInputError(
          f"Expected {name} to be a positive multiple of {align}; got {val}."
      )


def token_block_candidates(
    tokens: int,
    num_lanes: int,
    num_sublanes: int,
    multipliers: int | tuple[int, ...] = (1, 2, 3, 4, 6, 8),
) -> list[int]:
  """Returns lane-aligned or sublane-aligned block divisors of `tokens`."""
  if tokens <= 0:
    return [num_lanes]
  mults = (
      range(1, multipliers + 1)
      if isinstance(multipliers, int)
      else multipliers
  )
  lane_divs = [
      mult * num_lanes for mult in mults if tokens % (mult * num_lanes) == 0
  ]
  if lane_divs:
    return lane_divs
  sublane_divs = [
      mult * num_sublanes
      for mult in range(2 * num_lanes // num_sublanes, 0, -1)
      if tokens % (mult * num_sublanes) == 0
  ]
  return sublane_divs or [num_lanes]


def resolve_config(
    config: MhcKernelConfig,
    *,
    tokens: int | None = None,
    streams: int = 4,
    embedding: int = 512,
    num_permutations: int = 24,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    check_fwd: bool = True,
    check_bwd: bool = True,
) -> MhcKernelConfig:
  """Resolves `None` tiling and VMEM fields with safe hardware defaults."""
  del streams, num_permutations, dtype
  tpu_info = get_tpu_info()
  num_lanes, num_sublanes = tpu_info.num_lanes, tpu_info.num_sublanes
  validate_config_fields(
      config, num_lanes, num_sublanes, check_fwd=check_fwd, check_bwd=check_bwd
  )
  vmem_limit = _default(config.vmem_limit_bytes, tpu_info.vmem_capacity_bytes)
  wide_or_interpret = config.interpret or embedding > 1024
  if tokens is None or tokens <= 0:
    block_size = _default(config.block_size, num_lanes)
    fwd_pipe = _default(config.fwd_pipeline_blocks, 2)
    chunk = block_size * fwd_pipe
    bwd_block_size = _default(config.bwd_block_size, 2 * num_lanes)
    num_bwd_blocks = 4
  else:
    cands = token_block_candidates(tokens, num_lanes, num_sublanes, (1, 2))
    default_blk = min(cands) if wide_or_interpret else max(cands)
    block_size = _default(config.block_size, default_blk)
    default_pipe = (
        2
        if not wide_or_interpret and tokens % (2 * block_size) == 0
        else 1
    )
    fwd_pipe = _default(config.fwd_pipeline_blocks, default_pipe)
    chunk = chunk_tokens(tokens, block_size, fwd_pipe)
    bwd_block_size = _default(config.bwd_block_size, default_blk)
    num_bwd_blocks = max(1, tokens // max(1, bwd_block_size))

  max_strip_elems = 12 * num_lanes * 512
  if config.interpret or chunk * embedding > max_strip_elems:
    default_strip = math.gcd(chunk, num_lanes // 2) or num_sublanes
  else:
    default_strip = chunk
  default_fwd_sub = math.gcd(chunk, 2 * num_sublanes) or num_sublanes
  default_band = math.gcd(embedding, 2 * num_lanes) or num_lanes
  default_bwd_sub = math.gcd(bwd_block_size, 2 * num_sublanes) or num_sublanes
  default_bwd_chunks = math.gcd(num_bwd_blocks, 4)
  return dataclasses.replace(
      config,
      block_size=block_size,
      fwd_pipeline_blocks=fwd_pipe,
      fwd_sub_tokens=_default(config.fwd_sub_tokens, default_fwd_sub),
      fwd_strip_tokens=_default(config.fwd_strip_tokens, default_strip),
      fwd_band_width=_default(config.fwd_band_width, default_band),
      bwd_block_size=bwd_block_size,
      bwd_num_chunks=_default(config.bwd_num_chunks, default_bwd_chunks),
      bwd_sub_tokens=_default(config.bwd_sub_tokens, default_bwd_sub),
      vmem_limit_bytes=vmem_limit,
  )


def prepare_kernel_config(
    config: Any,
    tokens: int,
    streams: int,
    embedding: int,
    num_permutations: int,
    dtype: jax.typing.DTypeLike,
    *,
    is_bwd: bool = False,
    rms_epsilon: float | None = None,
    pre_mapping_epsilon: float | None = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[Any, float, float, int, bool]:
  """Validates `config` and resolves concrete kernel execution parameters."""
  check_fwd, check_bwd = (False, True) if is_bwd else (True, False)
  if isinstance(config, MhcKernelConfig):
    config = resolve_config(
        config,
        tokens=tokens,
        streams=streams,
        embedding=embedding,
        num_permutations=num_permutations,
        dtype=dtype,
        check_fwd=check_fwd,
        check_bwd=check_bwd,
    )
  else:
    tpu_info = get_tpu_info()
    validate_config_fields(
        config,
        tpu_info.num_lanes,
        tpu_info.num_sublanes,
        check_fwd=check_fwd,
        check_bwd=check_bwd,
    )
  if rms_epsilon is None:
    rms_epsilon = getattr(config, "rms_epsilon", 1e-5)
  if pre_mapping_epsilon is None:
    pre_mapping_epsilon = getattr(config, "pre_mapping_epsilon", 1e-6)
  if vmem_limit_bytes is None:
    vmem_limit_bytes = (
        getattr(config, "vmem_limit_bytes", None)
        or get_tpu_info().vmem_capacity_bytes
    )
  resolved_interpret = interpret or getattr(config, "interpret", False)
  return (
      config,
      rms_epsilon,
      pre_mapping_epsilon,
      vmem_limit_bytes,
      resolved_interpret,
  )


def pre_cost(
    tokens: int,
    streams: int,
    embedding: int,
    num_permutations: int = 24,
    *,
    is_bwd: bool = False,
) -> pl.CostEstimate:
  """Cost estimate for `pre` forward or backward (`T=tokens, k=streams, d=emb, P=perms, C=2k+P`).

  Returns:
    pl.CostEstimate with:
    - FLOPs (fwd): `2*T*(k*d)*C` (x @ phi) + `2*T*P*k^2` (w_res @ P) + `2*T*k*d`
      (h_pre * x); backward scales by 2x (recompute + VJP GEMMs and
      d_x/d_h_pre).
    - Transcendentals: `T*C` (sigmoid + softmax).
    - Bytes: `2*T*k*d` (read bf16 x; 3x in bwd for x, d_x_acc, write d_x) +
      `4*(k*d)*C` (read f32 phi; 2x in bwd for read phi, write d_phi).
  """
  flat = streams * embedding
  phi_cols = 2 * streams + num_permutations
  scale = 2 if is_bwd else 1
  x_reads = 3 if is_bwd else 1
  return pl.CostEstimate(
      flops=scale
      * 2
      * tokens
      * (flat * (phi_cols + 1) + num_permutations * streams * streams),
      transcendentals=tokens * phi_cols,
      bytes_accessed=x_reads * 2 * tokens * flat + scale * 4 * flat * phi_cols,
  )


def post_cost(
    tokens: int,
    streams: int,
    embedding: int,
    *,
    is_bwd: bool = False,
) -> pl.CostEstimate:
  """Cost estimate for `post` forward or backward (`T=tokens, k=streams, d=emb`).

  Returns:
    pl.CostEstimate with:
    - FLOPs (fwd): `2*T*k^2*d` (residual @ x) + `2*T*k*d` (h_post * layer_out +
      res_mix); backward scales by 2x (VJP of residual^T @ d_out, d_layer_out,
      d_h_post, d_res).
    - Transcendentals: `0`.
    - Bytes: `4*T*k*d` (read/write bf16 x, out; `6*T*k*d` in bwd for read x,
      d_out, write d_x) + `4*T*d` (read layer_out + gates; `8*T*d` in bwd for
      read/write layer_out, d_layer_out).
  """
  scale = 2 if is_bwd else 1
  x_reads = 3 if is_bwd else 2
  return pl.CostEstimate(
      flops=scale * 2 * tokens * streams * (streams + 1) * embedding,
      transcendentals=0,
      bytes_accessed=x_reads * 2 * tokens * streams * embedding
      + scale * 4 * tokens * embedding,
  )


def post_and_pre_cost(
    tokens: int,
    streams: int,
    embedding: int,
    num_permutations: int = 24,
    *,
    is_bwd: bool = False,
) -> pl.CostEstimate:
  """Cost estimate for fused `post_and_pre` forward or backward (`T=tokens, k=streams, d=emb`).

  Returns:
    pl.CostEstimate summing `post_cost` and `pre_cost` while subtracting the
    avoided HBM roundtrip (`2*T*(k-1)*d` in fwd for in-VMEM `x_next` reuse, or
    `4*T*k*d` in bwd for fused `d_x_next` handoff).
  """
  post = post_cost(tokens, streams, embedding, is_bwd=is_bwd)
  pre = pre_cost(tokens, streams, embedding, num_permutations, is_bwd=is_bwd)
  saved_bytes = (
      4 * tokens * streams * embedding
      if is_bwd
      else 2 * tokens * (streams - 1) * embedding
  )
  return pl.CostEstimate(
      flops=post.flops + pre.flops,
      transcendentals=post.transcendentals + pre.transcendentals,
      bytes_accessed=post.bytes_accessed + pre.bytes_accessed - saved_bytes,
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MhcWeights:
  """Structured layer weights container for mHC-lite."""

  norm_scale: jax.Array
  w_pre: jax.Array
  b_pre: jax.Array
  alpha_pre: jax.Array
  w_post: jax.Array
  b_post: jax.Array
  alpha_post: jax.Array
  w_res: jax.Array
  b_res: jax.Array
  alpha_res: jax.Array

  def fold_norm_scale(self) -> jax.Array:
    return fold_norm_scale(self.norm_scale, self.w_pre, self.w_post, self.w_res)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HcWeights:
  """Structured layer weights container for static Hyper-Connections (HC)."""

  w_pre: jax.Array
  w_post: jax.Array
  w_res: jax.Array


def _token_index_map(num_trailing_dims: int, i):
  """Index map tiling the leading token dimension."""
  return (i,) + (0,) * num_trailing_dims


def token_block_spec(shape: tuple[int, ...], block_size: int) -> pl.BlockSpec:
  """Returns a BlockSpec tiling the leading token dimension."""
  block_shape = (block_size,) + shape[1:]
  return pl.BlockSpec(
      block_shape, functools.partial(_token_index_map, len(shape) - 1)
  )


def fold_norm_scale(norm_scale, w_pre, w_post, w_res) -> jax.Array:
  """Folds the RMSNorm channel scale into the three projection matrices."""
  with jax.named_scope("fold_norm_scale"):
    weights = jnp.concatenate((w_pre, w_post, w_res), axis=-1).astype(
        jnp.float32
    )
    return norm_scale.astype(jnp.float32)[:, None] * weights


def validate_token_block_size(
    tokens: int, block_size: int, *, name: str = "block_size"
) -> None:
  """Validates a token-axis Pallas block size."""
  if tokens <= 0:
    raise UnsupportedInputError(f"Expected positive token count; got {tokens}.")
  num_sublanes = get_tpu_info().num_sublanes
  if block_size < num_sublanes or block_size % num_sublanes:
    raise UnsupportedInputError(
        f"Expected {name} to be a positive multiple of {num_sublanes}; got"
        f" {block_size}."
    )
  if tokens % block_size:
    raise UnsupportedInputError(
        f"The per-device token count ({tokens}) must be divisible by {name}"
        f" ({block_size})."
    )


def validate_inputs(
    x: jax.Array,
    block_size: int | None = None,
    permutations_shape: tuple[int, ...] | None = None,
    *,
    block_size_name: str = "block_size",
) -> None:
  """Validates the shape, dtype, and forward token block constraints."""
  if x.dtype != jnp.bfloat16:
    raise UnsupportedInputError(
        f"The mHC Pallas kernel requires bfloat16 activations; got {x.dtype}."
    )
  if x.ndim == 2:
    tokens, flattened_size = x.shape
    streams = 4
    if flattened_size % streams:
      raise UnsupportedInputError(
          "Expected 2D x to have trailing dimension divisible by 4; got"
          f" {x.shape}."
      )
    embedding = flattened_size // streams
  elif x.ndim == 4:
    batch, sequence, streams, embedding = x.shape
    tokens = batch * sequence
  else:
    raise UnsupportedInputError(
        "Expected x to have shape (batch, sequence, streams, embedding) or"
        f" (tokens, streams * embedding); got {x.shape}."
    )
  if streams != 4 or (
      permutations_shape is not None and permutations_shape != (24, 4, 4)
  ):
    raise UnsupportedInputError(
        "The optimized mHC Pallas kernel currently supports mHC-lite with"
        f" expansion rate 4 only; got x.shape={x.shape} and"
        f" permutations.shape={permutations_shape}."
    )
  tpu_info = get_tpu_info()
  if embedding % tpu_info.num_lanes:
    raise UnsupportedInputError(
        f"Expected embedding dim to be a multiple of {tpu_info.num_lanes}; got"
        f" {embedding}."
    )
  resolved_block_size = tpu_info.num_lanes if block_size is None else block_size
  validate_token_block_size(tokens, resolved_block_size, name=block_size_name)
