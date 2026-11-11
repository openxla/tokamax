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


@dataclasses.dataclass(frozen=True)
class MhcKernelConfig:
  """Compiler and tiling configuration for mHC Pallas kernels."""

  block_size: int | None = None
  fwd_pipeline_blocks: int = 2
  bwd_block_size: int | None = None
  vmem_limit_bytes: int | None = None
  rms_epsilon: float = 1e-5
  pre_mapping_epsilon: float = 1e-6
  interpret: bool = False


def resolve_config(config: MhcKernelConfig) -> MhcKernelConfig:
  """Resolves `None` tiling and VMEM parameters using `get_tpu_info()`."""
  tpu_info = get_tpu_info()
  block_size = (
      tpu_info.num_lanes if config.block_size is None else config.block_size
  )
  bwd_block_size = (
      2 * tpu_info.num_lanes
      if config.bwd_block_size is None
      else config.bwd_block_size
  )
  vmem_limit_bytes = (
      tpu_info.vmem_capacity_bytes
      if config.vmem_limit_bytes is None
      else config.vmem_limit_bytes
  )
  return dataclasses.replace(
      config,
      block_size=block_size,
      bwd_block_size=bwd_block_size,
      vmem_limit_bytes=vmem_limit_bytes,
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
