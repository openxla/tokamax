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
"""Pallas TPU kernels for the DeepSeek-V4 RoPE.

Ported from vllm-torchtpu `src/vllm_torchtpu/kernels/deepseek_v4/rope.py`.
"""

import dataclasses
import functools
from typing import cast

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Native lane count; the roped channels are the tail of the last lane block.
LANE = 128

# Upstream's limit. Mosaic caps it at the VMEM capacity (64 MiB on TPU7x).
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def largest_divisor(x: int, cap: int) -> int:
  """Largest divisor of ``x`` that is <= ``cap``."""
  for candidate in range(min(x, cap), 0, -1):
    if x % candidate == 0:
      return candidate
  return 1


def dtype_max(dtype: jax.typing.DTypeLike) -> float:
  """The largest finite value of ``dtype``."""
  dtype = jnp.dtype(dtype)
  if jnp.issubdtype(dtype, jnp.floating):
    return float(jnp.finfo(dtype).max)
  return float(jnp.iinfo(dtype).max)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CosSinRef(pltpu.BufferedRef):
  """Per-token gather of ``cos_sin_cache`` rows into a ``(tile_n, rotary_dim)``"""

  tile_n: int = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pyrefly: ignore[bad-override]
      cls,
      *,
      spec: pl.BlockSpec,
      dtype: jnp.dtype,
      buffer_count: int,
      tile_n: int,
  ) -> "CosSinRef":
    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype,
        buffer_type=pltpu.BufferType.INPUT,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=False,
    )
    return cls(
        tile_n=tile_n,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ) -> None:
    cos_sin_cache_ref, positions_ref = src_ref
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]  # pyrefly: ignore[missing-attribute]
    pid = grid_indices[0]

    # (tile_n, rotary_dim)
    dest_ref = self.window_ref.at[slot]  # pyrefly: ignore[missing-attribute]
    width = self.window_ref.shape[-1]  # pyrefly: ignore[missing-attribute]

    @pl.loop(0, self.tile_n, unroll=True)
    def _(i):
      position = positions_ref[pid * self.tile_n + i]
      src = cos_sin_cache_ref.at[position, pl.ds(0, width)]
      pltpu.make_async_copy(src, dest_ref.at[i, :], sem).start()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ) -> None:
    del src_ref, grid_indices
    slot = self.current_wait_in_slot
    vmem_ref = self.window_ref.at[slot]  # pyrefly: ignore[missing-attribute]
    pltpu.make_async_copy(
        vmem_ref,
        vmem_ref,
        self.sem_recvs.at[slot],  # pyrefly: ignore[missing-attribute]
    ).wait()


def _lane_gather(operand: jax.Array, indices: jax.Array) -> jax.Array:
  """``out[b..., i] = operand[b..., indices[i]]`` over the last axis.

  The leading axes come along as batch dimensions. ``indices`` is a constant
  1-D array.
  """
  rank = operand.ndim
  lanes = indices.shape[0]
  coords = jnp.broadcast_to(indices, (*operand.shape[:-1], lanes))[..., None]
  batching = tuple(range(rank - 1))
  dimension_numbers = jax.lax.GatherDimensionNumbers(
      offset_dims=(),
      collapsed_slice_dims=(rank - 1,),
      start_index_map=(rank - 1,),
      operand_batching_dims=batching,
      start_indices_batching_dims=batching,
  )
  return jax.lax.gather(
      operand,
      coords,
      dimension_numbers=dimension_numbers,
      slice_sizes=(1,) * rank,
      unique_indices=False,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )


def _cos_sin_lanes(
    cos_sin: jax.Array, *, rotary_dim: int, inverse: bool
) -> tuple[jax.Array, jax.Array]:
  """The reference's ``np.split`` + ``np.repeat(..., 2)``, over a whole block.

  ``cos_sin`` is one packed ``[cos | sin]`` row per token. The reference splits
  it in half, repeats each half so both channels of a pair share a frequency,
  rotates ``x[..., -rotary_dim:]`` and concatenates the NoPE part back on.

  A kernel block is ``LANE`` lanes wide and the roped channels are its tail, so
  rather than slicing them out we widen cos/sin to the full block and hand the
  NoPE lanes the identity rotation -- multiplying by ``cos = 1``, ``sin = 0``
  is exactly the reference's pass-through concatenation::

      lane      0 ... 63 | 64  65  66  67 ...   (LANE 128, rotary_dim 64)
      channel   -- NoPE -|  0   1   2   3
      cos       1 ...  1 | c0  c0  c1  c1       <- np.repeat(cos, 2)
      sin       0 ...  0 | s0  s0  s1  s1
  """
  half = rotary_dim // 2
  channel = jnp.arange(LANE) - (LANE - rotary_dim)  # < 0 on the NoPE lanes
  is_rot = channel >= 0
  # ``np.repeat(..., 2)``: channels 2k and 2k+1 both read frequency k. Clamped
  # to keep the gather in bounds on the NoPE lanes, replaced just below.
  freq = jnp.maximum(channel, 0) // 2

  cos = _lane_gather(cos_sin, freq)  # np.split(...)[0], repeated
  sin = _lane_gather(cos_sin, half + freq)  # np.split(...)[1], repeated
  if inverse:
    sin = -sin
  return jnp.where(is_rot, cos, 1.0), jnp.where(is_rot, sin, 0.0)


def _rotate_gptj(x: jax.Array) -> jax.Array:
  """The reference's ``rotated``: ``[-x1, x0, -x3, x2, ...]`` along the lanes.

  ``np.stack([-x[..., 1::2], x[..., 0::2]], -1).reshape(...)`` says, lane by
  lane, "take the partner lane ``i ^ 1``, negated on the even ones" -- a
  constant-index lane shuffle and a sign, no reshape. Lane parity may be read
  for channel parity because ``LANE - rotary_dim`` is even.
  """
  lane = jnp.arange(LANE)
  sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
  return _lane_gather(x, lane ^ 1) * sign


def _rope_body(
    x_ref: jax.Ref,
    cos_sin_ref: jax.Ref,
    *out_refs: jax.Ref,
    rotary_dim: int,
    inverse: bool,
    out_dtype: jnp.dtype,
    quant_dtype: jnp.dtype | None,
) -> None:
  """Process one ``(tile_n, [num_heads,] LANE)`` chunk.

  Without ``quant_dtype`` the rotation is written back into ``x_ref`` in place.
  With it, the rotated chunk is quantized along the lanes and written to the
  ``(q_ref, scale_ref)`` outputs instead.
  """
  x = x_ref[...].astype(jnp.float32)
  cos, sin = _cos_sin_lanes(
      cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse
  )
  if x.ndim == 3:  # [tokens, heads, lanes]: cos/sin are per token
    cos = cos[:, None, :]
    sin = sin[:, None, :]

  rotated = _rotate_gptj(x)
  y = x * cos + rotated * sin

  if quant_dtype is None:
    x_ref[...] = y.astype(out_dtype)
    return

  q_ref, scale_ref = out_refs
  qmax = dtype_max(quant_dtype)
  # One scale per row, i.e. over the whole ``LANE``-wide channel axis.
  abs_max = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
  scale = abs_max / qmax
  scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
  q_ref[...] = (y * scale_inv).astype(quant_dtype)
  scale_ref[...] = scale.reshape(scale_ref.shape)


def _kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM inputs
    x_hbm_ref: jax.Ref,
    cos_sin_cache_hbm_ref: jax.Ref,
    # HBM outputs: (out,) aliased onto x_hbm_ref, or (q, scales) when quantizing
    *out_hbm_refs: jax.Ref,
    num_tiles: int,
    tile_n: int,
    x_block_shape: tuple[int, ...],
    scale_block_shape: tuple[int, ...],
    lane_block: int,
    rotary_dim: int,
    inverse: bool,
    in_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    cos_sin_dtype: jnp.dtype,
    quant_dtype: jnp.dtype | None,
) -> None:
  def x_index_map(i):
    if len(x_block_shape) == 3:
      return (i, 0, lane_block)
    return (i, lane_block)

  x_spec = pl.BlockSpec(
      block_shape=x_block_shape, memory_space=pltpu.VMEM, index_map=x_index_map
  )

  cos_sin_spec = pl.BlockSpec(
      block_shape=(tile_n, rotary_dim),
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i, 0),
  )
  cos_sin_alloc = CosSinRef.create(
      spec=cos_sin_spec, dtype=cos_sin_dtype, buffer_count=2, tile_n=tile_n
  )

  if quant_dtype is None:
    del x_hbm_ref  # aliased to the output; read/written through the latter.
    (out_hbm_ref,) = out_hbm_refs
    x_ref_args = (out_hbm_ref,)
    out_ref_args = ()
    out_specs = []
    # input_output since we only update the last rope_dim channels in place.
    x_alloc = pltpu.BufferedRef.input_output(x_spec, out_dtype, buffer_count=2)
    out_allocs = ()
  else:
    q_hbm_ref, scale_hbm_ref = out_hbm_refs
    x_ref_args = (x_hbm_ref,)
    out_ref_args = (q_hbm_ref, scale_hbm_ref)
    q_spec = pl.BlockSpec(
        block_shape=x_block_shape,
        memory_space=pltpu.VMEM,
        index_map=x_index_map,
    )
    scale_spec = pl.BlockSpec(
        block_shape=scale_block_shape,
        memory_space=pltpu.VMEM,
        # Same block as ``q``, minus the channel axis it reduces away.
        index_map=lambda i: x_index_map(i)[:-1],
    )
    out_specs = [q_spec, scale_spec]
    x_alloc = pltpu.BufferedRef.input(x_spec, in_dtype, buffer_count=2)
    out_allocs = (
        pltpu.BufferedRef.output(q_spec, quant_dtype, buffer_count=2),
        pltpu.BufferedRef.output(scale_spec, jnp.float32, buffer_count=2),
    )

  pipeline = pltpu.emit_pipeline(
      functools.partial(
          _rope_body,
          rotary_dim=rotary_dim,
          inverse=inverse,
          out_dtype=out_dtype,
          quant_dtype=quant_dtype,
      ),
      grid=(num_tiles,),
      in_specs=[x_spec, cos_sin_spec],
      out_specs=out_specs,
  )

  @pl.with_scoped(allocations=(x_alloc, cos_sin_alloc, *out_allocs))
  def _run(allocations):
    pipeline(
        *x_ref_args,
        (cos_sin_cache_hbm_ref, positions_ref),
        *out_ref_args,
        allocations=allocations,
    )

  _run()


def _qnorm_rope_body(
    x_ref: jax.Ref,
    cos_sin_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    rotary_dim: int,
    eps: float,
    inverse: bool,
    out_dtype: jnp.dtype,
) -> None:
  """Process one ``(tile_n, num_heads, head_dim)`` chunk.

  Per-head RMSNorm (no weight) over the whole ``head_dim``, then the same
  rotation ``_rope_body`` applies.
  """
  x = x_ref[...].astype(jnp.float32)
  rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
  y = x * rms

  # Only the last lane block holds roped channels; the rest of the row passes
  # through with nothing but the norm applied.
  tail = y[..., -LANE:]
  cos, sin = _cos_sin_lanes(
      cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse
  )
  assert tail.ndim == 3
  cos = cos[:, None, :]
  sin = sin[:, None, :]
  roped = tail * cos + _rotate_gptj(tail) * sin

  out_ref[...] = jnp.concatenate([y[..., :-LANE], roped], axis=-1).astype(
      out_dtype
  )


def _qnorm_rope_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM inputs
    x_hbm_ref: jax.Ref,
    cos_sin_cache_hbm_ref: jax.Ref,
    # HBM output, aliased onto x_hbm_ref
    out_hbm_ref: jax.Ref,
    *,
    num_tiles: int,
    tile_n: int,
    x_block_shape: tuple[int, ...],
    rotary_dim: int,
    eps: float,
    inverse: bool,
    in_dtype: jnp.dtype,
    cos_sin_dtype: jnp.dtype,
) -> None:
  x_spec = pl.BlockSpec(
      block_shape=x_block_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i, 0, 0),
  )
  cos_sin_spec = pl.BlockSpec(
      block_shape=(tile_n, rotary_dim),
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i, 0),
  )
  cos_sin_alloc = CosSinRef.create(
      spec=cos_sin_spec, dtype=cos_sin_dtype, buffer_count=2, tile_n=tile_n
  )

  pipeline = pltpu.emit_pipeline(
      functools.partial(
          _qnorm_rope_body,
          rotary_dim=rotary_dim,
          eps=eps,
          inverse=inverse,
          out_dtype=in_dtype,
      ),
      grid=(num_tiles,),
      in_specs=[x_spec, cos_sin_spec],
      out_specs=[x_spec],
  )
  allocations = (
      pltpu.BufferedRef.input(x_spec, in_dtype, buffer_count=2),
      cos_sin_alloc,
      pltpu.BufferedRef.output(x_spec, in_dtype, buffer_count=2),
  )

  @pl.with_scoped(allocations=allocations)
  def _run(allocations):
    pipeline(
        x_hbm_ref,
        (cos_sin_cache_hbm_ref, positions_ref),
        out_hbm_ref,
        allocations=allocations,
    )

  _run()


def _rope_call(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    inverse: bool,
    quant_dtype: jax.typing.DTypeLike | None,
    tile_n: int | None,
    name: str,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Builds the ``pallas_call``; see the public wrappers for the contract."""
  if x.ndim not in (2, 3):
    raise ValueError(f"x must be rank 2 or 3, got {x.shape}")
  assert positions.ndim == 1 and positions.shape[0] == x.shape[0]
  assert cos_sin_cache.ndim == 2

  num_tokens = x.shape[0]
  head_dim = x.shape[-1]
  rows_per_token = x.shape[1] if x.ndim == 3 else 1
  rotary_dim = cos_sin_cache.shape[1]

  assert head_dim % LANE == 0
  assert rotary_dim % 2 == 0 and rotary_dim <= LANE
  if quant_dtype is not None and head_dim != LANE:
    # Only the last lane block is pipelined in, but the scale is a reduction
    # over the whole row, so the two have to coincide (the indexer's head_dim
    # is exactly LANE).
    raise ValueError(
        f"quantization requires head_dim == {LANE}, got {head_dim}"
    )

  if tile_n is None:
    tile_n = largest_divisor(num_tokens, cap=128)
  assert num_tokens % tile_n == 0

  x_block_shape = (
      (tile_n, rows_per_token, LANE) if x.ndim == 3 else (tile_n, LANE)
  )
  # One scale per row: x.shape minus the channel axis.
  scale_block_shape = x_block_shape[:-1]
  kernel = functools.partial(
      _kernel,
      num_tiles=num_tokens // tile_n,
      tile_n=tile_n,
      x_block_shape=x_block_shape,
      scale_block_shape=scale_block_shape,
      lane_block=head_dim // LANE - 1,  # last lane block has the roped channels
      rotary_dim=rotary_dim,
      inverse=inverse,
      in_dtype=x.dtype,
      out_dtype=x.dtype,
      cos_sin_dtype=cos_sin_cache.dtype,
      quant_dtype=quant_dtype,  # pyrefly: ignore[bad-argument-type]
  )

  if quant_dtype is None:
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
    input_output_aliases = {1: 0}  # x (after the scalar prefetch) -> out
  else:
    out_shape = (
        jax.ShapeDtypeStruct(x.shape, quant_dtype),
        jax.ShapeDtypeStruct(x.shape[:-1], jnp.float32),
    )
    out_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    )
    input_output_aliases = {}

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      in_specs=(
          pl.BlockSpec(memory_space=pltpu.HBM),  # x
          pl.BlockSpec(memory_space=pltpu.HBM),  # cos_sin_cache
      ),
      out_specs=out_specs,
  )

  return pl.pallas_call(
      kernel,
      out_shape=out_shape,
      grid_spec=grid_spec,
      input_output_aliases=input_output_aliases,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
          disable_bounds_checks=True,
      ),
      name=name,
  )(positions, x, cos_sin_cache)


@functools.partial(
    jax.jit,
    static_argnames=("inverse", "tile_n", "name"),
    donate_argnames=("x",),
)
def rope(
    x: jax.Array,  # [num_tokens, head_dim] | [num_tokens, num_heads, head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool = False,
    tile_n: int | None = None,
    name: str = "rope",
) -> jax.Array:
  """Applies DeepSeek-V4 RoPE to the trailing ``rotary_dim`` channels of ``x``.

  Args:
    x: ``[num_tokens, head_dim]`` or ``[num_tokens, num_heads, head_dim]``.
    positions: ``[num_tokens]``, the RoPE position of each token.
    cos_sin_cache: ``[max_position, rotary_dim]``, ``[cos | sin]`` packed side
      by side (``rotary_dim // 2`` columns each), as built by
      ``DeepseekV4ScalingRotaryEmbedding``.
    inverse: negate ``sin``, i.e. apply the transposed rotation.
    tile_n: tokens per grid step, a divisor of ``num_tokens``. Defaults to the
      largest divisor of ``num_tokens`` up to 128.
    name: kernel name.

  Returns:
    ``x`` with the trailing ``rotary_dim`` channels rotated, same shape and
    dtype. ``x`` is donated -- the kernel updates the buffer in place.
  """
  return cast(
      jax.Array,
      _rope_call(
          x,
          positions,
          cos_sin_cache,
          inverse=inverse,
          quant_dtype=None,
          tile_n=tile_n,
          name=name,
      ),
  )


@functools.partial(
    jax.jit,
    static_argnames=("eps", "inverse", "tile_n", "name"),
    donate_argnames=("x",),
)
def qnorm_rope(
    x: jax.Array,  # [num_tokens, num_heads, head_dim] | [num_tokens, head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    eps: float = 1e-6,
    inverse: bool = False,
    tile_n: int | None = None,
    name: str = "qnorm_rope",
) -> jax.Array:
  """Per-head RMSNorm (no weight) fused with DeepSeek-V4 RoPE.

  ``x`` is donated -- the output is aliased onto its buffer. ``tile_n`` is the
  number of tokens per grid step, a divisor of ``num_tokens``; it defaults to
  the largest divisor of ``num_tokens`` up to 64.
  """
  assert x.ndim == 3
  assert positions.ndim == 1 and positions.shape[0] == x.shape[0]
  assert cos_sin_cache.ndim == 2

  num_tokens = x.shape[0]
  head_dim = x.shape[-1]
  rows_per_token = x.shape[1]
  rotary_dim = cos_sin_cache.shape[1]
  assert head_dim % LANE == 0
  assert rotary_dim % 2 == 0 and rotary_dim <= LANE
  if tile_n is None:
    tile_n = largest_divisor(num_tokens, cap=64)
  assert num_tokens % tile_n == 0

  x_block_shape = (tile_n, rows_per_token, head_dim)
  kernel = functools.partial(
      _qnorm_rope_kernel,
      num_tiles=num_tokens // tile_n,
      tile_n=tile_n,
      x_block_shape=x_block_shape,
      rotary_dim=rotary_dim,
      eps=eps,
      inverse=inverse,
      in_dtype=x.dtype,
      cos_sin_dtype=cos_sin_cache.dtype,
  )

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      in_specs=(
          pl.BlockSpec(memory_space=pltpu.HBM),  # x
          pl.BlockSpec(memory_space=pltpu.HBM),  # cos_sin_cache
      ),
      out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
  )

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      grid_spec=grid_spec,
      input_output_aliases={1: 0},  # x (after the scalar prefetch) -> out
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
          disable_bounds_checks=True,
      ),
      name=name,
  )(positions, x, cos_sin_cache)


@functools.partial(
    jax.jit, static_argnames=("inverse", "quant_dtype", "tile_n", "name")
)
def rope_quant(
    x: jax.Array,  # [num_tokens, head_dim] | [num_tokens, num_heads, head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool = False,
    quant_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    tile_n: int | None = None,
    name: str = "rope_quant",
) -> tuple[jax.Array, jax.Array]:
  """RoPE fused with per-row dynamic quantization of the rotated values.

  Same rotation as ``rope``, but the result is quantized to ``quant_dtype``
  inside the kernel. ``tile_n`` is as in ``rope``.
  """
  return cast(
      tuple[jax.Array, jax.Array],
      _rope_call(
          x,
          positions,
          cos_sin_cache,
          inverse=inverse,
          quant_dtype=quant_dtype,
          tile_n=tile_n,
          name=name,
      ),
  )
