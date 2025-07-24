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
"""Pallas block utilities."""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools
import threading
from typing import Any, TypeAlias
from unittest import mock

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


_zip = functools.partial(zip, strict=True)


Indexer: TypeAlias = int | slice | pl.Slice | jax.Array


def dslice(idx: int | jax.Array, size: int) -> slice | pl.Slice:
  """Returns a block-aligned `pl.Slice`."""
  return pl.dslice(idx * size, size)


ds = dslice


def _is_scalar_indexer(idx: Indexer) -> bool:
  return isinstance(idx, int) or (isinstance(idx, jax.Array) and idx.ndim == 0)


def _pids() -> tuple[jax.Array, ...]:
  pids = []
  try:
    for i in itertools.count():
      pids.append(pl.program_id(i))
  except ValueError:
    pass  # We iterate until we hit an invalid axis.
  return tuple(pids)


@dataclasses.dataclass(frozen=True)
class BlockRef:
  """A reference to data with a `BlockSpec`.

  NOTE: The bounds-checking logic here assumes that, while the block may go out
  of the bounds of the referenced array, that any user-supplied indexes are
  within the bounds of the block.
  """

  ref: Any
  full_shape: tuple[int, ...]
  spec: pl.BlockSpec

  @property
  def at(self) -> "BlockRefIndexer":
    return BlockRefIndexer(self)

  @property
  def bounds(self) -> tuple[int, ...]:
    """Returns the bounds of the reference in each dimension.

    Returns:
      The bounds of the reference in each dimension. Note that this is not the
      same as `full_shape` as it doesn't include dimensions in which
      `block_shape` is `None` or those sliced out using `at`.
    """
    visible_axes = self._visible_axes
    return tuple(d for i, d in enumerate(self.full_shape) if i in visible_axes)

  @property
  def bounds_checked(self) -> tuple[bool, ...]:
    """Indicates which dimensions require bounds checking."""
    if any(isinstance(bs, pl.Element) for bs in self.spec.block_shape):
      return (True,) * self.ndim
    bounds = self.bounds
    return tuple((bound % dim != 0) for bound, dim in _zip(bounds, self.shape))

  @property
  def inbounds_masks(self) -> tuple[jax.Array | None, ...]:
    """Returns a boolean mask for each dimension indicating if in-bounds.

    Returns:
      Boolean mask indicating if each element is in-bounds, for each dimension.
      Each mask will be broadcastable to the shape of the block. If a dimension,
      doesn't require bounds checking, `None` is returned.
    """
    if (indexer := self._ndindexer) is None:
      idx = (slice(None),) * self.ndim
    else:
      idx = indexer.indices

    start_idxs = self.spec.index_map(*_pids())
    idx = iter(idx)
    masks = []
    block_shape = self.spec.block_shape
    for start_idx, dim, bound in _zip(start_idxs, block_shape, self.full_shape):
      if dim is None:
        continue

      if isinstance(dim, pl.Element):
        if dim.padding is not None:
          raise ValueError("Padding is not supported for unblocked indexing.")
      else:
        start_idx *= dim

      dim_idx = next(idx)
      if _is_scalar_indexer(dim_idx):
        continue

      bcast_axes = [a for a in range(self.ndim) if a != len(masks)]

      if isinstance(dim_idx, jax.Array):
        # TODO: Support multi-dimensional indices.
        if dim_idx.ndim > 1:
          raise ValueError("Only one-dimensional indices are supported.")
        masks.append(jnp.expand_dims(dim_idx < (bound - start_idx), bcast_axes))
        continue

      if dim_idx is None:
        dim_idx = pl.dslice(0, dim)
      elif isinstance(dim_idx, slice):
        dim_idx = pl.Slice.from_slice(dim_idx, dim)

      if not isinstance(dim_idx, pl.Slice):
        raise NotImplementedError("Index must an integer or slice.")

      start = start_idx + dim_idx.start
      mask = dim_idx.stride * jnp.arange(dim_idx.size) < (bound - start)
      masks.append(jnp.expand_dims(mask, bcast_axes))

    return tuple(
        m if check else None for m, check in _zip(masks, self.bounds_checked)
    )

  def inbounds_mask(
      self, *, bounds_check: bool | Sequence[bool] = True
  ) -> jax.Array | None:
    """Returns a boolean mask indicating if each element is in-bounds.

    Args:
      bounds_check: A `bool` or sequence of `bool`s indicating which dimensions'
        masks to include. The mask for dimension `d` will be included if
        `bounds_check[d] and self.bounds_checked[d]` is `True`.

    Returns:
      A boolean mask indicating if each element is in-bounds. This mask will be
      broadcastable to the shape of the block. If no dimensions are bounds
      checked, `None` is returned.
    """
    if isinstance(bounds_check, bool):
      bounds_check = (bounds_check,) * self.ndim
    mask = None
    for dim_mask, check in _zip(self.inbounds_masks, bounds_check):
      if dim_mask is None or not check:
        continue
      mask = dim_mask if mask is None else (mask & dim_mask)
    return mask

  def load(
      self,
      *,
      bounds_check: bool | Sequence[bool] = True,
      other: Any = None,
      **kwargs,
  ) -> jax.Array:
    """Loads a block with `mask=inbounds_mask(bounds_check=bounds_check)`."""
    uses_element_indexing = any(
        isinstance(bd, pl.Element) for bd in self.spec.block_shape
    )
    if "eviction_policy" not in kwargs and uses_element_indexing:
      # If the index mapping function is a function of every program ID, then
      # it means that block is only read by one program ID. It is not likely to
      # be beneficial to keep the values in the cache. Here, we simply check if
      # the index mapping contains every program ID.
      # TODO: Can we detect more complicated functions of the program IDs?
      pids = _pids()
      start_idxs = tuple(map(id, self.spec.index_map(*pids)))
      if all(pid in start_idxs for pid in map(id, pids)):
        kwargs["eviction_policy"] = "evict_first"

    mask = self.inbounds_mask(bounds_check=bounds_check)
    if mask is None:
      other = None
    elif other is None:
      other = 0
    return pl.load(self.ref, (), mask=mask, other=other, **kwargs)

  def store(self, val: jax.Array, **kwargs):
    """Stores a block with `mask=inbounds_mask()`."""
    if (mask := self.inbounds_mask()) is None:
      mask = kwargs.pop("mask", None)
    else:
      mask &= kwargs.pop("mask", True)
    pl.store(self.ref, (), val, mask=mask, **kwargs)

  def __getattr__(self, name: str) -> Any:
    return getattr(self.ref, name)

  @property
  def _visible_axes(self) -> tuple[int, ...]:
    """Returns the axes of the referenced array that are visible."""
    axes = (a for a, d in enumerate(self.spec.block_shape) if d is not None)
    if (indexer := self._ndindexer) is not None:
      idxs = indexer.indices
      axes = (a for a, idx in _zip(axes, idxs) if not _is_scalar_indexer(idx))
    return tuple(axes)

  @property
  def _ndindexer(self) -> Any | None:
    """Returns the indexer into the block, or `None`."""
    if not hasattr(self.ref, "transforms"):
      return None

    if len(self.ref.transforms) != 1:
      raise NotImplementedError("`BlockRef` only supports one transform.")

    if not hasattr(self.ref.transforms[0], "indices"):
      raise NotImplementedError("`BlockRef` only supports indexer transforms.")

    return self.ref.transforms[0]


@dataclasses.dataclass(frozen=True)
class BlockRefIndexer:
  """Indexer into a `BlockRef`."""

  ref: BlockRef

  def __getitem__(self, idx: Indexer | Sequence[Indexer]) -> BlockRef:
    return dataclasses.replace(self.ref, ref=self.ref.ref.at[idx])


_PL_LOAD_STORE_PATCH_LOCK = threading.Lock()


def _block_ref(ref, value, spec):
  if spec is pl.no_block_spec:
    spec = pl.BlockSpec(value.shape, lambda *_: (0,) * value.ndim)
  return BlockRef(ref, value.shape, spec)


def _as_tuple(x):
  return tuple(x) if isinstance(x, Sequence) else (x,)


def pallas_call(
    kernel: Callable[..., None],
    out_shape: Any,
    *,
    grid: tuple[int, ...] = (),
    grid_spec: pl.GridSpec | None = None,
    in_specs: Any = pl.no_block_spec,
    out_specs: Any = pl.no_block_spec,
    filter_specs: bool = False,
    **kwargs,
) -> Callable[..., Any]:
  """Invokes `pallas_call`, wrapping refs with a `BlockSpec` as `BlockRef`s."""
  if grid_spec is None:
    grid_spec = pl.GridSpec(grid, in_specs, out_specs)
  elif grid:
    raise ValueError(
        f"If `grid_spec` is specified, then `grid` must be `()`. It is {grid}"
    )
  elif in_specs is not pl.no_block_spec:
    raise ValueError(
        "If `grid_spec` is specified, then `in_specs` must be `no_block_spec`. "
        f"It is {in_specs}"
    )
  elif out_specs is not pl.no_block_spec:
    raise ValueError(
        "If `grid_spec` is specified, then `out_specs` must be `no_block_spec`."
        f" It is {out_specs}"
    )

  filter_spec = lambda spec, x: None if x is None else spec

  out_specs = grid_spec.out_specs
  if out_specs is pl.no_block_spec:
    out_specs = jax.tree.map(lambda _: pl.no_block_spec, out_shape)
  elif filter_specs:
    out_specs = jax.tree.map(filter_spec, out_specs, out_shape)

  def helper(*args, grid_spec=grid_spec):
    num_args = len(args)

    in_specs = grid_spec.in_specs
    if in_specs is pl.no_block_spec:
      in_specs = jax.tree.map(lambda _: pl.no_block_spec, args)
    elif filter_specs:
      in_specs = jax.tree.map(filter_spec, _as_tuple(in_specs), args)

    @functools.wraps(kernel)
    def wrapped_kernel(*refs):
      in_refs, out_refs = refs[:num_args], refs[num_args:]
      in_refs = jax.tree.map(_block_ref, in_refs, args, in_specs)
      out_shape_, out_specs_ = map(_as_tuple, (out_shape, out_specs))
      out_refs = jax.tree.map(_block_ref, out_refs, out_shape_, out_specs_)

      # Patch `plgpu.load` and `plgpu.store` to allow `BlockRef` to be passed.
      # `mock.patch` is not thread-safe, so hold the lock while patching.
      def ld_st(fn, ref, *args, **kwargs):
        if isinstance(ref, BlockRef):
          ref = ref.ref
        return fn(ref, *args, **kwargs)

      with (
          _PL_LOAD_STORE_PATCH_LOCK,
          mock.patch.object(pl, "load", functools.partial(ld_st, pl.load)),
          mock.patch.object(pl, "store", functools.partial(ld_st, pl.store)),
      ):
        return kernel(*in_refs, *out_refs)

    grid_spec = pl.GridSpec(
        grid=grid_spec.grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=grid_spec.scratch_shapes,
    )
    return pl.pallas_call(
        wrapped_kernel,
        out_shape,
        grid_spec=grid_spec,
        **kwargs,
    )(*args)

  return helper
