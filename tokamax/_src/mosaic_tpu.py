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
"""Mosaic-TPU utils."""

import dataclasses
import functools
from typing import Any, Callable, Sequence

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import qwix


QArray = qwix.QArray

LANES = 128


@functools.lru_cache
def _adaptive_sublane_size() -> int:
  """Returns the sublane size based on the generation.

  The standard `pltpu.get_tpu_info().num_sublanes` is 8 for all generations, and
  this returns 16 for TPU generation 7 and above.
  """
  return 16 if pltpu.get_tpu_info().generation >= 7 else 8


# TODO: Add tests for this file.


@jax.tree_util.register_static
@dataclasses.dataclass
class ScalesTilingInfo:
  true_scales_per_tile: int
  scales_tile_size: int | None
  ti_to_sti: Callable[[int], int]  # tile index to scale tile index

  @property
  def scales_inflation(self):
    return (self.scales_tile_size or 1) // self.true_scales_per_tile


def _get_scale_tile_info(
    eps: int, tile_size: int | None, axis_size: int, min_addressable_size: int
) -> ScalesTilingInfo:
  if tile_size is None:
    if min_addressable_size != 1:
      raise ValueError(
          "We can only collapse a dimension when it is"
          " individually addressable. Pallas Mosaic TPU has the"
          " same restriction."
      )
    return ScalesTilingInfo(
        true_scales_per_tile=1,
        scales_tile_size=None,
        ti_to_sti=lambda tile_idx: tile_idx // eps,
    )
  if eps == axis_size:
    return ScalesTilingInfo(
        true_scales_per_tile=1,
        scales_tile_size=min_addressable_size,
        ti_to_sti=lambda tile_idx: 0,
    )

  assert eps % tile_size == 0 or tile_size % eps == 0
  scales_per_tile = max(1, tile_size // eps)
  if eps > tile_size:
    tile_idx_to_scales_tile_idx = lambda tile_idx: tile_idx * tile_size // eps
    scales_tile_size = 1
  else:  # eps <= tile_size
    tile_idx_to_scales_tile_idx = lambda tile_idx: tile_idx
    scales_tile_size = tile_size // eps

  if eps == 1:
    assert scales_per_tile % min_addressable_size == 0
    scales_tile_size = scales_per_tile
  else:
    scales_tile_size = scales_tile_size * min_addressable_size

  return ScalesTilingInfo(
      scales_per_tile, scales_tile_size, tile_idx_to_scales_tile_idx
  )


def quant_block_spec(
    x: QArray, x_spec: pl.BlockSpec, reduction_axis: int
) -> tuple[QArray, QArray]:
  """Broadcast scales so that they are addressable by Pallas via a BlockSpec."""
  x_values, x_scales = x.qvalue, x.scale

  eps_list = [pl.cdiv(xs, ss) for xs, ss in zip(x_values.shape, x_scales.shape)]
  tile_sizes = x_spec.block_shape
  min_addressable_sizes = (
      [1] * x.ndim + [_adaptive_sublane_size(), pltpu.get_tpu_info().num_lanes]
  )[-x.ndim :]

  # Limitation 1: Currently, we only support full-axis scales or 1 scale per
  # each element for non-reduction axes, this is supported by the block-spec,
  # but not by the kernel implementation.
  for axis, eps in enumerate(eps_list):
    if axis != reduction_axis and eps not in (1, x_values.shape[axis]):
      raise NotImplementedError(
          "Non-reduction axes must have an either 1 scale per each element or a"
          " scalar full-axis scale, but"
          f" {jax.tree.map(jax.typeof, x)=} for {reduction_axis=}."
      )

  # Limitation 2: The number of elements per single scale along the reduction
  # axis must be greater than or equal to the min_addressable_size, because we
  # don't want to pay the in-kernel cost of individually addressing the scales.
  if eps_list[reduction_axis] < min_addressable_sizes[reduction_axis]:
    raise NotImplementedError(
        "The number of elements per single scale along the reduction axis must"
        " be greater than or equal to the min_addressable_size."
    )

  for axis, (eps, mas) in enumerate(zip(eps_list, min_addressable_sizes)):
    if eps != 1 and (eps % mas != 0 and eps != x_values.shape[axis]):
      raise NotImplementedError(
          "The number of elements per single scale must be divisible by the"
          f" min_addressable_size in {axis=}. Got {eps=} and {mas=}."
      )
  scale_info_per_axis = [
      _get_scale_tile_info(eps, tile_size, s, mas)
      for eps, tile_size, s, mas in zip(
          eps_list, tile_sizes, x_values.shape, min_addressable_sizes
      )
  ]

  # construct the blockspec
  def index_map(*args):
    idxs = list(x_spec.index_map(*args))
    return [scale_info_per_axis[i].ti_to_sti(idx) for i, idx in enumerate(idxs)]

  scales_block_shape = [info.scales_tile_size for info in scale_info_per_axis]
  scales_blockspec = dataclasses.replace(
      x_spec, block_shape=scales_block_shape, index_map=index_map
  )

  # repeat the scales to make them individually addressable
  for ax, info in enumerate(scale_info_per_axis):
    x_scales = jnp.repeat(x_scales, info.scales_inflation, axis=ax)

  # return the results
  x_new = dataclasses.replace(x, scale=x_scales)
  x_spec_new = dataclasses.replace(x, qvalue=x_spec, scale=scales_blockspec)  # pytype: disable=wrong-arg-types
  return x_new, x_spec_new


def custom_buffered_pallas_call(
    kernel: Callable[..., Any],
    out_shape: jax.ShapeDtypeStruct,
    grid_spec: pltpu.PrefetchScalarGridSpec,
    compiler_params: pltpu.CompilerParams,
    input_buffer_count: Sequence[int] | None = None,
    **kw,
):
  # pylint: disable=invalid-name
  """Custom PrefetchScalarGrid pallas_call using emit_pipeline."""
  num_scalar_prefetch = grid_spec.num_scalar_prefetch
  len_ = lambda x: len(x) if isinstance(x, (list, tuple)) else 1
  args_len = len_(grid_spec.in_specs) + len_(grid_spec.out_specs)

  def _augment_blockspec(bs, smem_refs):
    index_map_ = lambda *idxs: bs.index_map(*idxs, *smem_refs)
    return pl.BlockSpec(bs.block_shape, index_map_)

  grid_static = tuple(
      dim if isinstance(dim, int) else None for dim in grid_spec.grid
  )
  grid_dynamic = tuple(
      None if isinstance(dim, int) else jnp.atleast_1d(dim)
      for dim in grid_spec.grid
  )

  def _bind_pipeline(spec, count):
    if count == 2:
      return spec
    return dataclasses.replace(
        spec, pipeline_mode=pl.Buffered(buffer_count=count, use_lookahead=True)
    )

  def pallas_call(*args):
    smem_args = args[:num_scalar_prefetch]  # close over smem data

    def pipeline(*args_refs):
      # unpack the dynamic grid elements from smem
      grid = tuple(
          d[0] if d is not None else s
          for d, s in zip(args_refs[0], grid_static)
      )

      # unpack the smem prefetch values and bind them to the inspecs
      smem_refs = args_refs[1 : num_scalar_prefetch + 1]
      _bind_smem = functools.partial(_augment_blockspec, smem_refs=smem_refs)
      in_specs_ = jax.tree.map(_bind_smem, grid_spec.in_specs)
      if input_buffer_count is not None:
        if len(input_buffer_count) != len(in_specs_):
          raise ValueError(
              f"`{input_buffer_count=}` must a list[int] equal in length to"
              f" {len(in_specs_)=}."
          )

        in_specs_ = tuple(
            jax.tree.map(functools.partial(_bind_pipeline, count=c), spec)
            for spec, c in zip(in_specs_, input_buffer_count)
        )
      out_specs_ = jax.tree.map(_bind_smem, grid_spec.out_specs)

      # unpack inputs/outputs and scratch refs
      input_output_refs = args_refs[
          num_scalar_prefetch + 1 : num_scalar_prefetch + args_len + 1
      ]
      scratch_refs = args_refs[num_scalar_prefetch + args_len + 1 :]

      # bind smem and scratch to the pipeline body
      # specify dimension semantic from the scalar prefetch grid and emit
      pltpu.emit_pipeline(
          lambda *args: kernel(*smem_refs, *args, *scratch_refs),
          grid=grid,
          in_specs=in_specs_,
          out_specs=out_specs_,
          dimension_semantics=compiler_params.dimension_semantics,
      )(*input_output_refs)

    bs_smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    bs_hbm = pl.BlockSpec(memory_space=pl.ANY)

    smem_specs = (jax.tree.map(lambda _: bs_smem, grid_dynamic),)
    smem_specs += jax.tree.map(lambda _: bs_smem, smem_args)
    in_specs = jax.tree.map(lambda _: bs_hbm, tuple(grid_spec.in_specs))
    out_specs = jax.tree.map(lambda _: bs_hbm, grid_spec.out_specs)

    params = dataclasses.replace(compiler_params, dimension_semantics=())
    return pl.pallas_call(
        pipeline,
        out_shape,
        compiler_params=params,
        in_specs=smem_specs + in_specs,
        out_specs=out_specs,
        scratch_shapes=grid_spec.scratch_shapes,
        **kw,
    )(grid_dynamic, *args)

  # pylint: enable=invalid-name
  return pallas_call
