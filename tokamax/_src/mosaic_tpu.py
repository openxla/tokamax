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
from functools import lru_cache  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member
from typing import Final, Any, Callable

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp

from tokamax._src.quantization import QuantizedArray  # pylint: disable=g-importing-member

LANES = 128
_sublane_size = lru_cache(lambda: 16 if tpu_generation() >= 7 else 8)


# TODO: Add tests for this file.

_SUPPORTED_TPU_GENERATIONS: Final[dict[str, int]] = {
    "TPU v4": 4,
    "TPU v5 lite": 5,
    "TPU v5": 5,
    "TPU v5e": 5,
    "TPU v5p": 5,
    "TPU v6 lite": 6,
    "TPU7x": 7,
}


def tpu_generation() -> int:
  """Generation number of the currently attached TPU."""
  device_kind = backend.get_default_device().device_kind
  try:
    return _SUPPORTED_TPU_GENERATIONS[device_kind]
  except KeyError as e:
    raise ValueError(f"{device_kind} is not a supported TPU device") from e


def has_mosaic_tpu_support() -> bool:
  """Checks if Mosaic TPU is supported on the attached TPU."""
  return "TPU" in jax.devices()[0].device_kind and tpu_generation() >= 4


def supports_bfloat16_matmul() -> bool:
  """Checks TPU generation to determine if bfloat16 matmul is supported."""
  return "TPU" in jax.devices()[0].device_kind and tpu_generation() >= 4


def assert_is_supported_dtype(dtype: jnp.dtype) -> None:
  if dtype not in (jnp.bfloat16, jnp.float32):
    raise ValueError(f"Expected bfloat16 or float32 array but got {dtype}.")


def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
  """A type to which both input should be adapted to before dot product."""
  # bf16xbf16 matmul is only supported since TPUv4 generation. In case of mixed
  # input precision, we need to convert bf16 argument to fp32 beforehand.
  if not has_mosaic_tpu_support():
    raise ValueError("Mosaic TPU is not supported on this platform.")
  if lhs.dtype == rhs.dtype == jnp.bfloat16:
    return jnp.bfloat16
  else:
    return jnp.float32


def default_quant_dot_dtype() -> jnp.dtype:
  """Returns the default quantized dot dtype."""
  if tpu_generation() >= 7:
    return jnp.float8_e4m3fn
  else:
    return jnp.int8


def quant_block_spec(
    x: QuantizedArray,
    x_spec: pl.BlockSpec,
    axis: int,
    fragment: int | None = None
) -> tuple[QuantizedArray, QuantizedArray]:
  """Broadcast scales so that they are addressable by Pallas via a BlockSpec."""
  axis = axis % x.ndim

  # `fragment` is the lowest addressable unit in a dimension (8, 128) for last 2
  if fragment is None:
    fragment = {
        x.ndim - 1: LANES, x.ndim - 2: _sublane_size(),
    }.get(axis, 1)

  block_shape = list(x_spec.block_shape)
  scale_block_shape = block_shape[:axis] + [fragment] + block_shape[axis+1:]
  scales_tile = pl.cdiv(x.values.shape[axis], x.scales.shape[axis])
  if scales_tile % block_shape[axis] != 0:
    raise NotImplementedError(
        f"{block_shape[axis]=} must divide {scales_tile=} for"
        f" {jax.tree.map(jax.typeof, x)=} with {axis=}"
    )

  def idx_map(*args):
    idxs = list(x_spec.index_map(*args))
    tile_size = x_spec.block_shape[axis]
    return idxs[:axis] + [idxs[axis] * tile_size // scales_tile] + idxs[axis+1:]

  scales = jnp.repeat(x.scales, fragment, axis=axis)
  sspec = pl.BlockSpec(scale_block_shape, idx_map)
  x = dataclasses.replace(x, scales=scales)
  return x, QuantizedArray(x_spec, sspec)  # pytype: disable=wrong-arg-types


def custom_buffered_pallas_call(
    kernel: Callable[..., Any],
    out_shape: jax.ShapeDtypeStruct,
    grid_spec: pltpu.PrefetchScalarGridSpec,
    compiler_params: pltpu.CompilerParams,
    input_buffer_count: list[int] | None = None,
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
      grid = tuple(d[0] if d is not None else s
                   for d, s in zip(args_refs[0], grid_static))

      # unpack the smem prefetch values and bind them to the inspecs
      smem_refs = args_refs[1:num_scalar_prefetch+1]
      _bind_smem = partial(_augment_blockspec, smem_refs=smem_refs)
      in_specs_ = jax.tree.map(_bind_smem, grid_spec.in_specs)
      if input_buffer_count is not None:
        if len(input_buffer_count) != len(in_specs_):
          raise ValueError(
              f"`{input_buffer_count=}` must a list[int] equal in length to"
              f" {len(in_specs_)=}."
          )

        in_specs_ = tuple(
            jax.tree.map(partial(_bind_pipeline, count=c), spec)
            for spec, c in zip(in_specs_, input_buffer_count)
        )
      out_specs_ = jax.tree.map(_bind_smem, grid_spec.out_specs)

      # unpack inputs/outputs and scratch refs
      input_output_refs = args_refs[
          num_scalar_prefetch+1:num_scalar_prefetch + args_len + 1
      ]
      scratch_refs = args_refs[num_scalar_prefetch + args_len + 1:]

      # bind smem and scratch to the pipeline body
      _pipeline = lambda *args: kernel(*smem_refs, *args, *scratch_refs)

      # specify dimension semantic from the scalar prefetch grid and emit
      dim_sem = compiler_params.dimension_semantics
      _emit_pipeline = partial(pltpu.emit_pipeline, dimension_semantics=dim_sem)
      _emit_pipeline(
          _pipeline, grid=grid, in_specs=in_specs_, out_specs=out_specs_
      )(*input_output_refs)

    bs_smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    bs_hbm = pl.BlockSpec(memory_space=pltpu.ANY)

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
        **kw
    )(grid_dynamic, *args)

  # pylint: enable=invalid-name
  return pallas_call
