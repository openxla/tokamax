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
"""Pallas fuser integration for GMM v2 and TGMM v2 kernels."""

import dataclasses
import inspect

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops import fuser_adapter
from tokamax._src.ops.gmm_v2 import gmm_v2
from tokamax._src.ops.gmm_v2 import tgmm_v2


def _to_2d_bounded_spec(spec_3d: pl.BlockSpec) -> pl.BlockSpec:
  """Converts a 3D sublane-reshaped BoundedSlice BlockSpec to a 2D BlockSpec."""
  assert spec_3d.block_shape is not None and spec_3d.index_map is not None
  bounded_gm, sublane_dim, col_dim = spec_3d.block_shape
  assert isinstance(bounded_gm, pl.BoundedSlice)
  sublane = int(sublane_dim)  # pyrefly: ignore[bad-argument-type]
  tile_m = bounded_gm.block_size * sublane
  index_map_3d = spec_3d.index_map

  def index_map_2d(*grid_indices):
    row_slice, _, col_idx = index_map_3d(*grid_indices)
    return (
        pl.ds(
            pl.multiple_of(row_slice.start * sublane, sublane),
            pl.multiple_of(row_slice.size * sublane, sublane),
        ),
        col_idx,
    )

  return pl.BlockSpec(
      (pl.BoundedSlice(tile_m), col_dim),
      index_map_2d,
      memory_space=spec_3d.memory_space,
      pipeline_mode=spec_3d.pipeline_mode,
  )


def _unwrap_trivial_input_fusion(
    x: jax.Array | fuser_adapter.InputFusion,
) -> jax.Array | fuser_adapter.InputFusion:
  """Unwraps identity `fuser.Fusion` closures that have no producer ops."""
  if isinstance(x, fuser.Fusion) and not jax.make_jaxpr(x)().jaxpr.eqns:
    return x()
  return x


@dataclasses.dataclass(frozen=True)
class GmmFusionSpec:
  """Fusion specification for the GMM v2 kernel.

  Attributes:
    lhs_fusion: Optional `fuser.Fusion` prologue for `lhs`, or None if unfused.
    rhs_fusion: Optional `fuser.Fusion` prologue for `rhs`, or None if unfused.
    out_fusions: Optional output epilogue fusion from `pallas.fuser`, or None.
  """

  lhs_fusion: fuser_adapter.InputFusion | None = None
  rhs_fusion: fuser_adapter.InputFusion | None = None
  out_fusions: fuser_adapter.OutputFusion | None = None

  @property
  def is_lhs_fused(self) -> bool:
    """Returns whether `lhs` carries a fused prologue."""
    return self.lhs_fusion is not None

  @property
  def is_rhs_fused(self) -> bool:
    """Returns whether `rhs` carries a fused prologue."""
    return self.rhs_fusion is not None

  @property
  def is_out_fused(self) -> bool:
    """Returns whether the output carries a fused epilogue."""
    return self.out_fusions is not None

  def prepare(
      self,
      *,
      lhs: jax.Array | fuser_adapter.InputFusion,
      rhs: jax.Array | fuser_adapter.InputFusion,
      lhs_scale: jax.Array | None,
      rhs_scale: jax.Array | None,
      rhs_bias: jax.Array | None,
      cfgs: gmm_v2.GmmConfigs,
      raw_out_type: jax.ShapeDtypeStruct,
  ) -> tuple[
      gmm_v2.LhsRef[jax.Array] | gmm_v2.FusedLhsRef[jax.Array],
      gmm_v2.WeightsRef | gmm_v2.PrologueWeightsRef[jax.Array],
      gmm_v2.OutFusionInputRefs,
      jax.ShapeDtypeStruct,
      pl.MemoryRef,
      gmm_v2.GmmSpecTransformFn,
  ]:
    """Unpacks fusion values and builds the GMM BlockSpec transform.

    Args:
      lhs: Left-hand side array or `fuser.Fusion` object.
      rhs: Right-hand side array or `fuser.Fusion` object.
      lhs_scale: Optional quantization scale for `lhs`, or None if unscaled.
      rhs_scale: Optional quantization scale for `rhs`, or None if unscaled.
      rhs_bias: Optional bias for `rhs`, or None if unbiased.
      cfgs: Resolved `GmmConfigs` for the kernel invocation.
      raw_out_type: Unfused output shape and dtype before any epilogue fusion.

    Returns:
      A 6-tuple `(lhs_in, rhs_weights, out_fusion_input_values, out_init,
      partial_out_shape, spec_transform_fn)`.

    Raises:
      NotImplementedError: If unsupported options (such as `partial_sum`,
        packed weights with fused `rhs`, or unaligned fused output dimensions)
        are combined with fusion.
    """
    dims = cfgs.dims
    tiles = cfgs.tiles
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_n = gmm_v2.align_to(cfgs.out_size_n, num_lanes)

    if cfgs.has_partial_sum:
      raise NotImplementedError("Fused gmm_v2 does not support partial_sum.")
    if self.is_rhs_fused and (
        cfgs.fuse_act is not None or cfgs.rhs_cfgs.should_unpack
    ):
      raise NotImplementedError(
          "Fused rhs in gmm_v2 is not supported with fuse_act or packed"
          " sub-byte weights."
      )

    if self.lhs_fusion is not None:
      lhs_fn, lhs_values, lhs_values_type = fuser_adapter.unpack_input_fusion(
          self.lhs_fusion
      )
      assert lhs_fn is not None and lhs_values_type is not None
      lhs_in: gmm_v2.LhsRef[jax.Array] | gmm_v2.FusedLhsRef[jax.Array] = (
          gmm_v2.FusedLhsRef(values=lhs_values, scale=lhs_scale)
      )
    else:
      assert not isinstance(lhs, fuser.Fusion)
      lhs_fn = lhs_values_type = None
      lhs_in = gmm_v2.LhsRef(value=lhs, scale=lhs_scale)

    if self.rhs_fusion is not None:
      rhs_fn, rhs_values, rhs_values_type = fuser_adapter.unpack_input_fusion(
          self.rhs_fusion
      )
      assert rhs_fn is not None and rhs_values_type is not None
      rhs_weights: (
          gmm_v2.WeightsRef | gmm_v2.PrologueWeightsRef[jax.Array]
      ) = gmm_v2.PrologueWeightsRef(
          weights=rhs_values, scale=rhs_scale, bias=rhs_bias
      )
    else:
      assert not isinstance(rhs, fuser.Fusion)
      rhs_fn = rhs_values_type = None
      rhs_weights = gmm_v2.WeightsRef(
          weight=rhs, scale=rhs_scale, bias=rhs_bias
      )

    if self.out_fusions is not None:
      if cfgs.out_size_n != aligned_n:
        raise NotImplementedError(
            f"Fused output in gmm_v2 requires out_size_n ({cfgs.out_size_n})"
            f" to be aligned to num_lanes ({num_lanes})."
        )
      out_fn, out_values, out_values_type, out_init = (
          fuser_adapter.unpack_output_fusion(self.out_fusions, raw_out_type)
      )
      assert out_values is not None
      out_fusion_input_values: gmm_v2.OutFusionInputRefs = out_values
      if (
          len(out_init.shape) != 2
          or out_init.shape[0] != dims.size_m
          or out_init.shape[1] % num_lanes != 0
      ):
        raise NotImplementedError(
            f"Unsupported fused output shape {out_init.shape} for gmm_v2"
            f" raw output {raw_out_type.shape}."
        )
      num_n = pl.cdiv(cfgs.out_size_n, tiles.tile_n)
      out_tile_n = out_init.shape[1] // num_n
    else:
      out_fn = out_values_type = None
      out_fusion_input_values = ()
      out_init = raw_out_type
      out_tile_n = tiles.tile_n

    partial_out_shape = pltpu.VMEM(
        (dims.size_lhs_sublane, out_tile_n), out_init.dtype
    )

    def spec_transform_fn(
        *,
        lhs_ref: gmm_v2.LhsRef[jax.Array] | gmm_v2.FusedLhsRef[jax.Array],
        out_ref: jax.Array,
        lhs_spec: gmm_v2.LhsSpec,
        rhs_spec: gmm_v2.WeightsRef,
        out_spec: pl.BlockSpec,
        cfgs: gmm_v2.GmmConfigs,
    ) -> tuple[
        gmm_v2.LhsRef[jax.Array] | gmm_v2.FusedLhsRef[jax.Array],
        jax.Array,
        gmm_v2.LhsSpec | gmm_v2.FusedLhsSpec,
        gmm_v2.WeightsRef | gmm_v2.PrologueWeightsSpec,
        gmm_v2.OutFusionInputSpecs,
        pl.BlockSpec,
        fuser_adapter.PrologueFn | None,
        fuser_adapter.PrologueFn | None,
        fuser_adapter.EpilogueFn | None,
    ]:
      if lhs_fn is not None:
        assert lhs_values_type is not None
        assert isinstance(lhs_ref, gmm_v2.FusedLhsRef)
        pulled_lhs_fn, lhs_val_specs = fuser_adapter.pull_input_spec(
            lhs_fn,
            _to_2d_bounded_spec(lhs_spec.value),
            lhs_values_type,
            grid_len=3,
        )
        k_lhs_spec: gmm_v2.LhsSpec | gmm_v2.FusedLhsSpec = gmm_v2.FusedLhsRef(
            values=lhs_val_specs, scale=lhs_spec.scale
        )
        k_lhs_in: gmm_v2.LhsRef[jax.Array] | gmm_v2.FusedLhsRef[jax.Array] = (
            lhs_ref
        )
      else:
        assert isinstance(lhs_ref, gmm_v2.LhsRef)
        pulled_lhs_fn = None
        k_lhs_spec = lhs_spec
        lhs_value_in = lhs_ref.value.reshape(
            -1, cfgs.dims.size_lhs_sublane, lhs_ref.value.shape[-1]
        )
        k_lhs_in = gmm_v2.LhsRef(value=lhs_value_in, scale=lhs_ref.scale)

      if rhs_fn is not None:
        assert rhs_values_type is not None
        pulled_rhs_fn, rhs_val_specs = fuser_adapter.pull_input_spec(
            rhs_fn, rhs_spec.weight, rhs_values_type, grid_len=3
        )
        k_rhs_spec: gmm_v2.WeightsRef | gmm_v2.PrologueWeightsSpec = (
            gmm_v2.PrologueWeightsRef(
                weights=rhs_val_specs,
                scale=rhs_spec.scale,
                bias=rhs_spec.bias,
            )
        )
      else:
        pulled_rhs_fn = None
        k_rhs_spec = rhs_spec

      if out_fn is not None:
        assert out_values_type is not None
        out_2d_spec = _to_2d_bounded_spec(out_spec)
        out_read_spec = _make_out_read_spec(lhs_spec.value, out_2d_spec)
        # Note that: We can NOT use the `out_fusion_input_specs` derived from
        # `fuser_adapter.push_pull_block_spec(..., out_2d_spec, ...)`
        # because `out_2d_spec` has a trick that does not write the last sublane
        # and then defer to the next grid iteration.
        # This is problematic because when we are reading, we always want to
        # read all the contents, including the last sublane.
        pulled_out_fn, out_fusion_input_specs, _ = (
            fuser_adapter.push_pull_block_spec(
                out_fn,
                out_read_spec,
                out_values_type,
                raw_out_type,
                grid_len=3,
            )
        )
        _, _, out_spec = fuser_adapter.push_pull_block_spec(
            out_fn,
            out_2d_spec,
            out_values_type,
            raw_out_type,
            grid_len=3,
        )
        k_out_in = out_ref
      else:
        pulled_out_fn = None
        out_fusion_input_specs: gmm_v2.OutFusionInputSpecs = ()
        k_out_in = out_ref.reshape(
            -1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1]
        )

      return (
          k_lhs_in,
          k_out_in,
          k_lhs_spec,
          k_rhs_spec,
          out_fusion_input_specs,
          out_spec,
          pulled_lhs_fn,
          pulled_rhs_fn,
          pulled_out_fn,
      )

    return (
        lhs_in,
        rhs_weights,
        out_fusion_input_values,
        out_init,
        partial_out_shape,
        spec_transform_fn,
    )


@dataclasses.dataclass(frozen=True)
class TgmmFusionSpec:
  """Fusion specification for the TGMM v2 kernel.

  Attributes:
    lhs_fusion: Optional `fuser.Fusion` prologue for `lhs`, or None if unfused.
    rhs_fusion: Optional `fuser.Fusion` prologue for `rhs`, or None if unfused.
    out_fusions: Optional output epilogue fusion from `pallas.fuser`, or None.
  """

  lhs_fusion: fuser_adapter.InputFusion | None = None
  rhs_fusion: fuser_adapter.InputFusion | None = None
  out_fusions: fuser_adapter.OutputFusion | None = None

  @property
  def is_lhs_fused(self) -> bool:
    """Returns whether `lhs` carries a fused prologue."""
    return self.lhs_fusion is not None

  @property
  def is_rhs_fused(self) -> bool:
    """Returns whether `rhs` carries a fused prologue."""
    return self.rhs_fusion is not None

  @property
  def is_out_fused(self) -> bool:
    """Returns whether the output carries a fused epilogue."""
    return self.out_fusions is not None

  def prepare(
      self,
      *,
      lhs: jax.Array | fuser_adapter.InputFusion,
      rhs: jax.Array | fuser_adapter.InputFusion,
      rhs_scale: jax.Array | None,
      cfgs: gmm_v2.GmmConfigs,
      raw_out_type: jax.ShapeDtypeStruct,
  ) -> tuple[
      jax.Array | tgmm_v2.PrologueRefs,
      tgmm_v2.OperandRef[jax.Array] | tgmm_v2.FusedOperandRef[jax.Array],
      gmm_v2.OutFusionInputRefs,
      jax.ShapeDtypeStruct,
      tgmm_v2.TgmmSpecTransformFn,
  ]:
    """Unpacks fusion values and builds the TGMM BlockSpec transform.

    Args:
      lhs: Left-hand side array or `fuser.Fusion` object.
      rhs: Right-hand side array or `fuser.Fusion` object.
      rhs_scale: Optional per-channel scale for `rhs`, or None if unscaled.
      cfgs: Resolved `GmmConfigs` for the TGMM kernel invocation.
      raw_out_type: Unfused output shape and dtype before any epilogue fusion.

    Returns:
      A 5-tuple `(lhs_in, rhs_operand, out_fusion_input_values, out_init,
      spec_transform_fn)`.

    Raises:
      NotImplementedError: If unsupported options (such as `partial_sum`,
        sublane-unaligned `size_m` with fused inputs, or tile-unaligned output
        dimensions with fused outputs) are combined with fusion.
    """
    dims = cfgs.dims
    tiles = cfgs.tiles
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_n = gmm_v2.align_to(dims.size_n, num_lanes)
    aligned_k = gmm_v2.align_to(dims.size_k, tiles.tile_k)

    if cfgs.has_partial_sum:
      raise NotImplementedError("Fused tgmm_v2 does not support partial_sum.")
    if (self.is_lhs_fused or self.is_rhs_fused) and (
        dims.size_m % dims.size_lhs_sublane != 0
    ):
      raise NotImplementedError(
          f"Fused tgmm_v2 requires size_m ({dims.size_m}) to be aligned to"
          f" size_lhs_sublane ({dims.size_lhs_sublane})."
      )

    if self.lhs_fusion is not None:
      lhs_fn, lhs_values, lhs_values_type = fuser_adapter.unpack_input_fusion(
          self.lhs_fusion
      )
      assert lhs_fn is not None and lhs_values_type is not None
      lhs_in: jax.Array | tgmm_v2.PrologueRefs = lhs_values
    else:
      assert not isinstance(lhs, fuser.Fusion)
      lhs_fn = lhs_values_type = None
      lhs_in = lhs

    if self.rhs_fusion is not None:
      rhs_fn, rhs_values, rhs_values_type = fuser_adapter.unpack_input_fusion(
          self.rhs_fusion
      )
      assert rhs_fn is not None and rhs_values_type is not None
      rhs_operand: (
          tgmm_v2.OperandRef[jax.Array] | tgmm_v2.FusedOperandRef[jax.Array]
      ) = tgmm_v2.FusedOperandRef(values=rhs_values, scale=rhs_scale)
    else:
      assert not isinstance(rhs, fuser.Fusion)
      rhs_fn = rhs_values_type = None
      rhs_operand = tgmm_v2.OperandRef(value=rhs, scale=rhs_scale)

    if self.out_fusions is not None:
      if dims.size_k != aligned_k or dims.size_n != aligned_n:
        raise NotImplementedError(
            "Fused output in tgmm_v2 requires size_k and size_n to be tile"
            " aligned."
        )
      out_fn, out_values, out_values_type, out_init = (
          fuser_adapter.unpack_output_fusion(self.out_fusions, raw_out_type)
      )
      assert out_values is not None
      out_fusion_input_values: gmm_v2.OutFusionInputRefs = out_values
      if (
          len(out_init.shape) != 3
          or out_init.shape[0] != dims.size_group
          or out_init.shape[2] % num_lanes != 0
      ):
        raise NotImplementedError(
            f"Unsupported fused output shape {out_init.shape} for tgmm_v2"
            f" raw output {raw_out_type.shape}."
        )
    else:
      out_fn = out_values_type = None
      out_fusion_input_values = ()
      out_init = raw_out_type

    def spec_transform_fn(
        *,
        lhs_ref: jax.Array | tgmm_v2.PrologueRefs,
        rhs_ref: (
            tgmm_v2.OperandRef[jax.Array] | tgmm_v2.FusedOperandRef[jax.Array]
        ),
        lhs_spec: pl.BlockSpec,
        rhs_spec: tgmm_v2.OperandSpec,
        out_specs: pl.BlockSpec,
        cfgs: gmm_v2.GmmConfigs,
    ) -> tuple[
        jax.Array | tgmm_v2.PrologueRefs,
        tgmm_v2.OperandRef[jax.Array] | tgmm_v2.FusedOperandRef[jax.Array],
        pl.BlockSpec | tgmm_v2.PrologueSpecs,
        tgmm_v2.OperandSpec | tgmm_v2.FusedOperandSpec,
        gmm_v2.OutFusionInputSpecs,
        pl.BlockSpec,
        fuser_adapter.PrologueFn | None,
        fuser_adapter.PrologueFn | None,
        fuser_adapter.EpilogueFn | None,
    ]:
      if lhs_fn is not None:
        assert lhs_values_type is not None
        assert isinstance(lhs_ref, tuple)
        pulled_lhs_fn, pulled_lhs_specs = fuser_adapter.pull_input_spec(
            lhs_fn, _to_2d_bounded_spec(lhs_spec), lhs_values_type, grid_len=3
        )
        k_lhs_spec: pl.BlockSpec | tgmm_v2.PrologueSpecs = pulled_lhs_specs
        k_lhs_in: jax.Array | tgmm_v2.PrologueRefs = lhs_ref
      else:
        assert not isinstance(lhs_ref, tuple)
        pulled_lhs_fn = None
        k_lhs_spec = lhs_spec
        k_lhs_in = lhs_ref.reshape(
            -1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1]
        )

      if rhs_fn is not None:
        assert rhs_values_type is not None
        assert isinstance(rhs_ref, tgmm_v2.FusedOperandRef)
        pulled_rhs_fn, rhs_val_spec = fuser_adapter.pull_input_spec(
            rhs_fn,
            _to_2d_bounded_spec(rhs_spec.value),
            rhs_values_type,
            grid_len=3,
        )
        k_rhs_spec: tgmm_v2.OperandSpec | tgmm_v2.FusedOperandSpec = (
            tgmm_v2.FusedOperandRef(values=rhs_val_spec, scale=rhs_spec.scale)
        )
        k_rhs_operand: (
            tgmm_v2.OperandRef[jax.Array] | tgmm_v2.FusedOperandRef[jax.Array]
        ) = rhs_ref
      else:
        assert isinstance(rhs_ref, tgmm_v2.OperandRef)
        pulled_rhs_fn = None
        k_rhs_spec = rhs_spec
        rhs_value = rhs_ref.value
        rhs_in = rhs_value.reshape(
            -1, cfgs.dims.size_lhs_sublane, rhs_value.shape[-1]
        )
        k_rhs_operand = tgmm_v2.OperandRef(value=rhs_in, scale=rhs_ref.scale)

      if out_fn is not None:
        assert out_values_type is not None
        pulled_out_fn, out_fusion_input_specs, out_specs = (
            fuser_adapter.push_pull_block_spec(
                out_fn,
                out_specs,
                out_values_type,
                raw_out_type,
                grid_len=3,
            )
        )
      else:
        pulled_out_fn = None
        out_fusion_input_specs: gmm_v2.OutFusionInputSpecs = ()

      return (
          k_lhs_in,
          k_rhs_operand,
          k_lhs_spec,
          k_rhs_spec,
          out_fusion_input_specs,
          out_specs,
          pulled_lhs_fn,
          pulled_rhs_fn,
          pulled_out_fn,
      )

    return (
        lhs_in,
        rhs_operand,
        out_fusion_input_values,
        out_init,
        spec_transform_fn,
    )


def _run_gmm_v2_fusible(
    lhs: jax.Array | fuser_adapter.InputFusion,
    rhs: jax.Array | fuser_adapter.InputFusion,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    lhs_scale: jax.Array | None = None,
    partial_sum: jax.Array | None = None,
    *,
    tile_info: gmm_v2.TileSizes | gmm_v2.TileFn = (
        gmm_v2.calculate_tiling_from_request
    ),
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    acc_dtype: jax.typing.DTypeLike | None = None,
    maybe_quantize_lhs: bool = True,
    lhs_quant_dtype: jax.typing.DTypeLike | None = None,
    rhs_quant_dtype: jax.typing.DTypeLike | None = None,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    transpose_rhs: bool = False,
    out_fusions: fuser_adapter.OutputFusion | None = None,
) -> jax.Array:
  """Runs unwrapped `gmm_v2` with an optional `GmmFusionSpec`."""
  lhs = _unwrap_trivial_input_fusion(lhs)
  rhs = _unwrap_trivial_input_fusion(rhs)
  fusion_spec = None
  if (
      isinstance(lhs, fuser.Fusion)
      or isinstance(rhs, fuser.Fusion)
      or out_fusions is not None
  ):
    fusion_spec = GmmFusionSpec(
        lhs_fusion=lhs if isinstance(lhs, fuser.Fusion) else None,
        rhs_fusion=rhs if isinstance(rhs, fuser.Fusion) else None,
        out_fusions=out_fusions,
    )
  return inspect.unwrap(gmm_v2.gmm_v2)(
      lhs,
      rhs,
      group_sizes,
      rhs_scale,
      rhs_bias,
      group_offset,
      lhs_scale,
      partial_sum,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      precision=precision,
      preferred_element_type=preferred_element_type,
      acc_dtype=acc_dtype,
      maybe_quantize_lhs=maybe_quantize_lhs,
      lhs_quant_dtype=lhs_quant_dtype,
      rhs_quant_dtype=rhs_quant_dtype,
      zero_initialize=zero_initialize,
      fuse_act=fuse_act,
      transpose_rhs=transpose_rhs,
      fusion_spec=fusion_spec,
  )


def gmm_v2_fusible(
    lhs: jax.Array | fuser_adapter.InputFusion,
    rhs: jax.Array | fuser_adapter.InputFusion,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    lhs_scale: jax.Array | None = None,
    partial_sum: jax.Array | None = None,
    *,
    tile_info: gmm_v2.TileSizes | gmm_v2.TileFn = (
        gmm_v2.calculate_tiling_from_request
    ),
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    acc_dtype: jax.typing.DTypeLike | None = None,
    maybe_quantize_lhs: bool = True,
    lhs_quant_dtype: jax.typing.DTypeLike | None = None,
    rhs_quant_dtype: jax.typing.DTypeLike | None = None,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    transpose_rhs: bool = False,
    out_fusions: fuser_adapter.OutputFusion | None = None,
) -> jax.Array:
  """Fusible GMM v2 entrypoint for `@fuser.fuse` / `@op_fusion.fuse` callers.

  Can be called directly on `jax.Array` operands inside `@fuser.fuse` or
  `@op_fusion.fuse` (which binds a `fuser.fusible` primitive over `lhs` and
  `rhs`), or from `Op._fusible_fwd` with pre-extracted `fuser.Fusion` objects.

  Args:
    lhs: Left-hand side `jax.Array` or `fuser.Fusion` prologue.
    rhs: Right-hand side `jax.Array` or `fuser.Fusion` prologue.
    group_sizes: Per-group row counts (`int32[size_lhs_group]`).
    rhs_scale: Optional quantization scale for `rhs`.
    rhs_bias: Optional bias for `rhs`.
    group_offset: Optional starting group index offset.
    lhs_scale: Optional quantization scale for `lhs`.
    partial_sum: Optional partial sum buffer (unsupported when fused).
    tile_info: Tile sizes or tiling callback.
    vmem_limit_bytes: Optional VMEM capacity limit in bytes.
    precision: Unused LAX precision parameter for signature compatibility.
    preferred_element_type: Optional output element dtype.
    acc_dtype: Optional accumulator dtype.
    maybe_quantize_lhs: Whether to dynamically quantize `lhs` when `rhs` is
      quantized.
    lhs_quant_dtype: Optional quantization dtype for `lhs`.
    rhs_quant_dtype: Optional quantization dtype for `rhs`.
    zero_initialize: Whether to zero-initialize unvisited output rows.
    fuse_act: Optional activation name fused with gate/up projection.
    transpose_rhs: Whether `rhs` is transposed in-core.
    out_fusions: Optional output epilogue fusion from `pallas.fuser`, or None.

  Returns:
    The GMM output `jax.Array` (with any prologue/epilogue fusions applied).
  """
  if (
      isinstance(lhs, fuser.Fusion)
      or isinstance(rhs, fuser.Fusion)
      or out_fusions is not None
  ):
    return _run_gmm_v2_fusible(
        lhs,
        rhs,
        group_sizes,
        rhs_scale,
        rhs_bias,
        group_offset,
        lhs_scale,
        partial_sum,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        lhs_quant_dtype=lhs_quant_dtype,
        rhs_quant_dtype=rhs_quant_dtype,
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
        transpose_rhs=transpose_rhs,
        out_fusions=out_fusions,
    )

  def _fusible(
      lhs_f,
      rhs_f,
      group_sizes_f,
      rhs_scale_f,
      rhs_bias_f,
      group_offset_f,
      lhs_scale_f,
      partial_sum_f,
      out_fusions_f,
  ):
    return _run_gmm_v2_fusible(
        lhs_f,
        rhs_f,
        fuser_adapter.materialize(group_sizes_f),
        fuser_adapter.materialize(rhs_scale_f),
        fuser_adapter.materialize(rhs_bias_f),
        fuser_adapter.materialize(group_offset_f),
        fuser_adapter.materialize(lhs_scale_f),
        fuser_adapter.materialize(partial_sum_f),
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        lhs_quant_dtype=lhs_quant_dtype,
        rhs_quant_dtype=rhs_quant_dtype,
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
        transpose_rhs=transpose_rhs,
        out_fusions=out_fusions_f,
    )

  return fuser.fusible(_fusible)(
      lhs,
      rhs,
      group_sizes,
      rhs_scale,
      rhs_bias,
      group_offset,
      lhs_scale,
      partial_sum,
  )


def _run_tgmm_v2_fusible(
    lhs: jax.Array | fuser_adapter.InputFusion,
    rhs: jax.Array | fuser_adapter.InputFusion,
    group_sizes: jax.Array,
    num_actual_groups: int,
    rhs_scale: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    partial_sum: jax.Array | None = None,
    *,
    tile_info: gmm_v2.TileSizes | tgmm_v2.TileTgmmFn = (
        tgmm_v2.calculate_tgmm_tiling_from_request
    ),
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    acc_dtype: jax.typing.DTypeLike | None = None,
    out_fusions: fuser_adapter.OutputFusion | None = None,
) -> jax.Array:
  """Runs unwrapped `tgmm_v2` with an optional `TgmmFusionSpec`."""
  lhs = _unwrap_trivial_input_fusion(lhs)
  rhs = _unwrap_trivial_input_fusion(rhs)
  fusion_spec = None
  if (
      isinstance(lhs, fuser.Fusion)
      or isinstance(rhs, fuser.Fusion)
      or out_fusions is not None
  ):
    fusion_spec = TgmmFusionSpec(
        lhs_fusion=lhs if isinstance(lhs, fuser.Fusion) else None,
        rhs_fusion=rhs if isinstance(rhs, fuser.Fusion) else None,
        out_fusions=out_fusions,
    )
  return inspect.unwrap(tgmm_v2.tgmm_v2)(
      lhs,
      rhs,
      group_sizes,
      num_actual_groups,
      rhs_scale,
      group_offset,
      partial_sum,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      precision=precision,
      preferred_element_type=preferred_element_type,
      acc_dtype=acc_dtype,
      fusion_spec=fusion_spec,
  )


def tgmm_v2_fusible(
    lhs: jax.Array | fuser_adapter.InputFusion,
    rhs: jax.Array | fuser_adapter.InputFusion,
    group_sizes: jax.Array,
    num_actual_groups: int,
    rhs_scale: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    partial_sum: jax.Array | None = None,
    *,
    tile_info: gmm_v2.TileSizes | tgmm_v2.TileTgmmFn = (
        tgmm_v2.calculate_tgmm_tiling_from_request
    ),
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    acc_dtype: jax.typing.DTypeLike | None = None,
    out_fusions: fuser_adapter.OutputFusion | None = None,
) -> jax.Array:
  """Fusible TGMM v2 entrypoint for `@fuser.fuse` / `@op_fusion.fuse` callers.

  Can be called directly on `jax.Array` operands inside `@fuser.fuse` or
  `@op_fusion.fuse` (which binds a `fuser.fusible` primitive over `lhs` and
  `rhs`), or from `Op._fusible_fwd` with pre-extracted `fuser.Fusion` objects.

  Args:
    lhs: Left-hand side `jax.Array` or `fuser.Fusion` prologue.
    rhs: Right-hand side `jax.Array` or `fuser.Fusion` prologue.
    group_sizes: Per-group row counts (`int32[size_lhs_group]`).
    num_actual_groups: Number of local groups (`weight.shape[0]`).
    rhs_scale: Optional per-N scale for `rhs`.
    group_offset: Optional starting group index offset.
    partial_sum: Optional partial sum buffer (unsupported when fused).
    tile_info: Tile sizes or TGMM tiling callback.
    vmem_limit_bytes: Optional VMEM capacity limit in bytes.
    precision: Unused LAX precision parameter for signature compatibility.
    preferred_element_type: Optional output element dtype.
    acc_dtype: Optional accumulator dtype.
    out_fusions: Optional output epilogue fusion from `pallas.fuser`, or None.

  Returns:
    The TGMM output `jax.Array` (with any prologue/epilogue fusions applied).
  """
  if (
      isinstance(lhs, fuser.Fusion)
      or isinstance(rhs, fuser.Fusion)
      or out_fusions is not None
  ):
    return _run_tgmm_v2_fusible(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups,
        rhs_scale,
        group_offset,
        partial_sum,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        acc_dtype=acc_dtype,
        out_fusions=out_fusions,
    )

  def _fusible(
      lhs_f,
      rhs_f,
      group_sizes_f,
      rhs_scale_f,
      group_offset_f,
      partial_sum_f,
      out_fusions_f,
  ):
    return _run_tgmm_v2_fusible(
        lhs_f,
        rhs_f,
        fuser_adapter.materialize(group_sizes_f),
        num_actual_groups,
        fuser_adapter.materialize(rhs_scale_f),
        fuser_adapter.materialize(group_offset_f),
        fuser_adapter.materialize(partial_sum_f),
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        acc_dtype=acc_dtype,
        out_fusions=out_fusions_f,
    )

  return fuser.fusible(_fusible)(
      lhs, rhs, group_sizes, rhs_scale, group_offset, partial_sum
  )


def _make_out_read_spec(
    lhs_spec_3d: pl.BlockSpec, out_spec_2d: pl.BlockSpec
) -> pl.BlockSpec:
  """Builds a 2D BlockSpec for `(m, n)` that does not throw away the last sublane."""
  lhs_spec_2d = _to_2d_bounded_spec(lhs_spec_3d)
  assert lhs_spec_2d.index_map is not None
  assert out_spec_2d.index_map is not None
  lhs_index_map = lhs_spec_2d.index_map
  out_index_map = out_spec_2d.index_map

  def out_read_index_map(n_id, gm_id, _):
    row_slice, _ = lhs_index_map(n_id, gm_id, 0)
    _, col_idx = out_index_map(n_id, gm_id, 0)
    return (row_slice, col_idx)

  return pl.BlockSpec(
      out_spec_2d.block_shape,
      out_read_index_map,
      memory_space=out_spec_2d.memory_space,
      pipeline_mode=out_spec_2d.pipeline_mode,
  )

