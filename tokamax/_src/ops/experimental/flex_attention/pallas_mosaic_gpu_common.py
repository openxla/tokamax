# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Common utilities for Mosaic GPU FlexAttention implementations."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.extend as jex
import jax.numpy as jnp
from tokamax._src import quantization
from tokamax._src import shape as shape_lib
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as attn_common

_WGMMA = plgpu.Layout.WGMMA
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)
_WGMMA_COL = plgpu.Layout.WGMMA.reduce(0)


def prepare_qkv(q, k, v, precision):
  """Dequantizes Q, K and V and casts them to the given precision."""
  q, k, v = map(quantization.as_array, (q, k, v))
  return attn_common.cast_qkv(q, k, v, precision)


@dataclasses.dataclass(frozen=True, slots=True)
class ModValueSpec:
  """Metadata for loading and padding a closed-over value in score/mask mod."""

  spec: pl.BlockSpec
  dim_axes: tuple[int | None, ...]
  is_ki_invariant: bool
  block_q: int
  block_kv: int
  dtype: jnp.dtype


def _is_squeezed(bd: Any) -> bool:
  return bd is None or isinstance(bd, pl.Squeezed)


def _get_block_size(bd: Any) -> int:
  return bd if isinstance(bd, int) else bd.block_size


def _layout_for_axes(axes: tuple[int, ...]) -> Any:
  if not axes:
    return None
  if axes == (0,):
    return _WGMMA_ROW
  if axes == (1,):
    return _WGMMA_COL
  if axes == (0, 1):
    return _WGMMA
  raise NotImplementedError(f"Unsupported active tile axes: {axes}")


def _cast_to_axes_layout(x: jax.Array, axes: tuple[int, ...]) -> jax.Array:
  layout = _layout_for_axes(axes)
  return x if layout is None else plgpu.layout_cast(x, layout)


def _pid_avals(num_pids: int) -> tuple[jax.ShapeDtypeStruct, ...]:
  return (jax.ShapeDtypeStruct((), jnp.int32),) * num_pids


def _shape_for_axes(
    axes: tuple[int, ...], block_q: int, block_kv: int
) -> tuple[int, ...]:
  return tuple(block_q if a == 0 else block_kv for a in axes)


def _broadcast_to_axes(
    val: jax.Array,
    src_axes: tuple[int, ...],
    dst_axes: tuple[int, ...],
    block_q: int,
    block_kv: int,
) -> jax.Array:
  """Broadcasts `val` from `src_axes` to `dst_axes` with Mosaic GPU layout."""
  if src_axes == dst_axes:
    return val
  if src_axes == (1, 0) and dst_axes == (0, 1):
    val = val.T
    src_axes = (0, 1)
  dst_shape = _shape_for_axes(dst_axes, block_q, block_kv)
  bcast_dims = tuple(dst_axes.index(a) for a in src_axes)
  res = lax.broadcast_in_dim(val, dst_shape, bcast_dims)
  return _cast_to_axes_layout(res, dst_axes)


def _aval(v: jex.core.Var | jex.core.Literal) -> jax.core.ShapedArray:
  assert isinstance(v.aval, jax.core.ShapedArray)
  return v.aval


def _is_non_tileable_const_eqn(eqn: jex.core.JaxprEqn) -> bool:
  """Returns True if `eqn` on closed-over constants should run on the host."""
  if eqn.primitive is lax.reshape_p:
    if eqn.params.get("dimensions") is not None:
      return True
    in_non_unit = [d for d in _aval(eqn.invars[0]).shape if d != 1]
    out_non_unit = [d for d in _aval(eqn.outvars[0]).shape if d != 1]
    return in_non_unit != out_non_unit

  non_tileable_prims = (
      lax.transpose_p,
      lax.tile_p,
      lax.rev_p,
      lax.pad_p,
      lax.concatenate_p,
      lax.gather_p,
      lax.slice_p,
      lax.dynamic_slice_p,
      lax.dot_general_p,
      lax.reduce_sum_p,
      lax.reduce_max_p,
      lax.reduce_min_p,
      lax.reduce_prod_p,
      lax.reduce_or_p,
      lax.reduce_and_p,
  )
  return eqn.primitive in non_tileable_prims


def _pre_evaluate_const_subdags(
    jaxpr: jex.core.Jaxpr, values: Sequence[Any]
) -> tuple[jex.core.Jaxpr, tuple[Any, ...]]:
  """Pre-evaluates non-tileable constant-only sub-DAGs outside the kernel."""
  if not jaxpr.constvars:
    return jaxpr, tuple(values)

  const_vars: set[jex.core.Var] = set(jaxpr.constvars)
  const_eqn_idxs: list[int] = []
  for i, eqn in enumerate(jaxpr.eqns):
    if (
        eqn.invars
        and eqn.primitive is not lax.iota_p
        and all(
            isinstance(v, jex.core.Literal) or v in const_vars
            for v in eqn.invars
        )
    ):
      const_vars.update(eqn.outvars)
      const_eqn_idxs.append(i)

  host_idxs = {
      i for i in const_eqn_idxs if _is_non_tileable_const_eqn(jaxpr.eqns[i])
  }
  if not host_idxs:
    return jaxpr, tuple(values)

  # Also evaluate on the host every constant equation that feeds into one.
  needed_vars: set[jex.core.Var] = set()
  for i in reversed(const_eqn_idxs):
    eqn = jaxpr.eqns[i]
    if i in host_idxs or any(v in needed_vars for v in eqn.outvars):
      host_idxs.add(i)
      needed_vars.update(
          v
          for v in eqn.invars
          if isinstance(v, jex.core.Var) and v in const_vars
      )

  host_eqns = [e for i, e in enumerate(jaxpr.eqns) if i in host_idxs]
  kernel_eqns = [e for i, e in enumerate(jaxpr.eqns) if i not in host_idxs]
  kernel_defined: set[jex.core.Var] = set(jaxpr.invars)
  kernel_consumed: list[Any] = []
  for eqn in kernel_eqns:
    kernel_defined.update(eqn.outvars)
    kernel_consumed.extend(eqn.invars)
  kernel_consumed.extend(jaxpr.outvars)

  # Values used by the kernel but not defined in it become the new constants.
  new_constvars = list(
      dict.fromkeys(
          v
          for v in kernel_consumed
          if isinstance(v, jex.core.Var) and v not in kernel_defined
      )
  )
  host_jaxpr = jex.core.Jaxpr(
      constvars=(),
      invars=list(jaxpr.constvars),
      outvars=new_constvars,
      eqns=host_eqns,
  )
  values = jax.core.eval_jaxpr(host_jaxpr, (), *values)
  jaxpr = jaxpr.replace(constvars=new_constvars, eqns=kernel_eqns)
  return jaxpr, tuple(values)


def _compute_var_axes(
    block_jaxpr: jex.core.Jaxpr, num_pids: int
) -> dict[jex.core.Var, tuple[int | None, ...]]:
  """Infers for each variable in `block_jaxpr` which dims map to Q(0) or K(1)."""
  var_axes: dict[jex.core.Var, tuple[int | None, ...]] = {}
  for v in block_jaxpr.invars[:num_pids]:
    var_axes[v] = ()

  scores_invar = block_jaxpr.invars[-1]
  var_axes[scores_invar] = (0, 1)

  out_var = block_jaxpr.outvars[0]
  if isinstance(out_var, jex.core.Var):
    var_axes[out_var] = (0, 1)

  def merge_axes(v: jex.core.Var, new_ax: tuple[int | None, ...]):
    new_ax = tuple(
        None if dim_size == 1 else ax
        for dim_size, ax in zip(_aval(v).shape, new_ax, strict=True)
    )
    if v in var_axes:
      new_ax = tuple(
          b if a is None else a
          for a, b in zip(var_axes[v], new_ax, strict=True)
      )
    var_axes[v] = new_ax

  for eqn in reversed(block_jaxpr.eqns):
    out_v = eqn.outvars[0]
    if out_v not in var_axes:
      if _aval(out_v).ndim == 0:
        var_axes[out_v] = ()
      else:
        continue
    out_ax = var_axes[out_v]
    for in_v in eqn.invars:
      if not isinstance(in_v, jex.core.Var):
        continue
      in_aval = _aval(in_v)
      if in_aval.ndim == 0:
        merge_axes(in_v, ())
      elif eqn.primitive is lax.broadcast_in_dim_p:
        bdims = eqn.params["broadcast_dimensions"]
        merge_axes(in_v, tuple(out_ax[d] for d in bdims))
      elif eqn.primitive is lax.squeeze_p:
        squeezed = set(eqn.params["dimensions"])
        it = iter(out_ax)
        merge_axes(
            in_v,
            tuple(
                None if i in squeezed else next(it) for i in range(in_aval.ndim)
            ),
        )
      elif eqn.primitive is lax.reshape_p:
        out_aval = _aval(out_v)
        in_non_unit = [d for d in in_aval.shape if d != 1]
        out_non_unit = [d for d in out_aval.shape if d != 1]
        active_axes = [a for a in out_ax if a is not None]
        if (
            eqn.params.get("dimensions") is not None
            or in_non_unit != out_non_unit
            or len(in_non_unit) != len(active_axes)
        ):
          raise NotImplementedError(
              f"Unsupported reshape in score_mod/mask_mod: {in_aval.shape} ->"
              f" {out_aval.shape}"
          )
        active = iter(active_axes)
        merge_axes(
            in_v, tuple(next(active) if d != 1 else None for d in in_aval.shape)
        )
      elif eqn.primitive is lax.transpose_p:
        perm = eqn.params["permutation"]
        in_ax: list[int | None] = [None] * in_aval.ndim
        for out_i, in_i in enumerate(perm):
          in_ax[in_i] = out_ax[out_i]
        merge_axes(in_v, tuple(in_ax))
      else:
        merge_axes(in_v, out_ax)

  for v in list(block_jaxpr.constvars) + list(block_jaxpr.invars):
    if v not in var_axes:
      v_aval = _aval(v)
      if v_aval.ndim == 0:
        var_axes[v] = ()
      elif v_aval.ndim == 2:
        merge_axes(v, (0, 1))
      else:
        var_axes[v] = (None,) * v_aval.ndim

  return var_axes


def _make_mgpu_mod_fn(
    block_jaxpr: jex.core.Jaxpr,
    block_consts: Sequence[Any],
    var_axes: dict[jex.core.Var, tuple[int | None, ...]],
    num_pids: int,
    num_values: int,
    block_q: int,
    block_kv: int,
) -> Callable[..., jax.Array]:
  """Builds a block-level function using squeezed Mosaic GPU register layouts."""

  def active_axes(atom: Any) -> tuple[int, ...]:
    if isinstance(atom, jex.core.Literal):
      return ()
    return tuple(a for a in var_axes[atom] if a is not None)

  def mod_fn(pids, scalar_prefetch, value_blocks, scores_block):
    del scalar_prefetch
    env: dict[jex.core.Var, Any] = {}
    for v, c in zip(block_jaxpr.constvars, block_consts, strict=True):
      c_arr = jnp.asarray(c)
      squeeze_dims = tuple(i for i, d in enumerate(c_arr.shape) if d == 1)
      if squeeze_dims:
        c_arr = jnp.squeeze(c_arr, axis=squeeze_dims)
      env[v] = _cast_to_axes_layout(c_arr, active_axes(v))

    for v, pid in zip(block_jaxpr.invars[:num_pids], pids, strict=True):
      env[v] = pid

    value_invars = block_jaxpr.invars[num_pids : num_pids + num_values]
    for v, val_block in zip(value_invars, value_blocks, strict=True):
      env[v] = val_block

    env[block_jaxpr.invars[-1]] = scores_block

    def read(atom: Any) -> Any:
      if isinstance(atom, jex.core.Literal):
        return jnp.asarray(atom.val, dtype=_aval(atom).dtype)
      return env[atom]

    for eqn in block_jaxpr.eqns:
      out_v = eqn.outvars[0]
      if out_v not in var_axes:
        continue
      dst_axes = active_axes(out_v)
      if eqn.primitive is lax.iota_p:
        dtype = eqn.params["dtype"]
        dim = eqn.params["dimension"]
        iota_axis = var_axes[out_v][dim]
        dst_shape = _shape_for_axes(dst_axes, block_q, block_kv)
        if iota_axis is None:
          res = jnp.zeros(dst_shape, dtype=dtype)
          env[out_v] = _cast_to_axes_layout(res, dst_axes)
        else:
          env[out_v] = plgpu.broadcasted_iota(
              dtype,
              dst_shape,
              dst_axes.index(iota_axis),
              layout=_layout_for_axes(dst_axes),
          )
      elif eqn.primitive in (
          lax.broadcast_in_dim_p,
          lax.squeeze_p,
          lax.reshape_p,
          lax.transpose_p,
      ):
        in_v = eqn.invars[0]
        env[out_v] = _broadcast_to_axes(
            read(in_v), active_axes(in_v), dst_axes, block_q, block_kv
        )
      else:
        in_vals = [
            _broadcast_to_axes(
                read(in_v), active_axes(in_v), dst_axes, block_q, block_kv
            )
            for in_v in eqn.invars
        ]
        if (
            eqn.primitive is lax.convert_element_type_p
            and eqn.params["new_dtype"] == jnp.bool_
        ):
          if in_vals[0].dtype == jnp.bool_:
            res = in_vals[0]
          else:
            res = in_vals[0] != 0
        else:
          res = eqn.primitive.bind(*in_vals, **eqn.params)
        env[out_v] = _cast_to_axes_layout(res, dst_axes)

    out_var = block_jaxpr.outvars[0]
    out_val = _broadcast_to_axes(
        read(out_var), active_axes(out_var), (0, 1), block_q, block_kv
    )
    return plgpu.layout_cast(out_val, _WGMMA)

  return mod_fn


def _check_is_ki_invariant(
    val_shape: tuple[int, ...],
    spec: pl.BlockSpec,
    dim_axes: tuple[int | None, ...],
    num_pids: int,
) -> bool:
  """Returns True if loading `val` does not depend on the last pid (`ki`)."""

  def used_indices_fn(*pids):
    assert spec.index_map is not None
    b_indices = spec.index_map(*pids)
    return tuple(
        0 if (s == 1 and ax is None) else b_idx
        for s, ax, b_idx in zip(val_shape, dim_axes, b_indices, strict=True)
    )

  idx_jaxpr = jax.make_jaxpr(used_indices_fn)(*_pid_avals(num_pids)).jaxpr
  ki_dependent: set[jex.core.Var] = {idx_jaxpr.invars[-1]}
  for eqn in idx_jaxpr.eqns:
    if any(
        isinstance(v, jex.core.Var) and v in ki_dependent for v in eqn.invars
    ):
      ki_dependent.update(eqn.outvars)
  return not any(
      isinstance(v, jex.core.Var) and v in ki_dependent
      for v in idx_jaxpr.outvars
  )


def tile_score_mod(
    score_mod: Callable[[jax.Array], jax.Array] | None,
    scores_shape: tuple[int, ...],
    block_q: int,
    block_kv: int,
    *,
    out_dtype: jax.typing.DTypeLike = jnp.float32,
) -> tuple[
    Callable[..., jax.Array] | None,
    tuple[jax.Array, ...],
    tuple[ModValueSpec, ...],
]:
  """Tiles `score_mod` and canonicalizes it for Mosaic GPU registers."""
  if score_mod is None:
    return None, (), ()

  scores = jax.ShapeDtypeStruct(scores_shape, jnp.float32)

  def wrapped_score_mod(s):
    return jnp.broadcast_to(score_mod(s).astype(out_dtype), s.shape)

  closed_jaxpr = jax.make_jaxpr(wrapped_score_mod)(scores)
  jaxpr, const_values = _pre_evaluate_const_subdags(
      closed_jaxpr.jaxpr, closed_jaxpr.consts
  )

  def fn_with_consts(s):
    return jax.core.eval_jaxpr(jaxpr, const_values, s)[0]

  fn, raw_values, prefetch = fuser.get_fusion_values(fn_with_consts, scores)
  if prefetch:
    raise NotImplementedError(
        "Scalar prefetch in score_mod/mask_mod is not supported."
    )
  values: tuple[jax.Array, ...] = tuple(jnp.asarray(v) for v in raw_values)

  block_shape = (None,) * (scores.ndim - 2) + (block_q, block_kv)
  spec = pl.BlockSpec(block_shape, lambda *pids: pids)
  num_blocks_q = pl.cdiv(scores.shape[-2], block_q)
  num_blocks_k = pl.cdiv(scores.shape[-1], block_kv)
  grid = (*scores.shape[:-2], num_blocks_q, num_blocks_k)
  fn_spec_puller = fuser.pull_block_spec(fn, spec, grid_len=len(grid))
  block_fn, (value_specs, _), _ = fn_spec_puller(values, scores)

  num_pids = scores.ndim
  num_values = len(values)
  value_blocks_aval = tuple(
      jax.ShapeDtypeStruct(
          tuple(
              _get_block_size(bd)
              for bd in v_spec.block_shape
              if not _is_squeezed(bd)
          ),
          v.dtype,
      )
      for v, v_spec in zip(values, value_specs, strict=True)
  )
  scores_block_aval = jax.ShapeDtypeStruct((block_q, block_kv), jnp.float32)
  closed_block_jaxpr = jax.make_jaxpr(block_fn)(
      _pid_avals(num_pids), (), value_blocks_aval, scores_block_aval
  )
  block_jaxpr = closed_block_jaxpr.jaxpr
  block_consts = closed_block_jaxpr.consts

  var_axes = _compute_var_axes(block_jaxpr, num_pids)
  value_invars = block_jaxpr.invars[num_pids : num_pids + num_values]

  mod_value_specs: list[ModValueSpec] = []
  for v, v_spec, v_invar in zip(values, value_specs, value_invars, strict=True):
    invar_axes = iter(var_axes[v_invar])
    dim_axes_list: list[int | None] = []
    for dim_size, bd in zip(v.shape, v_spec.block_shape, strict=True):
      if _is_squeezed(bd):
        dim_axes_list.append(None)
      else:
        ax = next(invar_axes)
        if dim_size == 1 or _get_block_size(bd) == 1:
          dim_axes_list.append(None)
        else:
          dim_axes_list.append(ax)
    dim_axes = tuple(dim_axes_list)
    is_ki_invariant = _check_is_ki_invariant(
        v.shape, v_spec, dim_axes, num_pids
    )
    mod_value_specs.append(
        ModValueSpec(
            spec=v_spec,
            dim_axes=dim_axes,
            is_ki_invariant=is_ki_invariant,
            block_q=block_q,
            block_kv=block_kv,
            dtype=v.dtype,
        )
    )

  mgpu_mod_fn = _make_mgpu_mod_fn(
      block_jaxpr,
      block_consts,
      var_axes,
      num_pids,
      num_values,
      block_q,
      block_kv,
  )
  return mgpu_mod_fn, values, tuple(mod_value_specs)


def tile_mask_mod(
    mask_mod: Callable[[tuple[int, ...]], jax.Array] | None,
    scores_shape: tuple[int, ...],
    block_q: int,
    block_kv: int,
) -> tuple[
    Callable[..., jax.Array] | None,
    tuple[jax.Array, ...],
    tuple[ModValueSpec, ...],
]:
  """Tiles `mask_mod` and canonicalizes it for Mosaic GPU registers."""
  if mask_mod is None:
    return None, (), ()

  return tile_score_mod(
      lambda s: mask_mod(s.shape),
      scores_shape,
      block_q,
      block_kv,
      out_dtype=jnp.bool_,
  )


def pad_mod_values(
    values: tuple[jax.Array, ...],
    specs: tuple[ModValueSpec, ...],
    pad_block_q: int,
    pad_block_kv: int,
) -> tuple[jax.Array, ...]:
  """Pads closed-over values along Q and K tiled dimensions."""
  padded_values = []
  for val, mod_spec in zip(values, specs, strict=True):
    for d, axis in enumerate(mod_spec.dim_axes):
      if axis == 0:
        val = shape_lib.pad_to_next_multiple_of(val, pad_block_q, d)
      elif axis == 1:
        val = shape_lib.pad_to_next_multiple_of(val, pad_block_kv, d)
    if val.dtype == jnp.bool_:
      val = val.astype(jnp.int8)
    padded_values.append(val)
  return tuple(padded_values)


def load_mod_value(
    ref: Any,
    mod_spec: ModValueSpec,
    mod_fn_pids: tuple[int | jax.Array, ...],
) -> jax.Array:
  """Loads a block or scalar of a closed-over value from GMEM."""
  assert mod_spec.spec.index_map is not None
  b_indices = mod_spec.spec.index_map(*mod_fn_pids)
  idx = []
  active_axes = []
  for dim_size, b_idx, axis in zip(
      ref.shape, b_indices, mod_spec.dim_axes, strict=True
  ):
    if axis is None:
      idx.append(0 if dim_size == 1 else b_idx)
    else:
      block_size = mod_spec.block_q if axis == 0 else mod_spec.block_kv
      idx.append(pl.ds(b_idx * block_size, block_size))
      active_axes.append(axis)
  idx_tuple = tuple(idx)
  layout = _layout_for_axes(tuple(active_axes))
  if layout is None:
    res = ref[idx_tuple]
  else:
    res = plgpu.load(ref.at[idx_tuple], layout=layout, optimized=False)
  if mod_spec.dtype == jnp.bool_:
    return res != 0
  return res.astype(mod_spec.dtype)


def preload_mod_values(
    refs: Sequence[Any],
    specs: Sequence[ModValueSpec],
    mod_fn_pids: tuple[int | jax.Array, ...],
) -> tuple[jax.Array | None, ...]:
  """Loads the closed-over values that don't depend on `ki`."""
  return tuple(
      load_mod_value(ref, spec, mod_fn_pids) if spec.is_ki_invariant else None
      for ref, spec in zip(refs, specs, strict=True)
  )


def load_mod_values(
    preloaded: Sequence[jax.Array | None],
    refs: Sequence[Any],
    specs: Sequence[ModValueSpec],
    mod_fn_pids: tuple[int | jax.Array, ...],
) -> tuple[jax.Array, ...]:
  """Loads the closed-over values, reusing those from `preload_mod_values`."""
  return tuple(
      load_mod_value(ref, spec, mod_fn_pids) if pre is None else pre
      for pre, ref, spec in zip(preloaded, refs, specs, strict=True)
  )
