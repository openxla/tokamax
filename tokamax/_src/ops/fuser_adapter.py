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
"""Adapter and helpers for JAX Pallas `fuser` integration (`@op_fusion.fuse`)."""

from collections.abc import Callable, Iterable, Sequence
import contextlib
import functools
import inspect
from typing import Any, ClassVar, Protocol, cast, overload

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
import jax.numpy as jnp
import numpy as np
from tokamax._src import utils

type PyTree = Any
type InputFusion = fuser.Fusion[[], jax.Array]
type OutputFusion = fuser.Fusion[[jax.Array], tuple[jax.Array]]
type ExtractedInputFn = Callable[[tuple[jax.Array, ...]], jax.Array]
type ExtractedOutputFn = Callable[
    [tuple[jax.Array, ...], jax.Array], tuple[jax.Array]
]
type PrologueFn = Callable[
    [tuple[int | jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...]],
    jax.Array,
]
type EpilogueFn = Callable[
    [
        tuple[int | jax.Array, ...],
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array,
    ],
    jax.Array,
]


class _FusibleOp[_T, _C](Protocol):
  """Protocol satisfied by any Tokamax `Op` supporting `@op_fusion.fuse`."""

  fusible_argnames: ClassVar[tuple[str, ...] | None]

  def _fusible_fwd(
      self,
      *args: Any,
      out_fusions: OutputFusion | None = None,
      config: _C,
      **kwargs: Any,
  ) -> _T:
    ...


class _BoundArgs[_C](Protocol):
  """Protocol satisfied by `op.BoundArguments` inside `call_fusible_op`."""

  @property
  def default_config(self) -> _C:
    ...

  @property
  def signature(self) -> inspect.Signature:
    ...

  @property
  def args(self) -> tuple[Any, ...]:
    ...

  @property
  def kwargs(self) -> dict[str, Any]:
    ...

  def _tokamax_metadata(
      self, config: _C
  ) -> contextlib.AbstractContextManager[None]:
    ...


_IN_OP_FUSION_CONTEXT = jax.make_user_context(False)


@overload
def fuse[_F: Callable[..., Any]](
    func: _F,
    *,
    resolve_fusion_dtypes: bool = True,
    debug: bool = False,
    strict_mode: bool = True,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
) -> _F:
  ...


@overload
def fuse(
    func: None = None,
    *,
    resolve_fusion_dtypes: bool = True,
    debug: bool = False,
    strict_mode: bool = True,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  ...


def fuse(
    func: Callable[..., Any] | None = None,
    *,
    resolve_fusion_dtypes: bool = True,
    debug: bool = False,
    strict_mode: bool = True,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
) -> Callable[..., Any]:
  """Tokamax wrapper around `jax.experimental.pallas.fuser.fuse`.

  Args:
    func: Callable to trace with Pallas fuser, or None to return a decorator.
    resolve_fusion_dtypes: Whether to resolve fusion dtypes automatically.
    debug: Whether to print fusion debugging output.
    strict_mode: Whether to enforce fuser strict mode.
    static_argnums: Indices of static positional arguments.
    static_argnames: Names of static keyword arguments.

  Returns:
    The wrapped callable when `func` is given, or a decorator when `func` is
    None.
  """

  def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
      with _IN_OP_FUSION_CONTEXT(True):
        return fn(*args, **kwargs)

    return fuser.fuse(
        wrapped,
        resolve_fusion_dtypes=resolve_fusion_dtypes,
        debug=debug,
        strict_mode=strict_mode,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )

  return decorator(func) if func is not None else decorator


def in_op_fusion_context() -> bool:
  """Returns whether execution is currently inside `@op_fusion.fuse` tracing."""
  return bool(_IN_OP_FUSION_CONTEXT.value)


def _is_jax_array(x: object) -> bool:
  """Returns whether `x` is a concrete or traced JAX/NumPy array."""
  return isinstance(x, (jax.Array, np.ndarray))


def _is_jax_array_wrapper(x: object) -> bool:
  """Returns whether `x` is a non-array wrapper implementing `__jax_array__`."""
  return not _is_jax_array(x) and hasattr(x, "__jax_array__")


def _unwrap_jax_array_wrappers(pytree: PyTree) -> PyTree:
  """Converts PyTree wrappers with `__jax_array__` (e.g. `GroupSizes`)."""
  return jax.tree.map(
      lambda x: jnp.asarray(x) if _is_jax_array_wrapper(x) else x,
      pytree,
      is_leaf=_is_jax_array_wrapper,
  )


def call_fusible_op[_T, _C](
    op_instance: _FusibleOp[_T, _C],
    bound_args: _BoundArgs[_C],
    return_residuals: bool = False,
) -> _T:
  """Executes `op_instance` as a `@fuser.fusible` primitive.

  Args:
    op_instance: The Tokamax `Op` instance invoked inside `@op_fusion.fuse`.
    bound_args: `BoundArguments` capturing the op's positional/keyword inputs.
    return_residuals: Whether residuals were requested by the caller.

  Returns:
    The result of tracing `op_instance._fusible_fwd` through `fuser.fusible`.

  Raises:
    NotImplementedError: If `return_residuals` is True.
  """
  if return_residuals:
    raise NotImplementedError(
        "`return_residuals=True` is not supported in fusible mode."
    )

  with _IN_OP_FUSION_CONTEXT(False):
    config = bound_args.default_config
  args, kwargs = _unwrap_jax_array_wrappers(
      (bound_args.args, bound_args.kwargs)
  )
  args_flat, args_tree = jax.tree.flatten((args, kwargs))
  arrays, other, merge = utils.split_merge(_is_jax_array, args_flat)

  def raw_fusible_call(*in_fusions_and_out_fusions: Any) -> _T:
    *in_fusions, out_fusions = in_fusions_and_out_fusions
    with _IN_OP_FUSION_CONTEXT(False):
      f_args, f_kwargs = args_tree.unflatten(merge(in_fusions, other))
      if op_instance.fusible_argnames is not None:
        # Evaluate non-fusible inputs (e.g. `group_sizes`) outside the kernel
        # so `_fusible_fwd` receives regular `jax.Array`s for them.
        fusible_names = set(op_instance.fusible_argnames)
        sig_ba = bound_args.signature.bind_partial(*f_args, **f_kwargs)
        for arg_name, arg_val in sig_ba.arguments.items():
          if arg_name not in fusible_names:
            sig_ba.arguments[arg_name] = materialize(arg_val)
        f_args, f_kwargs = sig_ba.args, sig_ba.kwargs
      f_kwargs.pop("return_residuals", None)
      # pylint: disable-next=protected-access
      with bound_args._tokamax_metadata(config):
        return op_instance._fusible_fwd(  # pylint: disable=protected-access
            *f_args, out_fusions=out_fusions, config=config, **f_kwargs
        )

  fusible_call = fuser.fusible(
      raw_fusible_call,
      output_fusion_prefix=getattr(op_instance, "output_fusion_prefix", True),
  )
  return fusible_call(*arrays)


def materialize(pytree: PyTree) -> PyTree:
  """Evaluates `fuser.Fusion` leaves in `pytree` into HBM `jax.Array`s.

  `fuser.fusible` wraps every `jax.Array` leaf of every op argument in a
  `fuser.Fusion` closure. Unpacking a `Fusion` fuses its upstream ops into the
  kernel, whereas calling `x()` traces them outside the kernel to produce a
  regular `jax.Array`. Use `materialize` for inputs that should not be fused
  into the kernel, such as metadata arguments excluded by `Op.fusible_argnames`
  (e.g. `group_sizes`) or PyTree variants like pre-quantized `QArray(qvalue,
  scale)` inside `_fusible_fwd`.

  Args:
    pytree: Arbitrary PyTree whose leaves may contain `fuser.Fusion` closures.

  Returns:
    A PyTree with the same structure where every `fuser.Fusion` leaf is
    evaluated into a `jax.Array`.
  """
  return jax.tree.map(
      lambda x: x() if isinstance(x, fuser.Fusion) else x,
      pytree,
      is_leaf=lambda x: isinstance(x, fuser.Fusion),
  )


def unpack_input_fusion[_T](
    x: _T | InputFusion,
) -> (
    tuple[None, _T, None]
    | tuple[
        ExtractedInputFn,
        tuple[jax.Array, ...],
        tuple[jax.ShapeDtypeStruct, ...],
    ]
):
  """Unpacks an optional input `fuser.Fusion` object.

  Args:
    x: Either a `fuser.Fusion` input prologue or an unfused array/PyTree.

  Returns:
    A 3-tuple `(fn, values, values_type)`. When `x` is a `fuser.Fusion`, `fn` is
    the prologue callable, `values` are the captured input arrays, and
    `values_type` holds their `ShapeDtypeStruct` descriptors; otherwise `fn` and
    `values_type` are None and `values` is `x` unchanged.
  """
  if not isinstance(x, fuser.Fusion):
    return None, x, None
  raw_fn, raw_values, _ = fuser.get_fusion_values(x)
  fn = cast(ExtractedInputFn, raw_fn)
  values = cast(tuple[jax.Array, ...], raw_values)
  values_type = tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values)
  return fn, values, values_type


def unpack_output_fusion(
    out_fusions: OutputFusion | None,
    raw_out_type: jax.ShapeDtypeStruct,
) -> (
    tuple[None, None, None, jax.ShapeDtypeStruct]
    | tuple[
        ExtractedOutputFn,
        tuple[jax.Array, ...],
        tuple[jax.ShapeDtypeStruct, ...],
        jax.ShapeDtypeStruct,
    ]
):
  """Unpacks an optional single-output `out_fusions` tuple.

  Args:
    out_fusions: Output fusion specification passed by `fuser.fusible`, or None
      when no output epilogue is fused.
    raw_out_type: Unfused kernel output `ShapeDtypeStruct`.

  Returns:
    A 4-tuple `(out_fn, out_values, out_values_type, out_init)`. When
    `out_fusions` is None, `out_fn`, `out_values`, and `out_values_type` are
    None and `out_init` equals `raw_out_type`.

  Raises:
    NotImplementedError: If the epilogue produces multiple outputs.
  """
  if out_fusions is None:
    return None, None, None, raw_out_type
  raw_out_fn, raw_out_values, _ = fuser.get_fusion_values(
      out_fusions, raw_out_type
  )
  out_fn = cast(ExtractedOutputFn, raw_out_fn)
  out_values = cast(tuple[jax.Array, ...], raw_out_values)
  out_values_type = tuple(
      jax.ShapeDtypeStruct(v.shape, v.dtype) for v in out_values
  )
  raw_out_init = jax.eval_shape(out_fn, out_values_type, raw_out_type)
  if isinstance(raw_out_init, tuple):
    if len(raw_out_init) != 1:
      raise NotImplementedError(
          "Multi-output epilogue fusion is not supported"
          f" (got {len(raw_out_init)} outputs)."
      )
    raw_out_init = raw_out_init[0]
  out_init = cast(jax.ShapeDtypeStruct, raw_out_init)
  return out_fn, out_values, out_values_type, out_init


def pull_input_spec(
    fn: ExtractedInputFn,
    spec: pl.BlockSpec,
    values_type: tuple[jax.ShapeDtypeStruct, ...],
    *,
    grid_len: int,
    num_scalar_prefetch: int = 0,
) -> tuple[PrologueFn, tuple[pl.BlockSpec, ...]]:
  """Pulls `spec` backward through an input prologue `fn`.

  Args:
    fn: Prologue function extracted by `unpack_input_fusion`.
    spec: Consumer `BlockSpec` for the prologue's output tile.
    values_type: `ShapeDtypeStruct` descriptors for the prologue's inputs.
    grid_len: Number of grid dimensions in the enclosing Pallas kernel.
    num_scalar_prefetch: Number of scalar prefetch arguments in the kernel.

  Returns:
    A 2-tuple `(pulled_fn, val_specs)` containing the tile-level prologue
    callable and the pulled `BlockSpec`s for each captured input array.
  """
  pulled_fn, (val_specs,), _ = fuser.pull_block_spec(
      fn,
      spec,
      scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(
          num_scalar_prefetch
      ),
      grid_len=grid_len,
      strict_mode=False,
  )(values_type)
  return pulled_fn, val_specs


def _canonical_block_dim(dim: pl.BlockDim | int | None) -> pl.BlockDim:
  if dim is None:
    return pl.squeezed
  if isinstance(dim, int):
    return pl.Blocked(dim)
  return dim


def _block_specs_compatible(
    expected: pl.BlockSpec, actual: pl.BlockSpec | Any
) -> bool:
  """Returns whether `actual` matches `expected` modulo `index_map`."""
  if actual is pl.no_block_spec or not isinstance(actual, pl.BlockSpec):
    return False
  if (expected.block_shape is None) != (actual.block_shape is None):
    return False
  if expected.block_shape is not None and actual.block_shape is not None:
    if tuple(map(_canonical_block_dim, expected.block_shape)) != tuple(
        map(_canonical_block_dim, actual.block_shape)
    ):
      return False
  return (
      expected.memory_space == actual.memory_space
      and expected.pipeline_mode == actual.pipeline_mode
  )


def _full_blocked_axes(
    block_shape: Sequence[pl.BlockDim | int | None] | None,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
  if block_shape is None or len(block_shape) != len(shape):
    return ()
  axes = []
  for i, (dim, size) in enumerate(zip(block_shape, shape, strict=True)):
    canon = _canonical_block_dim(dim)
    if isinstance(canon, pl.Blocked) and canon.block_size == size:
      axes.append(i)
  return tuple(axes)


def _pull_shape_preserving_epilogue(
    out_fn: ExtractedOutputFn,
    out_spec: pl.BlockSpec,
    out_values_type: tuple[jax.ShapeDtypeStruct, ...],
    raw_out_type: jax.ShapeDtypeStruct,
    *,
    out_shape: tuple[int, ...],
    prefetch_handler: Any,
    grid_len: int,
) -> tuple[Any, tuple[pl.BlockSpec, ...], pl.BlockSpec]:
  """Pulls `out_spec` through a shape-preserving epilogue in strict mode."""
  out_specs = (out_spec,)
  try:
    raw_pulled_fn, (out_values_specs, pulled_raw_spec), _ = (
        fuser.pull_block_spec(
            out_fn,
            out_specs,
            scalar_prefetch_handler=prefetch_handler,
            grid_len=grid_len,
            strict_mode=True,
        )(out_values_type, raw_out_type)
    )
    return raw_pulled_fn, out_values_specs, pulled_raw_spec
  except ValueError as e:
    full_axes = _full_blocked_axes(out_spec.block_shape, out_shape)
    if (
        "Fusion contains a DAG" not in str(e)
        or not full_axes
        or out_spec.block_shape is None
    ):
      raise
    # Older JAX releases compare single-block full dimensions with symbolic
    # indices rather than 0 during strict DAG checks; reducing `full_axes`
    # pins those indices to 0 while keeping strict checks on tiled axes.
    full_axes_set = set(full_axes)
    reduced_block_shape = tuple(
        d
        for i, d in enumerate(out_spec.block_shape)
        if i not in full_axes_set
    )
    index_map = out_spec.index_map
    reduced_index_map = None
    if index_map is not None:
      reduced_index_map = lambda *args: tuple(
          idx
          for i, idx in enumerate(index_map(*args))
          if i not in full_axes_set
      )
    reduced_spec = pl.BlockSpec(
        reduced_block_shape,
        reduced_index_map,
        memory_space=out_spec.memory_space,
        pipeline_mode=out_spec.pipeline_mode,
    )

    def reduced_out_fn(vals: Any, raw_out: Any) -> tuple[jax.Array, ...]:
      out = out_fn(vals, raw_out)
      y = out[0] if isinstance(out, tuple) else out
      return (jax.lax.reduce_sum(y.astype(jnp.float32), axes=full_axes),)

    fuser.pull_block_spec(
        reduced_out_fn,
        (reduced_spec,),
        scalar_prefetch_handler=prefetch_handler,
        grid_len=grid_len,
        strict_mode=True,
    )(out_values_type, raw_out_type)
    raw_pulled_fn, (out_values_specs, pulled_raw_spec), _ = (
        fuser.pull_block_spec(
            out_fn,
            out_specs,
            scalar_prefetch_handler=prefetch_handler,
            grid_len=grid_len,
            strict_mode=False,
        )(out_values_type, raw_out_type)
    )
    return raw_pulled_fn, out_values_specs, pulled_raw_spec


def push_pull_block_spec(
    out_fn: ExtractedOutputFn,
    out_spec: pl.BlockSpec,
    out_values_type: tuple[jax.ShapeDtypeStruct, ...],
    raw_out_type: jax.ShapeDtypeStruct,
    *,
    grid_len: int,
    num_scalar_prefetch: int = 0,
) -> tuple[EpilogueFn, tuple[pl.BlockSpec, ...], pl.BlockSpec]:
  """Pushes and pulls `out_spec` through a single-output epilogue `out_fn`.

  Args:
    out_fn: Epilogue function extracted by `unpack_output_fusion`.
    out_spec: Unfused kernel output `BlockSpec`.
    out_values_type: `ShapeDtypeStruct` descriptors for epilogue side inputs.
    raw_out_type: Unfused kernel output `ShapeDtypeStruct`.
    grid_len: Number of grid dimensions in the enclosing Pallas kernel.
    num_scalar_prefetch: Number of scalar prefetch arguments in the kernel.

  Returns:
    A 3-tuple `(pulled_out_fn, out_values_specs, final_out_spec)` containing the
    unwrapped tile-level epilogue callable, pulled `BlockSpec`s for epilogue
    side inputs, and the pushed/pulled kernel output `BlockSpec`.

  Raises:
    ValueError: If `out_fn` produces inconsistent or incompatible `BlockSpec`s.
  """
  prefetch_handler = fuser.make_scalar_prefetch_handler(num_scalar_prefetch)
  values_block_specs = jax.tree.map(lambda _: pl.no_block_spec, out_values_type)
  try:
    out_specs = fuser.push_block_spec(
        out_fn,
        values_block_specs,
        out_spec,
        scalar_prefetch_handler=prefetch_handler,
        grid_len=grid_len,
    )(out_values_type, raw_out_type)
  except ValueError as e:
    out_type = jax.eval_shape(out_fn, out_values_type, raw_out_type)
    single_out_type = out_type[0] if isinstance(out_type, tuple) else out_type
    if (
        "Illegal binary push" not in str(e)
        or single_out_type.shape != raw_out_type.shape
    ):
      raise
    # JAX's `_binop_push_rule` rejects implicit size-1 broadcasts when both
    # operands carry `BlockSpec`s (e.g. RMSNorm's `x * rsqrt(mean(x**2))`).
    out_specs = (out_spec,)
    raw_pulled_fn, out_values_specs, pulled_raw_spec = (
        _pull_shape_preserving_epilogue(
            out_fn,
            out_spec,
            out_values_type,
            raw_out_type,
            out_shape=single_out_type.shape,
            prefetch_handler=prefetch_handler,
            grid_len=grid_len,
        )
    )
  else:
    raw_pulled_fn, (out_values_specs, pulled_raw_spec), _ = (
        fuser.pull_block_spec(
            out_fn,
            out_specs,
            scalar_prefetch_handler=prefetch_handler,
            grid_len=grid_len,
            strict_mode=False,
        )(out_values_type, raw_out_type)
    )

  if not _block_specs_compatible(out_spec, pulled_raw_spec):
    raise ValueError(
        "Inconsistent block specs in push_pull_block_spec: the fusion uses"
        f" kernel output with {pulled_raw_spec}, but the kernel writes it"
        f" with {out_spec}."
    )

  final_out_spec = (
      out_specs[0] if isinstance(out_specs, tuple) else out_specs
  )

  def pulled_out_fn(
      pids: tuple[int | jax.Array, ...],
      scalar_prefetch: tuple[jax.Array, ...],
      out_vals: tuple[jax.Array, ...],
      raw_out: jax.Array,
  ) -> jax.Array:
    pulled_output = raw_pulled_fn(pids, scalar_prefetch, out_vals, raw_out)
    return (
        pulled_output[0]
        if isinstance(pulled_output, tuple)
        else pulled_output
    )

  return pulled_out_fn, out_values_specs, final_out_spec
