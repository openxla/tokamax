## Operator Fusion

`tokamax.ops.op_fusion` lets you fuse surrounding JAX operations (prologues and
epilogues) directly into a fusible Tokamax Pallas TPU kernel via
[JAX Pallas Fuser](https://docs.jax.dev/en/latest/pallas/tpu/fuser.html).
Instead of materializing intermediate activations to HBM between a normalization
or activation and a custom kernel, the fused operations run in VMEM on each tile
inside the kernel loop.

### Using `@op_fusion.fuse`

#### Basic Usage (Prologue and Epilogue)

Import `op_fusion` from `tokamax.ops` and wrap the function containing a fusible
Tokamax operator with `@op_fusion.fuse` (typically inside `@jax.jit`):

```python
import jax
import jax.numpy as jnp
import tokamax
from tokamax.ops import op_fusion


def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
  ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
  return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps)).astype(
      x.dtype
  ) * weight


@jax.jit
@op_fusion.fuse
def fused_ragged_dot(x, rms_weight, rhs, out_bias, group_sizes):
  # Prologue: fused into the kernel's LHS tile load.
  x_norm = rms_norm(x, rms_weight)
  out = tokamax.ragged_dot(
      x_norm,
      rhs,
      group_sizes=group_sizes,
      implementation="mosaic_tpu_v2",
  )
  # Epilogue: fused into the kernel's output store.
  return jax.nn.silu(out + out_bias)
```

You can call either the high-level functional API with an explicit fusible
implementation (`tokamax.ragged_dot(..., implementation="mosaic_tpu_v2")`) or a
specific `tokamax.Op` instance directly (such as
[`PallasMosaicTpuV2RaggedDot`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py)).
If an operator inside `@op_fusion.fuse` does not support fusion, Tokamax raises
`NotImplementedError`.

#### Decorator Options

`op_fusion.fuse` forwards its keyword arguments to
`jax.experimental.pallas.fuser.fuse`:

```python
@op_fusion.fuse(
    resolve_fusion_dtypes=True,
    debug=False,
    strict_mode=True,
    static_argnums=None,
    static_argnames=None,
)
```

*   `static_argnums` / `static_argnames`: Mark non-array Python arguments (for
    example, scalar config values or boolean flags) as static when calling a
    fused function.
*   `debug=True`: Prints the traced fusion jaxprs to help diagnose why an op did
    not fuse or failed block-spec propagation.
*   `strict_mode`: Controls Pallas Fuser's strict block-spec equivalence checks.

#### Quantized Inputs and Built-In Kernel Activations

Fused prologues compose with on-the-fly activation quantization (`AsQArray`),
pre-quantized weights (`qwix.QArray`), and built-in fused gate/up activations on
`PallasMosaicTpuV2RaggedDot`:

```python
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2
from tokamax.ops import op_fusion

gmm_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()


@op_fusion.fuse
def fused_rms_norm_gate_up(x, rms_weight, w_q, w_scale, group_sizes):
  x_norm = rms_norm(x, rms_weight)
  return gmm_op(
      quantization.AsQArray(x_norm, jnp.float8_e4m3fn),
      qwix.QArray(w_q, w_scale),
      group_sizes=group_sizes,
      zero_initialize=False,
      fuse_gateup_activation="silu",
      preferred_element_type=jnp.bfloat16,
  )
```

#### One Fusible Op per `@op_fusion.fuse` Block

JAX Pallas Fuser supports a single fusible kernel per `@op_fusion.fuse` scope.
When chaining multiple kernels (for example, `gate_up` followed by `down` in an
MoE block), wrap each fusible stage in its own `@op_fusion.fuse` function inside
an outer `@jax.jit`:

```python
@jax.jit
def moe_layer(x, rms_weight, w1_q, w1_scale, w2_q, w2_scale, group_sizes):
  mid = fused_rms_norm_gate_up(x, rms_weight, w1_q, w1_scale, group_sizes)
  return gmm_op(
      quantization.AsQArray(mid, jnp.float8_e4m3fn),
      qwix.QArray(w2_q, w2_scale),
      group_sizes=group_sizes,
      zero_initialize=False,
      preferred_element_type=jnp.bfloat16,
  )
```

#### Automatic Differentiation with Explicit `@jax.custom_vjp`

`@op_fusion.fuse` fuses the forward computation block and does not automatically
synthesize fused backward passes through `jax.grad`. When differentiating a
fused block, wrap the forward and backward calls in an explicit
`@jax.custom_vjp` so you can fuse prologues/epilogues into the forward and
backward (`drhs` / `dlhs`) kernels independently:

```python
import jax
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2
from tokamax.ops import op_fusion

op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()
drhs_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(
    num_actual_groups=num_groups
)


@op_fusion.fuse
def fused_fwd_kernel(x, w, r, group_sizes):
  return op(rms_norm(x, w), r, group_sizes=group_sizes, zero_initialize=True)


@op_fusion.fuse
def fused_drhs_kernel(x, w, dy, group_sizes):
  return drhs_op(
      rms_norm(x, w),
      dy,
      group_sizes=group_sizes,
      ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
      preferred_element_type=dy.dtype,
  )


@jax.custom_vjp
def fused_gmm(x, w, r, group_sizes):
  return fused_fwd_kernel(x, w, r, group_sizes)


def fused_gmm_fwd(x, w, r, group_sizes):
  return fused_fwd_kernel(x, w, r, group_sizes), (x, w, r, group_sizes)


def fused_gmm_bwd(residuals, dy):
  x, w, r, group_sizes = residuals
  dr = fused_drhs_kernel(x, w, dy, group_sizes)
  dx_norm = op(
      dy,
      r,
      group_sizes=group_sizes,
      ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DLHS_RAGGED_DOT_DIM_NUMS,
      zero_initialize=True,
  )
  _, rms_vjp = jax.vjp(rms_norm, x, w)
  dx, dw = rms_vjp(dx_norm)
  return dx, dw, dr, None


fused_gmm.defvjp(fused_gmm_fwd, fused_gmm_bwd)
```

#### Using Bare Fusible Kernels

Prefer `@op_fusion.fuse` with Tokamax operator wrappers (`tokamax.ragged_dot`
or `PallasMosaicTpuV2RaggedDot`) over bare fusible kernels and
`@jax.experimental.pallas.fuser.fuse` to maintain backward compatibility and
retain input canonicalization, autotuning, and metadata serialization. If you
invoke low-level `gmm_v2` / `tgmm_v2` kernels directly, use
`gmm_fuser.gmm_v2_fusible` or `gmm_fuser.tgmm_v2_fusible` inside either
`@op_fusion.fuse` or `@jax.experimental.pallas.fuser.fuse`:

```python
from jax.experimental.pallas import fuser
from tokamax._src.ops.gmm_v2 import gmm_fuser


@jax.jit
@fuser.fuse
def fused_bare_gmm(x, w, rhs, out_bias, group_sizes):
  out = gmm_fuser.gmm_v2_fusible(rms_norm(x, w), rhs, group_sizes)
  return jax.nn.silu(out + out_bias)
```

When no prologue or epilogue ops surround `gmm_v2_fusible` / `tgmm_v2_fusible`,
trivial 0-equation fusions are unwrapped so the compiled kernel takes the exact
unfused path.

#### Supported Operations and Constraints

*   **Currently supported Tokamax ops**: `PallasMosaicTpuV2RaggedDot`
    (`implementation="mosaic_tpu_v2"`, covering both forward/dlhs `gmm_v2` and
    drhs `tgmm_v2`).
*   **Supported prologue / epilogue JAX primitives**: Governed by
    [JAX Pallas Fuser](https://docs.jax.dev/en/latest/pallas/tpu/fuser.html).
    Commonly supported ops include pointwise unary/binary math,
    `select`/`clamp`, dtype casts, `broadcast_in_dim`, `transpose`,
    `concatenate`/`stack`/`split`, and `reduce_sum` across a full block axis
    (such as `RMSNorm`). Ops without fuser block-spec rules—such as `lax.pad`,
    control flow (`scan`/`while`/`cond`), and `reduce_max`/`reduce_min`—cannot
    be fused.
*   **`PallasMosaicTpuV2RaggedDot` fusion constraints**:
    *   `size_m` must be aligned to the TPU sublane tiling (`size_lhs_sublane`)
        when `lhs` or `out` is fused in `gmm_v2` (and when `lhs` or `rhs` is
        fused in `tgmm_v2`).
    *   When `out` is fused, `out_size_n` (and `size_k` for `tgmm_v2`) must be
        aligned to `num_lanes` (128) / tile size, and only single-output
        epilogues are supported.
    *   `partial_sum` is not supported with fusion.
    *   Fused `rhs` in `gmm_v2` is not supported together with `fuse_act` or
        sub-byte packed weights.

---

### Authoring a Fusion-Compatible Kernel and Op

#### Architecture Overview

Fusion dispatch is split between the public facade, the internal adapter, the
`Op` class, and the kernel's companion fuser module:

1.  [`tokamax/ops/op_fusion.py`](https://github.com/openxla/tokamax/blob/main/tokamax/ops/op_fusion.py)
    re-exports `fuse` as the public entry point.
2.  [`tokamax._src.ops.fuser_adapter`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/fuser_adapter.py)
    sets a JAX user context flag (`in_op_fusion_context()`) around
    `jax.experimental.pallas.fuser.fuse`.
3.  When `Op.__call__` runs inside that context, it checks whether the subclass
    overrides `_fusible_fwd` (raising `NotImplementedError` if not) and
    delegates to `fuser_adapter.call_fusible_op`.
4.  `call_fusible_op` suspends the fusion context while resolving
    `bound_args.default_config`, binds a `fuser.fusible` primitive over the
    array leaves, materializes any argument not listed in `Op.fusible_argnames`,
    and invokes `Op._fusible_fwd(*args, out_fusions=out_fusions, config=config,
    **kwargs)` inside `bound_args._tokamax_metadata(config)`.

#### Step 1: Declare `fusible_argnames` and Override `_fusible_fwd` on Your `Op`

On your backend `Op` subclass (see
[`PallasMosaicTpuV2RaggedDot`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py)
for a full reference):

1.  Set `fusible_argnames` to the parameter names whose array leaves should
    remain `fuser.Fusion` closures. All other array arguments (such as metadata,
    indices, or `group_sizes`) are automatically materialized to regular
    `jax.Array`s by `call_fusible_op` before `_fusible_fwd` runs.
2.  Override `_fusible_fwd`. Unlike `_fwd` (which returns `(out, residuals)`),
    `_fusible_fwd` returns the output `T` directly and receives `out_fusions`
    and the resolved `config`.
3.  For composite PyTree operands whose leaves should not be fused (for example,
    pre-quantized `QArray(qvalue, scale)` weights), materialize them explicitly
    with `fuser_adapter.materialize(...)`.

```python
from typing import ClassVar, override
import jax
from tokamax._src.ops import fuser_adapter


class MyPallasTpuOp(MyBaseOp[Config, None]):
  config_cls: ClassVar[type[Config]] = Config
  fusible_argnames: ClassVar[tuple[str, ...]] = ("lhs", "rhs")

  @override
  def _fusible_fwd(
      self,
      lhs: jax.Array | fuser_adapter.InputFusion,
      rhs: jax.Array | fuser_adapter.InputFusion,
      *,
      out_fusions: fuser_adapter.OutputFusion | None = None,
      config: Config,
      **kwargs,
  ) -> jax.Array:
    return my_kernel_fusible(
        lhs, rhs, config=config, out_fusions=out_fusions, **kwargs
    )
```

#### Step 2: Keep Fuser Integration in a Companion `*_fuser.py` Module

Follow Tokamax's separation between the core kernel (`gmm_v2.py`, `tgmm_v2.py`)
and its fuser integration (`gmm_fuser.py`):

*   Keep `jax.experimental.pallas.fuser` imports out of the core kernel file so
    the unfused kernel remains self-contained and easy to read.
*   Unwrap 0-equation trivial `fuser.Fusion` closures before building your
    fusion spec:

    ```python
    def _unwrap_trivial_input_fusion(x):
      if isinstance(x, fuser.Fusion) and not jax.make_jaxpr(x)().jaxpr.eqns:
        return x()
      return x
    ```

    `fuser.fusible` wraps every input array leaf in a `Fusion`, even when that
    operand has no producer ops. Unwrapping trivial closures keeps untouched
    operands on the unmodified unfused `BlockSpec` path with zero runtime
    overhead.
*   If your base kernel is decorated with `@jax.jit`, call
    `inspect.unwrap(base_kernel)` on the fusible path—`@jax.jit` rejects
    `fuser.Fusion` arguments.
*   If bare-kernel callers should also be able to invoke your fusible entrypoint
    directly inside `@fuser.fuse`, detect bare `jax.Array` calls and bind
    `fuser.fusible` automatically (as `gmm_fuser.gmm_v2_fusible` does).

#### Step 3: Unpack Fusions and Transform `BlockSpec`s with `fuser_adapter`

Use the shared helpers in
[`tokamax._src.ops.fuser_adapter`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/fuser_adapter.py)
rather than calling low-level `fuser` APIs directly:

*   **Unpack an input prologue**:

    ```python
    lhs_fn, lhs_values, lhs_values_type = fuser_adapter.unpack_input_fusion(lhs)
    ```

    Returns `(fn, values, values_type)` where `values` is the tuple of HBM
    arrays closed over by the prologue and `values_type` holds their
    `ShapeDtypeStruct`s.
*   **Unpack a single-output epilogue**:

    ```python
    out_fn, out_values, out_values_type, out_init = (
        fuser_adapter.unpack_output_fusion(out_fusions, raw_out_type)
    )
    ```

    Unwraps Pallas Fuser's 1-tuple output convention into a single
    `ShapeDtypeStruct` (`out_init`) and raises `NotImplementedError` on
    multi-output epilogues.
*   **Pull an input `BlockSpec` backward through a prologue**:

    ```python
    pulled_lhs_fn, lhs_val_specs = fuser_adapter.pull_input_spec(
        lhs_fn,
        lhs_spec,
        lhs_values_type,
        grid_len=grid_len,
        num_scalar_prefetch=0,
    )
    ```

*   **Push and pull an output `BlockSpec` through an epilogue**:

    ```python
    pulled_out_fn, out_fusion_input_specs, final_out_spec = (
        fuser_adapter.push_pull_block_spec(
            out_fn,
            out_spec,
            out_values_type,
            raw_out_type,
            grid_len=grid_len,
            num_scalar_prefetch=0,
        )
    )
    ```

    `fuser_adapter.push_pull_block_spec` handles shape-preserving reductions
    with size-1 broadcasts (such as `RMSNorm` epilogues that trip raw
    `fuser.push_block_spec`), verifies that the pulled spec matches the kernel's
    written `out_spec`, unwraps the 1-tuple return value, and maintains
    compatibility across supported JAX releases.

#### Step 4: Wire Pulled Specs and Callbacks Into the Pallas Kernel

1.  **Pass captured HBM inputs alongside their pulled `BlockSpec`s**:
    *   Input prologue `values` ride in place of the fused operand (paired with
        `lhs_val_specs`).
    *   Output epilogue side inputs (`out_values`, such as an epilogue bias or
        residual) are passed as read-only HBM kernel inputs paired with
        `out_fusion_input_specs` so `pltpu.emit_pipeline` prefetches their tiles
        into VMEM. When unfused, pass an empty tuple `()` for both refs and
        specs (0 PyTree leaves).
2.  **Use distinct PyTree types per fusion mode**:
    *   `pltpu.emit_pipeline` maps `jax.tree.map(..., in_specs, in_refs)`, so
        each input ref container and its `in_spec` container must share the
        exact same `@jax.tree_util.register_dataclass` PyTree type (for example,
        `LhsRef[T]` for unfused `lhs` vs. `FusedLhsRef[T]` for prologue-fused
        `lhs`).
3.  **Evaluate the pulled callbacks on VMEM tiles inside the kernel body**:
    *   Every pulled function has signature
        `fn(program_ids, scalar_prefetch, *args)`.
    *   Inside `pltpu.emit_pipeline(..., _explicit_indices=True)`, pass
        `pids = step.index` (otherwise `tuple(pl.program_id(i) for i in
        range(grid_len))`) and `()` when `num_scalar_prefetch=0`:

    ```python
    # Prologue: compute the input tile from prefetched VMEM refs.
    tiled_lhs = lhs_fn(pids, (), tuple(r[...] for r in lhs_val_refs))

    # Epilogue: transform the accumulated output tile before storing.
    out_vals = tuple(r[...] for r in tiled_out_fusion_input_refs)
    out_tile = out_fn(pids, (), out_vals, acc_tile)
    ```

#### Common Pitfalls

*   **`pl.BoundedSlice` and `reshape`**: Pallas Fuser's reshape pull rule
    rejects `pl.BoundedSlice`. If your unfused kernel reshapes 2D HBM inputs to
    3D sublane-tiled refs before `emit_pipeline`, convert the spec to a 2D
    `BlockSpec((pl.BoundedSlice(tile_m), col_dim), ...)` before pulling or
    pushing through fused operands (see `_to_2d_bounded_spec` in
    [`gmm_fuser.py`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gmm_v2/gmm_fuser.py)).
*   **Epilogue ordering with row masking**: If your kernel masks out-of-group or
    padding rows to zero and combines partial sublane tiles across steps, run
    `out_fn` *before* applying the zero mask so non-zero epilogues (like
    `+ bias`) do not corrupt masked rows.
*   **`pl.kernel` with `Sequence` `out_type`**: `pl.kernel` unpacks a `tuple` or
    `list` `out_type` into multiple positional `out_ref` arguments. Using
    `fuser_adapter.unpack_output_fusion` keeps single-output `out_init` as a
    bare `ShapeDtypeStruct`.
*   **Host-side padding vs. fused shapes**: Because Pallas Fuser does not have a
    `lax.pad` rule or unaligned `lax.slice` push/pull rules, validate alignment
    requirements early in your fusion spec's `prepare()` method and raise
    `NotImplementedError` for unsupported shapes rather than padding `Fusion`
    objects.
*   **Helper `Op`s inside `_fusible_fwd`**: `call_fusible_op` suspends
    `in_op_fusion_context()` during `default_config` resolution and inside
    `_fusible_fwd`, so calling an unfused helper `Op` inside `_fusible_fwd` runs
    its normal `_fwd` path without triggering nested fusion errors.

#### Testing Checklist

When adding fusion support to an operator, cover:

1.  **Prologue fusion** (pointwise and full-axis reduction such as `RMSNorm`),
    **epilogue fusion** (with and without external side inputs like bias or
    residual), and **combined prologue + epilogue**.
2.  **No-fusion parity**: Verify calling the fusible entrypoint with no
    surrounding ops (plain, `@op_fusion.fuse`, and `@fuser.fuse`) produces
    bitwise-identical output (`chex.assert_trees_all_equal`) to the unfused
    kernel.
3.  **Explicit `@jax.custom_vjp` gradients**: Verify forward values and
    gradients match the unfused reference.
4.  **Unsupported combinations**: Verify clear `NotImplementedError` messages
    for unaligned shapes or incompatible kernel options.

#### Reference Implementation

*   Public facade:
    [`tokamax/ops/op_fusion.py`](https://github.com/openxla/tokamax/blob/main/tokamax/ops/op_fusion.py)
*   `Op` dispatch and shared helpers:
    [`tokamax/_src/ops/fuser_adapter.py`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/fuser_adapter.py)
*   Kernel fuser integration (`GmmFusionSpec` / `TgmmFusionSpec`):
    [`tokamax/_src/ops/gmm_v2/gmm_fuser.py`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gmm_v2/gmm_fuser.py)
*   `Op._fusible_fwd` integration:
    [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py)
*   Kernel and `Op` fusion tests:
    [`tokamax/_src/ops/gmm_v2/gmm_fuser_test.py`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gmm_v2/gmm_fuser_test.py)
    and
    [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py)
