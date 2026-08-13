# Minimal repro: Mosaic GPU layout inference fails on a partial reduction
# with `keepdims=True`.
#
# No GPU required -- this reproduces at lowering time. Run with:
#
#     uv run python jax-bug-mgpu-keepdims/repro.py
#
# Expected output:
#     FAIL  keepdims=True   ValueError: Failed to infer a possible set of layouts...
#     OK    keepdims=False
#
# The two kernels compute the same thing. The only difference is whether the
# row mean is produced as a rank-preserving `(BLOCK_M, 1)` value or as a
# rank-reducing `(BLOCK_M,)` value that is then broadcast back.

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

M, N, BLOCK_M = 256, 128, 64
DTYPE = jnp.float32
BLOCK = (BLOCK_M, N)

# Tiled + swizzled SMEM, so that layout inference has a tiled layout to work
# with at all. Without transforms the system is unsatisfiable even for the
# `keepdims=False` case.
SWIZZLE = plgpu.find_swizzle(N * 32, "x")
TRANSFORMS = (
    plgpu.TilingTransform((8, 8 * SWIZZLE // 32)),
    plgpu.SwizzleTransform(SWIZZLE),
)


def block_spec():
  return plgpu.BlockSpec(BLOCK, lambda i: (i, 0), transforms=TRANSFORMS)


def with_keepdims(x_ref, y_ref):
  x = x_ref[...]
  y_ref[...] = x - jnp.mean(x, axis=-1, keepdims=True)


def without_keepdims(x_ref, y_ref):
  x = x_ref[...]
  mean = jnp.mean(x, axis=-1)  # rank-reducing: (BLOCK_M,)
  y_ref[...] = x - jax.lax.broadcast_in_dim(mean, BLOCK, (0,))


def lower(body):
  f = pl.pallas_call(
      body,
      out_shape=jax.ShapeDtypeStruct((M, N), DTYPE),
      grid=(M // BLOCK_M,),
      in_specs=[block_spec()],
      out_specs=block_spec(),
      compiler_params=plgpu.CompilerParams(
          lowering_semantics=plgpu.LoweringSemantics.Warpgroup
      ),
  )
  x = jnp.zeros((M, N), DTYPE)
  return jax.jit(f).trace(x).lower(lowering_platforms=("cuda",)).as_text()


for name, body in (("keepdims=True ", with_keepdims),
                   ("keepdims=False", without_keepdims)):
  try:
    lower(body)
    print(f"OK    {name}")
  except Exception as e:  # pylint: disable=broad-except
    print(f"FAIL  {name}  {type(e).__name__}: {str(e)[:120]}")
