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
"""Pallas TPU kernel for mHC post: the stream recombine."""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.mhc import utils
from tokamax._src.ops.experimental.tpu.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES  # pylint: disable=g-importing-member


def _post_kernel(
    x_ref, res_ref, post_ref, comb_ref, out_ref, *, hc_mult, hidden_size
):
  """One token block of the unrolled stream recombine."""
  x = x_ref[...].astype(jnp.float32)  # (tb, hidden)
  post = post_ref[...]  # (tb, hc_mult) f32
  comb = comb_ref[...]  # (tb, hc_mult * hc_mult) f32, row-major (i, j)

  streams = [
      res_ref[:, i * hidden_size : (i + 1) * hidden_size].astype(jnp.float32)
      for i in range(hc_mult)
  ]
  for j in range(hc_mult):
    acc = post[:, j : j + 1] * x
    for i in range(hc_mult):
      k = i * hc_mult + j
      acc = acc + comb[:, k : k + 1] * streams[i]
    out_ref[:, j * hidden_size : (j + 1) * hidden_size] = acc.astype(
        out_ref.dtype
    )


@functools.partial(
    jax.jit, static_argnames=("token_block_size", "vmem_limit_bytes")
)
def mhc_post_2d(
    x: jax.Array,
    res2d: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
  """2D-native entry: the kernel's own layout as public API.

  The kernel consumes and produces flat `(num_tokens, hc_mult * hidden_size)`
  streams. Callers that can keep the residual flat (a composed seam, 2D model
  plumbing) should call this directly.

  Args:
    x: (num_tokens, hidden_size), bfloat16 — the sublayer output.
    res2d: (num_tokens, hc_mult * hidden_size), bfloat16.
    post_layer_mix: (num_tokens, hc_mult, 1) or (num_tokens, hc_mult),
      float32.
    comb_res_mix: (num_tokens, hc_mult, hc_mult), float32.
    token_block_size: Tokens per grid step.
    vmem_limit_bytes: Scoped-VMEM budget for the kernel.

  Returns:
    (num_tokens, hc_mult * hidden_size) in `res2d.dtype`.
  """
  assert res2d.dtype == jnp.bfloat16, res2d.dtype
  assert post_layer_mix.dtype == jnp.float32, post_layer_mix.dtype
  assert comb_res_mix.dtype == jnp.float32, comb_res_mix.dtype

  hc_mult = comb_res_mix.shape[-1]
  hidden_size = res2d.shape[-1] // hc_mult

  x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
  post2d = post_layer_mix.reshape(-1, hc_mult)
  comb2d = comb_res_mix.reshape(-1, hc_mult * hc_mult)
  num_tokens = res2d.shape[0]

  def _vmem_need(tb: int) -> int:
    # Double-buffered bf16 in/out blocks (x + old streams + new
    # streams) plus f32 accumulator transients; x2 spill headroom.
    return 2 * (
        tb * hidden_size * (2 + 2 * hc_mult * 2) * 2 + tb * hidden_size * 4 * 2
    )

  tb, padded_tokens = utils.select_token_block(
      num_tokens,
      token_block_size,
      vmem_need=_vmem_need,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  x2d, res2d, post2d, comb2d = utils.pad_to(
      padded_tokens, x2d, res2d, post2d, comb2d
  )

  compiler_params = pltpu.CompilerParams(
      dimension_semantics=("parallel",),
      vmem_limit_bytes=vmem_limit_bytes,
      disable_bounds_checks=True,
  )

  out = pl.pallas_call(
      functools.partial(_post_kernel, hc_mult=hc_mult, hidden_size=hidden_size),
      grid=(padded_tokens // tb,),
      in_specs=[
          pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
          pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0)),
          pl.BlockSpec((tb, hc_mult), lambda i: (i, 0)),
          pl.BlockSpec((tb, hc_mult * hc_mult), lambda i: (i, 0)),
      ],
      out_specs=pl.BlockSpec((tb, hc_mult * hidden_size), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct(
          (padded_tokens, hc_mult * hidden_size), res2d.dtype
      ),
      compiler_params=compiler_params,
  )(x2d, res2d, post2d, comb2d)

  (out,) = utils.trim_to(num_tokens, out)
  return out


def mhc_post(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
  """Pallas mHC post on `(..., hc_mult, hidden_size)` residual streams.

  Thin adapter over `mhc_post_2d` that takes and returns the 3D layout.

  Args:
    x: (..., hidden_size), bfloat16 — the sublayer output.
    residual: (..., hc_mult, hidden_size), bfloat16.
    post_layer_mix: (..., hc_mult, 1), float32.
    comb_res_mix: (..., hc_mult, hc_mult), float32.
    token_block_size: Tokens per grid step.
    vmem_limit_bytes: Scoped-VMEM budget for the kernel.

  Returns:
    (..., hc_mult, hidden_size) in `residual.dtype`.
  """
  assert residual.dtype == jnp.bfloat16, residual.dtype

  outer_shape = residual.shape[:-2]
  hc_mult, hidden_size = residual.shape[-2:]

  out2d = mhc_post_2d(
      x.reshape(-1, hidden_size),
      residual.reshape(-1, hc_mult * hidden_size),
      post_layer_mix.reshape(-1, hc_mult, 1),
      comb_res_mix.reshape(-1, hc_mult, hc_mult),
      token_block_size=token_block_size,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  return out2d.reshape(*outer_shape, hc_mult, hidden_size)
