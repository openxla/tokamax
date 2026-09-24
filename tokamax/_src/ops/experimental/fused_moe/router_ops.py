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
"""Top-k expert selection for the fused expert-parallel MoE router:
pallas_select is an in-VMEM max-and-mask pass replacing XLA's lax.top_k.

An all-NaN score row selects the lowest expert on every slot, with NEG as
the weight of each: NaN is flushed to the sentinel before the first max, so
every row has a maximum some column equals and every index returned is a
real expert. What that row's weights MEAN is not decided here. The kernel
layer masks such a row explicitly, on jnp.any(jnp.isfinite(scores)), rather
than relying on the sentinels to sum to something the renormalization turns
into zero -- that outcome was a property of the accumulation dtype and the
association order rather than of the design.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

NEG = -3.0e38  # below any real softmax score (>=0); avoids inf


def _select_kernel(x_ref, w_ref, i_ref, *, topk, n, v_ref=None):
    """One block of rows: take the top `topk` by repeated max-and-mask."""
    # NaN is flushed to the sentinel so every row has a maximum some
    # column equals; an all-NaN row selects expert 0 with weight NEG.
    x = jnp.where(jnp.isnan(x_ref[...]), NEG, x_ref[...])  # [R, n] f32
    v = None if v_ref is None else v_ref[...]
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)  # [R,1] rowmax
        ismax = x == m
        # Lowest matching column; non-matching columns fill with the last
        # expert so every index returned is a real expert.
        idx = jnp.min(jnp.where(ismax, iota, n - 1), axis=1)
        picked = iota == idx[:, None]
        # noaux selects with biased scores but emits unbiased weights.
        # A masked sum reads the resident tile without a dynamic lane gather.
        w_ref[:, j] = (m[:, 0] if v is None else jnp.sum(
            jnp.where(picked, v, 0.0), axis=1))
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(picked, NEG, x)


def _select_kernel_split(x_ref, v_ref, w_ref, i_ref, *, topk, n):
    """Select by x_ref and read the selected weights from v_ref.

    Pallas passes input refs before output refs, so the two-input call needs
    this signature. Forward v_ref as a keyword to reuse _select_kernel's
    selection loop while keeping its one-input call unchanged.
    """
    _select_kernel(x_ref, w_ref, i_ref, topk=topk, n=n, v_ref=v_ref)


def pallas_select(scores,
                  topk=10,
                  block_rows=256,
                  weight_scores=None,
                  interpret=False):
    """Top-k over [R, N] scores -> (weights f32, indices i32) [R, topk].

    With `weight_scores=None`, return the selected scores as weights.
    Otherwise, select experts by `scores` and read their weights from the
    separate [R, N] `weight_scores` table. DeepSeek-V4's `noaux` uses this
    to select with `scores=raw_scores + correction_bias` while returning
    unbiased weights from `weight_scores=raw_scores`. Here `raw_scores` are
    the scores after the scoring function, before adding the bias.
    Normalization and routed scaling are applied by the caller.

    `interpret` runs the kernel body on the host, which is what lets the
    selection rule be tested without a chip; it does not gate Mosaic
    lowering. Serving never sets it.
    """
    R, N = scores.shape
    assert R % block_rows == 0, (R, block_rows)
    if weight_scores is not None and weight_scores.shape != scores.shape:
        raise ValueError(
            f"weight_scores {weight_scores.shape} must match scores "
            f"{scores.shape}")
    grid = (R // block_rows, )
    out_shapes = (
        jax.ShapeDtypeStruct((R, topk), jnp.float32),
        jax.ShapeDtypeStruct((R, topk), jnp.int32),
    )
    in_spec = pl.BlockSpec((block_rows, N), lambda i: (i, 0))
    bs = (in_spec, ) if weight_scores is None else (in_spec, in_spec)
    os = (
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
    )
    kernel = (_select_kernel
              if weight_scores is None else _select_kernel_split)
    operands = ((scores, ) if weight_scores is None else
                (scores, weight_scores))
    return pl.pallas_call(
        functools.partial(kernel, topk=topk, n=N),
        grid=grid,
        in_specs=list(bs),
        out_specs=list(os),
        out_shape=list(out_shapes),
        interpret=interpret,
    )(*operands)
