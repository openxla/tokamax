# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native CP workspace equivalence with global labels larger than local capacity."""

import dataclasses
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest
from tokamax._src import jaxtyping
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import cp_compact
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata


@pytest.mark.parametrize(
    "cp,split,padding",
    [
        (4, False, False),
        (4, True, False),
        (4, True, True),
        (8, True, False),
        (4, True, "all"),
        (4, True, "rank"),
        (8, True, "rank"),
    ],
)
def test_compact_cp(cp, split, padding):
  _run_compact_cp(cp, split, padding)


@pytest.mark.parametrize(
    "heads,local_tokens,remat,outer_remat,scan",
    [
        (64, 128, False, True, True),
        (64, 1024, True, False, False),
        (64, 4096, False, False, False),
        (8, 32768, False, False, False),
    ],
)
def test_compact_cp_large_and_nested(
    heads, local_tokens, remat, outer_remat, scan
):
  _run_compact_cp(
      8,
      True,
      "rank",
      heads=heads,
      local_tokens=local_tokens,
      remat=remat,
      outer_remat=outer_remat,
      scan=scan,
  )


@pytest.mark.parametrize(
    "heads,local_tokens,nested",
    [(4, 128, False), (64, 128, True), (64, 16384, False)],
)
def test_compact_cp16(heads, local_tokens, nested):
  _run_compact_cp(
      16,
      True,
      "rank",
      heads=heads,
      local_tokens=local_tokens,
      outer_remat=nested,
      scan=nested,
  )


@pytest.mark.parametrize("cp", [8, 16])
def test_compact_cp_random_cross_rank(cp):
  _run_compact_cp(
      cp, True, "rank", heads=8, local_tokens=512, random_packing=True
  )


def _run_compact_cp(
    cp,
    split,
    padding,
    *,
    heads=4,
    local_tokens=128,
    remat=False,
    outer_remat=False,
    scan=False,
    random_packing=False,
):
  batch, over = 1, False
  if jax.default_backend() != "tpu" or jax.device_count() < cp:
    pytest.skip("Requires a native TPU mesh of the requested CP size")
  dim = 128
  tokens = cp * local_tokens
  rng = np.random.default_rng(619)
  shape = (heads, batch, tokens, dim)
  arrays = [
      jnp.asarray(rng.normal(size=shape) * 0.05, jnp.bfloat16) for _ in range(4)
  ]
  q, k, v, g = arrays
  beta = jnp.full(shape[:-1], 0.5, jnp.bfloat16)
  a = jnp.full((heads,), -0.5, jnp.float32)
  bias = jnp.full((heads * dim,), -3.5, jnp.float32)
  dout = jnp.asarray(rng.normal(size=shape) * 0.05, jnp.bfloat16)
  ids = np.zeros((batch, tokens), np.int32)
  for rank in range(cp):
    start = rank * local_tokens
    ids[:, start : start + 65] = 2 * rank + 1
    ids[:, start + 65 : start + local_tokens] = 2 * rank + 3
    if split:
      ids[:, start + 32 : start + 65] = 2 * rank + 2
    if padding:
      ids[:, start + 17 : start + 24] = 0
  if over:
    ids[-1, -local_tokens + 32 : -local_tokens + 65] = 2 * cp
  if padding == "rank":
    ids[:, local_tokens * (cp // 2) : local_tokens * (cp // 2 + 1)] = 0
  elif padding == "all":
    ids[:] = 0
  elif padding:
    ids[-1, -31:] = 0
  if random_packing:
    lengths = rng.integers(17, 180, size=tokens // 17 + 1)
    ids = np.repeat(np.arange(1, len(lengths) + 1, dtype=np.int32), lengths)[
        :tokens
    ][None]
    ids[:, 13::41] = 0
    ids[:, local_tokens * (cp // 2) : local_tokens * (cp // 2 + 1)] = 0
  ids = jnp.asarray(np.where(ids > 0, ids + 262100, 0))
  mesh = Mesh(np.array(jax.devices()[:cp]), ("context",))
  meta = ContextParallelMetadata(mesh=mesh, axis_name="context")
  config = mosaic.Config(
      fuse_forward=True,
      fuse_backward=True,
      fuse_cp_backward=True,
      fuse_rematerialization=True,
      rematerialize_for_backward=remat,
  )
  attention = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=dataclasses.replace(config, cp_local_state_indices=True)
  )
  spec = P(None, None, "context", None)
  inputs = (q, k, v, g, beta, a, bias, dout, ids)
  if jax.process_count() > 1:
    from jax.sharding import NamedSharding

    input_specs = (spec,) * 4 + (
        P(None, None, "context"),
        P(),
        P(),
        spec,
        P(None, "context"),
    )
    host_inputs = tuple(np.asarray(x) for x in inputs)
    inputs = tuple(
        jax.make_array_from_callback(
            x.shape,
            NamedSharding(mesh, layout),
            lambda index, value=x: value[index],
        )
        for x, layout in zip(host_inputs, input_specs, strict=True)
    )
  results = []
  routes = (
      ("reference_recurrence", "compact")
      if random_packing
      else ("reference_capacity", "compact")
  )
  for route in routes:

    def local(q, k, v, g, beta, a, bias, dout, ids):
      def call(q, k, v, g, beta, a, bias):
        options = dict(
            a_log=a,
            delta_time_bias=bias,
            lower_bound=-1.0,
            use_gate_in_kernel=True,
            segment_ids=ids,
            context_parallel_metadata=meta,
        )
        if route == "reference_recurrence":
          return _grouped_recurrence(
              q, k, v, g, beta, a, bias, ids, context=meta
          )
        if route == "reference_capacity":
          return attention(q, k, v, g, beta, max_num_segments=32, **options)[0]
        return cp_compact.kimi_delta_attention_cp_compact(
            q,
            k,
            v,
            g,
            beta,
            config=config,
            staged_run_cap=2 if split else 64,
            **options,
        )[0]

      if outer_remat:
        call = jax.checkpoint(call)
      if scan:
        step_call = call

        def call(*xs):
          def body(carry, unused):
            return carry, step_call(*xs)

          return jax.lax.scan(body, 0, None, length=2)[1].mean(axis=0)

      out, back = jax.vjp(call, q, k, v, g, beta, a, bias)
      grads = back(dout)
      return out, (
          *grads[:5],
          jax.lax.psum(grads[5], "context"),
          jax.lax.psum(grads[6], "context"),
      )

    with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
      fn = jax.jit(
          jax.shard_map(
              local,
              mesh=mesh,
              check_vma=False,
              in_specs=(spec,) * 4
              + (P(None, None, "context"), P(), P(), spec, P(None, "context")),
              out_specs=(
                  spec,
                  (spec, spec, spec, spec, P(None, None, "context"), P(), P()),
              ),
          )
      )
      results.append(jax.block_until_ready(fn(*inputs)))
  for result in results[1:]:
    for actual, expected in zip(
        jax.tree.leaves(result), jax.tree.leaves(results[0]), strict=True
    ):
      # Compare local shards on multi-host meshes without gathering full tensors.
      actual_shards = actual.addressable_shards
      expected_shards = expected.addressable_shards
      for lhs, rhs in zip(actual_shards, expected_shards, strict=True):
        actual_host = np.asarray(lhs.data, np.float64)
        expected_host = np.asarray(rhs.data, np.float64)
        assert (
            np.isfinite(actual_host).all() and np.isfinite(expected_host).all()
        )
        rms = np.sqrt(np.mean((actual_host - expected_host) ** 2))
        reference_rms = max(np.sqrt(np.mean(expected_host**2)), 1e-30)
        assert rms / reference_rms <= 0.02


def _grouped_recurrence(q, k, v, g, beta, a, bias, ids, *, context=None):
  """Independent FP32 recurrence for grouped labels; padding preserves state.

  Keep one active state instead of an unused-label state table, so a long CP
  gradient check does not retain that entire table at every scan iteration.
  """
  local_tokens = q.shape[2]
  if context is not None:
    axis = context.axis_name
    q, k, v, g, beta = [
        jax.lax.all_gather(x, axis, axis=2, tiled=True)
        for x in (q, k, v, g, beta)
    ]
    ids = jax.lax.all_gather(ids, axis, axis=1, tiled=True)
  heads, _, _, dim = q.shape
  gates = -jax.nn.sigmoid(
      jnp.exp(a.astype(jnp.float32))[:, None, None, None]
      * (g.astype(jnp.float32) + bias.reshape(heads, 1, 1, dim))
  )
  xs = (
      q[:, 0].transpose(1, 0, 2),
      k[:, 0].transpose(1, 0, 2),
      v[:, 0].transpose(1, 0, 2),
      gates[:, 0].transpose(1, 0, 2),
      beta[:, 0].T,
      ids[0],
  )

  def step(carry, x):
    state, last = carry
    qi, ki, vi, gi, bi, label = x
    qi, ki, vi, gi, bi = [z.astype(jnp.float32) for z in (qi, ki, vi, gi, bi)]
    state = jnp.where((label > 0) & (label != last), 0.0, state)
    decayed = state * jnp.exp(gi)[..., None]
    prediction = jnp.einsum(
        "hk,hkv->hv", ki, decayed, precision=jax.lax.Precision.HIGHEST
    )
    updated = (
        decayed + (ki * bi[:, None])[..., None] * (vi - prediction)[:, None, :]
    )
    output = jnp.einsum(
        "hk,hkv->hv",
        qi * dim**-0.5,
        updated,
        precision=jax.lax.Precision.HIGHEST,
    )
    return (
        jnp.where(label > 0, updated, state),
        jnp.where(label > 0, label, last),
    ), jnp.where(label > 0, output, 0).astype(q.dtype)

  _, ys = jax.lax.scan(
      jax.checkpoint(step),
      (jnp.zeros((heads, dim, v.shape[-1]), jnp.float32), jnp.int32(0)),
      xs,
  )
  output = ys.transpose(1, 0, 2)[:, None]
  if context is not None:
    output = jax.lax.dynamic_slice_in_dim(
        output,
        jax.lax.axis_index(context.axis_name) * local_tokens,
        local_tokens,
        axis=2,
    )
  return output


def test_grouped_recurrence_matches_existing_plain_jax():
  from tokamax._src.ops.kda import base

  rng = np.random.default_rng(619)
  shape = (2, 1, 17, 4)
  q, k, v, g = [
      jnp.asarray(rng.normal(size=shape) * 0.05, jnp.float32) for _ in range(4)
  ]
  beta = jnp.full(shape[:-1], 0.5, jnp.float32)
  a = jnp.full((2,), -0.5, jnp.float32)
  bias = jnp.full((8,), -3.5, jnp.float32)
  ids = jnp.asarray([[0, 1, 1, 0, 1, 2, 2, 2, 0, 3, 3, 0, 3, 3, 3, 0, 0]])
  reference = base.KimiDeltaAttention()

  def plain(q, k, v, g, beta, a, bias):
    return reference(
        q,
        k,
        v,
        g,
        beta,
        a_log=a,
        delta_time_bias=bias,
        segment_ids=ids,
        max_num_segments=3,
        use_gate_in_kernel=True,
        lower_bound=-1.0,
    )[0]

  def evaluate(fn):
    output, pb = jax.vjp(fn, q, k, v, g, beta, a, bias)
    return output, pb(jnp.full_like(output, 0.125))

  # The oracle models FP32 arithmetic, including its contractions. TPU's
  # default dot precision otherwise rounds FP32 operands before the dot.
  with jax.default_matmul_precision("highest"):
    actual = evaluate(lambda *xs: _grouped_recurrence(*xs, ids))
    expected = evaluate(plain)
  for lhs, rhs in zip(
      jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
  ):
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-5, atol=1e-9
    )
