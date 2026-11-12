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
"""8-device parity for the fused MoE decode kernel.

Tests routed + shared expert + in-kernel all-reduce (or all-to-all + local sum
DP mode), with the experts TP-sliced (every rank holds all E) or EP-sharded
(rank r holds global experts [r*E, (r+1)*E)), whole or streamed in intermediate
blocks; the shared expert is always a TP slice, gated or ungated with a routed
scale.
"""

import functools

from absl import app
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pytest
from tokamax._src.ops.experimental.fused_moe import reference
from tokamax._src.ops.experimental.fused_moe_decode.fused_moe_decode_kernel import (  # pylint: disable=g-multiple-import,g-importing-member
    fused_moe_decode,
    plan_weight_ring,
    vmem_bytes,
)

# pylint: disable=missing-function-docstring,redefined-outer-name,g-doc-args,g-doc-return-or-yield,protected-access

pytestmark = pytest.mark.multichip

N, H, INTER, INTER_S, E, TOPK = 8, 1024, 256, 128, 64, 10
F8 = jnp.float8_e4m3fn
ROUTED_SCALE = 2.5


@pytest.fixture(scope="module")
def setup():
  if (
      jax.default_backend() != "tpu"
      or jax.device_count() < N
      or pltpu.get_tpu_info().generation < 7
  ):
    pytest.skip("requires 8 TPU v7x devices")
  mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
  sh = NamedSharding(mesh, P("d"))
  rank = jax.device_put(np.arange(N, dtype=np.int32).reshape(N, 1), sh)

  def gen(r):
    ks = jax.random.split(jax.random.fold_in(jax.random.key(3), r[0, 0]), 4)

    def q(w, axis):
      s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
      return (w / s).astype(F8), s

    w1, w1_scale = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
    w2, w2_scale = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
    shared_w1, shared_w1_scale = q(
        jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0
    )
    shared_w2, shared_w2_scale = q(
        jax.random.normal(ks[3], (INTER_S, H)) * 0.02, 0
    )
    return (
        w1,
        w2,
        w1_scale[:, :, None, :],
        w2_scale[:, :, None, :],
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
    )

  weights = jax.jit(
      jax.shard_map(
          gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
      )
  )(rank)
  shared_gate_weight = jax.device_put(
      jnp.asarray(
          np.random.default_rng(1).normal(size=(H, 1)) * 0.02, jnp.bfloat16
      ),
      NamedSharding(mesh, P()),
  )
  return mesh, sh, rank, weights, shared_gate_weight


def _reference(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    skip_expert=None,
    ep=False,
    ungated=False,
    scale_shared=False,
):
  """The decomposed layer in f32: routed experts + gated shared expert + psum.

  ungated: routed x ROUTED_SCALE + ungated shared expert; scale_shared=True
  scales the shared expert too (wrong).
  """
  expert_start = rank.reshape(()) * E if ep else None
  if skip_expert is not None:
    local_ti = ti if expert_start is None else (ti - expert_start)
    tw = jnp.where(local_ti == skip_expert, 0.0, tw)
  if ungated and scale_shared:
    shared_w2_scale = shared_w2_scale * ROUTED_SCALE
  return reference.fused_moe_reference(
      x=x,
      w1=w1,
      w2=w2,
      w1_scale=w1_scale,
      w2_scale=w2_scale,
      topk_weights=tw,
      topk_ids=ti,
      shared_w1=shared_w1,
      shared_w1_scale=shared_w1_scale,
      shared_w2=shared_w2,
      shared_w2_scale=shared_w2_scale,
      shared_gate_weight=None if ungated else shared_gate_weight,
      rank=rank.reshape(-1),
      n_ranks=N,
      expert_start=expert_start,
      routed_scale=ROUTED_SCALE if ungated else 1.0,
  )


def _fused(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    ep=False,
    ungated=False,
    i_block=None,
):
  return fused_moe_decode(
      rank.reshape(-1),
      x,
      w1,
      w2,
      w1_scale,
      w2_scale,
      shared_w1,
      shared_w1_scale,
      shared_w2,
      shared_w2_scale,
      None if ungated else shared_gate_weight,
      tw,
      ti,
      n_ranks=N,
      expert_start=rank.reshape(()) * E if ep else None,
      routed_scale=ROUTED_SCALE if ungated else 1.0,
      i_block=i_block,
  )


def _inputs(setup_fixture, m, ep, padded, seed):
  """(run, ids): random routing for m tokens, replicated on every rank.

  run(fn) gives each rank's output, [N, m, H].
  """
  mesh, sh, rank, weights, shared_gate_weight = setup_fixture
  rng = np.random.default_rng(seed)
  x = rng.normal(size=(m, H)).astype(np.float32)
  n_global = N * E if ep else E
  ids = np.stack(
      [rng.choice(n_global, TOPK, replace=False) for _ in range(m)]
  ).astype(np.int32)
  tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
  if padded:
    tw[m - 3 :] = 0
  args = (
      rank,
      jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
      *weights,
      shared_gate_weight,
      jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
      jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
  )
  specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

  def run(fn):
    out = jax.jit(
        jax.shard_map(
            fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
        )
    )(*args)
    return np.asarray(out.astype(jnp.float32)).reshape(N, m, H)

  return run, ids


# "blocked" forces blocked mode with ib 128: two blocks per expert.
_PATHS = {
    "kernel": _fused,
    "blocked": functools.partial(_fused, i_block=128),
}


def _max_rel(a, ref):
  return (
      np.linalg.norm(a - ref, axis=1)
      / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
  ).max()


@pytest.mark.parametrize("m", [8, 32])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("path", ["kernel", "blocked"])
def test_matches_decomposed_layer(setup, m, padded, ep, path):
  if path != "kernel" and padded:
    pytest.skip("one padded case per mode")
  run, ids = _inputs(setup, m, ep, padded, seed=m)
  fn = _PATHS[path]
  got = run(functools.partial(fn, ep=ep))
  ref = run(functools.partial(_reference, ep=ep))
  assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
  assert np.isfinite(got).all()
  # FP8 activations and a BF16 all-reduce against f32 math: a few percent per
  # row.
  assert _max_rel(got[0], ref[0]) < 0.08
  # Positive control: a reference missing one routed expert must fail the bound.
  # In EP mode the local id of expert ids[0, 0] is only that on its owner rank.
  broken = run(
      functools.partial(_reference, skip_expert=int(ids[0, 0]) % E, ep=ep)
  )
  assert _max_rel(got[0], broken[0]) > 0.08


@pytest.mark.parametrize("path", ["kernel", "blocked"])
def test_ungated_shared_expert_routed_scale(setup, path):
  """Under EP: no shared-expert gate and routed_scale 2.5.

  Applies routed_scale on the routed experts only, against the kernel-off
  composition. Must-fail control: the scale applied to the shared expert too.
  """
  run, _ = _inputs(setup, 32, ep=True, padded=True, seed=200)
  fn = _PATHS[path]
  got = run(functools.partial(fn, ep=True, ungated=True))
  ref = run(functools.partial(_reference, ep=True, ungated=True))
  assert np.isfinite(got).all()
  assert _max_rel(got[0], ref[0]) < 0.08
  wrong = run(
      functools.partial(_reference, ep=True, ungated=True, scale_shared=True)
  )
  assert _max_rel(got[0], wrong[0]) > 0.08


def test_vmem_bound_matches_build_limits():
  """The arming gate's estimate at 8 and 16 ranks.

  TP (512 experts, i 128) fits at 512 tokens and not 1024; EP (64 experts, i
  1024) fits at 256 and not 512, both with whole experts at any token count.
  Large EP experts (16 experts of i 2048) stream blocks and fit the default cap
  of 256.
  """
  if (
      jax.default_backend() != "tpu"
      or pltpu.get_tpu_info().generation < 7
      or pltpu.get_tpu_info().vmem_capacity_bytes != (64 << 20)
  ):
    pytest.skip("requires a TPU v7x with 64 MiB VMEM")

  have = pltpu.get_tpu_info().vmem_capacity_bytes
  assert vmem_bytes(512, 4096, 128, 128, 512, 8, jnp.float8_e4m3fn) <= have
  assert vmem_bytes(1024, 4096, 128, 128, 512, 8, jnp.float8_e4m3fn) > have
  assert vmem_bytes(256, 4096, 1024, 128, 64, 8, jnp.float8_e4m3fn) <= have
  assert vmem_bytes(512, 4096, 1024, 128, 64, 8, jnp.float8_e4m3fn) > have
  for m in (16, 256, 4096):
    assert plan_weight_ring(m, 4096, 128, 128, 512, 8, F8)[0] is None
    assert plan_weight_ring(m, 4096, 1024, 128, 64, 8, F8)[0] is None
  assert plan_weight_ring(16, 6144, 2048, 128, 16, 16, F8) == (256, 6)
  assert vmem_bytes(256, 6144, 2048, 128, 16, 16, F8) <= have


@pytest.mark.parametrize("m", [8, 32, 128])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("i_block", [None, 128])
def test_scatter_rows_is_own_slice_of_all_reduce(setup, m, ep, i_block):
  """scatter_rows=True (DP attention: all-to-all + local sum, no all-gather).

  Returns on rank r exactly rows [r*m/N, (r+1)*m/N) of the all-reduce mode's
  output. The BF16 wire (m/N % 16 == 0) is bit-identical; the FP32 wire used for
  small row counts differs only by the all-reduce's BF16 rounding. i_block: the
  same in blocked mode.
  """
  if i_block and m != 128:
    pytest.skip("one blocked case per mode")
  mesh, sh, rank, weights, shared_gate_weight = setup
  rng = np.random.default_rng(100 + m)
  x = rng.normal(size=(m, H)).astype(np.float32)
  n_global = N * E if ep else E
  ids = np.stack(
      [rng.choice(n_global, TOPK, replace=False) for _ in range(m)]
  ).astype(np.int32)
  tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
  args = (
      rank,
      jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
      *weights,
      shared_gate_weight,
      jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
      jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
  )
  specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

  def call(r, x_in, *rest, scatter_rows):
    return fused_moe_decode(
        r.reshape(-1),
        x_in,
        *rest,
        n_ranks=N,
        expert_start=r.reshape(()) * E if ep else None,
        scatter_rows=scatter_rows,
        i_block=i_block,
    )

  def run(scatter_rows):
    out = jax.jit(
        jax.shard_map(
            functools.partial(call, scatter_rows=scatter_rows),
            mesh=mesh,
            in_specs=specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )(*args)
    return np.asarray(out.astype(jnp.float32))

  full = run(False).reshape(N, m, H)[0]
  own = run(True)
  assert own.shape == (m, H)
  assert np.isfinite(own).all()
  if (m // N) % 16 == 0:
    np.testing.assert_array_equal(own, full)
  else:
    rel = np.linalg.norm(own - full, axis=1) / np.maximum(
        np.linalg.norm(full, axis=1), 1e-30
    )
    assert rel.max() < 1e-2
  # Positive control: a one-row shift of the slices must not match.
  assert not np.allclose(np.roll(own, 1, axis=0), full, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
  app.run(
      lambda argv: pytest.main(
          ["-o", "cache_dir=/tmp/pytest_cache", __file__, *argv[1:]]
      )
  )
