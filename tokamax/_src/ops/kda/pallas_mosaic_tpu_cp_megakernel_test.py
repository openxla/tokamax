# Copyright 2026 Ant Group. All Rights Reserved.
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
"""CP saved-state and rematerialized local fusion across real shard_map collectives."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from tokamax._src import jaxtyping
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as f
from tokamax._src.ops.kda import pallas_mosaic_tpu_bwd_kernel as kernels

interpret_on_cpu = f.interpret_on_cpu


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("state_policy", ["saved", "remat"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_megakernel(dtype, split, state_policy, cp_size):
  if jax.device_count() < cp_size:
    pytest.skip(
        "Requires two devices; CPU interpret uses XLA host device count=2"
    )
  mesh = Mesh(np.array(jax.devices()[:cp_size]), ("context",))
  meta = ContextParallelMetadata(mesh=mesh, axis_name="context")
  tokens = 128 * cp_size
  keys = jax.random.split(jax.random.key(97), 3)
  q = (jax.random.normal(keys[0], (2, 2, tokens, 128)) * 0.05).astype(dtype)
  k = (jax.random.normal(keys[1], q.shape) * 0.05).astype(dtype)
  v = jax.random.normal(keys[2], q.shape).astype(dtype)
  g = jnp.full_like(q, -0.01)
  beta = jnp.full(q.shape[:-1], 0.5, dtype)
  segments = (
      jnp.array(
          [[1] * 64 + [2] * (tokens - 64), [1] * (tokens - 64) + [2] * 64],
          jnp.int32,
      )
      if split
      else jnp.ones((2, tokens), jnp.int32)
  )
  kernels.chunk_kda_bwd_custom.clear_cache()
  results = []
  for fused in (False, True, "reference"):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, 
            cp_megakernel=fused is True,
            rematerialize_for_backward=state_policy != "saved",
            fuse_rematerialization=state_policy == "remat",
        )
    )

    if fused == "reference":
      op = lambda *args, **kwargs: api.kimi_delta_attention(
          *args, **kwargs, implementation="xla"
      )

    def local(q, k, v, g, beta, segments):
      def forward(q, k, v, g, beta):
        return op(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=segments,
            max_num_segments=2,
            context_parallel_metadata=meta,
        )[0]

      out, back = jax.vjp(forward, q, k, v, g, beta)
      return back(jnp.ones_like(out))

    spec = P(None, None, "context", None)
    # Match the existing CP suite's batched-metadata annotation workaround.
    with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
      grads = jax.jit(
          jax.shard_map(
              local,
              mesh=mesh,
              in_specs=(spec,) * 4
              + (P(None, None, "context"), P(None, "context")),
              out_specs=(spec,) * 4 + (P(None, None, "context"),),
              check_vma=False,
          )
      )(q, k, v, g, beta, segments)
    results.append(grads)
  f._assert_close(
      results[1], results[0], tolerance=0.01 if dtype == jnp.bfloat16 else 1e-5
  )

  # The source CP megakernel is a BF16 kernel. For BF16, check both paths
  # against the independent recurrent reference, not just each other. FP32
  # exercises the guarded staged fallback above; the staged-vs-XLA tolerance
  # belongs to the existing CP implementation rather than this megakernel.
  if dtype == jnp.bfloat16:
    for gradients in results[:2]:
      f._assert_close(gradients, results[2], tolerance=0.02)


@pytest.mark.parametrize("state_policy", ["saved", "remat"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_megakernel_initial_state_grad(state_policy, cp_size):
  """The CP megakernel must return dh0 for a caller-reported initial state.

  Regression test: ``return_dh0=False`` used to discard the initial-state
  gradient, so a VJP would substitute zeros and training could not propagate
  through the supplied state. The public op currently rejects CP with an
  initial_state, so this exercises the backward boundary directly.
  """
  if jax.device_count() < cp_size:
    pytest.skip("Requires enough devices for the CP mesh")
  mesh = Mesh(np.array(jax.devices()[:cp_size]), ("context",))
  meta = ContextParallelMetadata(mesh=mesh, axis_name="context")
  tokens = 128 * cp_size
  keys = jax.random.split(jax.random.key(103), 4)
  q = (jax.random.normal(keys[0], (2, 2, tokens, 128)) * 0.05).astype(jnp.bfloat16)
  k = (jax.random.normal(keys[1], q.shape) * 0.05).astype(jnp.bfloat16)
  v = jax.random.normal(keys[2], q.shape).astype(jnp.bfloat16)
  g = jnp.full_like(q, -0.01)
  beta = jnp.full(q.shape[:-1], 0.5, jnp.bfloat16)
  segments = jnp.ones((2, tokens), jnp.int32)
  # Simulated user-supplied initial state, varlen form [B, N, H, K, V].
  initial_state = jax.random.normal(
      keys[3], (2, 2, 2, 128, 128), jnp.float32
  ) * 0.05

  kernels.chunk_kda_bwd_custom.clear_cache()
  results = []
  for fused in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, 
            cp_megakernel=fused is True,
            rematerialize_for_backward=state_policy != "saved",
            fuse_rematerialization=state_policy == "remat",
        )
    )

    def local(q, k, v, g, beta, segments, initial_state):
      (o, _), residuals = op(
          q,
          k,
          v,
          g,
          beta,
          segment_ids=segments,
          max_num_segments=2,
          context_parallel_metadata=meta,
          return_residuals=True,
      )
      # The forward replaced the (absent) user state with the CP-prepared
      # zeros state; simulate a caller that did supply one.
      residuals = dataclasses.replace(residuals, initial_state=initial_state)
      result = kernels.chunk_kda_bwd_custom(
          128**-0.5,
          False,
          False,
          None,
          residuals.h is not None,
          meta,
          64,
          2,
          True,
          residuals,
          (jnp.ones_like(o), None),
          fuse_backward=True,
          fuse_rematerialization=state_policy == "remat",
          cp_megakernel=fused is True,
      )
      # Return a uniform pytree: the backward tuple keeps its None leaves
      # (dA, dbias, placeholder), which shard_map out_specs cannot describe.
      # The staged CP path frees initial_state before its state collective
      # and returns dh0=None; substitute zeros to keep the trees identical.
      dh0 = result[7]
      if dh0 is None:
        dh0 = jnp.zeros_like(initial_state)
      return (result[0], result[1], result[2], result[4], result[3], dh0)

    spec = P(None, None, "context", None)
    with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
      grads = jax.jit(
          jax.shard_map(
              local,
              mesh=mesh,
              in_specs=(spec,) * 4
              + (P(None, None, "context"), P(None, "context"), None),
              # (dq, dk, dv, db, dg, dh0): db is rank-3 [H, B, T] and the
              # replicated dh0 is [B, N, H, K, V].
              out_specs=(spec,) * 3
              + (P(None, None, "context"), spec, P(None, None)),
              check_vma=False,
          )
      )(q, k, v, g, beta, segments, initial_state)
    results.append(grads)

  # The fused CP megakernel must match the staged CP backward on the token
  # gradients (dq, dk, dv, db, dg). The staged path never returns dh0 (its
  # zeros placeholder occupies index 5), so the initial-state gradient is
  # asserted on the fused path directly.
  f._assert_close(results[1][:5], results[0][:5], tolerance=0.02)
  max_grad = jnp.max(jnp.abs(results[1][5]))
  assert max_grad > 1e-6, (
      "CP megakernel initial_state gradient is zero; dh0 was discarded"
  )
