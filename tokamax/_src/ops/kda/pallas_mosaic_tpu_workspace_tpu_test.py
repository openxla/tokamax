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

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest
from tokamax._src import jaxtyping
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_workspace as workspace
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata


@pytest.mark.parametrize(
    "cp,batch,remat,over,padding,capacity,outer_remat",
    [
        (4, 1, False, False, False, 8192, False),
        (8, 2, False, True, False, 262144, False),
        (8, 2, True, False, False, 262144, False),
        (8, 1, False, False, True, 8192, False),
        (8, 1, False, False, "rank", 8192, False),
        (8, 1, False, False, "all", 262144, False),
        (8, 2, False, True, False, 262144, True),
    ],
)
def test_public_cp_workspace(
    cp, batch, remat, over, padding, capacity, outer_remat
):
  if jax.default_backend() != "tpu" or jax.device_count() < cp:
    pytest.skip("Requires a native TPU mesh of the requested CP size")
  heads, local_tokens, dim = 4, 128, 128
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
  if over:
    ids[-1, -local_tokens + 32 : -local_tokens + 65] = 2 * cp
  if padding == "rank":
    ids[:, local_tokens * (cp // 2) : local_tokens * (cp // 2 + 1)] = 0
  elif padding == "all":
    ids[:] = 0
  elif padding:
    ids[-1, -31:] = 0
  ids = jnp.asarray(ids)
  mesh = Mesh(np.array(jax.devices()[:cp]), ("context",))
  meta = ContextParallelMetadata(mesh=mesh, axis_name="context")
  config = mosaic.Config(
      fuse_forward=True,
      fuse_backward=True,
      fuse_cp_backward=True,
      fuse_rematerialization=True,
      rematerialize_for_backward=remat,
  )
  attention = mosaic.PallasMosaicTpuKimiDeltaAttention(config=config)
  spec = P(None, None, "context", None)
  inputs = (q, k, v, g, beta, a, bias, dout, ids)
  results = []
  for route in ("reference_capacity", "bucket", "grouped"):

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
        if route == "reference_capacity":
          return attention(q, k, v, g, beta, max_num_segments=32, **options)[0]
        return workspace.kimi_delta_attention_cp(
            q,
            k,
            v,
            g,
            beta,
            config=config,
            max_num_segments=capacity,
            head_group_size=2,
            fast_segment_capacity=2 if route == "bucket" else 0,
            **options,
        )[0]

      if outer_remat:
        call = jax.checkpoint(call)
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
      actual, expected = np.asarray(actual, np.float64), np.asarray(
          expected, np.float64
      )
      assert np.isfinite(actual).all() and np.isfinite(expected).all()
      rms = np.sqrt(np.mean((actual - expected) ** 2))
      reference_rms = max(np.sqrt(np.mean(expected**2)), 1e-30)
      assert rms / reference_rms <= 1e-5
