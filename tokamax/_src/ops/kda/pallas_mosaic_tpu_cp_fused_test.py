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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from tokamax._src import jaxtyping
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as f
from tokamax._src.ops.kda import pallas_mosaic_tpu_bwd_kernel as kernels

interpret_on_cpu = f.interpret_on_cpu


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("state_policy", ["saved", "remat", "remat_staged"])
def test_cp_local_fusion(dtype, split, state_policy, monkeypatch):
  if jax.device_count() < 2:
    pytest.skip(
        "Requires two devices; CPU interpret uses XLA host device count=2"
    )
  mesh = Mesh(np.array(jax.devices()[:2]), ("context",))
  meta = ContextParallelMetadata(mesh=mesh, axis_name="context")
  keys = jax.random.split(jax.random.key(97), 3)
  q = (jax.random.normal(keys[0], (2, 2, 256, 128)) * 0.05).astype(dtype)
  k = (jax.random.normal(keys[1], q.shape) * 0.05).astype(dtype)
  v = jax.random.normal(keys[2], q.shape).astype(dtype)
  g = jnp.full_like(q, -0.01)
  beta = jnp.full(q.shape[:-1], 0.5, dtype)
  segments = (
      jnp.array([[1] * 64 + [2] * 192, [1] * 192 + [2] * 64], jnp.int32)
      if split
      else jnp.ones((2, 256), jnp.int32)
  )
  # Record traced dispatch to ensure equality is not just staged-vs-staged.
  dispatch = []
  original_reverse = kernels._fused_dhu_wy_intra_cumsum_pallas_jit

  def record_reverse(*args, **kwargs):
    dispatch.append((kwargs["fuse_recompute"], kwargs["v_new"].dtype))
    return original_reverse(*args, **kwargs)

  monkeypatch.setattr(
      kernels, "_fused_dhu_wy_intra_cumsum_pallas_jit", record_reverse
  )
  # The outer jit's cache otherwise hides dispatch across parameter cases.
  kernels.chunk_kda_bwd_custom.clear_cache()
  results = []
  for fused in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, 
            fuse_cp_backward=fused,
            rematerialize_for_backward=state_policy != "saved",
            fuse_rematerialization=state_policy == "remat",
        )
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
  assert [enabled for enabled, _ in dispatch] == [
      False,
      state_policy != "remat_staged",
  ]
  if state_policy != "saved":
    assert all(dtype == jnp.float32 for _, dtype in dispatch)
  f._assert_close(
      results[1], results[0], tolerance=0.002 if dtype == jnp.bfloat16 else 1e-5
  )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dv_only_matches_staged(dtype):
  args, _ = f._inputs(dtype, False)
  q, k, v = args[:3]
  attention = jnp.broadcast_to(
      jnp.eye(64, dtype=dtype), (1, 1, 2, 64, 64)
  ).reshape(1, 1, 128, 64)
  do = jnp.ones_like(v)
  expected_da, expected_dv = kernels.chunk_kda_bwd_dAv_kernel(
      q, k, v, do, attention, scale=0.125
  )
  actual_da, actual_dv = kernels.chunk_kda_bwd_dAv_kernel(
      q, k, v, do, attention, scale=0.125, store_dAqk=False
  )
  assert expected_da is not None
  assert actual_da is None
  np.testing.assert_array_equal(actual_dv, expected_dv)
