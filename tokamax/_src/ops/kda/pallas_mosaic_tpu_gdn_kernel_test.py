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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import pallas_mosaic_tpu
from tokamax._src.ops.kda import pallas_mosaic_tpu_gdn_kernel


def _inputs(dtype, h_qk=2, h_v=4, b=2, t=128, d=128):
  keys = jax.random.split(jax.random.PRNGKey(0), 6)
  q = jax.random.normal(keys[0], (b, t, h_qk, d), dtype)
  k = jax.random.normal(keys[1], (b, t, h_qk, d), dtype)
  v = (0.5 * jax.random.normal(keys[2], (b, t, h_v, d))).astype(dtype)
  gate = -jax.nn.softplus(jax.random.normal(keys[3], (b, t, h_v)))
  beta = jax.random.normal(keys[4], (b, t, h_v)).astype(dtype)
  do = jax.random.normal(keys[5], (b, t, h_v, d), dtype)
  return (q, k, v, gate, beta), do


def _rel(a, b):
  a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


class GdnFusedForwardTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "gpu":
      self.skipTest("Pallas Mosaic TPU kernel is not supported on GPU.")
    if jax.default_backend() == "cpu":
      self.enterContext(
          mock.patch.object(
              pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet,
              "supported_on",
              return_value=True,
          )
      )

  @parameterized.product(h_qk=(2, 4), rematerialize=(False, True))
  def test_matches_xla(self, h_qk, rematerialize):
    args, do = _inputs(jnp.bfloat16, h_qk=h_qk)
    remat_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
        config=pallas_mosaic_tpu.Config(rematerialize_for_backward=True)
    )

    def run(implementation):
      def f(*a):
        kwargs = dict(use_beta_sigmoid_in_kernel=True, chunk_size=64)
        if implementation == "remat":
          out, _ = remat_op(*a, **kwargs)
        else:
          out, _ = api.gated_delta_net(
              *a, implementation=implementation, **kwargs
          )
        return out

      out, vjp = jax.vjp(f, *args)
      return out, vjp(do)

    with mock.patch.object(
        pallas_mosaic_tpu_gdn_kernel,
        "gdn_fwd",
        wraps=pallas_mosaic_tpu_gdn_kernel.gdn_fwd,
    ) as fused:
      out, grads = run("remat" if rematerialize else "mosaic")
    self.assertTrue(fused.called)
    out_ref, grads_ref = run("xla")
    self.assertLess(_rel(out, out_ref), 2e-2)
    for name, g, g_ref in zip(("q", "k", "v", "g", "beta"), grads, grads_ref):
      self.assertLess(_rel(g, g_ref), 2e-2, name)

  @parameterized.named_parameters(
      ("float32", jnp.float32, False), ("initial_state", jnp.bfloat16, True)
  )
  def test_not_used(self, dtype, with_initial_state):
    (q, k, v, gate, beta), _ = _inputs(dtype)
    h0 = (
        jnp.zeros((q.shape[0], v.shape[2], q.shape[-1], v.shape[-1]))
        if with_initial_state
        else None
    )
    with mock.patch.object(pallas_mosaic_tpu_gdn_kernel, "gdn_fwd") as fused:
      api.gated_delta_net(
          q,
          k,
          v,
          gate,
          beta,
          initial_state=h0,
          chunk_size=64,
          implementation="mosaic",
      )
    fused.assert_not_called()


if __name__ == "__main__":
  absltest.main()
