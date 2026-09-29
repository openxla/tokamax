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
"""Tests for Pallas Mosaic TPU GDN attention."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.causal_conv1d_gated_delta_rule import base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import pallas_mosaic_tpu
from tokamax._src.ops.causal_conv1d_gated_delta_rule import test_base


_OP = pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule


class GDNAttentionTest(test_base.CausalConv1dGatedDeltaRuleTestBase):

  def __init__(self, *args):
    super().__init__(
        *args,
        gdn_fn=_OP(),
        static_argnames=test_base.OP_STATIC_ARGNAMES,
    )


class GDNSecurityTest(test_base.CausalConv1dGatedDeltaRuleSecurityTestBase):

  def __init__(self, *args):
    super().__init__(*args, gdn_fn=_OP())


class GDNAutotuningTest(parameterized.TestCase):

  def _kwargs(self, n_kq, n_v):
    d_k, d_v, kernel_size = 128, 128, 4
    # Four one-token decodes: compiles fast, still admits several tile sizes.
    num_reqs = self.num_tokens
    dim_kq, dim_v = n_kq * d_k, n_v * d_v
    conv_dim = 2 * dim_kq + dim_v

    rngs = iter(jax.random.split(jax.random.key(0), 8))
    normal = lambda *shape: jax.random.normal(next(rngs), shape)
    q_loc = jnp.arange(num_reqs + 1)

    return dict(
        qkv=normal(num_reqs, conv_dim),
        b=normal(num_reqs, n_v),
        a=normal(num_reqs, n_v),
        conv_state=jnp.zeros((num_reqs + 1, kernel_size - 1, conv_dim)),
        recurrent_state=jnp.zeros((num_reqs + 1, n_v, d_k, d_v)),
        conv_weight=normal(conv_dim, 1, kernel_size),
        conv_bias=normal(conv_dim),
        a_log=normal(n_v),
        dt_bias=normal(n_v),
        query_start_loc=q_loc,
        # Slot 0 is the null block for padded tokens.
        state_indices=jnp.arange(1, num_reqs + 1),
        distribution=jnp.array([num_reqs] * 3, dtype=jnp.int32),
        seq_lens=jnp.ones((num_reqs,), dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

  def setUp(self):
    super().setUp()
    test_base.skip_if_unsupported(self)
    self.num_tokens = 4
    self.op = pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule()

  def _jit(self, config):
    static = ["n_kq", "n_v", "d_k", "d_v", "kernel_size"]
    return jax.jit(self.op.replace(config=config), static_argnames=static)

  # Qwen3.5-397B has 16 key/query heads and 64 value heads (d_k = d_v = 128).
  @parameterized.named_parameters(
      # TP8 shards heads 8 ways: 16 / 8 = 2 kq heads, 64 / 8 = 8 v heads.
      dict(testcase_name="qwen3_5_397b_tp8", n_kq=2, n_v=8),
      # DP8 keeps all heads on each chip; n_v >= 64 also exercises the
      # heuristic's reduced mixed-tile cap.
      dict(testcase_name="qwen3_5_397b_dp8", n_kq=16, n_v=64),
  )
  def test_autotuning_configs(self, n_kq, n_v):
    kwargs = self._kwargs(n_kq, n_v)
    configs = self.op.bind(**kwargs).autotuning_configs

    # The heuristic is always a candidate, so tuning can never lose to it.
    self.assertIn(pallas_mosaic_tpu.Config(), configs)

    states_ref, out_ref = base.CausalConv1dGatedDeltaRule()(**kwargs)
    lowerings = set()
    for config in configs:
      with self.subTest(f"{config=}"):
        # A larger tile would clamp back down to the token count and duplicate
        # a smaller candidate.
        self.assertLessEqual(config.decode_tile_size or 1, self.num_tokens)
        self.assertLessEqual(config.mixed_tile_size or 1, self.num_tokens)

        fn = self._jit(config)
        lowerings.add(fn.lower(**kwargs).as_text())
        states, out = fn(**kwargs)

        np.testing.assert_allclose(out, out_ref, rtol=2e-2, atol=2e-2)
        for state, state_ref in zip(states, states_ref, strict=True):
          np.testing.assert_allclose(state, state_ref, rtol=2e-2, atol=2e-2)

    # Tile sizes that differ must lower to different kernels. If they don't,
    # the config is being dropped before it reaches the kernel.
    self.assertGreater(len(lowerings), 1)


if __name__ == "__main__":
  absltest.main()
