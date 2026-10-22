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
"""Compare standalone native inference with the existing Mosaic forward."""

from absl import logging
from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
import tokamax
from tokamax._src.ops.kda import api, inference


class KdaInferenceBenchmark(parameterized.TestCase):

  @parameterized.product(native=(False, True), tokens=(512, 8192))
  def test_forward(self, native, tokens):
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU")
    keys = jax.random.split(jax.random.key(19), 4)
    q, k, v, g = [
        jax.random.normal(key, (1, tokens, 8, 128), jnp.bfloat16)
        for key in keys
    ]
    beta = jnp.full((1, tokens, 8), 0.5, jnp.bfloat16)
    ids = jnp.where(
        jnp.arange(tokens)[None, :] < tokens // 2 - 17, 1, 2
    ).astype(jnp.int32)
    kwargs = dict(
        segment_ids=ids,
        a_log=jnp.zeros((8,)),
        delta_time_bias=jnp.full((1024,), -2.0),
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        max_num_segments=2,
    )

    def run(q, k, v, g, beta):
      if native:
        return inference.kimi_delta_attention_inference(
            q, k, v, g, beta, **kwargs, use_qk_l2norm_in_kernel=True
        )
      return api.kimi_delta_attention(
          *[x.transpose(2, 0, 1, 3) for x in (q, k, v, g)],
          beta.transpose(2, 0, 1),
          **kwargs,
          use_qk_l2norm=True,
          implementation="mosaic",
      )

    result = tokamax.benchmark(jax.jit(run), (q, k, v, g, beta))
    logging.info(
        "device=%s native=%s T=%s median_ms=%s",
        jax.devices()[0].device_kind,
        native,
        tokens,
        result.median_evaluation_time_ms,
    )


if __name__ == "__main__":
  absltest.main()
