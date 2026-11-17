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

try:
  from tokamax._src.ops.kda import api
except ModuleNotFoundError as error:
  if error.name != "tokamax._src.ops.kda":
    raise
  from tokamax._src.ops.experimental.kda import api


def make_example(tokens, native):
  """Return a single-argument benchmark callable and its batch-first inputs."""
  keys = jax.random.split(jax.random.key(19), 4)
  q, k, v, g = [
      jax.random.normal(key, (1, tokens, 8, 128), jnp.bfloat16) for key in keys
  ]
  beta = jnp.full((1, tokens, 8), 0.5, jnp.bfloat16)
  ids = jnp.where(jnp.arange(tokens)[None, :] < tokens // 2 - 17, 1, 2).astype(
      jnp.int32
  )
  kwargs = dict(
      segment_ids=ids,
      a_log=jnp.zeros((8,)),
      delta_time_bias=jnp.full((1024,), -2.0),
      use_gate_in_kernel=True,
      lower_bound=-5.0,
      max_num_segments=2,
  )

  def run(xs):
    if native:
      from tokamax._src.ops.kda import inference

      return inference.kimi_delta_attention_inference(
          *xs, **kwargs, use_qk_l2norm_in_kernel=True
      )
    q, k, v, g, beta = xs
    output, final_state = api.kimi_delta_attention(
        *[x.transpose(2, 0, 1, 3) for x in (q, k, v, g)],
        beta.transpose(2, 0, 1),
        **kwargs,
        use_qk_l2norm=True,
        implementation="mosaic",
    )
    return output.transpose(1, 2, 0, 3), final_state

  return run, (q, k, v, g, beta)


class KdaInferenceBenchmark(parameterized.TestCase):

  @parameterized.product(native=(False, True), tokens=(512, 8192))
  def test_forward(self, native, tokens):
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU")
    fn, args = make_example(tokens, native)

    result = tokamax.benchmark(jax.jit(fn), args)
    logging.info(
        "device=%s native=%s T=%s median_ms=%s",
        jax.devices()[0].device_kind,
        native,
        tokens,
        result.median_evaluation_time_ms,
    )


if __name__ == "__main__":
  absltest.main()
