# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import test_base

jax.config.update("jax_enable_x64", True)


class GatedLinearUnitTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    super().__init__(*args, glu_fn=base.GatedLinearUnit())

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float32, jnp.float64],
      precision=[
          # TODO: Enable this after figuring out why it gives large
          # differences on A100/H100 on FP32.
          # jax.lax.Precision.DEFAULT,
          jax.lax.Precision.HIGH,
          jax.lax.Precision.HIGHEST,
      ],
      batch_dim=[(), (2, 3, 1)],
  )
  def test_explicit_vjp(self, dtype, batch_dim, precision):
    activation = jax.nn.swish
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(key=rng0, shape=batch_dim + (2, 3), dtype=dtype)
    w = jax.random.normal(key=rng1, shape=(3, 2, 5), dtype=dtype)
    do = jax.random.normal(key=rng2, shape=batch_dim + (2, 5), dtype=dtype)

    f = functools.partial(
        base.GatedLinearUnit(vjp=base.GatedLinearUnitVjp()),
        activation=activation,
        precision=precision,
    )
    f_ref = functools.partial(
        base.GatedLinearUnit(), activation=activation, precision=precision
    )

    @jax.jit
    def f_vjp(x, w, do):
      out, vjp = jax.vjp(f, x, w)
      return out, vjp(do)

    @jax.jit
    def f_vjp_ref(x, w, do):
      out, vjp = jax.vjp(f_ref, x, w)
      return out, vjp(do)

    out, (dx, dw) = f_vjp(x, w, do)
    out_ref, (dx_ref, dw_ref) = f_vjp_ref(x, w, do)

    chex.assert_trees_all_close(out, out_ref)

    atol = 1e-09 if dtype == jnp.float64 else 2e-06
    chex.assert_trees_all_close(dw, dw_ref, atol=atol)
    chex.assert_trees_all_close(dx, dx_ref, atol=atol)


if __name__ == "__main__":
  absltest.main()
