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
"""Capacity is logical API state space, not the number of local CP tokens."""

import dataclasses
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.kda import packing


@dataclasses.dataclass(frozen=True)
class Config:
  canonicalize_packed_inputs: bool = True


class CapacityProbe:

  def __init__(self, config=None, bypass_device_check=False):
    self.config = config or Config()
    self.bypass_device_check = bypass_device_check

  def __call__(self, **options):
    return jnp.full_like(options["query"], options["max_num_segments"]), None


def test_default_cp_preserves_logical_capacity():
  context = SimpleNamespace(is_cp_enabled=True, axis_name="context")

  def run(ids):
    x = jnp.ones((1, 1, 4, 1), jnp.float32)
    arguments = {name: None for name in packing.ARRAY_NAMES}
    arguments.update(
        query=x,
        key=x,
        value=x,
        gate=x,
        beta=x[..., 0],
        initial_state=None,
        output_final_state=False,
        use_conv1d_in_kernel=False,
        segment_ids=ids,
        max_num_segments=512,
        return_residuals=False,
        context_parallel_metadata=context,
    )
    return packing.invoke(CapacityProbe(), Config(), arguments)[0][0]

  actual = jax.jit(jax.vmap(run, axis_name="context"))(
      jnp.full((1, 1, 4), 500, jnp.int32)
  )
  np.testing.assert_array_equal(np.asarray(actual), 512)


def test_short_convolution_ignores_nonexistent_history():
  for tokens in (1, 2):
    x = jnp.ones((1, 1, tokens, 2), jnp.float32)
    weight = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2) / 8
    bias = jnp.full((1, 2), 0.125, jnp.float32)
    ids = jnp.ones((1, tokens), jnp.int32)
    expected = x * weight[:, None, -1:, :] + bias[:, None, None, :]
    if tokens == 2:
      expected = (
          expected
          + jnp.pad(x[:, :, :-1], ((0, 0), (0, 0), (1, 0), (0, 0)))
          * weight[:, None, -2:-1, :]
      )
    expected = jax.nn.silu(expected)
    np.testing.assert_array_equal(packing._conv(x, weight, bias, ids), expected)
