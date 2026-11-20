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
"""Regression for label-preserving interior-padding canonicalization."""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as shared

interpret_on_cpu = shared.interpret_on_cpu


@pytest.mark.parametrize("case", ["holes", "leading", "reused", "empty"])
@pytest.mark.parametrize("remat", [False, True])
@pytest.mark.parametrize("conv", [False, True])
def test_padding_output_state_and_gradients(case, remat, conv):
  args, kw = shared._inputs(jnp.bfloat16, True, heads=2, batch=2)
  patterns = {
      "holes": [1] * 17 + [0] * 20 + [1] * 27 + [2] * 31 + [0] * 33,
      "leading": [0] * 65 + [2] * 31 + [0] * 32,
      "reused": [2] * 17 + [1] * 31 + [2] * 48 + [0] * 32,
      "empty": [0] * 128,
  }
  kw["segment_ids"] = jnp.asarray([patterns[case]] * 2, jnp.int32)
  if conv:
    kw["use_conv1d_in_kernel"] = True
    for label, width in (("q", 128), ("k", 128), ("v", 128)):
      kw["conv_weight_" + label] = jnp.full((2, 3, width), 0.1, jnp.bfloat16)
      kw["conv_bias_" + label] = jnp.full((2, width), 0.01, jnp.bfloat16)
  config = mosaic.Config(
      canonicalize_packed_inputs=True, fuse_forward=True,
      rematerialize_for_backward=remat,
  )
  call = functools.partial(
      shared._call, mosaic.PallasMosaicTpuKimiDeltaAttention(config=config), kw
  )
  ref = functools.partial(
      shared._call,
      functools.partial(api.kimi_delta_attention, implementation="xla"),
      kw,
  )
  expected, pb = jax.vjp(ref, *args)
  actual, back = jax.vjp(call, *args)
  cot = jax.tree.map(lambda x: jnp.full_like(x, 0.1), expected)
  # Same tolerances as the existing fused forward/XLA contract test.
  shared._assert_close(actual, expected, tolerance=0.05)
  shared._assert_close(back(cot), pb(cot), tolerance=0.05)
  invalid = np.asarray(kw["segment_ids"]) == 0
  np.testing.assert_array_equal(np.asarray(actual[0])[:, invalid], 0)
  for grad in back(cot)[:5]:
    np.testing.assert_array_equal(np.asarray(grad)[:, invalid], 0)
