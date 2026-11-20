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
"""Leading padding must not expose allocator contents as recurrent state."""

import inspect
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import pytest
from tokamax._src.ops.kda import compact, compact_forward, api
from tokamax._src.ops.kda.pallas_mosaic_tpu_compact_forward_test import operands, assert_reference


def test_leading_padding_clears_poisoned_carry(monkeypatch):
  if jax.default_backend() != "tpu":
    pytest.skip("requires native TPU scratch semantics")
  original = compact_forward._fwd_mega_kernel_native_segids
  names = list(inspect.signature(original).parameters)

  def poisoned(*refs, **kwargs):
    scratch = refs[names.index("scratch_ref")]
    previous = refs[names.index("prev_seg_ref")]

    @pl.when(pl.program_id(2) == 0)
    def poison():
      scratch[...] = jnp.full(scratch.shape, jnp.nan, jnp.float32)
      previous[...] = jnp.ones(previous.shape, jnp.int32)

    return original(*refs, **kwargs)

  jax.clear_caches()
  try:
    monkeypatch.setattr(
        compact_forward, "_fwd_mega_kernel_native_segids", poisoned
    )
    xs, ids = operands("leading", heads=2)
    actual = compact.forward(
        *xs,
        a_log=None,
        delta_time_bias=None,
        scale=128**-0.5,
        segment_ids=ids,
        lower_bound=None,
        use_gate_in_kernel=False,
        return_residuals=True,
        sorted_segments=False,
        max_num_segments=128
    )[0][0]
    jax.block_until_ready(actual)
    expected = api.kimi_delta_attention(
        *xs, segment_ids=ids, max_num_segments=128, implementation="xla"
    )[0]
    assert_reference(actual, expected)
  finally:
    jax.clear_caches()
