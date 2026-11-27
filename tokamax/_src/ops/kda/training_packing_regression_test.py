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
"""Training packing shares one pullback without changing padding semantics."""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import api, pallas_mosaic_tpu as mosaic, utils
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as shared

interpret_on_cpu = shared.interpret_on_cpu


@pytest.mark.parametrize("zero_guard", [False, True])
@pytest.mark.parametrize("capacity", [2, 8, 9, 25, 8193])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_unalignment_masks_unwritten_padding(capacity, scalar, permuted, zero_guard):
  lengths = jnp.array([0, 2, 3] + [3] * (capacity - 2), jnp.int32)
  aligned = jnp.array([0, 4, 8] + [8] * (capacity - 2), jnp.int32)
  x = jnp.array([1, 2, np.nan, np.nan, 3] + [np.nan] * 7)[None, None]
  if not scalar:
    x = x[..., None]
  order = jnp.array([3, 1, 0, 2]) if permuted else None
  fn = functools.partial(utils._unalign_output, zero_guard=zero_guard)
  actual = jax.jit(fn, static_argnums=3)(x, lengths, aligned, 4, order)
  expected = jnp.array([1, 2, 3, 0])[None, None]
  if permuted:
    expected = expected[:, :, order]
  if not scalar:
    expected = expected[..., None]
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "layout,heads,canonicalize",
    [
        ("canonical", 2, False),
        ("canonical", 2, True),
        ("holes", 2, True),
        ("interleaved", 2, True),
        ("empty", 2, True),
        ("canonical", 32, False),
        ("canonical", 32, True),
        ("holes", 32, True),
    ],
)
@pytest.mark.parametrize("normalize", [False, True])
def test_runtime_packed_training(layout, heads, normalize, canonicalize, monkeypatch):
  if not canonicalize:
    def unexpected_adapter(*args, **kwargs):
      pytest.fail("Default aligned execution entered the packing adapter")
    monkeypatch.setattr(mosaic.packing, "invoke", unexpected_adapter)
  packing_options = {"canonicalize_packed_inputs": True} if canonicalize else {}
  args, options = shared._inputs(jnp.bfloat16, True, heads=heads, batch=2)
  rows = {
      "canonical": [1] * 17 + [2] * 79 + [0] * 32,
      "holes": [1] * 17 + [0] * 16 + [2] * 63 + [0] * 32,
      "interleaved": [1, 2] * 48 + [0] * 32,
      "empty": [0] * 128,
  }
  ids = jnp.array([rows[layout], rows["canonical"]], jnp.int32)
  options = {**options, "use_qk_l2norm": normalize}
  results = []
  for fn in (
      mosaic.PallasMosaicTpuKimiDeltaAttention(config=mosaic.Config(**packing_options)),
      mosaic.PallasMosaicTpuKimiDeltaAttention(
          config=mosaic.Config(
              fuse_forward=True, packed_forward=True, packed_output=True, **packing_options
          )
      ),
      functools.partial(api.kimi_delta_attention, implementation="xla"),
  ):

    def run(ids, *xs):
      call = functools.partial(shared._call, fn, {**options, "segment_ids": ids})
      y, pb = jax.vjp(call, *xs)
      return y, pb(jax.tree.map(lambda x: jnp.full_like(x, 0.1), y))

    results.append(jax.jit(run)(ids, *args))
  shared._assert_close(results[1], results[0], tolerance=0.002)
  shared._assert_close(results[1], results[2], tolerance=0.05)
  for grad in results[1][1][:5]:
    mask = (ids == 0)[None]
    if grad.ndim == 4:
      mask = mask[..., None]
    mask = jnp.broadcast_to(mask, grad.shape)
    np.testing.assert_array_equal(np.asarray(grad)[np.asarray(mask)], 0)


@pytest.mark.parametrize("lengths", [(1, 1, 1), (4, 4, 4), (0, 0, 0), (1, 0, 3)])
@pytest.mark.parametrize("extra", [0, 1])
@pytest.mark.parametrize("scalar", [False, True])
def test_zero_guard_preserves_valid_rows_and_padding_gradients(lengths, extra, scalar):
  tokens = max(1, sum(lengths)) + extra
  cu = jnp.array((0,) + tuple(np.cumsum(lengths)), jnp.int32)
  padded = jnp.array(
      (0,) + tuple(np.cumsum([(x + 3) // 4 * 4 for x in lengths])), jnp.int32
  )
  allocated = utils._aligned_token_bound(tokens, len(lengths), 4)
  shape = (2, 1, allocated) + (() if scalar else (3,))
  raw = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
  expected_fn = functools.partial(utils._unalign_output, T_out=tokens)
  expected = expected_fn(raw, cu, padded)
  valid = jnp.zeros(allocated, dtype=bool)
  for i, length in enumerate(lengths):
    start = int(padded[i])
    valid = valid.at[start : start + length].set(True)
  mask = valid.reshape((1, 1, allocated) + (() if scalar else (1,)))
  poisoned = jnp.where(mask, raw, jnp.nan)
  fn = functools.partial(utils._unalign_output, T_out=tokens, zero_guard=True)
  actual, pullback = jax.vjp(lambda x: fn(x, cu, padded), poisoned)
  np.testing.assert_array_equal(actual, expected)
  grad = pullback(jnp.ones_like(actual))[0]
  np.testing.assert_array_equal(jnp.where(mask, 0, grad), jnp.zeros_like(grad))
