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
"""Default packing composition retains full operand/parameter gradients."""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import api, pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as shared

interpret_on_cpu = shared.interpret_on_cpu


@pytest.mark.parametrize("holes", [False, True])
def test_canonicalized_convolution_parameter_vjp(holes):
  args, kwargs = shared._inputs(jnp.bfloat16, True, heads=2)
  if holes:
    kwargs["segment_ids"] = jnp.array(
        [[1] * 17 + [0] * 20 + [1] * 27 + [2] * 31 + [0] * 33], jnp.int32
    )
  # Exercise the short canonicalization path; long prefill keeps the fused implementation.
  args = tuple(x[:, :, :64] if i < 5 else x for i, x in enumerate(args))
  kwargs["segment_ids"] = kwargs["segment_ids"][:, :64]
  kwargs["use_conv1d_in_kernel"] = True
  keys = jax.random.split(jax.random.key(619), 6)
  weights = tuple(
      (jax.random.normal(key, (2, 3, 128)) * 0.1).astype(jnp.float32)
      for key in keys[:3]
  )
  biases = tuple(
      (jax.random.normal(key, (2, 128)) * 0.01).astype(jnp.float32)
      for key in keys[3:]
  )

  def call(fn, *values):
    operands, weights, biases = values[:8], values[8:11], values[11:]
    options = dict(kwargs)
    for n, w, b in zip(("q", "k", "v"), weights, biases, strict=True):
      options["conv_weight_" + n] = w
      options["conv_bias_" + n] = b
    return shared._call(fn, options, *operands)

  reference = functools.partial(api.kimi_delta_attention, implementation="xla")
  results = []
  for fn in (
      reference,
      mosaic.PallasMosaicTpuKimiDeltaAttention(
          config=mosaic.Config(canonicalize_packed_inputs=True)
      ),
  ):
    y, pb = jax.vjp(functools.partial(call, fn), *args, *weights, *biases)
    results.append((y, pb(jax.tree.map(lambda x: jnp.full_like(x, 0.1), y))))
  shared._assert_close(results[1], results[0], tolerance=0.05)
  for actual, expected in zip(
      results[1][1][8:], results[0][1][8:], strict=True
  ):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert np.isfinite(a).all()
    assert np.linalg.norm(a - e) <= 0.02 * np.linalg.norm(e) + 1e-7


@pytest.mark.parametrize("state", [False, True])
def test_sparse_large_logical_state_capacity(state):
  shape = (2, 2, 4, 8)
  keys = jax.random.split(jax.random.key(619), 6)
  xs = tuple(
      (jax.random.normal(k, shape) * 0.1).astype(jnp.float32) for k in keys[:4]
  )
  beta = jnp.full(shape[:-1], 0.25, jnp.float32)
  h0 = (jax.random.normal(keys[4], (2, 8, 2, 8, 8)) * 0.01) if state else None
  ids = jnp.array([[7, 0, 2, 7], [0, 3, 0, 3]], jnp.int32)
  options = dict(
      segment_ids=ids,
      max_num_segments=8,
      initial_state=h0,
      output_final_state=True,
      use_qk_l2norm=True,
  )
  g = -jax.nn.softplus(xs[3]) * 0.1
  operands = (*xs[:3], g, beta, h0)

  def call(fn, q, k, v, g, b, h):
    return fn(q, k, v, g, b, **{**options, "initial_state": h})

  results = []
  for fn in (
      functools.partial(api.kimi_delta_attention, implementation="xla"),
      mosaic.PallasMosaicTpuKimiDeltaAttention(
          config=mosaic.Config(canonicalize_packed_inputs=True)
      ),
  ):
    y, pb = jax.vjp(functools.partial(call, fn), *operands)
    results.append((y, pb(jax.tree.map(lambda x: jnp.full_like(x, 0.1), y))))
  shared._assert_close(results[1], results[0], tolerance=0.002)


def test_permutation_composes_with_alignment_and_its_pullback():
  from tokamax._src.ops.kda import utils, packing

  x = jnp.arange(96, dtype=jnp.float32).reshape(2, 2, 8, 3)
  order = jnp.array(
      [[2, 0, 3, 7, 1, 6, 4, 5], [5, 4, 0, 1, 2, 3, 6, 7]], jnp.int32
  )
  inverse = (
      jnp.zeros_like(order)
      .at[jnp.arange(2)[:, None], order]
      .set(jnp.arange(8)[None])
  )
  cu = jnp.array([[0, 2, 5, 8], [0, 1, 4, 8]], jnp.int32)
  expected, _, aligned, _ = utils._align_seqs(
      [packing._take(x, order)], [], cu, align=4
  )
  actual, _, _, _ = utils._align_seqs([x], [], cu, align=4, token_order=order)
  np.testing.assert_array_equal(actual[0], expected[0])
  y = jnp.arange(actual[0].size, dtype=jnp.float32).reshape(actual[0].shape)
  expected = packing._take(utils._unalign_output(y, cu, aligned, 8), inverse)
  actual = utils._unalign_output(y, cu, aligned, 8, token_order=inverse)
  np.testing.assert_array_equal(actual, expected)


def test_adjacent_metadata_check_matches_canonical_contract():
  import itertools
  from tokamax._src.ops.kda import packing

  rows = np.asarray(list(itertools.product(range(-1, 5), repeat=4)), np.int32)
  expected = []
  for row in rows.tolist():
    canonical = True
    last = 0
    seen_padding = False
    for label in row:
      if not 0 <= label <= 3:
        canonical = False
        break
      if label == 0:
        seen_padding = True
        continue
      if seen_padding:
        canonical = False
        break
      if label != last:
        if label != last + 1:
          canonical = False
          break
        last = label
    expected.append(not canonical)

  def check(row):
    ids = row[None]
    valid = (ids > 0) & (ids <= 3)
    clean = jnp.where(valid, ids, 0)
    return packing._needs_packing(ids, clean, valid)

  actual = jax.jit(jax.vmap(check))(jnp.asarray(rows))
  np.testing.assert_array_equal(actual, np.asarray(expected))
