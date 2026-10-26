# Copyright 2026 Ant Group. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native TPU M5 gradients against an independent recurrence."""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda.pallas_mosaic_tpu_compact_forward_test import operands, assert_reference


@pytest.fixture(autouse=True)
def native_tpu():
  if jax.default_backend() != "tpu":
    pytest.skip(
        "M5 scoped DMA requires native TPU; metadata/forward interpretation is tested separately"
    )


@pytest.mark.parametrize(
    "case,sorted_segments",
    [
        (c, False)
        for c in ["full", "two", "many", "reused", "holes", "padding", "empty"]
    ]
    + [(c, True) for c in ["full", "two", "many", "padding", "empty"]],
)
@pytest.mark.parametrize("raw_gate", [False, True])
def test_compact_gradients(case, sorted_segments, raw_gate):
  xs, ids = operands(case, heads=8)
  q, k, v, g, beta = xs
  if raw_gate:
    g = jnp.full(q.shape, -1.0, q.dtype)
    a_log = jnp.linspace(-1.0, 0.0, q.shape[0], dtype=jnp.float32)
    bias = jnp.full((q.shape[0] * q.shape[-1],), -2.0, jnp.float32)
  else:
    a_log = bias = None
  args = (q, k, v, g, beta, a_log, bias)
  kwargs = dict(
      segment_ids=ids,
      max_num_segments=262144,
      use_gate_in_kernel=raw_gate,
      lower_bound=-5.0 if raw_gate else None,
  )
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(
          compact_forward=True, compact_sorted_segments=sorted_segments
      )
  )
  reference = functools.partial(api.kimi_delta_attention, implementation="xla")

  def run(fn, dynamic_ids, *args):
    def call(q, k, v, g, beta, a_log, bias):
      options = dict(kwargs)
      options["segment_ids"] = dynamic_ids
      if fn is reference:
        options["max_num_segments"] = (
            128 if case == "many" else 8 if case == "reused" else 2
        )
      return fn(q, k, v, g, beta, a_log=a_log, delta_time_bias=bias, **options)[
          0
      ]

    output, back = jax.vjp(call, *args)
    return output, back(jnp.full_like(output, 0.125))

  actual = jax.jit(lambda dynamic_ids, *args: run(op, dynamic_ids, *args))(
      ids, *args
  )
  expected = jax.jit(
      lambda dynamic_ids, *args: run(reference, dynamic_ids, *args)
  )(ids, *args)
  jax.block_until_ready(actual)
  for index, (a, b) in enumerate(
      zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
  ):
    # Retain the pinned source's 5% gate-parameter budget; source/port A/B
    # separately checks that the migration introduces no numerical drift.
    assert_reference(a, b, relative_rms_limit=0.05 if index >= 6 else 0.02)
  invalid = np.asarray(ids) == 0
  for gradient in actual[1][:5]:
    np.testing.assert_array_equal(np.asarray(gradient)[:, invalid], 0)


@pytest.mark.parametrize(
    "batch,heads,key_dim,value_dim",
    [(2, 8, 128, 256), (1, 8, 256, 128), (1, 1, 64, 192)],
)
def test_compact_gradient_layouts(batch, heads, key_dim, value_dim):
  xs, ids = operands(
      "two", batch=batch, heads=heads, key_dim=key_dim, value_dim=value_dim
  )
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(compact_forward=True, compact_sorted_segments=True)
  )
  reference = functools.partial(api.kimi_delta_attention, implementation="xla")

  def evaluate(fn, dynamic_ids, *xs):
    call = lambda *values: fn(
        *values, segment_ids=dynamic_ids, max_num_segments=2
    )[0]
    y, pb = jax.vjp(call, *xs)
    return y, pb(jnp.full_like(y, 0.125))

  actual = jax.jit(lambda dynamic_ids, *xs: evaluate(op, dynamic_ids, *xs))(
      ids, *xs
  )
  expected = jax.jit(
      lambda dynamic_ids, *xs: evaluate(reference, dynamic_ids, *xs)
  )(ids, *xs)
  for a, b in zip(
      jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
  ):
    assert_reference(a, b)


def test_one_executable_accepts_new_segment_patterns():
  xs, _ = operands("two", heads=8)
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(compact_forward=True)
  )
  reference = functools.partial(api.kimi_delta_attention, implementation="xla")

  def evaluate(fn, ids, *xs):
    call = lambda *values: fn(*values, segment_ids=ids, max_num_segments=8)[0]
    y, pb = jax.vjp(call, *xs)
    return y, pb(jnp.full_like(y, 0.125))

  actual_fn = (
      jax.jit(functools.partial(evaluate, op))
      .lower(jnp.ones((1, 128), jnp.int32), *xs)
      .compile()
  )
  expected_fn = (
      jax.jit(functools.partial(evaluate, reference))
      .lower(jnp.ones((1, 128), jnp.int32), *xs)
      .compile()
  )
  for case in ["full", "holes", "reused", "empty"]:
    _, ids = operands(case, heads=8)
    actual, expected = actual_fn(ids, *xs), expected_fn(ids, *xs)
    for a, b in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
      assert_reference(a, b)
