# Copyright 2026 Ant Group. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed gate-parameter gradients against an independent FP32 recurrence."""

import jax
import jax.numpy as jnp
from jax.experimental import checkify
import ml_dtypes
import numpy as np
import pytest
from tokamax._src.ops.kda import compact


def _inputs():
  # Preserve the failing benchmark's random stream, then select eight heads
  # including its three elementwise failures. This bounds reference memory.
  rng = np.random.default_rng(619)
  shape = (1, 8192, 32, 128)
  selected = np.array([8, 9, 10, 27, 24, 25, 26, 28])
  arrays = [
      (rng.standard_normal(shape, dtype=np.float32) * std)
      .astype(ml_dtypes.bfloat16)
      .transpose(2, 0, 1, 3)[selected]
      for std in (0.05, 0.05, 0.05, 0.1)
  ]
  logits = rng.standard_normal(shape[:-1], dtype=np.float32) * 0.1
  beta = (1 / (1 + np.exp(-logits))).astype(ml_dtypes.bfloat16)
  beta = beta.transpose(2, 0, 1)[selected]
  a = rng.uniform(-1, 0, 32).astype(np.float32)[selected]
  bias = rng.uniform(-4, -3, 32 * 128).astype(np.float32)
  bias = bias.reshape(32, 128)[selected].reshape(-1)
  dout = (rng.standard_normal(shape, dtype=np.float32) * 0.05).astype(
      ml_dtypes.bfloat16
  )
  dout = dout.transpose(2, 0, 1, 3)[selected]
  lengths = np.full(25, 8192 // 25, np.int32)
  lengths[: 8192 % 25] += 1
  ids = np.repeat(np.arange(1, 26, dtype=np.int32), lengths)[None]
  return (
      tuple(map(jnp.asarray, (*arrays, beta, a, bias))),
      jnp.asarray(ids),
      jnp.asarray(dout),
  )


def _reference(q, k, v, g, beta, a, bias, ids):
  """One active state suffices for these globally grouped labels."""
  heads, _, _, dim = q.shape
  gate = -5 * jax.nn.sigmoid(
      jnp.exp(a)[:, None, None, None]
      * (g.astype(jnp.float32) + bias.reshape(heads, 1, 1, dim))
  )
  xs = tuple(x[:, 0].transpose(1, 0, 2) for x in (q, k, v, gate))
  xs = (*xs, beta[:, 0].T, ids[0])

  def step(carry, x):
    state, previous = carry
    qi, ki, vi, gi, bi, label = x
    qi, ki, vi, gi, bi = (y.astype(jnp.float32) for y in (qi, ki, vi, gi, bi))
    state = jnp.where(label != previous, 0.0, state)
    state = state * jnp.exp(gi)[..., None]
    prediction = jnp.einsum(
        "hk,hkv->hv", ki, state, precision=jax.lax.Precision.HIGHEST
    )
    state = (
        state + (ki * bi[:, None])[..., None] * (vi - prediction)[:, None, :]
    )
    output = jnp.einsum(
        "hk,hkv->hv",
        qi * dim**-0.5,
        state,
        precision=jax.lax.Precision.HIGHEST,
    )
    return (state, label), output.astype(q.dtype)

  _, output = jax.lax.scan(
      jax.checkpoint(step),
      (jnp.zeros((heads, dim, v.shape[-1]), jnp.float32), jnp.int32(0)),
      xs,
  )
  return output.transpose(1, 0, 2)[:, None]


def test_packed_gate_parameters_keep_fp32_until_reduction():
  if jax.default_backend() != "tpu":
    pytest.skip("The compact M5 kernel requires native TPU")
  inputs, ids, dout = _inputs()

  def actual_call(*xs):
    options = dict(
        a_log=xs[5],
        delta_time_bias=xs[6],
        segment_ids=ids,
        max_num_segments=25,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        scale=128**-0.5,
    )
    output, tape = compact.forward(
        *xs[:5], return_residuals=True, sorted_segments=True, **options
    )
    grads = compact.backward(
        tape,
        (dout, None),
        query=xs[0],
        value=xs[2],
        gate=xs[3],
        a_log=xs[5],
        delta_time_bias=xs[6],
        scale=128**-0.5,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
    )
    names = (
        "query",
        "key",
        "value",
        "gate",
        "beta",
        "a_log",
        "delta_time_bias",
    )
    return output[0], tuple(
        grads[name].astype(x.dtype) for name, x in zip(names, xs, strict=True)
    )

  def evaluate(fn, *xs):
    output, back = jax.vjp(fn, *xs)
    return output, back(dout)

  errors, actual = jax.jit(checkify.checkify(actual_call))(*inputs)
  errors.throw()
  expected = jax.jit(
      lambda *xs: evaluate(lambda *ys: _reference(*ys, ids), *xs)
  )(*inputs)
  for index, (lhs, rhs) in enumerate(
      zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
  ):
    lhs, rhs = np.asarray(lhs, np.float64), np.asarray(rhs, np.float64)
    assert np.isfinite(lhs).all() and np.isfinite(rhs).all()
    norm = np.linalg.norm(rhs)
    # Match the existing compact-backward norm and elementwise budgets.
    if norm > 1e-8:
      assert np.linalg.norm(lhs - rhs) / norm <= (0.05 if index >= 6 else 0.02)
    else:
      np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-7)
    np.testing.assert_allclose(lhs, rhs, rtol=0.05, atol=1e-4)
