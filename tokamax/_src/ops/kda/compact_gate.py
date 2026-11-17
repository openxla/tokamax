# Copyright 2026 Ant Group. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Ported from primatrix/pallas-kernel dca6a09f.
import jax
import jax.numpy as jnp


def kda_gate_bwd(
    g: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array | None = None,
    dyg: jax.Array | None = None,
    lower_bound: float | None = None,
    head_first: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Backward pass for the KDA gate function.

  Computes gradients w.r.t. the original gate input g, the log-parameter
  A_log, and the optional dt_bias.

  Forward semantics (for reference):
    - Without lower_bound:  yg = -exp(A_log) * softplus(g + dt_bias)
    - With lower_bound:     yg = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))

  Args:
      g:           [H, B, T, K] — original gate input (before activation),
                    heads-first layout.
      A_log:       [H]           — log of the A parameter.
      dt_bias:     [H * K] or None — optional bias added to g.
      dyg:         [H, B, T, K] — upstream gradient w.r.t. yg.
      lower_bound: float or None — if set, use sigmoid mode.

  Returns:
      dg:    [H, B, T, K] — gradient w.r.t. g (cast to g.dtype).
      dA:    [H]          — gradient w.r.t. A_log.
      dbias: [H * K] or None — gradient w.r.t. dt_bias.
  """
  H, K = (g.shape[0] if head_first else g.shape[2]), g.shape[-1]
  layout = "[H, B, T, K]" if head_first else "[B, T, H, K]"
  assert g.ndim == 4, f"g must be 4-D {layout}, got {g.ndim}-D"
  assert dyg is not None, "dyg must be provided"
  assert A_log.shape == (H,), f"A_log shape {A_log.shape} != ({H},)"

  g_f = g.astype(jnp.float32)
  dyg_f = dyg.astype(jnp.float32)
  head_broadcast = (H, 1, 1, 1) if head_first else (1, 1, H, 1)

  # Apply bias if present
  if dt_bias is not None:
    assert dt_bias.shape == (
        H * K,
    ), f"dt_bias shape {dt_bias.shape} != ({H * K},)"
    bias_shape = (H, 1, 1, K) if head_first else (1, 1, H, K)
    g_f = g_f + dt_bias.reshape(bias_shape).astype(jnp.float32)

  if lower_bound is None:
    # Forward: yg = -exp(A_log) * softplus(g + bias)
    # softplus(x) = log(1 + exp(x)), d/dx softplus(x) = sigmoid(x)
    b_A = -jnp.exp(A_log.astype(jnp.float32))  # [H]
    b_yg = b_A.reshape(head_broadcast) * jax.nn.softplus(g_f)
    dg_f = b_A.reshape(head_broadcast) * (dyg_f * jax.nn.sigmoid(g_f))
    dA_per_elem = dyg_f * b_yg  # [H, B, T, K]
  else:
    # Forward: yg = lower_bound * sigmoid(exp(A_log) * g)
    b_A = jnp.exp(A_log.astype(jnp.float32))  # [H]
    b_inner = b_A.reshape(head_broadcast) * g_f
    b_sig = jax.nn.sigmoid(b_inner)
    b_dsig = b_sig * (1.0 - b_sig)
    dg_f = dyg_f * (lower_bound * b_dsig) * b_A.reshape(head_broadcast)
    dA_per_elem = dg_f * g_f  # [H, B, T, K]

  # dA: reduce over all dims except H (axis 0) → [H]
  reduce_axes = (1, 2, 3) if head_first else (0, 1, 3)
  dA = jnp.sum(dA_per_elem, axis=reduce_axes).astype(A_log.dtype)

  # Cast dg back to input dtype
  dg = dg_f.astype(g.dtype)

  # dbias: sum dg over B, T (axes 1, 2) → [H, K] → [H*K]
  if dt_bias is not None:
    bias_reduce_axes = (1, 2) if head_first else (0, 1)
    dbias = jnp.sum(dg_f, axis=bias_reduce_axes).reshape(-1)  # [H*K]
    dbias = dbias.astype(dt_bias.dtype)
  else:
    dbias = None

  return dg, dA, dbias
