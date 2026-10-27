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
"""Inference-only Kimi Delta Attention forward operator.

This module deliberately does not import ``chunk.py`` or register a custom
VJP.  It provides a small public API around the native segment-ID Pallas
mega-kernel so inference optimization cannot change the training path.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from tokamax._src import jaxtyping

from tokamax._src.ops.kda.pallas_mosaic_tpu_inference_kernel import _chunk_kda_fwd_native_segids


@jaxtyping.jaxtyped
def kimi_delta_attention_inference(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    v: Float[Array, "B T H V"],
    g: Float[Array, "B T H K"],
    beta: Float[Array, "B T H"],
    *,
    segment_ids: Int[Array, "B T"] | Int[Array, "T"],
    a_log: Float[Array, "H"] | None = None,
    delta_time_bias: Float[Array, "H*K"] | None = None,
    scale: float | None = None,
    initial_state: (
        Float[Array, "B N H K V"] | Float[Array, "N H K V"] | None
    ) = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    max_num_segments: int | None = None,
) -> tuple[Float[Array, "B T H V"], Float[Array, "B N H K V"] | None]:
  """Run the standalone KDA inference forward kernel.

  Args:
    q: Query tensor ``[B, T, H, K]`` in BF16.
    k: Key tensor ``[B, T, H, K]`` with the same dtype as ``q``.
    v: Value tensor ``[B, T, H, V]`` with the same dtype as ``q``.
    g: Raw gate tensor ``[B, T, H, K]`` when
      ``use_gate_in_kernel=True``; otherwise an activated log-space gate.
    beta: Update coefficient ``[B, T, H]``.
    segment_ids: One-indexed packed-sequence IDs ``[B, T]`` or ``[T]``;
      zero denotes padding.
    a_log: Per-head log decay ``[H]``. Required for fused gate activation.
    delta_time_bias: Flattened gate bias ``[H*K]``. Required for fused gate activation.
    scale: Attention scale. Defaults to ``K**-0.5``.
    initial_state: Optional FP32 state ``[B, N, H, K, V]`` or
      ``[N, H, K, V]``.
    output_final_state: Return FP32 final state ``[B, N, H, K, V]``.
    use_qk_l2norm_in_kernel: Must be True. Normalization bounds the
      triangular-solve coefficients.
    use_gate_in_kernel: Must be True so gate decay is bounded in the kernel.
    safe_gate: Must be True.
    lower_bound: Sigmoid-gate lower bound in the safe range ``[-5, 0)``.
    max_num_segments: Static maximum number of packed segments.

  Returns:
    A pair ``(output, final_state)``. Output is ``[B, T, H, V]`` and has
    the input dtype. ``final_state`` is FP32 when requested, otherwise None.

  Raises:
    ValueError: If shapes, dtypes, or static inference options are invalid.
  """
  if any(d <= 0 for d in q.shape) or v.shape[-1] <= 0:
    raise ValueError("tensor dimensions must be positive")
  if q.dtype != jnp.bfloat16:
    raise ValueError(
        f"kimi_delta_attention_inference currently requires BF16, got {q.dtype}"
    )
  if not (q.dtype == k.dtype == v.dtype == g.dtype):
    raise ValueError("q, k, v, and g must have the same dtype")
  if beta.dtype != q.dtype:
    raise ValueError("beta must have the same dtype as q")
  if q.shape[:3] != k.shape[:3] or q.shape != g.shape:
    raise ValueError("q, k, and g must agree on [B, T, H, K]")
  if q.shape[:3] != v.shape[:3] or q.shape[:3] != beta.shape:
    raise ValueError("v and beta must agree with q on [B, T, H]")
  if q.shape[-1] != k.shape[-1]:
    raise ValueError("q and k must have the same key dimension")
  if q.shape[1] % 64:
    raise ValueError("T must be padded to a multiple of 64")
  if segment_ids.ndim == 1:
    # A shared [T] segment map covers every batch element, so broadcast it
    # to the documented [B, T] form. The native kernel already accepts this
    # broadcast semantics for a 1D map; only the shape check rejected it.
    segment_ids = jnp.broadcast_to(
        segment_ids[None, :], (q.shape[0], segment_ids.shape[0])
    )
  if segment_ids.shape != q.shape[:2]:
    raise ValueError(
        f"segment_ids must have shape {q.shape[:2]}, got {segment_ids.shape}"
    )
  if segment_ids.dtype != jnp.int32:
    raise ValueError(f"segment_ids must use int32, got {segment_ids.dtype}")
  if use_gate_in_kernel and (a_log is None or delta_time_bias is None):
    raise ValueError("fused gate activation requires a_log and delta_time_bias")
  if not use_qk_l2norm_in_kernel:
    raise ValueError(
        "standalone inference requires in-kernel Q/K L2 normalization "
        "to bound the triangular solve"
    )
  if not use_gate_in_kernel or not safe_gate:
    raise ValueError("standalone inference requires safe fused gate activation")
  if lower_bound is None or not (-5 <= lower_bound < 0):
    raise ValueError(
        "safe fused gate activation requires lower_bound in [-5, 0)"
    )
  if max_num_segments is None:
    if initial_state is not None:
      max_num_segments = initial_state.shape[-4]
    else:
      raise ValueError(
          "max_num_segments is required when initial_state is None"
      )
  if max_num_segments <= 0:
    raise ValueError("max_num_segments must be positive")
  if a_log.shape != (q.shape[2],) or delta_time_bias.shape != (
      q.shape[2] * q.shape[-1],
  ):
    raise ValueError("gate parameters must match the head and key dimensions")
  if initial_state is not None and initial_state.shape not in (
      (q.shape[0], max_num_segments, q.shape[2], q.shape[-1], v.shape[-1]),
      (max_num_segments, q.shape[2], q.shape[-1], v.shape[-1]),
  ):
    raise ValueError(
        "initial_state shape must match batch, segments, heads, K and V"
    )
  if initial_state is not None and initial_state.dtype != jnp.float32:
    raise ValueError("initial_state must use float32")

  # The native inference kernel clips beta before the Stage 2 solve. Keeping
  # the batch-first tensor here avoids a separate head-first layout module.
  # exp(80) is finite in FP32 and already far beyond the sigmoid saturation
  # point. This preserves valid model values while preventing exp(a_log) from
  # becoming inf before gate activation.
  a_log = jnp.minimum(a_log, jnp.asarray(80.0, dtype=a_log.dtype))

  actual_scale = q.shape[-1] ** -0.5 if scale is None else scale
  output, final_state, *_ = _chunk_kda_fwd_native_segids(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      segment_ids=segment_ids,
      initial_state=initial_state,
      output_final_state=output_final_state,
      scale=actual_scale,
      chunk_size=64,
      store_h=False,
      store_v_new=False,
      disable_recompute=False,
      only_fwd=True,
      safe_gate=safe_gate,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      use_gate_in_kernel=use_gate_in_kernel,
      A_log=a_log,
      dt_bias=delta_time_bias,
      lower_bound=lower_bound,
      mini_batch=None,
      N_max=max_num_segments,
      residual_chunk_layout=False,
      batch_first=True,
  )
  return output, final_state
