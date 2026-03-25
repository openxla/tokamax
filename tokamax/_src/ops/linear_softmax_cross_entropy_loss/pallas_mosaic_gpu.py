# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pallas-Mosaic-GPU Op implementation of linear softmax cross-entropy loss.

Forward pass: SM90 WGMMA + TMA kernel (H100+).
Backward pass: chunked scan over V using cuBLAS GEMMs (no atomics, near-XLA speed).
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
from jax.extend import backend
from jaxtyping import Array, Integer, Real, Scalar
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
from tokamax._src.ops.linear_softmax_cross_entropy_loss import (
    pallas_mosaic_gpu_common as common,
)
import tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_gpu_kernel_sm90 as kernel_sm90
from typing_extensions import override


Config = common.Config
Key = common.Key


def linear_softmax_cross_entropy_loss_bwd_chunked_scan(
    dout,
    lse,
    x,
    labels,
    w,
    *,
    reduction,
    chunk_size=4096,
):
  """Chunked-scan backward: padded chunks for full cuBLAS utilisation.

  Uses chunk_size-wide GEMMs throughout — the last chunk is zero-padded and
  masked so padded positions contribute nothing to either gradient. This gives
  square GEMMs for any vocab size (including irregular sizes like V=128256).
  Never materialises the full (B, V) logit matrix.
  """
  b_dim, h_dim = x.shape
  v_dim = w.shape[1]

  x_f32 = x.astype(jnp.float32)
  w_f32 = w.astype(jnp.float32)
  lse_f32 = lse.astype(jnp.float32)
  scale = (
      dout.astype(jnp.float32) / b_dim
      if reduction == "mean"
      else dout.astype(jnp.float32)
  )

  num_chunks = (v_dim + chunk_size - 1) // chunk_size
  v_padded = num_chunks * chunk_size

  # Pad w to v_padded (last chunk may be partial; extra cols are zero).
  w_padded = jnp.pad(w_f32, ((0, 0), (0, v_padded - v_dim)))  # (H, v_padded)
  # Reshape into (num_chunks, H, chunk_size) for scan.
  w_chunks = w_padded.reshape(h_dim, num_chunks, chunk_size).transpose(1, 0, 2)

  def scan_fn(x_grad_carry, args):
    chunk_idx, w_chunk = args          # w_chunk: (H, chunk_size)
    v_start = chunk_idx * chunk_size
    logit_chunk = x_f32 @ w_chunk     # (B, chunk_size)
    softmax_chunk = jnp.exp(logit_chunk - lse_f32[:, None])
    col_idx = jnp.arange(chunk_size) + v_start
    one_hot_chunk = (col_idx[None, :] == labels[:, None]).astype(jnp.float32)
    # Zero out padded positions so they don't contribute to either gradient.
    valid = (col_idx < v_dim).astype(jnp.float32)[None, :]
    s_chunk = scale * (softmax_chunk - one_hot_chunk) * valid
    x_grad_carry = x_grad_carry + s_chunk @ w_chunk.T  # (B, H)
    w_grad_chunk = x_f32.T @ s_chunk                   # (H, chunk_size)
    return x_grad_carry, w_grad_chunk

  x_grad, w_grad_chunks = jax.lax.scan(
      scan_fn,
      jnp.zeros((b_dim, h_dim), dtype=jnp.float32),
      (jnp.arange(num_chunks), w_chunks),
  )
  # w_grad_chunks: (num_chunks, H, chunk_size) → (H, v_padded) → (H, V)
  w_grad = w_grad_chunks.transpose(1, 0, 2).reshape(h_dim, v_padded)[:, :v_dim]
  return x_grad, w_grad


def _mosaic_vjp(
    residuals: base.Residuals,
    out: jax.Array,
    dout: jax.Array,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    *,
    reduction: str = "sum",
    return_residuals: bool = False,
):
  """Mosaic GPU backward: chunked scan over V (no atomics, cuBLAS per chunk)."""
  del out, return_residuals
  (lse,) = residuals
  x_grad, w_grad = linear_softmax_cross_entropy_loss_bwd_chunked_scan(
      dout,
      lse,
      x,
      labels,
      w,
      reduction=reduction,
  )
  labels_grad = jnp.zeros_like(labels)
  return (x_grad, labels_grad, w_grad)


@dataclass(frozen=True, kw_only=True)
class PallasMosaicGpuLinearSoftmaxCrossEntropyLoss(
    base.LinearSoftmaxCrossEntropyLoss[Config]
):
  """Pallas/Mosaic-GPU SM90 forward + backward for linear softmax CE loss.

  Both forward and backward use WGMMA + TMA pipelining on H100 (SM90).
  No Triton dependency.
  """

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    object.__setattr__(self, "vjp", _mosaic_vjp)

  @override
  def _fwd(
      self,
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean"] = "sum",
      config: Config,
      return_residuals: bool,
  ) -> tuple[jax.Array, base.Residuals]:
    device_kind = backend.get_default_device().device_kind.lower()
    if not (gpu_utils.is_sm90() or gpu_utils.is_sm100()):
      raise NotImplementedError(
          f"Mosaic GPU kernel requires SM90 or SM100; got {device_kind!r}."
      )

    loss, lse = kernel_sm90.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_gpu_sm90(
        x,
        labels,
        w,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        num_stages=config.num_stages,
        reduction=reduction,
    )
    return loss, (lse,)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return common.get_heuristics_config(ba.arguments["x"], ba.arguments["w"])

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return common.get_autotuning_configs(ba.arguments["x"], ba.arguments["w"])

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return common.get_key(**ba.arguments)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
