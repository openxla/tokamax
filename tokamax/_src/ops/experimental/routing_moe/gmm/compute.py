# Copyright 2026 Google LLC
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

"""Functions required to perform matrix multiplication."""

from collections.abc import Callable

import jax
from jax.experimental import hijax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.common import quant_util
from tokamax._src.ops.experimental.routing_moe.gmm import config


def _swigluoai(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
  """Activation used in some models such as GPT-OSS."""

  # Use the limit value provided in GPT-OSS config.
  # TODO: Add ability configure limit values.
  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  glu = gate * jax.nn.sigmoid(alpha * gate)
  return (up + 1.0) * glu


def _silu_and_mul_with_clamp(
    gate: jax.Array, up: jax.Array, limit: float = 10.0
) -> jax.Array:
  """Activation used in some models DeepSeek V4."""
  # Use the limit value provided in DSV4's config.
  # TODO: Add ability configure limit values.
  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  return jax.nn.silu(gate) * up


def apply_act_fn(acc: jax.Array, fuse_act: str | None) -> jax.Array:
  """Optionally applies a fused activation function to the accumulator."""

  if fuse_act is None:
    return acc

  acc_gate, acc_up = layout_util.large_deinterleave_lane(acc)
  match fuse_act:
    case "silu":
      return jax.nn.silu(acc_gate) * acc_up
    case "gelu":
      return jax.nn.gelu(acc_gate) * acc_up
    case "gelu_tanh":
      return jax.nn.gelu(acc_gate, approximate=True) * acc_up
    case "swigluoai":
      return _swigluoai(acc_gate, acc_up)
    case "silu_and_mul_with_clamp":
      return _silu_and_mul_with_clamp(acc_gate, acc_up)
    case _:
      raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def matmul(
    cfg: config.ComputeConfig,
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array | None,
    w_bias: jax.Array | None,
    *,
    is_last_k_step: bool = True,
    is_first_k_step: bool = True,
) -> jax.Array:
  """Performs stateless matrix multiplication for a single tile."""
  # TODO: Support multiple N/K tiles.
  assert is_last_k_step and is_first_k_step

  tpu_info = cfg.tpu_info
  mxu_size = tpu_info.mxu_column_size

  # Step 1: Input pre-processing.
  tile_m, tile_k = x.shape
  tile_n = w.shape[-1]

  # Step 2: Matmul.
  acc_list = []

  if cfg.dynamic_quant is None:
    block_acc = jnp.matmul(x, w, preferred_element_type=jnp.float32)
    acc_list.append(block_acc.astype(cfg.acc_dtype))
  else:
    # Quantized matmul path.
    x_q_dtype = cfg.dynamic_quant.dtype
    q_block_size = cfg.dynamic_quant.block_size

    if jnp.issubdtype(x_q_dtype, jnp.floating):
      preferred_element_type = jnp.float32
    else:
      preferred_element_type = jnp.int32

    # Weight quantization block along K.
    if w_scale is None:
      w_q_block_size = tile_k
    else:
      # w_scale: [tile_k // w_q_block_size, 1, tile_n]
      num_w_blocks = w_scale.shape[0]
      assert tile_k % num_w_blocks == 0, (
          f"{num_w_blocks=} must divide {tile_k=}."
      )
      w_q_block_size = tile_k // num_w_blocks
      # With a single weight block (per-channel), every x block lies within it,
      # so the block sizes need not divide one another.
      if num_w_blocks > 1:
        block_sizes = (q_block_size, w_q_block_size)
        assert max(block_sizes) % min(block_sizes) == 0, (
            f"{q_block_size=} and {w_q_block_size=} must divide one another."
        )

    # Without n outer loop, result of quantized matmul becomes available only
    # at the last iteration of the loop. This means [tile_m, tile_n] value
    # needs to be stored until the last iteration. By adding n outer loop,
    # result of [tile_m, mxu_size] becomes available at the end of every k
    # inner loop which can be used to pipeline subsequent VPU or VST ops with
    # MXU ops for the next [tile_m, mxu_size].
    for start_n in range(0, tile_n, mxu_size):
      end_n = min(tile_n, start_n + mxu_size)
      col_size = end_n - start_n

      acc_n = jnp.zeros((tile_m, col_size), dtype=cfg.acc_dtype)
      for start_k in range(0, tile_k, q_block_size):
        end_k = min(tile_k, start_k + q_block_size)

        block_x = x[:, start_k:end_k]
        block_w = w[start_k:end_k, start_n:end_n]

        # Perform lhs quantization. Note that for every block_lhs,
        # same computation will be performed tiles_n//mxu_size times.
        # But we can let compiler perform CSE and avoid recomputation.
        block_scale, block_scale_inv = quant_util.compute_scale(
            block_x, cfg.dynamic_quant
        )
        block_x_q = quant_util.quantize(block_x, block_scale_inv, x_q_dtype)

        # Unlike unquantized path, compiler may not perform implicit type
        # conversion due to numeric concerns. As this can cause unsupported
        # matmul error, explicit type conversion is performed.
        if not tpu_info.is_matmul_supported(x_q_dtype, block_w.dtype):
          block_w = block_w.astype(x_q_dtype)

        # If w_q_block_size < q_block_size, this loops
        # q_block_size // w_q_block_size times. Otherwise it runs once over the
        # whole x block (end_w is clipped to end_k - start_k), and consecutive
        # x blocks share one weight scale.
        # Don't initialize block_acc to jnp.zero because jnp.zeros causes zero
        # initialization values to materialize.
        block_acc = None
        for start_w in range(0, end_k - start_k, w_q_block_size):
          end_w = min(end_k - start_k, start_w + w_q_block_size)
          sub_acc = jnp.matmul(
              # block_x_q: [tile_m, end_k - start_k]
              block_x_q[:, start_w:end_w],
              # block_w: [end_k - start_k, col_size]
              block_w[start_w:end_w],
              preferred_element_type=preferred_element_type,
          ).astype(cfg.acc_dtype)
          if w_scale is not None:
            # w_scale: [tile_k // w_q_block_size, 1, tile_n]
            scale_idx = (start_k + start_w) // w_q_block_size
            sub_acc *= w_scale[scale_idx, :, start_n:end_n].astype(
                cfg.acc_dtype
            )
          block_acc = sub_acc if block_acc is None else block_acc + sub_acc

        assert block_acc is not None
        block_acc *= block_scale.astype(cfg.acc_dtype)

        acc_n += block_acc
      acc_list.append(acc_n)
  acc = jnp.concatenate(acc_list, axis=1)

  if w_bias is not None:
    acc += w_bias

  return acc


def quantize_rows(
    dst: hijax.TransformedRef,
    first_row: int,
    scale: jax.Array,
    inv_scale: jax.Array,
    load_chunk: Callable[[int], jax.Array],
    out: config.OutputConfig,
) -> None:
  """Quantizes `scale.shape[0]` rows and writes payload and scale chunks.

  Args:
    dst: Destination compact output VMEM slot.
    first_row: Starting row index in `dst` for this `scale.shape[0]`-row slice.
    scale: Per-lane scales of shape `f32[rows, lanes]`.
    inv_scale: Broadcastable reciprocal scales matching `load_chunk` outputs.
    load_chunk: Callable returning the unquantized values for payload chunk `c`.
    out: Output configuration describing payload chunks and quantization dtype.
  """
  assert out.dynamic_quant is not None
  num_chunks = out.num_payload_chunks
  layout_util.store_compact_chunk(
      dst, first_row, num_chunks, pltpu.bitcast(scale, jnp.uint32)
  )
  for chunk in range(num_chunks):
    payload = quant_util.quantize(
        load_chunk(chunk), inv_scale, out.dynamic_quant.dtype
    )
    words = pltpu.bitcast(payload, jnp.uint32)
    layout_util.store_compact_chunk(dst, first_row, chunk, words)
