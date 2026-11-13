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
"""Unit tests for compute.matmul, run as regular JAX outside a kernel."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import quant_util
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.gmm import compute
from tokamax._src.ops.experimental.routing_moe.gmm import config

_TILE_M, _TILE_K, _TILE_N = 16, 1024, 512
_X_Q_BLOCK_SIZE = 512


def _compute_config(
    mode: config.MatmulMode, matmul_dtype: jax.typing.DTypeLike
) -> config.ComputeConfig:
  return config.ComputeConfig(
      matmul_mode=mode,
      matmul_dtype=jnp.dtype(matmul_dtype),
      acc_dtype=jnp.dtype(jnp.bfloat16),
      tpu_info=pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
      dynamic_quant=spec.SubchannelQuantSpec(block_size=_X_Q_BLOCK_SIZE)
      if mode == config.MatmulMode.QUANTIZED
      else None,
  )


def _inputs(
    has_bias: bool,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  kx, kw, kb = jax.random.split(jax.random.key(0), 3)
  x = jax.random.normal(kx, (_TILE_M, _TILE_K), jnp.bfloat16)
  w = jax.random.normal(kw, (_TILE_K, _TILE_N), jnp.float32) / _TILE_K**0.5
  bias = None
  if has_bias:
    bias = jax.random.normal(kb, (1, _TILE_N), jnp.float32)
  return x, w, bias


def _quantize_w(
    w: jax.Array, dtype: jax.typing.DTypeLike, block_size: int
) -> tuple[jax.Array, jax.Array]:
  """Returns `w` quantized per K block and its `(num_blocks, 1, n)` scale."""
  k, n = w.shape
  blocks = w.reshape(k // block_size, block_size, n)
  absmax = jnp.max(jnp.abs(blocks), axis=1, keepdims=True)
  scale = absmax / quant_util.dtype_max(dtype)
  w_q = (blocks / scale).astype(dtype).reshape(k, n)
  return w_q, scale


def _reference(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array | None,
    bias: jax.Array | None,
) -> np.ndarray:
  w = w.astype(jnp.float32)
  if w_scale is not None:
    num_blocks = w_scale.shape[0]
    w = (w.reshape(num_blocks, -1, w.shape[1]) * w_scale).reshape(w.shape)
  out = x.astype(jnp.float32) @ w
  if bias is not None:
    out += bias
  return np.asarray(out)


class MatmulTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_unquantized(self, has_bias: bool):
    x, w, bias = _inputs(has_bias)
    w = w.astype(jnp.bfloat16)
    cfg = _compute_config(config.MatmulMode.UNQUANTIZED, jnp.bfloat16)

    actual = compute.matmul(cfg, x, w, None, bias)

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        _reference(x, w, None, bias),
        rtol=2e-2,
        atol=2e-2,
    )

  @parameterized.product(
      mode=[
          config.MatmulMode.WEIGHT_ONLY_SCALE_AFTER,
          config.MatmulMode.WEIGHT_ONLY_SCALE_BEFORE,
      ],
      w_dtype=[jnp.float8_e4m3fn, jnp.int8],
      w_q_block_size=[128, 512, 1024],
      has_bias=[False, True],
  )
  def test_weight_only(
      self,
      mode: config.MatmulMode,
      w_dtype: jnp.dtype,
      w_q_block_size: int,
      has_bias: bool,
  ):
    x, w, bias = _inputs(has_bias)
    w_q, w_scale = _quantize_w(w, w_dtype, w_q_block_size)
    cfg = _compute_config(mode, jnp.bfloat16)

    actual = compute.matmul(cfg, x, w_q, w_scale, bias)

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        _reference(x, w_q, w_scale, bias),
        rtol=5e-2,
        atol=5e-2,
    )

  @parameterized.product(
      w_dtype=[jnp.float8_e4m3fn, jnp.int8],
      w_q_block_size=[128, 512, 1024],
      has_bias=[False, True],
  )
  def test_quantized(
      self, w_dtype: jnp.dtype, w_q_block_size: int, has_bias: bool
  ):
    x, w, bias = _inputs(has_bias)
    w_q, w_scale = _quantize_w(w, w_dtype, w_q_block_size)
    cfg = _compute_config(config.MatmulMode.QUANTIZED, w_dtype)

    actual = compute.matmul(cfg, x, w_q, w_scale, bias)

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        _reference(x, w_q, w_scale, bias),
        rtol=1e-1,
        atol=1e-1,
    )


if __name__ == "__main__":
  absltest.main()
