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
"""Tests of common/: norm, quant, rope and layout_util.

norm, quant and rope are checked against vLLM's formulas. Each layout
primitive runs alone in a small Pallas kernel gridded over token tiles, on
values that are distinct per element.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.fused_qkv_proj import cos_sin
from tokamax._src.ops.experimental.fused_qkv_proj import test_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import norm
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import quantized_matmul
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import rope

TILE = 256
ROWS = 128  # Rows per store, like the kernels' matmul_rows.


def _tiled(kernel, x, out_shape, num_tokens):
  """Runs kernel(x_ref, o_ref) per TILE tokens of x [T, ...]."""

  def spec(shape):
    return pl.BlockSpec(
        (TILE, *shape[1:]), lambda i, n=len(shape): (i,) + (0,) * (n - 1)
    )

  return pl.pallas_call(
      kernel,
      grid=(num_tokens // TILE,),
      in_specs=[spec(x.shape)],
      out_specs=spec(out_shape.shape),
      out_shape=out_shape,
  )(x)


def _vllm_rms_norm(x, weight, eps):
  # ir.ops.rms_norm.
  xf = x.astype(jnp.float32)
  xf = xf * jax.lax.rsqrt(jnp.mean(xf**2, -1, keepdims=True) + eps)
  if weight is not None:
    xf = xf.astype(weight.dtype) * weight
  return xf.astype(x.dtype)


def _vllm_rope(x, cos, sin, rotary_dim, fp32=True):
  # RotaryEmbedding.forward_static: rotate x[..., :rotary_dim] NeoX-style.
  # vllm-torchtpu passes enable_fp32_compute=True (rotary_embedding.py): the
  # cos/sin cache is matched to x's dtype, then x, cos and sin are upcast.
  rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
  x1, x2 = jnp.split(rot, 2, -1)
  cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
  if fp32:
    x1, x2, cos, sin = (a.astype(jnp.float32) for a in (x1, x2, cos, sin))
  out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
  return jnp.concatenate([out.astype(x.dtype), rest], -1)


class _ReferenceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("TPU-only")
    ks = jax.random.split(jax.random.key(0), 2)
    self.x = jax.random.normal(ks[0], (64, 4, 512), jnp.bfloat16)
    self.w = 1 + 0.1 * jax.random.normal(ks[1], (512,), jnp.bfloat16)


class NormTest(_ReferenceTest):

  def test_rms(self):
    test_util.assert_equal(
        norm.norm(model_config.Norm.RMS, self.x, self.w, 1e-6),
        _vllm_rms_norm(self.x, self.w, 1e-6),
    )

  def test_rms_one_plus_w(self):
    # GemmaRMSNorm.forward_native passes weight.float() + 1.
    w = self.w - 1
    test_util.assert_equal(
        norm.norm(model_config.Norm.RMS_ONE_PLUS_W, self.x, w, 1e-6),
        _vllm_rms_norm(self.x, w.astype(jnp.float32) + 1.0, 1e-6),
    )

  def test_rms_noscale(self):
    test_util.assert_equal(
        norm.norm(model_config.Norm.RMS_NOSCALE, self.x, None, 1e-6),
        _vllm_rms_norm(self.x, None, 1e-6),
    )

  def test_fused_add_rms(self):
    # ir.ops.fused_add_rms_norm: f32 add, bf16 residual, norm of the f32 sum.
    residual = self.x[::-1]
    xf = self.x.astype(jnp.float32) + residual.astype(jnp.float32)
    got, got_residual = norm.add_norm(
        model_config.Norm.RMS, self.x, residual, self.w, 1e-6
    )
    assert got_residual is not None
    test_util.assert_equal(got_residual, xf.astype(jnp.bfloat16))
    want = xf * jax.lax.rsqrt(jnp.mean(xf**2, -1, keepdims=True) + 1e-6)
    test_util.assert_equal(
        got, (want.astype(jnp.bfloat16) * self.w).astype(jnp.bfloat16)
    )

  def test_f32_weight(self):
    k, block = 2048, 1024
    w = test_util.distinct((1, k), jnp.bfloat16)
    got = norm.f32_weight(model_config.Norm.RMS_ONE_PLUS_W, w, block)
    self.assertEqual(got.shape, (k // block, 1, block))
    self.assertEqual(got.dtype, jnp.float32)
    test_util.assert_equal(got.reshape(1, k), w.astype(jnp.float32) + 1.0)

  def test_packed_bf16_weight(self):
    k, block = 2048, 1024
    w = test_util.distinct((1, k), jnp.bfloat16)
    got = np.asarray(norm.packed_bf16_weight(model_config.Norm.RMS, w, block))
    self.assertEqual(got.shape, (k // block, 1, block))
    bits = np.asarray(w).view(np.uint16).astype(np.uint32).reshape(got.shape)
    np.testing.assert_array_equal(got, bits | (bits << 16))


class QuantTest(_ReferenceTest):

  def test_quantize_fp8_per_token(self):
    # vllm_torchtpu quantized_matmul/util.py quantize_block(x, axis=-1).
    x = self.x.at[0].set(0)
    abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = abs_max / jnp.finfo(quantized_matmul.F8).max.astype(jnp.float32)
    scale = jnp.where(scale == 0, 1.0, scale)
    want = (
        (x / scale)
        .clip(-quantized_matmul.F8_MAX, quantized_matmul.F8_MAX)
        .astype(quantized_matmul.F8)
    )
    got, got_scale = quantized_matmul.quantize_fp8_per_token(x)
    test_util.assert_equal(got_scale, scale)
    test_util.assert_equal(got, want)


class RopeTest(_ReferenceTest):

  def test_partial_rope(self):
    pos = jnp.arange(64)
    cos, sin = cos_sin.cos_sin_default(pos, 64, 1e7)
    x = self.x[..., :256]
    test_util.assert_equal(
        rope.rope(model_config.Rope.neox(64), x, cos[:, None], sin[:, None]),
        _vllm_rope(x, cos[:, None], sin[:, None], 64),
    )

  def test_partial_rope_bf16(self):
    pos = jnp.arange(64)
    cos, sin = cos_sin.cos_sin_default(pos, 64, 1e7)
    x = self.x[..., :256]
    test_util.assert_equal(
        rope.rope(
            model_config.Rope.neox(64),
            x,
            cos[:, None],
            sin[:, None],
            model_config.RopePrecision.BF16,
        ),
        _vllm_rope(x, cos[:, None], sin[:, None], 64, fp32=False),
    )

  def test_proportional_rope(self):
    # Gemma4RotaryEmbedding: full-width rotate_half with identity padding.
    pos = jnp.arange(64) * 97
    cos, sin = cos_sin.cos_sin_proportional(pos, 512, 0.25, 1e6)
    self.assertEqual(cos.shape, (64, 64))
    pad = lambda t, v: jnp.pad(t, ((0, 0), (0, 192)), constant_values=v)
    full_cos, full_sin = pad(cos, 1), pad(sin, 0)
    test_util.assert_equal(
        rope.rope(
            model_config.Rope(64, 256), self.x, cos[:, None], sin[:, None]
        ),
        _vllm_rope(self.x, full_cos[:, None], full_sin[:, None], 512),
    )

  def test_yarn_matches_formula(self):
    # gpt-oss-120b: low/high correction dims are 8.09 and 17.39.
    pos = jnp.array([0, 1, 4096, 131071])
    cos, sin = cos_sin.cos_sin_yarn(
        pos, 64, 150000.0, 32.0, 4096, truncate=False
    )
    mscale = 0.1 * np.log(32.0) + 1.0
    np.testing.assert_allclose(
        np.asarray(cos[0], np.float32), mscale, rtol=1e-2
    )
    np.testing.assert_array_equal(np.asarray(sin[0], np.float32), 0)
    # Highest-frequency pair is extrapolated: angle = pos * 1.
    np.testing.assert_allclose(
        np.asarray(sin[1, 0], np.float32), mscale * np.sin(1.0), rtol=1e-2
    )


class LayoutsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("TPU-only")

  def test_pack_heads_bf16(self):
    lo, hi = (
        test_util.distinct((8, 128), jnp.bfloat16),
        test_util.distinct((8, 128))[::-1],
    )
    hi = hi.astype(jnp.bfloat16)
    packed = np.asarray(layout_util.pack_heads([lo, hi]))
    bits = lambda a: np.asarray(a).view(np.uint16).astype(np.uint32)
    np.testing.assert_array_equal(packed, bits(lo) | (bits(hi) << 16))
    got_lo, got_hi = layout_util.unpack_heads(jnp.asarray(packed), jnp.bfloat16)
    test_util.assert_equal(got_lo, lo)
    test_util.assert_equal(got_hi, hi)

  @parameterized.parameters((4, 128), (2, 256), (4, 512))
  def test_strided_load_heads(self, num_groups, d):
    """strided_store_heads then strided_load_heads recovers every head pair."""
    t = 512
    x = test_util.distinct((t, num_groups, 2, d), jnp.bfloat16)

    def kernel(x_ref, o_ref):
      for m in range(0, TILE, ROWS):
        rows = pl.ds(m, ROWS)
        for g in range(num_groups):
          packed = layout_util.strided_load_heads(x_ref, rows, g, d)
          h0, h1 = layout_util.unpack_heads(packed, jnp.bfloat16)
          col = 2 * g * d
          o_ref[rows, pl.ds(col, d)] = h0
          o_ref[rows, pl.ds(col + d, d)] = h1

    out = jax.ShapeDtypeStruct((t, 2 * num_groups * d), jnp.bfloat16)
    test_util.assert_equal(_tiled(kernel, x, out, t), x.reshape(t, -1))

  @parameterized.parameters(1536, 512)
  def test_strided_store_column_blocks(self, width):
    """Column blocks of (T, W) f32 -> compact (T, 1, W): mixed_qkv."""
    t, block = 512, 256
    x = test_util.distinct((t, width))

    def kernel(x_ref, o_ref):
      for m in range(0, TILE, ROWS):
        rows = pl.ds(m, ROWS)
        for c in range(0, width, block):
          layout_util.strided_store(
              o_ref, m, 1, x_ref[rows, pl.ds(c, block)], col=c
          )

    out = jax.ShapeDtypeStruct((t, 1, width), jnp.float32)
    test_util.assert_equal(_tiled(kernel, x, out, t), x.reshape(t, 1, width))

  @parameterized.parameters(8, 4)
  def test_store_head_pair(self, num_heads):
    """Head pairs of f32 (T, H * 128) -> head-major bf16 (T, H, 128): z."""
    t, d = 512, 128
    x = test_util.distinct((t, num_heads * d))

    def kernel(x_ref, o_ref):
      for m in range(0, TILE, ROWS):
        rows = pl.ds(m, ROWS)
        for p in range(num_heads // 2):
          val = x_ref[rows, pl.ds(2 * d * p, 2 * d)]
          layout_util.store_head_pair(o_ref, rows, p, val)

    out = jax.ShapeDtypeStruct((t, num_heads, d), jnp.bfloat16)
    want = x.astype(jnp.bfloat16).reshape(t, num_heads, d)
    test_util.assert_equal(_tiled(kernel, x, out, t), want)

  def test_strided_store_rows(self):
    """f32 (T, 128) -> lane-padded (T, 1, 128): b and a."""
    t = 512
    x = test_util.distinct((t, 128))

    def kernel(x_ref, o_ref):
      for m in range(0, TILE, ROWS):
        layout_util.strided_store(o_ref, m, 1, x_ref[pl.ds(m, ROWS)])

    out = jax.ShapeDtypeStruct((t, 1, 128), jnp.float32)
    test_util.assert_equal(_tiled(kernel, x, out, t), x.reshape(t, 1, 128))


if __name__ == "__main__":
  absltest.main()
