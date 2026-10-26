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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src import benchmarking
from tokamax._src import test_utils
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.io_relayout import output_reduce

jax.config.parse_flags_with_absl()


def ref_topk_reduce(rows2d, weights, out_dtype=jnp.bfloat16):
  """Pure-JAX reference: rows2d[T * K, H] in (token, pick) order."""
  num_tokens, topk = weights.shape
  w = weights.astype(jnp.float32).reshape(-1, 1)
  x = jnp.where(w != 0, rows2d.astype(jnp.float32), 0.0) * w
  return x.reshape(num_tokens, topk, -1).sum(axis=1).astype(out_dtype)


def local_pick_weights(
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    expert_lo,
    num_local_experts: int,
) -> jax.Array:
  """``topk_weights`` with the picks other shards serve set to 0."""
  local = jnp.logical_and(
      topk_indices >= expert_lo, topk_indices < expert_lo + num_local_experts
  )
  return jnp.where(local, topk_weights, 0)


def xla_output_reduce_unpack(
    rows_compact: jax.Array,
    weights: jax.Array,
    topk_indices: jax.Array,
    group_offset: jax.Array,
    num_local_experts: int,
    *,
    out_dtype=jnp.bfloat16,
) -> jax.Array:
  w_local = local_pick_weights(
      weights, topk_indices, group_offset, num_local_experts
  )
  rows_2d = rows_compact.reshape(layout_util.to_large_shape(rows_compact.shape))
  return ref_topk_reduce(rows_2d, w_local, out_dtype=out_dtype)


class OutputReduceTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest("Only tested on TPUv7x for now.")
    if jax.__version_info__ < (0, 11, 2):
      self.skipTest("routing_moe requires JAX >= 0.11.2.")
    super().setUp()

  def _benchmark_and_profile(self, xla, kernel):
    latency_ms = {}
    for name, (fn, args) in [
        ("xla", xla),
        ("kernel", kernel),
    ]:
      latency_ms[name] = benchmarking.benchmark(
          fn, args, method="hermetic_xprof"
      ).median_evaluation_time_ms
      logging.info("%s: %.3f ms", name, latency_ms[name])

  def _test_output_reduce(
      self,
      num_tokens: int,
      top_k: int,
      num_experts: int,
      hidden_size: int,
      ep_size: int = 1,
      active_expert_ratio: float = 1.0,
      renormalize_topk_logits: bool = True,
      out_dtype: jnp.dtype = jnp.bfloat16,
      seed: int = 1234,
      atol: float | None = None,
      rtol: float = 2e-2,
      capture_xprof: bool = False,
  ):
    m = num_tokens * top_k
    num_local_experts = num_experts // ep_size

    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    gating_output = jax.random.uniform(
        k1, (num_tokens, num_experts), minval=0, maxval=1, dtype=jnp.float32
    )
    if active_expert_ratio < 1.0:
      num_active_experts = max(1, int(num_experts * active_expert_ratio))
      active_experts = jax.random.normal(k3, (num_experts,))
      val, _ = jax.lax.top_k(active_experts, k=num_active_experts)
      mask = active_experts >= val[-1]
      gating_output = jnp.where(mask, gating_output, 0.0)

    topk_weights = jax.nn.softmax(gating_output, axis=-1).astype(jnp.float32)
    topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=top_k)
    topk_indices = topk_indices.astype(jnp.int32)
    if renormalize_topk_logits:
      topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)

    group_offset = jnp.array([0], dtype=jnp.int32)
    w_local = local_pick_weights(
        topk_weights, topk_indices, group_offset, num_local_experts
    )

    rows_clean = jax.random.normal(k2, (m, hidden_size), jnp.bfloat16) / 10
    valid_row = w_local.reshape(-1)[:, None] != 0
    rows_with_nan = jnp.where(
        valid_row, rows_clean, jnp.full_like(rows_clean, jnp.nan)
    )
    rows_compact = rows_with_nan.reshape(
        layout_util.to_compact_shape(rows_with_nan.shape, rows_with_nan.dtype)
    )

    expected = ref_topk_reduce(rows_clean, w_local, out_dtype=out_dtype)

    if ep_size == 1:
      actual = output_reduce.output_reduce_unpack(
          rows_compact, topk_weights, out_dtype=out_dtype
      )
    else:
      actual = output_reduce.output_reduce_unpack(
          rows_compact,
          topk_weights,
          topk_indices,
          group_offset,
          num_local_experts,
          out_dtype=out_dtype,
      )

    expected.block_until_ready()
    actual.block_until_ready()

    self.assertFalse(bool(jnp.isnan(actual).any()))
    if atol is None:
      atol = rtol * max(float(jnp.abs(expected).max()), 1e-3)
    test_utils.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

    if capture_xprof:
      rows_clean_compact = rows_clean.reshape(
          layout_util.to_compact_shape(rows_clean.shape, rows_clean.dtype)
      )
      xla = benchmarking.standardize_function(
          xla_output_reduce_unpack,
          rows_clean_compact,
          topk_weights,
          topk_indices,
          group_offset.reshape(()),
          kwargs=dict(num_local_experts=num_local_experts, out_dtype=out_dtype),
      )
      kernel = benchmarking.standardize_function(
          output_reduce.output_reduce_unpack,
          rows_compact,
          topk_weights,
          topk_indices,
          group_offset,
          kwargs=dict(num_local_experts=num_local_experts, out_dtype=out_dtype),
      )
      self._benchmark_and_profile(xla, kernel)

  @parameterized.product(
      num_tokens=[48, 128, 300, 512],
      top_k=[1, 2, 4, 10],
      ep_size=[1, 8],
      out_dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_basic(self, num_tokens, top_k, ep_size, out_dtype):
    self._test_output_reduce(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=128,
        hidden_size=1024,
        ep_size=ep_size,
        out_dtype=out_dtype,
        seed=42,
    )

  def test_ep_all_shards_sum_to_dense(self):
    num_tokens = 256
    top_k = 10
    num_experts = 512
    hidden_size = 2048
    ep_size = 8
    m = num_tokens * top_k
    local_experts = num_experts // ep_size

    k1, k2 = jax.random.split(jax.random.key(99), 2)
    logits = jax.random.normal(k1, (num_tokens, num_experts), jnp.float32)
    topk_vals, topk_indices = jax.lax.top_k(logits, top_k)
    topk_indices = topk_indices.astype(jnp.int32)
    topk_weights = jax.nn.softmax(topk_vals, axis=-1).astype(jnp.float32)

    rows_clean = jax.random.normal(k2, (m, hidden_size), jnp.bfloat16) / 10

    shard_sum = jnp.zeros((num_tokens, hidden_size), dtype=jnp.float32)
    for shard in range(ep_size):
      w_shard = local_pick_weights(
          topk_weights, topk_indices, shard * local_experts, local_experts
      )
      valid_row = w_shard.reshape(-1)[:, None] != 0
      rows_shard = jnp.where(
          valid_row, rows_clean, jnp.full_like(rows_clean, jnp.nan)
      ).reshape(
          layout_util.to_compact_shape(rows_clean.shape, rows_clean.dtype)
      )
      out_shard = output_reduce.output_reduce_unpack(
          rows_shard,
          topk_weights,
          topk_indices,
          shard * local_experts,
          local_experts,
          out_dtype=jnp.float32,
      )
      self.assertFalse(bool(jnp.isnan(out_shard).any()))
      shard_sum = shard_sum + out_shard

    ref_dense = ref_topk_reduce(rows_clean, topk_weights, out_dtype=jnp.float32)
    np.testing.assert_allclose(shard_sum, ref_dense, rtol=2e-2, atol=2e-2)

  def test_all_zero_weights(self):
    num_tokens = 256
    top_k = 10
    hidden_size = 2048
    m = num_tokens * top_k
    rows_nan = jnp.full(
        layout_util.to_compact_shape((m, hidden_size), jnp.bfloat16),
        jnp.nan,
        dtype=jnp.bfloat16,
    )
    w_zero = jnp.zeros((num_tokens, top_k), dtype=jnp.float32)
    out = output_reduce.output_reduce_unpack(rows_nan, w_zero)
    np.testing.assert_array_equal(
        out, jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16)
    )

  @parameterized.product(
      use_ep=[True, False],
  )
  def test_benchmark_qwen_235(self, use_ep):
    self._test_output_reduce(
        num_tokens=8 * 64,
        top_k=8,
        num_experts=128,
        hidden_size=4096,
        ep_size=8 if use_ep else 1,
        seed=54321,
        capture_xprof=True,
    )

  @parameterized.product(
      use_ep=[True, False],
  )
  def test_benchmark_qwen_30b_a3b(self, use_ep):
    self._test_output_reduce(
        num_tokens=8 * 64,
        top_k=8,
        num_experts=128,
        hidden_size=2048,
        ep_size=8 if use_ep else 1,
        seed=54321,
        capture_xprof=True,
    )

  @parameterized.product(
      num_tokens=[64, 8192],
      use_ep=[True, False],
      active_expert_ratio=[0.6],
  )
  def test_benchmark_qwen_coder_480b(
      self, num_tokens, use_ep, active_expert_ratio
  ):
    self._test_output_reduce(
        num_tokens=num_tokens,
        top_k=8,
        num_experts=160,
        hidden_size=6144,
        ep_size=8 if use_ep else 1,
        active_expert_ratio=active_expert_ratio,
        seed=54321,
        capture_xprof=True,
    )

  @parameterized.product(
      num_tokens=[2048, 4096],
      use_ep=[True, False],
      active_expert_ratio=[0.6],
  )
  def test_benchmark_mistral_3_large(
      self, num_tokens, use_ep, active_expert_ratio
  ):
    self._test_output_reduce(
        num_tokens=num_tokens,
        top_k=4,
        num_experts=128,
        hidden_size=7168,
        ep_size=8 if use_ep else 1,
        active_expert_ratio=active_expert_ratio,
        seed=54321,
        capture_xprof=True,
    )

  @parameterized.product(
      num_tokens=[64, 128, 256, 512, 8192],
      use_ep=[True, False],
      active_expert_ratio=[1.0],
  )
  def test_benchmark_qwen_35_397b(
      self, num_tokens, use_ep, active_expert_ratio
  ):
    self._test_output_reduce(
        num_tokens=num_tokens,
        top_k=10,
        num_experts=512,
        hidden_size=4096,
        ep_size=8 if use_ep else 1,
        active_expert_ratio=active_expert_ratio,
        seed=54321,
        capture_xprof=True,
    )

  @parameterized.product(
      num_tokens=[64],
      use_ep=[True, False],
  )
  def test_benchmark_gpt_oss_120(self, num_tokens, use_ep):
    self._test_output_reduce(
        num_tokens=num_tokens,
        top_k=4,
        num_experts=128,
        hidden_size=3072,
        ep_size=2 if use_ep else 1,
        seed=54321,
        capture_xprof=True,
    )

  def test_benchmark_gmm_top1_200(self):
    self._test_output_reduce(
        num_tokens=200,
        top_k=1,
        num_experts=1,
        hidden_size=2048,
        ep_size=1,
        seed=0,
        capture_xprof=True,
    )


if __name__ == "__main__":
  absltest.main()
