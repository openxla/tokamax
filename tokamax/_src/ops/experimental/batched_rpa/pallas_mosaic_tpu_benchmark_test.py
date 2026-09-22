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
"""Microbenchmarks and autotuning for Batched RPA Mosaic TPU kernel on accelerators."""

from absl.testing import absltest
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import benchmarking
from tokamax._src.ops.experimental.batched_rpa import base
from tokamax._src.ops.experimental.batched_rpa import pallas_mosaic_tpu


class PallasMosaicTpuBatchedRpaBenchmarkTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      tpu_info = pltpu.get_tpu_info()
    except Exception:
      self.skipTest("Failed to get TPU info.")
      return
    if tpu_info.generation < 5:
      self.skipTest("Pallas TPU kernel requires TPU v5 or newer.")

  def test_benchmark_and_autotune_decode(self) -> None:
    """Benchmarks and autotunes Batched RPA decode workload (BS=16, seq_len=4096)."""
    device = jax.devices()[0]
    tpu_info = pltpu.get_tpu_info()
    print("\n" + "=" * 80, flush=True)
    print(f"DEVICE: {device.device_kind} (generation {tpu_info.generation})", flush=True)
    print(f"VMEM CAPACITY: {tpu_info.vmem_capacity_bytes / (1024 * 1024):.1f} MiB", flush=True)
    print("=" * 80, flush=True)

    total_q_tokens = 16
    max_num_seqs = 16
    seq_len = 4096
    page_size = 256
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128
    head_dim_aligned = 128
    pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = max_num_seqs * pages_per_seq

    k1, k2, k3, k4 = jax.random.split(jax.random.key(42), 4)
    queries = jax.random.normal(k1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
    keys = jax.random.normal(k2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    values = jax.random.normal(k3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    kv_packing = 2  # bf16
    kv_cache = jax.random.normal(
        k4,
        (
            total_pages,
            page_size,
            num_kv_heads * 2 // kv_packing,
            kv_packing,
            head_dim_aligned,
        ),
        dtype=jnp.bfloat16,
    )
    kv_lens = jnp.full((max_num_seqs,), seq_len, dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    cu_q_lens = jnp.arange(max_num_seqs + 1, dtype=jnp.int32)
    distribution = jnp.array([max_num_seqs, max_num_seqs, max_num_seqs], dtype=jnp.int32)

    op_base = base.BatchedRpa()
    op_pallas = pallas_mosaic_tpu.PallasTpuBatchedRpa()

    bound_args_base = op_base.bind(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )
    bound_args = op_pallas.bind(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )

    print("\n--- Step 1: Reference Baseline Benchmark ---", flush=True)
    ref_bench = bound_args_base.benchmark()
    print(f"Reference Mean Evaluation Time: {ref_bench.median_evaluation_time_ms:.3f} ms", flush=True)
    print(f"Reference Peak Memory: {ref_bench.peak_memory_mb:.2f} MiB", flush=True)

    print("\n--- Step 2: Default Heuristics Benchmark ---", flush=True)
    heuristics_bench = bound_args.benchmark()
    heuristics_cfg = bound_args.heuristics_config
    print(f"Heuristics Config: {heuristics_cfg}", flush=True)
    print(f"Heuristics Mean Evaluation Time: {heuristics_bench.median_evaluation_time_ms:.3f} ms", flush=True)
    print(f"Heuristics Compile Time: {heuristics_bench.compile_time_ms:.3f} ms", flush=True)
    print(f"Heuristics Peak Memory: {heuristics_bench.peak_memory_mb:.2f} MiB", flush=True)
    speedup = ref_bench.median_evaluation_time_ms / heuristics_bench.median_evaluation_time_ms
    print(f"Speedup vs Reference: {speedup:.2f}x", flush=True)

    print("\n--- Step 3: Autotuning Kernel Configurations ---", flush=True)
    configs = {
        pallas_mosaic_tpu.Config(
            prefill_bq_sz=128,
            bkv_sz=bkv,
            decode_batch_size=dbs,
            prefill_batch_size=2,
        )
        for bkv in (256, 512, 1024)
        for dbs in (4, 8)
    }

    autotuned_data = bound_args.autotune(configs=configs, cache_results=False)
    print(f"\nAutotuning Complete ({len(autotuned_data)} configurations evaluated):", flush=True)
    best_config = autotuned_data.fastest_config
    valid_data = autotuned_data.prune_errors()
    best_data = valid_data[best_config]
    assert isinstance(best_data, benchmarking.BenchmarkData)
    print(f"  Best Config: {best_config}", flush=True)
    print(f"  Best Execution Time: {best_data.median_evaluation_time_ms:.3f} ms", flush=True)
    print(f"  Compile Time: {best_data.compile_time_ms:.3f} ms", flush=True)

    print("\nConfiguration Breakdown:", flush=True)
    print(
        f"{'bkv_sz':>8} {'decode_bs':>10} | "
        f"{'Eval Time (ms)':>15} | {'Compile Time (ms)':>18}",
        flush=True,
    )
    print("-" * 60, flush=True)
    for cfg, data in sorted(valid_data.items(), key=lambda x: x[1].median_evaluation_time_ms):
      print(
          f"{cfg.bkv_sz:>8} {cfg.decode_batch_size:>10} | "
          f"{data.median_evaluation_time_ms:>15.3f} | "
          f"{data.compile_time_ms:>18.3f}",
          flush=True,
      )

    best_speedup = ref_bench.median_evaluation_time_ms / best_data.median_evaluation_time_ms
    print(f"\nFinal Autotuned Speedup vs Reference: {best_speedup:.2f}x", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
  absltest.main()
