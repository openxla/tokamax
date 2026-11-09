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
"""Test shard definitions for `.github/workflows/ci-build.yml`.

Splits the test suite into parallel shards, one GitHub Actions job per
(shard, runner). Stdlib only so CI can run this before installing packages.

Every test file is named individually with no directory targets or shared
prefixes. Tests too slow for one job can be split by pytest node ID
(`path/to/test.py::SomeTest`) at the class boundary.

Usage:
  python3 .github/ci/shards.py matrix   # emit GitHub Actions matrix outputs
  python3 .github/ci/shards.py check    # check all test files are claimed
  python3 .github/ci/shards.py list     # print human-readable summary
  python3 .github/ci/shards.py explain  # explain test selection for changes
"""

from __future__ import annotations

import argparse
import ast
import collections
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from typing import TypedDict

# Sibling module, same directory, stdlib-only: computes affected test files.
import deps

# Runner label -> (device extra installed, short name used in job titles).
# Every shard runs on every runner.
RUNNERS = {
    'linux-x86-a3-8g-h100-1gpu': ('cuda', 'h100'),
    'linux-x86-ct6e-44-1tpu': ('tpu', 'tpu6e'),
    'linux-x86-tpu7x-56-1tpu': ('tpu', 'tpu7x'),
}

# pytest default `python_files` patterns.
PYTEST_FILE_PATTERNS = ('test_*.py', '*_test.py')

# Flags emitted as the pytest_flags workflow output to select tests.
PYTEST_SELECT_FLAGS = (
    # Skip tests marked `long` (marker declared in pyproject.toml).
    '-m',
    'not long',
    # Abstract base classes only; direct instantiation raises TypeError.
    '--ignore-glob=*/test_base.py',
    # Helper functions only; contains no test classes or functions.
    '--ignore-glob=*/test_utils.py',
    # Skipped until torch_tpu dependencies are available in CI.
    '--ignore-glob=*/torch_tpu/*',
)

# Patterns extracted from PYTEST_SELECT_FLAGS so checks ignore the same files.
IGNORED_GLOBS = tuple(
    flag.removeprefix('--ignore-glob=')
    for flag in PYTEST_SELECT_FLAGS
    if flag.startswith('--ignore-glob=')
)

# A shard's name must be a theme or start with `<theme>-`. Jobs are ordered by
# `shard_order`: catch-all first, then longest `minutes`, theme breaks ties.
THEMES = (
    'attention',
    'splash',
    'ragged-dot',
    'gmm',
    'ops',
    'experimental',
    'core',
    'catch-all',
)

# Test files intentionally run by no shard.
EXCLUDED_TESTS = ()

# JAX versions tested in CI: `latest_jax()` for the shard jobs, `older_jaxs()`
# for compat-tests. The oldest must match the floor in pyproject.toml.
JAX_VERSIONS = ('0.12.0', '0.11.2')

# TODO: Enable the backward compatibility test for GPU kernels by
# adding 'linux-x86-a3-8g-h100-1gpu' to COMPAT_RUNNERS.
COMPAT_RUNNERS = ('linux-x86-tpu7x-56-1tpu',)

# Target size in minutes for packing compat smoke tests into jobs.
COMPAT_BIN_MINUTES = 20

# Repo-relative path of this module.
SHARDS_PATH = '.github/ci/shards.py'

# Thresholds for table-only diffs before selection runs all shards.
MAX_DIFF_SHARDS = 10
MAX_DIFF_FRACTION = 0.25

# Assumed duration for unmeasured shards so they start alongside mid-sized jobs
# rather than last.
UNMEASURED_MINUTES = 10


class _RequiredSpec(TypedDict):
  """Required shard spec fields."""

  # None only on 'catch-all', populated by resolve_shards.
  paths: tuple[str, ...] | None


class Spec(_RequiredSpec, total=False):
  """Specification for one test shard in SHARDS."""

  minutes: float
  compat_minutes: float


# Mapping of shard name to Spec.
ShardMap = dict[str, Spec]

# Entry in GitHub Actions strategy.matrix.include.
MatrixEntry = dict[str, str]

# Shard definitions for CI test matrix generation.
#
# Spec fields:
#   paths: Test files or `path/to/test.py::ClassName` node IDs. Every test file
#     must be named individually with no directory targets; check_consistency
#     enforces this. None only on 'catch-all', populated by resolve_shards.
#   minutes: Estimated runtime in minutes, set by hand from CI 'Run shard' step
#     times (worst runner, rounded up, minimum 1). Omitted if unmeasured.
#   compat_minutes: Runtime in compat smoke mode on COMPAT_RUNNERS, used to
#     pack compat bins. Falls back to `minutes` when omitted.
#
# pylint: disable=line-too-long
SHARDS: ShardMap = {
    # -- attention ---------------------------------------------------------
    # Split across two shards by class node ID to balance runtime.
    'attention-base-vjp': Spec(
        paths=(
            'tokamax/_src/ops/attention/base_test.py::DotProductAttentionWithExplicitVjpTest',
        ),
        minutes=13,
        compat_minutes=9,
    ),
    'attention-base': Spec(
        paths=(
            'tokamax/_src/ops/attention/base_test.py::MaskTest',
            'tokamax/_src/ops/attention/base_test.py::DotProductAttentionTest',
        ),
        minutes=11,
        compat_minutes=8,
    ),
    'attention-xla-chunked': Spec(
        paths=('tokamax/_src/ops/attention/xla_chunked_test.py',),
        minutes=6,
        compat_minutes=3,
    ),
    # GPU-only Triton tests skip on TPU.
    'attention-triton': Spec(
        paths=('tokamax/_src/ops/attention/triton_test.py',),
        minutes=4,
        compat_minutes=1,
    ),
    'attention-api': Spec(
        paths=('tokamax/_src/ops/attention/api_test.py',),
        minutes=2,
        compat_minutes=1,
    ),
    # Tests skip when running on TPU.
    'attention-jax-nn': Spec(
        paths=('tokamax/_src/ops/attention/jax_nn_test.py',),
        minutes=2,
        compat_minutes=1,
    ),
    'attention-misc': Spec(
        paths=(
            'tokamax/_src/ops/attention/api_sharding_test.py',
            'tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py',
            'tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # -- splash ------------------------------------------------------------
    'splash-kernel': Spec(
        paths=(
            'tokamax/_src/ops/splash_attention/splash_attention_kernel_test.py',
        ),
        minutes=42,
        compat_minutes=2,
    ),
    'splash-misc': Spec(
        paths=(
            'tokamax/_src/ops/splash_attention/api_test.py',
            'tokamax/_src/ops/splash_attention/base_test.py',
            'tokamax/_src/ops/splash_attention/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/splash_attention/ring_attention_kernel_test.py',
            'tokamax/_src/ops/splash_attention/splash_attention_kernel_sharded_test.py',
            'tokamax/_src/ops/splash_attention/splash_attention_mask_test.py',
            'tokamax/_src/ops/splash_attention/splash_attention_segments_test.py',
        ),
        minutes=2,
        compat_minutes=2,
    ),
    # -- ragged-dot --------------------------------------------------------
    'ragged-dot-api': Spec(
        paths=('tokamax/_src/ops/ragged_dot/api_test.py',),
        minutes=25,
        compat_minutes=1,
    ),
    'ragged-dot-mosaic-gpu': Spec(
        paths=('tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py',),
        minutes=11,
        compat_minutes=1,
    ),
    'ragged-dot-misc': Spec(
        paths=(
            'tokamax/_src/ops/ragged_dot/base_test.py',
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py',
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py',
        ),
        minutes=7,
        compat_minutes=2,
    ),
    'ragged-dot-triton': Spec(
        paths=('tokamax/_src/ops/ragged_dot/triton_test.py',),
        minutes=6,
        compat_minutes=1,
    ),
    'ragged-dot-sm100': Spec(
        paths=(
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_quant_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # -- gmm ---------------------------------------------------------------
    'gmm-v2-kernel': Spec(
        paths=('tokamax/_src/ops/gmm_v2/gmm_v2_test.py',),
        minutes=4,
        compat_minutes=2,
    ),
    'gmm-v2-perf': Spec(
        paths=('tokamax/_src/ops/gmm_v2/gmm_v2_perf_test.py',),
        minutes=4,
        compat_minutes=3,
    ),
    # -- ops ---------------------------------------------------------------
    'ops-linear-softmax-xent': Spec(
        paths=(
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/base_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py',
        ),
        minutes=20,
        compat_minutes=4,
    ),
    'ops-causal-conv1d': Spec(
        paths=(
            'tokamax/_src/ops/causal_conv1d_gated_delta_rule/base_test.py',
            'tokamax/_src/ops/causal_conv1d_gated_delta_rule/gdn_attention_test.py',
            'tokamax/_src/ops/causal_conv1d_gated_delta_rule/kda_attention_test.py',
            'tokamax/_src/ops/causal_conv1d_gated_delta_rule/pallas_mosaic_tpu_test.py',
        ),
        minutes=15,
        compat_minutes=8,
    ),
    'ops-ragged-gather': Spec(
        paths=(
            'tokamax/_src/ops/ragged_gather/api_test.py',
            'tokamax/_src/ops/ragged_gather/base_test.py',
            'tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_test.py',
        ),
        minutes=7,
        compat_minutes=1,
    ),
    'ops-normalization': Spec(
        paths=(
            'tokamax/_src/ops/normalization/api_test.py',
            'tokamax/_src/ops/normalization/base_test.py',
            'tokamax/_src/ops/normalization/triton_test.py',
        ),
        minutes=3,
        compat_minutes=1,
    ),
    'ops-ragged-gather-reduce': Spec(
        paths=(
            'tokamax/_src/ops/ragged_gather_reduce/api_test.py',
            'tokamax/_src/ops/ragged_gather_reduce/base_test.py',
            'tokamax/_src/ops/ragged_gather_reduce/pallas_mosaic_tpu_test.py',
        ),
        minutes=3,
        compat_minutes=1,
    ),
    'ops-flex-attention': Spec(
        paths=(
            'tokamax/_src/ops/experimental/flex_attention/api_test.py',
            'tokamax/_src/ops/experimental/flex_attention/base_test.py',
        ),
        minutes=3,
        compat_minutes=1,
    ),
    'ops-triangle-multiplication': Spec(
        paths=(
            'tokamax/_src/ops/triangle_multiplication/api_test.py',
            'tokamax/_src/ops/triangle_multiplication/base_test.py',
            'tokamax/_src/ops/triangle_multiplication/pallas_mosaic_gpu_test.py',
            'tokamax/_src/ops/triangle_multiplication/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/triangle_multiplication/triton_test.py',
        ),
        minutes=2,
        compat_minutes=1,
    ),
    'ops-gated-linear-unit': Spec(
        paths=(
            'tokamax/_src/ops/gated_linear_unit/api_test.py',
            'tokamax/_src/ops/gated_linear_unit/base_test.py',
            'tokamax/_src/ops/gated_linear_unit/cutedsl_test.py',
            'tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_test.py',
            'tokamax/_src/ops/gated_linear_unit/triton_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # -- experimental ------------------------------------------------------
    'experimental-mla-v2-kernel': Spec(
        paths=('tokamax/_src/ops/experimental/mla/v2/mla_kernel_v2_test.py',),
        minutes=35,
        compat_minutes=4,
    ),
    # TPU-only kernel test; separated from experimental-kda to isolate runtimes.
    'experimental-kda-kernel': Spec(
        paths=('tokamax/_src/ops/kda/pallas_mosaic_tpu_kernel_test.py',),
        minutes=16,
        compat_minutes=1,
    ),
    'experimental-routing-moe': Spec(
        paths=(
            'tokamax/_src/ops/experimental/routing_moe/test/gmm/gmm_prod_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/layer/moe_prod_test.py',
        ),
        minutes=15,
        compat_minutes=1,
    ),
    # MoE GMM unit tests, split from the layer tests to run in parallel.
    'experimental-routing-moe-gmm-unit': Spec(
        paths=(
            'tokamax/_src/ops/experimental/routing_moe/test/common/dataclass_util_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/common/layout_util_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/common/module_scope_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/common/out_format_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/common/quant_util_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/gmm/config_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/gmm/gmm_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/gmm/metadata_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/gmm/plan_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/gmm/spec_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/helper/helper_inputs_unit_test.py',
        ),
        minutes=12,
    ),
    'experimental-masked-dense-mla': Spec(
        paths=(
            'tokamax/_src/ops/masked_dense_mla/api_test.py',
            'tokamax/_src/ops/masked_dense_mla/base_test.py',
            'tokamax/_src/ops/masked_dense_mla/kv_cache_utils_test.py',
            'tokamax/_src/ops/masked_dense_mla/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/masked_dense_mla/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/masked_dense_mla/scatter_test.py',
        ),
        minutes=10,
        compat_minutes=3,
    ),
    'experimental-routing-moe-layer-unit': Spec(
        paths=(
            'tokamax/_src/ops/experimental/routing_moe/test/layer/modular_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/layer/moe_plan_unit_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/layer/monolithic_unit_test.py',
        ),
        minutes=9,
    ),
    'experimental-routing-moe-output-reduce': Spec(
        paths=(
            'tokamax/_src/ops/experimental/routing_moe/test/io_relayout/output_reduce_prod_test.py',
            'tokamax/_src/ops/experimental/routing_moe/test/io_relayout/output_reduce_unit_test.py',
        ),
        minutes=8,
    ),
    'experimental-proj-and-save-state': Spec(
        paths=(
            'tokamax/_src/ops/proj_and_save_state/api_test.py',
            'tokamax/_src/ops/proj_and_save_state/base_test.py',
            'tokamax/_src/ops/proj_and_save_state/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/proj_and_save_state/pallas_mosaic_tpu_test.py',
        ),
        minutes=8,
        compat_minutes=3,
    ),
    'experimental-batched-rpa': Spec(
        paths=(
            'tokamax/_src/ops/batched_rpa/base_test.py',
            'tokamax/_src/ops/batched_rpa/kernel/configs_test.py',
            'tokamax/_src/ops/batched_rpa/kernel/seq_along_lane_parity_test.py',
            'tokamax/_src/ops/batched_rpa/kernel/speculative_decode_test.py',
            'tokamax/_src/ops/batched_rpa/kernel/stacked_rpa_kv_writeback_test.py',
            'tokamax/_src/ops/batched_rpa/pallas_mosaic_tpu_test.py',
        ),
        minutes=6,
        compat_minutes=2,
    ),
    'experimental-mla-v2': Spec(
        paths=(
            'tokamax/_src/ops/experimental/mla/v2/mla_transpose_test.py',
            'tokamax/_src/ops/experimental/mla/v2/test_mla_tuned_params.py',
        ),
        minutes=6,
        compat_minutes=1,
    ),
    'experimental-mhc': Spec(
        paths=(
            'tokamax/_src/ops/mhc/api_test.py',
            'tokamax/_src/ops/mhc/base_test.py',
            'tokamax/_src/ops/mhc/mhc_test.py',
            'tokamax/_src/ops/mhc/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/mhc/pallas_mosaic_tpu_test.py',
        ),
        minutes=5,
        compat_minutes=2,
    ),
    'experimental-dsv4-compressor': Spec(
        paths=(
            'tokamax/_src/ops/dsv4_compressor/api_test.py',
            'tokamax/_src/ops/dsv4_compressor/base_test.py',
            'tokamax/_src/ops/dsv4_compressor/dsv4_compressor_test.py',
            'tokamax/_src/ops/dsv4_compressor/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/dsv4_compressor/pallas_mosaic_tpu_test.py',
        ),
        minutes=4,
        compat_minutes=2,
    ),
    'experimental-fused-fp8-matmul': Spec(
        paths=(
            'tokamax/_src/ops/experimental/fused_fp8_matmul/base_test.py',
            'tokamax/_src/ops/experimental/fused_fp8_matmul/pallas_mosaic_tpu_dlhs_kernel_test.py',
            'tokamax/_src/ops/experimental/fused_fp8_matmul/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/experimental/fused_fp8_matmul/pallas_mosaic_tpu_vjp_test.py',
        ),
        minutes=4,
        compat_minutes=2,
    ),
    'experimental-grouped-lowrank-proj': Spec(
        paths=(
            'tokamax/_src/ops/grouped_lowrank_proj/api_test.py',
            'tokamax/_src/ops/grouped_lowrank_proj/base_test.py',
            'tokamax/_src/ops/grouped_lowrank_proj/grouped_lowrank_proj_test.py',
            'tokamax/_src/ops/grouped_lowrank_proj/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/grouped_lowrank_proj/pallas_mosaic_tpu_test.py',
        ),
        minutes=4,
        compat_minutes=2,
    ),
    'experimental-lightning-indexer': Spec(
        paths=(
            'tokamax/_src/ops/lightning_indexer/api_test.py',
            'tokamax/_src/ops/lightning_indexer/base_test.py',
            'tokamax/_src/ops/lightning_indexer/kernel/metadata_test.py',
            'tokamax/_src/ops/lightning_indexer/kernel/streamindex_topk_dcp_test.py',
            'tokamax/_src/ops/lightning_indexer/kernel/streamindex_topk_test.py',
            'tokamax/_src/ops/lightning_indexer/kernel/tc_streamindex_topk_test.py',
            'tokamax/_src/ops/lightning_indexer/kernel/tc_topk_test.py',
            'tokamax/_src/ops/lightning_indexer/pallas_mosaic_tpu_test.py',
        ),
        minutes=10,
        compat_minutes=3,
    ),
    'experimental-mhc-lite': Spec(
        paths=(
            'tokamax/_src/ops/experimental/mhc_lite/api_test.py',
            'tokamax/_src/ops/experimental/mhc_lite/mhc_test.py',
        ),
        minutes=4,
        compat_minutes=1,
    ),
    'experimental-mla-swa': Spec(
        paths=(
            'tokamax/_src/ops/experimental/mla_swa/api_test.py',
            'tokamax/_src/ops/experimental/mla_swa/base_test.py',
            'tokamax/_src/ops/experimental/mla_swa/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/experimental/mla_swa/pallas_mosaic_tpu_test.py',
        ),
        minutes=4,
        compat_minutes=3,
    ),
    'experimental-dedup-moe': Spec(
        paths=(
            'tokamax/_src/ops/experimental/dedup_moe/test/dedup_plan/dedup_plan_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/dedup_plan/source_plan_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/dedup_plan/topology_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/dispatch/wave_dispatch_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/dispatch/wave_order_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/layer/dedup_moe_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/layer/layer_interpret_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/layer/mode_registry_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/layer/modular_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/layer/padding_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/layer/plan_test.py',
            'tokamax/_src/ops/experimental/dedup_moe/test/layer/tpu_correctness_test.py',
        ),
        minutes=3,
    ),
    'experimental-deepseek-rope': Spec(
        paths=(
            'tokamax/_src/ops/deepseek_rope/api_test.py',
            'tokamax/_src/ops/deepseek_rope/base_test.py',
            'tokamax/_src/ops/deepseek_rope/deepseek_rope_test.py',
            'tokamax/_src/ops/deepseek_rope/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/deepseek_rope/pallas_mosaic_tpu_test.py',
        ),
        minutes=3,
        compat_minutes=1,
    ),
    'experimental-csa-gather': Spec(
        paths=(
            'tokamax/_src/ops/csa_gather/api_test.py',
            'tokamax/_src/ops/csa_gather/base_test.py',
            'tokamax/_src/ops/csa_gather/csa_gather_test.py',
            'tokamax/_src/ops/csa_gather/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/csa_gather/pallas_mosaic_tpu_test.py',
        ),
        minutes=2,
        compat_minutes=1,
    ),
    'experimental-kda': Spec(
        paths=(
            'tokamax/_src/ops/kda/base_test.py',
            'tokamax/_src/ops/kda/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/kda/xla_chunked_test.py',
        ),
        minutes=2,
        compat_minutes=1,
    ),
    'experimental-mla': Spec(
        paths=(
            'tokamax/_src/ops/experimental/mla/base_test.py',
            'tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py',
        ),
        minutes=2,
        compat_minutes=2,
    ),
    'experimental-router-topk': Spec(
        paths=(
            'tokamax/_src/ops/router_topk/api_test.py',
            'tokamax/_src/ops/router_topk/base_test.py',
            'tokamax/_src/ops/router_topk/pallas_mosaic_tpu_test.py',
        ),
        minutes=2,
        compat_minutes=1,
    ),
    'experimental-sparse-mla': Spec(
        paths=(
            'tokamax/_src/ops/sparse_mla/api_test.py',
            'tokamax/_src/ops/sparse_mla/base_test.py',
            'tokamax/_src/ops/sparse_mla/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/sparse_mla/pallas_mosaic_tpu_test.py',
        ),
        minutes=2,
        compat_minutes=2,
    ),
    'experimental-fused-moe': Spec(
        paths=(
            'tokamax/_src/ops/experimental/fused_moe/api_test.py',
            'tokamax/_src/ops/experimental/fused_moe/base_test.py',
            'tokamax/_src/ops/experimental/fused_moe/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/experimental/fused_moe/test_fused_ep_moe_v2.py',
            'tokamax/_src/ops/experimental/fused_moe/test_fused_ep_moe_v2_router.py',
            'tokamax/_src/ops/experimental/fused_moe/test_fused_ep_moe_v2_tables.py',
            'tokamax/_src/ops/experimental/fused_moe/test_fused_ep_moe_v2_tp_tokens.py',
            'tokamax/_src/ops/experimental/fused_moe/test_fused_ep_moe_v2_weight_slots.py',
            'tokamax/_src/ops/experimental/fused_moe/test_tp_token_layout.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    'experimental-fused-moe-decode': Spec(
        paths=(
            'tokamax/_src/ops/experimental/fused_moe_decode/api_test.py',
            'tokamax/_src/ops/experimental/fused_moe_decode/base_test.py',
            'tokamax/_src/ops/experimental/fused_moe_decode/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/experimental/fused_moe_decode/test_fused_moe_decode_kernel.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    'experimental-gdn2': Spec(
        paths=(
            'tokamax/_src/ops/experimental/gdn2/base_test.py',
            'tokamax/_src/ops/experimental/gdn2/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    'experimental-mla-v3': Spec(
        paths=(
            'tokamax/_src/ops/experimental/mla/v3/flash_attention_test.py',
            'tokamax/_src/ops/experimental/mla/v3/mla_kernel_v3_test.py',
            'tokamax/_src/ops/experimental/mla/v3/schedule_test.py',
        ),
        minutes=40,
        compat_minutes=1,
    ),
    # See experimental-gdn2.
    'experimental-moba': Spec(
        paths=(
            'tokamax/_src/ops/experimental/moba/base_test.py',
            'tokamax/_src/ops/experimental/moba/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # See experimental-gdn2.
    'experimental-nsa': Spec(
        paths=(
            'tokamax/_src/ops/experimental/nsa/base_test.py',
            'tokamax/_src/ops/experimental/nsa/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # See experimental-gdn2.
    'experimental-raven': Spec(
        paths=(
            'tokamax/_src/ops/experimental/raven/base_test.py',
            'tokamax/_src/ops/experimental/raven/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # See experimental-gdn2.
    'experimental-rwkv7': Spec(
        paths=(
            'tokamax/_src/ops/experimental/rwkv7/base_test.py',
            'tokamax/_src/ops/experimental/rwkv7/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # Separated from experimental-topk-kernel to isolate TPU-only tests.
    'experimental-topk': Spec(
        paths=(
            'tokamax/_src/ops/topk/api_test.py',
            'tokamax/_src/ops/topk/base_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    'experimental-topk-kernel': Spec(
        paths=(
            'tokamax/_src/ops/topk/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/topk/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # -- core --------------------------------------------------------------
    'core-autotuning': Spec(
        paths=(
            'tokamax/_src/autotuning/api_test.py',
            'tokamax/_src/autotuning/cache_test.py',
        ),
        minutes=2,
        compat_minutes=1,
    ),
    # Public-API smoke test for import or packaging breaks.
    'core-api': Spec(
        paths=('tokamax/tokamax_test.py',),
        minutes=1,
        compat_minutes=1,
    ),
    # Internal library tests across tokamax/_src/.
    'core-utils': Spec(
        paths=(
            'tokamax/_src/ad_test.py',
            'tokamax/_src/batching_test.py',
            'tokamax/_src/benchmarking_test.py',
            'tokamax/_src/config_test.py',
            'tokamax/_src/gpu_utils_test.py',
            'tokamax/_src/hlo_utils_common_test.py',
            'tokamax/_src/hlo_utils_test.py',
            'tokamax/_src/jaxtyping_test.py',
            'tokamax/_src/mosaic_gpu_test.py',
            'tokamax/_src/numerics_test.py',
            'tokamax/_src/ops/op_test.py',
            'tokamax/_src/ops/ops_list_test.py',
            'tokamax/_src/precision_test.py',
            'tokamax/_src/pydantic_test.py',
            'tokamax/_src/shape_test.py',
            'tokamax/_src/test_utils_test.py',
        ),
        minutes=1,
        compat_minutes=1,
    ),
    # -- catch-all ---------------------------------------------------------
    # Populated by resolve_shards with tests not claimed by any other shard.
    'catch-all': Spec(
        paths=None,
        minutes=1,
    ),
}
# pylint: enable=line-too-long

_CATCH_ALL_SHARD = 'catch-all'


# Matches requirement floors like 'jax>=0.11.0' or 'jax[tpu]>=0.11.0'.
_JAX_REQUIREMENT_RE = re.compile(r'(jax|jaxlib)(\[[^\]]*\])?>=([0-9][^,;\s]*)')


def jax_floor(pyproject: str = 'pyproject.toml') -> str:
  """Returns the oldest JAX that pyproject.toml claims to support."""
  with open(pyproject, 'rb') as f:
    project = tomllib.load(f)['project']

  requirements = list(project.get('dependencies', ()))
  for extra in project.get('optional-dependencies', {}).values():
    requirements.extend(extra)

  floors = {
      m[3]
      for r in requirements
      if (m := _JAX_REQUIREMENT_RE.fullmatch(r.strip()))
  }
  if not floors:
    raise ValueError(f'{pyproject} declares no `jax>=` requirement')
  if len(floors) > 1:
    raise ValueError(
        f'{pyproject} declares more than one JAX floor: '
        + ', '.join(sorted(floors))
    )
  return floors.pop()


def _version_key(version: str) -> tuple[int, ...]:
  return tuple(int(part) for part in version.split('.'))


def sorted_jax_versions() -> tuple[str, ...]:
  """Returns JAX_VERSIONS in version order, newest first."""
  return tuple(sorted(JAX_VERSIONS, key=_version_key, reverse=True))


def latest_jax() -> str:
  """Returns the newest JAX version CI tests."""
  return sorted_jax_versions()[0]


def older_jaxs() -> tuple[str, ...]:
  """Returns all supported JAX versions except the latest."""
  return sorted_jax_versions()[1:]


def oldest_jax() -> str:
  """Returns the oldest supported JAX version."""
  return sorted_jax_versions()[-1]


def _is_test_filename(name: str) -> bool:
  """Returns whether pytest would collect a file with this base name.

  Args:
    name: Base name of a file, without directory components.

  Returns:
    True if the name matches `PYTEST_FILE_PATTERNS`.
  """
  return any(fnmatch.fnmatch(name, p) for p in PYTEST_FILE_PATTERNS)


def all_test_files(root: str = 'tokamax') -> list[str]:
  """Returns sorted repo-relative paths matching PYTEST_FILE_PATTERNS.

  Args:
    root: Directory to walk, relative to the repository root.

  Returns:
    Sorted repo-relative paths matching pytest patterns.
  """
  found = []
  for dirpath, _, filenames in os.walk(root):
    if '__pycache__' in dirpath:
      continue
    for name in filenames:
      if _is_test_filename(name):
        found.append(os.path.join(dirpath, name).replace(os.sep, '/'))
  return sorted(found)


def split_node_id(path: str) -> tuple[str, str | None]:
  """Splits a paths entry into (file_path, class_name or None).

  Args:
    path: Shard path, either a test file or `path/to/test.py::ClassName`.

  Returns:
    Tuple of (file_path, class_name or None).
  """
  file, sep, selector = path.partition('::')
  return file, (selector if sep else None)


def declared_test_classes(path: str) -> list[str]:
  """Returns top-level class names in path that a split shard must name.

  Private classes (_*) are excluded unless inheriting from a test base.

  Args:
    path: Path to the test file to parse.

  Returns:
    Top-level test class names in source order.

  Raises:
    OSError: If the file cannot be read.
    SyntaxError: If the file contains invalid Python syntax.
  """
  with open(path, encoding='utf-8') as f:
    tree = ast.parse(f.read(), filename=path)

  # Traverse statements that can hold module-level classes (if/try blocks).
  body = list(tree.body)
  names = []
  while body:
    node = body.pop(0)
    if isinstance(node, (ast.If, ast.Try)):
      body = [
          *node.body,
          *node.orelse,
          *getattr(node, 'finalbody', []),
          *[s for h in getattr(node, 'handlers', []) for s in h.body],
          *body,
      ]
      continue
    if not isinstance(node, ast.ClassDef):
      continue
    bases = [
        b.attr if isinstance(b, ast.Attribute) else getattr(b, 'id', '')
        for b in node.bases
    ]
    private = node.name.startswith('_') and not any(
        b.endswith(('Test', 'TestCase', 'TestBase')) for b in bases
    )
    if not private:
      names.append(node.name)
  return names


def _matches_glob(path: str, globs: Iterable[str]) -> bool:
  """Returns whether path matches any of the given fnmatch globs.

  Args:
    path: Repo-relative file path.
    globs: Glob patterns to match against.

  Returns:
    True if the path matches any pattern, else False.
  """
  return any(fnmatch.fnmatch(path, g) for g in globs)


def collected_by(paths: Iterable[str], test_files: Collection[str]) -> set[str]:
  """Returns the subset of test_files named in paths, minus IGNORED_GLOBS.

  Args:
    paths: Shard path specs or node IDs.
    test_files: Test file paths to check against.

  Returns:
    Subset of test files collected, excluding `IGNORED_GLOBS`.
  """
  named = {split_node_id(p)[0] for p in paths}
  return {
      f
      for f in test_files
      if f in named and not _matches_glob(f, IGNORED_GLOBS)
  }


def _matches(test_files: Iterable[str], globs: Iterable[str]) -> set[str]:
  """Returns the subset of test_files matching any glob.

  Args:
    test_files: Repo-relative file paths.
    globs: Glob patterns to match against.

  Returns:
    Paths matching at least one glob pattern.
  """
  return {f for f in test_files if _matches_glob(f, globs)}


def resolve_shards(
    test_files: Collection[str] | None = None,
    shard_map: Mapping[str, Spec] | None = None,
) -> tuple[ShardMap, list[str], set[str], set[str]]:
  """Resolves the catch-all shard and partitions test_files.

  Args:
    test_files: Test file paths to shard, or None to discover from disk.
    shard_map: Shard definitions to resolve, defaulting to `SHARDS`.

  Returns:
    Tuple of (resolved_shards, catch_all_files, excluded_files, ignored_files).
    The catch-all shard is omitted if empty.

  Raises:
    ValueError: If catch-all shard is missing, has paths defined, or another
      shard has paths=None.
  """
  test_files = all_test_files() if test_files is None else test_files
  source = SHARDS if shard_map is None else shard_map
  if (
      _CATCH_ALL_SHARD not in source
      or source[_CATCH_ALL_SHARD].get('paths') is not None
  ):
    raise ValueError(
        f'{_CATCH_ALL_SHARD!r} shard must be present with paths=None'
    )
  for name, spec in source.items():
    if name != _CATCH_ALL_SHARD and spec.get('paths') is None:
      raise ValueError(f'shard {name!r} must not have paths=None')
  shards = {n: dict(s) for n, s in source.items()}

  claimed = set()
  for spec in shards.values():
    if spec['paths'] is not None:
      claimed |= collected_by(spec['paths'], test_files)

  ignored = _matches(test_files, IGNORED_GLOBS)
  excluded = _matches(test_files, EXCLUDED_TESTS)
  catch_all = sorted(set(test_files) - claimed - excluded - ignored)

  if catch_all:
    shards[_CATCH_ALL_SHARD]['paths'] = tuple(catch_all)
  else:
    # Pytest with empty paths collects everything; delete empty catch-all.
    del shards[_CATCH_ALL_SHARD]
  return shards, catch_all, excluded, ignored


jax_version_re = re.compile(r'\d+(\.\d+)*')


def check_consistency(
    shards: ShardMap,
    test_files: Collection[str],
    excluded: Collection[str],
    ignored: Collection[str],
    class_reader: Callable[[str], list[str]] = declared_test_classes,
    pyproject: str | None = 'pyproject.toml',
) -> list[str]:
  """Validates shard configuration and returns a list of error descriptions.

  Args:
    shards: Resolved shard mapping to check.
    test_files: Universe of test files the shards were resolved against.
    excluded: Intentionally unrun test files.
    ignored: Test files ignored by pytest configuration.
    class_reader: Callable returning declared test classes in a file.
    pyproject: Path to pyproject.toml for JAX floor verification, or None.

  Returns:
    List of error messages describing configuration issues, empty if valid.
  """
  errors = []
  claims = collections.defaultdict(list)
  for name, spec in sorted(shards.items()):
    if not spec['paths']:
      errors.append(f'shard {name!r} has no paths')
      continue
    if flags := [t for t in spec['paths'] if t.startswith('-')]:
      errors.append(
          f'shard {name!r} passes flags in paths: {", ".join(flags)} -- paths'
          ' lists test files only, because a flag can narrow what pytest'
          ' collects in a way `collected_by` cannot see. Put selection flags'
          ' in PYTEST_SELECT_FLAGS.'
      )
    targets = [split_node_id(p) for p in spec['paths'] if p not in flags]
    missing = sorted({f for f, _ in targets} - set(test_files))
    if missing:
      errors.append(
          f'shard {name!r} names paths that are not test files:'
          f' {", ".join(missing)}'
      )

    files = collected_by(spec['paths'], test_files)
    if not files:
      errors.append(
          f'shard {name!r} collects no test files -- a target was probably'
          ' moved or renamed'
      )
    for file, selector in targets:
      if file in files:
        claims[file].append((name, selector))
    for key in ('minutes', 'compat_minutes'):
      value = spec.get(key)
      if value is not None and (
          not isinstance(value, (int, float)) or value <= 0
      ):
        errors.append(
            f'shard {name!r} has {key}={value!r}: expected a positive'
            ' number, or the key omitted if unmeasured'
        )
    if 'devices' in spec:
      errors.append(
          f'shard {name!r} has a `devices` key: every shard runs on every'
          ' runner, so this pins nothing. Drop it.'
      )

  uncovered = sorted(set(test_files) - set(claims) - excluded - ignored)
  if uncovered:
    errors.append(f'test files no shard runs: {", ".join(uncovered)}')

  if strays := sorted(n for n in shards if shard_theme(n) is None):
    errors.append(
        'shards whose name starts with no theme in THEMES:'
        f' {", ".join(strays)} -- rename to `<theme>-<what it runs>`, or add'
        ' a theme'
    )

  short_counts = collections.Counter(short for _, short in RUNNERS.values())
  if dupes := sorted(s for s, n in short_counts.items() if n > 1):
    errors.append(f'runner short names used twice: {", ".join(dupes)}')

  for glob in EXCLUDED_TESTS:
    if not _matches(test_files, (glob,)):
      errors.append(f'EXCLUDED_TESTS entry {glob!r} matches no test file')

  for file, entries in sorted(claims.items()):
    selectors = [s for _, s in entries]
    where = ', '.join(sorted({n for n, _ in entries}))
    if selectors.count(None) > 1:
      errors.append(
          f'{file} is run by more than one shard, in {where}: the tests still'
          ' run, just twice, on scarce hardware'
      )
      continue
    if len(entries) > 1 and None in selectors:
      errors.append(
          f'{file} is claimed both whole and by node ID, in {where}: the'
          ' whole-file claim already runs every class in it, so each node ID'
          ' runs its class a second time'
      )
      continue
    if dupes := sorted({s for s in selectors if selectors.count(s) > 1}):
      errors.append(
          f'{file} is claimed more than once, in {where}: {", ".join(dupes)}'
      )
      continue
    if selectors == [None]:
      continue

    # Split by node ID: ensure all declared classes are assigned to shards.
    declared = set(class_reader(file))
    if orphans := sorted(declared - set(selectors)):
      errors.append(
          f'{file} is split by node ID across {where}, but no shard names'
          f' {", ".join(orphans)} -- add each to a shard, or, if it holds no'
          ' tests, give it a leading underscore'
      )
    if unknown := sorted(set(selectors) - declared):
      errors.append(
          f'{file} is split by node ID across {where}, which name classes it'
          f' does not declare: {", ".join(unknown)}'
      )

  for runner in COMPAT_RUNNERS:
    if runner not in RUNNERS:
      errors.append(f'COMPAT_RUNNERS {runner!r} is not in RUNNERS.')

  if len(JAX_VERSIONS) != 2:
    errors.append(
        f'JAX_VERSIONS is {JAX_VERSIONS}: tokamax supports 2 latest JAX'
        ' versions for backward compatibility.'
    )
  if sorted({v for v in JAX_VERSIONS if JAX_VERSIONS.count(v) > 1}):
    errors.append(f'Duplicated JAX version found in {JAX_VERSIONS}.')
  if bad := sorted(v for v in JAX_VERSIONS if not jax_version_re.fullmatch(v)):
    errors.append(
        'Invalid JAX version found in JAX_VERSIONS: ' + ', '.join(bad)
    )
    return errors

  if pyproject is not None:
    try:
      if (floor := jax_floor(pyproject)) != oldest_jax():
        errors.append(
            f'{pyproject} declares jax>={floor}, but the oldest version in'
            f' JAX_VERSIONS is {oldest_jax()}: the JAX lower bound and the'
            ' oldest JAX version the CI tests against have to be the same.'
        )
    except ValueError as exc:
      errors.append(str(exc))
  return errors


def shard_theme(name: str) -> tuple[int, str] | None:
  """Returns (index, theme) if the shard name matches a theme, else None.

  Args:
    name: Shard name.

  Returns:
    Tuple of (theme_index, theme_name), or None if no theme matches.
  """
  for index, theme in enumerate(THEMES):
    if name == theme or name.startswith(f'{theme}-'):
      return index, theme
  return None


def shard_order(item: tuple[str, Spec]) -> tuple[bool, float, int, str]:
  """Sort key: catch-all first, then longest runtime, then theme, then name.

  Args:
    item: Tuple of (shard_name, spec).

  Returns:
    Tuple sort key ordering catch-all first, descending duration, theme, name.
  """
  name, spec = item
  theme = shard_theme(name)
  minutes = spec.get('minutes')
  return (
      name != _CATCH_ALL_SHARD,
      -(UNMEASURED_MINUTES if minutes is None else minutes),
      len(THEMES) if theme is None else theme[0],
      name,
  )


def shards_for_tests(
    shards: ShardMap, test_files: Collection[str], affected: Collection[str]
) -> set[str]:
  """Returns the names of shards that run any of the affected test files.

  Args:
    shards: Resolved shard mapping.
    test_files: Universe of all test files.
    affected: Test file paths reached by a change.

  Returns:
    Set of shard names collecting any affected test file.
  """
  affected = set(affected)
  return {
      name
      for name, spec in shards.items()
      if spec['paths'] and collected_by(spec['paths'], test_files) & affected
  }


def unclaimed_tests(
    shards: ShardMap, test_files: Collection[str], affected: Collection[str]
) -> set[str]:
  """Returns affected test files that are not covered by any shard.

  Ignores files matching IGNORED_GLOBS or EXCLUDED_TESTS.

  Args:
    shards: Resolved shard mapping.
    test_files: Universe of all test files.
    affected: Test file paths reached by a change.

  Returns:
    Affected test files not collected by any shard.
  """
  claimed = set()
  for spec in shards.values():
    if spec['paths']:
      claimed |= collected_by(spec['paths'], test_files)
  deliberate = IGNORED_GLOBS + EXCLUDED_TESTS
  return {
      f
      for f in affected
      if f not in claimed and not _matches_glob(f, deliberate)
  }


def build_matrix(
    shards: ShardMap, only: Collection[str] | None = None
) -> list[MatrixEntry]:
  """Builds strategy.matrix.include entries for primary shard jobs.

  Args:
    shards: Resolved shard mapping.
    only: Shard names to include. None includes all shards.

  Returns:
    Matrix entries for each (shard, runner) combination.
  """
  ordered = sorted(shards.items(), key=shard_order)
  jobs = []
  for runner, (device, short) in RUNNERS.items():
    for name, spec in ordered:
      if only is not None and name not in only:
        continue
      if not spec['paths']:
        continue
      jobs.append({
          'runner': runner,
          'runner_short': f'{short}-jax{latest_jax()}',
          'device': device,
          'shard_name': name,
          'test_paths': shlex.join(spec['paths']),
          'jax_pin': latest_jax(),
      })
  return jobs


def compat_size(spec: Spec) -> float:
  """Estimated runtime in minutes for a shard in a compat bin.

  Args:
    spec: Shard specification.

  Returns:
    Estimated runtime in minutes, or `COMPAT_BIN_MINUTES` if unmeasured.
  """
  return spec.get('compat_minutes') or spec.get('minutes') or COMPAT_BIN_MINUTES


def compat_bins(
    shards: ShardMap, only: Collection[str] | None = None
) -> list[list[str]]:
  """Packs shards into bins of COMPAT_BIN_MINUTES using first-fit decreasing.

  Args:
    shards: Resolved shard mapping.
    only: Shard names to include, or None for all shards.

  Returns:
    List of bins, each containing shard names packed together.
  """
  sizes = {
      name: compat_size(spec)
      for name, spec in shards.items()
      if spec['paths'] and (only is None or name in only)
  }
  bins: list[list[str]] = []
  loads: list[float] = []
  for name in sorted(sizes, key=lambda n: (-sizes[n], n)):
    for i, load in enumerate(loads):
      if load + sizes[name] <= COMPAT_BIN_MINUTES:
        loads[i] += sizes[name]
        bins[i].append(name)
        break
    else:
      bins.append([name])
      loads.append(sizes[name])
  return [bins[i] for i in sorted(range(len(bins)), key=lambda i: -loads[i])]


def build_compat_matrix(
    shards: ShardMap, only: Collection[str] | None = None
) -> list[MatrixEntry]:
  """Builds strategy.matrix.include entries for compat-tests jobs.

  Args:
    shards: Resolved shard mapping.
    only: Shard names to include, or None for all shards.

  Returns:
    Matrix entries for each compat job combination.
  """
  bins = compat_bins(shards, only)
  jobs = []
  for runner in COMPAT_RUNNERS:
    device, short = RUNNERS[runner]
    for index, names in enumerate(bins, start=1):
      paths = [p for name in names for p in shards[name]['paths']]
      for version in older_jaxs():
        jobs.append({
            'runner': runner,
            'runner_short': f'{short}-jax{version}',
            'device': device,
            'shard_name': f'compat-{index}',
            'shards': ' '.join(names),
            'test_paths': shlex.join(paths),
            'jax_pin': version,
        })
  return jobs


def read_changed(path: str) -> list[str]:
  """Reads a NUL-separated path list from file.

  Args:
    path: Path to file containing NUL-separated paths.

  Returns:
    List of non-empty paths read from the file.
  """
  with open(path, encoding='utf-8') as f:
    return [p for p in f.read().split('\0') if p]


def _assigns_shards(node: ast.stmt) -> bool:
  """Returns whether node is a statement assigning to SHARDS."""
  if isinstance(node, ast.AnnAssign):
    return isinstance(node.target, ast.Name) and node.target.id == 'SHARDS'
  if isinstance(node, ast.Assign):
    return any(
        isinstance(t, ast.Name) and t.id == 'SHARDS' for t in node.targets
    )
  return False


def _literal_spec(node: ast.expr) -> Spec:
  """Parses an AST node into a literal Spec dictionary without executing it.

  Args:
    node: AST expression defining a shard spec.

  Returns:
    Parsed Spec dictionary.

  Raises:
    ValueError: If node is not a valid literal Spec or missing required paths.
  """
  try:
    if isinstance(node, ast.Dict):
      spec = ast.literal_eval(node)
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ('Spec', 'dict')
        and not node.args
        and all(k.arg is not None for k in node.keywords)
    ):
      spec = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
      if len(spec) != len(node.keywords):
        raise ValueError('a shard spec repeats a keyword')
    else:
      raise ValueError(f'not a literal shard spec: {ast.unparse(node)}')
  except TypeError as exc:
    raise ValueError(f'not a literal shard spec: {ast.unparse(node)}') from exc
  if not isinstance(spec, dict) or 'paths' not in spec:
    raise ValueError(f'shard spec has no paths: {ast.unparse(node)}')
  paths = spec['paths']
  if paths is not None and not (
      isinstance(paths, tuple) and all(isinstance(p, str) for p in paths)
  ):
    raise ValueError(f'shard paths is not a tuple of strings: {paths!r}')
  return spec


def _split_shards_table(source: str) -> tuple[str, ShardMap]:
  """Separates source into an AST dump (with SHARDS=None) and parsed table.

  Args:
    source: Python source code of shards.py.

  Returns:
    Tuple of (ast_dump_without_table, parsed_shards_map).

  Raises:
    SyntaxError: If source cannot be parsed as valid Python.
    ValueError: If SHARDS is not uniquely defined as a literal dictionary.
  """
  tree = ast.parse(source)
  assignments = [node for node in tree.body if _assigns_shards(node)]
  if len(assignments) != 1:
    raise ValueError(f'{len(assignments)} assignments to SHARDS, expected 1')
  (statement,) = assignments
  if not isinstance(statement.value, ast.Dict):
    raise ValueError('SHARDS is not a dict display')

  table = {}
  for key, value in zip(
      statement.value.keys, statement.value.values, strict=True
  ):
    if key is None:
      raise ValueError('SHARDS unpacks another mapping')
    try:
      name = ast.literal_eval(key)
    except TypeError as exc:
      raise ValueError(
          f'shard name is not a string: {ast.unparse(key)}'
      ) from exc
    if not isinstance(name, str):
      raise ValueError(f'shard name is not a string: {name!r}')
    if name in table:
      raise ValueError(f'shard {name!r} is declared twice')
    spec = _literal_spec(value)
    if name != _CATCH_ALL_SHARD and spec['paths'] is None:
      raise ValueError(f'shard {name!r} must not have paths=None')
    table[name] = spec

  statement.value = ast.Constant(value=None)
  return ast.dump(tree), table


def read_base_shards() -> str:
  """Returns shards.py as of the merge base of HEAD and origin/<base branch>.

  Returns:
    Source code of shards.py at the merge base.

  Raises:
    OSError: If the git command cannot be executed.
    subprocess.SubprocessError: If git fails to find merge base or read file.
  """
  base = os.environ.get('BASE_REF') or os.environ.get('GITHUB_BASE_REF')
  ref = f'origin/{base or "main"}'
  fork = subprocess.run(
      ['git', 'merge-base', ref, 'HEAD'],
      capture_output=True,
      text=True,
      check=True,
  ).stdout.strip()
  return subprocess.run(
      ['git', 'show', f'{fork}:{SHARDS_PATH}'],
      capture_output=True,
      text=True,
      check=True,
  ).stdout


def read_head_shards() -> str:
  """Reads the current version of this file."""
  with open(__file__, encoding='utf-8') as f:
    return f.read()


def shards_from_table_diff(
    test_files: Collection[str],
    base_source: str | None = None,
    head_source: str | None = None,
    class_reader: Callable[[str], list[str]] = declared_test_classes,
) -> tuple[set[str] | None, str]:
  """Selects shards affected by changes to the SHARDS table.

  Args:
    test_files: Universe of test files to resolve against.
    base_source: Source code of base shards.py, or None to read from git.
    head_source: Source code of changed shards.py, or None to read current file.
    class_reader: Callable returning declared test classes in a file.

  Returns:
    Tuple of (shards_to_run, reason). Shards is None if all shards must run.
  """
  try:
    base_source = read_base_shards() if base_source is None else base_source
  except (OSError, subprocess.SubprocessError):
    return None, 'shards.py: could not load the base version'
  try:
    head_source = read_head_shards() if head_source is None else head_source
  except OSError:
    return None, 'shards.py: could not load the head version'

  try:
    base_rest, base_table = _split_shards_table(base_source)
    head_rest, head_table = _split_shards_table(head_source)
  except (SyntaxError, TypeError, ValueError) as exc:
    return None, f'shards.py: could not read the SHARDS table ({exc})'
  if base_rest != head_rest:
    return None, 'shards.py: changes outside the SHARDS table'

  try:
    base, *_ = resolve_shards(test_files, base_table)
    head, _, excluded, ignored = resolve_shards(test_files, head_table)
  except (KeyError, TypeError, ValueError) as exc:
    return None, f'shards.py: could not resolve the SHARDS table ({exc})'
  if errors := check_consistency(
      head, test_files, excluded, ignored, class_reader, pyproject=None
  ):
    return None, f'shards.py: head table fails check_consistency: {errors[0]}'

  # Find shards whose resolved paths differ from any base shard.
  before = {
      frozenset(s['paths']) for n, s in base.items() if n != _CATCH_ALL_SHARD
  }
  base_catch_all = frozenset(base.get(_CATCH_ALL_SHARD, {}).get('paths', ()))
  names = set()
  for name, spec in head.items():
    paths = frozenset(spec['paths'])
    if name == _CATCH_ALL_SHARD:
      if paths != base_catch_all:
        names.add(name)
    elif paths not in before:
      names.add(name)

  if len(names) > MAX_DIFF_SHARDS or len(names) > MAX_DIFF_FRACTION * len(head):
    return None, (
        f'shards.py: table diff touches {len(names)} of {len(head)} shards,'
        ' over the diff-aware bound'
    )
  return names, (
      f"shards.py: table-only change; {len(names)} of {len(head)} shards'"
      ' test paths changed'
  )


def _table_diff(
    shards: ShardMap, test_files: Collection[str], changed: Sequence[str]
) -> tuple[list[str], set[str] | None, str | None]:
  """Splits off SHARDS_PATH; returns (other_files, table_shards, reason).

  Args:
    shards: Resolved shard mapping.
    test_files: Universe of all test files.
    changed: Changed file paths.

  Returns:
    Tuple of (other_files, table_shards, reason).
  """
  if SHARDS_PATH not in changed:
    return list(changed), set(), None
  others = [p for p in changed if p != SHARDS_PATH]
  names, why = shards_from_table_diff(test_files)
  if names is not None and (unknown := sorted(names - set(shards))):
    return (
        others,
        None,
        f'shards.py: table diff names unknown shard {unknown[0]}',
    )
  return others, names, why


def select(
    shards: ShardMap, test_files: Collection[str], changed_from: str | None
) -> tuple[set[str] | None, str]:
  """Selects shards to run for changed files, returning (shard_names, reason).

  Args:
    shards: Resolved shard mapping.
    test_files: Universe of all test files.
    changed_from: File containing NUL-separated changed paths, or None for all.

  Returns:
    Tuple of (shard_names, reason). Shards is None if all shards must run.
  """
  if changed_from is None:
    return None, 'no changed-file list: every shard runs'

  changed = read_changed(changed_from)
  if not changed:
    return None, 'changed-file list is empty: every shard runs'

  others, table_names, table_why = _table_diff(shards, test_files, changed)
  if table_names is None:
    return None, f'every shard runs: {table_why}'

  affected = set()
  if others:
    affected, why = deps.affected_tests(others)
    if affected is None:
      return None, f'every shard runs: {why}'

  # Fall back to all shards if any affected test is unassigned.
  if orphans := unclaimed_tests(shards, test_files, affected):
    return None, (
        f'every shard runs: {len(orphans)} affected test file(s) belong to no'
        f' shard, first {sorted(orphans)[0]}'
    )

  names = shards_for_tests(shards, test_files, affected) | table_names
  reason = (
      f'{len(changed)} changed file(s) reach {len(affected)} test file(s)'
      f' in {len(names)} of {len(shards)} shards'
  )
  return names, f'{reason} ({table_why})' if table_why else reason


def explain(
    shards: ShardMap, test_files: Collection[str], changed_from: str
) -> list[str]:
  """Returns log lines explaining shard selection for changed files.

  Args:
    shards: Resolved shard mapping.
    test_files: Universe of all test files.
    changed_from: File containing NUL-separated changed paths.

  Returns:
    Log lines explaining shard selection in `shard_order`.
  """
  changed = read_changed(changed_from)
  if not changed:
    return ['changed-file list is empty: every shard runs']

  others, table_names, table_why = _table_diff(shards, test_files, changed)
  if table_names is None:
    return [f'every shard runs: {table_why}']

  affected = set()
  if others:
    affected, why = deps.affected_tests(others)
    if affected is None:
      return [f'every shard runs: {why}']

  edited = set(changed)
  tests_edited = {f for f in edited if deps.is_test_file(f)}
  affected = set(affected)
  names = shards_for_tests(shards, test_files, affected) | table_names
  noun = 'a test file' if len(tests_edited) == 1 else 'test files'
  out = [f'{len(changed)} changed file(s), {len(tests_edited)} of them {noun}:']
  out += [f'      {f}' for f in sorted(changed)]
  out += [
      '',
      (
          f'reaching {len(affected)} test file(s) in {len(names)} of'
          f' {len(shards)} shards.'
      ),
  ]
  if table_why:
    out.append(f'  {table_why}; those shards are marked `table`.')
  out += [
      '  `edited` was changed directly, `import` reaches one that was.',
      (
          '  the clock is measured wall time: what shards are sized on, not'
          ' file count.'
      ),
      '',
  ]
  claimed = set()
  for name, spec in sorted(shards.items(), key=shard_order):
    if not spec['paths']:
      continue
    files = collected_by(spec['paths'], test_files)
    claimed |= files
    reached = sorted(files & affected)
    if not reached and name not in table_names:
      continue
    classes = collections.defaultdict(list)
    for path in spec['paths']:
      file, selector = split_node_id(path)
      if selector:
        classes[file].append(selector)
    minutes = spec.get('minutes')
    out.append(
        f'  {name:32s} {f"{minutes}m" if minutes else "-":>4}'
        f'  {len(reached)} of {len(files)} file(s) reached'
    )
    if name in table_names:
      out.append(f'      table   test paths changed in {SHARDS_PATH}')
    for file in reached:
      tag = 'edited' if file in edited else 'import'
      runs = f'  [runs ::{", ::".join(classes[file])}]' if classes[file] else ''
      out.append(f'      {tag}  {file}{runs}')

  if leftover := sorted(affected - claimed):
    out.append('')
    for file in leftover:
      if _matches_glob(file, IGNORED_GLOBS):
        out.append(f'      {file} -- never collected, selects nothing')
      elif _matches_glob(file, EXCLUDED_TESTS):
        out.append(f'      {file} -- in EXCLUDED_TESTS, selects nothing')
      else:
        out.append(f'      {file} -- NO SHARD, forces a full run')
  return out


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the matrix, check, list, or explain CLI subcommand.

  Args:
    argv: Command-line arguments, or None to use sys.argv[1:].

  Returns:
    Process exit code (0 on success).

  Raises:
    SystemExit: If consistency checks find configuration errors.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('command', choices=('matrix', 'check', 'list', 'explain'))
  parser.add_argument(
      '--changed-from',
      metavar='FILE',
      help=(
          'Path to file containing NUL-separated changed file paths (e.g. from'
          ' `git diff --name-only -z`). Narrows the test matrix to affected'
          ' shards. If omitted, all shards run.'
      ),
  )
  parser.add_argument(
      '--dry-run',
      action='store_true',
      help=(
          'Calculate and log shard selection without filtering the matrix'
          ' (emits all shards). Used to validate selection behavior before'
          ' enforcing it.'
      ),
  )
  args = parser.parse_args(argv)

  test_files = all_test_files()
  shards, catch_all, excluded, ignored = resolve_shards(test_files)

  if errors := check_consistency(shards, test_files, excluded, ignored):
    raise SystemExit(
        'shard configuration is inconsistent:\n  ' + '\n  '.join(errors)
    )

  if args.command == 'check':
    covered = len(test_files) - len(excluded) - len(ignored)
    print(
        f'{len(shards)} shards cover {covered} test files;'
        f' {len(excluded)} excluded by design,'
        f' {len(ignored)} never collected'
    )
    return 0

  if args.command == 'explain':
    if not args.changed_from:
      parser.error('explain needs --changed-from')
    print('\n'.join(explain(shards, test_files, args.changed_from)))
    return 0

  if args.command == 'matrix':
    only, reason = select(shards, test_files, args.changed_from)
    combos = build_matrix(shards, None if args.dry_run else only)
    compat = build_compat_matrix(shards, None if args.dry_run else only)
    flags = shlex.join(PYTEST_SELECT_FLAGS)
    if args.dry_run and only is not None:
      reason = f'{reason} -- DRY RUN, not applied: every shard runs'
    print(f'selection: {reason}', file=sys.stderr)
    print(
        f'{len(shards)} shards, {len(combos)} jobs, {len(compat)} compat jobs',
        file=sys.stderr,
    )
    if only is not None:
      verb = 'would select' if args.dry_run else 'selected'
      print(f'{verb} {len(only)} of {len(shards)} shards:', file=sys.stderr)
      for name in sorted(only):
        print(f'  {name}', file=sys.stderr)
    for index, names in enumerate(
        compat_bins(shards, None if args.dry_run else only), start=1
    ):
      print(f'compat-{index}: {" ".join(names)}', file=sys.stderr)
    print(f'not run by design: {sorted(excluded)}', file=sys.stderr)
    print(f'pytest select flags: {flags}', file=sys.stderr)
    print(f'include={json.dumps(combos)}')
    print(f'compat_include={json.dumps(compat)}')
    print(f'pytest_flags={flags}')
    print(f'catch_all={json.dumps(catch_all)}')
    return 0

  combos = build_matrix(shards)
  compat = build_compat_matrix(shards)
  per_shard = collections.Counter(c['shard_name'] for c in combos)
  print(f'{len(shards)} shards, {len(combos)} jobs, {len(compat)} compat jobs')
  order = sorted(shards.items(), key=shard_order)
  for name, spec in order:
    print(f'  {name:32s} x{per_shard[name]}')
    for path in spec['paths']:
      print(f'      {path}')

  targets = [p for s in shards.values() for p in s['paths']]
  files = {split_node_id(p)[0] for p in targets}
  print()
  print(
      f'  {len(test_files)} test files on disk = {len(files)} run'
      f' + {len(excluded)} excluded + {len(ignored)} never collected'
  )
  print(f'  {len(targets)} targets across {len(shards)} shards')
  for path in sorted(excluded):
    print(f'      excluded         {path}')
  for path in sorted(ignored):
    print(f'      never collected  {path}')

  print()
  print(
      f'  compat bins, each run once per {", ".join(older_jaxs())} on'
      f' {", ".join(RUNNERS[r][1] for r in COMPAT_RUNNERS)}:'
  )
  for index, names in enumerate(compat_bins(shards), start=1):
    load = sum(compat_size(shards[n]) for n in names)
    print(f'  compat-{index:<3d} {load:5.1f}m  {" ".join(names)}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
