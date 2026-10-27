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

"""Benchmarks for Kimi Delta Attention (KDA)."""

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
import jax
import jax.numpy as jnp
import tokamax
from tokamax._src import numerics
from tokamax._src.ops.kda import api as kda_api
from tokamax._src.ops.kda import arg_specs as kda_specs
from tokamax._src.ops.kda import pallas_mosaic_tpu
from tokamax.benchmarks import common

_TENSORBOARD_OUTPUT_ENV_VAR = flags.DEFINE_string(
    "tensorboard_output_env_var",
    "TENSORBOARD_OUTPUT_DIR",
    "Environment variable to use to retrieve TensorBoard output directory.",
)
_SKIP_IMPLEMENTATIONS = flags.DEFINE_list(
    "skip_implementations",
    [],
    "A comma-separated list of implementations to skip.",
)

EXAMPLES = immutabledict.immutabledict(
    {
        spec.name: spec.args
        for spec in kda_specs.ARG_SPECS
        if "primary" in spec.tags
    }
)


# The compact profile consumes pre-normalized packed Q/K. Keep this workload
# distinct so its benchmark cannot silently select the aligned fallback.
EXAMPLES = immutabledict.immutabledict(
    {
        **EXAMPLES,
        "packed_training_b1_t8192_n25_bf16_pre_normalized": {
            **EXAMPLES["packed_training_b1_t8192_n25_bf16"],
            "use_qk_l2norm": False,
        },
    }
)


def _make_example(args_spec_name: str, implementation: str):
  """Initializes one valid KDA benchmark input."""
  example = dict(EXAMPLES[args_spec_name])
  example["implementation"] = implementation
  example = numerics.random_initialize(example)

  # The public KDA API consumes post-activation beta values in [0, 1]. Keep
  # beta deterministic and numerically stable. The remaining KDA parameters
  # retain the randomized representative-workload values.
  example["beta"] = jnp.full_like(example["beta"], 0.5)
  return example


class KdaBenchmark(parameterized.TestCase):
  """Performance benchmarks for the XLA and Mosaic KDA implementations."""

  @parameterized.product(
      implementation=(
          "xla",
          "xla_chunked",
          "mosaic",
          "mosaic_staged",
          "mosaic_fwd_fused",
          "mosaic_packed_inputs",
          "mosaic_packed_outputs",
          "mosaic_compact",
          "mosaic_fused",
          "mosaic_packed_backward",
          "mosaic_packed_gradients",
          "mosaic_remat",
          "mosaic_remat_fused",
      ),
      benchmark_mode=("forward", "forward_and_vjp"),
      args_spec_name=tuple(EXAMPLES.keys()),
  )
  def test_kda(self, implementation, benchmark_mode, args_spec_name):
    """Benchmarks one KDA implementation and training workload."""
    logging.info("device_kind=%s", jax.devices()[0].device_kind)

    if implementation in _SKIP_IMPLEMENTATIONS.value:
      self.skipTest(
          f"Skipping implementation '{implementation}' as per"
          " --skip_implementations flag."
      )
    if jax.default_backend() != "tpu" and implementation not in (
        "xla",
        "xla_chunked",
    ):
      self.skipTest("Mosaic TPU implementation is only supported on TPU.")

    example = _make_example(
        args_spec_name,
        implementation
        if implementation in ("xla", "xla_chunked", "mosaic")
        else "mosaic",
    )
    attention = kda_api.kimi_delta_attention
    if implementation == "mosaic_compact" and (
        example.get("use_qk_l2norm", False)
        or example.get("segment_ids") is None
    ):
      self.skipTest("Compact benchmark requires pre-normalized packed inputs.")
    if implementation not in ("xla", "xla_chunked", "mosaic"):
      example.pop("implementation")
      attention = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention(
          config=pallas_mosaic_tpu.Config(
              fuse_forward=implementation != "mosaic_staged",
              packed_forward=implementation
              in (
                  "mosaic_packed_inputs",
                  "mosaic_packed_outputs",
                  "mosaic_packed_backward",
                  "mosaic_packed_gradients",
              ),
              packed_output=implementation
              in (
                  "mosaic_packed_outputs",
                  "mosaic_packed_backward",
                  "mosaic_packed_gradients",
              ),
              fuse_backward=implementation
              in (
                  "mosaic_fused",
                  "mosaic_packed_inputs",
                  "mosaic_packed_outputs",
                  "mosaic_packed_backward",
                  "mosaic_packed_gradients",
                  "mosaic_remat",
                  "mosaic_remat_fused",
              ),
              packed_backward=implementation
              in ("mosaic_packed_backward", "mosaic_packed_gradients"),
              packed_gradients=implementation == "mosaic_packed_gradients",
              rematerialize_for_backward=implementation.startswith(
                  "mosaic_remat"
              ),
              fuse_rematerialization=implementation == "mosaic_remat_fused",
              compact_forward=implementation == "mosaic_compact",
              compact_sorted_segments=implementation == "mosaic_compact",
          )
      )
    fn, args = tokamax.standardize_function(
        attention,
        kwargs=example,
        mode=benchmark_mode,
    )
    fn = jax.jit(fn)

    result = tokamax.benchmark(fn, args)
    logging.info(
        "median_time_ms=%s",
        result.median_evaluation_time_ms,
    )

    common.write_tensorboard_logs(
        tensorboard_output=_TENSORBOARD_OUTPUT_ENV_VAR.value,
        value=result.evaluation_times_ms,
        metric_tag=(f"kda/{args_spec_name}/{implementation}/{benchmark_mode}"),
    )


if __name__ == "__main__":
  absltest.main()
