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
"""Tests for the Pallas/Mosaic SparseMLA operator on TPU."""

from absl.testing import absltest
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.ops.sparse_mla import arg_specs
from tokamax._src.ops.sparse_mla import base
from tokamax._src.ops.sparse_mla import pallas_mosaic_tpu
from tokamax._src.ops.sparse_mla import pallas_mosaic_tpu_tc_kernel
from tokamax._src.ops.sparse_mla import test_base


class PallasTpuSparseMlaTest(test_base.SparseMlaTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, sparse_mla_fn=pallas_mosaic_tpu.PallasTpuSparseMla()
    )

  def setUp(self):
    super().setUp()
    if not self._sparse_mla_fn.supported_on(backend.get_default_device()):
      self.skipTest("PallasTpuSparseMla is not supported on this device.")

  def test_autotuning_configs(self):
    """Checks every autotuning config runs and matches the reference."""
    inputs, actual_tokens = test_base.make_inputs(
        batch_size=4, topk=512, pad_tokens=True
    )
    op = pallas_mosaic_tpu.PallasTpuSparseMla()
    ba = op.bind(*inputs, sm_scale=1.0)
    configs = ba.autotuning_configs
    self.assertIn(ba.default_config, configs)
    self.assertGreater(len(configs), 1)

    for config in configs:
      with self.subTest(str(config)):
        actual = op.replace(config=config)(*inputs, sm_scale=1.0)
        test_base.assert_matches_reference(actual, inputs, actual_tokens)

  def test_tc_gather_configs(self):
    """Checks TensorCore gather path matches the reference."""
    tpu_info = pltpu.get_tpu_info()
    if tpu_info.generation >= 8 or tpu_info.sparse_core is None:
      self.skipTest("Default path already uses TensorCore gather.")
    inputs, actual_tokens = test_base.make_inputs(
        batch_size=4, topk=512, pad_tokens=True
    )
    op = pallas_mosaic_tpu.PallasTpuSparseMla()
    ba = op.bind(*inputs, sm_scale=1.0)
    for config in ba.autotuning_configs:
      with self.subTest(str(config)):
        actual = pallas_mosaic_tpu_tc_kernel.sparse_ragged_paged_attention(
            *inputs,
            sm_scale=1.0,
            gather_and_attention_chunk_size=(
                config.gather_and_attention_chunk_size
            ),
            attention_kernel_batch_size=config.attention_kernel_batch_size,
            vmem_limit_bytes=config.vmem_limit_bytes,
        )
        test_base.assert_matches_reference(actual, inputs, actual_tokens)

  def test_arg_spec_numerics(self):
    """Checks a realistic ArgSpec initializes cleanly with non-trivial diff."""
    spec = next(
        s
        for s in arg_specs.ARG_SPECS
        if s.name == "prefill_b1_q1024_kv1024_p256_topk2048_h64"
    )
    args = {
        k: jax.device_put(np.asarray(v)) if isinstance(v, np.ndarray) else v
        for k, v in numerics.random_initialize(spec.args, seed=0).items()
    }
    ref = base.SparseMla()(**args)
    actual_op = pallas_mosaic_tpu.PallasTpuSparseMla()(**args)
    self.assertTrue(bool(jnp.all(jnp.isfinite(ref))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(actual_op))))
    op_summary = numerics.array_diff_summary(
        ref, actual_op, rtol=1.6e-2, atol=1e-2
    )
    self.assertGreater(op_summary.percent_close, 0.99)
    tc_out = pallas_mosaic_tpu_tc_kernel.sparse_ragged_paged_attention(
        args["q"],
        args["cache_kv_nope"],
        args["cache_kv_rope"],
        args["topk_indices"],
        args["page_indices"],
        args["cu_q_lens"],
        args["distribution"],
        args["attention_sinks"],
        args["swa_accumution"],
        args["swa_l"],
        args["swa_m"],
        sm_scale=args["sm_scale"],
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(tc_out))))
    tc_summary = numerics.array_diff_summary(
        ref, tc_out, rtol=1.6e-2, atol=1e-2
    )
    self.assertGreater(tc_summary.percent_close, 0.99)
    # Perturbing cached KV rows must meaningfully change the output so Perfmon
    # numerics stay sensitive to gather-index, byte-order, or lane-mapping bugs.
    bad_args = dict(args)
    bad_args["cache_kv_nope"] = jnp.roll(args["cache_kv_nope"], 1, axis=1)
    bad_ref = base.SparseMla()(**bad_args)
    bad_summary = numerics.array_diff_summary(
        ref, bad_ref, rtol=1.6e-2, atol=1e-2
    )
    self.assertLess(bad_summary.percent_close, 0.5)


if __name__ == "__main__":
  absltest.main()
