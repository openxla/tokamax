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
"""End-to-end tests of the monolithic layer: CPU interpret mode or TPU v8i.

On CPU (default) the layer (dedup_ep_moe with wave dispatch) runs on eight CPU
devices: every Pallas kernel (router top-k, fused EP MoE kernel, combine,
token-gather assembly) is interpreted with simulated TPU memory, DMAs and
semaphores (interpret_util.interpret_pallas), against the TPU v8i chip record.
The split-phase exchange is emulated (interpret_util.emulate_wave_exchange).
On TPU v8i devices the same cases run on four devices with the real kernels
(hw_util).

  * the direct wave dispatch with W=4 (the bit-exactness gate) with bf16 and
    fp8 (W8A8) weights, against a dense f32 reference
    (reference.moe_reference);
  * other wave counts: bit equality with the W=4 gate, plus the same
    f32-reference check;
  * one case at t_local = 256, where the combine runs as its Mosaic kernel
    (output_reduce.combine_step_tokens) instead of the XLA epilogue;
  * the native split-phase path. TPU interpret mode cannot run it, so on CPU
    this is a canary that fails if interpret mode starts accepting it; on
    hardware it is a real check that the native path is bit-exact with the
    W=4 gate (and close to the f32 reference).
"""

import functools
import time
import types
from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.common.arg import WeightFormat
from tokamax._src.ops.experimental.dedup_moe.io_relayout import output_reduce
from tokamax._src.ops.experimental.dedup_moe.layer.monolithic import dedup_ep_moe
from tokamax._src.ops.experimental.dedup_moe.test.common import hw_util
from tokamax._src.ops.experimental.dedup_moe.test.common import interpret_util
from tokamax._src.ops.experimental.dedup_moe.test.common import moe_inputs
from tokamax._src.ops.experimental.dedup_moe.test.common import reference

# 8 CPU devices in interpret mode; on hardware 4 TPU v8i devices (one host).
CPU_DEVICES = 8
EXPERTS_PER_SHARD = 4
FORMATS = ("bf16", "fp8")
TOPK = 4
ROWS_PER_SHARD = 16
# The combine-kernel case: t_local >= output_reduce.COMBINE_MIN_TOKENS, taller
# tiles to keep the interpreted kernel's tile count down.
COMBINE_ROWS_PER_SHARD = 256


@functools.cache
def _shapes() -> types.SimpleNamespace:
  """The layer shapes (hw_util.pick needs JAX, so not at import time)."""
  ep = hw_util.pick(8, 4)
  return types.SimpleNamespace(
      ep=ep,
      e_total=ep * EXPERTS_PER_SHARD,
      tokens=ep * ROWS_PER_SHARD,
      hidden=hw_util.pick(256, hw_util.HW_HIDDEN),
      inter=hw_util.pick(256, hw_util.HW_INTER),
      capacity=hw_util.pick(16, hw_util.HW_CAPACITY),
      combine_capacity=hw_util.pick(64, hw_util.HW_CAPACITY),
  )


# The bit-exactness gate: the direct wave dispatch with this many waves.
GATE_WAVES = 4
# rel-L2 bound against the f32 reference. bf16 rounds the intermediate and
# the output; fp8 also quantizes the weights per channel, the rows, the
# intermediate and the wire.
REL_L2_BOUND = {"bf16": 2e-2, "fp8": 1e-1}


def setUpModule():
  hw_util.setup_module("tpu8i")


class LayerInterpretTest(parameterized.TestCase):
  """The layer against the f32 reference and the direct W=4 gate."""

  # The mesh, weights and inputs, built by the first test that runs.
  _cache: dict[str, Any] = {}
  # (fmt, routing) -> direct W=4 output, shared across cases that run in the
  # same test shard.
  _gates: dict[tuple[str, str], np.ndarray] = {}

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)
    if "mesh" not in self._cache:
      mesh = hw_util.make_mesh(_shapes().ep, self)
      self._cache["mesh"] = mesh
      self._cache["weights"] = {
          fmt: moe_inputs.random_weights(
              mesh,
              e_total=_shapes().e_total,
              hidden=_shapes().hidden,
              inter=_shapes().inter,
              weight_format=fmt,
              seed=0,
          )
          for fmt in FORMATS
      }
      self._cache["inputs"] = moe_inputs.random_inputs(
          mesh,
          tokens=_shapes().tokens,
          hidden=_shapes().hidden,
          e_total=_shapes().e_total,
          seed=1,
      )
    self.mesh = self._cache["mesh"]
    self.weights = self._cache["weights"]
    self.inputs = self._cache["inputs"]
    hw_util.reset_interpret_state()

  def _hash_ids(self, routing, tokens=None):
    if tokens is None:
      tokens = _shapes().tokens
    if routing == "router":
      return None
    return moe_inputs.expert_ids(
        routing, tokens, _shapes().e_total, TOPK, EXPERTS_PER_SHARD, seed=1
    )

  def _layer(
      self,
      fmt,
      *,
      waves,
      routing="router",
      emulate=True,
      x=None,
      gating=None,
      capacity=None,
  ):
    if capacity is None:
      capacity = _shapes().capacity
    x = self.inputs[0] if x is None else x
    gating = self.inputs[1] if gating is None else gating
    hash_ids = self._hash_ids(routing, x.shape[0])
    if hash_ids is not None:
      hash_ids = moe_inputs.shard_rows(self.mesh, hash_ids)
    mesh = self.mesh

    def fn(x, w1, w2, w1_scale, w2_scale, gating, hash_ids):
      return dedup_ep_moe(
          x,
          w1,
          w2,
          w1_scale,
          w2_scale,
          gating,
          topk=TOPK,
          renormalize=True,
          mesh=mesh,
          capacity=capacity,
          weight_format=WeightFormat(fmt),
          hash_topk_ids=hash_ids,
          waves=waves,
      )

    start = time.perf_counter()
    with hw_util.cpu_layer(emulate=emulate):
      out = hw_util.run(fn, x, *self.weights[fmt], gating, hash_ids)
      out = np.asarray(out, np.float32)
    logging.info(
        "layer [%s] %s routing=%s T=%d waves=%d emulate=%s: %.1f s",
        hw_util.mode_name(),
        fmt,
        routing,
        x.shape[0],
        waves,
        emulate,
        time.perf_counter() - start,
    )
    self.assertEqual(out.shape, x.shape)
    self.assertTrue(np.isfinite(out).all())
    return out

  def _gate(self, fmt, routing):
    """The direct W=4 output, once per (fmt, routing) per test shard."""
    key = (fmt, routing)
    if key not in self._gates:
      self._gates[key] = self._layer(fmt, routing=routing, waves=GATE_WAVES)
    return self._gates[key]

  def _assert_close_to_reference(
      self, got, fmt, routing="router", x=None, gating=None
  ):
    x = self.inputs[0] if x is None else x
    gating = self.inputs[1] if gating is None else gating
    with hw_util.host_only():
      # Host copies as jax arrays (on hardware the layer's are on the TPUs).
      w1, w2, w1_scale, w2_scale = (
          None if a is None else jnp.asarray(jax.device_get(a))
          for a in self.weights[fmt]
      )
      want = np.asarray(
          reference.moe_reference(
              jnp.asarray(jax.device_get(x)),
              w1,
              w2,
              w1_scale,
              w2_scale,
              jnp.asarray(jax.device_get(gating)),
              topk=TOPK,
              hash_topk_ids=self._hash_ids(routing, x.shape[0]),
          )
      )
    err = reference.rel_l2(got, want)
    logging.info(
        "%s routing=%s T=%d: rel-L2 %.4g, max-abs-rel %.4g vs f32 reference",
        fmt,
        routing,
        x.shape[0],
        err,
        reference.max_abs_rel(got, want),
    )
    self.assertLess(err, REL_L2_BOUND[fmt])

  @parameterized.parameters(*FORMATS)
  def test_gate_matches_reference(self, fmt):
    self._assert_close_to_reference(self._gate(fmt, "router"), fmt)

  @parameterized.named_parameters(
      dict(testcase_name="bf16_w2", fmt="bf16", waves=2),
      dict(testcase_name="fp8_w1", fmt="fp8", waves=1),
      dict(testcase_name="fp8_w2", fmt="fp8", waves=2),
      dict(testcase_name="fp8_w2_skewed", fmt="fp8", waves=2, routing="skewed"),
      dict(
          testcase_name="fp8_w1_dominant",
          fmt="fp8",
          waves=1,
          routing="dominant",
      ),
  )
  def test_waves_are_bit_exact_with_the_w4_gate(
      self, fmt, waves, routing="router"
  ):
    want = self._gate(fmt, routing)
    got = self._layer(fmt, routing=routing, waves=waves)
    np.testing.assert_array_equal(got, want)
    self._assert_close_to_reference(got, fmt, routing)

  def test_combine_kernel_path_is_bit_exact_and_close(self):
    """t_local >= COMBINE_MIN_TOKENS: the Mosaic combine kernel runs.

    Every other case has t_local = ROWS_PER_SHARD (16), where the combine is
    the XLA epilogue; from t_local = 256 up the layer takes the kernel instead.
    """
    t_local = COMBINE_ROWS_PER_SHARD
    self.assertGreater(output_reduce.combine_step_tokens(t_local), 0)
    x, gating = moe_inputs.random_inputs(
        self.mesh,
        tokens=_shapes().ep * t_local,
        hidden=_shapes().hidden,
        e_total=_shapes().e_total,
        seed=2,
    )
    want = self._layer(
        "fp8",
        waves=GATE_WAVES,
        x=x,
        gating=gating,
        capacity=_shapes().combine_capacity,
    )
    got = self._layer(
        "fp8", waves=2, x=x, gating=gating, capacity=_shapes().combine_capacity
    )
    np.testing.assert_array_equal(got, want)
    self._assert_close_to_reference(got, "fp8", x=x, gating=gating)

  def test_native_split_phase_is_tpu_only(self):
    """CPU canary / hardware check of the native split-phase exchange.

    On CPU: interpret mode cannot hand DMA semaphores between kernels. If
    this starts passing through (interpret mode gained semaphore outputs),
    run the wave cases natively and drop emulate_wave_exchange.

    On hardware the split phase is real: the native W=4 and W=2 runs are
    bit-exact with the W=4 gate (run separately) and close to the f32
    reference.
    """
    if hw_util.on_hardware():
      want = self._gate("fp8", "router")
      for waves in (GATE_WAVES, 2):
        got = self._layer("fp8", emulate=False, waves=waves)
        np.testing.assert_array_equal(got, want, err_msg=f"waves={waves}")
        self._assert_close_to_reference(got, "fp8")
      return
    try:
      with self.assertRaises(Exception) as caught:  # pylint: disable=g-error-prone-assert-raises
        self._layer("fp8", emulate=False, waves=GATE_WAVES)
      logging.info(
          "native split-phase wave dispatch in interpret mode: %s: %s",
          type(caught.exception).__name__,
          (str(caught.exception).splitlines() or [""])[0][:500],
      )
    finally:
      interpret_util.reset_interpret_state()
      interpret_util.clear_layer_cache()


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", CPU_DEVICES)
  absltest.main()
