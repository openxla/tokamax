# Copyright 2026 Google LLC
# Copyright 2026 Rabdos AI
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
"""Tests for the fused causal Conv1D GDN Op."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src import benchmarking
from tokamax._src.autotuning import autotuner
from tokamax._src.autotuning import cache
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.causal_conv1d_gated_delta_rule import test_base
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import fused_conv1d_gdn as fused_op

_OP = fused_op.FusedCausalConv1dGatedDeltaRule


class GDNConfigTest(parameterized.TestCase):
  """Config serialization through Tokamax's cache and HLO metadata adapters."""

  def _bound_args(
      self, *, compute_precision=jnp.float32
  ) -> op_lib.BoundArguments:
    """Bind shape-only inputs with the requested compute dtype."""
    shape = jax.ShapeDtypeStruct
    config = fused_op.Config(
        compute_precision=compute_precision,
        decode_tile_size=2,
        mixed_tile_size=128,
    )
    tokens, requests, slots = 1, 1, 2
    n_kq, n_v = 2, 8
    d_k, d_v = 128, 128
    kernel_size = 4
    qkv_width = 2 * n_kq * d_k + n_v * d_v
    return _OP(config=config).bind(
        qkv=shape((tokens, qkv_width), jnp.bfloat16),
        b=shape((tokens, n_v), jnp.bfloat16),
        a=shape((tokens, n_v), jnp.bfloat16),
        conv_state=shape((slots, kernel_size - 1, qkv_width), jnp.bfloat16),
        recurrent_state=shape((slots, n_v, d_k, d_v), jnp.float32),
        conv_weight=shape((qkv_width, 1, kernel_size), jnp.float32),
        conv_bias=shape((qkv_width,), jnp.float32),
        a_log=shape((n_v,), jnp.float32),
        dt_bias=shape((n_v,), jnp.float32),
        query_start_loc=shape((requests + 1,), jnp.int32),
        state_indices=shape((requests,), jnp.int32),
        distribution=shape((3,), jnp.int32),
        seq_lens=shape((requests,), jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

  @parameterized.named_parameters(
      ("dtype_class", jnp.float32),
      ("dtype_instance", jnp.dtype(jnp.float32)),
      ("dtype_string", "float32"),
  )
  def test_autotuning_cache_roundtrip(self, compute_precision):
    """Round-trip cache entries for class, dtype, and string FP32 forms."""
    bound = self._bound_args(compute_precision=compute_precision)
    config = bound.op.config
    benchmark = benchmarking.BenchmarkData(
        compile_time_ms=1.0,
        lower_time_ms=1.0,
        evaluation_times_ms=(0.1, 0.2, 0.3),
        metadata={},
    )
    data = {
        bound.autotuning_cache_key: autotuner.AutotuningData(
            {config: benchmark}
        )
    }
    # This adapter builds Config directly, unlike bound-argument serialization.
    adapter = cache._get_cache_adapter(bound.op)  # pylint: disable=protected-access
    self.assertEmpty(adapter.validate_json("{}"))
    restored = adapter.validate_json(adapter.dump_json(data, round_trip=True))
    self.assertEqual(restored, data)
    self.assertEqual(
        restored[bound.autotuning_cache_key].fastest_config, config
    )

  def test_bound_arguments_roundtrip(self):
    """Check bound operation arguments survive serialization."""
    bound = self._bound_args()
    adapter = op_lib.BOUND_ARGS_ADAPTER
    restored = adapter.validate_json(adapter.dump_json(bound, round_trip=True))
    self.assertEqual(restored.arguments, bound.arguments)
    self.assertEqual(restored.op.config, bound.op.config)

  def test_symbolic_shapes_are_rejected_before_device_resolution(self):
    """Check symbolic shapes fail before device resolution is attempted."""
    bound = self._bound_args()
    (tokens,) = jax.export.symbolic_shape("tokens")
    args = dict(bound.arguments)
    qkv = args["qkv"]
    args["qkv"] = jax.ShapeDtypeStruct((tokens, qkv.shape[1]), qkv.dtype)
    with self.assertRaisesRegex(NotImplementedError, "symbolic shapes"):
      bound.op(**args)


# The shared suite exercises float32 activations and float32 cache state.
class GDNAttentionTest(test_base.CausalConv1dGatedDeltaRuleTestBase):
  """Binds the fused operation to upstream's shared attention correctness suite."""

  def __init__(self, *args):
    """Pass the fused operation to the inherited correctness tests."""
    super().__init__(
        *args,
        gdn_fn=_OP(),
        static_argnames=test_base.OP_STATIC_ARGNAMES,
    )


class GDNSecurityTest(test_base.CausalConv1dGatedDeltaRuleSecurityTestBase):
  """Binds the fused operation to upstream's shared security/contract suite."""

  def __init__(self, *args):
    """Supply the fused operation to the inherited security-test constructor."""
    super().__init__(*args, gdn_fn=_OP())


if __name__ == "__main__":
  absltest.main()
