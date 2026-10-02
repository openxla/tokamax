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
"""Fused GDN dispatch, numerical parity, and cache-ownership tests.

TPU tests use the upstream wrapper as oracle and check cache preservation.
"""

from collections.abc import Sequence
import dataclasses
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.causal_conv1d_gated_delta_rule import test_base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import metadata as fused_metadata
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import prefill_decode
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import tiling as fused_tiling

_D_K = 128
_D_V = 128

# Cache tolerances; output comparisons scale each row separately.
_RTOL = 2e-2
_CACHE_ATOL = 2e-2
_OUTPUT_ATOL = 2e-2  # After scaling each row to its reference peak.

# Positional arguments in fused_conv1d_gdn signature order.
_ARG_ORDER = (
    "qkv",
    "b",
    "a",
    "conv_state",
    "recurrent_state",
    "conv_weight",
    "conv_bias",
    "a_log",
    "dt_bias",
    "query_start_loc",
    "state_indices",
    "distribution",
    "seq_lens",
)


@dataclasses.dataclass(frozen=True)
class _Request:
  """One request: tokens submitted now, history consumed earlier, and cache
  slot.

  Zero history ignores cached contents; decode requests submit one token.
  """

  tokens: int
  history: int
  slot: int


def _decode(slot: int, *, history: int = 8) -> _Request:
  """Build a one-token request for a nonzero slot.

  history counts tokens already consumed.
  """
  return _Request(tokens=1, history=history, slot=slot)


def _prefill(slot: int, tokens: int, *, history: int = 0) -> _Request:
  """Build a prefill request for a nonzero slot.

  history counts tokens already consumed.
  """
  return _Request(tokens=tokens, history=history, slot=slot)


@dataclasses.dataclass(frozen=True)
class _Case:
  """Call inputs and expected active token/cache ownership.

  values holds the 13 positional arguments by name; conv_bias may be None.
  """

  values: dict[str, np.ndarray | None]
  n_kq: int
  n_v: int
  kernel_size: int
  slots: int
  active_tokens: int
  active_slots: tuple[int, ...]

  @property
  def options(self) -> dict[str, Any]:
    """Return the fixture's static head dimensions and convolution options."""
    return dict(
        n_kq=self.n_kq,
        n_v=self.n_v,
        d_k=_D_K,
        d_v=_D_V,
        kernel_size=self.kernel_size,
    )


@dataclasses.dataclass(frozen=True)
class _Result:
  """Holds host copies of convolution state, recurrent state, and output."""

  conv: np.ndarray
  recurrent: np.ndarray
  out: np.ndarray


def _build_case(
    active: Sequence[_Request],
    *,
    num_decode: int,
    n_kq: int,
    n_v: int,
    kernel_size: int = 4,
    padding_slots: Sequence[int] = (),
    slots: int | None = None,
    conv_bias: bool = True,
    conv_bias_scale: float = 0.01,
    conv_state_dtype: Any = jnp.bfloat16,
    recurrent_state_dtype: Any = jnp.float32,
    activation_dtype: Any = jnp.bfloat16,
    seed: int = 0,
) -> _Case:
  """Build a valid schedule with decodes preceding prefills.

  Options that preserve shapes also preserve RNG draw order, allowing
  same-seed comparisons. Padding slots may repeat or be out of range. The
  default bias is small; tests detecting ignored bias must raise its
  magnitude above the comparison tolerance.

  Args:
    active: Active request fixtures, with decodes before prefills.
    num_decode: Number of active one-token requests at the start of the batch.
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.
    kernel_size: Positive static convolution window size.
    padding_slots: Inactive request slot IDs; they may repeat or be out of
      range.
    slots: Cache-pool size, or None for request entries plus reserved slot zero.
    conv_bias: Whether the fixture includes convolution bias.
    conv_bias_scale: Magnitude of the seeded convolution-bias fixture.
    conv_state_dtype: Dtype of the convolution-cache fixture.
    recurrent_state_dtype: Dtype of the recurrent-cache fixture.
    activation_dtype: Dtype of QKV and gate inputs.
    seed: Seed for deterministic NumPy fixture generation.

  Returns:
    Input fixture with expected active token count and cache ownership.
  """
  if any(request.tokens != 1 for request in active[:num_decode]):
    raise ValueError("decode requests submit exactly one token")
  if any(request.tokens < 1 for request in active):
    raise ValueError("active requests submit at least one token")
  width = 2 * n_kq * _D_K + n_v * _D_V
  lengths = [request.tokens for request in active] + [0] * len(padding_slots)
  entries = len(lengths)
  tokens = sum(lengths)
  if slots is None:
    slots = entries + 1
  rng = np.random.default_rng(seed)

  def uniform(
      shape: tuple[int, ...], scale: float, dtype: Any = np.float32
  ) -> np.ndarray:
    """Draw a seeded NumPy array in [-scale, scale] with the requested dtype."""
    return rng.uniform(-scale, scale, shape).astype(np.float32).astype(dtype)

  # Draw caches and weights first so changing token count preserves them.
  bias = uniform((width,), conv_bias_scale)
  values = {
      "conv_state": uniform(
          (slots, kernel_size - 1, width), 0.08, conv_state_dtype
      ),
      "recurrent_state": uniform(
          (slots, n_v, _D_K, _D_V), 0.04, recurrent_state_dtype
      ),
      "conv_weight": uniform((width, 1, kernel_size), 0.04),
      "conv_bias": bias if conv_bias else None,
      # Negative log weights keep the decay gate from saturating.
      "a_log": uniform((n_v,), 1.5) - np.float32(2.0),
      "dt_bias": uniform((n_v,), 0.2),
      "qkv": uniform((tokens, width), 0.2, activation_dtype),
      "b": uniform((tokens, n_v), 0.25, activation_dtype),
      "a": uniform((tokens, n_v), 0.25, activation_dtype),
      "query_start_loc": np.array([0] + list(np.cumsum(lengths)), np.int32),
      "state_indices": np.array(
          [request.slot for request in active] + list(padding_slots), np.int32
      ),
      "distribution": np.array(
          [num_decode, len(active), len(active)], np.int32
      ),
      "seq_lens": np.array(
          [request.tokens + request.history for request in active]
          + [0] * len(padding_slots),
          np.int32,
      ),
  }
  return _Case(
      values=values,
      n_kq=n_kq,
      n_v=n_v,
      kernel_size=kernel_size,
      slots=slots,
      active_tokens=tokens,
      active_slots=tuple(request.slot for request in active),
  )


def _replace(case: _Case, **edits: np.ndarray | None) -> _Case:
  """Copy a case with input replacements, retaining its other fields."""
  values = dict(case.values)
  values.update(edits)
  return dataclasses.replace(case, values=values)


def _pad_token_bucket(case: _Case, rows: int, *, seed: int = 1) -> _Case:
  """Append seeded noise without changing submitted rows or request offsets.

  Args:
    case: Input fixture and its expected active token/cache ownership.
    rows: Total padded bucket rows; must cover all existing input rows.
    seed: Seed for deterministic NumPy fixture generation.

  Returns:
    New case with padded activation/gate arrays and unchanged active metadata.
  """
  extra = rows - case.values["qkv"].shape[0]
  if extra < 0:
    raise ValueError("the bucket must hold every submitted token")
  rng = np.random.default_rng(seed)

  def extend(name: str, scale: float) -> np.ndarray:
    """Append uniform noise rows in [-scale, scale], preserving input dtype."""
    array = case.values[name]
    tail = rng.uniform(-scale, scale, (extra,) + array.shape[1:])
    return np.concatenate([array, tail.astype(np.float32).astype(array.dtype)])

  return _replace(
      case,
      qkv=extend("qkv", 0.2),
      b=extend("b", 0.25),
      a=extend("a", 0.25),
  )


def _swap_rows(array: np.ndarray, first: int, second: int) -> np.ndarray:
  """Return a copy of array with rows first and second exchanged."""
  swapped = array.copy()
  swapped[[first, second]] = swapped[[second, first]]
  return swapped


def _swap_slots(case: _Case, first: int, second: int) -> _Case:
  """Exchange two nonzero cache slots and their request IDs; slot zero is
  reserved.

  Args:
    case: Input fixture and its expected active token/cache ownership.
    first: Index of the first row or cache slot to exchange.
    second: Index of the second row or cache slot to exchange.

  Returns:
    New case with cache contents, slot IDs, and expected ownership consistently
    relabeled.
  """
  indices = case.values["state_indices"]
  relabeled = np.where(
      indices == first,
      second,
      np.where(indices == second, first, indices),
  ).astype(np.int32)
  swapped = _replace(
      case,
      conv_state=_swap_rows(case.values["conv_state"], first, second),
      recurrent_state=_swap_rows(case.values["recurrent_state"], first, second),
      state_indices=relabeled,
  )
  return dataclasses.replace(
      swapped,
      active_slots=tuple(
          second if slot == first else first if slot == second else slot
          for slot in case.active_slots
      ),
  )


def _single_request_case(case: _Case, index: int) -> _Case:
  """Isolate one request with inactive padding, retaining its inputs and cache
  pool.

  Args:
    case: Input fixture and its expected active token/cache ownership.
    index: Zero-based element or request index.

  Returns:
    Case containing only the selected request, plus inactive padding and the
    original cache pool.
  """
  starts = case.values["query_start_loc"]
  begin, end = int(starts[index]), int(starts[index + 1])
  length = end - begin
  decode = index < int(case.values["distribution"][0])
  slot = int(case.values["state_indices"][index])
  carved = _replace(
      case,
      qkv=case.values["qkv"][begin:end],
      b=case.values["b"][begin:end],
      a=case.values["a"][begin:end],
      query_start_loc=np.array([0, length, length], np.int32),
      state_indices=np.array([slot, 0], np.int32),
      distribution=np.array([1 if decode else 0, 1, 1], np.int32),
      seq_lens=np.array([int(case.values["seq_lens"][index]), 0], np.int32),
  )
  return dataclasses.replace(carved, active_tokens=length, active_slots=(slot,))


def _device_args(case: _Case) -> tuple[Any, ...]:
  """Materialize _ARG_ORDER inputs with fresh caches for donation.

  Preserve an absent convolution bias as None.
  """
  return tuple(
      None if case.values[name] is None else jnp.asarray(case.values[name])
      for name in _ARG_ORDER
  )


def _run(impl: Any, case: _Case, **options: Any) -> _Result:
  """Run an implementation with case/options and return host copies of its
  results.

  Args:
    impl: Implementation callable accepting the public fused-entry-point
      signature.
    case: Input fixture and its expected active token/cache ownership.
    **options: Entry-point keyword overrides, including optional dynamic read
      addresses.

  Returns:
    Host copies of updated convolution/recurrent caches and output.
  """
  merged = dict(case.options)
  merged.update(options)
  (conv, recurrent), out = impl(*_device_args(case), **merged)
  return _Result(
      conv=np.asarray(conv),
      recurrent=np.asarray(recurrent),
      out=np.asarray(out),
  )


def _static_eligibility(case: _Case, **options: Any) -> bool:
  """Call static eligibility using a fixture and default options.

  Args:
    case: Input fixture and its expected active token/cache ownership.
    **options: Static configuration overrides forwarded to the selected entry
      point.

  Returns:
    Whether this exact static configuration is eligible for fused execution.
  """
  merged = dict(
      case.options,
      zero_initialize_out=True,
      compute_precision=jnp.float32,
      decode_tile_size=None,
      mixed_tile_size=None,
  )
  merged.update(options)
  args = tuple(case.values[name] for name in _ARG_ORDER)
  # pylint: disable-next=protected-access
  return fused_tiling.static_eligible(args, **merged)


def _selected_tiles(case: _Case, **options: Any):
  """Call tile selection using a fixture and optional hints.

  Args:
    case: Input fixture and its expected active token/cache ownership.
    **options: Static configuration overrides forwarded to the selected entry
      point.

  Returns:
    (eligible, decode_tile_size, mixed_tile_size) from the tile planner.
  """
  merged = dict(
      case.options,
      zero_initialize_out=True,
      compute_precision=jnp.float32,
      decode_tile_size=None,
      mixed_tile_size=None,
  )
  merged.update(options)
  hint = merged.pop("mixed_tile_size")
  args = tuple(case.values[name] for name in _ARG_ORDER)
  return fused_tiling.select_tiles(args, hint, **merged)


class GDNStaticDispatchTest(parameterized.TestCase):
  """Trace-time dispatch. These tests need no accelerator."""

  def _case(self, **kwargs: Any) -> _Case:
    """Build the small mixed fixture used by static dispatch tests."""
    return _build_case(
        [_decode(1), _prefill(2, 6)],
        num_decode=1,
        n_kq=2,
        n_v=8,
        **kwargs,
    )

  def test_reference_configuration_is_eligible(self):
    """Check the reference layout is eligible."""
    self.assertTrue(_static_eligibility(self._case()))

  @parameterized.named_parameters(
      ("negative", 2, -1),
      ("negative_past_smem", 2, -4098),
      ("int32_min", 2, -(2**31)),
      ("empty", 2, 0),
      ("full", 2, 2),
      ("upstream_single_prefill", 1, 3),
      ("int32_max", 2, 2**31 - 1),
  )
  def test_active_endpoint_is_safe_for_smem_indices(self, requests, endpoint):
    """Check extreme active endpoints clamp to safe scalar-memory request
    indices.

    Args:
      requests: Number of request entries, including inactive padding.
      endpoint: Raw active-request endpoint to clamp to the request count.
    """
    distribution = jnp.array([0, 0, endpoint], dtype=jnp.int32)
    active = int(fused_metadata.active_request_count(distribution, requests))
    self.assertGreaterEqual(active, 0)
    self.assertLessEqual(active, requests)
    if endpoint <= 0:
      self.assertEqual(active, 0)
    elif endpoint >= requests:
      self.assertEqual(active, requests)

  @parameterized.named_parameters(
      ("float32_conv_state", dict(conv_state_dtype=jnp.float32)),
      ("bfloat16_recurrent_state", dict(recurrent_state_dtype=jnp.bfloat16)),
      ("float32_activations", dict(activation_dtype=jnp.float32)),
      ("without_conv_bias", dict(conv_bias=False)),
      ("kernel_size_one", dict(kernel_size=1)),
      ("kernel_size_three", dict(kernel_size=3)),
  )
  def test_input_variants_are_eligible(self, kwargs):
    """Check supported dtypes, bias choices, and convolution windows remain
    eligible.

    Args:
      kwargs: Keyword overrides used to construct the parameterized input
        fixture.
    """
    self.assertTrue(_static_eligibility(self._case(**kwargs)))

  @parameterized.named_parameters(
      ("explicit_tile_64", dict(mixed_tile_size=64)),
      ("explicit_tile_128", dict(mixed_tile_size=128)),
      ("explicit_tile_256", dict(mixed_tile_size=256)),
      ("explicit_tile_512", dict(mixed_tile_size=512)),
      ("decode_group_2", dict(decode_tile_size=2)),
      ("decode_group_8", dict(decode_tile_size=8)),
      ("decode_tile_1", dict(decode_tile_size=1)),
      ("zero_initialize_out_off", dict(zero_initialize_out=False)),
  )
  def test_option_variants_stay_eligible(self, options):
    """Check valid tile hints and output-initialization options remain eligible.

    Args:
      options: Static option overrides for the selected test case.
    """
    self.assertTrue(_static_eligibility(self._case(), **options))

  @parameterized.named_parameters(
      ("head_dim_64", dict(d_k=64)),
      ("value_dim_64", dict(d_v=64)),
      ("zero_kq_heads", dict(n_kq=0)),
      ("v_heads_not_grouped", dict(n_kq=3)),
      ("bfloat16_compute", dict(compute_precision=jnp.bfloat16)),
      ("unaligned_tile", dict(mixed_tile_size=32)),
  )
  def test_options_outside_the_envelope_are_ineligible(self, options):
    """Check unsupported dimensions, head grouping, compute precision, and tiles
    are ineligible.

    Args:
      options: Static option overrides for the selected test case.
    """
    self.assertFalse(_static_eligibility(self._case(), **options))

  @parameterized.named_parameters(
      ("float32_qkv_only", "qkv", lambda x: x.astype(np.float32)),
      ("float32_gate_b", "b", lambda x: x.astype(np.float32)),
      ("float16_conv_state", "conv_state", lambda x: x.astype(np.float16)),
      ("float32_schedule", "state_indices", lambda x: x.astype(np.float32)),
      ("int64_offsets", "query_start_loc", lambda x: x.astype(np.int64)),
      ("rank_three_qkv", "qkv", lambda x: x[None]),
      ("short_query_start_loc", "query_start_loc", lambda x: x[:-1]),
      ("wrong_gate_width", "b", lambda x: x[:, :-1]),
      ("wrong_weight_shape", "conv_weight", lambda x: x[..., 0]),
      ("wrong_bias_shape", "conv_bias", lambda x: x[:-1]),
      ("wrong_a_log_shape", "a_log", lambda x: x[:-1]),
      ("empty_token_bucket", "qkv", lambda x: x[:0]),
      ("empty_request_array", "state_indices", lambda x: x[:0]),
  )
  def test_inputs_outside_the_envelope_are_ineligible(self, name, edit):
    """Check malformed input shapes/dtypes and empty inputs are ineligible.

    Args:
      name: Name of the fixture input or option under inspection.
      edit: Callable transforming one fixture input into an unsupported variant.
    """
    case = self._case()
    self.assertFalse(
        _static_eligibility(_replace(case, **{name: edit(case.values[name])}))
    )

  @parameterized.named_parameters(
      (
          "unsupported_prefill_tile",
          dict(mixed_tile_size=32),
          "mixed_tile_size",
          None,
      ),
      (
          "supported_prefill_tile",
          dict(mixed_tile_size=128),
          "mixed_tile_size",
          128,
      ),
      (
          "unsupported_decode_group",
          dict(decode_tile_size=9),
          "decode_tile_size",
          None,
      ),
  )
  def test_public_dispatch_normalizes_tile_hints(self, options, name, expected):
    """Check public dispatch normalizes unsupported hints while preserving valid
    ones.

    Args:
      options: Static option overrides for the selected test case.
      name: Name of the fixture input or option under inspection.
      expected: Expected value for the selected parameterized case.
    """
    case = self._case()
    with (
        jax.disable_jit(),
        mock.patch.object(
            prefill_decode,
            "_fused_conv1d_gdn_fast",
            return_value=mock.sentinel.result,
        ) as fused,
        mock.patch.object(wrapper, "fused_conv1d_gdn") as fallback,
    ):
      result = prefill_decode.fused_conv1d_gdn(
          *_device_args(case), **case.options, **options
      )
    self.assertIs(result, mock.sentinel.result)
    fused.assert_called_once()
    fallback.assert_not_called()
    self.assertEqual(fused.call_args.kwargs[name], expected)

  @parameterized.named_parameters(
      ("compute_dtype", dict(compute_precision=jnp.bfloat16)),
      ("speculative_window", dict(num_spec_tokens=1)),
  )
  def test_public_fallback_preserves_options_and_read_addresses(self, options):
    """Check fallback receives the caller's original options and read arrays.

    Args:
      options: Static option overrides for the selected test case.
    """
    case = self._case()
    reads = jnp.asarray(case.values["state_indices"])
    offsets = jnp.zeros_like(reads)
    with (
        jax.disable_jit(),
        mock.patch.object(prefill_decode, "_fused_conv1d_gdn_fast") as fused,
        mock.patch.object(
            wrapper, "fused_conv1d_gdn", return_value=mock.sentinel.result
        ) as fallback,
    ):
      result = prefill_decode.fused_conv1d_gdn(
          *_device_args(case), reads, offsets, **case.options, **options
      )
    self.assertIs(result, mock.sentinel.result)
    fused.assert_not_called()
    fallback.assert_called_once()
    self.assertIs(fallback.call_args.args[13], reads)
    self.assertIs(fallback.call_args.args[14], offsets)
    for name, value in options.items():
      self.assertEqual(fallback.call_args.kwargs[name], value)

  def test_speculative_decoding_requires_read_offsets(self):
    """Check speculative decoding requires read offsets."""
    case = self._case()
    with self.assertRaises(ValueError):
      prefill_decode.fused_conv1d_gdn(
          *_device_args(case), num_spec_tokens=1, **case.options
      )


class GDNVMEMPlanningTest(parameterized.TestCase):
  """CPU checks of the allocator's shapes and both VMEM budget decisions."""

  @parameterized.product(
      activation_dtype=(jnp.bfloat16, jnp.float32),
      recurrent_dtype=(jnp.bfloat16, jnp.float32),
      group=(4, 8),
      kernel=(1, 4),
      bias=(False, True),
      layout=(
          (4, 12, 256, 8, 16),
          (16, 48, 128, 8, 16),
          (1, 4, 512, 16, 24),  # Narrow many-request pool.
          (1, 4, 512, 16, 32),  # Same workload with more inactive slots.
      ),
  )
  def test_ring_thresholds_agree_with_eligibility(
      self,
      activation_dtype,
      recurrent_dtype,
      group,
      kernel,
      bias,
      layout,
  ):
    """Capture real allocation descriptors and compare full/reduced-ring
    thresholds with eligibility across layouts.

    Args:
      activation_dtype: Dtype of QKV and gate inputs.
      recurrent_dtype: Recurrent-cache dtype for the parameterized fixture.
      group: Requested number of decode members in each group.
      kernel: Convolution window size for the allocation fixture.
      bias: Whether the allocation fixture includes convolution bias.
      layout: Fixture tuple (Q/K heads, value heads, tile rows, requests, cache
        slots).
    """
    # Capture the allocator's real descriptors before a Pallas call is built.
    # This catches SMEM leaves and shape changes a copied fixture would miss.
    n_kq, n_v, tile, requests, slots = layout
    width = 2 * n_kq * _D_K + n_v * _D_V
    ring = fused_tiling.state_ring_depth(n_kq, n_v)
    desc = jax.ShapeDtypeStruct
    args = (
        desc((tile, width), activation_dtype),
        desc((tile, n_v), activation_dtype),
        desc((tile, n_v), activation_dtype),
        desc((slots, kernel - 1, width), jnp.bfloat16),
        desc((slots, n_v, _D_K, _D_V), recurrent_dtype),
        desc((width, 1, kernel), jnp.float32),
        desc((width,), jnp.float32) if bias else None,
        desc((n_v,), jnp.float32),
        desc((n_v,), jnp.float32),
        desc((requests + 1,), jnp.int32),
        desc((requests,), jnp.int32),
        desc((3,), jnp.int32),
        desc((requests,), jnp.int32),
        desc((requests,), jnp.int32),
        desc((requests,), jnp.int32),
    )
    captured = {}

    class ScratchCaptured(Exception):
      """Sentinel exception that stops tracing after descriptors and operands
      have been captured.
      """

      pass

    def intercept_pallas_call(*unused, **kwargs):
      """Intercept Pallas construction to collect scratch shapes and cache
      aliases.

      Args:
        *unused: Unused positional arguments to the mocked Pallas constructor.
        **kwargs: Pallas constructor options carrying scratch shapes and
          input/output aliases.

      Returns:
        Operand-capture callback replacing the constructed kernel call.
      """
      captured["scratch"] = kwargs["scratch_shapes"]
      captured["aliases"] = kwargs["input_output_aliases"]

      def capture_operands(*operands):
        """Capture staged weight descriptors and stop before any kernel
        execution.

        Args:
          *operands: Flattened Pallas operands captured before kernel execution.

        Raises:
          ScratchCaptured: Always, after collecting the staged operands.
        """
        captured["weights"] = jax.tree.map(
            lambda x: desc(x.shape, x.dtype), operands[-2]
        )
        captured["dense_conv"] = jax.tree.map(
            lambda x: desc(x.shape, x.dtype), operands[-1]
        )
        raise ScratchCaptured

      return capture_operands

    with (
        mock.patch.object(
            prefill_decode.pl, "pallas_call", side_effect=intercept_pallas_call
        ),
        mock.patch.object(
            fused_tiling.config,
            "get_vmem_limit_bytes",
            return_value=int(0.8 * 128 * 1024 * 1024),
        ),
        self.assertRaises(ScratchCaptured),
    ):
      jax.eval_shape(
          lambda *values: prefill_decode._fused_conv1d_gdn_fast.__wrapped__(
              *values,
              n_kq=n_kq,
              n_v=n_v,
              d_k=_D_K,
              d_v=_D_V,
              kernel_size=kernel,
              decode_tile_size=group,
              mixed_tile_size=tile,
          ),
          *args,
      )
    scratch = captured["scratch"]
    self.assertEqual(captured["aliases"], {9: 1, 10: 2})
    self.assertFalse(
        {
            "preserve_fallback_state_ref",
            "recurrent_preserve_scratch_ref",
            "recurrent_preserve_slots_ref",
            "recurrent_preserve_load_sem_ref",
            "recurrent_preserve_store_sem_ref",
        }
        & scratch.keys()
    )
    # These are added only after the buffer decision. Retain all other leaves,
    # including real SMEM descriptors, to exercise the memory-space filter.
    state_keys = {
        "decode_member0_state_ref",
        "decode_extra_state_refs",
        "decode_state_load_sem_refs",
        "decode_state_store_sem_refs",
        "metadata_ref",
        "request_prefix_ref",
        "occupancy_ref",
    }
    resident = {
        key: value for key, value in scratch.items() if key not in state_keys
    }
    weights, dense_conv = captured["weights"], captured["dense_conv"]

    def ring_depths(limit):
      """Query the allocator's ring choices for a supplied memory limit.

      Args:
        limit: VMEM budget in bytes; None retains full recurrent rings.

      Returns:
        Per-member recurrent ring depths selected at the supplied VMEM limit.
      """
      return fused_tiling.decode_state_ring_depths(
          group,
          resident,
          weights,
          dense_conv,
          limit,
          member0_buffered=recurrent_dtype != jnp.float32,
      )

    low, high = 0, 256 * 1024 * 1024
    full = (ring,) * group
    while low < high:
      mid = (low + high) // 2
      if ring_depths(mid) == full:
        high = mid
      else:
        low = mid + 1
    full_budget = low
    state_bytes = fused_tiling.padded_vmem_bytes(
        (n_v, _D_K, _D_V), jnp.dtype(recurrent_dtype).itemsize
    )
    reduced_budget = full_budget - (group - 2) * (ring - 1) * state_bytes

    def fits(limit):
      """Query resource eligibility under mocked VMEM and SMEM capacities.

      Args:
        limit: VMEM budget in bytes used for the resource estimate.

      Returns:
        Whether the fixture's resource estimate fits the supplied budget.
      """
      with (
          mock.patch.object(
              fused_tiling.config, "get_vmem_limit_bytes", return_value=limit
          ),
          mock.patch.object(
              fused_tiling.pltpu,
              "get_tpu_info",
              return_value=SimpleNamespace(smem_capacity_bytes=1024 * 1024),
          ),
      ):
        # pylint: disable-next=protected-access
        return fused_tiling._resources_fit(
            args[:13],
            n_kq=n_kq,
            n_v=n_v,
            chunk_size=tile,
            decode_tile_size=group,
        )

    self.assertEqual(ring_depths(full_budget), full)
    self.assertNotEqual(ring_depths(full_budget - 1), full)
    self.assertTrue(fits(full_budget))
    self.assertTrue(fits(reduced_budget))
    self.assertFalse(fits(reduced_budget - 1))
    v6e_budget = int(0.8 * 128 * 1024 * 1024)
    self.assertEqual(fits(v6e_budget), reduced_budget <= v6e_budget)
    self.assertEqual(
        tuple(ref.shape[0] for ref in scratch["decode_state_load_sem_refs"]),
        ring_depths(v6e_budget),
    )

  def test_only_vmem_descriptors_and_array_operands_use_vmem_budget(self):
    """Check only VMEM descriptors and array operands contribute to the VMEM
    estimate.
    """
    space = fused_tiling.pltpu
    # pylint: disable-next=protected-access
    count = fused_tiling._padded_storage_estimate(
        (
            space.SMEM((24,), jnp.int32),
            space.HBM((24,), jnp.int32),
            space.VMEM((24,), jnp.int32),
            jax.ShapeDtypeStruct((24,), jnp.int32),
        )
    )
    self.assertEqual(count, 2 * 128 * 4)

  @parameterized.named_parameters(
      ("three_slots", 4, 12, 3),
      ("wide_values", 4, 13, 2),
      ("wide_queries", 5, 12, 2),
  )
  def test_ring_depth_boundary(self, n_kq, n_v, expected):
    """Check the head-count boundary between three-slot and two-slot rings.

    Args:
      n_kq: Number of query/key heads.
      n_v: Number of value heads; supported layouts group them evenly over Q/K
        heads.
      expected: Expected value for the selected parameterized case.
    """
    self.assertEqual(fused_tiling.state_ring_depth(n_kq, n_v), expected)


class _GDNKernelTestBase(parameterized.TestCase):
  """Shared native-test helpers for device eligibility, parity, and cache
  preservation.
  """

  def setUp(self):
    """Skip native tests when the upstream hardware support check fails."""
    super().setUp()
    test_base.skip_if_unsupported(self)

  def assert_close(self, actual, desired, message: str) -> None:
    """Compare FP32-converted cache values using _RTOL and _CACHE_ATOL."""
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(desired, np.float32),
        rtol=_RTOL,
        atol=_CACHE_ATOL,
        err_msg=message,
    )

  def assert_output_close(self, actual, desired, message: str) -> None:
    """Compare each output row relative to its reference peak magnitude.

    Args:
      actual: Observed array to check.
      desired: Reference array defining the expected result.
      message: Context included in an assertion failure.
    """
    actual = np.asarray(actual, np.float32)
    desired = np.asarray(desired, np.float32)
    peaks = np.max(np.abs(desired), axis=-1, keepdims=True, initial=0.0)
    zero_rows = peaks[:, 0] == 0
    np.testing.assert_allclose(
        actual[zero_rows],
        desired[zero_rows],
        rtol=0,
        atol=0,
        equal_nan=False,
        err_msg=message,
    )
    scales = np.where(peaks == 0, 1.0, peaks)
    np.testing.assert_allclose(
        actual / scales,
        desired / scales,
        rtol=_RTOL,
        atol=_OUTPUT_ATOL,
        equal_nan=False,
        err_msg=message,
    )

  def assert_identical(self, actual, desired, message: str) -> None:
    """Compare arrays numerically after FP32 conversion, not by raw bits."""
    np.testing.assert_array_equal(
        np.asarray(actual, np.float32),
        np.asarray(desired, np.float32),
        err_msg=message,
    )

  def assert_output_is_live(self, case: _Case, result: _Result) -> None:
    """Check the active output is not entirely erased to zero."""
    rows = np.asarray(result.out[: case.active_tokens], np.float32)
    self.assertGreater(
        np.abs(rows).max(initial=0.0), 0.0, "active output rows are all zero"
    )

  def assert_inactive_slots_preserved(
      self, case: _Case, result: _Result
  ) -> None:
    """Check cache shape/dtype and unchanged bytes in all inactive slots."""
    for name, actual in (
        ("conv_state", result.conv),
        ("recurrent_state", result.recurrent),
    ):
      expected = case.values[name]
      self.assertEqual(actual.dtype, expected.dtype)
      self.assertEqual(actual.shape, expected.shape)
      for slot in range(case.slots):
        if slot not in case.active_slots:
          self.assertTrue(
              actual[slot].tobytes() == expected[slot].tobytes(),
              f"{name} cache slot {slot} was modified",
          )

  def run_fused(
      self,
      case: _Case,
      *,
      read_state_indices=None,
      read_offsets=None,
      **options: Any,
  ) -> _Result:
    """Run the entry point after checking it takes the fused path.

    Args:
      case: Input fixture and its expected active token/cache ownership.
      read_state_indices: Optional [requests] initial-state slots; defaults to
        state_indices.
      read_offsets: Optional [requests] read-slot offsets; decode applies them
        and prefill uses base slots.
      **options: Static configuration overrides forwarded to the selected entry
        point.

    Returns:
      Host result from the fused entry point after confirming fused tile
      eligibility.
    """
    self.assertTrue(
        _selected_tiles(case, **options)[0],
        "this case does not select the fused implementation",
    )
    return _run(
        prefill_decode.fused_conv1d_gdn,
        case,
        read_state_indices=read_state_indices,
        read_offsets=read_offsets,
        **options,
    )

  def assert_matches_fallback(self, case: _Case, **options: Any) -> _Result:
    """Compare active output rows and owned cache slots against the upstream
    wrapper.

    Return the fused result for further assertions.

    Args:
      case: Input fixture and its expected active token/cache ownership.
      **options: Entry-point overrides shared by the fused and reference calls.

    Returns:
      Fused host result after output and active-cache parity checks.
    """
    fused = self.run_fused(case, **options)
    reference = _run(wrapper.fused_conv1d_gdn, case, **options)
    rows = case.active_tokens
    self.assert_output_close(
        fused.out[:rows], reference.out[:rows], "output rows"
    )
    for slot in case.active_slots:
      self.assert_close(
          fused.conv[slot],
          reference.conv[slot],
          f"convolution cache slot {slot}",
      )
      self.assert_close(
          fused.recurrent[slot],
          reference.recurrent[slot],
          f"recurrent cache slot {slot}",
      )
    self.assert_output_is_live(case, fused)
    return fused


class GDNReferenceComparisonTest(parameterized.TestCase):
  """Supported parity cases retain failures and detect erased output."""

  @parameterized.product(
      cache=("conv", "recurrent"),
      bits=((0x00000000, 0x80000000), (0x7FC00001, 0x7FC00002)),
  )
  def test_inactive_cache_bit_change_is_rejected(self, cache, bits):
    """Check inactive-cache preservation detects signed-zero and NaN-payload bit
    changes.

    Args:
      cache: Cache name to corrupt: convolution or recurrent.
      bits: Original and replacement uint32 bit patterns for an inactive cache
        element.
    """
    case = _build_case(
        [_prefill(1, 2)],
        num_decode=0,
        n_kq=2,
        n_v=8,
        conv_state_dtype=jnp.float32,
        recurrent_state_dtype=jnp.float32,
    )
    expected = case.values[f"{cache}_state"]
    expected.view(np.uint32).flat[0] = bits[0]
    actual = expected.copy()
    actual.view(np.uint32).flat[0] = bits[1]
    result = _Result(
        conv=case.values["conv_state"],
        recurrent=case.values["recurrent_state"],
        out=np.empty((0, case.n_v * _D_V), np.float32),
    )
    helper = _GDNKernelTestBase(methodName="runTest")
    helper.assert_inactive_slots_preserved(case, result)
    # Numeric equality hides signed-zero and NaN-payload changes.
    helper.assert_identical(actual[0], expected[0], "numerically equal")
    result = dataclasses.replace(result, **{cache: actual})
    with self.assertRaisesRegex(AssertionError, "cache slot 0"):
      helper.assert_inactive_slots_preserved(case, result)

  def test_unexpected_reference_failure_is_not_skipped(self):
    """Check unexpected upstream reference failures propagate instead of
    becoming skips.
    """
    case = _build_case([_decode(1)], num_decode=1, n_kq=2, n_v=8)
    helper = _GDNKernelTestBase(methodName="runTest")
    error = RuntimeError("unexpected reference compilation failure")
    try:
      with (
          mock.patch.object(helper, "run_fused"),
          mock.patch(f"{__name__}._run", side_effect=error),
      ):
        with self.assertRaises(RuntimeError) as caught:
          helper.assert_matches_fallback(case)
    except SkipTest as skipped:
      self.fail(f"reference failure became a test skip: {skipped}")
    self.assertIs(caught.exception, error)

  @parameterized.named_parameters(
      ("tiny_constant", "tiny_constant"),
      ("erased_fresh_row", "erased_fresh_row"),
  )
  def test_erased_output_fails_parity(self, corruption):
    """Check tiny constant or erased fresh-row output fails numerical parity.

    Args:
      corruption: Output corruption variant used to test the parity oracle.
    """
    case = _build_case(
        [_decode(1), _prefill(2, 1, history=0)], num_decode=1, n_kq=2, n_v=8
    )
    fresh = np.linspace(-1.5e-4, 1.5e-4, case.n_v * _D_V, dtype=np.float32)
    reference = _Result(
        conv=case.values["conv_state"],
        recurrent=case.values["recurrent_state"],
        out=np.stack((50 * fresh, fresh)),
    )
    if corruption == "tiny_constant":
      out = np.full_like(reference.out, 1e-8)
    else:
      out = reference.out.copy()
      out[1] = 0
    fused = dataclasses.replace(reference, out=out)
    helper = _GDNKernelTestBase(methodName="runTest")
    with (
        mock.patch.object(helper, "run_fused", return_value=fused),
        mock.patch(f"{__name__}._run", return_value=reference),
    ):
      with self.assertRaises(AssertionError):
        helper.assert_matches_fallback(case)

  @parameterized.named_parameters(
      (
          "narrow_fp32",
          "GDNNarrowServingLayoutTest",
          "test_three_decode_groups_preserve_inactive_cache_fp32_state",
      ),
      (
          "narrow_bf16",
          "GDNNarrowServingLayoutTest",
          "test_three_decode_groups_preserve_inactive_cache_bf16_state",
      ),
      *(
          (
              f"prefix_offset_{index}",
              "GDNPrefixCacheTest",
              f"test_prefill_offsets_do_not_change_prefix_cache_reads{index}",
          )
          for index in range(8)
      ),
  )
  def test_direct_native_output_oracles_reject_erased_output(
      self, class_name, method_name
  ):
    """Check selected native tests' own output assertions reject erased results,
    with kernels mocked.

    Args:
      class_name: Native test class whose output assertions are exercised with
        mocks.
      method_name: Parameterized native test method whose assertions are
        exercised.
    """
    helper = globals()[class_name](methodName=method_name)

    def result(case, *, erased=False):
      """Build normal or deliberately erased synthetic results for that oracle
      test.

      Args:
        case: Input fixture and its expected active token/cache ownership.
        erased: Whether to replace the synthetic output with a tiny constant.

      Returns:
        Synthetic host result with normal or deliberately erased output.
      """
      width = case.n_v * _D_V
      row = np.linspace(-1.5e-4, 1.5e-4, width, dtype=np.float32)
      out = np.broadcast_to(row, (case.active_tokens, width)).copy()
      if erased:
        out.fill(1e-8)
      return _Result(
          conv=case.values["conv_state"],
          recurrent=case.values["recurrent_state"],
          out=out.astype(case.values["qkv"].dtype),
      )

    # Calling the method directly omits only the native-device setUp guard.
    # Kernels remain mocked; fixture construction and all assertions execute.
    for erased in (False, True):
      with (
          mock.patch.object(
              helper,
              "run_fused",
              side_effect=lambda case, **unused: result(case, erased=erased),
          ),
          mock.patch(
              __name__ + "._run",
              side_effect=lambda unused_impl, case, **unused: result(case),
          ),
      ):
        if erased:
          with self.assertRaises(AssertionError):
            getattr(helper, method_name)()
        else:
          getattr(helper, method_name)()


class GDNFallbackParityTest(_GDNKernelTestBase):
  """Native parity cases across decode/prefill mixes, boundaries, and supported
  options.
  """

  @parameterized.named_parameters(
      # Exercise both sides of the 16-row output boundary.
      dict(testcase_name="one_decode", decodes=1, prefills=()),
      dict(
          testcase_name="fifteen_fresh_decodes",
          decodes=15,
          prefills=(),
          history=0,
      ),
      dict(testcase_name="sixteen_decodes", decodes=16, prefills=()),
      dict(testcase_name="seventeen_decodes", decodes=17, prefills=()),
      dict(testcase_name="fresh_prefill", decodes=0, prefills=(9,), history=0),
      dict(
          testcase_name="continuing_prefill",
          decodes=0,
          prefills=(9,),
          history=20,
      ),
      # Exercise both sides of this layout's default 256-row tile.
      dict(
          testcase_name="prefill_of_exactly_one_tile",
          decodes=0,
          prefills=(256,),
      ),
      dict(testcase_name="prefill_across_tiles", decodes=0, prefills=(300,)),
      dict(testcase_name="decodes_and_prefill", decodes=3, prefills=(9,)),
      dict(
          testcase_name="decodes_and_two_prefills",
          decodes=2,
          prefills=(5, 11),
      ),
      dict(testcase_name="single_token_prefill", decodes=2, prefills=(1,)),
      dict(
          testcase_name="float32_conv_state",
          decodes=2,
          prefills=(7,),
          conv_state_dtype=jnp.float32,
      ),
      dict(
          testcase_name="kernel_size_three",
          decodes=2,
          prefills=(7,),
          kernel_size=3,
      ),
      dict(
          testcase_name="without_conv_bias",
          decodes=2,
          prefills=(7,),
          conv_bias=False,
      ),
  )
  def test_matches_fallback(
      self,
      decodes: int,
      prefills: tuple[int, ...],
      *,
      history: int = 8,
      kernel_size: int = 4,
      conv_bias: bool = True,
      conv_state_dtype: Any = jnp.bfloat16,
  ):
    """Check parameterized mixed workloads and inactive caches against upstream.

    Args:
      decodes: Number of one-token requests before the prefill requests.
      prefills: Tuple of submitted token counts for the prefill requests.
      history: Number of tokens already consumed before this request.
      kernel_size: Positive static convolution window size.
      conv_bias: Whether the fixture includes convolution bias.
      conv_state_dtype: Dtype of the convolution-cache fixture.
    """
    active = [_decode(1 + index, history=history) for index in range(decodes)]
    active += [
        _prefill(1 + decodes + index, length, history=history)
        for index, length in enumerate(prefills)
    ]
    case = _build_case(
        active,
        num_decode=decodes,
        n_kq=2,
        n_v=8,
        kernel_size=kernel_size,
        conv_bias=conv_bias,
        conv_state_dtype=conv_state_dtype,
    )
    result = self.assert_matches_fallback(case)
    self.assert_inactive_slots_preserved(case, result)

  def test_matches_fallback_with_16_kq_and_32_value_heads(self):
    """Check parity for the wider 16-Q/K, 32-value-head layout."""
    case = _build_case(
        [_decode(1), _decode(2), _prefill(3, 10)],
        num_decode=2,
        n_kq=16,
        n_v=32,
    )
    self.assert_matches_fallback(case)

  def test_conv_bias_reaches_the_computation(self):
    # A small bias could disappear within numerical tolerances.
    """Check a deliberately large convolution bias actually changes output."""
    active = [_decode(1), _prefill(2, 7)]
    with_bias = _build_case(
        active, num_decode=1, n_kq=2, n_v=8, conv_bias_scale=1.0
    )
    without_bias = _build_case(
        active, num_decode=1, n_kq=2, n_v=8, conv_bias=False
    )
    biased = np.asarray(self.run_fused(with_bias).out, np.float32)
    unbiased = np.asarray(self.run_fused(without_bias).out, np.float32)
    # Normalization can hide bias within tolerances; require an output bit change.
    self.assertFalse(
        np.array_equal(biased, unbiased),
        "the convolution bias did not change the output",
    )


class GDNAlignedPrefillTest(_GDNKernelTestBase):
  """Native tests of the special long-prefill alignment path."""

  @parameterized.named_parameters(
      ("bf16_long", jnp.bfloat16, 1, (4096,), False),
      ("fp32_long", jnp.float32, 1, (4096,), False),
      ("bf16_shared_prefix", jnp.bfloat16, 3, (4096, 4103), True),
      ("fp32_shared_prefix", jnp.float32, 3, (4096, 4103), True),
  )
  def test_matches_fallback(self, dtype, decodes, lengths, shared_prefix):
    """Check BF16/FP32 long prefills, including multiple requests sharing an
    unwritten prefix.

    Args:
      dtype: Shared dtype for activations and both caches.
      decodes: Number of one-token requests before the prefill requests.
      lengths: Submitted token lengths of the prefill requests.
      shared_prefix: Whether requests read one shared, unwritten checkpoint
        slot.
    """
    requests = [_decode(i + 1, history=1024) for i in range(decodes)]
    requests += [
        _prefill(decodes + i + 1, length, history=1024)
        for i, length in enumerate(lengths)
    ]
    case = _build_case(
        requests,
        num_decode=decodes,
        n_kq=8,
        n_v=24,
        slots=len(requests) + 4,
        activation_dtype=dtype,
        conv_state_dtype=dtype,
        recurrent_state_dtype=dtype,
        seed=1729,
    )
    options = dict(decode_tile_size=8, mixed_tile_size=128)
    if shared_prefix:
      options.update(
          read_state_indices=np.full(len(requests), case.slots - 1, np.int32),
          read_offsets=np.zeros(len(requests), np.int32),
      )
    result = self.assert_matches_fallback(case, **options)
    self.assert_inactive_slots_preserved(case, result)


class GDNReducedDecodeGroupTest(_GDNKernelTestBase):
  """Native parity for wide FP32 layouts that require a smaller decode group."""

  @parameterized.named_parameters(
      ("mixed_history_tail", 8, (65,), False),
      ("shared_prefix_tail", 17, (129,), True),
      ("pure_decode", 8, (), False),
      ("pure_prefill", 0, (1, 64, 65, 129), False),
  )
  def test_matches_fallback(self, decodes, prefills, shared_prefix):
    """Check the selected three-member/64-row plan across mixed, pure, and
    shared-prefix cases.

    Args:
      decodes: Number of one-token requests before the prefill requests.
      prefills: Tuple of submitted token counts for the prefill requests.
      shared_prefix: Whether requests read one shared, unwritten checkpoint
        slot.
    """
    requests = [
        _decode(i + 1, history=1024 if shared_prefix or i % 3 else 0)
        for i in range(decodes)
    ]
    requests += [
        _prefill(decodes + i + 1, length, history=1024)
        for i, length in enumerate(prefills)
    ]
    case = _build_case(
        requests,
        num_decode=decodes,
        n_kq=16,
        n_v=64,
        padding_slots=(0,),
        slots=len(requests) + 4,
        activation_dtype=jnp.float32,
        conv_state_dtype=jnp.float32,
        recurrent_state_dtype=jnp.float32,
        seed=1729,
    )
    self.assertEqual(_selected_tiles(case), (True, 3, 64))
    options = {}
    if shared_prefix:
      options.update(
          read_state_indices=np.full(
              len(requests) + 1, case.slots - 1, np.int32
          ),
          read_offsets=np.zeros(len(requests) + 1, np.int32),
      )
    result = self.assert_matches_fallback(case, **options)
    self.assert_inactive_slots_preserved(case, result)


class GDNCacheSlotTest(_GDNKernelTestBase):
  """Native tests of write-slot ownership and invalid schedule rejection."""

  @parameterized.named_parameters(
      ("decode_moves_slot", 0),
      ("prefill_moves_slot", 2),
  )
  def test_nonzero_slots_are_exchangeable(self, request_index: int):
    # Relabeling nonzero slots must preserve output and exchange only cache rows.
    """Check relabeling nonzero cache slots only relabels the resulting caches.

    Args:
      request_index: Index of the active request whose write slot is relabeled.
    """
    case = _build_case(
        [_decode(1), _decode(2), _prefill(3, 7)],
        num_decode=2,
        n_kq=2,
        n_v=8,
        slots=6,
    )
    owned = case.active_slots[request_index]
    spare = 4
    relabeled = _swap_slots(case, spare, owned)
    self.assertEqual(relabeled.active_slots[request_index], spare)
    baseline = self.run_fused(case)
    moved = self.run_fused(relabeled)
    self.assert_output_is_live(case, baseline)
    self.assert_output_is_live(relabeled, moved)
    self.assert_identical(moved.out, baseline.out, "output rows")
    self.assert_identical(
        moved.conv,
        _swap_rows(baseline.conv, spare, owned),
        "convolution cache",
    )
    self.assert_identical(
        moved.recurrent,
        _swap_rows(baseline.recurrent, spare, owned),
        "recurrent cache",
    )

  def test_every_nonzero_slot_of_the_pool_is_usable(self):
    # Every usable slot is active; only reserved slot zero stays unchanged.
    """Check every nonzero slot can be active while reserved slot zero stays
    unchanged.
    """
    case = _build_case(
        [_decode(1), _decode(2), _prefill(3, 7)],
        num_decode=2,
        n_kq=2,
        n_v=8,
        slots=4,
    )
    result = self.run_fused(case)
    self.assert_output_is_live(case, result)
    self.assert_inactive_slots_preserved(case, result)
    for slot in case.active_slots:
      self.assertFalse(
          np.array_equal(
              np.asarray(result.recurrent[slot], np.float32),
              np.asarray(case.values["recurrent_state"][slot], np.float32),
          ),
          f"recurrent cache slot {slot} was not advanced",
      )

  def _fail_closed_case(self, many_requests: bool) -> _Case:
    """Return a valid case with a small or many-request cache pool.

    Args:
      many_requests: Whether to use the larger narrow-head request fixture.

    Returns:
      Valid fixture to corrupt for schedule-rejection checks.
    """
    if many_requests:
      # Narrow batches with at least 16 requests exercise cache preservation.
      active = [_decode(1 + index) for index in range(15)]
      active.append(_prefill(16, 5))
      return _build_case(
          active,
          num_decode=15,
          n_kq=1,
          n_v=2,
          padding_slots=(0, 0),
          slots=19,
      )
    return _build_case(
        [_decode(1), _decode(2), _prefill(3, 5)],
        num_decode=2,
        n_kq=2,
        n_v=8,
        padding_slots=(0, 0),
    )

  # Padded lengths may be nonzero or negative; they must never address caches.
  @parameterized.product(
      defect=(
          "decode_owns_slot_zero",
          "prefill_owns_slot_zero",
          "negative_slot",
          "negative_active_end",
          "negative_active_end_past_smem",
          "slot_equal_to_the_pool_size",
          "slot_far_past_the_pool",
          "repeated_slot",
          "decode_with_several_tokens",
          "active_request_shorter_than_its_tokens",
          "offsets_not_starting_at_zero",
      ),
      many_requests=(False, True),
  )
  def test_invalid_schedule_fails_closed(
      self, defect: str, many_requests: bool
  ):
    # Rejection preserves caches for both request layouts.
    """Inject invalid ownership/metadata and require zero output plus unchanged
    caches.

    Args:
      defect: Schedule corruption to inject into an otherwise valid fixture.
      many_requests: Whether to use the larger narrow-head request fixture.
    """
    case = self._fail_closed_case(many_requests)
    slots = case.values["state_indices"].copy()
    distribution = case.values["distribution"].copy()
    starts = case.values["query_start_loc"].copy()
    seq_lens = case.values["seq_lens"].copy()
    if defect == "decode_owns_slot_zero":
      slots[0] = 0
    elif defect == "prefill_owns_slot_zero":
      slots[len(case.active_slots) - 1] = 0
    elif defect == "negative_slot":
      slots[0] = -1
    elif defect == "negative_active_end":
      distribution[2] = -1
    elif defect == "negative_active_end_past_smem":
      distribution[2] = -(len(slots) + 4096)
    elif defect == "slot_equal_to_the_pool_size":
      slots[1] = case.slots
    elif defect == "slot_far_past_the_pool":
      slots[1] = case.slots + 4096
    elif defect == "repeated_slot":
      slots[1] = slots[0]
    elif defect == "decode_with_several_tokens":
      distribution[0] = distribution[2]
    elif defect == "active_request_shorter_than_its_tokens":
      seq_lens[len(case.active_slots) - 1] = 0
    elif defect == "offsets_not_starting_at_zero":
      starts[0] = 1
    else:
      raise ValueError(f"unknown defect {defect}")
    broken = _replace(
        case,
        state_indices=slots,
        distribution=distribution,
        query_start_loc=starts,
        seq_lens=seq_lens,
    )
    result = self.run_fused(broken)
    self.assert_identical(
        result.out,
        np.zeros_like(np.asarray(result.out, np.float32)),
        "output",
    )
    self.assert_identical(
        result.conv, case.values["conv_state"], "convolution cache"
    )
    self.assert_identical(
        result.recurrent, case.values["recurrent_state"], "recurrent cache"
    )


class GDNPaddedBatchTest(_GDNKernelTestBase):
  """Native tests that inactive requests and token padding do not affect active
  work.
  """

  def _padded_case(self, padding_slots: Sequence[int]) -> _Case:
    """Build a padded mixed batch with caller-selected inactive slot IDs.

    Args:
      padding_slots: Inactive request slot IDs; they may repeat or be out of
        range.

    Returns:
      Mixed fixture with the requested inactive slot IDs.
    """
    active = [_decode(1), _decode(2), _decode(3), _prefill(4, 7)]
    active.append(_prefill(5, 3))
    return _build_case(
        active,
        num_decode=3,
        n_kq=2,
        n_v=8,
        padding_slots=padding_slots,
        slots=9,
    )

  def test_padding_slot_ids_are_never_dereferenced(self):
    # Inactive IDs may repeat active IDs or lie outside the pool.
    """Check hostile inactive slot IDs are never used to address caches."""
    clean = self._padded_case((0, 0, 0))
    hostile = self._padded_case((3, -1, 1 << 20))
    expected = self.run_fused(clean)
    actual = self.run_fused(hostile)
    self.assert_output_is_live(clean, expected)
    self.assert_identical(actual.out, expected.out, "output rows")
    self.assert_identical(actual.conv, expected.conv, "convolution cache")
    self.assert_identical(
        actual.recurrent, expected.recurrent, "recurrent cache"
    )

  def test_inactive_entries_do_not_change_the_active_result(self):
    """Check adding inactive request entries preserves active outputs and
    states.
    """
    padded = self._padded_case((0, 0, 0))
    unpadded = self._padded_case(())
    with_padding = self.run_fused(padded)
    without_padding = self.run_fused(unpadded)
    rows = padded.active_tokens
    self.assert_identical(
        with_padding.out[:rows], without_padding.out[:rows], "output rows"
    )
    for slot in padded.active_slots:
      self.assert_identical(
          with_padding.conv[slot],
          without_padding.conv[slot],
          f"convolution cache slot {slot}",
      )
      self.assert_identical(
          with_padding.recurrent[slot],
          without_padding.recurrent[slot],
          f"recurrent cache slot {slot}",
      )
    self.assert_inactive_slots_preserved(padded, with_padding)

  def test_padded_token_bucket_is_zero_filled(self):
    """Check unused token-bucket rows are zero-filled while active rows match."""
    active = [_decode(1), _decode(2), _prefill(3, 3)]
    packed = _build_case(active, num_decode=2, n_kq=2, n_v=8)
    padded = _pad_token_bucket(packed, 32)
    rows = packed.active_tokens
    self.assertEqual(padded.values["qkv"].shape[0], 32)
    expected = self.run_fused(packed)
    actual = self.run_fused(padded)
    self.assert_output_is_live(packed, expected)
    self.assert_output_close(
        actual.out[:rows], expected.out[:rows], "output rows"
    )
    self.assert_identical(
        actual.out[rows:],
        np.zeros_like(np.asarray(actual.out[rows:], np.float32)),
        "rows outside the active prefix",
    )


class GDNNarrowServingLayoutTest(_GDNKernelTestBase):
  """Native tests of narrow serving layouts spanning multiple decode groups."""

  @parameterized.named_parameters(
      ("fp32_state", jnp.float32),
      ("bf16_state", jnp.bfloat16),
  )
  def test_three_decode_groups_preserve_inactive_cache(self, recurrent_dtype):
    """Check three-group decode parity and inactive-cache preservation for both
    cache dtypes.

    Args:
      recurrent_dtype: Recurrent-cache dtype for the parameterized fixture.
    """
    active = [_decode(1 + index, history=1) for index in range(17)]
    active.append(_prefill(18, 5, history=1))
    case = _build_case(
        active,
        num_decode=17,
        n_kq=1,
        n_v=1,
        slots=31,
        kernel_size=4,
        recurrent_state_dtype=recurrent_dtype,
    )
    result = self.run_fused(case)
    reference = _run(wrapper.fused_conv1d_gdn, case)
    self.assert_output_close(result.out, reference.out, "output rows")
    for slot in case.active_slots:
      self.assert_close(
          result.recurrent[slot], reference.recurrent[slot], "active state"
      )
    self.assert_output_is_live(case, result)
    self.assert_inactive_slots_preserved(case, result)

  def _serving_case(self, slots: int) -> _Case:
    """Build the narrow serving fixture with a selectable cache-pool size.

    Args:
      slots: Number of cache slots, including reserved slot zero.

    Returns:
      Narrow-head serving fixture with the requested cache-pool size.
    """
    active = [_decode(1 + index) for index in range(15)]
    active.append(_prefill(16, 5))
    return _build_case(
        active,
        num_decode=15,
        n_kq=1,
        n_v=2,
        padding_slots=(0, 0),
        slots=slots,
    )

  @parameterized.named_parameters(
      ("tight_pool", 19),
      ("larger_pool", 30),
  )
  def test_batch_equals_one_call_per_request(self, slots: int):
    """Check one batched call agrees with running each request alone.

    Args:
      slots: Number of cache slots, including reserved slot zero.
    """
    case = self._serving_case(slots)
    batch = self.run_fused(case)
    self.assert_output_is_live(case, batch)
    starts = case.values["query_start_loc"]
    for index, slot in enumerate(case.active_slots):
      alone = self.run_fused(_single_request_case(case, index))
      begin, end = int(starts[index]), int(starts[index + 1])
      self.assert_output_close(
          batch.out[begin:end], alone.out, f"output rows of request {index}"
      )
      self.assert_close(
          batch.conv[slot],
          alone.conv[slot],
          f"convolution cache slot {slot}",
      )
      self.assert_close(
          batch.recurrent[slot],
          alone.recurrent[slot],
          f"recurrent cache slot {slot}",
      )
    self.assert_inactive_slots_preserved(case, batch)


class GDNReadOffsetDispatchTest(parameterized.TestCase):
  """Decode offsets and prefill base slots retain upstream dispatch semantics."""

  @parameterized.parameters(0, 1, 2)
  def test_only_decode_offsets_reach_the_fused_kernel(self, num_decode):
    """Check only decode offsets reach the fused kernel.

    Args:
      num_decode: Number of active one-token requests at the start of the batch.
    """
    case = _build_case([_decode(1), _decode(2)], num_decode=2, n_kq=2, n_v=8)
    case = _replace(
        case, distribution=np.array([num_decode, 2, 2], dtype=np.int32)
    )
    offsets = jnp.array([1, 7], dtype=jnp.int32)
    with (
        mock.patch.object(
            fused_tiling, "select_tiles", return_value=(True, None, 128)
        ),
        mock.patch.object(
            prefill_decode,
            "_fused_conv1d_gdn_fast",
            side_effect=lambda *a, **_: a[14],
        ),
    ):
      actual = prefill_decode.fused_conv1d_gdn.__wrapped__(
          *_device_args(case), read_offsets=offsets, **case.options
      )
    expected = ([0, 0], [1, 0], [1, 7])[num_decode]
    np.testing.assert_array_equal(actual, expected)

  def test_native_helper_keeps_read_arrays_out_of_static_options(self):
    """Check the native test helper keeps dynamic read arrays separate from
    static tile options.
    """
    case = _build_case(
        [_decode(1), _prefill(2, 7)], num_decode=1, n_kq=2, n_v=8
    )
    reads = jnp.array([1, 1], dtype=jnp.int32)
    offsets = jnp.array([0, 99], dtype=jnp.int32)
    helper = _GDNKernelTestBase(methodName="runTest")
    with mock.patch(
        __name__ + "._run", return_value=mock.sentinel.result
    ) as run:
      actual = helper.run_fused(
          case, read_state_indices=reads, read_offsets=offsets
      )
    self.assertIs(actual, mock.sentinel.result)
    self.assertIs(run.call_args.kwargs["read_state_indices"], reads)
    self.assertIs(run.call_args.kwargs["read_offsets"], offsets)

  def test_fallback_receives_original_read_offsets(self):
    """Check fallback receives the original offsets rather than fused-path
    masking.
    """
    case = _build_case(
        [_decode(1), _prefill(2, 7)], num_decode=1, n_kq=2, n_v=8
    )
    offsets = jnp.array([1, 7], dtype=jnp.int32)
    with (
        mock.patch.object(
            fused_tiling, "select_tiles", return_value=(False, None, None)
        ),
        mock.patch.object(
            wrapper, "fused_conv1d_gdn", side_effect=lambda *a, **_: a[14]
        ),
    ):
      actual = prefill_decode.fused_conv1d_gdn.__wrapped__(
          *_device_args(case), read_offsets=offsets, **case.options
      )
    self.assertIs(actual, offsets)


class GDNPrefixCacheTest(_GDNKernelTestBase):
  """Warm prefills use shared, unwritten base slots independently of offsets."""

  @parameterized.product(
      recurrent_dtype=(jnp.bfloat16, jnp.float32), mixed=(False, True)
  )
  def test_shared_prefix_survives_in_place_state_output(
      self, recurrent_dtype, mixed
  ):
    """Check shared checkpoint reads agree with copying the checkpoint into each
    destination.

    Args:
      recurrent_dtype: Recurrent-cache dtype for the parameterized fixture.
      mixed: Whether the fixture combines decode and prefill requests.
    """
    requests = [
        _prefill(slot, 5, history=8) if mixed and slot > 8 else _decode(slot)
        for slot in range(1, 17)
    ]
    case = _build_case(
        requests,
        num_decode=8 if mixed else 16,
        n_kq=1,
        n_v=4,
        slots=19,
        recurrent_state_dtype=recurrent_dtype,
    )
    # Materializing the checkpoint in each destination defines the same call
    # without shared reads. Slot 17 itself must remain untouched.
    caches = {}
    for name in ("conv_state", "recurrent_state"):
      caches[name] = case.values[name].copy()
      caches[name][1:17] = caches[name][17]
    expected = self.run_fused(_replace(case, **caches))
    actual = self.run_fused(
        case, read_state_indices=jnp.full((16,), 17, dtype=jnp.int32)
    )
    self.assert_output_is_live(case, actual)
    self.assert_identical(actual.out, expected.out, "shared checkpoint output")
    self.assert_identical(
        actual.conv, expected.conv, "shared checkpoint convolution"
    )
    self.assert_identical(
        actual.recurrent,
        expected.recurrent,
        "shared checkpoint recurrent state",
    )
    self.assert_inactive_slots_preserved(case, actual)

  @parameterized.product(
      recurrent_dtype=(jnp.bfloat16, jnp.float32),
      mixed=(False, True),
      prefill_offset=(1, 99),
  )
  def test_prefill_offsets_do_not_change_prefix_cache_reads(
      self, recurrent_dtype, mixed, prefill_offset
  ):
    """Check prefill offsets are ignored and the result agrees with upstream.

    Args:
      recurrent_dtype: Recurrent-cache dtype for the parameterized fixture.
      mixed: Whether the fixture combines decode and prefill requests.
      prefill_offset: Offset injected into prefill requests to verify it is
        ignored.
    """
    requests = [
        _decode(2) if mixed else _prefill(2, 5, history=8),
        _prefill(3, 7, history=8),
    ]
    case = _build_case(
        requests,
        num_decode=int(mixed),
        n_kq=2,
        n_v=8,
        slots=4,
        recurrent_state_dtype=recurrent_dtype,
    )
    read_slots = jnp.ones((2,), dtype=jnp.int32)
    normal = jnp.array([int(mixed), 0], dtype=jnp.int32)
    changed = jnp.array(
        [1 if mixed else prefill_offset, prefill_offset], dtype=jnp.int32
    )
    options = dict(read_state_indices=read_slots)
    fused = self.run_fused(case, read_offsets=changed, **options)
    expected = self.run_fused(case, read_offsets=normal, **options)
    self.assert_output_is_live(case, fused)
    self.assert_identical(fused.out, expected.out, "prefill offset output")
    self.assert_identical(
        fused.conv, expected.conv, "prefill offset convolution"
    )
    self.assert_identical(
        fused.recurrent, expected.recurrent, "prefill offset recurrent state"
    )
    self.assert_inactive_slots_preserved(case, fused)
    # Supported native fixtures must retain unexpected upstream failures.
    reference = _run(
        wrapper.fused_conv1d_gdn, case, read_offsets=changed, **options
    )
    for slot in case.active_slots:
      self.assert_close(
          fused.conv[slot], reference.conv[slot], "prefix convolution"
      )
      self.assert_close(
          fused.recurrent[slot],
          reference.recurrent[slot],
          "prefix recurrent state",
      )
    self.assert_output_close(fused.out, reference.out, "prefix output")


class GDNEntryPointContractTest(_GDNKernelTestBase):
  """Native tests of public entry-point defaults, cache addressing, and
  fallback.
  """

  def _case(self, **kwargs: Any) -> _Case:
    """Build the standard mixed fixture for entry-point checks."""
    return _build_case(
        [_decode(1), _decode(2), _prefill(3, 7)],
        num_decode=2,
        n_kq=2,
        n_v=8,
        **kwargs,
    )

  def test_required_arguments_only(self):
    """Check required-only calls preserve output/cache shapes and dtypes and
    agree with helper defaults.
    """
    case = self._case()
    values = case.values
    (conv, recurrent), out = prefill_decode.fused_conv1d_gdn(
        jnp.asarray(values["qkv"]),
        jnp.asarray(values["b"]),
        jnp.asarray(values["a"]),
        jnp.asarray(values["conv_state"]),
        jnp.asarray(values["recurrent_state"]),
        jnp.asarray(values["conv_weight"]),
        jnp.asarray(values["conv_bias"]),
        jnp.asarray(values["a_log"]),
        jnp.asarray(values["dt_bias"]),
        jnp.asarray(values["query_start_loc"]),
        jnp.asarray(values["state_indices"]),
        jnp.asarray(values["distribution"]),
        jnp.asarray(values["seq_lens"]),
        n_kq=2,
        n_v=8,
        d_k=_D_K,
        d_v=_D_V,
        kernel_size=4,
    )
    result = _Result(
        conv=np.asarray(conv),
        recurrent=np.asarray(recurrent),
        out=np.asarray(out),
    )
    self.assertEqual(result.out.shape, (case.active_tokens, 8 * _D_V))
    self.assertEqual(result.out.dtype, values["qkv"].dtype)
    self.assertEqual(result.conv.shape, values["conv_state"].shape)
    self.assertEqual(result.conv.dtype, values["conv_state"].dtype)
    self.assertEqual(result.recurrent.shape, values["recurrent_state"].shape)
    self.assertEqual(result.recurrent.dtype, values["recurrent_state"].dtype)
    self.assertTrue(np.isfinite(np.asarray(result.out, np.float32)).all())
    self.assert_output_is_live(case, result)
    self.assert_identical(
        result.out, self.run_fused(case).out, "defaulted output rows"
    )

  def test_explicit_default_read_address(self):
    """Check explicit default read arrays match omitted read arrays."""
    case = self._case()
    expected = self.run_fused(case)
    actual = _run(
        prefill_decode.fused_conv1d_gdn,
        case,
        read_state_indices=jnp.asarray(case.values["state_indices"]),
        read_offsets=jnp.zeros_like(jnp.asarray(case.values["state_indices"])),
    )
    self.assert_identical(actual.out, expected.out, "output rows")
    self.assert_identical(actual.conv, expected.conv, "convolution cache")
    self.assert_identical(
        actual.recurrent, expected.recurrent, "recurrent cache"
    )

  def test_read_offsets_address_the_same_slot(self):
    # Decode read slots split into base + offset, including base zero.
    # Prefill read slots use the base directly. Active write slot zero rejects.
    """Check decode base-plus-offset addressing matches direct slots while
    prefills use base slots.
    """
    case = self._case()
    indices = case.values["state_indices"]
    self.assertTrue((indices[: len(case.active_slots)] > 0).all())
    expected = self.run_fused(case)
    actual = _run(
        prefill_decode.fused_conv1d_gdn,
        case,
        read_state_indices=jnp.asarray(indices) - jnp.array([1, 1, 0]),
        read_offsets=jnp.array([1, 1, 0], dtype=jnp.int32),
    )
    self.assert_identical(actual.out, expected.out, "output rows")
    self.assert_identical(actual.conv, expected.conv, "convolution cache")
    self.assert_identical(
        actual.recurrent, expected.recurrent, "recurrent cache"
    )

  @parameterized.named_parameters(
      ("bf16_recurrent_cache", jnp.bfloat16),
      ("fp32_recurrent_cache", jnp.float32),
  )
  def test_shared_prefix_read_slot_matches_fallback(self, recurrent_dtype):
    """Check shared unwritten prefix reads match upstream for both recurrent
    cache dtypes.

    Args:
      recurrent_dtype: Recurrent-cache dtype for the parameterized fixture.
    """
    case = _build_case(
        [_decode(1, history=20), _prefill(2, 7, history=20)],
        num_decode=1,
        n_kq=2,
        n_v=8,
        slots=5,
        recurrent_state_dtype=recurrent_dtype,
    )
    # Share an unwritten checkpoint while writing distinct slots.
    result = self.assert_matches_fallback(
        case,
        read_state_indices=jnp.array([3, 3], dtype=jnp.int32),
        read_offsets=jnp.zeros((2,), dtype=jnp.int32),
    )
    self.assert_inactive_slots_preserved(case, result)

  def test_ineligible_configuration_uses_the_fallback(self):
    # BF16 compute must delegate with the caller's options intact.
    """Check ineligible compute precision delegates exactly to upstream."""
    case = self._case()
    options = dict(compute_precision=jnp.bfloat16)
    self.assertFalse(_static_eligibility(case, **options))
    reference = _run(wrapper.fused_conv1d_gdn, case, **options)
    dispatched = _run(prefill_decode.fused_conv1d_gdn, case, **options)
    self.assert_identical(dispatched.out, reference.out, "output rows")
    self.assert_identical(dispatched.conv, reference.conv, "convolution cache")
    self.assert_identical(
        dispatched.recurrent, reference.recurrent, "recurrent cache"
    )

  def test_prefix_caching_outside_the_envelope_is_delegated(self):
    """Check parity with explicit default read slots on an unsupported layout."""
    case = self._case()
    options = dict(
        compute_precision=jnp.bfloat16,
        read_state_indices=jnp.asarray(case.values["state_indices"]),
    )
    self.assertFalse(_static_eligibility(case, compute_precision=jnp.bfloat16))
    reference = _run(wrapper.fused_conv1d_gdn, case, **options)
    dispatched = _run(prefill_decode.fused_conv1d_gdn, case, **options)
    self.assert_identical(dispatched.out, reference.out, "output rows")
    self.assert_identical(dispatched.conv, reference.conv, "convolution cache")
    self.assert_identical(
        dispatched.recurrent, reference.recurrent, "recurrent cache"
    )


if __name__ == "__main__":
  absltest.main()
