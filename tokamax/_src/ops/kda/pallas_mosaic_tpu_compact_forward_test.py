# Copyright 2026 Ant Group. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original-token forward correctness, metadata, and compact tape contract."""

import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.kda import api, compact
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as shared

interpret_on_cpu = shared.interpret_on_cpu


def operands(case, *, heads=2, batch=1, key_dim=128, value_dim=128):
  tokens = 128
  patterns = {
      "full": [1] * 128,
      "two": [1] * 17 + [2] * 111,
      "many": list(range(1, 129)),
      "reused": [7] * 16 + [8] * 16 + [7] * 96,
      "holes": [1] * 16 + [0] * 8 + [2] * 16 + [1] * 24 + [0] * 64,
      "padding": [1] * 60 + [0] * 68,
      "empty": [0] * 128,
  }
  ids = jnp.asarray([patterns[case]] * batch, jnp.int32)
  if batch == 2:
    ids = ids.at[1].set(jnp.asarray([1] * 65 + [2] * 31 + [0] * 32, jnp.int32))
  keys = jax.random.split(jax.random.key(619), 5)
  q = (
      jax.random.normal(keys[0], (heads, batch, tokens, key_dim)) * 0.05
  ).astype(jnp.bfloat16)
  k = (jax.random.normal(keys[1], q.shape) * 0.05).astype(q.dtype)
  v = (
      jax.random.normal(keys[2], (heads, batch, tokens, value_dim)) * 0.05
  ).astype(q.dtype)
  g = jnp.full(q.shape, -0.1, jnp.float32)
  beta = jnp.full(q.shape[:-1], 0.25, q.dtype)
  return (q, k, v, g, beta), ids


def assert_reference(actual, reference, *, relative_rms_limit=0.02):
  a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
  assert np.isfinite(a).all()
  norm = np.linalg.norm(b)
  if norm > 1e-8:
    error = np.linalg.norm(a - b) / norm
    assert error <= relative_rms_limit, f"relative RMS {error}"
  else:
    # Zero-reference derivatives need an absolute roundoff bound.
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
  np.testing.assert_allclose(a, b, rtol=0.05, atol=1e-4)


@pytest.mark.parametrize(
    "case,sorted_segments",
    [
        (c, False)
        for c in ["full", "two", "many", "reused", "holes", "padding", "empty"]
    ]
    + [(c, True) for c in ["full", "two", "many", "padding", "empty"]],
)
def test_forward_and_compact_tape(case, sorted_segments):
  xs, ids = operands(case)
  kwargs = dict(segment_ids=ids, max_num_segments=262144)
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(
          compact_forward=True, compact_sorted_segments=sorted_segments
      )
  )
  output, residuals = op(*xs, return_residuals=True, **kwargs)
  expected = api.kimi_delta_attention(
      *xs, implementation="xla", segment_ids=ids, max_num_segments=128
  )
  assert_reference(output[0], expected[0])
  assert isinstance(residuals, compact.CompactResiduals)
  assert residuals.aqk.shape == (2, 1, 128, 64)
  assert residuals.h.shape == (2, 1, 2, 128, 128)
  assert residuals.cu_seqlens.shape == (1, 129)
  assert output[1] is None
  invalid = np.asarray(ids) == 0
  np.testing.assert_array_equal(np.asarray(output[0])[:, invalid], 0)


def test_multi_batch_and_unequal_widths():
  xs, ids = operands("two", batch=2, key_dim=128, value_dim=256)
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(compact_forward=True)
  )
  output = op(*xs, segment_ids=ids, max_num_segments=2)
  expected = api.kimi_delta_attention(
      *xs, segment_ids=ids, max_num_segments=2, implementation="xla"
  )
  assert_reference(output[0], expected[0])


def test_fast_path_bypasses_aligned_preprocessing(monkeypatch):
  xs, ids = operands("two")

  def unexpected(*args, **kwargs):
    raise AssertionError("compact path must not align token tensors")

  monkeypatch.setattr(
      mosaic.PallasMosaicTpuKimiDeltaAttention, "_preprocess_inputs", unexpected
  )
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(compact_forward=True)
  )
  output = op(*xs, segment_ids=ids, max_num_segments=2)
  assert np.isfinite(np.asarray(output[0])).all()


def test_interleaved_labels_are_grouped_stably():
  ids = jnp.asarray([[7, 7, 8, 7, 7, 0, 7, 0]], jnp.int32)
  actual, permutation, inverse, ordered = compact._packing(ids)
  assert not bool(ordered)
  np.testing.assert_array_equal(actual, [[1, 1, 1, 1, 1, 2, 0, 0]])
  np.testing.assert_array_equal(permutation, [[0, 1, 3, 4, 6, 2, 5, 7]])
  np.testing.assert_array_equal(
      jnp.take_along_axis(permutation, inverse, axis=1), jnp.arange(8)[None]
  )


def test_sorted_contract_has_no_permutation_tape():
  xs, ids = operands("two")
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(compact_forward=True, compact_sorted_segments=True)
  )
  _, r = op(*xs, segment_ids=ids, max_num_segments=2, return_residuals=True)
  assert (
      r.permutation is None
      and r.inverse_permutation is None
      and r.ordered is None
  )
  assert r.cu_seqlens.shape == (1, 3)


@pytest.mark.parametrize("ids", [[1, 2, 1, 0], [1, 0, 2, 0], [1, 3, 0, 0]])
def test_sorted_contract_rejects_invalid_metadata(ids):
  checked = checkify.checkify(
      lambda values: compact._check_sorted_segments(values, 2)
  )
  error, _ = jax.jit(checked)(jnp.asarray([ids], jnp.int32))
  with pytest.raises(checkify.JaxRuntimeError, match="grouped positive labels"):
    error.throw()


def test_largest_label_still_sorts_before_padding():
  label = jnp.iinfo(jnp.int32).max
  ids = jnp.asarray([[label, 0, label]], jnp.int32)
  actual, permutation, _, _ = compact._packing(ids)
  np.testing.assert_array_equal(permutation, [[0, 2, 1]])
  np.testing.assert_array_equal(actual, [[1, 1, 0]])


@pytest.mark.parametrize(
    "option", [{"rematerialize_for_backward": True}, {"safe_gate": False}]
)
def test_unsupported_profile_keeps_aligned_path(option):
  xs, ids = operands("two")
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(compact_forward=True, **option)
  )
  _, residuals = op(
      *xs, segment_ids=ids, max_num_segments=2, return_residuals=True
  )
  assert not isinstance(residuals, compact.CompactResiduals)
