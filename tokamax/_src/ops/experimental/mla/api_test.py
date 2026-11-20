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
"""Tests for the MLA API."""

import functools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.mla import api
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla import test_base

_NUM_HEADS = 128
_LKV_DIM = 512
_R_DIM = 64
_SM_SCALE = (_LKV_DIM + _R_DIM) ** -0.5
_SEQ_LENS = ((1, 129), (3, 18))


def _make_inputs() -> dict[str, Any]:
  names = (
      "ql_nope",
      "q_pe",
      "new_kv_c",
      "new_k_pe",
      "cache_kv",
      "kv_lens",
      "page_indices",
      "cu_q_lens",
      "distribution",
  )
  values = test_base.generate_mla_inputs(
      _SEQ_LENS,
      _NUM_HEADS,
      _LKV_DIM,
      _R_DIM,
      page_size=256,
      q_dtype=jnp.bfloat16,
      kv_dtype=jnp.bfloat16,
      num_pages=16,
      rng=0,
  )
  return dict(zip(names, values, strict=True))


def _valid_kv_mask(inputs: dict[str, Any]) -> np.ndarray:
  cache_kv = inputs["cache_kv"]
  kv_lens = np.asarray(inputs["kv_lens"])
  page_indices = np.asarray(inputs["page_indices"])
  page_size = cache_kv.shape[-1]
  pages_per_seq = page_indices.shape[0] // kv_lens.shape[0]
  mask = np.zeros(cache_kv.shape, dtype=bool)
  for i, kv_len in enumerate(kv_lens):
    for t in range(kv_len):
      page = page_indices[i * pages_per_seq + t // page_size]
      mask[page, :, t % page_size] = True
  return mask


def _mosaic_supported() -> bool:
  mosaic_impl = api.IMPLEMENTATIONS.get("mosaic_tpu")
  return mosaic_impl is not None and mosaic_impl.supported_on(
      backend.get_default_device()
  )


class ApiTest(parameterized.TestCase):

  def test_default_implementation(self):
    inputs = _make_inputs()
    fn = jax.jit(
        functools.partial(api.multi_head_latent_attention, sm_scale=_SM_SCALE)
    )

    out, updated_kv = fn(**inputs)
    ref_out, ref_updated_kv = base.MultiHeadLatentAttention()(
        **inputs, sm_scale=_SM_SCALE, s_dtype=jnp.float32
    )

    np.testing.assert_allclose(
        np.asarray(out, np.float32),
        np.asarray(ref_out, np.float32),
        atol=0.1,
        rtol=0.2,
    )
    mask = _valid_kv_mask(inputs)
    np.testing.assert_array_equal(
        np.asarray(updated_kv)[mask], np.asarray(ref_updated_kv)[mask]
    )

    opspecs = hlo_utils.get_opspecs(
        fn.lower(**inputs), include_xla_kernels=False
    )
    if _mosaic_supported():
      self.assertNotEmpty(opspecs)
      for opspec in opspecs:
        self.assertIsInstance(
            opspec.op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )
    else:
      self.assertEmpty(opspecs)

  @parameterized.parameters("xla", "mosaic", "mosaic_tpu")
  def test_explicit_implementation(self, impl: api.Implementation):
    inputs = _make_inputs()
    fn = functools.partial(
        api.multi_head_latent_attention,
        sm_scale=_SM_SCALE,
        implementation=impl,
    )

    if impl in ("mosaic", "mosaic_tpu"):
      if "mosaic_tpu" not in api.IMPLEMENTATIONS:
        with self.assertRaisesRegex(ValueError, "Unknown implementation"):
          fn(**inputs)
        return
      if not _mosaic_supported():
        with self.assertRaises(NotImplementedError):
          fn(**inputs)
        return

    jitted_fn = jax.jit(fn)
    out, updated_kv = jitted_fn(**inputs)
    ref_out, ref_updated_kv = base.MultiHeadLatentAttention()(
        **inputs, sm_scale=_SM_SCALE, s_dtype=jnp.float32
    )

    np.testing.assert_allclose(
        np.asarray(out, np.float32),
        np.asarray(ref_out, np.float32),
        atol=0.1,
        rtol=0.2,
    )
    mask = _valid_kv_mask(inputs)
    np.testing.assert_array_equal(
        np.asarray(updated_kv)[mask], np.asarray(ref_updated_kv)[mask]
    )

    opspecs = hlo_utils.get_opspecs(
        jitted_fn.lower(**inputs), include_xla_kernels=False
    )
    if impl == "xla":
      self.assertEmpty(opspecs)
    else:
      self.assertNotEmpty(opspecs)
      for opspec in opspecs:
        self.assertIsInstance(
            opspec.op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )

  def test_callable_and_sequence_fallback(self):
    inputs = _make_inputs()
    calls: list[str] = []

    def unsupported_impl(*args, **kwargs):
      del args, kwargs
      calls.append("unsupported")
      raise NotImplementedError("unsupported backend")

    def custom_impl(ql_nope, *args, **kwargs):
      del args, kwargs
      calls.append("custom")
      return jnp.zeros_like(ql_nope), inputs["cache_kv"]

    out, _ = api.multi_head_latent_attention(
        **inputs, implementation=custom_impl
    )
    self.assertEqual(calls, ["custom"])
    np.testing.assert_array_equal(
        np.asarray(out), np.zeros(inputs["ql_nope"].shape, dtype=jnp.bfloat16)
    )

    calls.clear()
    fallback_out, _ = api.multi_head_latent_attention(
        **inputs, implementation=(unsupported_impl, custom_impl)
    )
    self.assertEqual(calls, ["unsupported", "custom"])
    np.testing.assert_array_equal(
        np.asarray(fallback_out),
        np.zeros(inputs["ql_nope"].shape, dtype=jnp.bfloat16),
    )

    with self.assertRaises(ExceptionGroup) as ctx:
      api.multi_head_latent_attention(
          **inputs, implementation=(unsupported_impl, unsupported_impl)
      )
    self.assertLen(ctx.exception.exceptions, 2)

  def test_invalid_implementation_raises(self):
    inputs = _make_inputs()

    with self.assertRaisesRegex(ValueError, "empty sequence"):
      api.multi_head_latent_attention(**inputs, implementation=())

    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.multi_head_latent_attention(
          **inputs, implementation="nonexistent"  # pyrefly: ignore[bad-argument-type]
      )


if __name__ == "__main__":
  absltest.main()
