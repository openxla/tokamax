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
"""Tests for the MLA v3 Tokamax op."""

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import jaxtyping
import numpy as np
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.experimental.mla import api
from tokamax._src.ops.experimental.mla import arg_specs
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla import pallas_mosaic_tpu_v3
from tokamax._src.ops.experimental.mla import test_base
from tokamax._src.ops.experimental.mla.v3 import bref_override
from tokamax._src.ops.experimental.mla.v3 import configs as v3_configs
from tokamax._src.ops.experimental.mla.v3 import utils as v3_utils

jax.config.parse_flags_with_absl()

Config = pallas_mosaic_tpu_v3.Config
MlaV3 = pallas_mosaic_tpu_v3.PallasTpuMultiHeadLatentAttentionV3
HEAD_MAJOR = base.MlaLayout.HEAD_MAJOR_SEQ_ALONG_LANE

_NUM_HEADS = 128
_LKV_DIM = 512
_R_DIM = 64

# One decode-only and one mixed batch, both exercising partial pages.
_DECODE_SEQ_LENS = ((1, 129), (1, 300), (1, 1107), (1, 18), (1, 2049))
_MIXED_SEQ_LENS = ((1, 129), (1, 1107), (5, 597), (31, 1328), (3, 18))


def _inputs(
    seq_lens: Sequence[tuple[int, int]],
    page_size: int,
    dtype: Any,
    num_pages: int = 64,
) -> dict[str, Any]:
  """Generates head-major / lanes-major inputs for the v3 op."""
  (
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
  ) = test_base.generate_mla_inputs(
      seq_lens,
      _NUM_HEADS,
      _LKV_DIM,
      _R_DIM,
      page_size,
      dtype,
      dtype,
      num_pages,
      rng=0,
  )
  return dict(
      ql_nope=jnp.swapaxes(ql_nope, 0, 1),
      q_pe=q_pe,
      new_kv_c=new_kv_c,
      new_k_pe=new_k_pe,
      cache_kv=v3_utils.transpose_kv_cache_to_seq_along_lane(cache_kv),
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
      layout=HEAD_MAJOR,
  )


def _valid_kv_mask(inputs: dict[str, Any]) -> np.ndarray:
  """Mask of the cache entries that hold a real token after the update.

  The kernel writes back whole lane tiles, so slots past a sequence's end may
  differ from the reference; only real tokens are compared.
  """
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


class PallasTpuMultiHeadLatentAttentionV3Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if (
        backend.get_default_device().platform != "tpu"
        or pltpu.get_tpu_info().generation < 7
    ):
      self.skipTest("Only supported on TPU v7+.")
    if not bref_override.SUPPORTS_IN_OUT_BUFFER_COUNT:
      self.skipTest("Requires (in, out) buffer_count support in Pallas.")

  @parameterized.product(
      seq_lens=(_DECODE_SEQ_LENS, _MIXED_SEQ_LENS),
      page_size=(256, 1024),
      dtype=(jnp.bfloat16, jnp.float8_e4m3fn),
  )
  def test_matches_reference(self, seq_lens, page_size, dtype):
    inputs = _inputs(seq_lens, page_size, dtype)

    out, updated_kv = MlaV3()(**inputs)
    ref_out, ref_updated_kv = base.MultiHeadLatentAttention()(
        **inputs, s_dtype=jnp.float32
    )

    num_tokens = sum(q_len for q_len, _ in seq_lens)
    self.assertEqual(out.shape, (_NUM_HEADS, num_tokens, _LKV_DIM))
    self.assertEqual(updated_kv.shape, inputs["cache_kv"].shape)
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

  def test_precomputed_schedules_match_inline(self):
    inputs = _inputs(_MIXED_SEQ_LENS, 1024, jnp.bfloat16)
    op = MlaV3()

    schedules = op.build_schedules(**inputs)
    self.assertSameElements(
        schedules.keys(),
        (
            v3_configs.MlaCase.DECODE,
            v3_configs.MlaCase.PREFILL,
            v3_configs.MlaCase.MIXED,
        ),
    )
    out, updated_kv = op(**inputs, schedules=schedules)
    expected_out, expected_kv = op(**inputs)

    np.testing.assert_array_equal(out, expected_out)
    np.testing.assert_array_equal(updated_kv, expected_kv)

  def test_schedules_for_another_config_are_rejected(self):
    inputs = _inputs(_DECODE_SEQ_LENS, 1024, jnp.bfloat16)
    schedules = MlaV3(config=Config(decode_batch_size=2)).build_schedules(
        **inputs
    )
    with self.assertRaisesRegex(ValueError, "different configs"):
      MlaV3(config=Config(decode_batch_size=4))(**inputs, schedules=schedules)

  def test_token_major_layout_is_rejected(self):
    inputs = _inputs(_DECODE_SEQ_LENS, 1024, jnp.bfloat16)
    inputs["ql_nope"] = jnp.swapaxes(inputs["ql_nope"], 0, 1)
    inputs["cache_kv"] = v3_utils.transpose_kv_cache_from_seq_along_lane(
        inputs["cache_kv"]
    )
    inputs["layout"] = base.MlaLayout.TOKEN_MAJOR
    # `ql_nope` and `q_pe` then disagree on the head and token dims.
    with self.assertRaises(jaxtyping.TypeCheckError):
      MlaV3()(**inputs)

  @parameterized.parameters(
      # (tokens, page_size, decode_pages, decode_batch, prefill_pages)
      (4, 1024, 2, 2, 1),
      (8, 1024, 2, 4, 1),
      (64, 1024, 2, 4, 1),
      (8, 256, 8, 4, 4),
  )
  def test_heuristics_config(
      self, num_tokens, page_size, decode_pages, decode_batch, prefill_pages
  ):
    ba = MlaV3().bind(
        ql_nope=jax.ShapeDtypeStruct(
            (_NUM_HEADS, num_tokens, _LKV_DIM), jnp.float8_e4m3fn
        ),
        q_pe=jax.ShapeDtypeStruct(
            (num_tokens, _NUM_HEADS, _R_DIM), jnp.float8_e4m3fn
        ),
        new_kv_c=jax.ShapeDtypeStruct(
            (num_tokens, _LKV_DIM), jnp.float8_e4m3fn
        ),
        new_k_pe=jax.ShapeDtypeStruct((num_tokens, _R_DIM), jnp.float8_e4m3fn),
        cache_kv=jax.ShapeDtypeStruct((16, 640, page_size), jnp.float8_e4m3fn),
        kv_lens=jax.ShapeDtypeStruct((32,), jnp.int32),
        page_indices=jax.ShapeDtypeStruct((288,), jnp.int32),
        cu_q_lens=jax.ShapeDtypeStruct((33,), jnp.int32),
        distribution=jax.ShapeDtypeStruct((3,), jnp.int32),
        layout=HEAD_MAJOR,
    )
    self.assertEqual(
        ba.heuristics_config,
        Config(
            decode_num_kv_pages_per_block=decode_pages,
            decode_batch_size=decode_batch,
            prefill_num_kv_pages_per_block=prefill_pages,
        ),
    )

  def test_config_validation(self):
    with self.assertRaisesRegex(ValueError, "must divide"):
      Config(prefill_num_queries_per_block=16, prefill_q_split=3)

  def test_registered_in_api(self):
    self.assertIsInstance(api.IMPLEMENTATIONS["mosaic_tpu_v3"], MlaV3)


def _v3_spec(name: str) -> arg_spec.ArgSpec:
  (spec,) = (s for s in arg_specs.V3_ARG_SPECS if s.name == name)
  return spec


class AutotuningTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if (
        backend.get_default_device().platform != "tpu"
        or pltpu.get_tpu_info().generation < 7
    ):
      self.skipTest("Only supported on TPU v7+.")
    if not bref_override.SUPPORTS_IN_OUT_BUFFER_COUNT:
      self.skipTest("Requires (in, out) buffer_count support in Pallas.")

  def test_cache_key_ignores_page_count_and_scales(self):
    args = dict(_v3_spec("mla_v3_decode_rr05_t8").args)
    key = MlaV3().bind(**args).autotuning_cache_key

    cache_kv = args["cache_kv"]
    args["cache_kv"] = jax.ShapeDtypeStruct(
        (12345, *cache_kv.shape[1:]), cache_kv.dtype
    )
    args.update(sm_scale=0.1, q_scale=0.5, k_scale=None, v_scale=2.0)
    self.assertEqual(MlaV3().bind(**args).autotuning_cache_key, key)

    other = MlaV3().bind(**_v3_spec("mla_v3_decode_rr05_t32").args)
    self.assertNotEqual(other.autotuning_cache_key, key)

  def test_cache_key_ignores_schedules(self):
    inputs = _inputs(_DECODE_SEQ_LENS, 1024, jnp.bfloat16)
    op = MlaV3()
    schedules = op.build_schedules(**inputs)
    self.assertEqual(
        op.bind(**inputs, schedules=schedules).autotuning_cache_key,
        op.bind(**inputs).autotuning_cache_key,
    )

  @parameterized.parameters("mla_v3_decode_rr05_t8", "mla_v3_mixed_rr05_t1024")
  def test_autotuning_configs(self, name):
    ba = MlaV3().bind(**_v3_spec(name).args)
    heuristic = ba.heuristics_config
    configs = ba.op._get_autotuning_configs(ba)  # pylint: disable=protected-access

    # Both passes are varied for every shape, whatever the batch's bands.
    self.assertGreater(len({c.decode_batch_size for c in configs}), 1)
    self.assertGreater(
        len({c.prefill_num_queries_per_block for c in configs}), 1
    )
    # KV blocks need not divide the sequence length in pages.
    self.assertIn(3, {c.decode_num_kv_pages_per_block for c in configs})
    self.assertIn(3, {c.prefill_num_kv_pages_per_block for c in configs})
    for c in configs:
      self.assertEqual(c.prefill_num_queries_per_block % c.prefill_q_split, 0)
      # Each sweep varies one pass and leaves the other at the heuristic.
      decode_at_heuristic = (
          c.decode_num_kv_pages_per_block,
          c.decode_batch_size,
          c.decode_n_buffer,
      ) == (
          heuristic.decode_num_kv_pages_per_block,
          heuristic.decode_batch_size,
          heuristic.decode_n_buffer,
      )
      prefill_at_heuristic = (
          c.prefill_num_kv_pages_per_block,
          c.prefill_num_queries_per_block,
          c.prefill_q_split,
      ) == (
          heuristic.prefill_num_kv_pages_per_block,
          heuristic.prefill_num_queries_per_block,
          heuristic.prefill_q_split,
      )
      self.assertTrue(decode_at_heuristic or prefill_at_heuristic)
      # Buffering depth is only swept for the decode pass.
      self.assertEqual(c.prefill_n_buffer, heuristic.prefill_n_buffer)
      if prefill_at_heuristic:
        self.assertIn(c.decode_n_buffer, (2, 3, 4))
      else:
        self.assertEqual(c.decode_n_buffer, heuristic.decode_n_buffer)

  def test_production_specs_hit_the_cache(self):
    device_kind = jax.devices()[0].device_kind.lower().replace(" ", "_")
    if device_kind != "tpu7x":
      self.skipTest("The cache is only populated for tpu7x.")
    for spec in arg_specs.V3_ARG_SPECS:
      with self.subTest(spec.name):
        self.assertIsNotNone(MlaV3().bind(**spec.args).cached_autotuning_data)


if __name__ == "__main__":
  absltest.main()
