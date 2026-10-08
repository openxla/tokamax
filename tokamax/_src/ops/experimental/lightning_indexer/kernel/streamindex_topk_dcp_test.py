# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Tests for DCP (decode context parallel) StreamIndex Top-K"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.lightning_indexer.kernel import (
    dcp_sc_compact,
    metadata,
)
from tokamax._src.ops.experimental.lightning_indexer.kernel.streamindex_topk import (
    DCP_AXIS_NAME,
    cp_local_to_global,
    cp_rank_as_data,
    streamindex_topk,
    streamindex_topk_dcp,
)

P = jax.sharding.PartitionSpec

H_IDX, D_IDX, K = 32, 128, 2048
Q_LEN, KV_LEN, PAGE_SIZE = 256, 32768, 1024
WIDTH = 256


class StreamIndexTopKDcpTest(parameterized.TestCase):

  def _require_sparsecore(self):
    """Probe in a test body: a chip touch at collection breaks spawned workers."""
    try:
      if pltpu.get_tpu_info().sparse_core is not None:
        return
    except Exception:  # no TPU at all
      pass
    self.skipTest("needs SparseCore")

  def _xla_pack_reference(self, vals):
    """The scatter the kernel replaced, unreachable in `src`, kept as an oracle."""
    num_rows, width = vals.shape
    keep = vals >= 0

    # Exclusive prefix count of kept entries gives the destination slot, < width.
    slot = jnp.cumsum(keep, axis=1, dtype=jnp.int32) - keep
    rows = jnp.arange(num_rows, dtype=jnp.int32)[:, None]

    # Column `width` is a dump for the dropped entries, sliced off at the end.
    out = jnp.full((num_rows, width + 1), -1, dtype=jnp.int32)
    out = out.at[rows, jnp.where(keep, slot, width)].set(
        jnp.where(keep, vals, -1)
    )
    return out[:, :width]

  def _sparse_fixture(
      self, num_rows=16, width=128, keep_rate=0.5, seed=20260906
  ):
    """Non-negative values with `-1` holes; row 0 is full and row 1 is empty."""
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 1 << 20, size=(num_rows, width)).astype(np.int32)
    vals[rng.random((num_rows, width)) > keep_rate] = -1
    vals[0] = rng.integers(0, 1 << 20, size=width).astype(np.int32)
    if num_rows > 1:
      vals[1] = -1
    return vals

  def _simulate_merge(
      self, num_rows, k, span, dcp_size, interleave_c, seed=20260929
  ):
    """Stages 1-3 of `streamindex_topk_dcp` in numpy, as `resolve` receives them.

    Returns each rank's rank-local candidate list, the merged flat slot columns,
    and the true global top-k the ranks must reassemble between them.
    """
    rng = np.random.default_rng(seed)
    # Distinct scores, so the top-k is unambiguous and the partition is exact.
    scores = (
        rng.permutation(num_rows * span)
        .reshape(num_rows, span)
        .astype(np.float32)
    )
    owner = (np.arange(span) // interleave_c) % dcp_size

    local_idxs = np.full((dcp_size, num_rows, k), -1, np.int32)
    local_scores = np.full((dcp_size, num_rows, k), -np.inf, np.float32)
    for r in range(dcp_size):
      owned = np.flatnonzero(owner == r)
      for t in range(num_rows):
        best = owned[np.argsort(-scores[t, owned])][:k]
        local_idxs[r, t, : len(best)] = np.asarray(
            metadata.cp_global_to_local(
                jnp.asarray(best), dcp_size, interleave_c
            )
        )
        local_scores[r, t, : len(best)] = scores[t, best]

    # Stage 2 is rank-major, so rank r's k candidates land at columns [r*k, r*k+k).
    gathered = np.concatenate(list(local_scores), axis=1)
    slots = np.full((num_rows, k), -1, np.int32)
    for t in range(num_rows):
      order = np.argsort(-gathered[t])[:k]
      order = order[np.isfinite(gathered[t, order])]
      slots[t, : len(order)] = order

    want = [set(np.argsort(-scores[t])[:k].tolist()) for t in range(num_rows)]
    return local_idxs, slots, want

  def _global_records(self, kv_len, seed=7):
    """One record array; both paths are views of it, so any diff is a real bug."""
    vals = jax.random.normal(jax.random.key(seed), (kv_len, D_IDX), jnp.float32)
    fp8 = jax.lax.bitcast_convert_type(
        vals.astype(jnp.float8_e4m3fn), jnp.uint8
    ).reshape(kv_len, D_IDX)
    # 127 is the e8m0 exponent bias, i.e. a scale of exactly 1.0.
    rec = jnp.concatenate([fp8, jnp.full((kv_len, 1), 127, jnp.uint8)], -1)
    return jnp.pad(rec, ((0, 0), (0, WIDTH - rec.shape[-1])))

  def _need_devices(self, dcp_size):
    self._require_sparsecore()
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest(
          "StreamIndex Top-K Pallas TPU kernel requires TPU v7 or newer."
      )
    n = len(jax.devices())
    if n < dcp_size:
      self.skipTest(f"needs {dcp_size} devices, have {n}")

  def _dcp_case(self, dcp_size, seed):
    """Records plus the positional args and kwargs of one sharded DCP call."""
    rec = self._global_records(KV_LEN)
    kq, kw = jax.random.split(jax.random.key(seed))
    q = (
        jax.random.normal(kq, (Q_LEN, H_IDX, D_IDX), jnp.float32) * 0.5
    ).astype(jnp.float8_e4m3fn)
    weights = jax.random.normal(kw, (Q_LEN, H_IDX), jnp.float32).astype(
        jnp.bfloat16
    )

    # interleave_size=1: global position g lives on rank g % dcp_size at local
    # index g // dcp_size, so the shards are a strided de-interleave.
    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    shards = jnp.concatenate(
        [
            rec[r::dcp_size].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
            for r in range(dcp_size)
        ],
        0,
    )
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:dcp_size]), (DCP_AXIS_NAME,)
    )
    shards = jax.device_put(
        shards, jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME))
    )

    args = (
        q,
        weights,
        shards,
        jnp.array([KV_LEN], jnp.int32),
        jnp.arange(vpages, dtype=jnp.int32),
        jnp.array([0, Q_LEN], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
    )
    kwargs = dict(
        mesh=mesh,
        k=K,
        compression_ratio=1,
        dcp_size=dcp_size,
        interleave_size=1,
        num_kv_pages_per_block=2,
        num_queries_per_block=128,
    )
    return rec, args, kwargs

  @parameterized.parameters([2, 4, 8])
  def test_candidate_all_gather_is_rank_major(self, dcp_size):
    """The merge resolves a winner as rank `slot // k`, position `slot % k`.

    Both readings assume this layout, and nothing else in the merge would
    notice it changing: every rank would resolve the wrong candidate and still
    return a full, plausible, silently wrong list. Asserting the full
    rank-major arange pins order and chunk width together.
    """
    devices = jax.devices()
    if len(devices) < dcp_size:
      self.skipTest(f"needs {dcp_size} devices, have {len(devices)}")

    rows, k = 4, 8
    mesh = jax.sharding.Mesh(np.array(devices[:dcp_size]), (DCP_AXIS_NAME,))

    def _local(_):
      rank = cp_rank_as_data(DCP_AXIS_NAME, dcp_size)
      # Rank r writes r*k + j, so a rank-major gather reads back as arange.
      mine = rank * k + jnp.arange(k, dtype=jnp.int32)
      return jax.lax.all_gather(
          jnp.broadcast_to(mine, (rows, k)), DCP_AXIS_NAME, axis=1, tiled=True
      )

    gathered = jax.jit(
        jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=P(DCP_AXIS_NAME),
            check_vma=False,
        )
    )(jnp.zeros((rows, 1), jnp.int32))

    got = np.asarray(gathered).reshape(dcp_size, rows, dcp_size * k)
    want = np.broadcast_to(
        np.arange(dcp_size * k, dtype=np.int32), (rows, dcp_size * k)
    )
    for r in range(dcp_size):
      np.testing.assert_array_equal(
          got[r], want, err_msg=f"rank {r} saw a non-rank-major candidate axis"
      )

  @parameterized.product(
      [
          {"num_rows": 16, "width": 128},
          {"num_rows": 1, "width": 16},
          {"num_rows": 33, "width": 256},
      ],
      keep_rate=[0.1, 0.5, 0.9],
  )
  def test_sparsecore_pack_matches_xla(self, num_rows, width, keep_rate):
    """Elementwise, not just as a set: both packs must emit in ascending order."""
    self._require_sparsecore()
    vals = self._sparse_fixture(num_rows, width, keep_rate)

    want = np.asarray(self._xla_pack_reference(jnp.asarray(vals)))
    got = np.asarray(dcp_sc_compact.pack_nonnegative(jnp.asarray(vals)))

    np.testing.assert_array_equal(got, want)

  @parameterized.parameters((2, 1), (4, 4), (8, 4), (8, 64))
  def test_resolve_partitions_the_global_topk(self, dcp_size, interleave_c):
    """The ranks' packed outputs reassemble the global top-k, each entry once."""
    self._require_sparsecore()
    num_rows, k, span = 8, 32, 4096
    local_idxs, slots, want = self._simulate_merge(
        num_rows, k, span, dcp_size, interleave_c
    )

    lists = np.stack([
        np.asarray(
            dcp_sc_compact.resolve_owned_winners(
                jnp.asarray(local_idxs[r]), jnp.asarray(slots), k, jnp.int32(r)
            )
        )
        for r in range(dcp_size)
    ])
    assert lists.shape == (dcp_size, num_rows, k)

    for t in range(num_rows):
      got = []
      for r in range(dcp_size):
        row = lists[r, t]
        for local in row[row >= 0].tolist():
          g = int(cp_local_to_global(local, r, dcp_size, interleave_c))
          # A rank must only ever keep positions it actually owns.
          assert (g // interleave_c) % dcp_size == r
          got.append(g)
      # Multiplicity, not just set equality: the LSE merge double-counts a
      # position delivered to two ranks.
      assert len(got) == len(set(got))
      assert set(got) == want[t]

  # ===================================================================== #
  # End-to-end against the unsharded kernel.
  # ===================================================================== #

  @parameterized.parameters([2, 4, 8])
  def test_dcp_reassembles_the_unsharded_global_topk(self, dcp_size):
    """Union the per-rank lists back together; it must be the global top-k"""
    self._need_devices(dcp_size)
    rec, args, kwargs = self._dcp_case(dcp_size, 0)
    q, weights = args[0], args[1]

    pages = KV_LEN // PAGE_SIZE
    reference = np.asarray(
        streamindex_topk(
            q,
            weights,
            rec.reshape(pages, PAGE_SIZE // 4, 4, WIDTH),
            jnp.array([KV_LEN], jnp.int32),
            jnp.arange(pages, dtype=jnp.int32),
            jnp.array([0, Q_LEN], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
            k=K,
            compression_ratio=1,
            num_kv_pages_per_block=2,
            num_queries_per_block=128,
        )
    )

    # Global leading dim is dcp_size * Q_LEN, rank-major.
    local_topk = np.asarray(streamindex_topk_dcp(*args, **kwargs)).reshape(
        dcp_size, Q_LEN, K
    )
    assert local_topk.max() < KV_LEN // dcp_size, "a rank got a foreign index"

    for t in range(Q_LEN):
      got = set()
      for r in range(dcp_size):
        row = local_topk[r, t]
        for local in row[row >= 0].tolist():
          got.add(int(cp_local_to_global(local, r, dcp_size, 1)))
      want = set(reference[t][reference[t] >= 0].tolist())
      assert (
          got == want
      ), f"token {t}: {len(want - got)} missing, {len(got - want)} spurious"

  @parameterized.product(
      dcp_size=[2, 4, 8],
      chunk_tokens=[32, 64, 128],
  )
  def test_dcp_chunking_does_not_change_the_result(
      self, chunk_tokens, dcp_size
  ):
    self._need_devices(dcp_size)
    _, args, kwargs = self._dcp_case(dcp_size, 11)

    hlo = streamindex_topk_dcp.lower(
        *args, **kwargs, chunk_tokens=chunk_tokens
    ).as_text()
    assert "_scheduling_group_id" in hlo, (
        "no scheduling group annotation: the DCP path ran a single unchunked"
        " pass, so this comparison would pass trivially"
    )

    unchunked = np.asarray(streamindex_topk_dcp(*args, **kwargs))
    chunked = np.asarray(
        streamindex_topk_dcp(*args, **kwargs, chunk_tokens=chunk_tokens)
    )
    np.testing.assert_array_equal(chunked, unchunked)


if __name__ == "__main__":
  absltest.main()
