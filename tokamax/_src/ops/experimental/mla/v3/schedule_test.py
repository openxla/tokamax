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
"""Unit tests for the Batched MLA v3 metadata scheduler."""

from collections.abc import Sequence
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import schedule


@dataclasses.dataclass(frozen=True)
class DmaQ:
  hbm_off: int
  sz: int


@dataclasses.dataclass(frozen=True)
class DmaCache:
  phys_page: int
  sz: int


@dataclasses.dataclass(frozen=True)
class DmaNewFetch:
  hbm_off: int
  vmem_off: int
  sz: int


@dataclasses.dataclass(frozen=True)
class DmaNewWb:
  phys_page: int
  page_off: int
  vmem_off: int
  sz: int


@dataclasses.dataclass(frozen=True)
class LaneSchedule:
  """Unpacked view of one (step, lane) schedule entry."""

  s_idx: int
  q_idx: int
  k_idx: int
  is_last_k: int
  do_writeback: int
  dma_q: DmaQ
  dma_kv_cache: list[DmaCache]
  dma_kv_new_fetch: DmaNewFetch
  dma_kv_new_wb: list[DmaNewWb]


@dataclasses.dataclass(frozen=True)
class StepWaits:
  """Unpacked DMA wait counters for a single schedule step."""

  kv_in: int
  kv_out: int
  q_in: int
  o_out: int


class MlaScheduleTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()
    self.model_cfg = configs.MlaModelConfigs(
        num_q_heads=128,
        lkv_dim=512,
        r_dim=64,
        mask_value=-1e9,
    )

  def _make_schedule(
      self,
      seq_lens: Sequence[tuple[int, int]],
      page_indices: Sequence[int],
      *,
      mode: configs.MlaCase,
      page_size: int = 256,
      bq_sz: int = 1,
      bkv_sz: int = 512,
      batch_size: int = 2,
      distribution: tuple[int, int, int] | None = None,
      smem_fraction_limit: float = 0.33,
  ) -> schedule.MlaSchedule:
    num_seqs = len(seq_lens)
    q_lens = [q for q, _ in seq_lens]
    kv_lens = [kv for _, kv in seq_lens]
    cu_q_lens = [0]
    for q in q_lens:
      cu_q_lens.append(cu_q_lens[-1] + q)

    if distribution is None:
      if mode == configs.MlaCase.DECODE:
        distribution = (num_seqs, num_seqs, num_seqs)
      elif mode == configs.MlaCase.PREFILL:
        distribution = (0, num_seqs, num_seqs)
      else:
        distribution = (0, 0, num_seqs)

    mla_cfg = configs.MlaConfigs(
        block=configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=bq_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=2,
        ),
        model=self.model_cfg,
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=page_size,
            total_q_tokens=cu_q_lens[-1],
            num_page_indices=len(page_indices),
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            smem_fraction_limit_for_schedule_generation=smem_fraction_limit,
        ),
        mode=mode,
        vmem_limit_bytes=16 * 1024 * 1024,
    )
    return schedule.generate_mla_metadata(
        jnp.asarray(cu_q_lens, dtype=jnp.int32),
        jnp.asarray(kv_lens, dtype=jnp.int32),
        jnp.asarray(page_indices, dtype=jnp.int32),
        jnp.asarray(distribution, dtype=jnp.int32),
        mla_cfg,
    )

  def _get_lane(
      self, sched: schedule.MlaSchedule, step: int, lane: int
  ) -> LaneSchedule:
    cfgs = sched.cfgs
    step_arr = jnp.int32(step)
    q_hbm, q_sz = sched.get_dma_q(step_arr, lane)
    cache_dmas = []
    for i in range(cfgs.bkv_p):
      p_idx, sz = sched.get_dma_kv_cache(step_arr, lane, i)
      cache_dmas.append(DmaCache(phys_page=int(p_idx), sz=int(sz)))

    fetch_hbm, fetch_vmem, fetch_sz = sched.get_dma_kv_new_fetch(step_arr, lane)
    wb_dmas = []
    for i in range(cfgs.bkv_p_wb):
      wb_hbm, wb_vmem, wb_sz = sched.get_dma_kv_new_wb(step_arr, lane, i)
      wb_hbm = int(wb_hbm)
      wb_dmas.append(
          DmaNewWb(
              phys_page=wb_hbm >> cfgs.serve.page_size_log2,
              page_off=wb_hbm & cfgs.serve.page_size_mask,
              vmem_off=int(wb_vmem),
              sz=int(wb_sz),
          )
      )

    return LaneSchedule(
        s_idx=int(sched.s_idx[step, lane]),
        q_idx=int(sched.q_idx[step, lane]),
        k_idx=int(sched.k_idx[step, lane]),
        is_last_k=int(sched.is_last_k[step, lane]),
        do_writeback=int(sched.do_writeback[step, lane]),
        dma_q=DmaQ(hbm_off=int(q_hbm), sz=int(q_sz)),
        dma_kv_cache=cache_dmas,
        dma_kv_new_fetch=DmaNewFetch(
            hbm_off=int(fetch_hbm), vmem_off=int(fetch_vmem), sz=int(fetch_sz)
        ),
        dma_kv_new_wb=wb_dmas,
    )

  def _get_waits(self, sched: schedule.MlaSchedule, step: int) -> StepWaits:
    return StepWaits(
        kv_in=int(sched.total_wait_kv_in[step]),
        kv_out=int(sched.total_wait_kv_out[step]),
        q_in=int(sched.total_wait_q_in[step]),
        o_out=int(sched.total_wait_o_out[step]),
    )

  def test_decode_schedule(self):
    """Verifies decode lane packing, tile-rounded DMAs, waits, and masking.

    Setup (page_size=256, bkv_sz=512 -> 2 pages/block, batch_size=2):
      - Seq 0 (q_len=1, kv_len=300): 299 cached + 1 new -> 1 K-block (k_idx=0).
        Page 0 is full (256), Page 1 holds 43 cached tokens (rounds to 128).
        New token lands at seq_tok_idx=299 (page 1, offset 43 -> tile [0, 128)).
      - Seq 1 (q_len=1, kv_len=681): 680 cached + 1 new -> 2 K-blocks:
        * k_idx=0 ([0, 512)): pure cache (512 cached, new_sz=0, no writeback).
        * k_idx=1 ([512, 1024)): 168 cached (rounds to 256) + 1 new token at
          seq_tok_idx=680 (page 2, offset 168 -> tile [128, 256)).
      - 3 total tasks across 2 lanes -> 2 steps (step 1 lane 1 is masked out).
    """
    sched = self._make_schedule(
        seq_lens=[(1, 300), (1, 681)],
        page_indices=[7, 9, -1, 10, 11, 12],
        mode=configs.MlaCase.DECODE,
        page_size=256,
        bq_sz=1,
        bkv_sz=512,
        batch_size=2,
    )
    self.assertEqual(int(sched.actual_steps[0]), 2)

    # Step 0, Lane 0: Seq 0, k_idx=0 (final K-block, 299 cached + 1 new)
    self.assertEqual(
        self._get_lane(sched, step=0, lane=0),
        LaneSchedule(
            s_idx=0,
            q_idx=0,
            k_idx=0,
            is_last_k=1,
            do_writeback=1,
            dma_q=DmaQ(hbm_off=0, sz=1),
            dma_kv_cache=[
                DmaCache(phys_page=7, sz=256),
                DmaCache(phys_page=9, sz=128),
            ],
            dma_kv_new_fetch=DmaNewFetch(hbm_off=0, vmem_off=384, sz=128),
            dma_kv_new_wb=[
                DmaNewWb(phys_page=9, page_off=0, vmem_off=256, sz=128)
            ],
        ),
    )

    # Step 0, Lane 1: Seq 1, k_idx=0 (earlier K-block, 512 cached + 0 new)
    self.assertEqual(
        self._get_lane(sched, step=0, lane=1),
        LaneSchedule(
            s_idx=1,
            q_idx=0,
            k_idx=0,
            is_last_k=0,
            do_writeback=0,
            dma_q=DmaQ(hbm_off=1, sz=1),
            dma_kv_cache=[
                DmaCache(phys_page=10, sz=256),
                DmaCache(phys_page=11, sz=256),
            ],
            dma_kv_new_fetch=DmaNewFetch(hbm_off=0, vmem_off=512, sz=0),
            dma_kv_new_wb=[
                DmaNewWb(phys_page=12, page_off=0, vmem_off=512, sz=0)
            ],
        ),
    )
    self.assertEqual(
        self._get_waits(sched, step=0),
        StepWaits(
            kv_in=(256 + 128 + 128) + (256 + 256),
            kv_out=128,
            q_in=2,
            o_out=1,  # Only Lane 0 has is_last_k=1
        ),
    )

    # Step 1, Lane 0: Seq 1, k_idx=1 (final K-block, 168 cached + 1 new)
    self.assertEqual(
        self._get_lane(sched, step=1, lane=0),
        LaneSchedule(
            s_idx=1,
            q_idx=0,
            k_idx=1,
            is_last_k=1,
            do_writeback=1,
            dma_q=DmaQ(hbm_off=1, sz=1),
            dma_kv_cache=[
                DmaCache(phys_page=12, sz=256),
                DmaCache(phys_page=12, sz=0),
            ],
            dma_kv_new_fetch=DmaNewFetch(hbm_off=0, vmem_off=256, sz=128),
            dma_kv_new_wb=[
                DmaNewWb(phys_page=12, page_off=128, vmem_off=128, sz=128)
            ],
        ),
    )

    # Step 1, Lane 1: Masked-out padding lane (s_idx = -1)
    self.assertEqual(
        self._get_lane(sched, step=1, lane=1),
        LaneSchedule(
            s_idx=-1,
            q_idx=0,
            k_idx=0,
            is_last_k=0,
            do_writeback=0,
            dma_q=DmaQ(hbm_off=0, sz=0),
            dma_kv_cache=[
                DmaCache(phys_page=0, sz=0),
                DmaCache(phys_page=0, sz=0),
            ],
            dma_kv_new_fetch=DmaNewFetch(hbm_off=0, vmem_off=0, sz=0),
            dma_kv_new_wb=[DmaNewWb(phys_page=0, page_off=0, vmem_off=0, sz=0)],
        ),
    )
    self.assertEqual(
        self._get_waits(sched, step=1),
        StepWaits(kv_in=256 + 128, kv_out=128, q_in=1, o_out=1),
    )

  def test_chunked_prefill_schedule(self):
    """Verifies causal KV fetch growth, multi-page WB, and multi-chunk flush.

    Setup (page_size=256, bkv_sz=512, bq_sz=128, batch_size=1,
    smem_fraction_limit=0.0 -> max_steps_ub=128):
      - Seq 0: (50, 50) in DECODE band (skipped by PREFILL mode; offsets Seq 1's
        q_start to 50).
      - Seq 1: q_len=400, kv_len=1080 (historical_kv_len=680), pages [16..21]:
        * k_idx=0 ([0, 512)): pure cache (512 cached, new_sz=0).
        * k_idx=1 ([512, 1024)): 168 cached + up to 344 new (680..1023).
          Causal new_sz grows across q_idx=0..2 (128 -> 256 -> 344).
          Writeback triggers once at q_wb=2 across pages 18 and 19.
        * k_idx=2 ([1024, 1536)): 0 cached + up to 56 new (1024..1079).
          Visited by q_idx=2 (new_sz=40, no WB) and q_idx=3 (new_sz=56, WB=1).
      - Seq 2: q_len=3584, kv_len=4096 (historical_kv_len=512), pages [32..47]:
        28 Q-blocks across k_idx=0..7 produce 4 * (2+3+4+5+6+7+8) = 140 steps,
        pushing total steps to 150 (> max_steps_ub=128) so the mid-loop `flush`
        in `k_loop` fires at step 128 and chunk 1 flushes steps 128..149.
    """
    sched = self._make_schedule(
        seq_lens=[(50, 50), (400, 1080), (3584, 4096)],
        page_indices=list(range(48)),
        mode=configs.MlaCase.PREFILL,
        page_size=256,
        bq_sz=128,
        bkv_sz=512,
        batch_size=1,
        distribution=(1, 3, 3),
        smem_fraction_limit=0.0,
    )
    # 10 steps for Seq 1 (steps 0..9) + 140 steps for Seq 2 (steps 10..149):
    #   Seq 1:
    #     q_idx=0 -> k_idx=0 (step 0), k_idx=1 (step 1)
    #     q_idx=1 -> k_idx=0 (step 2), k_idx=1 (step 3)
    #     q_idx=2 -> k_idx=0 (step 4), k_idx=1 (step 5), k_idx=2 (step 6)
    #     q_idx=3 -> k_idx=0 (step 7), k_idx=1 (step 8), k_idx=2 (step 9)
    self.assertEqual(sched.cfgs.max_steps_ub, 128)
    self.assertEqual(int(sched.actual_steps[0]), 150)

    # 1. k_idx=0 (step 0): pure cache block
    lane_q0_k0 = self._get_lane(sched, step=0, lane=0)
    self.assertEqual(lane_q0_k0.k_idx, 0)
    self.assertEqual(lane_q0_k0.do_writeback, 0)
    self.assertEqual(
        lane_q0_k0.dma_kv_cache,
        [DmaCache(phys_page=16, sz=256), DmaCache(phys_page=17, sz=256)],
    )
    self.assertEqual(
        lane_q0_k0.dma_kv_new_fetch, DmaNewFetch(hbm_off=0, vmem_off=512, sz=0)
    )

    # 2. k_idx=1 ([512, 1024)) across q_idx = 0, 1, 2, 3:
    # q_idx=0 (step 1): new_sz=128 (HBM [50, 178) -> [0, 256)), no writeback
    lane_q0_k1 = self._get_lane(sched, step=1, lane=0)
    self.assertEqual(
        lane_q0_k1.dma_kv_new_fetch,
        DmaNewFetch(hbm_off=0, vmem_off=256, sz=256),
    )
    self.assertEqual(lane_q0_k1.do_writeback, 0)
    self.assertEqual(lane_q0_k1.is_last_k, 1)

    # q_idx=1 (step 3): new_sz=256 (HBM [50, 306) -> [0, 384)), no writeback
    lane_q1_k1 = self._get_lane(sched, step=3, lane=0)
    self.assertEqual(
        lane_q1_k1.dma_kv_new_fetch,
        DmaNewFetch(hbm_off=0, vmem_off=256, sz=384),
    )
    self.assertEqual(lane_q1_k1.do_writeback, 0)
    self.assertEqual(lane_q1_k1.is_last_k, 1)

    # q_idx=2 (step 5): reaches kv_block_end=1024 -> new_sz=344, do_writeback=1
    self.assertEqual(
        self._get_lane(sched, step=5, lane=0),
        LaneSchedule(
            s_idx=1,
            q_idx=2,
            k_idx=1,
            is_last_k=0,
            do_writeback=1,
            dma_q=DmaQ(hbm_off=306, sz=128),
            dma_kv_cache=[
                DmaCache(phys_page=18, sz=256),
                DmaCache(phys_page=19, sz=0),
            ],
            dma_kv_new_fetch=DmaNewFetch(hbm_off=0, vmem_off=256, sz=512),
            dma_kv_new_wb=[
                DmaNewWb(phys_page=18, page_off=128, vmem_off=128, sz=128),
                DmaNewWb(phys_page=19, page_off=0, vmem_off=256, sz=256),
            ],
        ),
    )

    # q_idx=3 (step 8): revisits k_idx=1 with do_writeback=0
    lane_q3_k1 = self._get_lane(sched, step=8, lane=0)
    self.assertEqual(
        lane_q3_k1.dma_kv_new_fetch,
        DmaNewFetch(hbm_off=0, vmem_off=256, sz=512),
    )
    self.assertEqual(lane_q3_k1.do_writeback, 0)

    # 3. k_idx=2 ([1024, 1536), pure new KV) across q_idx = 2, 3:
    # q_idx=2 (step 6): new_sz=40 (1024..1063), do_writeback=0
    lane_q2_k2 = self._get_lane(sched, step=6, lane=0)
    self.assertEqual(
        lane_q2_k2.dma_kv_new_fetch,
        DmaNewFetch(hbm_off=384, vmem_off=0, sz=128),
    )
    self.assertEqual(lane_q2_k2.do_writeback, 0)
    self.assertEqual(lane_q2_k2.is_last_k, 1)

    # q_idx=3 (step 9): partial q_sz_task=16, new_sz=56, do_writeback=1
    self.assertEqual(
        self._get_lane(sched, step=9, lane=0),
        LaneSchedule(
            s_idx=1,
            q_idx=3,
            k_idx=2,
            is_last_k=1,
            do_writeback=1,
            dma_q=DmaQ(hbm_off=434, sz=16),
            dma_kv_cache=[
                DmaCache(phys_page=20, sz=0),
                DmaCache(phys_page=21, sz=0),
            ],
            dma_kv_new_fetch=DmaNewFetch(hbm_off=384, vmem_off=0, sz=128),
            dma_kv_new_wb=[
                DmaNewWb(phys_page=20, page_off=0, vmem_off=0, sz=128),
                DmaNewWb(phys_page=21, page_off=0, vmem_off=256, sz=0),
            ],
        ),
    )

    # 4. Chunk 1 (flushed after the mid-loop flush at step 128):
    # Final step (step 149) is Seq 2, q_idx=27, k_idx=7 (last K-block, WB=1).
    lane_last = self._get_lane(sched, step=149, lane=0)
    self.assertEqual(lane_last.s_idx, 2)
    self.assertEqual(lane_last.q_idx, 27)
    self.assertEqual(lane_last.k_idx, 7)
    self.assertEqual(lane_last.is_last_k, 1)
    self.assertEqual(lane_last.do_writeback, 1)


if __name__ == "__main__":
  absltest.main()
