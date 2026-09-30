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
import inspect
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.experimental.batched_rpa.kernel import schedule


def _off(cls, field: str) -> int:
  """Offset of a FieldOffset field, read without invoking the descriptor."""
  return inspect.getattr_static(cls, field).offset


def _create_mock_tpu_info():
  mock_tpu = mock.MagicMock()
  mock_tpu.generation = 7
  mock_tpu.chip_version = "v7x"
  mock_tpu.num_lanes = 128
  mock_tpu.num_sublanes = 8
  mock_tpu.mxu_column_size = 128
  mock_tpu.vmem_capacity_bytes = 64 * 1024 * 1024
  mock_tpu.smem_capacity_bytes = 16 * 1024 * 1024
  return mock_tpu


class ScheduleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.patcher = mock.patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(),
    )
    self.patcher.start()

  def tearDown(self):
    self.patcher.stop()
    super().tearDown()

  def test_verify_metadata_scheduler(self):
    batch = 2
    bq_sz = 64
    bkv_sz = 64
    page_size = 64

    # S0: Prefill (128 tokens -> 2 bq blocks, 2 bkv blocks)
    # S1: Decode (1 token -> 1 bq block, 1 bkv block)
    kv_lens = jnp.array([128, 64], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 128, 129], dtype=jnp.int32)

    # Simple page mapping: Sequence 0 uses pages [10, 11], Sequence 1 uses page [20]
    page_indices = jnp.array([10, 11, 20, -1], dtype=jnp.int32)

    from tokamax._src.ops.experimental.batched_rpa.kernel import configs
    
    block = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch,
        n_buffer=3,
    )
    model = configs.ModelConfigs(
        num_q_heads=1,
        num_kv_heads=1,
        head_dim=128,
        mask_value=0.0,
    )
    serve = configs.ServingConfigs(
        num_seqs=len(kv_lens),
        page_size=page_size,
        total_q_tokens=int(cu_q_lens[-1]),
        num_page_indices=len(page_indices),
        dtype_q=jnp.dtype(jnp.bfloat16),
        dtype_kv=jnp.dtype(jnp.bfloat16),
        dtype_out=jnp.dtype(jnp.bfloat16),
    )
    config = configs.RpaConfigs(
        block=block,
        model=model,
        serve=serve,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=10**9,
    )
    
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    kv_cache_lens = kv_lens - q_lens
    kv_new_lens = q_lens
    q_offsets = kv_cache_lens
    distribution = jnp.array([0, 0, len(kv_lens)], dtype=jnp.int32)

    results = schedule.generate_rpa_metadata(
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
        distribution,
        cfgs=config,
        interpret=True,
    )
    dma_kv_new = results.dma_kv_new
    actual_len = results.actual_steps

    # lane 0: (s0, q0, k0), (s1, q0, k0)
    # lane 1: (s0, q1, k0), (s0, q1, k1)
    self.assertEqual(actual_len[0], 2)

    pos = dma_kv_new._get_pos((0, 0, 0))
    # hbm_cache_dst should point to sequence 0, page 0 (offset 0)
    self.assertEqual(
        int(dma_kv_new.data[pos + _off(schedule.HeadAlongSublaneDmaNew, "wb_hbm")]),
        0,
    )
    # src_hbm_new should be q_start (0)
    self.assertEqual(
        int(dma_kv_new.data[pos + _off(schedule.HeadAlongSublaneDmaNew, "fetch_hbm")]),
        0,
    )
    # size should be 64
    self.assertEqual(
        int(dma_kv_new.data[pos + _off(schedule.HeadAlongSublaneDmaNew, "_flags")]),
        64,
    )

    s_idx = results.s_idx
    q_idx = results.q_idx
    k_idx = results.k_idx
    is_last_k = results.is_last_k

    # Check lane 0: step 0 is (s0, q0, k0), step 1 is (s0, q1, k1)
    np.testing.assert_array_equal([s_idx[step, 0] for step in range(2)], [0, 0])
    np.testing.assert_array_equal([q_idx[step, 0] for step in range(2)], [0, 1])
    np.testing.assert_array_equal([k_idx[step, 0] for step in range(2)], [0, 1])
    np.testing.assert_array_equal([is_last_k[step, 0] for step in range(2)], [1, 1])

    # Check lane 1: step 0 is (s0, q1, k0), step 1 is (s1, q0, k0)
    np.testing.assert_array_equal([s_idx[step, 1] for step in range(2)], [0, 1])
    np.testing.assert_array_equal([q_idx[step, 1] for step in range(2)], [1, 0])
    np.testing.assert_array_equal([k_idx[step, 1] for step in range(2)], [0, 0])
    np.testing.assert_array_equal([is_last_k[step, 1] for step in range(2)], [0, 1])

  def test_complex_scheduling(self):
    batch = 2
    bq_sz = 32
    bkv_sz = 64
    page_size = 64
    sliding_window = 64

    # S0: 65 q tokens, 129 k tokens. q_blocks=3, k_blocks=3
    #     q0=[0,31], q1=[32,63], q2=[64,64].
    #     k-range for q0: only k=1 attends context, k=0 if sw=64
    #     k_len=129, q_len=65.
    #     q0: 129-65+0-64+1 = 1. k_start=max(0,1//64)=0.
    #         end_k = min(3, (129-65+0+32-1)//64+1)=min(3, 95//64+1)=min(3,2)=2
    #         q0 -> k=0,1
    #     q1: 129-65+32-64+1=33. k_start=max(0,33//64)=0.
    #         end_k = min(3, (129-65+32+32-1)//64+1)=min(3, 127//64+1)=min(3,2)=2
    #         q1 -> k=0,1
    #     q2: 129-65+64-64+1=65. k_start=max(0,65//64)=1.
    #         end_k = min(3, (129-65+64+1-1)//64+1)=min(3, 129//64+1)=min(3,3)=3
    #         q2 -> k=1,2
    # S1: 33 q tokens, 65 k tokens. q_blocks=2, k_blocks=2
    #     q0=[0,31], q1=[32,32]
    #     k_len=65, q_len=33
    #     q0: 65-33+0-64+1=-31. k_start=0.
    #         end_k=min(2, (65-33+0+32-1)//64+1)=min(2, 63//64+1)=1
    #         q0 -> k=0
    #     q1: 65-33+32-64+1=1. k_start=0
    #         end_k=min(2, (65-33+32+1-1)//64+1)=min(2, 65//64+1)=2
    #         q1 -> k=0,1
    # S2: 1 q token, 128 k tokens. q_blocks=1, k_blocks=2
    #     q0=[0,0]
    #     k_len=128, q_len=1
    #     q0: 128-1+0-64+1=64. k_start=1
    #         end_k=min(2, (128-1+0+1-1)//64+1)=min(2,127//64+1)=2
    #         q0 -> k=1

    # Scheduling:
    # s0q0(k=0,1): l0,s0,1=[1,0]. len[0]=2
    # s0q1(k=0,1): l1,s0,1=[2,1]. len[1]=2
    # s0q2(k=1,2): l0,s2,3=[2,3]. len[0]=2+2=4
    # s1q0(k=0):   l1,s2=[4,3]. len[1]=2+1=3
    # s1q1(k=0,1): l1,s3,4=[4,4]. len[1]=3+2=5
    # s2q0(k=1):   l0,s4=[5,5]. len[0]=4+1=5
    # Actual steps = 5.

    kv_lens = jnp.array([129, 65, 128], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 65, 98, 99], dtype=jnp.int32)
    page_indices = jnp.array(
        [10, 11, 12, 20, 21, 22, 30, 31, 32], dtype=jnp.int32
    )

    from tokamax._src.ops.experimental.batched_rpa.kernel import configs
    
    block = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch,
        n_buffer=3,
    )
    model = configs.ModelConfigs(
        num_q_heads=1,
        num_kv_heads=1,
        head_dim=128,
        mask_value=0.0,
        sliding_window=sliding_window,
    )
    serve = configs.ServingConfigs(
        num_seqs=len(kv_lens),
        page_size=page_size,
        total_q_tokens=int(cu_q_lens[-1]),
        num_page_indices=len(page_indices),
        dtype_q=jnp.dtype(jnp.bfloat16),
        dtype_kv=jnp.dtype(jnp.bfloat16),
        dtype_out=jnp.dtype(jnp.bfloat16),
    )
    config = configs.RpaConfigs(
        block=block,
        model=model,
        serve=serve,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=10**9,
    )
    
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    kv_cache_lens = kv_lens - q_lens
    kv_new_lens = q_lens
    q_offsets = kv_cache_lens
    distribution = jnp.array([0, 0, len(kv_lens)], dtype=jnp.int32)
    results = schedule.generate_rpa_metadata(
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
        distribution,
        cfgs=config,
        interpret=True,
    )
    actual_len = results.actual_steps
    self.assertEqual(actual_len[0], 5)
    s_idx, q_idx, k_idx, is_last_k = (
        results.s_idx,
        results.q_idx,
        results.k_idx,
        results.is_last_k,
    )

    # Check lane 0 schedule
    # Step 0: (s0, q0, k0)
    # Step 1: (s0, q1, k0)
    # Step 2: (s0, q2, k1)
    # Step 3: (s1, q0, k0)
    # Step 4: (s1, q1, k1)
    np.testing.assert_array_equal([s_idx[step, 0] for step in range(5)], [0, 0, 0, 1, 1])
    np.testing.assert_array_equal([q_idx[step, 0] for step in range(5)], [0, 1, 2, 0, 1])
    np.testing.assert_array_equal([k_idx[step, 0] for step in range(5)], [0, 0, 1, 0, 1])
    np.testing.assert_array_equal([is_last_k[step, 0] for step in range(5)], [0, 0, 0, 1, 1])

    # Check lane 1 schedule
    # Step 0: (s0, q0, k1)
    # Step 1: (s0, q1, k1)
    # Step 2: (s0, q2, k2)
    # Step 3: (s1, q1, k0)
    # Step 4: (s2, q0, k1)
    np.testing.assert_array_equal([s_idx[step, 1] for step in range(5)], [0, 0, 0, 1, 2])
    np.testing.assert_array_equal([q_idx[step, 1] for step in range(5)], [0, 1, 2, 1, 0])
    np.testing.assert_array_equal([k_idx[step, 1] for step in range(5)], [1, 1, 2, 0, 1])
    np.testing.assert_array_equal([is_last_k[step, 1] for step in range(5)], [1, 1, 1, 0, 1])

  def test_verify_metadata_scheduler_seq_along_lane(self):
    batch = 2
    bq_sz = 64
    bkv_sz = 64
    page_size = 64

    kv_lens = jnp.array([128, 64], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 128, 129], dtype=jnp.int32)
    page_indices = jnp.array([10, 11, 20, -1], dtype=jnp.int32)

    from tokamax._src.ops.experimental.batched_rpa.kernel import configs

    block = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch,
        n_buffer=3,
    )
    model = configs.ModelConfigs(
        num_q_heads=1,
        num_kv_heads=1,
        head_dim=128,
        mask_value=0.0,
    )
    serve = configs.ServingConfigs(
        num_seqs=len(kv_lens),
        page_size=page_size,
        total_q_tokens=int(cu_q_lens[-1]),
        num_page_indices=len(page_indices),
        dtype_q=jnp.dtype(jnp.bfloat16),
        dtype_kv=jnp.dtype(jnp.bfloat16),
        dtype_out=jnp.dtype(jnp.bfloat16),
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    config = configs.RpaConfigs(
        block=block,
        model=model,
        serve=serve,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=10**9,
    )

    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    kv_cache_lens = kv_lens - q_lens
    kv_new_lens = q_lens
    q_offsets = kv_cache_lens
    distribution = jnp.array([0, 0, len(kv_lens)], dtype=jnp.int32)

    results = schedule.generate_rpa_metadata(
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
        distribution,
        cfgs=config,
        interpret=True,
    )
    dma_kv_new = results.dma_kv_new
    pos = dma_kv_new._get_pos((0, 0, 0))
    self.assertEqual(
        int(dma_kv_new.data[pos + _off(schedule.SeqAlongLaneDmaNew, "wb_hbm")]), 0
    )
    self.assertEqual(
        int(dma_kv_new.data[pos + _off(schedule.SeqAlongLaneDmaNew, "fetch_hbm")]), 0
    )
    flags = int(dma_kv_new.data[pos + _off(schedule.SeqAlongLaneDmaNew, "_flags")])
    # Tokamax (like vllm-torchtpu) still uses page-aligned SEQ_ALONG_LANE DMAs
    # with 1-bit fetch/writeback flags; 128-lane alignment with 16-bit token
    # counts hasn't been synced into Tokamax yet.
    fetch_val = flags & 1
    wb_val = (flags >> 1) & 1
    self.assertEqual(fetch_val, 1)
    self.assertEqual(wb_val, 1)


if __name__ == "__main__":
  absltest.main()
