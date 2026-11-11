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
"""Tests of the De-Dup dispatch of the monolithic layer.

The cases cover both supported board sizes: 16 devices as 2 boards of 8 and
as 4 boards of 4, and 8 devices as 2 boards of 4. They run on CPU devices in
interpret mode, or on TPU v8i devices with the real kernels (hw_util; test 1
only, when the host has enough devices):

  1. The layer with De-Dup dispatch (`dedup_topology=BoardTopology(...)`) is
     bit-identical (`np.testing.assert_array_equal`) to the direct wave
     dispatch with the same wave count (`dedup_topology=None`), for `bf16` and
     `fp8` weights, `router` and `skewed` routing, W in {2, 4}, a forward
     table small enough to force the overflow fallback to direct sends, and a
     shared expert (`ws1`, `ws2`). Both are close to the dense f32 reference.
  2. A single-call Pallas harness with `detect_races=True` (CPU only: the race
     detector exists only in interpret mode): runs
     `tc_sender._start_waves_kernel(has_stage=True)`,
     `memory_ref.forward_staged_wave_body`, the per-wave receive waits, and
     `tc_sender._send_done_kernel` inside one `pallas_call` on the CPU
     devices, so the Pallas race detector checks the stage, gateway forward,
     and wave semaphore protocol, and every needed row lands at
     `row_on_dest`.
"""

import functools
import types
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dedup_plan.topology import BoardTopology
from tokamax._src.ops.experimental.dedup_moe.dispatch import source_plan
from tokamax._src.ops.experimental.dedup_moe.dispatch import tc_sender
from tokamax._src.ops.experimental.dedup_moe.dispatch import wave_dispatch
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import memory_ref
from tokamax._src.ops.experimental.dedup_moe.fused_kernel.config import AXIS
from tokamax._src.ops.experimental.dedup_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.dedup_moe.layer.monolithic import dedup_ep_moe
from tokamax._src.ops.experimental.dedup_moe.test.common import hw_util
from tokamax._src.ops.experimental.dedup_moe.test.common import interpret_util
from tokamax._src.ops.experimental.dedup_moe.test.common import moe_inputs
from tokamax._src.ops.experimental.dedup_moe.test.common import phase1_ref
from tokamax._src.ops.experimental.dedup_moe.test.common import reference
from tokamax._src.ops.experimental.fused_moe import host as fused_host

# CPU devices the test process creates (the largest ep below).
CPU_DEVICES = 16
EXPERTS_PER_SHARD = 4
TOPK = 4
# Token-gather list rows per peer in the race harness (the lists are empty).
LIST_ROWS = 8


@functools.cache
def _shapes() -> types.SimpleNamespace:
  """(hidden, inter) and the tile height (hw_util.pick needs JAX)."""
  hidden_inter = hw_util.pick((256, 256), (hw_util.HW_HIDDEN, hw_util.HW_INTER))
  return types.SimpleNamespace(
      dims={"bf16": hidden_inter, "fp8": hidden_inter},
      # Layer tile height: a multiple of hw_util.HW_CAPACITY on hardware.
      capacity=hw_util.pick(16, hw_util.HW_CAPACITY),
  )


ROWS_PER_SHARD = 16
# rel-L2 bound against the f32 reference (as in the layer interpret test).
REL_L2_BOUND = {"bf16": 2e-2, "fp8": 1e-1}


def setUpModule():
  hw_util.setup_module("tpu8i")


def _topo_name(ep, board_size):
  return f"ep{ep}_bs{board_size}"


def _layer_case(ep, board_size, name, **kwargs):
  return dict(
      testcase_name=f"{_topo_name(ep, board_size)}_{name}",
      ep=ep,
      board_size=board_size,
      **kwargs,
  )


_LAYER_CASES = (
    # 2 boards of 8.
    _layer_case(16, 8, "bf16_w4", fmt="bf16", waves=4),
    _layer_case(16, 8, "fp8_w4", fmt="fp8", waves=4),
    _layer_case(16, 8, "bf16_w2", fmt="bf16", waves=2),
    _layer_case(16, 8, "bf16_w4_skewed", fmt="bf16", waves=4, routing="skewed"),
    # 4 forward-table entries per (slot, wave): the plan demotes the staged
    # tokens that do not fit to direct sends.
    _layer_case(
        16, 8, "fp8_w4_fwd_overflow", fmt="fp8", waves=4, fwd_max_runs=4
    ),
    _layer_case(16, 8, "bf16_w4_shared", fmt="bf16", waves=4, shared=True),
    # 4 boards of 4.
    _layer_case(16, 4, "bf16_w4", fmt="bf16", waves=4),
    _layer_case(16, 4, "fp8_w2_skewed", fmt="fp8", waves=2, routing="skewed"),
    _layer_case(
        16, 4, "fp8_w4_fwd_overflow", fmt="fp8", waves=4, fwd_max_runs=4
    ),
    _layer_case(16, 4, "bf16_w4_shared", fmt="bf16", waves=4, shared=True),
    # 2 boards of 4.
    _layer_case(8, 4, "fp8_w4", fmt="fp8", waves=4),
    _layer_case(8, 4, "bf16_w2", fmt="bf16", waves=2),
    _layer_case(8, 4, "fp8_w4_skewed", fmt="fp8", waves=4, routing="skewed"),
)


class DedupMoeTest(parameterized.TestCase):
  """De-Dup == direct, bit for bit, and both close to the f32 reference."""

  # ep -> mesh and (ep, fmt) -> weights, built on first use.
  _meshes: dict[int, Any] = {}
  _weights: dict[tuple[int, str], Any] = {}

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)
    hw_util.reset_interpret_state()

  def _mesh(self, ep):
    if ep not in self._meshes:
      self._meshes[ep] = hw_util.make_mesh(ep, self)
    return self._meshes[ep]

  def _layer_weights(self, ep, fmt):
    key = (ep, fmt)
    if key not in self._weights:
      hidden, inter = _shapes().dims[fmt]
      self._weights[key] = moe_inputs.random_weights(
          self._mesh(ep),
          e_total=ep * EXPERTS_PER_SHARD,
          hidden=hidden,
          inter=inter,
          weight_format=fmt,
          seed=0,
      )
    return self._weights[key]

  def _run_layer(
      self,
      mesh,
      weights,
      fmt,
      x,
      gating,
      *,
      waves,
      hash_ids=None,
      shared=None,
      **kwargs,
  ):
    """One layer call, as f32 numpy.

    Args:
      mesh: The EP mesh.
      weights: (w1, w2, w1_scale, w2_scale).
      fmt: The weight format name.
      x: [T, hidden] token rows.
      gating: [T, E] router logits.
      waves: W.
      hash_ids: Optional [T, topk] forced expert ids.
      shared: Optional (ws1, ws2) shared expert weights.
      **kwargs: Static dedup_ep_moe arguments (closed over, never traced).

    Returns:
      [T, hidden] f32.
    """

    def fn(x, w1, w2, w1_scale, w2_scale, gating, hash_ids, shared):
      ws1, ws2 = shared if shared is not None else (None, None)
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
          capacity=_shapes().capacity,
          weight_format=arg.WeightFormat(fmt),
          hash_topk_ids=hash_ids,
          waves=waves,
          ws1=ws1,
          ws2=ws2,
          **kwargs,
      )

    with hw_util.cpu_layer():
      out = hw_util.run(fn, x, *weights, gating, hash_ids, shared)
      return np.asarray(out, np.float32)

  @parameterized.named_parameters(*_LAYER_CASES)
  def test_dedup_dispatch_is_bit_identical(
      self,
      ep,
      board_size,
      fmt,
      waves,
      routing="router",
      fwd_max_runs=None,
      shared=False,
  ):
    mesh = self._mesh(ep)
    weights = self._layer_weights(ep, fmt)
    e_total = ep * EXPERTS_PER_SHARD
    hidden, inter = _shapes().dims[fmt]
    tokens = ep * ROWS_PER_SHARD
    x, gating = moe_inputs.random_inputs(
        mesh, tokens=tokens, hidden=hidden, e_total=e_total, seed=19
    )
    ids = None
    hash_ids = None
    if routing != "router":
      ids = moe_inputs.expert_ids(
          routing, tokens, e_total, TOPK, EXPERTS_PER_SHARD, seed=7
      )
      hash_ids = moe_inputs.shard_rows(mesh, ids)
    shared_w = None
    if shared:
      k1, k2 = jax.random.split(jax.random.key(123))
      shared_w = (
          (
              jax.random.normal(k1, (hidden, 2 * inter), dtype=jnp.float32)
              * 0.05
          ).astype(jnp.bfloat16),
          (
              jax.random.normal(k2, (inter, hidden), dtype=jnp.float32) * 0.05
          ).astype(jnp.bfloat16),
      )
    common = dict(waves=waves, hash_ids=hash_ids, shared=shared_w)
    want = self._run_layer(mesh, weights, fmt, x, gating, **common)
    dedup_kw: dict[str, Any] = dict(
        dedup_topology=BoardTopology.uniform(ep, board_size)
    )
    if fwd_max_runs is not None:
      dedup_kw["dedup_fwd_max_runs"] = fwd_max_runs
    got = self._run_layer(mesh, weights, fmt, x, gating, **common, **dedup_kw)
    np.testing.assert_array_equal(got, want)

    routed = want
    if shared_w is not None:
      # The shared expert contributes shared_expert_ffn on top of the routed
      # experts.
      routed = self._run_layer(
          mesh, weights, fmt, x, gating, waves=waves, hash_ids=hash_ids
      )
      with hw_util.host_only():
        expected_shared = np.asarray(
            input_pack.shared_expert_ffn(
                jnp.asarray(jax.device_get(x)),
                jnp.asarray(jax.device_get(shared_w[0])),
                jnp.asarray(jax.device_get(shared_w[1])),
            ).reshape(tokens, hidden),
            np.float32,
        )
      diff_shared = float(np.max(np.abs(want - (routed + expected_shared))))
      self.assertLess(diff_shared, 4e-2)
    # The routed experts against the dense f32 reference.
    with hw_util.host_only():
      # Host copies as jax arrays (on hardware the layer's are on the TPUs).
      w1, w2, w1_scale, w2_scale = (
          None if a is None else jnp.asarray(jax.device_get(a)) for a in weights
      )
      ref = np.asarray(
          reference.moe_reference(
              jnp.asarray(jax.device_get(x)),
              w1,
              w2,
              w1_scale,
              w2_scale,
              jnp.asarray(jax.device_get(gating)),
              topk=TOPK,
              hash_topk_ids=ids,
          )
      )
    self.assertLess(reference.rel_l2(routed, ref), REL_L2_BOUND[fmt])


def _race_case(ep, board_size, name, **kwargs):
  return dict(
      testcase_name=f"{_topo_name(ep, board_size)}_{name}",
      ep=ep,
      board_size=board_size,
      **kwargs,
  )


_RACE_CASES = (
    _race_case(16, 8, "w2_ptw", waves=2, per_target_wave=True),
    _race_case(16, 8, "w4_ptw", waves=4, per_target_wave=True),
    _race_case(16, 8, "w4_d4", waves=4, per_target_wave=False),
    _race_case(16, 4, "w4_ptw", waves=4, per_target_wave=True),
    _race_case(16, 4, "w2_d4", waves=2, per_target_wave=False),
    _race_case(8, 4, "w4_ptw", waves=4, per_target_wave=True),
)


class DedupRaceTest(parameterized.TestCase):
  """The stage, forward and wave protocol under the Pallas race detector."""

  def setUp(self):
    super().setUp()
    hw_util.skip_unless_v8i(self)
    hw_util.skip_on_hardware(
        self, "detect_races exists only in Pallas TPU interpret mode"
    )
    hw_util.reset_interpret_state()

  @parameterized.named_parameters(*_RACE_CASES)
  def test_single_call_pallas_stage_and_forward_race_free(
      self, ep, board_size, waves, per_target_wave
  ):
    mesh = hw_util.make_mesh(ep, self)
    e_total = ep * EXPERTS_PER_SHARD
    t_local = 16
    tokens = ep * t_local
    hidden = 256
    lane_blocks = hidden // fused_host.HIDDEN_LANE_BLOCK
    dedup_tables = BoardTopology.uniform(ep, board_size).tables()
    num_boards = dedup_tables.num_boards
    num_slots = dedup_tables.num_slots
    # 8 t runs per (sigma, wave) is the bound every (row, target) pair being
    # its own run gives, so no routing overflows the table.
    fwd_schedule = dedup_plan_mod.FwdSchedule(
        per_target_wave=per_target_wave, max_runs=8 * t_local
    )
    send_alloc = fused_host.align_up(t_local * min(TOPK, ep), fused_host.ROWBLK)
    stage_alloc = fused_host.align_up(num_slots * t_local, fused_host.ROWBLK)
    recv_alloc = fused_host.align_up(tokens, fused_host.ROWBLK)

    rng = np.random.default_rng(42)
    pairs_np = np.stack(
        [rng.choice(e_total, size=TOPK, replace=False) for _ in range(tokens)],
        axis=0,
    ).astype(np.int32)
    rows_np = (
        (
            np.arange(tokens * hidden, dtype=np.int32).reshape(
                tokens, lane_blocks, fused_host.HIDDEN_LANE_BLOCK
            )
            % 251
            + 1
        )
        .astype(np.float32)
        .astype(jnp.bfloat16)
    )

    P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
    NS = jax.sharding.NamedSharding  # pylint: disable=invalid-name
    rows_sharded = jax.device_put(rows_np, NS(mesh, P(AXIS)))
    pairs_rep = jax.device_put(jnp.asarray(pairs_np), NS(mesh, P()))
    ranks_sharded = jax.device_put(
        jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1), NS(mesh, P(AXIS))
    )

    params = interpret_util.default_interpret_params(detect_races=True)

    def shard_body(rows_l, pairs_g, rank_l):
      me = rank_l[0, 0]
      block = fused_host.routing_block(t_local, TOPK)
      local_blocks = lax.dynamic_slice(
          pairs_g, (me * t_local, 0), (t_local, TOPK)
      ).reshape(t_local * TOPK // block, block)
      block_hist = fused_host.pair_block_hist(local_blocks, e_total)
      rows_by_dest = lax.all_gather(
          block_hist.sum(axis=0), AXIS, axis=0, tiled=False
      )
      wave_of = phase1_ref.wave_of_from_rows_by_dest(
          rows_by_dest, g=EXPERTS_PER_SHARD, num_waves=waves
      )
      plan_direct = source_plan.source_plan(
          pairs_g,
          me,
          ep=ep,
          t_local=t_local,
          g_local=EXPERTS_PER_SHARD,
          waves=waves,
          wave_of=wave_of,
      )
      plan_kw = dict(
          tables=dedup_tables,
          num_waves=waves,
          min_fanout=2,
          schedule=fwd_schedule,
      )
      dplan = dedup_plan_mod.dedup_plan(
          plan_direct.fw, me, need=plan_direct.need, **plan_kw
      )
      fw_served = dedup_plan_mod.served_fw(
          pairs_g,
          wave_of,
          me,
          tables=dedup_tables,
          t=t_local,
          num_waves=waves,
      )
      gplan = dedup_plan_mod.gateway_plan(fw_served, me, **plan_kw)
      fw_my_board = phase1_ref.board_fw(
          pairs_g,
          wave_of,
          me,
          t=t_local,
          num_waves=waves,
          board_size=board_size,
      )
      recv_cnt = dedup_plan_mod.receiver_counts(fw_my_board, me, **plan_kw)
      disp_tab = dedup_plan_mod.dedup_disp_tab(dplan, recv_cnt.recv_cnt)
      smem_ops = dedup_plan_mod.pack_smem_operands(dplan, gplan)
      stage_send_tab = jnp.concatenate(
          [
              dplan.stage_send_off,
              dplan.cnt_stage,
              dplan.stage_dst,
              jnp.broadcast_to(dplan.gw[:, None], (num_boards, waves)),
              jnp.broadcast_to(dplan.sigma[:, None], (num_boards, waves)),
          ],
          axis=0,
      ).astype(jnp.int32)
      list_tab = jnp.zeros((ep + 1,), jnp.int32)

      def combined_pallas_body(
          disp_tab_smem,
          me_smem,
          perm_smem,
          list_tab_smem,
          stage_send_tab_smem,
          stage_tab_smem,
          fwd_tab_smem,
          fwd_cnt_smem,
          rows_hbm,
          send_in_hbm,
          recv_in_hbm,
          stage_in_hbm,
          send_hbm,
          recv_hbm,
          stage_hbm,
          send_sem,
          wave_sems,
          stage_sems,
          fwd_send_sem,
          pack_sem,
          list_send_sem,
          tg_sem,
          tg_lists_vm,
          tg_land_vm,
      ):
        tc_sender._start_waves_kernel(  # pylint: disable=protected-access
            disp_tab_smem,
            me_smem,
            send_in_hbm,
            recv_in_hbm,
            perm_smem,
            rows_hbm,
            list_tab_smem,
            tg_lists_vm,
            tg_land_vm,
            stage_send_tab_smem,
            stage_in_hbm,
            send_sem,
            wave_sems,
            send_hbm,
            recv_hbm,
            list_send_sem,
            tg_sem,
            tg_lists_vm,
            tg_land_vm,
            stage_sems,
            stage_hbm,
            pack_sem,
            ep=ep,
            waves=waves,
            chunk_rows=wave_dispatch.CHUNK_ROWS,
            list_rows=LIST_ROWS,
            has_stage=True,
            num_boards=num_boards,
            board_size=board_size,
        )
        me = me_smem[0]
        for w in range(waves):
          # The fused kernel's own gateway step, on the flat tables it gets.
          memory_ref.forward_staged_wave_body(
              jnp.int32(w),
              me=me,
              stage_tab_sm=stage_tab_smem,
              fwd_tab_sm=fwd_tab_smem,
              fwd_cnt_sm=fwd_cnt_smem,
              stage_sems=stage_sems,
              stage_hbm=stage_hbm,
              lhs_hbm=recv_hbm,
              wave_sems=wave_sems,
              fwd_send_sem=fwd_send_sem,
              num_slots=num_slots,
              num_waves=waves,
              num_boards=num_boards,
              fwd_max_runs=fwd_schedule.max_runs,
          )
          rows_w = disp_tab_smem[3 * ep, w]

          @pl.when(rows_w > 0)
          def _wait_wave(w=w, rows_w=rows_w):
            memory_ref.rows_wait(wave_sems.at[w], recv_hbm, rows_w)

        # Drain the gateway's remote forward sends.
        fwd_row = (2 * num_boards + num_slots) * waves
        total_fwd_rows = stage_tab_smem[fwd_row]
        for w in range(1, waves):
          total_fwd_rows = total_fwd_rows + stage_tab_smem[fwd_row + w]

        @pl.when(total_fwd_rows > 0)
        def _drain_fwd():
          memory_ref.rows_wait_send(fwd_send_sem, recv_hbm, total_fwd_rows)

        # Direct and stage sends share send_sem; disp_tab row 3 ep + 1 counts
        # both.
        tc_sender._send_done_kernel(  # pylint: disable=protected-access
            disp_tab_smem,
            send_hbm,
            send_sem,
            list_tab_smem,
            tg_lists_vm,
            list_send_sem,
            send_hbm,
            send_hbm,
            ep=ep,
        )

      hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
      send_init = jnp.zeros(
          (send_alloc, lane_blocks, fused_host.HIDDEN_LANE_BLOCK), jnp.bfloat16
      )
      recv_init = jnp.zeros(
          (recv_alloc, lane_blocks, fused_host.HIDDEN_LANE_BLOCK), jnp.bfloat16
      )
      stage_init = jnp.zeros(
          (stage_alloc, lane_blocks, fused_host.HIDDEN_LANE_BLOCK),
          jnp.bfloat16,
      )
      scratch_shapes = (
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA((waves,)),
          pltpu.SemaphoreType.DMA((num_slots * waves,)),
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
          pltpu.VMEM((ep * LIST_ROWS,), jnp.int32),
          pltpu.VMEM((ep * LIST_ROWS,), jnp.int32),
      )
      _, recv_out, _ = pl.pallas_call(
          combined_pallas_body,
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=8,
              in_specs=[hbm, hbm, hbm, hbm],
              out_specs=[hbm, hbm, hbm],
              scratch_shapes=scratch_shapes,
              grid=(),
          ),
          out_shape=[
              jax.ShapeDtypeStruct(send_init.shape, jnp.bfloat16),
              jax.ShapeDtypeStruct(recv_init.shape, jnp.bfloat16),
              jax.ShapeDtypeStruct(stage_init.shape, jnp.bfloat16),
          ],
          input_output_aliases={9: 0, 10: 1, 11: 2},
          compiler_params=pltpu.CompilerParams(
              collective_id=tc_sender.A2A_COLLECTIVE_ID
          ),
          interpret=params,
          name="test_dedup_stage_and_forward",
      )(
          disp_tab,
          rank_l.reshape(1),
          dplan.perm[:send_alloc],
          list_tab,
          stage_send_tab,
          smem_ops.stage_tab.reshape(-1),
          smem_ops.fwd_tab.reshape(-1),
          smem_ops.fwd_cnt.reshape(-1),
          rows_l,
          send_init,
          recv_init,
          stage_init,
      )
      return recv_out, dplan.row_on_dest, plan_direct.need

    fn = jax.jit(
        jax.shard_map(
            shard_body,
            mesh=mesh,
            in_specs=(P(AXIS), P(), P(AXIS)),
            out_specs=(P(AXIS), P(AXIS), P(AXIS)),
            check_vma=False,
        )
    )
    recv_all, row_on_dest_all, need_all = fn(
        rows_sharded, pairs_rep, ranks_sharded
    )
    recv_np = np.asarray(recv_all, np.float32).reshape(
        ep, recv_alloc, lane_blocks, fused_host.HIDDEN_LANE_BLOCK
    )
    rows_expected = np.asarray(rows_np, np.float32)
    rod_np = np.asarray(row_on_dest_all).reshape(ep, ep, t_local)
    need_np = np.asarray(need_all).reshape(ep, ep, t_local)
    self.assertFalse(interpret_util.races_found())
    checked = 0
    for s in range(ep):
      for d in range(ep):
        for tok in range(t_local):
          if need_np[s, d, tok]:
            r = int(rod_np[s, d, tok])
            self.assertGreaterEqual(r, s * t_local)
            self.assertLess(r, (s + 1) * t_local)
            np.testing.assert_array_equal(
                recv_np[d, r], rows_expected[s * t_local + tok]
            )
            checked += 1
    self.assertGreater(checked, ep * t_local)


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", CPU_DEVICES)
  absltest.main()
