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
"""CPU harness for the De-Dup MoE layer: run it in Pallas TPU interpret mode.

Nothing in the layer's modules (fused_kernel/*, common/*, layer/*, dispatch/*,
io_relayout/*) is edited to run on CPU. Each helper below patches a module
ATTRIBUTE for the duration of a `with` block and restores it on exit, so the
TPU path is byte-for-byte the upstream code:

  * `register_cpu_chip` answers `pltpu.get_tpu_info()` with a TPU chip record
    on a CPU host. fused_kernel.plan.vmem_limit / fit_weight_slots /
    chip_generation read the device record at kernel build time and raise on
    "cpu" otherwise.
  * `interpret_pallas` swaps `jax.experimental.pallas.pallas_call` for a
    wrapper that adds `interpret=pltpu.InterpretParams(...)` to every call
    that does not ask for interpret mode itself. Every fork module reaches
    pallas_call through that module attribute (`pl.pallas_call`), so the fused
    kernel, the combine kernel, the router's top-k selector and the wave
    dispatch's kernels all run interpreted, with simulated remote DMAs,
    semaphores and barriers across the CPU devices.
  * `emulate_wave_exchange` replaces only the split-phase exchange
    (tc_sender.ragged_a2a_start_waves / ragged_a2a_send_done) by the
    same data movement written with `lax.all_gather` (De-Dup's stage rows and
    gateway forwards too), and builds the fused kernel without its per-wave
    semaphore waits and forwards (wait_for_exchange=False). TPU interpret mode
    cannot run the split-phase pair: it gives every pallas_call fresh
    semaphores and clears its simulated shared memory between calls, so a DMA
    semaphore cannot be handed from the start kernel to the fused kernel (a
    semaphore output fails in interpret mode). Everything else on the wave
    path runs as on TPU: the expert visit order, the wave plan, the De-Dup
    plan and gateway tables, the packed send rows, the receive-buffer layout,
    the token-gather lists and tg_assemble, and the fused kernel computing on
    the wave-ordered buffer. What stays TPU-only: the start/send-done
    kernels' DMAs and the per-wave waits and forwards inside the fused
    kernel.
"""

import contextlib
import functools
import importlib
from typing import Any, Iterator

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dispatch import tc_sender as dispatch_kernels
from tokamax._src.ops.experimental.dedup_moe.dispatch import wave_dispatch
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import config as kernel
from tokamax._src.ops.experimental.dedup_moe.layer import monolithic as layer

CHIPS = {
    "tpu7": pltpu.ChipVersion.TPU_7,
    "tpu8i": pltpu.ChipVersion.TPU_8I,
}


def chip_info(chip: str = "tpu8i") -> Any:
  """The Pallas TPU record of `chip` (one TensorCore per logical device)."""
  if chip not in CHIPS:
    raise ValueError(f"unknown chip {chip!r}; one of {sorted(CHIPS)}")
  return pltpu.get_tpu_info_for_chip(CHIPS[chip], 1)


def register_cpu_chip(chip: str = "tpu8i") -> Any:
  """Makes `pltpu.get_tpu_info()` return `chip`'s record on a CPU host.

  Args:
    chip: A key of `CHIPS` (`tpu7` for TPU v7x, `tpu8i` for TPU v8i).

  Returns:
    The record `pltpu.get_tpu_info()` now returns.
  """
  tpu_info = importlib.import_module("jax._src.tpu_info")
  info = chip_info(chip)
  tpu_info.registry["cpu"] = lambda: info
  # get_tpu_info memoizes its answer; drop one taken before registration.
  cache_clear = getattr(tpu_info.get_tpu_info, "cache_clear", None)
  if cache_clear is not None:
    cache_clear()
  return pltpu.get_tpu_info()


def default_interpret_params(**overrides: Any) -> Any:
  """The InterpretParams the interpret tests run with (overridable)."""
  kwargs: dict[str, Any] = dict(
      dma_execution_mode="on_wait", detect_races=False
  )
  kwargs.update(overrides)
  return pltpu.InterpretParams(**kwargs)


@contextlib.contextmanager
def interpret_pallas(params: Any = None) -> Iterator[Any]:
  """Runs every `pl.pallas_call` traced inside the block in interpret mode.

  Args:
    params: The `pltpu.InterpretParams`; `default_interpret_params()` if None.

  Yields:
    The params in effect.
  """
  params = default_interpret_params() if params is None else params
  original = pl.pallas_call

  @functools.wraps(original)
  def interpreted_pallas_call(*args, interpret=False, **kwargs):
    return original(*args, interpret=interpret or params, **kwargs)

  pl.pallas_call = interpreted_pallas_call
  try:
    yield params
  finally:
    pl.pallas_call = original


def reset_interpret_state() -> None:
  """Clears TPU interpret mode's simulated state (needed after a failure)."""
  pltpu.reset_tpu_interpret_mode_state()


def races_found() -> bool:
  """Whether the interpret-mode race detector has reported a race."""
  mod = importlib.import_module(
      "jax._src.pallas.mosaic.interpret.interpret_pallas_call"
  )
  return bool(mod.races.races_found)


def emulated_exchange(send, recv_rows, disp_tab, me, *, ep, axis_name):
  """The receive buffer `ragged_a2a_start_waves` fills, built with a gather.

  Args:
    send: [send_alloc, ...] this shard's packed send rows.
    recv_rows: Height of the receive buffer.
    disp_tab: [4 * ep, waves] i32 dispatch plan (source_plan.dispatch_table).
    me: This shard's index (data, as in the layer).
    ep: Expert-parallel width.
    axis_name: The mesh axis the layer's shard_map runs over.

  Returns:
    [recv_rows, ...]: for every (source s, wave w), rows
    send_s[tab_s[me, w] : + tab_s[ep + me, w]] at tab_s[2 * ep + me, w]; rows
    no segment covers are zero (on TPU they are uninitialized and unread).
  """
  waves = disp_tab.shape[1]
  send_alloc = send.shape[0]
  all_send = lax.all_gather(send, axis_name)  # [src, send_alloc, ...]
  all_tab = lax.all_gather(
      disp_tab.astype(jnp.int32), axis_name
  )  # [src, 4ep, W]
  me = jnp.asarray(me, jnp.int32)
  src_off = lax.dynamic_index_in_dim(all_tab, me, axis=1, keepdims=False)
  cnt = lax.dynamic_index_in_dim(all_tab, ep + me, axis=1, keepdims=False)
  dst_off = lax.dynamic_index_in_dim(
      all_tab, 2 * ep + me, axis=1, keepdims=False
  )
  # Segments in (source, wave) order: [1, ep * waves].
  start = dst_off.reshape(1, ep * waves)
  stop = start + cnt.reshape(1, ep * waves)
  row = jnp.arange(recv_rows, dtype=jnp.int32)[:, None]  # [R, 1]
  hit = (row >= start) & (row < stop)  # [R, S]
  seg = jnp.argmax(hit, axis=1).astype(jnp.int32)  # [R]
  covered = jnp.any(hit, axis=1)
  src = seg // waves
  src_row = (
      jnp.take(src_off.reshape(-1), seg) + row[:, 0] - jnp.take(start[0], seg)
  )
  flat = jnp.clip(src * send_alloc + src_row, 0, ep * send_alloc - 1)
  rows = jnp.take(
      all_send.reshape(ep * send_alloc, *send.shape[1:]), flat, axis=0
  )
  mask = covered.reshape((recv_rows,) + (1,) * (send.ndim - 1))
  return jnp.where(mask, rows, jnp.zeros((), send.dtype))


def emulated_list_exchange(tg_lists, me, *, ep, axis_name):
  """The `tg_land` buffer `ragged_a2a_start_waves` fills with sender lists."""
  list_rows = tg_lists.shape[0] // ep
  all_lists = lax.all_gather(
      tg_lists.reshape(ep, list_rows, tg_lists.shape[1]), axis_name
  )
  me = jnp.asarray(me, jnp.int32)
  return lax.dynamic_index_in_dim(
      all_lists, me, axis=1, keepdims=False
  ).reshape(tg_lists.shape)


def emulated_stage_exchange(
    send,
    stage_rows,
    stage_send_tab,
    me,
    *,
    ep,
    num_boards,
    board_size,
    axis_name,
):
  """The `stage` buffer `ragged_a2a_start_waves` fills on each gateway."""
  waves = stage_send_tab.shape[1]
  send_alloc = send.shape[0]
  all_send = lax.all_gather(send, axis_name)  # [src, send_alloc, ...]
  all_tab = lax.all_gather(
      stage_send_tab.astype(jnp.int32), axis_name
  )  # [src, 5*nb, W]
  me = jnp.asarray(me, jnp.int32)
  nb = num_boards
  src_off = all_tab[:, 0:nb, :]  # [src, nb, W]
  cnt = all_tab[:, nb : 2 * nb, :]
  dst_off = all_tab[:, 2 * nb : 3 * nb, :]
  gw = all_tab[:, 3 * nb : 4 * nb, :]
  # Keep only segments whose gateway is `me` (and where src's board != b).
  src_board = jnp.arange(ep, dtype=jnp.int32)[:, None, None] // board_size
  b_idx = jnp.arange(nb, dtype=jnp.int32)[None, :, None]
  on_me = (gw == me) & (src_board != b_idx) & (cnt > 0)
  n_segs = ep * nb * waves
  start = jnp.where(on_me, dst_off, stage_rows + 1).reshape(1, n_segs)
  stop = jnp.where(on_me, dst_off + cnt, stage_rows + 1).reshape(1, n_segs)
  row = jnp.arange(stage_rows, dtype=jnp.int32)[:, None]
  hit = (row >= start) & (row < stop)
  seg = jnp.argmax(hit, axis=1).astype(jnp.int32)
  covered = jnp.any(hit, axis=1)
  src = seg // (nb * waves)
  src_row = (
      jnp.take(src_off.reshape(-1), seg) + row[:, 0] - jnp.take(start[0], seg)
  )
  flat = jnp.clip(src * send_alloc + src_row, 0, ep * send_alloc - 1)
  rows = jnp.take(
      all_send.reshape(ep * send_alloc, *send.shape[1:]), flat, axis=0
  )
  mask = covered.reshape((stage_rows,) + (1,) * (send.ndim - 1))
  return jnp.where(mask, rows, jnp.zeros((), send.dtype))


def emulated_forward(recv, stage, fwd_tab, fwd_cnt, me, *, ep, axis_name):
  """Overlay gateway-forwarded runs onto `recv` using `fwd_tab`/`fwd_cnt`.

  fwd_cnt is the kernel operand [S, W, FWD_SLOTS] (runs per ready slot, stored
  contiguously per (sigma, issue wave)); the dst field's signal-wave bits
  (dedup_plan.FWD_DST_SHIFT) are dropped: timing does not exist here.

  Args:
    recv: The receive buffer.
    stage: This gateway's stage buffer.
    fwd_tab: [S, W, max_runs, 4] forward table.
    fwd_cnt: [S, W, FWD_SLOTS] runs per ready slot.
    me: This device.
    ep: Expert-parallel width.
    axis_name: The mesh axis.

  Returns:
    `recv` with every forwarded run that targets `me` overlaid.
  """
  recv_rows = recv.shape[0]
  stage_rows = stage.shape[0]
  num_slots, waves, max_runs, _ = fwd_tab.shape
  all_stage = lax.all_gather(stage, axis_name)  # [gw, stage_rows, ...]
  all_ftab = lax.all_gather(fwd_tab.astype(jnp.int32), axis_name)
  all_fcnt = lax.all_gather(fwd_cnt.astype(jnp.int32), axis_name)
  if all_fcnt.ndim == 4:  # [gw, S, W, slots] -> runs per (sigma, issue wave)
    all_fcnt = jnp.sum(all_fcnt, axis=-1)
  me = jnp.asarray(me, jnp.int32)
  r_idx = jnp.arange(max_runs, dtype=jnp.int32)[None, None, None, :]
  valid = r_idx < all_fcnt[:, :, :, None]
  stage_row = all_ftab[..., 0]
  rows_len = jnp.where(valid, all_ftab[..., 1], 0)
  dst = all_ftab[..., 2] & dedup_plan_mod.FWD_DST_MASK
  dst_row = all_ftab[..., 3]
  on_me = valid & (dst == me) & (rows_len > 0)
  n_segs = ep * num_slots * waves * max_runs
  start = jnp.where(on_me, dst_row, recv_rows + 1).reshape(1, n_segs)
  stop = jnp.where(on_me, dst_row + rows_len, recv_rows + 1).reshape(1, n_segs)
  row = jnp.arange(recv_rows, dtype=jnp.int32)[:, None]
  hit = (row >= start) & (row < stop)
  seg = jnp.argmax(hit, axis=1).astype(jnp.int32)
  covered = jnp.any(hit, axis=1)
  gw = seg // (num_slots * waves * max_runs)
  s_row = (
      jnp.take(stage_row.reshape(-1), seg) + row[:, 0] - jnp.take(start[0], seg)
  )
  flat = jnp.clip(gw * stage_rows + s_row, 0, ep * stage_rows - 1)
  f_rows = jnp.take(
      all_stage.reshape(ep * stage_rows, *stage.shape[1:]), flat, axis=0
  )
  mask = covered.reshape((recv_rows,) + (1,) * (recv.ndim - 1))
  return jnp.where(mask, f_rows, recv)


def clear_layer_cache() -> None:
  """Drops dedup_ep_moe's cached shard_map callables (and their kernels)."""
  with layer._LAYER_SM_CACHE_LOCK:  # pylint: disable=protected-access
    layer._LAYER_SM_CACHE.clear()  # pylint: disable=protected-access


@contextlib.contextmanager
def emulate_wave_exchange(axis_name: str = kernel.AXIS) -> Iterator[None]:
  """Runs `dedup_ep_moe` without split-phase kernels.

  See the module docstring for what is and is not exercised.

  Args:
    axis_name: The single mesh axis name of the layer's mesh.

  Yields:
    None.
  """
  original_start = dispatch_kernels.ragged_a2a_start_waves
  original_done = dispatch_kernels.ragged_a2a_send_done
  original_build = layer.build_fused_ep_moe_kernel
  original_uninit = wave_dispatch._uninit_hbm  # pylint: disable=protected-access

  def start_waves(
      send,
      recv,
      disp_tab,
      me,
      *,
      ep,
      waves,
      chunk_rows,
      rows,
      perm,
      list_tab,
      tg_lists,
      tg_land,
      stage_send_tab=None,
      stage=None,
      num_boards=0,
      num_slots=0,
      board_size=0,
  ):
    del chunk_rows, list_tab, tg_land
    send = rows[perm]
    filled = emulated_exchange(
        send, recv.shape[0], disp_tab, me, ep=ep, axis_name=axis_name
    )
    out = [
        jnp.zeros((), jnp.int32),
        jnp.zeros((waves,), jnp.int32),
        send,
        filled,
    ]
    filled_lists = emulated_list_exchange(
        tg_lists, me, ep=ep, axis_name=axis_name
    )
    out.extend([
        jnp.zeros((), jnp.int32),
        jnp.zeros((), jnp.int32),
        tg_lists,
        filled_lists,
    ])
    if stage is not None and stage_send_tab is not None:
      filled_stage = emulated_stage_exchange(
          send,
          stage.shape[0],
          stage_send_tab,
          me,
          ep=ep,
          num_boards=num_boards,
          board_size=board_size,
          axis_name=axis_name,
      )
      out.extend([
          jnp.zeros((num_slots * waves,), jnp.int32),
          filled_stage,
      ])
    return tuple(out)

  def send_done(
      send_sem,
      send,
      disp_tab,
      carry,
      *,
      ep,
      list_tab,
      tg_lists,
      list_send_sem,
  ):
    del send_sem, send, disp_tab, ep, list_tab, tg_lists, list_send_sem
    return carry

  def build(**kwargs):
    kwargs.pop("dedup_slots", None)
    kwargs.pop("dedup_fwd_max_runs", None)
    kwargs.pop("dedup_board_size", None)
    # Same tiles and matmuls, no per-wave waits and no gateway forwards.
    fn = original_build(wait_for_exchange=False, **kwargs)
    ep = kwargs.get("ep")

    def call(
        *args,
        disp_tab=None,
        wave_sems=None,
        stage=None,
        stage_tab=None,
        fwd_tab=None,
        fwd_cnt=None,
        stage_sems=None,
        **call_kwargs,
    ):
      del disp_tab, wave_sems, stage_sems, stage_tab
      if stage is not None and fwd_tab is not None and fwd_cnt is not None:
        args = list(args)
        me = call_kwargs["rank"][0, 0]
        args[4] = emulated_forward(
            args[4],
            stage,
            fwd_tab,
            fwd_cnt,
            me,
            ep=ep,
            axis_name=axis_name,
        )
        args = tuple(args)
      return fn(*args, **call_kwargs)

    return call

  clear_layer_cache()
  dispatch_kernels.ragged_a2a_start_waves = start_waves
  dispatch_kernels.ragged_a2a_send_done = send_done
  layer.build_fused_ep_moe_kernel = build
  wave_dispatch._uninit_hbm = jnp.zeros  # pylint: disable=protected-access
  try:
    yield
  finally:
    dispatch_kernels.ragged_a2a_start_waves = original_start
    dispatch_kernels.ragged_a2a_send_done = original_done
    layer.build_fused_ep_moe_kernel = original_build
    wave_dispatch._uninit_hbm = original_uninit  # pylint: disable=protected-access
    clear_layer_cache()


def cpu_devices(n: int) -> list[Any]:
  """The first `n` CPU devices (set `jax_num_cpu_devices` before JAX starts)."""
  devices = jax.devices("cpu")
  if len(devices) < n:
    raise RuntimeError(
        f"need {n} CPU devices and JAX has {len(devices)}; set "
        "jax.config.update('jax_num_cpu_devices', n) before first use"
    )
  return devices[:n]
