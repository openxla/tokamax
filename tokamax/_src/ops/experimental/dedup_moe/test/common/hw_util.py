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
"""Dual-mode test harness: CPU interpret mode, or TPU v7x / v8i hardware.

Every kernel-executing test of the De-Dup MoE layer runs in one of two modes:

  * CPU (the default on a CPU host): the Pallas kernels run in TPU interpret
    mode on CPU devices and the split-phase exchange is emulated
    (interpret_util).
  * Hardware (the test runs on TPU v7x or v8i devices, `jax.devices()`): every
    context manager here is a no-op, so the real Mosaic kernels and the real
    split-phase exchange run, and the layer is jitted (`run`).

Hardware tests run on TPU v7x and v8i only: `skip_unless_supported` skips
them on any other host.
"""

import contextlib
import os
import sys
from typing import Any, Callable, Iterator
import unittest

from absl import flags
import jax
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import config as kernel_config
from tokamax._src.ops.experimental.dedup_moe.test.common import interpret_util
from tokamax._src.ops.experimental.fused_moe import host as fused_host

AXIS = kernel_config.AXIS
# The device kind prefixes of TPU v7x and TPU v8i devices.
V7X_DEVICE_KIND = "TPU7x"
V8I_DEVICE_KIND = "TPU8i"
# The TPUs the hardware tests run on (the kernels need TPU v7x or later).
SUPPORTED_DEVICE_KINDS = (V7X_DEVICE_KIND, V8I_DEVICE_KIND)
# Fewest TPU devices the hardware tests run on: wave dispatch needs ep >= 3,
# so a one-chip (two-device) host skips them.
MIN_HW_DEVICES = 4

# Hardware-legal layer shapes for the small dual-mode tests (the CPU interpret
# shapes stay as each test defines them). The fused kernel streams the int32
# token-gather table from HBM in 128-row aligned windows, so on hardware the
# tile height (capacity) must be a multiple of 128. hidden must be whole
# 128-lane blocks; 1024 keeps the intermediate at two 512-column chunks.
HW_CAPACITY = fused_host.TOKEN_GATHER_DMA_ALIGNMENT
HW_HIDDEN = 1024
HW_INTER = 1024

_DEVICES: list[Any] | None = None


def on_hardware() -> bool:
  """Whether the tests run on TPU devices.

  Must be called after absl has parsed flags (from setUp or a test, not at
  module import or in parameterized case lists): JAX cannot be initialized
  earlier, and the platform is only known once it is.

  Raises:
    RuntimeError: If called before absl.app.run.

  Returns:
    True on TPU devices.
  """
  if not flags.FLAGS.is_parsed():
    raise RuntimeError(
        "hw_util.on_hardware()/pick()/hw_shapes() must be called after"
        " absl.app.run (in setUp or a test), not at import time"
    )
  return jax.default_backend() == "tpu"


def on_v8i() -> bool:
  """Whether the tests run on TPU v8i devices."""
  return on_hardware() and jax.devices()[0].device_kind.startswith(
      V8I_DEVICE_KIND
  )


def on_supported_tpu() -> bool:
  """Whether the tests run on TPU v7x or TPU v8i devices."""
  return on_hardware() and jax.devices()[0].device_kind.startswith(
      SUPPORTED_DEVICE_KINDS
  )


def unsupported_reason() -> str | None:
  """Why this host can run neither mode, or None if it can run one.

  The tests run on TPU v7x or v8i hardware with at least MIN_HW_DEVICES
  devices, or in CPU interpret mode on a host whose default JAX backend is the
  CPU. Any other host (a GPU, an older TPU generation, or a one-chip TPU host)
  skips them.

  Returns:
    The skip reason, or None on TPU v7x / v8i or a CPU-default host.
  """
  backend = jax.default_backend()
  if backend == "cpu":
    return None
  kind = jax.devices()[0].device_kind
  if not on_supported_tpu():
    return (
        "dedup_moe kernel tests need TPU v7x / v8i or the CPU interpret"
        f" backend; got {backend}/{kind}"
    )
  if len(jax.devices()) < MIN_HW_DEVICES:
    return (
        f"dedup_moe kernel tests need >= {MIN_HW_DEVICES} TPU devices (wave"
        f" dispatch needs ep >= 3); got {len(jax.devices())} {kind}"
    )
  return None


def skip_unless_supported(test: Any) -> None:
  """Skips `test` unless it runs on TPU v7x / v8i or in CPU interpret mode."""
  reason = unsupported_reason()
  if reason is not None:
    test.skipTest(reason)


def require_v8i(test: Any) -> None:
  """Skips `test` unless it runs on TPU v8i hardware."""
  if not on_v8i():
    test.skipTest("TPU v8i only.")


def mode_name() -> str:
  return "tpu" if on_hardware() else "cpu-interpret"


def hw_shapes() -> bool:
  """Whether the hardware shape choices apply (`pick`, `check_capacity`).

  True on hardware, and on CPU with --hw_shapes (or DEDUP_MOE_HW_SHAPES=1): a
  CPU interpret pre-check of the shapes the hardware targets will run.

  Returns:
    Whether to use the hardware shapes.
  """
  if on_hardware():
    return True
  if os.environ.get("DEDUP_MOE_HW_SHAPES", "").lower() in ("1", "true", "yes"):
    return True
  return any(
      a in ("--hw_shapes", "--hw_shapes=true", "--hw_shapes=1")
      for a in sys.argv[1:]
  )


def pick(cpu: Any, hw: Any) -> Any:
  """`hw` under hardware shapes (hw_shapes()), else `cpu`."""
  return hw if hw_shapes() else cpu


def check_capacity(capacity: int) -> int:
  """Returns `capacity`; under hardware shapes it must be a multiple of 128."""
  if hw_shapes() and capacity % HW_CAPACITY:
    raise ValueError(
        f"capacity {capacity} cannot compile on TPU: it must be a multiple"
        f" of {HW_CAPACITY}"
    )
  return capacity


def _coords_key(d: Any) -> tuple[int, ...]:
  """Sort key that keeps the devices of a chip and of a board adjacent."""
  coords = tuple(getattr(d, "coords", None) or ())
  core = getattr(d, "core_on_chip", 0) or 0
  return tuple(reversed(coords)) + (core, d.id)


def setup_module(chip: str = "tpu8i") -> None:
  """The tests' setUpModule: fake the chip record in CPU interpret mode.

  Args:
    chip: The chip record for CPU interpret mode (interpret_util.CHIPS).

  Raises:
    unittest.SkipTest: On a host that is neither TPU v7x / v8i nor
      CPU-default.
  """
  reason = unsupported_reason()
  if reason is not None:
    raise unittest.SkipTest(reason)
  if not on_hardware():
    interpret_util.register_cpu_chip(chip)


def all_devices() -> list[Any]:
  """Every usable device: the TPUs sorted by coordinates, or the CPUs."""
  global _DEVICES
  if _DEVICES is None:
    if on_hardware():
      _DEVICES = sorted(jax.devices(), key=_coords_key)
    else:
      _DEVICES = list(jax.devices("cpu"))
  return _DEVICES


def num_devices() -> int:
  return len(all_devices())


def devices(ep: int) -> list[Any]:
  """The first `ep` devices."""
  if not on_hardware():
    return interpret_util.cpu_devices(ep)
  devs = all_devices()
  if len(devs) < ep:
    raise RuntimeError(f"need {ep} TPU devices and the host has {len(devs)}")
  return devs[:ep]


def fits(ep: int) -> bool:
  return ep <= num_devices()


def skip_unless_fits(test: Any, ep: int) -> None:
  """Skips `test` when there are fewer than `ep` devices."""
  if not fits(ep):
    test.skipTest(f"needs {ep} devices; {mode_name()} has {num_devices()}")


def make_mesh(ep: int, test: Any = None, axis: str = AXIS) -> Any:
  """A one-axis EP mesh over `devices(ep)`; skips `test` if it does not fit."""
  if test is not None:
    skip_unless_fits(test, ep)
  return jax.sharding.Mesh(np.asarray(devices(ep)), axis_names=(axis,))


def skip_on_hardware(test: Any, reason: str) -> None:
  if on_hardware():
    test.skipTest(f"CPU interpret mode only: {reason}")


def host_only() -> Any:
  """Context: pure host or plan logic runs on the host CPU, not the TPUs.

  Returns:
    A context manager.
  """
  if not on_hardware():
    return contextlib.nullcontext()
  return jax.default_device(jax.devices("cpu")[0])


def skip_unless_hardware(test: Any, reason: str) -> None:
  if not on_hardware():
    test.skipTest(f"TPU hardware only: {reason}")


@contextlib.contextmanager
def interpret_pallas(params: Any = None) -> Iterator[Any]:
  """interpret_util.interpret_pallas on CPU; the real kernels on hardware."""
  if on_hardware():
    yield None
  else:
    with interpret_util.interpret_pallas(params) as p:
      yield p


@contextlib.contextmanager
def emulate_wave_exchange(axis_name: str = AXIS) -> Iterator[None]:
  """interpret_util.emulate_wave_exchange on CPU; the real split phase on TPU."""
  if on_hardware():
    yield
  else:
    with interpret_util.emulate_wave_exchange(axis_name):
      yield


@contextlib.contextmanager
def cpu_layer(*, emulate: bool = True, axis_name: str = AXIS) -> Iterator[None]:
  """The layer harness: interpret mode (+ emulated exchange) on CPU.

  Args:
    emulate: Emulate the split-phase exchange on CPU.
    axis_name: The mesh axis.

  Yields:
    None. Nothing is patched on hardware.
  """
  with contextlib.ExitStack() as stack:
    if not on_hardware():
      stack.enter_context(interpret_util.interpret_pallas())
      if emulate:
        stack.enter_context(interpret_util.emulate_wave_exchange(axis_name))
    yield


def reset_interpret_state() -> None:
  """interpret_util.reset_interpret_state on CPU (nothing to reset on TPU)."""
  if not on_hardware():
    interpret_util.reset_interpret_state()


def run(fn: Callable[..., Any], *args: Any) -> Any:
  """Calls `fn(*args)`; on hardware inside jax.jit.

  Args:
    fn: The function; static configuration must be closed over and every array
      (weights included) passed in `args`, so nothing large becomes a jit
      constant.
    *args: Array (pytree) arguments.

  Returns:
    fn(*args).
  """
  if not on_hardware():
    return fn(*args)
  return jax.jit(fn)(*args)
