# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for Pallas Mosaic TPU Ragged Dot."""

import dataclasses
import functools
from typing import override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import qwix
from tokamax._src import mosaic_tpu as common
from tokamax._src import quantization
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu
from tokamax._src.ops.ragged_dot import test_base


AsQArray = quantization.AsQArray
P = jax.sharding.PartitionSpec


def _is_scale_tiling_supported(x: qwix.QArray, axis: int) -> bool:
  min_addressable_sizes = (
      [1] * x.ndim
      + [common._adaptive_sublane_size(), pltpu.get_tpu_info().num_lanes]
  )[-x.ndim :]
  cdiv = lambda x, y: (x + y - 1) // y
  eps_list = [cdiv(x, y) for x, y in zip(x.qvalue.shape, x.scale.shape)]
  for ax, (mas, eps) in enumerate(zip(min_addressable_sizes, eps_list)):
    if eps != 1 and eps % mas != 0:
      return False
    if ax != axis and not (eps == 1 or eps == x.qvalue.shape[ax]):
      return False
  return True


def _is_config_supported(
    lhs: jax.Array | qwix.QArray | AsQArray,
    rhs: jax.Array | qwix.QArray | AsQArray,
    config: pallas_mosaic_tpu.Config,
) -> bool:
  (m, k), (_, _, n) = lhs.shape, rhs.shape
  if m < config.tile_m or k < config.tile_k or n < config.tile_n:
    return False

  lhs_ = jax.eval_shape(quantization.as_array_or_qarray, lhs)
  rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)

  if isinstance(lhs_, qwix.QArray) and not _is_scale_tiling_supported(lhs_, 1):
    return False
  if isinstance(rhs_, qwix.QArray) and not _is_scale_tiling_supported(rhs_, 1):
    return False
  return True


# TODO : Add QWIX tests for ragged dot once QWIX is in Ragged Dot.
# TODO: Merge QWIX quantization tests into ragged dot API tests.
# also add shapes which tile sizes do not cleanly divide to test masking.
class PallasMosaicTpuRaggedDotTest(test_base.RaggedDotTestBase):
  """Pallas Mosaic TPU Ragged Dot tests."""

  def __init__(self, *args):

    def fn(lhs, rhs, *, config=None, **kwargs):
      config = config or pallas_mosaic_tpu.Config()
      op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)

      # skip unsupported tiling and quantization
      if _is_config_supported(lhs, rhs, config):
        return op(lhs, rhs, **kwargs)

      with self.assertRaises(NotImplementedError) as e:
        _ = op(lhs, rhs, **kwargs)
      self.skipTest(f"Test not supported: {e.msg}")

    super().__init__(*args, dot_fn=fn)

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @override
  def _test_simple(self, dtype):
    with test_base.override_chex_args(atol=1e-6):
      super()._test_simple(dtype)

  def test_vjp0(self):
    with test_base.override_chex_args(atol=0.2, rtol=0.01):
      super().test_vjp0()  # pyrefly: ignore[missing-attribute]

  def test_shard_map_manual_axis_type_varying(self):
    if jax.devices()[0].device_kind.startswith("TPU7x"):
      self.skipTest("Skip the test on TPU v7.")
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices, 1), ("x", "y"))
    # NB: Must not directly run jax.set_mesh in test method.
    # jax.set_mesh(mesh) sets a global mesh for the entire process but never
    # clears it. If this test runs before other tests, the global mesh remains
    # active and causes other tests to fail, while if you only run that failing
    # test, it passes.
    with jax.set_mesh(mesh):
      m, k, n = 256, 128, 128
      num_groups = 1

      lhs = jnp.ones((m, k))
      lhs = jax.device_put(
          lhs,
          P(
              "x",
          ),
      )
      rhs = jnp.ones((num_groups, k, n))
      rhs = jax.device_put(rhs, P(None, None, "y"))
      group_sizes = jnp.array([m], jnp.uint32)

      config = pallas_mosaic_tpu.Config(tile_m=128, tile_k=128, tile_n=128)
      op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)

      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(P("x", None), P(None, None, "y"), P(None)),
          out_specs=P("x", "y"),
          check_vma=True,
      )
      def f(lhs, rhs, sizes):
        manual_axis_type = jax.sharding.ManualAxisType(varying={"x", "y"})
        res = op(lhs, rhs, group_sizes=sizes, manual_axis_type=manual_axis_type)
        self.assertEqual(jax.typeof(res).manual_axis_type, manual_axis_type)
        return res

      _, vjpfunc = jax.vjp(
          f,
          lhs,
          rhs,
          group_sizes,
      )
      cotangent = jax.random.normal(
          jax.random.key(0), (m, n), dtype=jnp.float32
      )
      cotangent = jax.device_put(cotangent, P("x", "y"))
      vjpfunc(cotangent)[0].block_until_ready()

  def test_shard_map_manual_axis_type_unreduced(self):
    self.skipTest("Test is failing.")  # FIXME

    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices, 1), ("x", "y"))
    with jax.set_mesh(mesh):
      m, k, n = 256, 256, 256
      num_groups = 1

      lhs = jnp.ones((m, k))
      lhs = jax.device_put(lhs, P(None, "x"))
      rhs = jnp.ones((num_groups, k, n))
      rhs = jax.device_put(rhs, P(None, "x", None))
      group_sizes = jnp.array([m], jnp.uint32)

      config = pallas_mosaic_tpu.Config(tile_m=128, tile_k=128, tile_n=128)
      op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)

      @jax.jit
      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(P(None, "x"), P(None, "x", None), P(None)),
          out_specs=P(None, None, unreduced={"x"}),
          check_vma=True,
      )
      def f(lhs, rhs, sizes):
        manual_axis_type = jax.sharding.ManualAxisType(unreduced={"x"})
        res = op(lhs, rhs, group_sizes=sizes, manual_axis_type=manual_axis_type)
        self.assertEqual(jax.typeof(res).manual_axis_type, manual_axis_type)
        return res

      _, vjpfunc = jax.vjp(
          f,
          lhs,
          rhs,
          group_sizes,
      )
      cotangent = jax.random.normal(
          jax.random.key(0), (m, n), dtype=jnp.float32
      )
      cotangent = jax.device_put(cotangent, P(None, None, reduced={"x"}))
      vjpfunc(cotangent)[0].block_until_ready()

  @override
  def _test_quantized(
      self,
      a_dtype,
      b_dtype,
      a_tile_shape,
      b_tile_shape,
      use_as_qarray,
      activation=None,
      # (num_groups, m, k, n)
      task=(8, 512, 256, 512),
  ):
    with test_base.override_chex_args(atol=0.4, rtol=0.1):
      super()._test_quantized(
          a_dtype,
          b_dtype,
          a_tile_shape,
          b_tile_shape,
          use_as_qarray,
          activation,
          task,
      )

  @override
  def _test_bench(self, spec):
    if "i8xi8" in self._testMethodName:
      kwargs = dict(atol=2.0, rtol=0.5)  # This is really bad!
    elif "i4" in self._testMethodName:
      kwargs = dict(atol=0.7, rtol=0.1)
    else:
      kwargs = {}
    with test_base.override_chex_args(**kwargs):
      super()._test_bench(spec)

  def test_autotuning_configs(self):
    tpu_ragged_dot = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    ba = op_lib.BoundArguments(
        op=tpu_ragged_dot,
        arguments={
            "lhs": jax.ShapeDtypeStruct((262144, 7168), dtype=jnp.bfloat16),
            "rhs": jax.ShapeDtypeStruct((256, 7168, 2048), dtype=jnp.bfloat16),
        },
    )
    autotuning_configs = ba.autotuning_configs
    self.assertGreaterEqual(len(autotuning_configs), 3 * 3 * 3)

  @parameterized.parameters(
      (128, 128, 128),
      (1, 2048, 512),
      (65536, 2048, 512),
      (65536, 16384, 512),
      (65536, 2048, 16384),
      (65536, 32768, 32768),
  )
  def test_heuristics_config(self, m, k, n):
    num_groups = 256
    num_lanes = pltpu.get_tpu_info().num_lanes

    tpu_ragged_dot = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()

    ba_fwd = op_lib.BoundArguments(
        op=tpu_ragged_dot,
        arguments={
            "lhs": jax.ShapeDtypeStruct((m, k), dtype=jnp.bfloat16),
            "rhs": jax.ShapeDtypeStruct((num_groups, k, n), dtype=jnp.bfloat16),
            "group_sizes": base.generate_group_sizes(
                m=m, num_groups=num_groups
            ),
        },
    )
    fwd_heuristics = ba_fwd.heuristics_config
    self.assertIsInstance(fwd_heuristics, pallas_mosaic_tpu.Config)
    self.assertLessEqual(fwd_heuristics.tile_m, min(m, 8 * num_lanes))
    self.assertLessEqual(fwd_heuristics.tile_k, k)
    self.assertLessEqual(fwd_heuristics.tile_n, n)
    self.assertEqual(fwd_heuristics.input_buffer_count, 2)
    # Deflation order GMM: m -> n -> k.
    if fwd_heuristics.tile_n < n:
      self.assertEqual(fwd_heuristics.tile_m, num_lanes)
    if fwd_heuristics.tile_k < k:
      self.assertEqual(fwd_heuristics.tile_n, num_lanes)

    ba_vjp_dlhs = op_lib.BoundArguments(
        op=tpu_ragged_dot,
        arguments={
            "lhs": jax.ShapeDtypeStruct((m, n), dtype=jnp.bfloat16),
            "rhs": jax.ShapeDtypeStruct((num_groups, k, n), dtype=jnp.bfloat16),
            "group_sizes": base.generate_group_sizes(
                m=m, num_groups=num_groups
            ),
            "ragged_dot_dimension_numbers": (
                pallas_mosaic_tpu.DLHS_RAGGED_DOT_DIM_NUMS
            ),
        },
    )
    dlhs_heuristics = ba_vjp_dlhs.heuristics_config
    self.assertIsInstance(dlhs_heuristics, pallas_mosaic_tpu.Config)
    self.assertLessEqual(dlhs_heuristics.tile_m, min(m, 8 * num_lanes))
    self.assertLessEqual(dlhs_heuristics.tile_k, k)
    self.assertLessEqual(dlhs_heuristics.tile_n, n)
    self.assertEqual(dlhs_heuristics.input_buffer_count, 2)
    # Deflation order GMM: m -> n -> k.
    if dlhs_heuristics.tile_n < n:
      self.assertEqual(dlhs_heuristics.tile_m, num_lanes)
    if dlhs_heuristics.tile_k < k:
      self.assertEqual(dlhs_heuristics.tile_n, num_lanes)

    # Try DRHS VJP with heuristics config.
    ba_vjp_drhs = op_lib.BoundArguments(
        op=tpu_ragged_dot,
        arguments={
            "lhs": jax.ShapeDtypeStruct((m, k), dtype=jnp.bfloat16),
            "rhs": jax.ShapeDtypeStruct((m, n), dtype=jnp.bfloat16),
            "group_sizes": base.generate_group_sizes(
                m=m, num_groups=num_groups
            ),
            "ragged_dot_dimension_numbers": (
                pallas_mosaic_tpu.DRHS_RAGGED_DOT_DIM_NUMS
            ),
        },
    )
    drhs_heuristics = ba_vjp_drhs.heuristics_config
    self.assertIsInstance(drhs_heuristics, pallas_mosaic_tpu.Config)
    self.assertLessEqual(drhs_heuristics.tile_m, min(m, 8 * num_lanes))
    self.assertLessEqual(drhs_heuristics.tile_k, k)
    self.assertLessEqual(drhs_heuristics.tile_n, n)
    self.assertEqual(drhs_heuristics.input_buffer_count, 2)
    # TGMM picks the largest (k, n) tile that fits with
    # tile_m <= 2 * num_lanes, then the largest tile_m.
    if drhs_heuristics.tile_n < n:
      self.assertEqual(drhs_heuristics.tile_k, num_lanes)
    if drhs_heuristics.tile_m < min(m, 2 * num_lanes):
      self.assertEqual(drhs_heuristics.tile_k, num_lanes)
      self.assertEqual(drhs_heuristics.tile_n, num_lanes)

  @parameterized.parameters(
      (32, 196608, 1536, 1280),
      (16, 131072, 1280, 1536),
      (128, 98304, 2048, 1280),
      (32, 147456, 1280, 2048),
      (8, 8192, 4096, 14336),
  )
  def test_drhs_heuristics_config_compiles(
      self, num_groups: int, m: int, k: int, n: int
  ):
    lhs = jax.ShapeDtypeStruct((m, k), dtype=jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((m, n), dtype=jnp.bfloat16)
    group_sizes = jax.ShapeDtypeStruct((num_groups,), dtype=jnp.int32)
    dims = pallas_mosaic_tpu.DRHS_RAGGED_DOT_DIM_NUMS
    config = op_lib.BoundArguments(
        op=pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(),
        arguments={
            "lhs": lhs,
            "rhs": rhs,
            "group_sizes": group_sizes,
            "ragged_dot_dimension_numbers": dims,
        },
    ).heuristics_config
    op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)
    fn = functools.partial(op, ragged_dot_dimension_numbers=dims)
    # Compilation fails if the kernel exceeds the scoped VMEM limit.
    jax.jit(fn).lower(lhs, rhs, group_sizes=group_sizes).compile()


_MIB = 1024 * 1024

# Generation and VMEM capacity per TPU, as reported by `pltpu.get_tpu_info()`.
_TPU_INFOS = {
    "v5e": (5, 128 * _MIB),
    "v5p": (5, 64 * _MIB),
    "v6e": (6, 128 * _MIB),
    "tpu7x": (7, 64 * _MIB),
}


class TgmmHeuristicsTest(parameterized.TestCase):
  """Tests for the tgmm (drhs) heuristics, which don't need a TPU."""

  def _drhs_bound_args(
      self,
      m: int,
      k: int,
      n: int,
      *,
      num_groups: int = 32,
      preferred_element_type: jax.typing.DTypeLike | None = None,
  ) -> op_lib.BoundArguments:
    return op_lib.BoundArguments(
        op=pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(),
        arguments={
            "lhs": jax.ShapeDtypeStruct((m, k), dtype=jnp.bfloat16),
            "rhs": jax.ShapeDtypeStruct((m, n), dtype=jnp.bfloat16),
            "group_sizes": base.generate_group_sizes(
                m=m, num_groups=num_groups
            ),
            "ragged_dot_dimension_numbers": (
                pallas_mosaic_tpu.DRHS_RAGGED_DOT_DIM_NUMS
            ),
            "preferred_element_type": preferred_element_type,
        },
    )

  def _mock_tpu_info(self, tpu: str, num_lanes: int = 128) -> pltpu.TpuInfo:
    generation, vmem_capacity_bytes = _TPU_INFOS[tpu]
    return dataclasses.replace(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1),
        generation=generation,
        vmem_capacity_bytes=vmem_capacity_bytes,
        num_lanes=num_lanes,
    )

  def _heuristics_config(
      self, ba: op_lib.BoundArguments, tpu: str, num_lanes: int = 128
  ) -> pallas_mosaic_tpu.Config:
    tpu_info = self._mock_tpu_info(tpu, num_lanes)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=tpu_info):
      return ba.heuristics_config

  def _vmem_bytes(
      self,
      config: pallas_mosaic_tpu.Config,
      out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ) -> int:
    return pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()._tgmm_vmem_bytes(
        (config.tile_m, config.tile_k, config.tile_n),
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.bfloat16,
        out_dtype=out_dtype,
        input_buffer_count=config.input_buffer_count,
    )

  def test_vmem_bytes_counts_f32_accumulator_and_output_buffers(self):
    # This tiling needs 32.09 MiB of scoped VMEM on TPU7x (32 MiB limit).
    config = pallas_mosaic_tpu.Config(tile_m=1024, tile_k=1536, tile_n=1280)
    inputs = 2 * 1024 * (1536 + 1280) * 2 + 2 * 1024 * 1536 * 2
    self.assertEqual(self._vmem_bytes(config), inputs + 1536 * 1280 * 8)
    self.assertEqual(
        self._vmem_bytes(config, jnp.float32), inputs + 1536 * 1280 * 12
    )
    self.assertEqual(self._vmem_bytes(config), 32 * _MIB)

  @parameterized.parameters(
      # Fits the full (k, n) output tile with tile_m = 512.
      (196608, 1536, 1280, 32, (512, 1536, 1280)),
      (196608, 1280, 1536, 32, (512, 1280, 1536)),
      (131072, 1536, 1280, 16, (512, 1536, 1280)),
      (131072, 1280, 1536, 128, (512, 1280, 1536)),
      # Shrinks tile_k to fit with tile_m = 256, then grows tile_m.
      (98304, 2048, 1280, 128, (1024, 1024, 1280)),
      (98304, 1280, 2048, 128, (1024, 640, 2048)),
      (8192, 4096, 14336, 8, (512, 128, 7168)),
  )
  def test_tpu7x_drhs_config(
      self,
      m: int,
      k: int,
      n: int,
      num_groups: int,
      expected_tiling: tuple[int, int, int],
  ):
    ba = self._drhs_bound_args(m, k, n, num_groups=num_groups)
    config = self._heuristics_config(ba, "tpu7x")
    self.assertEqual(
        (config.tile_m, config.tile_k, config.tile_n), expected_tiling
    )
    self.assertEqual(config.input_buffer_count, 2)

  def test_f32_output_shrinks_tiles(self):
    m, k, n = 196608, 1536, 1280
    bf16_config = self._heuristics_config(
        self._drhs_bound_args(m, k, n), "tpu7x"
    )
    f32_config = self._heuristics_config(
        self._drhs_bound_args(m, k, n, preferred_element_type=jnp.float32),
        "tpu7x",
    )
    self.assertEqual(
        (bf16_config.tile_m, bf16_config.tile_k, bf16_config.tile_n),
        (512, 1536, 1280),
    )
    self.assertEqual(
        (f32_config.tile_m, f32_config.tile_k, f32_config.tile_n),
        (1024, 768, 1280),
    )
    self.assertLessEqual(self._vmem_bytes(f32_config, jnp.float32), 24 * _MIB)

  @parameterized.product(
      tpu=tuple(_TPU_INFOS),
      mkn=(
          (8, 4096, 4096),
          (4096, 4096, 4096),
          (131072, 1536, 1280),
          (131072, 7168, 1024),
          (262144, 256, 7168),
          (327680, 2880, 2880),
          (8192, 4096, 14336),
      ),
  )
  def test_drhs_config_fits_scoped_vmem(
      self, tpu: str, mkn: tuple[int, int, int]
  ):
    m, k, n = mkn
    config = self._heuristics_config(self._drhs_bound_args(m, k, n), tpu)
    generation, vmem_capacity_bytes = _TPU_INFOS[tpu]
    scoped_vmem_bytes = vmem_capacity_bytes / {5: 8, 6: 4}.get(generation, 2)
    self.assertLessEqual(self._vmem_bytes(config), 0.75 * scoped_vmem_bytes)
    self.assertLessEqual(config.tile_m, min(m, 1024))
    self.assertLessEqual(config.tile_k, k)
    self.assertLessEqual(config.tile_n, n)

  @parameterized.product(
      dims=("fwd", "dlhs", "drhs"),
      mkn=(
          (196608, 1536, 1280),
          (65536, 7168, 2048),
          (8192, 4096, 14336),
          (262144, 256, 7168),
      ),
  )
  def test_configs_respect_num_lanes(
      self, dims: str, mkn: tuple[int, int, int]
  ):
    m, k, n = mkn
    num_lanes, num_groups = 256, 32
    lhs_shape, rhs_shape, dim_nums = {
        "fwd": (
            (m, k),
            (num_groups, k, n),
            pallas_mosaic_tpu.DEFAULT_RAGGED_DOT_DIM_NUMS,
        ),
        "dlhs": (
            (m, n),
            (num_groups, k, n),
            pallas_mosaic_tpu.DLHS_RAGGED_DOT_DIM_NUMS,
        ),
        "drhs": ((m, k), (m, n), pallas_mosaic_tpu.DRHS_RAGGED_DOT_DIM_NUMS),
    }[dims]
    arguments = {
        "lhs": jax.ShapeDtypeStruct(lhs_shape, dtype=jnp.bfloat16),
        "rhs": jax.ShapeDtypeStruct(rhs_shape, dtype=jnp.bfloat16),
        "group_sizes": base.generate_group_sizes(m=m, num_groups=num_groups),
        "ragged_dot_dimension_numbers": dim_nums,
    }
    op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    ba = op_lib.BoundArguments(op=op, arguments=arguments)
    tpu_info = self._mock_tpu_info("tpu7x", num_lanes)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=tpu_info):
      config = ba.heuristics_config
      tiling = (config.tile_m, config.tile_k, config.tile_n)
      if dims == "drhs":
        fits = self._vmem_bytes(config) <= 0.75 * 32 * _MIB
      else:
        tile_m, tile_k, tile_n = tiling
        fits = op._fit_within_tpu_vmem(
            [(tile_m, tile_k, jnp.bfloat16), (tile_k, tile_n, jnp.bfloat16)],
            (tile_m, tile_n, jnp.bfloat16),
            config.input_buffer_count,
        )
    self.assertTrue(fits, tiling)
    self.assertLessEqual(config.tile_m, 8 * num_lanes)
    for tile in tiling:
      self.assertGreaterEqual(tile, num_lanes)
      self.assertEqual(tile % num_lanes, 0, tiling)


if __name__ == "__main__":
  absltest.main()
