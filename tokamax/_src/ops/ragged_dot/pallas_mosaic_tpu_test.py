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


_TPU_CHIPS = {
    "v5e": pltpu.ChipVersion.TPU_V5E,
    "v5p": pltpu.ChipVersion.TPU_V5P,
    "v6e": pltpu.ChipVersion.TPU_V6E,
    "tpu7x": pltpu.ChipVersion.TPU_7X,
    "tpu8i": pltpu.ChipVersion.TPU_8I,
}


class TgmmHeuristicsTest(parameterized.TestCase):
  """Tests for the tgmm (drhs) heuristics using mocked TPU info."""

  def _drhs_bound_args(
      self,
      m: int,
      k: int,
      n: int,
      *,
      num_groups: int = 32,
      lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      rhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      preferred_element_type: jax.typing.DTypeLike | None = None,
  ) -> op_lib.BoundArguments:
    return op_lib.BoundArguments(
        op=pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(),
        arguments={
            "lhs": jax.ShapeDtypeStruct((m, k), dtype=lhs_dtype),
            "rhs": jax.ShapeDtypeStruct((m, n), dtype=rhs_dtype),
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
    return dataclasses.replace(
        pltpu.get_tpu_info_for_chip(_TPU_CHIPS[tpu], 1),
        num_lanes=num_lanes,
    )

  def _vmem_bytes(
      self,
      config: pallas_mosaic_tpu.Config,
      *,
      lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      rhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ) -> int:
    return pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()._tgmm_vmem_bytes(
        (config.tile_m, config.tile_k, config.tile_n),
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        out_dtype=out_dtype,
        input_buffer_count=config.input_buffer_count,
    )

  @parameterized.named_parameters(
      ("v5e", "v5e", 16),
      ("v5p", "v5p", 16),
      ("v6e", "v6e", 32),
      ("tpu7x", "tpu7x", 32),
      ("tpu8i", "tpu8i", 96),
  )
  def test_scoped_vmem_capacity_bytes(self, tpu: str, expected_mib: int):
    tpu_info = self._mock_tpu_info(tpu)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=tpu_info):
      self.assertEqual(
          pallas_mosaic_tpu._scoped_vmem_capacity_bytes(),
          expected_mib * 1024 * 1024,
      )

  @parameterized.product(
      tpu=tuple(_TPU_CHIPS),
      mkn=(
          (8, 4096, 4096),
          (4096, 4096, 4096),
          (98304, 2048, 1280),
          (98304, 1280, 2048),
          (131072, 1536, 1280),
          (131072, 1280, 1536),
          (196608, 1536, 1280),
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
    ba = self._drhs_bound_args(m, k, n)
    tpu_info = self._mock_tpu_info(tpu)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=tpu_info):
      config = ba.heuristics_config
      scoped_vmem_bytes = pallas_mosaic_tpu._scoped_vmem_capacity_bytes()
    self.assertLessEqual(self._vmem_bytes(config), 0.75 * scoped_vmem_bytes)
    self.assertLessEqual(config.tile_m, min(m, 8 * tpu_info.num_lanes))
    self.assertLessEqual(config.tile_k, k)
    self.assertLessEqual(config.tile_n, n)
    self.assertEqual(config.input_buffer_count, 2)

  @parameterized.product(
      tpu=tuple(_TPU_CHIPS),
      dtypes=(
          (jnp.bfloat16, jnp.bfloat16, jnp.float32),
          (jnp.bfloat16, jnp.float32, None),
          (jnp.float32, jnp.bfloat16, None),
      ),
  )
  def test_drhs_f32_output_fits_scoped_vmem(
      self,
      tpu: str,
      dtypes: tuple[
          jax.typing.DTypeLike,
          jax.typing.DTypeLike,
          jax.typing.DTypeLike | None,
      ],
  ):
    lhs_dtype, rhs_dtype, preferred_element_type = dtypes
    out_dtype = (
        jnp.result_type(lhs_dtype, rhs_dtype)
        if preferred_element_type is None
        else preferred_element_type
    )
    ba = self._drhs_bound_args(
        196608,
        1536,
        1280,
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        preferred_element_type=preferred_element_type,
    )
    tpu_info = self._mock_tpu_info(tpu)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=tpu_info):
      config = ba.heuristics_config
      scoped_vmem_bytes = pallas_mosaic_tpu._scoped_vmem_capacity_bytes()
    self.assertLessEqual(
        self._vmem_bytes(
            config,
            lhs_dtype=lhs_dtype,
            rhs_dtype=rhs_dtype,
            out_dtype=out_dtype,
        ),
        0.75 * scoped_vmem_bytes,
    )

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
        scoped_vmem_bytes = pallas_mosaic_tpu._scoped_vmem_capacity_bytes()
        fits = self._vmem_bytes(config) <= 0.75 * scoped_vmem_bytes
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
