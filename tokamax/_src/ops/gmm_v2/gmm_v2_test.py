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
import collections
import inspect
import itertools
import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src import config
from tokamax._src import mosaic_tpu
from tokamax._src import test_utils
from tokamax._src.ops.gmm_v2 import gmm_v2
from tokamax._src.ops.gmm_v2 import tgmm_v2
from tokamax._src.ops.gmm_v2 import util

import pytest

jax.config.parse_flags_with_absl()


_GroupConfig = collections.namedtuple(
    "_GroupConfig", ["num_groups", "group_offset", "num_local_groups"]
)

get_group_sizes = util.get_group_sizes
quantize_tensor = util.quantize_tensor


def reference_gmm(
    lhs: jax.Array,  # [m, k]
    rhs: jax.Array,  # [num_groups, k, n]
    group_sizes: jax.Array,  # [num_groups]
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    partial_sum: jax.Array | None = None,  # [m, n]
    group_offset: jax.Array | None = None,  # int32[1]
    lhs_scale: jax.Array | None = None,
):
  num_tokens = lhs.shape[0]
  num_groups, in_size, out_size = rhs.shape
  assert num_groups > 0, f"rhs must have at least 1 group, got {num_groups}"
  assert lhs.shape[1] == in_size

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  if rhs_scale is not None:
    num_blocks = rhs_scale.shape[1]
  else:
    num_blocks = 1
  block_size = in_size // num_blocks

  start = 0
  gmm_out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]

    group = global_group - group_offset[0]
    end = min(start + group_size, num_tokens)
    group_size = end - start
    if 0 <= group and group < num_groups:
      lhs_slice = lhs[start:end]
      rhs_slice = rhs[group]

      out = jnp.array(0.0, dtype=jnp.float32)
      for block in range(num_blocks):
        block_start = block * block_size
        block_end = block_start + block_size
        lhs_block = lhs_slice[:, block_start:block_end].astype(jnp.float32)
        rhs_block = rhs_slice[block_start:block_end, :].astype(jnp.float32)

        acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
        if rhs_scale is not None:
          acc *= rhs_scale[group][block]
        out += acc
      if lhs_scale is not None:
        out *= lhs_scale
      if rhs_bias is not None:
        out = out + rhs_bias[group]
      if partial_sum is not None:
        out = out + partial_sum[start:end]
    else:
      out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)

    gmm_out.append(out.astype(lhs.dtype))
    start = end

  return jnp.concat(gmm_out, axis=0)


# For TGMM quantized case, we want the reference to use the most accurate
# computation of what the kernel is asked to compute
# (f32 and highest precision) as opposed to the hardware-mirroring way
# (reference also does native fp8 dot_general) which we measure noise against
# noise.
def reference_tgmm(
    lhs,  # [k, m]
    rhs,  # [m, n]
    group_sizes,  # [num_groups]
    # num_actual_groups comes from weights.shape[0]
    num_actual_groups,  # int32
    rhs_scale: jax.Array | None = None,
    # group_offset is obtained from
    # jnp.arange(0, num_experts, num_experts_per_shard)
    group_offset=None,
    out_dtype: jnp.dtype | None = None,
):  # [num_groups, k, n]
  # Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :]
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  assert group_sizes.size >= int(group_offset[0]) + num_actual_groups, (
      f"group_sizes.size ({group_sizes.size}) must be >= "
      f"group_offset ({int(group_offset[0])}) + num_actual_groups "
      f"({num_actual_groups})"
  )

  start = 0
  out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]
    group = global_group - group_offset[0]
    end = start + group_size
    if 0 <= group and group < num_actual_groups:
      if rhs_scale is None:
        out.append(lhs[:, start:end] @ rhs[start:end, :])
      else:
        # rhs_scale.shape==(1, 1, N).
        partial = jax.lax.dot_general(
            lhs[:, start:end].astype(jnp.float32),
            rhs[start:end, :].astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        partial *= rhs_scale[0]  # rhs_scale[0]: shape [1, N]
        output_dtype = out_dtype if out_dtype is not None else lhs.dtype
        out.append(partial.astype(output_dtype))
    start = end
  return jnp.stack(out)


# Default per-dtype tolerances, mirroring
# jax._src.public_test_util._default_tolerance. Extend this map if a new output
# dtype is introduced into a default-tolerance assertion.
_DTYPE_TOL = {
    jnp.dtype(jnp.bfloat16): 1e-1,
    jnp.dtype(jnp.float32): 5e-1,
}


def _lookup_tol(dtype):
  key = jnp.dtype(dtype)
  if key not in _DTYPE_TOL:
    raise KeyError(
        f"No default tolerance for dtype {key!r}. "
        "Add it to _DTYPE_TOL or pass explicit atol/rtol."
    )
  return _DTYPE_TOL[key]


def assert_arrays_all_close(actual, desired, *, atol=None, rtol=None):
  if atol is None:
    atol = max(_lookup_tol(actual.dtype), _lookup_tol(desired.dtype))
  if rtol is None:
    rtol = max(_lookup_tol(actual.dtype), _lookup_tol(desired.dtype))
  chex.assert_trees_all_close(actual, desired, atol=atol, rtol=rtol)


def _poison_vmem_all_cores(num_rows: int = 8192) -> None:
  """Fills VMEM scratch across all cores with NaNs to expose stale reads."""
  num_cores = pltpu.get_tpu_info().num_cores

  def _kernel(out_ref, scratch_ref):
    scratch_ref[...] = jnp.full(scratch_ref.shape, jnp.nan, jnp.float32)
    core_id = jax.lax.axis_index("core")
    pltpu.sync_copy(scratch_ref.at[:8, :], out_ref.at[core_id])

  out = pl.kernel(
      _kernel,
      out_type=pltpu.HBM((num_cores, 8, 128), jnp.float32),
      scratch_types=[pltpu.VMEM((num_rows, 128), jnp.float32)],
      mesh=pltpu.TensorCoreMesh(axis_name="core"),
  )()
  jax.block_until_ready(out)


def _reference_fill_metadata(
    group_sizes, group_offset, size_group, tile_m, sublane, max_num_gm
):
  """Python copy of fill_metadata's walk over every group from group 0."""
  group_ids = [-1] * max_num_gm
  m_offsets = [-1] * (max_num_gm + 1)
  m_offsets[0] = 0
  num_gm = 0
  start = 0
  for lhs_group_id in range(group_offset + size_group):
    size = int(group_sizes[lhs_group_id])
    end = start + size
    if size > 0 and lhs_group_id >= group_offset:
      curr_num_gm = math.ceil((size + start % sublane) / tile_m)
      curr = start
      for tm_id in range(num_gm, num_gm + curr_num_gm):
        tm_size = min(tile_m - curr % sublane, end - curr)
        group_ids[tm_id] = lhs_group_id - group_offset
        m_offsets[tm_id] = curr
        m_offsets[tm_id + 1] = curr + tm_size
        curr += tm_size
      num_gm += curr_num_gm
    start = end
  return num_gm, group_ids, m_offsets


def _run_fill_metadata(
    group_sizes, group_offset, cfgs, max_num_gm, chunked=True
):
  """Runs gmm_v2.fill_metadata in a kernel and returns its outputs."""

  def kernel(
      group_sizes_ref, group_offset_ref, num_gm_ref, group_id_ref, m_offset_ref
  ):
    # Sentinels, so that a stray or missing write shows up in the comparison.
    for i in range(max_num_gm):
      group_id_ref[i] = -1
    for i in range(max_num_gm + 1):
      m_offset_ref[i] = -1
    metadata_ref = gmm_v2.MetadataRef(
        gm_id_to_group_id=group_id_ref, gm_id_to_m_offset=m_offset_ref
    )
    num_gm_ref[0] = gmm_v2.fill_metadata(
        group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
        chunked=chunked,
    )

  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((1,), jnp.int32),
          jax.ShapeDtypeStruct((max_num_gm,), jnp.int32),
          jax.ShapeDtypeStruct((max_num_gm + 1,), jnp.int32),
      ),
      in_specs=[smem, smem],
      out_specs=(smem, smem, smem),
  )(group_sizes, group_offset)


class GmmTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16, 32],
      has_bias=[True],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_basic(
      self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1, k2 = jax.random.split(key, 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          k2, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs, rhs, group_sizes, rhs_bias=rhs_bias, group_offset=group_offset
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    assert_arrays_all_close(actual, expected)

  def test_group_offset_python_int(self):
    # Under `jax.disable_jit()` the kernel entry runs untraced and can receive
    # `group_offset` as a Python int. Reproduce that by tracing the unwrapped
    # body inside an outer jit, then compare with an int32[1].
    batch_size, in_size, out_size, num_groups = 128, 512, 512, 4
    group_offset = 1
    num_local_groups = num_groups - group_offset
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    out_grad = jax.random.normal(k2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)

    def run(lhs, rhs, out_grad, group_offset):
      out = inspect.unwrap(gmm_v2.gmm_v2)(
          lhs, rhs, group_sizes, group_offset=group_offset
      )
      drhs = inspect.unwrap(tgmm_v2.tgmm_v2)(
          lhs,
          out_grad,
          group_sizes,
          num_local_groups,
          group_offset=group_offset,
          preferred_element_type=jnp.bfloat16,
      )
      return out, drhs

    def run_with_python_int(lhs, rhs, out_grad):
      return run(lhs, rhs, out_grad, group_offset)

    expected = jax.jit(run)(
        lhs, rhs, out_grad, jnp.array([group_offset], dtype=jnp.int32)
    )
    actual = jax.jit(run_with_python_int)(lhs, rhs, out_grad)
    chex.assert_trees_all_equal(actual, expected)

  @parameterized.product(
      batch_size=[73, 117],
      disable_multi_core_mode=[False, True],
  )
  def test_gmm_unaligned_m(self, batch_size, disable_multi_core_mode):
    in_size = 512
    out_size = 512
    num_groups = 16

    key = jax.random.key(0)
    k0, k1 = jax.random.split(key, 2)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)
    expected = reference_gmm(lhs, rhs, group_sizes)

    with config.disable_multi_core_mode(disable_multi_core_mode):
      actual = gmm_v2.gmm_v2(lhs, rhs, group_sizes)

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_partial_sum(
      self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
  ):
    """Tests the fused residual add, including its interaction with bias."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1, k2, k3 = jax.random.split(key, 4)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          k2, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )
    partial_sum = jax.random.normal(
        k3, (batch_size, out_size), dtype=jnp.bfloat16
    )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        partial_sum=partial_sum,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        partial_sum=partial_sum,
        group_offset=group_offset,
    )

    assert_arrays_all_close(actual, expected)

  def test_gmm_partial_sum_matches_unfused_add(self):
    """A partial sum must be equivalent to adding it after an unfused GMM."""
    batch_size, in_size, out_size, num_groups = 128, 512, 512, 16
    key = jax.random.key(0)
    k0, k1, k2 = jax.random.split(key, 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    partial_sum = jax.random.normal(
        k2, (batch_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)

    fused = gmm_v2.gmm_v2(lhs, rhs, group_sizes, partial_sum=partial_sum)
    unfused = gmm_v2.gmm_v2(lhs, rhs, group_sizes) + partial_sum

    assert_arrays_all_close(fused, unfused)

  @parameterized.parameters(False, True)
  def test_gmm_tile_fn_receives_tiling_request(self, use_partial_sum):
    """A custom `tile_info` callback is called with a `TilingRequest`."""
    batch_size, in_size, out_size, num_groups = 128, 512, 512, 16
    key = jax.random.key(0)
    k0, k1, k2 = jax.random.split(key, 3)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    partial_sum = None
    if use_partial_sum:
      partial_sum = jax.random.normal(
          k2, (batch_size, out_size), dtype=jnp.bfloat16
      )
    group_sizes = get_group_sizes(batch_size, num_groups)
    vmem_limit_bytes = 32 * 1024 * 1024

    requests = []

    def tile_fn(req):
      requests.append(req)
      return gmm_v2.calculate_tiling_from_request(req)

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        partial_sum=partial_sum,
        tile_info=tile_fn,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    self.assertNotEmpty(requests)
    req = requests[0]
    self.assertIsInstance(req, gmm_v2.TilingRequest)
    self.assertEqual(req.dims.size_k, in_size)
    self.assertEqual(req.dims.size_n, out_size)
    self.assertEqual(req.vmem_limit_bytes, vmem_limit_bytes)
    self.assertIsNone(req.fuse_act)
    self.assertEqual(req.has_partial_sum, use_partial_sum)
    expected = gmm_v2.gmm_v2(lhs, rhs, group_sizes, partial_sum=partial_sum)
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[73, 117],
      out_size=[255, 512],
  )
  def test_gmm_partial_sum_unaligned(self, batch_size, out_size):
    """Covers padding of the partial sum."""
    in_size, num_groups = 512, 16
    key = jax.random.key(0)
    k0, k1, k2 = jax.random.split(key, 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    partial_sum = jax.random.normal(
        k2, (batch_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = reference_gmm(lhs, rhs, group_sizes, partial_sum=partial_sum)
    actual = gmm_v2.gmm_v2(lhs, rhs, group_sizes, partial_sum=partial_sum)

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      weight_dtype=[jnp.int8],
      block_size=[256],
      group_offset=[0],
  )
  def test_gmm_activation_weight_quantized_partial_sum(
      self, weight_dtype, block_size, group_offset
  ):
    """Quantized weights and lhs, with a bf16 partial sum carried in."""
    batch_size, in_size, out_size, num_groups = 256, 1024, 512, 16
    num_local_groups = num_groups - group_offset
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.uniform(k0, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        k1, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    partial_sum = jax.random.uniform(
        k2, (batch_size, out_size), jnp.bfloat16, -1, 1
    )
    group_sizes = get_group_sizes(batch_size, num_groups)
    # Rows of non-local groups are zero, with or without a partial sum.
    first_local_row = int(jnp.sum(group_sizes[:group_offset]))
    is_local_row = jnp.arange(batch_size)[:, None] >= first_local_row
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    kwargs = dict(
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
        preferred_element_type=jnp.bfloat16,
    )
    fused = gmm_v2.gmm_v2(
        lhs, rhs_q, group_sizes, partial_sum=partial_sum, **kwargs
    )
    unfused = gmm_v2.gmm_v2(lhs, rhs_q, group_sizes, **kwargs)
    expected = jnp.where(is_local_row, unfused + partial_sum, 0)

    assert_arrays_all_close(fused, expected)

  def test_gmm_partial_sum_unsupported(self):
    lhs = jnp.zeros((128, 512), jnp.bfloat16)
    rhs = jnp.zeros((16, 512, 512), jnp.bfloat16)
    group_sizes = get_group_sizes(128, 16)
    partial_sum = jnp.zeros((128, 512), jnp.bfloat16)

    with self.assertRaisesRegex(NotImplementedError, "fuse_act"):
      gmm_v2.gmm_v2(
          lhs,
          jnp.zeros((16, 512, 1024), jnp.bfloat16),
          group_sizes,
          partial_sum=partial_sum,
          fuse_act="silu",
      )
    with config.disable_multi_core_mode(True):
      with self.assertRaisesRegex(
          NotImplementedError, "disable_multi_core_mode"
      ):
        gmm_v2.gmm_v2(lhs, rhs, group_sizes, partial_sum=partial_sum)
    with self.assertRaisesRegex(ValueError, "must match output dtype"):
      gmm_v2.gmm_v2(
          lhs, rhs, group_sizes, partial_sum=partial_sum.astype(jnp.float32)
      )

  @parameterized.parameters(500, 512)
  def test_partial_sum_is_accumulated_in_place(self, n):
    """The partial sum must be aliased to the output rather than copied.

    Removing the extra HBM round-trip is the point of the feature. If the
    aliasing silently stops being emitted the numerics still pass, so assert it
    directly.
    """
    m, k, num_groups = 512, 512, 8
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(k0, (m, k), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, k, n), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(m, num_groups)
    partial_sum = jnp.zeros((m, n), jnp.bfloat16)

    plain = jax.jit(gmm_v2.gmm_v2)
    plain_text = plain.lower(lhs, rhs, group_sizes).compile().as_text()
    # `as_text` is typed as optional; an empty program would make the control
    # below pass vacuously, so fail loudly instead of defaulting to "".
    assert plain_text is not None
    self.assertEqual(
        plain_text.count("output_to_operand_aliasing"),
        0,
        "control failed: a GMM with no partial sum has nothing to alias, so "
        "the counter is not measuring what this test claims",
    )

    for has_pad_ref in sorted({gmm_v2.HAS_PAD_REF, False}):
      jax.clear_caches()
      with mock.patch.object(gmm_v2, "HAS_PAD_REF", has_pad_ref):
        fused = jax.jit(
            lambda l, r, g, p: gmm_v2.gmm_v2(l, r, g, partial_sum=p)
        )
        fused_text = (
            fused.lower(lhs, rhs, group_sizes, partial_sum).compile().as_text()
        )
      assert fused_text is not None
      self.assertGreater(
          fused_text.count("output_to_operand_aliasing"),
          0,
          "the partial sum was not aliased to the output",
      )
      if has_pad_ref:
        self.assertNotIn("pad(", fused_text)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_transpose_rhs(
      self, batch_size, in_size, out_size, num_groups, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1 = jax.random.split(key, 2)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    # When transpose_rhs=True, rhs shape is
    # (num_local_groups, out_size, in_size)
    rhs = jax.random.normal(
        k1, (num_local_groups, out_size, in_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset_arr = jnp.array(group_offset, dtype=jnp.int32)

    # Reference expects un-transposed RHS (num_local_groups, in_size, out_size)
    rhs_ref = jnp.swapaxes(rhs, 1, 2)
    expected = reference_gmm(
        lhs, rhs_ref, group_sizes, group_offset=group_offset_arr
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset_arr,
        transpose_rhs=True,
    )

    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[8],
      tile_k=[256],
      transpose_rhs=[False, True],
      group_offset=[0],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  @pytest.mark.long
  def test_gmm_multi_k_partial_bucket(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      tile_k,
      transpose_rhs,
      group_offset,
      dtype,
  ):
    """Tests multi-K contraction (num_k > 1) with partial M-buckets."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1 = jax.random.split(key, 2)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=dtype)
    if transpose_rhs:
      rhs = jax.random.normal(
          k1, (num_local_groups, out_size, in_size), dtype=dtype
      )
      rhs_ref = jnp.swapaxes(rhs, 1, 2)
    else:
      rhs = jax.random.normal(
          k1, (num_local_groups, in_size, out_size), dtype=dtype
      )
      rhs_ref = rhs

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset_arr = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs, rhs_ref, group_sizes, group_offset=group_offset_arr
    )

    tile_info = gmm_v2.TileSizes(
        tile_m=256,
        tile_k=tile_k,
        tile_n=min(out_size, 512),
        bucket_base=64,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset_arr,
        tile_info=tile_info,
        transpose_rhs=transpose_rhs,
    )

    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      bucket_base=[64, 128],
      num_buckets=[1, 2, 4],
  )
  def test_select_bucket_idx_exact_multiple(self, bucket_base, num_buckets):
    """Row counts that are exact multiples of bucket_base select their bucket.

    Bucket i covers bucket_base * (i + 1) rows, so a group of `rows` rows must
    select bucket ceil(rows / bucket_base) - 1, not the next larger one.

    Args:
      bucket_base: Number of rows in the smallest bucket.
      num_buckets: Number of buckets, i.e. tile_m // bucket_base.
    """
    branches = [lambda i=i: jnp.int32(i) for i in range(num_buckets)]

    @jax.jit
    def selected_bucket(m_end_local):
      # Mirrors inner_kernel: the index feeds lax.switch over the buckets.
      bucket_idx = gmm_v2.select_bucket_idx(m_end_local, bucket_base)
      return bucket_idx, jax.lax.switch(bucket_idx, branches)

    for multiple in range(1, num_buckets + 1):
      rows = multiple * bucket_base
      expected = math.ceil(rows / bucket_base) - 1
      with self.subTest(rows=rows):
        bucket_idx, selected = selected_bucket(jnp.int32(rows))
        self.assertEqual(int(bucket_idx), expected)
        self.assertEqual(int(selected), expected)

  def test_select_bucket_idx_not_multiple(self):
    bucket_base = 64
    for rows in (1, 63, 65, 127, 129, 255):
      expected = math.ceil(rows / bucket_base) - 1
      with self.subTest(rows=rows):
        self.assertEqual(
            int(gmm_v2.select_bucket_idx(jnp.int32(rows), bucket_base)),
            expected,
        )

  @parameterized.named_parameters(
      # name, num_groups, size_group, group_offset, tile_m, sublane, pattern
      ("offset_0", 16, 16, 0, 128, 16, "random"),
      ("offset_1", 64, 8, 1, 128, 16, "random"),
      ("offset_7", 64, 8, 7, 32, 8, "random"),
      ("offset_8", 64, 8, 8, 128, 16, "random"),
      ("offset_13", 64, 8, 13, 32, 16, "random"),
      ("last_shard", 64, 8, 56, 128, 16, "random"),
      ("empty_before_offset", 64, 8, 21, 32, 16, "empty_before"),
      ("empty_local_groups", 64, 8, 21, 32, 16, "empty_local"),
      ("ep16_384_last_shard", 384, 24, 360, 128, 16, "random"),
      ("ep8_512_last_shard", 512, 64, 448, 256, 16, "random"),
      ("offset_451", 512, 61, 451, 32, 8, "random"),
      ("decode_512_sparse_72x1", 512, 512, 0, 64, 8, "sparse_m1"),
      ("decode_512_single_active", 512, 512, 0, 64, 8, "single_active"),
      ("decode_512_all_empty", 512, 512, 0, 64, 8, "empty_local"),
      ("non_chunk_multiple_513_mixed", 513, 513, 0, 32, 8, "random"),
      # The compact walk used by tgmm_v2.
      ("compact_offset_7", 64, 8, 7, 32, 8, "random", False),
      ("compact_ep8_512_last_shard", 512, 64, 448, 256, 16, "random", False),
      ("compact_decode_sparse_72x1", 512, 512, 0, 64, 8, "sparse_m1", False),
      ("compact_empty_local_groups", 64, 8, 21, 32, 16, "empty_local", False),
  )
  def test_fill_metadata_matches_reference(
      self,
      num_groups,
      size_group,
      group_offset,
      tile_m,
      sublane,
      pattern,
      chunked=True,
  ):
    """The metadata matches a walk over every group from group 0.

    Covers small, large and odd offsets, and empty groups before the offset and
    in the local groups, whose sizes still leave the next group's start off a
    sublane boundary.

    Args:
      num_groups: Number of lhs groups.
      size_group: Number of local groups.
      group_offset: First local group.
      tile_m: Rows per gm tile.
      sublane: Rows per lhs sublane.
      pattern: How group sizes are drawn.
      chunked: Whether to use the chunked walk.
    """
    rng = np.random.default_rng(num_groups + group_offset)
    if pattern == "sparse_m1":
      group_sizes = np.zeros(num_groups, dtype=np.int32)
      group_sizes[np.linspace(0, num_groups - 1, 72, dtype=int)] = 1
    elif pattern == "single_active":
      group_sizes = np.zeros(num_groups, dtype=np.int32)
      group_sizes[num_groups // 2] = 1
    else:
      group_sizes = rng.integers(1, 3 * tile_m, num_groups)
      group_sizes[rng.random(num_groups) < 0.3] = 0
      if pattern == "empty_before":
        group_sizes[:group_offset] = 0
      elif pattern == "empty_local":
        group_sizes[group_offset : group_offset + size_group] = 0
    group_sizes = group_sizes.astype(np.int32)

    size_m = max(math.ceil(int(group_sizes.sum()) / sublane), 1) * sublane
    max_num_gm = size_group + math.ceil(size_m / tile_m) - 1
    input_cfgs = gmm_v2.InputConfigs(
        quant_dtype=None,
        quant_block_size=None,
        dtype=jnp.dtype(jnp.bfloat16),
    )
    cfgs = gmm_v2.GmmConfigs(
        tiles=gmm_v2.TileSizes(
            tile_m=tile_m, tile_k=128, tile_n=128, bucket_base=tile_m
        ),
        dims=gmm_v2.Dimensions(
            size_m=size_m,
            size_k=128,
            size_n=128,
            size_group=size_group,
            size_lhs_group=num_groups,
            size_lhs_sublane=sublane,
        ),
        lhs_cfgs=input_cfgs,
        rhs_cfgs=input_cfgs,
        out_dtype=jnp.dtype(jnp.bfloat16),
        acc_dtype=jnp.dtype(jnp.float32),
        zero_init=False,
        fuse_act=None,
    )

    num_gm, group_ids, m_offsets = _run_fill_metadata(
        jnp.asarray(group_sizes),
        jnp.array([group_offset], jnp.int32),
        cfgs,
        max_num_gm,
        chunked=chunked,
    )
    exp_num_gm, exp_group_ids, exp_m_offsets = _reference_fill_metadata(
        group_sizes, group_offset, size_group, tile_m, sublane, max_num_gm
    )
    self.assertEqual(int(num_gm[0]), exp_num_gm)
    np.testing.assert_array_equal(np.asarray(group_ids), exp_group_ids)
    np.testing.assert_array_equal(np.asarray(m_offsets), exp_m_offsets)

  @parameterized.parameters(False, True)
  def test_tgmm_tile_fn_receives_tiling_request(self, use_partial_sum):
    """A custom `tile_info` callback is called with a `TgmmTilingRequest`."""
    batch_size, in_size, out_size, num_groups = 256, 1024, 1024, 16
    key = jax.random.key(0)
    key1, key2, key3 = jax.random.split(key, 3)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    partial_sum = None
    if use_partial_sum:
      partial_sum = jax.random.normal(
          key3, (num_groups, in_size, out_size), dtype=jnp.bfloat16
      )
    group_sizes = get_group_sizes(batch_size, num_groups)
    vmem_limit_bytes = 32 * 1024 * 1024

    requests = []

    def tile_fn(req):
      requests.append(req)
      return tgmm_v2.calculate_tgmm_tiling_from_request(req)

    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_groups,
        partial_sum=partial_sum,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_fn,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    self.assertNotEmpty(requests)
    req = requests[0]
    self.assertIsInstance(req, tgmm_v2.TgmmTilingRequest)
    self.assertEqual(req.dims.size_k, in_size)
    self.assertEqual(req.dims.size_n, out_size)
    self.assertEqual(req.vmem_limit_bytes, vmem_limit_bytes)
    self.assertEqual(req.out_dtype, jnp.bfloat16)
    self.assertEqual(req.acc_dtype, jnp.float32)
    self.assertEqual(req.has_partial_sum, use_partial_sum)
    expected = reference_tgmm(
        lhs.swapaxes(0, 1), grad, group_sizes, num_groups
    )
    if use_partial_sum:
      expected = expected + partial_sum
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      rows_per_group=[64, 128, 192, 256],
      transpose_rhs=[False, True],
  )
  def test_gmm_equal_groups_exact_bucket_multiple(
      self, rows_per_group, transpose_rhs
  ):
    """Equal group sizes that are exact multiples of bucket_base."""
    num_groups = 8
    in_size = 512
    out_size = 512
    batch_size = rows_per_group * num_groups
    key = jax.random.key(0)
    k0, k1 = jax.random.split(key, 2)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = jnp.full((num_groups,), rows_per_group, dtype=jnp.int32)
    expected = reference_gmm(lhs, rhs, group_sizes)

    if transpose_rhs:
      rhs = jnp.swapaxes(rhs, 1, 2)
    tile_info = gmm_v2.TileSizes(
        tile_m=256,
        tile_k=in_size,
        tile_n=out_size,
        bucket_base=64,
    )
    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        tile_info=tile_info,
        transpose_rhs=transpose_rhs,
    )

    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[5, 16],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_tgmm_basic(
      self, batch_size, in_size, out_size, num_groups, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    )  # [m, k]
    grad = jax.random.normal(
        key2, (batch_size, out_size), dtype=jnp.bfloat16
    )  # [m, n]
    group_sizes = get_group_sizes(batch_size, num_groups)
    # if batch_size=128, num_groups=3, an example group_size is
    # group_sizes=Array([14, 14, ..., 7]).
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)  # [k, m]
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    # diff = jnp.abs(expected - actual)
    # max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
    # print(f"Output max diff: {jnp.max(diff)} at index {max_diff_idx}")
    # print(f"Output mean diff: {jnp.mean(jnp.abs(expected - actual))}")
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255],
      out_size=[255],
      num_groups=[16],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_tgmm_implicit_padding(
      self, batch_size, in_size, out_size, num_groups, group_offset
  ):
    # Test the case where there is implicit padding in the dim size_k (in_size)
    # and size_n (out_size).
    # Notice that tile_n and tile_k are aligned to the num_lanes in
    # calculate_tgmm_tiling.
    # The output shape is [num_groups, size_k, aligned_n] but there is implicit
    # padding on the k-dim to a multiple of sublanes. So the kernel is able to
    # write the full [i, aligned_tile_k, aligned_tile_n] to hbm with no problem
    # at the last k block.
    # Within the kernel, because k is not the contracting dim, so the padded k
    # is also not a problem.
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  def test_tgmm_multi_bucket_mixed_group_sizes(self) -> None:
    # Exercises all buckets (256, 512, 768) across single-tile, 2-tile,
    # 3+-tile, empty, and sublane-unaligned groups.
    group_sizes_list = [103, 0, 389, 768, 913, 1280, 1751, 256]
    batch_size = sum(group_sizes_list)
    # Pad batch_size to a multiple of 16 (TPU7 bf16 sublane tiling).
    pad_m = (-batch_size) % 16
    if pad_m:
      group_sizes_list[-1] += pad_m
      batch_size += pad_m
    in_size, out_size = 512, 512
    num_groups = len(group_sizes_list)

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array(group_sizes_list, dtype=jnp.int32)

    expected = reference_tgmm(lhs.swapaxes(0, 1), grad, group_sizes, num_groups)
    tile_info = gmm_v2.TileSizes(
        tile_m=768, tile_k=512, tile_n=512, bucket_base=256
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_groups,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[256],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16],
      group_offset=[0],
      tile_k=[256, 512],
      tile_n=[256, 512],
  )
  @pytest.mark.long
  def test_tgmm_with_tile_info(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      tile_k,
      tile_n,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )

    tile_m = 256
    tile_info = gmm_v2.TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=tile_m
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
      tile_k=[256, 512],
      tile_n=[256, 512],
      bucket_base=[64, 256],
  )
  @pytest.mark.long
  def test_tgmm_bucketing(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      tile_k,
      tile_n,
      bucket_base,
  ):
    """Verifies TGMM dynamic M-bucketing with num_buckets==1 and >1 across num_k=1,2 and num_n=1,2."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset_arr = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset_arr,
    )

    tile_m = 256
    tile_info = gmm_v2.TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=bucket_base
    )
    tgmm_v2.validate_tgmm_inputs(
        group_sizes, num_local_groups, group_offset_arr
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset_arr,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
      empty_group_index=[0, 1],
  )
  @pytest.mark.long
  def test_tgmm_empty_group(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      empty_group_index,
  ):
    """Test that TGMM correctly zeros output for empty groups."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    # Redistribute the empty group's tokens to the last group.
    group_sizes = group_sizes.at[-1].add(group_sizes[empty_group_index])
    group_sizes = group_sizes.at[empty_group_index].set(0)

    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  def test_tgmm_explicitly_exercises_all_branches(self):
    # Group 0 (size 4*tile_m, 4 gm tiles): matmul_new_group, matmul, matmul,
    # matmul_group_changing.
    # Group 1 (size 64, 1 gm tile): matmul_new_group_and_changing.

    tile_m = tile_k = tile_n = 256
    in_size = out_size = 256
    num_local_groups = 2
    g0, g1 = 4 * tile_m, 64
    batch_size = g0 + g1

    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([g0, g1], dtype=jnp.int32)
    group_offset = jnp.array(0, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    tile_info = gmm_v2.TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=tile_m
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  def test_tgmm_multi_tile(self):
    """num_k>1 and num_n>1 combined with multi-gm groups.
    """
    batch_size, in_size, out_size = 128, 256, 256
    group_sizes = jnp.array([8, 10, 50, 2, 3, 47, 8], dtype=jnp.int32)
    num_groups = group_sizes.size
    num_local_groups = num_groups
    assert jnp.sum(group_sizes) == batch_size

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_offset = jnp.array(0, dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1),
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=gmm_v2.TileSizes(
            tile_m=32, tile_k=128, tile_n=128, bucket_base=32
        ),
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  def test_tgmm_with_group_offset_and_empties(self):
    """group_offset>0 with empty groups both inside and outside the window.
    """
    batch_size, in_size, out_size = 128, 128, 128
    group_sizes = jnp.array([10, 0, 8, 0, 50, 2, 58, 0], dtype=jnp.int32)
    num_groups = group_sizes.size
    group_offset_val = 2
    num_local_groups = num_groups - group_offset_val
    assert jnp.sum(group_sizes) == batch_size

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_offset = jnp.array(group_offset_val, dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1),
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=gmm_v2.TileSizes(
            tile_m=32, tile_k=128, tile_n=128, bucket_base=32
        ),
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[256],
      out_size=[256],
      num_groups=[16],
      group_offset=[0],
      dtype_pair=[
          (jnp.float8_e4m3fn, jnp.float8_e5m2),       # production
          (jnp.float8_e4m3fn, jnp.float8_e4m3fn),     # symmetric fp8
      ],
  )
  @pytest.mark.long
  def test_tgmm_with_rhs_scale(
      self, batch_size, in_size, out_size, num_groups, group_offset, dtype_pair
  ):
    lhs_dtype, rhs_quant_dtype = dtype_pair
    num_local_groups = num_groups - group_offset

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    ).astype(lhs_dtype)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.float32)

    grad_q, grad_scale = quantize_tensor(
        grad,
        rhs_quant_dtype,
        axis=0,
        block_size=batch_size,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, N]
    assert grad_scale.shape == (1, 1, out_size)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset_arr = jnp.array([group_offset], dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1),
        grad_q,
        group_sizes,
        num_local_groups,
        rhs_scale=grad_scale,
        group_offset=group_offset_arr,
        out_dtype=jnp.bfloat16,  # pyrefly: ignore[bad-argument-type]
    )
    tgmm_v2.validate_tgmm_inputs(
        group_sizes, num_local_groups, group_offset_arr
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad_q,
        group_sizes,
        num_local_groups,
        rhs_scale=grad_scale,
        group_offset=group_offset_arr,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    chex.assert_trees_all_close(actual, expected, rtol=1e-2, atol=4e-1)

  @pytest.mark.long
  def test_tgmm_with_rhs_scale_n_padding(self):
    # Test the case where there is implicit padding in the dim size_n (out_size)
    # Pins tile_n=128 with out_size=300 so the kernel runs 3 n-tiles over an
    # aligned width of 384; the last tile (n_id=2) reads scale[..., 256:384]
    # where columns 300..383 are pad.

    batch_size, in_size, out_size = 128, 256, 300
    num_groups = 4
    rhs_quant_dtype = jnp.float8_e5m2

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.float32)

    grad_q, grad_scale = quantize_tensor(
        grad,
        rhs_quant_dtype,  # pyrefly: ignore[bad-argument-type]
        axis=0,
        block_size=batch_size,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, N]
    assert grad_scale.shape == (1, 1, out_size)

    group_sizes = get_group_sizes(batch_size, num_groups)
    tile_m = 128
    tile_info = gmm_v2.TileSizes(
        tile_m=tile_m, tile_k=256, tile_n=128, bucket_base=tile_m
    )

    expected = reference_tgmm(
        lhs.swapaxes(0, 1),
        grad_q,
        group_sizes,
        num_groups,
        rhs_scale=grad_scale,
        out_dtype=jnp.bfloat16,  # pyrefly: ignore[bad-argument-type]
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_groups)
    for has_pad_ref in sorted({gmm_v2.HAS_PAD_REF, False}):
      jax.clear_caches()
      with mock.patch.object(gmm_v2, "HAS_PAD_REF", has_pad_ref):
        actual = tgmm_v2.tgmm_v2(
            lhs,
            grad_q,
            group_sizes,
            num_groups,
            rhs_scale=grad_scale,
            tile_info=tile_info,
            preferred_element_type=jnp.bfloat16,
        )
      self.assertEqual(actual.shape, (num_groups, in_size, out_size))
      chex.assert_trees_all_close(actual, expected, rtol=1e-2, atol=4e-1)

  @parameterized.product(
      batch_size=[128, 512],
      in_size=[256, 512],
      out_size=[256, 512],
      group_offset=[0, 2],
  )
  def test_tgmm_partial_sum(self, batch_size, in_size, out_size, group_offset):
    """Tests the fused partial-sum add into the TGMM output."""
    num_groups = 16
    num_local_groups = num_groups - group_offset
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(k1, (batch_size, out_size), dtype=jnp.bfloat16)
    partial_sum = jax.random.normal(
        k2, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1),
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
    ) + partial_sum
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        partial_sum=partial_sum,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  def test_tgmm_partial_sum_matches_unfused_add(self):
    """A partial sum must be equivalent to adding it after an unfused TGMM."""
    batch_size, in_size, out_size, num_groups = 256, 512, 512, 8
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(k1, (batch_size, out_size), dtype=jnp.bfloat16)
    partial_sum = jax.random.normal(
        k2, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)

    fused = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_groups,
        partial_sum=partial_sum,
        preferred_element_type=jnp.bfloat16,
    )
    unfused = (
        tgmm_v2.tgmm_v2(
            lhs, grad, group_sizes, num_groups,
            preferred_element_type=jnp.bfloat16,
        )
        + partial_sum
    )
    assert_arrays_all_close(fused, unfused)

  def test_tgmm_partial_sum_empty_groups(self):
    """Groups with no rows must return their partial sum unchanged."""
    batch_size, in_size, out_size = 128, 128, 128
    group_sizes = jnp.array([10, 0, 8, 0, 50, 2, 58, 0], dtype=jnp.int32)
    num_groups = group_sizes.size
    group_offset_val = 2
    num_local_groups = num_groups - group_offset_val
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(k1, (batch_size, out_size), dtype=jnp.bfloat16)
    partial_sum = jax.random.normal(
        k2, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_offset = jnp.array(group_offset_val, dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1),
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
    ) + partial_sum
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        partial_sum=partial_sum,
        preferred_element_type=jnp.bfloat16,
        tile_info=gmm_v2.TileSizes(
            tile_m=32, tile_k=128, tile_n=128, bucket_base=32
        ),
    )
    assert_arrays_all_close(actual, expected)
    # Local groups 1 and 5 (global groups 3 and 7) are empty: exact passthrough.
    for local_group in (1, 5):
      chex.assert_trees_all_equal(
          actual[local_group], partial_sum[local_group]
      )

  @parameterized.product(
      in_size=[255, 500],
      out_size=[255, 500],
  )
  def test_tgmm_partial_sum_unaligned(self, in_size, out_size):
    """Covers padding of the partial sum along k and n."""
    batch_size, num_groups = 128, 16
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(k1, (batch_size, out_size), dtype=jnp.bfloat16)
    partial_sum = jax.random.normal(
        k2, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = (
        reference_tgmm(lhs.swapaxes(0, 1), grad, group_sizes, num_groups)
        + partial_sum
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_groups,
        partial_sum=partial_sum,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  def test_tgmm_partial_sum_with_rhs_scale(self):
    """The partial sum is added after the per-N rhs scale is applied."""
    batch_size, in_size, out_size, num_groups = 128, 256, 256, 16
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs = jax.random.normal(
        k0, (batch_size, in_size), dtype=jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    grad = jax.random.normal(k1, (batch_size, out_size), dtype=jnp.float32)
    grad_q, grad_scale = quantize_tensor(
        grad,
        jnp.float8_e5m2,  # pyrefly: ignore[bad-argument-type]
        axis=0,
        block_size=batch_size,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, N]
    partial_sum = jax.random.normal(
        k2, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1),
        grad_q,
        group_sizes,
        num_groups,
        rhs_scale=grad_scale,
        out_dtype=jnp.bfloat16,  # pyrefly: ignore[bad-argument-type]
    ) + partial_sum
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad_q,
        group_sizes,
        num_groups,
        rhs_scale=grad_scale,
        partial_sum=partial_sum,
        preferred_element_type=jnp.bfloat16,
    )
    chex.assert_trees_all_close(actual, expected, rtol=1e-2, atol=4e-1)

  def test_tgmm_partial_sum_unsupported(self):
    lhs = jnp.zeros((128, 256), jnp.bfloat16)
    grad = jnp.zeros((128, 256), jnp.bfloat16)
    group_sizes = get_group_sizes(128, 4)
    partial_sum = jnp.zeros((4, 256, 256), jnp.bfloat16)

    with config.disable_multi_core_mode(True):
      with self.assertRaisesRegex(
          NotImplementedError, "disable_multi_core_mode"
      ):
        tgmm_v2.tgmm_v2(
            lhs,
            grad,
            group_sizes,
            4,
            partial_sum=partial_sum,
            preferred_element_type=jnp.bfloat16,
        )
    with self.assertRaisesRegex(ValueError, "must match output dtype"):
      tgmm_v2.tgmm_v2(
          lhs,
          grad,
          group_sizes,
          4,
          partial_sum=partial_sum.astype(jnp.float32),
          preferred_element_type=jnp.bfloat16,
      )

  @parameterized.parameters(500, 512)
  def test_tgmm_partial_sum_is_accumulated_in_place(self, n):
    """The partial sum must be aliased to the output rather than copied."""
    m, k, num_groups = 512, 512, 8
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(k0, (m, k), dtype=jnp.bfloat16)
    grad = jax.random.normal(k1, (m, n), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(m, num_groups)
    partial_sum = jnp.zeros((num_groups, k, n), jnp.bfloat16)

    plain = jax.jit(
        lambda l, g, gs: tgmm_v2.tgmm_v2(
            l, g, gs, num_groups, preferred_element_type=jnp.bfloat16
        )
    )
    plain_text = plain.lower(lhs, grad, group_sizes).compile().as_text()
    assert plain_text is not None
    self.assertEqual(
        plain_text.count("output_to_operand_aliasing"),
        0,
        "control failed: a TGMM with no partial sum has nothing to alias, so "
        "the counter is not measuring what this test claims",
    )

    for has_pad_ref in sorted({gmm_v2.HAS_PAD_REF, False}):
      jax.clear_caches()
      with mock.patch.object(gmm_v2, "HAS_PAD_REF", has_pad_ref):
        fused = jax.jit(
            lambda l, g, gs, p: tgmm_v2.tgmm_v2(
                l,
                g,
                gs,
                num_groups,
                partial_sum=p,
                preferred_element_type=jnp.bfloat16,
            )
        )
        fused_text = (
            fused.lower(lhs, grad, group_sizes, partial_sum).compile().as_text()
        )
      assert fused_text is not None
      self.assertGreater(
          fused_text.count("output_to_operand_aliasing"),
          0,
          "the partial sum was not aliased to the output",
      )
      if has_pad_ref:
        self.assertNotIn("pad(", fused_text)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
      block_size=[64],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and test_utils.get_tpu_version() < 7:
      self.skipTest("Expect TPUv7+")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  def test_gmm_security_isolation(self):
    """Verifies that sequences (experts) are isolated from each other.

    This test checks that NaNs or extreme values in one expert group do not
    pollute the output of other expert groups, even if they share the same
    sublane tile.
    """
    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 4
    key = jax.random.key(42)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )

    # We use very small group sizes to force expert groups to share tiles.
    # sublane_size is typically 8 or 16.
    group_sizes = jnp.array([4, 4, 4, batch_size - 12], dtype=jnp.int32)

    # 1. Run baseline
    actual_clean = gmm_v2.gmm_v2(lhs, rhs, group_sizes)

    # 2. Inject NaNs into all experts except the first one.
    # If isolation fails, the NaNs will leak into the first expert's output.
    rhs_malicious = rhs.at[1:].set(jnp.nan)
    actual_malicious = gmm_v2.gmm_v2(lhs, rhs_malicious, group_sizes)

    # Verify that the first expert's output is identical and NaN-free.
    first_expert_size = group_sizes[0]
    chex.assert_trees_all_close(
        actual_malicious[:first_expert_size],
        actual_clean[:first_expert_size],
        atol=0.0,
        rtol=0.0,
    )
    self.assertFalse(jnp.any(jnp.isnan(actual_malicious[:first_expert_size])))

  def test_gmm_uninitialized_memory_robustness(self):
    """Verifies that the kernel is robust against uninitialized scratchpads.

    This test intentionally poisons TPU VMEM/SMEM with NaNs before running the
    GMM kernel. This ensures that  no stale data from previous sessions can leak
    into the output.
    """
    # 1. Poison TPU memory with NaNs
    mosaic_tpu.poison_tpu_memory()

    # 2. Run GMM kernel
    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 4
    key = jax.random.key(0)
    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = jnp.array([batch_size // 4] * 4, dtype=jnp.int32)

    actual = gmm_v2.gmm_v2(lhs, rhs, group_sizes)

    # 3. Verify that the output is NaN-free
    self.assertFalse(jnp.any(jnp.isnan(actual)))

  @parameterized.parameters(64, 129, 160)
  def test_gmm_unaligned_k_to_tpu_num_lanes_with_nans(self, k: int):
    """The last k tile requires proper masking when `% num_lanes !=0` especially with NaNs."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    self.assertNotEqual(k % num_lanes, 0)

    batch_size = 128
    out_size = 512
    num_groups = 4
    key = jax.random.key(10)
    k0, k1, k2 = jax.random.split(key, 3)

    # Mark half of the tokens as paddings to stress-test GMM
    is_padding = jax.random.bernoulli(k2, p=0.5, shape=(batch_size, 1))

    aligned_k = pl.cdiv(k, num_lanes) * num_lanes
    # On the k-dimension, we allocate `num_lanes`-aligned array so that we can
    # fill `[k:aligned_k]` as `jnp.nan`s and TPU will load those `[k:aligned_k]`
    # nans into VMEM.
    lhs_padded = jax.random.normal(
        k0, (batch_size, aligned_k), dtype=jnp.bfloat16
    )
    lhs_padded = jnp.where(is_padding, jnp.nan, lhs_padded)
    lhs_padded = lhs_padded.at[:, k:].set(jnp.nan)

    lhs = lhs_padded[:, :k]

    rhs = jax.random.normal(
        k1, (num_groups, k, out_size), dtype=jnp.bfloat16
    )
    group_sizes = jnp.array(
        [batch_size // num_groups] * num_groups, dtype=jnp.int32
    )

    # Re-masking because we only care about the output values on non-paddings
    # across different implementations.
    expected = reference_gmm(lhs, rhs, group_sizes)
    expected = jnp.where(is_padding, jnp.nan, expected)

    actual = gmm_v2.gmm_v2(lhs, rhs, group_sizes)
    actual = jnp.where(is_padding, jnp.nan, actual)

    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int8],
      block_size=[1024],
      tile_k=[128, 256],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test that quant_block_size > tile_k is handled correctly."""
    if weight_dtype == jnp.float4_e2m1fn and test_utils.get_tpu_version() < 7:
      self.skipTest("Expect TPUv7+")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = gmm_v2.TileSizes(
        tile_m=128, tile_k=tile_k, tile_n=out_size, bucket_base=128
    )
    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int4, jnp.int8],
      block_size=[1024],
      tile_k=[128, 256],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_activation_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test activation+weight quantized path with quant_block_size > tile_k."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = gmm_v2.TileSizes(
        tile_m=128, tile_k=tile_k, tile_n=out_size, bucket_base=128
    )
    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=True,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=1.2, rtol=1.2)

  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16, 32],
      weight_dtype=[jnp.int4, jnp.int8, jnp.float8_e4m3fn],
      activation_dtype=[
          None,
          jnp.int8,
          jnp.float8_e4m3fn,
          jnp.float8_e5m2
      ],
      block_size=[1024],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_activation_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      activation_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and test_utils.get_tpu_version() < 7:
      self.skipTest("Expect TPUv7+")
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    if activation_dtype is not None:
      tpu_info = pltpu.get_tpu_info()
      if not (
          tpu_info.is_matmul_supported(activation_dtype, weight_dtype)
      ) and not gmm_v2.is_manually_cast_matmul_dtype_combo(
          activation_dtype, weight_dtype
      ):
        self.skipTest(
            f"Combination {activation_dtype} and {weight_dtype} not supported"
            " by the kernel."
        )
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
        lhs_quant_dtype=activation_dtype,
    ).astype(lhs.dtype)

    # e5m2 introduces increased rounding error compared to e4m3
    if activation_dtype == jnp.float8_e5m2:
      atol, rtol = 2.25, 1.1
    else:
      atol, rtol = 1.1, 1.1

    chex.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  @parameterized.product(
      batch_size=[128],
      in_size=[768],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
      activation_dtype=[
          None,
          jnp.int8,
          jnp.float8_e4m3fn,
      ],
      block_size=[256, 768],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_activation_weight_quantized_misaligned_block_sizes(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      activation_dtype,
      block_size,
      group_offset,
  ):
    """Tests activation and weight quantization with misaligned block sizes."""
    if weight_dtype == jnp.float4_e2m1fn and test_utils.get_tpu_version() < 7:
      self.skipTest("Expect TPUv7+")
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    if in_size % block_size != 0:
      self.skipTest("in_size must be divisible by block_size")
    if activation_dtype is not None:
      tpu_info = pltpu.get_tpu_info()
      if not (
          tpu_info.is_matmul_supported(activation_dtype, weight_dtype)
      ) and not gmm_v2.is_manually_cast_matmul_dtype_combo(
          activation_dtype, weight_dtype
      ):
        self.skipTest(
            f"Combination {activation_dtype} and {weight_dtype} not supported"
            " by the kernel."
        )
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
        lhs_quant_dtype=activation_dtype,
    ).astype(lhs.dtype)

    # e5m2 introduces increased rounding error compared to e4m3
    if activation_dtype == jnp.float8_e5m2:
      atol, rtol = 2.25, 1.1
    else:
      atol, rtol = 1.1, 1.1

    chex.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16, 32],
      block_size=[1024],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_quantize_lhs_with_lhs_scale(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      block_size,
      group_offset,
  ):
    """LHS quantized with a user provided lhs_scale."""
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    if pltpu.get_tpu_info().generation <= 5:
      self.skipTest("FP8 is only supported after TPU generation 6")

    # Per-tensor fp8 quant scale (bound / finfo(fp8).max = 224 / 448),
    # matching qwix's "fixed,-224,224" act calibration.
    lhs_scale = jnp.full((1, 1), 224.0 / 448.0, dtype=jnp.float32)

    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    # Pin fp8 weights so rhs is dequantized after the matmul, which is the only
    # path that enables lhs quantization.
    rhs_q, rhs_scale = quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size  # pyrefly: ignore[bad-argument-type]
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    # The kernel quantizes LHS to fp8 internally (constant fixed scale,
    # clip-before-cast), which perturbs every LHS value before the matmul. We
    # apply the identical quantize->dequantize round-trip to the reference's LHS
    # so both sides carry the same quantization error. Otherwise the comparison
    # would measure fp8 quantization noise itself (large for fp8_e4m3fn) rather
    # than whether the kernel computes the correct grouped matmul. Because the
    # fixed scale is data-independent (not per-block absmax), the quantized
    # values are identical regardless of how the kernel slices K, so this
    # whole-tensor simulation is faithful; only accumulation order /
    # intermediate-dtype differences remain, within the tolerance below.
    scale = lhs_scale.item()
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    lhs_q = jnp.clip(lhs.astype(jnp.float32) / scale, -fp8_max, fp8_max).astype(
        jnp.float8_e4m3fn
    )
    lhs_simulated = (lhs_q.astype(jnp.float32) * scale).astype(lhs.dtype)

    expected = reference_gmm(
        lhs_simulated,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
        lhs_scale=lhs_scale,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=0.75, rtol=3e-2)

  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255],
      out_size=[255],
      num_groups=[16],
      has_bias=[True, False],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_implicit_padding(
      self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[500],
      num_groups=[16],
      has_bias=[True, False],
      weight_dtype=[jnp.int8],
      block_size=[512],
      group_offset=[0],
  )
  @pytest.mark.long
  def test_gmm_weight_quantized_padding(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    self.assertEqual(actual.shape, (batch_size, out_size))
    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      # group_config: (num_groups, group_offset, num_local_groups)
      group_config=[
          # groups 0-1: group<0, groups 2-5: local and active,
          # groups 6-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=2, num_local_groups=4),
          # no negative groups, groups 0-7: local and active,
          # groups 8-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=0, num_local_groups=8),
          # groups 0-3: group<0, groups 4-7: local and active,
          # groups 8-31: group>=num_local_groups
          _GroupConfig(num_groups=32, group_offset=4, num_local_groups=4),
      ],
  )
  def test_gmm_nonlocal_groups_produce_zeros(
      self, batch_size, in_size, out_size, group_config
  ):
    num_groups, group_offset, num_local_groups = group_config
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = jax.random.normal(
        key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
    )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.parameters([None, 25.0])
  def test_situ_and_mul(self, linear_beta):
    gate = jnp.asarray([-3.0, 0.5, 7.0])
    up = jnp.asarray([-30.0, 2.0, 40.0])
    expected_gate = 4.0 * jnp.tanh(gate / 4.0) * jax.nn.sigmoid(gate)
    expected_up = (
        up if linear_beta is None else linear_beta * jnp.tanh(up / linear_beta)
    )

    chex.assert_trees_all_close(
        gmm_v2.situ_and_mul(gate, up, 4.0, linear_beta),
        expected_gate * expected_up,
    )

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True, False],
      use_weight_scale=[True, False],
      maybe_quantize_lhs=[True, False],
      fuse_act=["silu", "swigluoai", "gelu", "situ:4.0:none", "situ:4.0:25.0"],
      group_offset=[0, 2],
      block_size=[256, 512],
  )
  def test_gmm_fused_activation(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      use_weight_scale,
      maybe_quantize_lhs,
      fuse_act,
      group_offset,
      block_size,
  ):
    if maybe_quantize_lhs and not use_weight_scale:
      self.skipTest(
          "LHS quantization requires RHS quantization/scale in this config."
      )
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    key = jax.random.key(0)
    final_out_size = out_size // 2
    num_local_groups = num_groups - group_offset

    # 1. Generate Inputs
    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )

    rhs_q = rhs
    rhs_scale = None
    if use_weight_scale:
      rhs_q, rhs_scale = quantize_tensor(
          rhs, jnp.int8, axis=1, block_size=block_size  # pyrefly: ignore[bad-argument-type]
      )
      rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array([group_offset], dtype=jnp.int32)

    # 2. Simulate LHS Quantization Noise
    lhs_simulated = lhs
    # because the kernel quantizes LHS in blocks, while reference does it at the
    # whole tensor level, and output is casted down we need to simulate that
    # quantization noise in the reference as well for a fair comparison
    if maybe_quantize_lhs:
      lhs_block_size = min(512, in_size)
      lhs_q, lhs_scale_factor = quantize_tensor(
          lhs, jnp.int8, axis=1, block_size=lhs_block_size  # pyrefly: ignore[bad-argument-type]
      )
      lhs_q_blocked = lhs_q.reshape(batch_size, -1, lhs_block_size).astype(
          jnp.float32
      )
      lhs_scale_expanded = jnp.expand_dims(lhs_scale_factor, axis=2)
      lhs_simulated = (
          (lhs_q_blocked * lhs_scale_expanded)
          .reshape(lhs.shape)
          .astype(lhs.dtype)
      )

    # 3. Compute Reference Output
    raw_expected = reference_gmm(
        lhs_simulated,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    # Slice the reference and apply the activation function
    if fuse_act is not None:
      raw_gate, raw_up = jnp.split(raw_expected, 2, axis=-1)
      raw_expected = gmm_v2.interleave_lane(raw_gate, raw_up)
    expected = gmm_v2.apply_act_fn(
        raw_expected.astype(jnp.float32), fuse_act
    ).astype(lhs.dtype)

    # 4. Compute Actual Kernel Output
    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
    ).astype(lhs.dtype)

    # 5. Compare Results
    self.assertEqual(actual.shape, (batch_size, final_out_size))

    # tolerances based quantization noise difference between reference and
    # gmm_v2
    if maybe_quantize_lhs:
      atol, rtol = 4.0, 2.0  # Act + Weight Quantization
    elif use_weight_scale:
      atol, rtol = 3e-1, 3e-1  # Weight Quantization Only
    else:
      atol, rtol = 5e-2, 5e-2  # Unquantized Path (bfloat16 precision diffs)

    chex.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  def test_gmm_v2_int4_packed_matches_reference(self):
    batch_size = 128
    logical_k = 512
    storage_k = logical_k // 8
    out_size = 256
    num_groups = 4
    block_size = 64
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, logical_k), jnp.bfloat16, -1, 1)

    weight_key, _ = jax.random.split(key)
    rhs = jax.random.uniform(
        weight_key,
        (num_groups, logical_k, out_size),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.bfloat16,
    )
    rhs_int4, rhs_scale = quantize_tensor(
        rhs,
        jnp.int4,  # pyrefly: ignore[bad-argument-type]
        axis=1,
        block_size=block_size,
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = reference_gmm(
        lhs,
        rhs_int4,
        group_sizes,
        rhs_scale=rhs_scale,
    )

    # We simulate packing uint4 weights since checkpoints are stored as packed
    # uint4 values
    rhs_uint4 = (rhs_int4.astype(jnp.uint32) + 8) & 0x0F

    packing_factor = gmm_v2.get_packing_factor(jnp.int32, jnp.int4)  # 8  # pyrefly: ignore[bad-argument-type]
    rhs_reshaped = rhs_uint4.reshape(
        num_groups, storage_k, packing_factor, out_size
    )

    # We pack 8 (packing factor) uint4 values into a single uint32
    shifts = jnp.arange(packing_factor, dtype=jnp.int32) * 4
    shifted = rhs_reshaped << shifts[None, None, :, None]
    rhs_packed_uint32 = jnp.sum(shifted, axis=2).astype(jnp.int32)

    # We use -2004318072 because this converts the packed uint4 back to int4.
    # In decimal -2004318072 is 0x88888888 which when XOR-ed with the uint4
    # values gives the corresponding packed int4 values.
    INT4_SIGN_XOR = -2004318072
    rhs_packed_int32_xor = (rhs_packed_uint32 ^ INT4_SIGN_XOR).astype(jnp.int32)

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_packed_int32_xor,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_quant_dtype=jnp.int4,
    )

    assert actual.shape == (batch_size, out_size)
    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @parameterized.product(
      rhs_dtype=[jnp.int8, jnp.float8_e4m3fn],
      in_size_and_blocks=[(1536, 3), (1280, 5)],
      maybe_quantize_lhs=[False, True],
  )
  def test_partial_k_tile_skips_stale_rhs_scale(
      self, rhs_dtype, in_size_and_blocks, maybe_quantize_lhs
  ):
    """Partial K tail tiles skip uninitialized rhs_scale blocks (b/576348326)."""
    if (
        rhs_dtype == jnp.float8_e4m3fn
        and pltpu.get_tpu_info().generation <= 5
    ):
      self.skipTest("FP8 is only supported after TPU generation 5")
    in_size, num_scale_blocks = in_size_and_blocks
    batch_size, out_size, num_groups = 256, 256, 4
    k0, k1, k2 = jax.random.split(jax.random.key(42), 3)
    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = (
        jax.random.normal(k1, (num_groups, in_size, out_size), jnp.float32)
        * 4.0
    ).astype(rhs_dtype)
    rhs_scale = (
        jnp.abs(
            jax.random.normal(
                k2, (num_groups, num_scale_blocks, 1, out_size), jnp.float32
            )
        )
        * 0.02
    )
    group_sizes = jnp.array([64, 64, 64, 64], dtype=jnp.int32)
    # tile_k=1024 does not divide in_size (valid_k=512 or 256 on final K step),
    # leaving trailing sub-blocks in rhs_scale VMEM uninitialized on step 1.
    tiles = gmm_v2.TileSizes(
        tile_m=128, tile_k=1024, tile_n=256, bucket_base=128
    )

    _poison_vmem_all_cores()
    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_scale=rhs_scale,
        tile_info=tiles,
        maybe_quantize_lhs=maybe_quantize_lhs,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(actual))))
    expected = reference_gmm(
        lhs, rhs, group_sizes, rhs_scale=rhs_scale
    ).astype(jnp.bfloat16)
    if maybe_quantize_lhs:
      chex.assert_trees_all_close(actual, expected, atol=6e-1, rtol=6e-1)
    else:
      assert_arrays_all_close(actual, expected)

  @parameterized.product(
      has_rhs_scale=[False, True],
      has_bias=[False, True],
      group_offset=[0, 2],
  )
  def test_gmm_prequantized_lhs_with_lhs_scale(
      self, has_rhs_scale, has_bias, group_offset
  ):
    """Pre-quantized int8 lhs with per-tensor post-matmul lhs_scale."""
    batch_size, in_size, out_size, num_groups = 256, 512, 256, 8
    num_local_groups = num_groups - group_offset
    k0, k1, k2 = jax.random.split(jax.random.key(7), 3)
    lhs_f = jax.random.uniform(k0, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs_f = jax.random.uniform(
        k1, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    lhs_q, lhs_scale = quantize_tensor(
        lhs_f.reshape(1, -1),
        jnp.dtype(jnp.int8),
        axis=1,
        block_size=lhs_f.size,
    )
    lhs_q = lhs_q.reshape(lhs_f.shape)
    lhs_scale = lhs_scale.reshape(1, 1)

    rhs_q, rhs_scale_q = quantize_tensor(
        rhs_f, jnp.dtype(jnp.int8), axis=1, block_size=256
    )
    rhs_scale = jnp.expand_dims(rhs_scale_q, axis=2) if has_rhs_scale else None

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.uniform(
          k2, (num_local_groups, 1, out_size), jnp.float32, -1, 1
      )
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset_arr = jnp.array(group_offset, dtype=jnp.int32)

    actual = gmm_v2.gmm_v2(
        lhs_q,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset_arr,
        lhs_scale=lhs_scale,
        maybe_quantize_lhs=False,
        preferred_element_type=jnp.float32,
    )
    expected = reference_gmm(
        lhs_q.astype(jnp.float32),
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset_arr,
        lhs_scale=lhs_scale,
    )
    assert_arrays_all_close(actual, expected)

  def test_int8_lhs_rejects_pre_matmul_rhs_dequant(self):
    """Integer lhs with sub-MXU rhs quant blocks raises ValueError."""
    batch_size, in_size, out_size, num_groups = 128, 256, 256, 4
    lhs = jnp.zeros((batch_size, in_size), dtype=jnp.int8)
    rhs = jnp.zeros((num_groups, in_size, out_size), dtype=jnp.int8)
    # quant_block_size = 256 // 8 = 32 < mxu_column_size (128 or 256).
    rhs_scale = jnp.ones((num_groups, 8, 1, out_size), dtype=jnp.float32)
    group_sizes = jnp.array([32, 32, 32, 32], dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "integer lhs"):
      gmm_v2.gmm_v2(
          lhs,
          rhs,
          group_sizes,
          rhs_scale=rhs_scale,
          maybe_quantize_lhs=False,
          preferred_element_type=jnp.float32,
      )

  @parameterized.product(
      has_rhs_scale=[False, True],
      group_offset=[0, 2],
  )
  def test_tgmm_int8(self, has_rhs_scale, group_offset):
    """tgmm_v2 uses int32 accumulation for int8 x int8 dots."""
    batch_size, in_size, out_size, num_groups = 256, 256, 256, 8
    num_local_groups = num_groups - group_offset
    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs_f = jax.random.uniform(key1, (batch_size, in_size), jnp.float32, -1, 1)
    grad_f = jax.random.uniform(
        key2, (batch_size, out_size), jnp.float32, -1, 1
    )
    lhs_q, lhs_scale = quantize_tensor(
        lhs_f, jnp.dtype(jnp.int8), axis=1, block_size=in_size
    )
    # Fold per-token lhs_scale into grad before quantizing grad per-channel
    # along out_size (matching how MaxText prepares tgmm_v2 inputs).
    grad_q, grad_scale = quantize_tensor(
        grad_f * lhs_scale,
        jnp.dtype(jnp.int8),
        axis=0,
        block_size=batch_size,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, out_size]
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset_arr = jnp.array([group_offset], dtype=jnp.int32)

    expected = reference_tgmm(
        lhs_f.swapaxes(0, 1),
        grad_f,
        group_sizes,
        num_local_groups,
        group_offset=group_offset_arr,
    )
    actual = tgmm_v2.tgmm_v2(
        lhs_q,
        grad_q,
        group_sizes,
        num_local_groups,
        rhs_scale=grad_scale if has_rhs_scale else None,
        group_offset=group_offset_arr,
        preferred_element_type=jnp.float32,
    )
    if not has_rhs_scale:
      actual = actual * grad_scale
    chex.assert_trees_all_close(actual, expected, atol=2e-1, rtol=5e-2)

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.int8, jnp.float8_e4m3fn],
      has_scale=[False, True],
      has_bias=[False, True],
      fuse_act=[None, "silu"],
  )
  def test_transpose_rhs_with_scale_bias_and_fuse_act(
      self, dtype, has_scale, has_bias, fuse_act
  ):
    """Native transpose_rhs matches swapped-axes 2D gmm_v2 across epilogues.

    This test sets maybe_quantize_lhs=False, so lhs must already be in a dtype
    compatible with rhs (`dtype`) for the TPU MXU.
    """
    if dtype == jnp.float8_e4m3fn and pltpu.get_tpu_info().generation <= 5:
      self.skipTest("FP8 is only supported after TPU generation 5")
    if dtype == jnp.bfloat16 and has_scale:
      self.skipTest("rhs_scale only applies to quantized rhs")

    batch_size, in_size, out_size, num_groups = 256, 512, 256, 4
    rhs_n = out_size * 2 if fuse_act is not None else out_size
    k0, k1, k2 = jax.random.split(jax.random.key(11), 3)
    lhs_f = jax.random.uniform(k0, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs_f = jax.random.uniform(
        k1, (num_groups, in_size, rhs_n), jnp.bfloat16, -1, 1
    )
    lhs_scale = None
    rhs_scale = None
    if dtype == jnp.bfloat16:
      lhs = lhs_f
      rhs = rhs_f
    elif dtype == jnp.float8_e4m3fn:
      lhs = lhs_f.astype(jnp.float8_e4m3fn)
      if has_scale:
        rhs, rhs_scale_q = quantize_tensor(
            rhs_f, jnp.dtype(dtype), axis=1, block_size=256
        )
        rhs_scale = jnp.expand_dims(rhs_scale_q, axis=2)
      else:
        rhs = rhs_f.astype(jnp.float8_e4m3fn)
    else:
      lhs, lhs_scale_q = quantize_tensor(
          lhs_f.reshape(1, -1),
          jnp.dtype(dtype),
          axis=1,
          block_size=lhs_f.size,
      )
      lhs = lhs.reshape(lhs_f.shape)
      lhs_scale = lhs_scale_q.reshape(1, 1)
      rhs, rhs_scale_q = quantize_tensor(
          rhs_f, jnp.dtype(dtype), axis=1, block_size=256
      )
      rhs_scale = jnp.expand_dims(rhs_scale_q, axis=2) if has_scale else None
    rhs_t = jnp.swapaxes(rhs, 1, 2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.uniform(
          k2, (num_groups, 1, rhs_n), jnp.bfloat16, -1, 1
      )
    group_sizes = jnp.array([60, 0, 100, 50], dtype=jnp.int32)
    tiles = gmm_v2.TileSizes(
        tile_m=128, tile_k=256, tile_n=128, bucket_base=128
    )

    common = dict(
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        lhs_scale=lhs_scale,
        tile_info=tiles,
        fuse_act=fuse_act,
        maybe_quantize_lhs=False,
        preferred_element_type=jnp.float32,
    )
    expected = gmm_v2.gmm_v2(lhs, rhs, group_sizes, **common)
    actual = gmm_v2.gmm_v2(
        lhs, rhs_t, group_sizes, transpose_rhs=True, **common
    )
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      rhs_dtype=[jnp.int8, jnp.float8_e4m3fn],
      fuse_act=[None, "silu"],
  )
  def test_transpose_rhs_with_lhs_quantization(self, rhs_dtype, fuse_act):
    """Native transpose_rhs matches 2D gmm_v2 with in-kernel lhs quantization."""
    if rhs_dtype == jnp.float8_e4m3fn and pltpu.get_tpu_info().generation <= 5:
      self.skipTest("FP8 is only supported after TPU generation 5")

    batch_size, in_size, out_size, num_groups = 256, 512, 256, 4
    rhs_n = out_size * 2 if fuse_act is not None else out_size
    k0, k1 = jax.random.split(jax.random.key(13), 2)
    lhs = jax.random.uniform(k0, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs_f = jax.random.uniform(
        k1, (num_groups, in_size, rhs_n), jnp.bfloat16, -1, 1
    )
    rhs, rhs_scale_q = quantize_tensor(
        rhs_f, jnp.dtype(rhs_dtype), axis=1, block_size=256
    )
    rhs_scale = jnp.expand_dims(rhs_scale_q, axis=2)
    rhs_t = jnp.swapaxes(rhs, 1, 2)
    group_sizes = jnp.array([60, 0, 100, 50], dtype=jnp.int32)
    tiles = gmm_v2.TileSizes(
        tile_m=128, tile_k=256, tile_n=128, bucket_base=128
    )

    common = dict(
        rhs_scale=rhs_scale,
        tile_info=tiles,
        fuse_act=fuse_act,
        maybe_quantize_lhs=True,
        preferred_element_type=jnp.float32,
    )
    expected = gmm_v2.gmm_v2(lhs, rhs, group_sizes, **common)
    actual = gmm_v2.gmm_v2(
        lhs, rhs_t, group_sizes, transpose_rhs=True, **common
    )
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      fuse_act=[None, "silu"],
  )
  def test_transpose_rhs_pre_matmul_dequant(self, fuse_act):
    """Native transpose_rhs matches 2D gmm_v2 for sub-MXU rhs quant blocks."""
    batch_size, in_size, out_size, num_groups = 256, 512, 256, 4
    rhs_n = out_size * 2 if fuse_act is not None else out_size
    k0, k1 = jax.random.split(jax.random.key(15), 2)
    lhs = jax.random.uniform(k0, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs_f = jax.random.uniform(
        k1, (num_groups, in_size, rhs_n), jnp.bfloat16, -1, 1
    )
    # block_size=64 < mxu_column_size (128 or 256) triggers pre-matmul dequant.
    rhs, rhs_scale_q = quantize_tensor(
        rhs_f, jnp.dtype(jnp.int8), axis=1, block_size=64
    )
    rhs_scale = jnp.expand_dims(rhs_scale_q, axis=2)
    rhs_t = jnp.swapaxes(rhs, 1, 2)
    group_sizes = jnp.array([60, 0, 100, 50], dtype=jnp.int32)
    tiles = gmm_v2.TileSizes(
        tile_m=128, tile_k=256, tile_n=128, bucket_base=128
    )

    common = dict(
        rhs_scale=rhs_scale,
        tile_info=tiles,
        fuse_act=fuse_act,
        maybe_quantize_lhs=False,
        preferred_element_type=jnp.float32,
    )
    expected = gmm_v2.gmm_v2(lhs, rhs, group_sizes, **common)
    actual = gmm_v2.gmm_v2(
        lhs, rhs_t, group_sizes, transpose_rhs=True, **common
    )
    assert_arrays_all_close(actual, expected)

  def test_transpose_rhs_shape_mismatch_raises(self):
    """validate_inputs raises ValueError when rhs K disagrees with lhs K."""
    lhs = jnp.zeros((128, 256), dtype=jnp.bfloat16)
    rhs_wrong_k = jnp.zeros((4, 128, 512), dtype=jnp.bfloat16)
    group_sizes = jnp.array([32, 32, 32, 32], dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "rhs shape mismatch"):
      gmm_v2.gmm_v2(lhs, rhs_wrong_k, group_sizes, transpose_rhs=True)
    with self.assertRaisesRegex(ValueError, "rhs shape mismatch"):
      gmm_v2.gmm_v2(lhs, rhs_wrong_k, group_sizes, transpose_rhs=False)


_VMEM_LIMIT_BYTES = int(64 * 1024 * 1024 * 0.9)


def _make_dims(size_m: int, size_k: int, size_n: int) -> gmm_v2.Dimensions:
  return gmm_v2.Dimensions(
      size_m=size_m,
      size_k=size_k,
      size_n=size_n,
      size_group=128,
      size_lhs_group=128,
      size_lhs_sublane=8,
  )


def _bf16_input_configs(is_transposed: bool = False) -> gmm_v2.InputConfigs:
  return gmm_v2.InputConfigs(
      dtype=jnp.dtype(jnp.bfloat16),
      quant_dtype=None,
      has_scale=False,
      has_bias=False,
      quant_block_size=None,
      is_transposed=is_transposed,
  )


def _mock_tpu_info(
    num_cores: int, generation: int = 5, mxu_column_size: int = 128
) -> mock.Mock:
  mock_tpu_info = mock.create_autospec(
      gmm_v2.pltpu.TpuInfo, instance=True, spec_set=False
  )
  mock_tpu_info.num_lanes = 128
  mock_tpu_info.mxu_column_size = mxu_column_size
  mock_tpu_info.generation = generation
  mock_tpu_info.num_cores = num_cores
  mock.seal(mock_tpu_info)
  return mock_tpu_info


class GmmV2CalculateTilingTest(parameterized.TestCase):
  """Tests for calculate_tiling edge cases regarding vmem floor limits."""

  @parameterized.named_parameters(
      dict(
          testcase_name="tile_n_floor_overshoot_memory_fits",
          # tile_n shrinks to limit, but actually goes slightly below it due to
          # alignment/math. The memory actually fits. We expect tile_n to step
          # back up to the limit (or its aligned nearest above limit) and tile_k
          # should not shrink because memory fits.
          vmem_limit=3000000,
          expected_tile_n_less_than_limit=False,
      ),
      dict(
          testcase_name="tile_n_floor_overshoot_memory_exceeded",
          # tile_n shrinks to limit, memory is still exceeded. tile_n steps back
          # up and then tile_k shrinks to fit.
          vmem_limit=1000000,
          expected_tile_n_less_than_limit=False,
      ),
  )
  @mock.patch("tokamax._src.ops.gmm_v2.gmm_v2.pltpu.get_tpu_info")
  def test_calculate_tiling_edge_cases(
      self, mock_get_tpu_info, vmem_limit, expected_tile_n_less_than_limit
  ):
    mock_tpu_info = mock.MagicMock()
    mock_tpu_info.num_lanes = 128
    mock_tpu_info.mxu_column_size = 128
    mock_get_tpu_info.return_value = mock_tpu_info

    dims = gmm_v2.Dimensions(
        size_m=512,
        size_k=4096,
        size_n=4096,
        size_group=1,
        size_lhs_group=1,
        size_lhs_sublane=8,
    )
    lhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.bfloat16),
        quant_dtype=None,
        has_scale=False,
        has_bias=False,
        quant_block_size=None,
    )
    rhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.bfloat16),
        quant_dtype=None,
        has_scale=False,
        has_bias=False,
        quant_block_size=None,
        is_transposed=False,
    )

    tiles = gmm_v2.calculate_tiling(
        dims=dims,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        vmem_limit_bytes=vmem_limit,
    )

    tile_n_limit = mock_tpu_info.mxu_column_size * 2
    if not expected_tile_n_less_than_limit:
      self.assertGreaterEqual(tiles.tile_n, tile_n_limit)

  @mock.patch.object(gmm_v2.pltpu, "get_tpu_info")
  def test_lhs_buffer_count_fallback(self, mock_get_tpu_info):
    mock_tpu_info = mock.create_autospec(
        gmm_v2.pltpu.TpuInfo, instance=True, spec_set=False
    )
    mock_tpu_info.num_lanes = 128
    mock_tpu_info.mxu_column_size = 128
    mock.seal(mock_tpu_info)
    mock_get_tpu_info.return_value = mock_tpu_info

    dims = gmm_v2.Dimensions(
        size_m=256,
        size_k=1024,
        size_n=1024,
        size_group=1,
        size_lhs_group=1,
        size_lhs_sublane=8,
    )
    lhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.bfloat16),
        quant_dtype=None,
        has_scale=False,
        has_bias=False,
        quant_block_size=None,
    )
    rhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.bfloat16),
        quant_dtype=None,
        has_scale=False,
        has_bias=False,
        quant_block_size=None,
        is_transposed=False,
    )

    # Test with very large memory limit -> should use 3 buffers
    tiles_large = gmm_v2.calculate_tiling(
        dims=dims,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        vmem_limit_bytes=100000000,
    )
    self.assertEqual(tiles_large.lhs_buffer_count, 3)

    # Calculate memory for a 3-buffer tile vs 2-buffer tile to find the threshold
    # Instead of guessing, we can try a limit that allows the exact tiles_large sizes
    # for 2 buffers, but rejects it for 3 buffers.
    # LHS vmem = lhs_buffer_count * tm * tk * 2 bytes.
    # RHS weight vmem = 3 * tk * tn * 2 bytes (always 3 buffers).
    tk, tn, tm = tiles_large.tile_k, tiles_large.tile_n, tiles_large.tile_m
    rhs_weight_size = tk * tn * 2
    lhs_tile_size = tm * tk * 2
    acc_vmem = tm * tn * 4
    out_vmem = 2 * tm * tn * 2
    base_vmem = acc_vmem + out_vmem + 3 * rhs_weight_size

    # Limit to enough memory for 2 buffers but not 3 buffers
    tight_limit = base_vmem + 2 * lhs_tile_size

    tiles_tight = gmm_v2.calculate_tiling(
        dims=dims,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        vmem_limit_bytes=tight_limit,
    )
    self.assertEqual(tiles_tight.lhs_buffer_count, 2)

    # Ensure tiles_tight was able to preserve the same tile sizes as tiles_large
    # (since the whole point of falling back to 2 is to avoid shrinking tiles)
    self.assertEqual(tiles_tight.tile_m, tiles_large.tile_m)
    self.assertEqual(tiles_tight.tile_n, tiles_large.tile_n)
    self.assertEqual(tiles_tight.tile_k, tiles_large.tile_k)

  @mock.patch.object(gmm_v2.pltpu, "get_tpu_info")
  def test_calculate_tiling_tpu_gen8(self, mock_get_tpu_info):
    mock_tpu_info = mock.create_autospec(
        gmm_v2.pltpu.TpuInfo, instance=True, spec_set=False
    )
    mock_tpu_info.num_lanes = 128
    mock_tpu_info.mxu_column_size = 128
    mock_tpu_info.generation = 8
    mock.seal(mock_tpu_info)
    mock_get_tpu_info.return_value = mock_tpu_info

    # 1. qwen_3_5_prefill_down: 81920 tokens, 64 groups of 1280
    dims_down = gmm_v2.Dimensions(
        size_m=81920,
        size_k=1024,
        size_n=4096,
        size_group=64,
        size_lhs_group=1,
        size_lhs_sublane=8,
    )
    lhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.bfloat16),
        quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
        has_scale=False,
        has_bias=False,
        quant_block_size=None,
    )
    rhs_cfgs_down = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.bfloat16),
        quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
        has_scale=True,
        has_bias=False,
        quant_block_size=1024,
        is_transposed=False,
    )
    vmem_limit_bytes = int(192 * 1024 * 1024 * 0.9)
    tiles_down = gmm_v2.calculate_tiling(
        dims=dims_down,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs_down,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    # At most 2 buckets on TPU8, to stay within instruction memory.
    self.assertEqual(tiles_down.tile_m, 512)
    self.assertEqual(tiles_down.bucket_base, 256)

    # 2. qwen_3_5_prefill_conc8_gate_up: 163840 tokens, 64 groups of 2560
    dims_conc8 = gmm_v2.Dimensions(
        size_m=163840,
        size_k=4096,
        size_n=2048,
        size_group=64,
        size_lhs_group=1,
        size_lhs_sublane=8,
    )
    rhs_cfgs_gateup = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.bfloat16),
        quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
        has_scale=True,
        has_bias=False,
        quant_block_size=4096,
        is_transposed=False,
    )
    tiles_conc8 = gmm_v2.calculate_tiling(
        dims=dims_conc8,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs_gateup,
        vmem_limit_bytes=vmem_limit_bytes,
        fuse_act="silu",
    )
    self.assertEqual(tiles_conc8.tile_m, 512)
    self.assertEqual(tiles_conc8.bucket_base, 256)
    self.assertGreaterEqual(tiles_conc8.lhs_buffer_count, 3)

  @parameterized.named_parameters(
      dict(
          testcase_name="fwd",
          size_k=2048,
          size_n=1280,
          is_transposed=False,
          fuse_act=None,
          expected_tile_n=640,
      ),
      dict(
          testcase_name="fwd_fused_act",
          size_k=2048,
          size_n=2 * 1280,
          is_transposed=False,
          fuse_act="silu",
          expected_tile_n=640,
      ),
      dict(
          testcase_name="dlhs",
          size_k=1280,
          size_n=2048,
          is_transposed=True,
          fuse_act=None,
          expected_tile_n=1024,
      ),
      dict(
          testcase_name="odd_lane_multiple_fwd",
          size_k=1408,
          size_n=1664,
          is_transposed=False,
          fuse_act=None,
          expected_tile_n=896,
      ),
      dict(
          testcase_name="odd_lane_multiple_dlhs",
          size_k=1664,
          size_n=1408,
          is_transposed=True,
          fuse_act=None,
          expected_tile_n=768,
      ),
  )
  def test_calculate_tiling_megacore_splits_n(
      self,
      size_k: int,
      size_n: int,
      is_transposed: bool,
      fuse_act: str | None,
      expected_tile_n: int,
  ) -> None:
    # With MegaCore, only the n grid dimension is split across TensorCores, so
    # the n tiles must be a multiple of num_cores even when a single n tile
    # fits in VMEM.
    dims = _make_dims(size_m=98304, size_k=size_k, size_n=size_n)
    rhs_cfgs = _bf16_input_configs(is_transposed=is_transposed)
    size_n_per_rhs = size_n // 2 if fuse_act else size_n

    for num_cores, disable_multi_core_mode, want_tile_n in (
        (1, False, size_n_per_rhs),
        (2, True, size_n_per_rhs),
        (2, False, expected_tile_n),
    ):
      with self.subTest(
          num_cores=num_cores, disable_multi_core_mode=disable_multi_core_mode
      ):
        with (
            mock.patch.object(
                gmm_v2.pltpu,
                "get_tpu_info",
                return_value=_mock_tpu_info(num_cores=num_cores),
            ),
            config.disable_multi_core_mode(disable_multi_core_mode),
        ):
          tiles = gmm_v2.calculate_tiling(
              dims=dims,
              lhs_cfgs=_bf16_input_configs(),
              rhs_cfgs=rhs_cfgs,
              vmem_limit_bytes=_VMEM_LIMIT_BYTES,
              fuse_act=fuse_act,
          )
        self.assertEqual(tiles.tile_n, want_tile_n)
        self.assertEqual(tiles.tile_k, size_k)
        self.assertEqual(tiles.tile_n % 128, 0)
        if num_cores > 1 and not disable_multi_core_mode:
          self.assertEqual(pl.cdiv(size_n_per_rhs, tiles.tile_n) % num_cores, 0)

  @parameterized.named_parameters(
      # Tiles of 128 would be below the MXU-stall floor of 256.
      dict(testcase_name="below_tile_n_floor", size_n=256, is_out_fused=False),
      # Fused outputs require exact tile_n divisibility, so odd lane multiples
      # cannot use a padded MegaCore split.
      dict(testcase_name="fused_output_odd_lanes", size_n=384, is_out_fused=True),
  )
  def test_calculate_tiling_megacore_keeps_unsplittable_n(
      self, size_n: int, is_out_fused: bool
  ) -> None:
    dims = _make_dims(size_m=4096, size_k=1024, size_n=size_n)
    with (
        mock.patch.object(
            gmm_v2.pltpu, "get_tpu_info", return_value=_mock_tpu_info(2)
        ),
        config.disable_multi_core_mode(False),
    ):
      tiles = gmm_v2.calculate_tiling(
          dims=dims,
          lhs_cfgs=_bf16_input_configs(),
          rhs_cfgs=_bf16_input_configs(is_transposed=False),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
          is_out_fused=is_out_fused,
      )
    self.assertEqual(tiles.tile_n, size_n)

  def test_calculate_tiling_megacore_rounds_up_vmem_split(self) -> None:
    # VMEM pressure alone yields 3 n tiles; MegaCore rounds that up to 4.
    dims = _make_dims(size_m=4096, size_k=4096, size_n=4096)
    vmem_limit_bytes = 48 * 1024 * 1024
    with mock.patch.object(
        gmm_v2.pltpu, "get_tpu_info", return_value=_mock_tpu_info(1)
    ):
      single_core_tiles = gmm_v2.calculate_tiling(
          dims=dims,
          lhs_cfgs=_bf16_input_configs(),
          rhs_cfgs=_bf16_input_configs(is_transposed=False),
          vmem_limit_bytes=vmem_limit_bytes,
      )
    with (
        mock.patch.object(
            gmm_v2.pltpu, "get_tpu_info", return_value=_mock_tpu_info(2)
        ),
        config.disable_multi_core_mode(False),
    ):
      megacore_tiles = gmm_v2.calculate_tiling(
          dims=dims,
          lhs_cfgs=_bf16_input_configs(),
          rhs_cfgs=_bf16_input_configs(is_transposed=False),
          vmem_limit_bytes=vmem_limit_bytes,
      )
    self.assertEqual(pl.cdiv(4096, single_core_tiles.tile_n), 3)
    self.assertEqual(megacore_tiles.tile_n, 1024)

  @parameterized.named_parameters(
      dict(testcase_name="avg_32_clamped_to_base", size_m=4096, want_tile_m=256),
      dict(testcase_name="avg_256", size_m=32768, want_tile_m=256),
      dict(testcase_name="avg_768", size_m=98304, want_tile_m=768),
      dict(testcase_name="avg_1024", size_m=131072, want_tile_m=1024),
  )
  def test_calculate_tiling_transposed_caps_tile_m_at_avg_group_size(
      self, size_m: int, want_tile_m: int
  ) -> None:
    dims = _make_dims(size_m=size_m, size_k=1280, size_n=2048)
    with (
        mock.patch.object(
            gmm_v2.pltpu, "get_tpu_info", return_value=_mock_tpu_info(2)
        ),
        config.disable_multi_core_mode(False),
    ):
      tiles = gmm_v2.calculate_tiling(
          dims=dims,
          lhs_cfgs=_bf16_input_configs(),
          rhs_cfgs=_bf16_input_configs(is_transposed=True),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      )
    self.assertEqual(tiles.tile_m, want_tile_m)
    self.assertEqual(tiles.bucket_base, 256)
    self.assertEqual(tiles.tile_n, 1024)

  def test_calculate_tiling_transposed_uses_global_group_count_for_ep(
      self,
  ) -> None:
    # Under expert parallelism `size_lhs_group` holds the global group count
    # across which `size_m` rows are routed (`131072 // 512 == 256`).
    dims = gmm_v2.Dimensions(
        size_m=131072,
        size_k=1280,
        size_n=2048,
        size_group=64,
        size_lhs_group=512,
        size_lhs_sublane=8,
    )
    with (
        mock.patch.object(
            gmm_v2.pltpu, "get_tpu_info", return_value=_mock_tpu_info(2)
        ),
        config.disable_multi_core_mode(False),
    ):
      tiles = gmm_v2.calculate_tiling(
          dims=dims,
          lhs_cfgs=_bf16_input_configs(),
          rhs_cfgs=_bf16_input_configs(is_transposed=True),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      )
    self.assertEqual(tiles.tile_m, 256)

  @parameterized.parameters(2048, 4096, 16384, 32768)
  def test_calculate_tiling_forward_preserves_tile_m_on_small_m(
      self, size_m: int
  ) -> None:
    dims = _make_dims(size_m=size_m, size_k=2048, size_n=1280)
    with (
        mock.patch.object(
            gmm_v2.pltpu, "get_tpu_info", return_value=_mock_tpu_info(2)
        ),
        config.disable_multi_core_mode(False),
    ):
      tiles = gmm_v2.calculate_tiling(
          dims=dims,
          lhs_cfgs=_bf16_input_configs(),
          rhs_cfgs=_bf16_input_configs(is_transposed=False),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      )
    self.assertEqual(tiles.tile_m, 512)
    self.assertEqual(tiles.bucket_base, 128)
    self.assertEqual(tiles.tile_n, 640)


class SplitForCoresTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(size=1280, num_tiles=1, min_tile=256, allow_padding=True, expected=(2, 640)),
      dict(size=2048, num_tiles=1, min_tile=256, allow_padding=True, expected=(2, 1024)),
      dict(size=4096, num_tiles=3, min_tile=256, allow_padding=True, expected=(4, 1024)),
      # 1536 / 2 = 768 is lane-aligned.
      dict(size=1536, num_tiles=1, min_tile=256, allow_padding=True, expected=(2, 768)),
      # Odd multiples of 128 lanes fall back to the fewest padded lane-aligned
      # tiles when allow_padding=True, and return None when allow_padding=False.
      dict(size=1408, num_tiles=1, min_tile=256, allow_padding=True, expected=(2, 768)),
      dict(size=1664, num_tiles=1, min_tile=256, allow_padding=True, expected=(2, 896)),
      dict(size=640, num_tiles=1, min_tile=256, allow_padding=True, expected=(2, 384)),
      dict(size=384, num_tiles=1, min_tile=256, allow_padding=True, expected=(2, 256)),
      dict(size=1408, num_tiles=1, min_tile=256, allow_padding=False, expected=None),
      dict(size=640, num_tiles=1, min_tile=256, allow_padding=False, expected=None),
      dict(size=640, num_tiles=1, min_tile=128, allow_padding=False, expected=None),
      dict(size=384, num_tiles=1, min_tile=128, allow_padding=False, expected=None),
      # Guard against cand where cdiv(size, tile) != cand (640 / 256 -> 3 != 4).
      dict(size=640, num_tiles=3, min_tile=256, allow_padding=True, expected=None),
      dict(size=256, num_tiles=1, min_tile=256, allow_padding=True, expected=None),
      dict(size=100, num_tiles=1, min_tile=100, allow_padding=True, expected=None),
  )
  def test_split_for_cores(
      self,
      size: int,
      num_tiles: int,
      min_tile: int,
      allow_padding: bool,
      expected: tuple[int, int] | None,
  ) -> None:
    self.assertEqual(
        gmm_v2.split_for_cores(
            size,
            num_tiles,
            num_cores=2,
            num_lanes=128,
            min_tile=min_tile,
            allow_padding=allow_padding,
        ),
        expected,
    )


class TgmmV2CalculateTilingTest(parameterized.TestCase):

  def _calculate_tgmm_tiling(
      self,
      size_k: int,
      size_n: int,
      num_cores: int,
      size_m: int = 98304,
      is_out_fused: bool = False,
      generation: int = 5,
  ) -> gmm_v2.TileSizes:
    dims = _make_dims(size_m=size_m, size_k=size_k, size_n=size_n)
    with (
        mock.patch.object(
            gmm_v2.pltpu,
            "get_tpu_info",
            return_value=_mock_tpu_info(
                num_cores=num_cores, generation=generation
            ),
        ),
        config.disable_multi_core_mode(False),
    ):
      return tgmm_v2.calculate_tgmm_tiling(
          dims=dims,
          lhs_cfgs=_bf16_input_configs(),
          rhs_cfgs=_bf16_input_configs(),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
          out_dtype=jnp.dtype(jnp.bfloat16),
          acc_dtype=jnp.dtype(jnp.float32),
          target_zero_ref_bytes=2 * 1024 * 1024,
          is_out_fused=is_out_fused,
      )

  @parameterized.named_parameters(
      # Split the larger dimension so each core re-reads only the smaller
      # operand.
      dict(
          testcase_name="split_k",
          size_k=2048,
          size_n=1280,
          expected_tile_k=1024,
          expected_tile_n=1280,
      ),
      dict(
          testcase_name="split_n",
          size_k=1280,
          size_n=2048,
          expected_tile_k=1280,
          expected_tile_n=1024,
      ),
      # 640 cannot be split evenly, so prefer the exact split on n (512 -> 256).
      dict(
          testcase_name="split_n_when_k_is_unsplittable",
          size_k=640,
          size_n=512,
          expected_tile_k=640,
          expected_tile_n=256,
      ),
      # When both dims are odd multiples of 128, fall back to a padded split on
      # n, which avoids padding the output along k.
      dict(
          testcase_name="padded_split_n_odd_lanes",
          size_k=1408,
          size_n=1664,
          expected_tile_k=1408,
          expected_tile_n=896,
      ),
      dict(
          testcase_name="padded_split_n_odd_lanes_larger_k",
          size_k=1664,
          size_n=1408,
          expected_tile_k=1664,
          expected_tile_n=768,
      ),
      # n is too small to split, so fall back to a padded split on k.
      dict(
          testcase_name="padded_split_k_when_n_is_unsplittable",
          size_k=1408,
          size_n=128,
          expected_tile_k=768,
          expected_tile_n=128,
      ),
  )
  def test_calculate_tgmm_tiling_megacore(
      self,
      size_k: int,
      size_n: int,
      expected_tile_k: int,
      expected_tile_n: int,
  ) -> None:
    single_core_tiles = self._calculate_tgmm_tiling(size_k, size_n, 1)
    self.assertEqual(single_core_tiles.tile_k, size_k)
    self.assertEqual(single_core_tiles.tile_n, size_n)

    tiles = self._calculate_tgmm_tiling(size_k, size_n, 2)
    self.assertEqual(tiles.tile_k, expected_tile_k)
    self.assertEqual(tiles.tile_n, expected_tile_n)
    num_parallel_tiles = pl.cdiv(size_k, tiles.tile_k) * pl.cdiv(
        size_n, tiles.tile_n
    )
    self.assertEqual(num_parallel_tiles % 2, 0)

  @parameterized.named_parameters(
      dict(
          testcase_name="below_floor",
          size_k=256,
          size_n=256,
          is_out_fused=False,
      ),
      dict(
          testcase_name="fused_output_odd_lanes",
          size_k=1408,
          size_n=1664,
          is_out_fused=True,
      ),
  )
  def test_calculate_tgmm_tiling_megacore_keeps_unsplittable(
      self, size_k: int, size_n: int, is_out_fused: bool
  ) -> None:
    tiles = self._calculate_tgmm_tiling(
        size_k, size_n, 2, is_out_fused=is_out_fused
    )
    self.assertEqual(tiles.tile_k, size_k)
    self.assertEqual(tiles.tile_n, size_n)

  @parameterized.named_parameters(
      dict(testcase_name="avg_32_clamped_to_base", size_m=4096, want_tile_m=256),
      dict(testcase_name="avg_256", size_m=32768, want_tile_m=256),
      dict(testcase_name="avg_768_capped_at_512", size_m=98304, want_tile_m=512),
      dict(
          testcase_name="avg_1024_capped_at_512", size_m=131072, want_tile_m=512
      ),
  )
  def test_calculate_tgmm_tiling_caps_tile_m(
      self, size_m: int, want_tile_m: int
  ) -> None:
    tiles = self._calculate_tgmm_tiling(2048, 1280, num_cores=2, size_m=size_m)
    self.assertEqual(tiles.tile_m, want_tile_m)
    self.assertEqual(tiles.bucket_base, 256)

  @parameterized.named_parameters(
      dict(testcase_name="tpu5", generation=5, num_cores=2, expected_buckets=2),
      dict(testcase_name="tpu6", generation=6, num_cores=1, expected_buckets=2),
      dict(testcase_name="tpu7", generation=7, num_cores=1, expected_buckets=3),
      dict(testcase_name="tpu8", generation=8, num_cores=1, expected_buckets=2),
  )
  def test_calculate_tgmm_tiling_max_num_buckets(
      self, generation: int, num_cores: int, expected_buckets: int
  ) -> None:
    tiles = self._calculate_tgmm_tiling(
        2048, 1280, num_cores=num_cores, size_m=131072, generation=generation
    )
    self.assertEqual(tiles.tile_m, tiles.bucket_base * expected_buckets)
    self.assertEqual(tiles.bucket_base, 256)

  @parameterized.named_parameters(
      dict(testcase_name="tpu7", generation=7, expected_num_buckets=4),
      # TPU8 has enough VMEM for tiles whose unrolled buckets would exceed
      # instruction memory, so it uses fewer buckets.
      dict(testcase_name="tpu8", generation=8, expected_num_buckets=2),
  )
  @mock.patch.object(gmm_v2.pltpu, "get_tpu_info")
  def test_calculate_tiling_max_num_buckets(
      self, mock_get_tpu_info, generation, expected_num_buckets
  ):
    mock_tpu_info = mock.MagicMock()
    mock_tpu_info.generation = generation
    mock_tpu_info.num_lanes = 128
    mock_tpu_info.mxu_column_size = 256
    mock_get_tpu_info.return_value = mock_tpu_info

    # DeepSeek-V3-like FFI: fp8 weights, bf16 activations, fused gate and up.
    dims = gmm_v2.Dimensions(
        size_m=8192,
        size_k=7168,
        size_n=2048,
        size_group=1,
        size_lhs_group=1,
        size_lhs_sublane=8,
    )
    lhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.bfloat16),
        quant_dtype=None,
        has_scale=False,
        has_bias=False,
        quant_block_size=None,
    )
    rhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(jnp.float8_e4m3fn),
        quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
        has_scale=True,
        has_bias=False,
        quant_block_size=256,
        is_transposed=False,
    )

    # VMEM and size_m are large enough that only the bucket cap limits tile_m.
    tiles = gmm_v2.calculate_tiling(
        dims=dims,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        vmem_limit_bytes=1 << 34,
        fuse_act="silu",
    )

    self.assertEqual(tiles.tile_m, tiles.bucket_base * expected_num_buckets)


class GmmV2GroupOffsetNormalizationTest(parameterized.TestCase):
  """Tests for group_offset normalization in gmm_v2 and tgmm_v2."""

  def test_validate_tgmm_inputs(self):
    group_sizes = jnp.array([32, 32, 32], dtype=jnp.int32)
    for offset in (None, 0, 1, jnp.int32(1), jnp.array([1], jnp.int32)):
      tgmm_v2.validate_tgmm_inputs(
          group_sizes, num_actual_groups=2, group_offset=offset
      )
    with self.assertRaises(ValueError):
      tgmm_v2.validate_tgmm_inputs(
          group_sizes, num_actual_groups=2, group_offset=2
      )

  @parameterized.named_parameters(
      ("none", lambda: None, 0),
      ("zero_int", lambda: 0, 0),
      ("pos_int", lambda: 1, 1),
      ("scalar_array", lambda: jnp.int32(1), 1),
      ("1d_array", lambda: jnp.array([1], dtype=jnp.int32), 1),
  )
  def test_normalize_group_offset(self, make_offset, expected):
    actual = gmm_v2.normalize_group_offset(make_offset())
    self.assertEqual(actual.shape, (1,))
    self.assertEqual(actual.dtype, jnp.int32)
    self.assertEqual(int(actual[0]), expected)


class GmmV2VmemStressTest(parameterized.TestCase):
  """AOT compilation and VMEM stress tests for GMM v2.

  Verifies that large model shapes and DLHS transpose_rhs configurations compile
  ahead-of-time and fit within TPU VMEM capacity without triggering
  RESOURCE_EXHAUSTED errors.
  """

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 5:
      self.skipTest("Only supported on TPU gen 5+.")
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name="vmem_stress_transpose_rhs",
          batch_size=65536,
          num_groups=128,
          in_size=3072,
          out_size=2048,
          transpose_rhs=True,
          tile_info=gmm_v2.TileSizes(
              tile_m=512, tile_k=3072, tile_n=2048, bucket_base=256
          ),
      ),
      dict(
          testcase_name="vmem_stress_normal_rhs",
          batch_size=65536,
          num_groups=128,
          in_size=3072,
          out_size=2048,
          transpose_rhs=False,
      ),
      dict(
          testcase_name="megablox_yt_moe_dlhs",
          batch_size=65536,
          num_groups=64,
          in_size=1536,
          out_size=1280,
          transpose_rhs=True,
      ),
  )
  def test_aot_compile_vmem_stress(
      self,
      batch_size,
      num_groups,
      in_size,
      out_size,
      transpose_rhs,
      tile_info=gmm_v2.calculate_tiling_from_request,
  ):
    group_sizes = get_group_sizes(batch_size, num_groups)
    lhs_spec = jax.ShapeDtypeStruct((batch_size, in_size), jnp.bfloat16)
    if transpose_rhs:
      rhs_spec = jax.ShapeDtypeStruct(
          (num_groups, out_size, in_size), jnp.bfloat16
      )
    else:
      rhs_spec = jax.ShapeDtypeStruct(
          (num_groups, in_size, out_size), jnp.bfloat16
      )

    def gmm_fn(lhs, rhs, sizes):
      return gmm_v2.gmm_v2(
          lhs, rhs, sizes, transpose_rhs=transpose_rhs, tile_info=tile_info
      )

    lowered = jax.jit(gmm_fn).lower(lhs_spec, rhs_spec, group_sizes)
    compiled = lowered.compile()
    self.assertIsNotNone(compiled)


class GmmV2VmapTest(parameterized.TestCase):
  """Tests verifying jax.vmap compatibility for GMM and TGMM v2."""

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 5:
      self.skipTest("Only supported on TPU gen 5+.")
    super().setUp()

  @parameterized.named_parameters(
      ("multi_core_mode", False),
      ("single_core_fallback", True),
  )
  def test_gmm_vmap(self, disable_multi_core_mode: bool):
    if not disable_multi_core_mode and jax.__version_info__ <= (0, 11, 2):
      self.skipTest("Multi-core vmap requires JAX > 0.11.2.")
    # Tests jax.vmap on gmm_v2 with batched LHS and group_sizes.
    batch_size = 128
    in_size = 256
    out_size = 256
    num_groups = 4
    vmap_size = 2

    key = jax.random.key(42)
    k0, k1 = jax.random.split(key, 2)
    lhs = jax.random.normal(
        k0, (vmap_size, batch_size, in_size), dtype=jnp.bfloat16
    )
    rhs = jax.random.normal(
        k1, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes_list = [
        get_group_sizes(batch_size, num_groups) for _ in range(vmap_size)
    ]
    group_sizes = jnp.stack(group_sizes_list)

    def gmm_fn(x, w, gs):
      return gmm_v2.gmm_v2(x, w, gs)

    with config.disable_multi_core_mode(disable_multi_core_mode):
      vmapped_fn = jax.jit(jax.vmap(gmm_fn, in_axes=(0, None, 0)))
      actual = vmapped_fn(lhs, rhs, group_sizes)

    # Verify numerical equivalence with batched reference.
    expected = jnp.stack([
        reference_gmm(lhs[i], rhs, group_sizes_list[i])
        for i in range(vmap_size)
    ])
    assert_arrays_all_close(actual, expected)

  @parameterized.named_parameters(
      ("multi_core_mode", False),
      ("single_core_fallback", True),
  )
  def test_tgmm_vmap(self, disable_multi_core_mode: bool):
    if not disable_multi_core_mode and jax.__version_info__ <= (0, 11, 2):
      self.skipTest("Multi-core vmap requires JAX > 0.11.2.")
    # Tests jax.vmap on tgmm_v2 with batched LHS, RHS, and group_sizes.
    batch_size = 128
    in_size = 256
    out_size = 256
    num_groups = 4
    vmap_size = 2

    key = jax.random.key(42)
    k0, k1 = jax.random.split(key, 2)
    lhs = jax.random.normal(
        k0, (vmap_size, batch_size, in_size), dtype=jnp.bfloat16
    )
    rhs = jax.random.normal(
        k1, (vmap_size, batch_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes_list = [
        get_group_sizes(batch_size, num_groups) for _ in range(vmap_size)
    ]
    group_sizes = jnp.stack(group_sizes_list)

    def tgmm_fn(x, y, gs):
      return tgmm_v2.tgmm_v2(
          x,
          y,
          gs,
          num_actual_groups=num_groups,
          preferred_element_type=jnp.bfloat16,
      )

    with config.disable_multi_core_mode(disable_multi_core_mode):
      vmapped_fn = jax.jit(jax.vmap(tgmm_fn, in_axes=(0, 0, 0)))
      actual = vmapped_fn(lhs, rhs, group_sizes)

    # Verify numerical equivalence with batched reference.
    expected = jnp.stack([
        reference_tgmm(
            lhs[i].swapaxes(0, 1), rhs[i], group_sizes_list[i], num_groups
        )
        for i in range(vmap_size)
    ])
    assert_arrays_all_close(actual, expected)


class GmmV2OddLaneMultipleMegaCoreTest(parameterized.TestCase):
  """Hardware tests for odd-lane-multiple dimensions under MegaCore splitting."""

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 5:
      self.skipTest("Only supported on TPU gen 5+.")
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name="fwd_1408x1664_f32_rhs",
          in_size=1408,
          out_size=1664,
          transpose_rhs=False,
          rhs_dtype=jnp.float32,
          tile_n=896,
      ),
      dict(
          testcase_name="fwd_1664x1408_f32_rhs",
          in_size=1664,
          out_size=1408,
          transpose_rhs=False,
          rhs_dtype=jnp.float32,
          tile_n=768,
      ),
      dict(
          testcase_name="dlhs_1664x1408_f32_rhs",
          in_size=1664,
          out_size=1408,
          transpose_rhs=True,
          rhs_dtype=jnp.float32,
          tile_n=768,
      ),
      dict(
          testcase_name="dlhs_1408x1664_f32_rhs",
          in_size=1408,
          out_size=1664,
          transpose_rhs=True,
          rhs_dtype=jnp.float32,
          tile_n=896,
      ),
      dict(
          testcase_name="fwd_1408x1664_bf16_rhs",
          in_size=1408,
          out_size=1664,
          transpose_rhs=False,
          rhs_dtype=jnp.bfloat16,
          tile_n=896,
      ),
  )
  def test_gmm_odd_lane_multiple_padded_tile_n(
      self,
      in_size: int,
      out_size: int,
      transpose_rhs: bool,
      rhs_dtype: jnp.dtype,
      tile_n: int,
  ):
    batch_size = 512
    num_groups = 8
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs_shape = (
        (num_groups, out_size, in_size)
        if transpose_rhs
        else (num_groups, in_size, out_size)
    )
    rhs = jax.random.normal(k1, rhs_shape, dtype=rhs_dtype)
    group_sizes = get_group_sizes(batch_size, num_groups)

    rhs_ref = rhs.swapaxes(1, 2) if transpose_rhs else rhs
    expected = reference_gmm(lhs, rhs_ref, group_sizes)

    # Test both explicit padded tile_n and the default MegaCore heuristic.
    for tile_info in (
        gmm_v2.TileSizes(
            tile_m=256, tile_k=in_size, tile_n=tile_n, bucket_base=128
        ),
        gmm_v2.calculate_tiling_from_request,
    ):
      with self.subTest(tile_info=tile_info):
        actual = gmm_v2.gmm_v2(
            lhs,
            rhs,
            group_sizes,
            transpose_rhs=transpose_rhs,
            tile_info=tile_info,
            preferred_element_type=jnp.bfloat16,
        )
        self.assertEqual(actual.shape, (batch_size, out_size))
        assert_arrays_all_close(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="split_n_1408x1664",
          in_size=1408,
          out_size=1664,
          tile_k=1408,
          tile_n=896,
      ),
      dict(
          testcase_name="split_k_1408x1664",
          in_size=1408,
          out_size=1664,
          tile_k=768,
          tile_n=1664,
      ),
      dict(
          testcase_name="split_k_1664x1408",
          in_size=1664,
          out_size=1408,
          tile_k=896,
          tile_n=1408,
      ),
      dict(
          testcase_name="split_n_1664x1408",
          in_size=1664,
          out_size=1408,
          tile_k=1664,
          tile_n=768,
      ),
      dict(
          testcase_name="split_k_1408x128",
          in_size=1408,
          out_size=128,
          tile_k=768,
          tile_n=128,
      ),
  )
  def test_tgmm_odd_lane_multiple_padded_tiles(
      self, in_size: int, out_size: int, tile_k: int, tile_n: int
  ):
    batch_size = 512
    num_groups = 8
    k0, k1 = jax.random.split(jax.random.key(1), 2)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k1, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = reference_tgmm(lhs.swapaxes(0, 1), rhs, group_sizes, num_groups)

    for tile_info in (
        gmm_v2.TileSizes(
            tile_m=256, tile_k=tile_k, tile_n=tile_n, bucket_base=256
        ),
        tgmm_v2.calculate_tgmm_tiling_from_request,
    ):
      with self.subTest(tile_info=tile_info):
        actual = tgmm_v2.tgmm_v2(
            lhs,
            rhs,
            group_sizes,
            num_actual_groups=num_groups,
            tile_info=tile_info,
            preferred_element_type=jnp.bfloat16,
        )
        self.assertEqual(actual.shape, (num_groups, in_size, out_size))
        assert_arrays_all_close(actual, expected)


class DecodeTilingAndSparseGroupsTest(parameterized.TestCase):
  """Tests for low-batch MoE decode tiling, chunked metadata, and edge cases."""

  @parameterized.named_parameters(
      ("e512_72x1_gen5", 72, 16, 64),
      ("e512_72x2_gen5", 144, 16, 64),
      ("e512_72x4_gen5", 288, 24, 96),
      ("e512_32x4_gen5", 128, 16, 64),
      ("e512_8x1_gen5", 8, 8, 8),
      ("e512_1x1_gen5", 1, 1, 1),
      dict(
          testcase_name="ep8_9x1_gen5",
          size_m=128,
          size_group=64,
          want_bucket_base=16,
          want_tile_m=64,
      ),
      ("e512_1x256_gen5", 256, 16, 64),
      ("e512_1x512_gen5", 512, 32, 128),
      dict(
          testcase_name="e512_1x512_gen5_split_k",
          size_m=512,
          want_bucket_base=32,
          want_tile_m=128,
          vmem_limit_bytes=3 * 1024**2,
          want_tile_k=2048,
          want_rhs_buffer_count=2,
      ),
      # The zero-initialization scratch does not leave room for three RHS
      # buffers at the largest bucket.
      dict(
          testcase_name="e512_1x512_gen5_split_k_zero_init",
          size_m=512,
          want_bucket_base=32,
          want_tile_m=128,
          vmem_limit_bytes=3328 * 1024,
          zero_initialize=True,
          want_tile_k=2048,
          want_rhs_buffer_count=2,
          want_lhs_buffer_count=2,
      ),
      # Two LHS and three RHS buffers fit, so the RHS keeps three.
      dict(
          testcase_name="e512_1x512_gen5_split_k_keeps_rhs_buffers",
          size_m=512,
          want_bucket_base=32,
          want_tile_m=128,
          vmem_limit_bytes=3552 * 1024,
          want_tile_k=2048,
          want_rhs_buffer_count=3,
          want_lhs_buffer_count=2,
      ),
      # With split K, each M tile streams the hot expert's weights again, so
      # the decode tiles keep the default tile_m of 256 and only shrink its
      # buckets.
      dict(
          testcase_name="e64_1x512_gen5_n384_zero_init",
          size_m=512,
          size_group=64,
          size_lhs_group=64,
          size_n=384,
          want_bucket_base=64,
          want_tile_m=256,
          vmem_limit_bytes=5324 * 1024,
          zero_initialize=True,
          want_tile_k=2048,
          want_tile_n=256,
          want_rhs_buffer_count=2,
          want_lhs_buffer_count=2,
      ),
      # Subchannel scales: the decode tiles keep the default (1536, 256) N and
      # K tiles instead of growing tile_k past them.
      dict(
          testcase_name="e64_1x192_gen5_subchannel_scales",
          size_m=192,
          size_group=64,
          size_lhs_group=64,
          size_k=3072,
          size_n=384,
          rhs_quant_block_size=192,
          want_bucket_base=16,
          want_tile_m=64,
          vmem_limit_bytes=3364 * 1024,
          zero_initialize=True,
          want_tile_k=1536,
          want_tile_n=256,
          want_rhs_buffer_count=3,
          want_lhs_buffer_count=3,
      ),
      ("e512_1x2048_gen5", 2048, 128, 512),
      dict(
          testcase_name="prefill_avg16_gen5",
          size_m=2048,
          size_group=128,
          size_lhs_group=128,
          want_bucket_base=128,
          want_tile_m=512,
      ),
      dict(
          testcase_name="e512_72x1_gen7x_fp8",
          size_m=80,
          generation=7,
          sublane=16,
          mxu_column_size=256,
          quant_dtype=jnp.float8_e4m3fn,
          want_bucket_base=16,
          want_tile_m=64,
      ),
      dict(
          testcase_name="e512_72x1_gen8i_prequantized_fp8",
          size_m=96,
          generation=8,
          sublane=32,
          mxu_column_size=256,
          quant_dtype=jnp.float8_e4m3fn,
          prequantized_lhs=True,
          want_bucket_base=32,
          want_tile_m=64,
      ),
      dict(
          testcase_name="e512_72x1_gen8t",
          size_m=96,
          generation=8,
          sublane=32,
          mxu_column_size=256,
          want_bucket_base=32,
          want_tile_m=64,
      ),
      dict(
          testcase_name="e512_72x1_gen5p_megacore",
          size_m=72,
          size_n=1024,
          num_cores=2,
          want_bucket_base=16,
          want_tile_m=64,
          want_tile_n=512,
      ),
  )
  def test_calculate_tiling_low_batch_decode(
      self,
      size_m: int,
      want_bucket_base: int,
      want_tile_m: int,
      *,
      size_group: int = 512,
      size_lhs_group: int = 512,
      size_k: int = 4096,
      size_n: int = 256,
      rhs_quant_block_size: int | None = None,
      generation: int = 5,
      sublane: int = 8,
      mxu_column_size: int = 128,
      num_cores: int = 1,
      quant_dtype: jax.typing.DTypeLike = jnp.int8,
      prequantized_lhs: bool = False,
      vmem_limit_bytes: int = _VMEM_LIMIT_BYTES,
      zero_initialize: bool = False,
      want_tile_k: int = 4096,
      want_tile_n: int = 256,
      want_rhs_buffer_count: int = 3,
      want_lhs_buffer_count: int | None = None,
  ):
    dims = gmm_v2.Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=min(sublane, size_m),
    )
    lhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(quant_dtype if prequantized_lhs else jnp.bfloat16),
        quant_dtype=jnp.dtype(quant_dtype),
        quant_block_size=size_k if prequantized_lhs else 512,
        has_scale=prequantized_lhs,
    )
    rhs_cfgs = gmm_v2.InputConfigs(
        dtype=jnp.dtype(quant_dtype),
        quant_dtype=jnp.dtype(quant_dtype),
        quant_block_size=rhs_quant_block_size or size_k,
        has_scale=True,
    )
    with mock.patch.object(
        gmm_v2.pltpu,
        "get_tpu_info",
        return_value=_mock_tpu_info(num_cores, generation, mxu_column_size),
    ):
      tiles = gmm_v2.calculate_tiling(
          dims=dims,
          lhs_cfgs=lhs_cfgs,
          rhs_cfgs=rhs_cfgs,
          vmem_limit_bytes=vmem_limit_bytes,
          zero_initialize=zero_initialize,
          out_dtype=jnp.dtype(jnp.bfloat16),
      )
    self.assertEqual(tiles.bucket_base, want_bucket_base)
    self.assertEqual(tiles.tile_m, want_tile_m)
    self.assertEqual(tiles.tile_k, want_tile_k)
    self.assertEqual(tiles.tile_n, want_tile_n)
    self.assertEqual(tiles.rhs_buffer_count, want_rhs_buffer_count)
    if want_lhs_buffer_count is not None:
      self.assertEqual(tiles.lhs_buffer_count, want_lhs_buffer_count)

  @parameterized.named_parameters(
      ("e1_bf16_n128_zero_init", 512, 1, 128, 5324),
      ("e64_dense_bf16_n128_zero_init", 576, 64, 128, 5340),
  )
  def test_calculate_tiling_keeps_default_outside_decode(
      self, size_m: int, num_groups: int, size_n: int, vmem_limit_kib: int
  ):
    dims = gmm_v2.Dimensions(
        size_m=size_m,
        size_k=4096,
        size_n=size_n,
        size_group=num_groups,
        size_lhs_group=num_groups,
        size_lhs_sublane=8,
    )
    with mock.patch.object(
        gmm_v2.pltpu, "get_tpu_info", return_value=_mock_tpu_info(1)
    ):
      tiles = gmm_v2.calculate_tiling(
          dims,
          _bf16_input_configs(),
          _bf16_input_configs(),
          vmem_limit_kib * 1024,
          zero_initialize=True,
          out_dtype=jnp.dtype(jnp.bfloat16),
      )
      default = gmm_v2._default_tiling(
          dims,
          _bf16_input_configs(),
          _bf16_input_configs(),
          vmem_limit_kib * 1024,
      )
    self.assertEqual(tiles, default)
    self.assertEqual(
        tiles,
        gmm_v2.TileSizes(tile_m=128, tile_k=4096, tile_n=128, bucket_base=128),
    )

  def test_decode_tiling_never_exceeds_default_buffers(self):
    """Each decode buffer is at most as large as for the default tiles."""
    int8 = jnp.dtype(jnp.int8)
    bf16 = jnp.dtype(jnp.bfloat16)
    for (
        (size_m, num_groups, size_k, size_n),
        quantized,
        block,
        limit_kib,
        zi,
    ) in itertools.product(
        (
            (72, 512, 4096, 256),
            (192, 64, 3072, 384),
            (512, 64, 4096, 384),
            (512, 512, 4096, 256),
            (1280, 512, 1024, 4096),
        ),
        (True, False),
        (None, 192),
        (2500, 3072, 3328, 5324, 9952, 58982),
        (True, False),
    ):
      if block is not None and (not quantized or size_k % block):
        continue
      dims = gmm_v2.Dimensions(
          size_m=size_m,
          size_k=size_k,
          size_n=size_n,
          size_group=num_groups,
          size_lhs_group=num_groups,
          size_lhs_sublane=8,
      )
      lhs_cfgs = gmm_v2.InputConfigs(
          dtype=bf16,
          quant_dtype=int8 if quantized else None,
          quant_block_size=512,
      )
      rhs_cfgs = gmm_v2.InputConfigs(
          dtype=int8 if quantized else bf16,
          quant_dtype=int8 if quantized else None,
          quant_block_size=block or size_k,
          has_scale=quantized,
      )
      args = (dims, lhs_cfgs, rhs_cfgs, limit_kib * 1024)
      with (
          self.subTest(
              dims=dims,
              quantized=quantized,
              block=block,
              limit_kib=limit_kib,
              zero_initialize=zi,
          ),
          mock.patch.object(
              gmm_v2.pltpu, "get_tpu_info", return_value=_mock_tpu_info(1)
          ),
      ):
        try:
          default = gmm_v2._default_tiling(*args)
        except ValueError:
          continue
        tiles = gmm_v2.calculate_tiling(
            *args, zero_initialize=zi, out_dtype=bf16
        )
        self.assertEqual(tiles.tile_k, default.tile_k)
        self.assertEqual(tiles.tile_n, default.tile_n)
        self.assertLessEqual(tiles.tile_m, default.tile_m)
        self.assertLessEqual(
            tiles.lhs_buffer_count * tiles.tile_m,
            default.lhs_buffer_count * default.tile_m,
        )
        self.assertLessEqual(tiles.rhs_buffer_count, default.rhs_buffer_count)
        self.assertEqual(tiles.tile_m % tiles.bucket_base, 0)
        self.assertLessEqual(tiles.tile_m // tiles.bucket_base, 4)

  @parameterized.named_parameters(
      dict(
          testcase_name="e512_72x1_bf16_zi_true",
          num_groups=512,
          pattern="sparse_72x1",
          quantized=False,
          zero_initialize=True,
          group_offset=0,
          local_groups=512,
          num_k_tiles=1,
      ),
      dict(
          testcase_name="e512_72x1_int8_zi_false",
          num_groups=512,
          pattern="sparse_72x1",
          quantized=True,
          zero_initialize=False,
          group_offset=0,
          local_groups=512,
          num_k_tiles=1,
      ),
      dict(
          testcase_name="e512_72x2_int8_numk2",
          num_groups=512,
          pattern="sparse_72x2",
          quantized=True,
          zero_initialize=True,
          group_offset=0,
          local_groups=512,
          num_k_tiles=2,
      ),
      dict(
          testcase_name="e512_72x2_bf16_numk2_bias_and_partial_sum",
          num_groups=512,
          pattern="sparse_72x2",
          quantized=False,
          zero_initialize=True,
          group_offset=0,
          local_groups=512,
          num_k_tiles=2,
          use_bias_and_partial_sum=True,
      ),
      dict(
          testcase_name="e512_skewed_int8",
          num_groups=512,
          pattern="skewed",
          quantized=True,
          zero_initialize=True,
          group_offset=0,
          local_groups=512,
          num_k_tiles=1,
      ),
      dict(
          testcase_name="e512_all_empty_zi_true",
          num_groups=512,
          pattern="all_empty",
          quantized=False,
          zero_initialize=True,
          group_offset=0,
          local_groups=512,
          num_k_tiles=1,
      ),
      dict(
          testcase_name="e512_single_active_zi_true",
          num_groups=512,
          pattern="single_active",
          quantized=False,
          zero_initialize=True,
          group_offset=0,
          local_groups=512,
          num_k_tiles=1,
      ),
      dict(
          testcase_name="e512_ep8_offset448_zi_true",
          num_groups=512,
          pattern="ep_offset",
          quantized=False,
          zero_initialize=True,
          group_offset=448,
          local_groups=64,
          num_k_tiles=1,
      ),
      dict(
          testcase_name="e512_ep8_offset448_zi_false",
          num_groups=512,
          pattern="ep_offset",
          quantized=False,
          zero_initialize=False,
          group_offset=448,
          local_groups=64,
          num_k_tiles=1,
      ),
  )
  def test_decode_sparse_groups_e2e(
      self,
      num_groups: int,
      pattern: str,
      quantized: bool,
      zero_initialize: bool,
      group_offset: int,
      local_groups: int,
      num_k_tiles: int,
      use_bias_and_partial_sum: bool = False,
  ):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

    in_size, out_size = 512, 256
    gs = np.zeros(num_groups, dtype=np.int32)
    if pattern == "sparse_72x1":
      gs[np.linspace(0, num_groups - 1, 72, dtype=int)] = 1
      batch_size = 72
    elif pattern == "sparse_72x2":
      gs[np.linspace(0, num_groups - 1, 72, dtype=int)] = 2
      batch_size = 144
    elif pattern == "skewed":
      idx = np.linspace(0, num_groups - 1, 72, dtype=int)
      gs[idx[:70]] = 1
      gs[idx[70:]] = 32
      batch_size = 134
    elif pattern == "all_empty":
      batch_size = 16
    elif pattern == "single_active":
      gs[257] = 3
      batch_size = 16
    elif pattern == "ep_offset":
      gs[:group_offset:8] = 1
      gs[group_offset + np.linspace(0, local_groups - 1, 9, dtype=int)] = 1
      batch_size = int(gs.sum()) + 4
    else:
      raise ValueError(f"Unknown pattern: {pattern}")

    k0, k1 = jax.random.split(jax.random.key(42), 2)
    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(gs)
    go = jnp.array([group_offset], dtype=jnp.int32) if group_offset else None
    if quantized:
      rhs = jax.random.randint(
          k1, (local_groups, in_size, out_size), -127, 127, dtype=jnp.int8
      )
      rhs_scale = (
          jnp.ones((local_groups, 1, 1, out_size), dtype=jnp.float32) * 0.001
      )
      maybe_q = True
    else:
      rhs = jax.random.normal(
          k1, (local_groups, in_size, out_size), dtype=jnp.bfloat16
      )
      rhs_scale = None
      maybe_q = False

    if num_k_tiles == 2:
      tile_info = gmm_v2.TileSizes(
          tile_m=64, tile_k=in_size // 2, tile_n=out_size, bucket_base=16
      )
    else:
      tile_info = gmm_v2.calculate_tiling_from_request

    rhs_bias = partial_sum = None
    if use_bias_and_partial_sum:
      rhs_bias = jnp.full((local_groups, 1, out_size), 4, dtype=jnp.bfloat16)
      partial_sum = jnp.full((batch_size, out_size), 6, dtype=jnp.bfloat16)

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        partial_sum=partial_sum,
        group_offset=go,
        tile_info=tile_info,
        maybe_quantize_lhs=maybe_q,
        zero_initialize=zero_initialize,
    )
    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        partial_sum=partial_sum,
        group_offset=go,
    )
    start = int(gs[:group_offset].sum())
    end = start + int(gs[group_offset : group_offset + local_groups].sum())
    assert_arrays_all_close(actual[start:end], expected[start:end])
    if zero_initialize:
      np.testing.assert_array_equal(actual[:start], 0)
      np.testing.assert_array_equal(actual[end:], 0)

  @parameterized.named_parameters(
      # On v5e, the default tiling compiles at 5,324 KiB and the round-3 decode
      # tiles did not.
      ("hot_n384_zero_init_5324k", 512, 64, 4096, 384, True, None, True, 5324),
      # The default tiling runs out of scoped VMEM at 3,328 KiB with zero init
      # and at 3,072 KiB without it. The decode tiles drop an RHS buffer.
      ("hot_n256_zero_init_3328k", 512, 64, 4096, 256, True, None, True, 3328),
      ("hot_n256_3072k", 512, 64, 4096, 256, True, None, False, 3072),
      # 192-wide subchannel RHS scales. The default tiling compiles here and
      # the round-3 decode tiles did not.
      ("hot_subchannel_3364k", 192, 64, 3072, 384, True, 192, True, 3364),
      # Not a decode call, so the default tiling is used unchanged.
      ("dense_n128_zi_5324k", 512, 1, 4096, 128, False, None, True, 5324),
  )
  def test_tiling_fits_tight_vmem_limit(
      self,
      size_m: int,
      num_groups: int,
      size_k: int,
      size_n: int,
      quantized: bool,
      scale_block: int | None,
      zero_initialize: bool,
      vmem_limit_kib: int,
  ):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_V5E:
      self.skipTest("The VMEM limits were measured on TPU v5e.")

    # Inputs with an exact result. LHS entries are +-1, so online int8
    # quantization is lossless. Each RHS column of each expert has a single
    # +-1 at an expert-dependent row, so every integer partial sum is 0 or
    # +-127 and stays exact in the bf16 accumulator, and each output checks
    # which expert and LHS row were used.
    rng = np.random.default_rng(size_m * size_n)
    lhs = rng.choice([-1.0, 1.0], (size_m, size_k)).astype(np.float32)
    rhs = np.zeros((num_groups, size_k, size_n), np.float32)
    cols = np.arange(size_n)
    for g in range(num_groups):
      rhs[g, (7 * cols + 13 * g) % size_k, cols] = rng.choice([-1, 1], size_n)
    gs = np.zeros(num_groups, dtype=np.int32)
    if num_groups > 1:
      # One hot expert owns all rows except a few routed to its neighbours.
      gs[[20, 22]] = 3
      gs[21] = size_m - 6
    else:
      gs[0] = size_m
    expected = np.zeros((size_m, size_n), np.float32)
    starts = np.concatenate([[0], np.cumsum(gs)])
    for g in np.flatnonzero(gs):
      expected[starts[g] : starts[g + 1]] = (
          lhs[starts[g] : starts[g + 1]] @ rhs[g]
      )

    rhs_scale = None
    if quantized:
      num_blocks = size_k // (scale_block or size_k)
      rhs_scale = jnp.full(
          (num_groups, num_blocks, 1, size_n), 2**-7, jnp.float32
      )
      expected *= 2**-7
    actual = gmm_v2.gmm_v2(
        jnp.asarray(lhs, jnp.bfloat16),
        jnp.asarray(rhs, jnp.int8 if quantized else jnp.bfloat16),
        jnp.asarray(gs),
        rhs_scale=rhs_scale,
        tile_info=gmm_v2.calculate_tiling_from_request,
        vmem_limit_bytes=vmem_limit_kib * 1024,
        maybe_quantize_lhs=quantized,
        zero_initialize=zero_initialize,
    )
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)


if __name__ == "__main__":
  absltest.main()
