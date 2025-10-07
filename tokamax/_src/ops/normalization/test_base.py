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
"""Normalization test base."""

import functools
import inspect
import math

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import test_utils
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import bench_arg_specs


NAMED_ARG_SPECS = {s.full_name: s.args for s in bench_arg_specs.ARG_SPECS}


# pylint: disable=missing-function-docstring
class NormalizationTestBase(parameterized.TestCase):
  """Base class for normalization op tests."""

  def __init__(self, *args, norm_fn, supports_vjp=True):
    super().__init__(*args)
    self._norm_fn = norm_fn
    self._supports_vjp = supports_vjp

  @parameterized.product(
      (
          dict(use_scale_and_offset=False, scale_offset=0.0),
          dict(use_scale_and_offset=True, scale_offset=0.0),
          dict(use_scale_and_offset=True, scale_offset=1.0),
      ),
      shape=((64,), (128, 32), (8, 128, 32), (256, 42)),
  )
  def test_layer_norm(
      self,
      shape,
      use_scale_and_offset,
      scale_offset,
  ):
    if use_scale_and_offset:
      scale_offset_shape = jax.ShapeDtypeStruct((shape[-1],), jnp.float32)
    else:
      scale_offset_shape = None
    self._run_test(
        jax.ShapeDtypeStruct(shape, jnp.float32),
        scale_offset_shape,
        scale_offset_shape,
        scale_offset=scale_offset,
    )

  @parameterized.parameters(0, 1, 2)
  def test_layer_norm_axis(self, axis):
    shape = (24, 32, 40)
    self._run_test(
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct((shape[axis],), jnp.float32),
        jax.ShapeDtypeStruct((shape[axis],), jnp.float32),
        axis=axis,
    )

  @parameterized.product(
      axis=(0, -1),
      vmap_in_axes=(
          ((0, None, None),),
          ((0, 0, None),),
          ((1, None, 0),),
          ((0, None, None), (0, None, None)),
      ),
  )
  def test_layer_norm_vmap(self, axis, vmap_in_axes):
    shape = (24, 32, 40)
    axis_sizes = []
    inner_shape = list(shape)
    for in_axes in vmap_in_axes[::-1]:
      axis_sizes.insert(0, inner_shape.pop(in_axes[0]))
    scale_shape = [inner_shape[axis]]
    offset_shape = [inner_shape[axis]]
    for axis_size, in_axes in zip(axis_sizes, vmap_in_axes):
      if in_axes[1] is not None:
        scale_shape.insert(in_axes[1], axis_size)
      if in_axes[2] is not None:
        offset_shape.insert(in_axes[2], axis_size)
    self._run_test(
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct(scale_shape, jnp.float32),
        jax.ShapeDtypeStruct(offset_shape, jnp.float32),
        axis=axis,
        vmap_in_axes=vmap_in_axes,
    )

  @parameterized.parameters(((128, 32),), ((8, 128, 32),), ((256, 42),))
  def test_rms_norm(self, shape):
    self._run_test(
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct((shape[-1],), jnp.float32),
        offset=None,
        subtract_mean=False,
    )

  def _run_test(
      self,
      x,
      scale,
      offset,
      *,
      axis=-1,
      epsilon=1e-6,
      scale_offset=0.0,
      subtract_mean=True,
      vmap_in_axes=(),
  ):
    rngs = list(jax.random.split(jax.random.PRNGKey(0), 4))
    x = jax.random.normal(rngs.pop(), x.shape, x.dtype)
    if scale is not None:
      scale = jax.random.uniform(rngs.pop(), scale.shape, scale.dtype)
    if offset is not None:
      offset = jax.random.uniform(rngs.pop(), offset.shape, offset.dtype)

    f = functools.partial(
        self._norm_fn,
        axis=axis,
        epsilon=epsilon,
        scale_offset=scale_offset,
        subtract_mean=subtract_mean,
    )

    def f_ref(x, scale, offset):
      x_ = x.astype(jnp.promote_types(x.dtype, jnp.float32))
      mean = None if subtract_mean else 0.0
      y = jax.nn.standardize(x_, axis=axis, mean=mean, epsilon=epsilon)

      param_bcast_axes = list(range(x.ndim))
      del param_bcast_axes[axis]
      if scale is not None:
        y *= jnp.expand_dims(scale + scale_offset, param_bcast_axes)
      if offset is not None:
        y += jnp.expand_dims(offset, param_bcast_axes)
      return y.astype(x.dtype)

    for in_axes in vmap_in_axes:
      f = jax.vmap(f, in_axes=in_axes, out_axes=in_axes[0])
      f_ref = jax.vmap(f_ref, in_axes=in_axes, out_axes=in_axes[0])

    y_expected = f_ref(x, scale, offset)
    y_actual = f(x, scale, offset)

    atols = {jnp.dtype(jnp.float32): 5e-7, jnp.dtype(jnp.bfloat16): 2e-3}
    atol = atols[x.dtype] * float(jnp.sqrt(x.shape[axis]))
    chex.assert_trees_all_close(y_actual, y_expected, atol=atol)

    if not self._supports_vjp:
      return

    y_expected, f_ref_vjp = jax.vjp(f_ref, x, scale, offset)
    y_actual, f_vjp = jax.vjp(f, x, scale, offset)
    chex.assert_trees_all_close(y_actual, y_expected, atol=atol)

    dy = jax.random.normal(rngs.pop(), x.shape, x.dtype)
    dx_expected, dscale_expected, doffset_expected = f_ref_vjp(dy)
    dx_actual, dscale_actual, doffset_actual = f_vjp(dy)
    chex.assert_trees_all_close(dx_actual, dx_expected, atol=atol)
    axis = axis if axis >= 0 else (axis + x.ndim)
    param_bcast_dims = [dim for i, dim in enumerate(x.shape) if i != axis]
    atol = atols[x.dtype] * 10 * float(jnp.sqrt(math.prod(param_bcast_dims)))
    chex.assert_trees_all_close(dscale_actual, dscale_expected, atol=atol)
    atol = atols[x.dtype] * 25 * float(jnp.sqrt(math.prod(param_bcast_dims)))
    chex.assert_trees_all_close(doffset_actual, doffset_expected, atol=atol)

  @parameterized.named_parameters(NAMED_ARG_SPECS.items())
  def test_bench(self, kwargs):
    ba = inspect.signature(base.Normalization.__call__).bind(None, **kwargs)
    ba.apply_defaults()
    ba.arguments.pop('return_residuals')
    self._run_test(*ba.args[1:], **ba.kwargs)


def base_names_and_params(test_name: str) -> list[tuple[str, str]]:
  return test_utils.get_names_and_params(NormalizationTestBase, test_name)


# pylint: enable=missing-function-docstring
