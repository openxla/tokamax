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
import functools
import json
import math
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import jax_triton as jt
from tokamax._src import batching
from tokamax._src import benchmarking
from tokamax._src import hlo_utils
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import numerics
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.gated_linear_unit import pallas_triton as pl_triton_glu
from tokamax._src.ops.normalization import pallas_triton as pl_norm
from tokamax._src.ops.normalization import pallas_triton_vjp as pl_norm_vjp
from tokamax._src.ops.ragged_dot import pallas_triton as pl_ragged_dot
import triton
import triton.language as tl

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import


def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


def add_vector_two(x_ref, o_ref):
  o_ref[...] = x_ref[...] + 2


@jax.jit
def add_vectors_pallas_triton(x: jax.Array, y: jax.Array) -> jax.Array:
  call_1 = pl.pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      name='add_vectors_kernel_1',
      compiler_params=plgpu.CompilerParams(num_warps=2, num_stages=1),
  )
  call_2 = pl.pallas_call(
      add_vector_two,
      grid=(8, 1, 1),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      name='add_vector_two',
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
  )
  out = call_1(x, y)
  out *= call_2(out)
  return jnp.sin(out) + jnp.cos(out)


class DumpHloLibTest(parameterized.TestCase):

  def test_pallas_gpu_tpu(self):
    # Example taken from https://docs.jax.dev/en/latest/pallas/quickstart.html.
    def add_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x + y

    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          add_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x, y)

    x = jnp.arange(8)
    out = add_vectors(x, x)
    out_ref = jnp.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=jnp.int32)
    self.assertTrue(jnp.array_equal(out, out_ref))

    (kernel_info,) = hlo_utils.get_kernel_info(
        add_vectors.lower(x, x), include_xla_kernels=False
    )

    expected_class = (
        hlo_utils.TritonKernelInfo
        if jax.default_backend() == 'gpu'
        else hlo_utils.MosaicTpuKernelInfo
    )
    self.assertIsInstance(kernel_info, expected_class)
    self.assertEqual(
        kernel_info.outputs,
        (jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32),),
    )

  def test_simple_pallas_triton(self):

    if jax.default_backend() != 'gpu':
      self.skipTest('This test only runs on GPU.')

    dtype = jnp.int32
    x = jnp.arange(8, dtype=dtype)
    lowered = add_vectors_pallas_triton.lower(x=x, y=x)
    kernel_info = hlo_utils.get_kernel_info(lowered, include_xla_kernels=False)
    self.assertLen(kernel_info, 2)
    kernel_1, kernel_2 = kernel_info

    self.assertIsInstance(kernel_1, hlo_utils.TritonKernelInfo)
    self.assertIsInstance(kernel_2, hlo_utils.TritonKernelInfo)

    self.assertEqual(kernel_1.kernel_name, 'add_vectors_kernel_1')
    self.assertEqual(kernel_2.kernel_name, 'add_vector_two')

    self.assertEqual(kernel_1.num_warps, 2)
    self.assertEqual(kernel_2.num_warps, 4)

    # TODO: Re-enable checks after bug is fixed.
    _ = """
    self.assertEqual(kernel_1.num_stages, 1)
    self.assertEqual(kernel_2.num_stages, 3)
    """

    shape = jax.ShapeDtypeStruct(shape=(8,), dtype=dtype)
    self.assertEqual(kernel_1.inputs, (shape, shape))
    self.assertEqual(kernel_2.inputs, (shape,))

    self.assertEqual(kernel_2.grid, (8, 1, 1))

  @parameterized.product(
      axis=(-1,),
  )
  def test_pallas_norm(
      self,
      axis,
  ):

    if jax.default_backend() != 'gpu':
      self.skipTest('This test only runs on GPU.')

    dtype = jnp.bfloat16
    x_shape = (16, 64, 128)
    param_shape = (x_shape[axis],)

    f = functools.partial(pl_norm.PallasTritonNormalization(), axis=axis)

    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), num=3)
    x = jax.random.normal(key=key1, shape=x_shape, dtype=dtype)
    scale = jax.random.normal(key=key2, shape=param_shape, dtype=jnp.float32)
    offset = jax.random.normal(key=key3, shape=param_shape, dtype=jnp.float32)

    def layer_norm_loss(x, scale, offset):
      return jnp.sum(f(x, scale, offset))

    f_grad = jax.grad(layer_norm_loss, argnums=(0, 1, 2))
    f_grad_lowered = jax.jit(f_grad).lower(x, scale, offset)

    forward, vjp = hlo_utils.get_kernel_info(
        f_grad_lowered, include_xla_kernels=False
    )

    self.assertIsInstance(forward, hlo_utils.TritonKernelInfo)
    self.assertIsInstance(vjp, hlo_utils.TritonKernelInfo)

    self.assertEqual(forward.kernel_name, 'pallas_layer_norm_fwd_res')
    self.assertEqual(vjp.kernel_name, 'pallas_layer_norm_vjp')

    self.assertLen(forward.inputs, 3)
    self.assertLen(forward.outputs, 3)

    # `x` is canonicalized to a 3D shape.
    x_canonical_shape = (math.prod(x_shape[:-1]), x_shape[-1], 1)
    inputs_ref = (
        jax.ShapeDtypeStruct(shape=x_canonical_shape, dtype=x.dtype),
        jax.ShapeDtypeStruct(shape=(*param_shape, 1), dtype=scale.dtype),
        jax.ShapeDtypeStruct(shape=(*param_shape, 1), dtype=offset.dtype),
    )
    self.assertEqual(forward.inputs, inputs_ref)

    # TODO: add tests for axis once this is in the Pallas HLO.

  def test_jax_triton_simple(self):
    self.skipTest('This test is not supported on OSS due to CUDA backend issues.')

    if jax.default_backend() != 'gpu':
      self.skipTest('This test only runs on GPU.')

    metadata = {'test': 1, 'test2': 'two'}
    metadata_json = bytes(json.dumps(metadata), 'utf-8')
    num_warps = 2

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        block_size: tl.constexpr,
    ):
      """Adds two vectors."""
      pid = tl.program_id(axis=0)
      block_start = pid * block_size
      offsets = block_start + tl.arange(0, block_size)
      mask = offsets < 8
      x = tl.load(x_ptr + offsets, mask=mask)
      y = tl.load(y_ptr + offsets, mask=mask)
      output = x + y
      tl.store(output_ptr + offsets, output, mask=mask)

    @jax.jit
    def add_jax_triton(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
      out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
      block_size = 8
      grid = (triton.cdiv(x.size, block_size),)
      return jt.triton_call(
          x,
          y,
          kernel=add_kernel,
          out_shape=out_shape,
          grid=grid,
          block_size=block_size,
          num_warps=num_warps,
          serialized_metadata=metadata_json,
          name='add_kernel_jax_triton',
      )

    x = jnp.arange(8)
    y = jnp.arange(8, 16)

    kernels = hlo_utils.get_kernel_info(
        add_jax_triton.lower(x, y), include_xla_kernels=False
    )
    self.assertLen(kernels, 1)
    kernels = kernels[0]
    self.assertIsInstance(kernels, hlo_utils.TritonKernelInfo)
    self.assertEqual(kernels.kernel_name, 'add_kernel')

    metadata_load = json.loads(kernels.metadata)
    self.assertEqual(metadata_load, metadata)
    self.assertEqual(kernels.num_warps, num_warps)
    self.assertEqual(kernels.grid, (1, 1, 1))
    self.assertEqual(kernels.compute_capability, jt.get_compute_capability(0))

  def test_get_opspecs_from_lowered_jax(self):

    if jax.default_backend() != 'gpu':
      self.skipTest('This test only runs on GPU.')

    norm_op = pl_norm.PallasTritonNormalization()
    glu_op = pl_triton_glu.PallasTritonGatedLinearUnit()

    # Create a string of Tokamax ops in Jax, lower it to HLO, and extract the
    # kernel spec from the name of the kernel.
    (key1, key2, key3, key4) = jax.random.split(jax.random.PRNGKey(0), 4)
    x_shape = (64, 128)
    param_shape = (x_shape[-1],)
    x = jax.random.normal(key1, x_shape, jnp.bfloat16)
    scale = jax.random.normal(key2, param_shape, jnp.bfloat16)
    offset = jax.random.normal(key3, param_shape, jnp.bfloat16)
    weights = jax.random.normal(key4, (128, 2, 128), jnp.bfloat16)

    def norm_and_glu(x, scale, offset, weights):
      x = norm_op(x, scale, offset)
      return jnp.sum(glu_op(x, weights, activation=jax.nn.swish))

    f_lowered = jax.jit(norm_and_glu).lower(x, scale, offset, weights)
    op_specs = hlo_utils.get_opspecs(f_lowered)

    norm_spec = norm_op.bind(  # pytype: disable=wrong-arg-types
        jax.ShapeDtypeStruct(x_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(param_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(param_shape, jnp.bfloat16),
    )
    glu_spec = glu_op.bind(  # pytype: disable=wrong-arg-types
        jax.ShapeDtypeStruct(x_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(weights.shape, jnp.bfloat16),
        activation=jax.nn.swish,
    )

    self.assertEqual(op_specs[0].op.config, norm_spec.default_config)
    self.assertEqual(op_specs[1].op.config, glu_spec.default_config)
    object.__setattr__(op_specs[0].op, 'config', None)
    object.__setattr__(op_specs[1].op, 'config', None)
    self.assertEqual(op_specs, (norm_spec, glu_spec))

    # Test VJP ops.
    norm_vjp = lambda x, scale, offset: jnp.sum(norm_op(x, scale, offset))
    norm_lowered = jax.jit(jax.value_and_grad(norm_vjp)).lower(x, scale, offset)
    op_specs = hlo_utils.get_opspecs(norm_lowered, include_xla_kernels=False)

    norm_spec = norm_op.bind(  # pytype: disable=wrong-arg-types
        jax.ShapeDtypeStruct(x_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(param_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(param_shape, jnp.bfloat16),
        return_residuals=True,
    )
    norm_vjp_op = cast(pl_norm_vjp.PallasTritonNormalizationVjp, norm_op.vjp)
    norm_vjp_spec = norm_vjp_op.bind(**norm_spec.vjp_arg_spec)
    self.assertEqual(op_specs[0].op.config, norm_spec.default_config)
    self.assertEqual(op_specs[1].op.config, norm_vjp_spec.default_config)
    object.__setattr__(op_specs[0].op, 'config', None)
    object.__setattr__(op_specs[1].op, 'config', None)
    self.assertEqual(op_specs, (norm_spec, norm_vjp_spec))

    # Lastly, test a regular jax function. This should not return any op specs.
    def sin_cos(x):
      return jnp.sin(x), jnp.cos(x)

    sin_cos_lowered = jax.jit(sin_cos).lower(x)
    op_specs = hlo_utils.get_opspecs(sin_cos_lowered)
    self.assertEmpty(op_specs)

  def test_normalization_spec_round_trip(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('This test only runs on GPU.')

    # TODO: Add a test for vmap.
    op = pl_norm.PallasTritonNormalization()
    ba = op.bind(  # pytype: disable=wrong-arg-types
        batching.BatchedShapeDtype((128, 256), jnp.bfloat16, vmap_axes=()),
        batching.BatchedShapeDtype((256,), jnp.bfloat16, vmap_axes=()),
        batching.BatchedShapeDtype((256,), jnp.bfloat16, vmap_axes=()),
    )

    fn, x = benchmarking.standardize_function(op, kwargs=ba.arguments)
    fn_lowered = jax.jit(fn).lower(x)
    (ba2,) = hlo_utils.get_opspecs(fn_lowered, include_xla_kernels=False)
    self.assertEqual(ba.default_config, ba2.op.config)
    object.__setattr__(ba2.op, 'config', None)
    self.assertEqual(ba, ba2)

    expected = fn_lowered.compile()(x)
    fn2, x2 = benchmarking.standardize_function(ba2.op, kwargs=ba2.arguments)
    diff_summary = numerics.array_diff_summary(expected, jax.jit(fn2)(x2))
    self.assertGreater(diff_summary.percent_close * 100, 99.99)

  def test_ragged_dot_spec_round_trip(self):
    if jax.default_backend() != 'gpu':
      self.skipTest('This test only runs on GPU.')

    op = pl_ragged_dot.PallasTritonRaggedDot()
    ba = op.bind(  # pytype: disable=wrong-arg-types
        jax.ShapeDtypeStruct((1024, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((8, 128, 256), jnp.bfloat16),
        group_sizes=[128] * 8,
    )

    fn, x = benchmarking.standardize_function(op, kwargs=ba.arguments)
    fn_lowered = jax.jit(fn).lower(x)
    (ba2,) = hlo_utils.get_opspecs(fn_lowered, include_xla_kernels=False)
    fn2, x2 = benchmarking.standardize_function(ba2.op, kwargs=ba2.arguments)

    arguments = dict(ba.arguments)
    arguments.pop('group_sizes')
    arguments2 = dict(ba2.arguments)
    actual = arguments2.pop('group_sizes')
    self.assertEqual(arguments, arguments2)
    chex.assert_trees_all_equal(actual.value, jnp.array([128] * 8, jnp.int32))
    self.assertEqual(actual.representative_value, (128,) * 8)

    self.assertEqual(ba.default_config, ba2.op.config)
    object.__setattr__(ba.op, 'vjp', None)
    object.__setattr__(ba2.op, 'config', None)
    object.__setattr__(ba2.op, 'vjp', None)
    self.assertEqual(ba.op, ba2.op)

    expected = fn_lowered.compile()(x)
    diff_summary = numerics.array_diff_summary(expected, jax.jit(fn2)(x2))
    self.assertGreater(diff_summary.percent_close * 100, 99.99)

  def test_empty_opspecs_from_triton_kernel(self):
    self.skipTest('This test is not supported on OSS due to CUDA backend issues.')

    if jax.default_backend() != 'gpu':
      self.skipTest('This test only runs on GPU.')

    # A non-Tokamax kernel should not return any op specs from a lowered Jax
    # function.
    @triton.jit
    def non_tokamax_add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        block_size: tl.constexpr,
    ):
      """Adds two vectors."""
      pid = tl.program_id(axis=0)
      block_start = pid * block_size
      offsets = block_start + tl.arange(0, block_size)
      mask = offsets < 8
      x = tl.load(x_ptr + offsets, mask=mask)
      y = tl.load(y_ptr + offsets, mask=mask)
      output = x + y
      tl.store(output_ptr + offsets, output, mask=mask)

    @jax.jit
    def add_jax_triton(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
      out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
      block_size = 8
      grid = (triton.cdiv(x.size, block_size),)
      return jt.triton_call(
          x,
          y,
          kernel=non_tokamax_add_kernel,
          out_shape=out_shape,
          grid=grid,
          block_size=block_size,
          name='add_kernel_jax_triton',
      )

    x = jnp.arange(8)
    y = jnp.arange(8, 16)
    kernels = hlo_utils.get_opspecs(add_jax_triton.lower(x, y))
    self.assertEmpty(kernels)

  @parameterized.parameters(
      ['mosaic', 'triton', 'xla', 'xla_chunked', 'cudnn', None]
  )
  def test_opspec_attention_all_implementations(self, implementation):
    """Tests that attention opspecs are returned for all implementations."""

    # TODO: Remove skipping None once fixed.
    if (
        implementation in ('mosaic', 'triton', 'cudnn', None)
        and jax.default_backend() != 'gpu'
    ):
      self.skipTest('This test only runs on GPU.')

    x = jnp.ones((32, 512, 16, 64), dtype=jnp.bfloat16)
    f = functools.partial(
        attention_api.dot_product_attention, implementation=implementation
    )
    f = jax.jit(f)
    opspecs = hlo_utils.get_opspecs(f.lower(x, x, x))
    self.assertNotEmpty(opspecs)
    opspec = opspecs[0]
    self.assertEqual(
        batching.BatchedShapeDtype(shape=x.shape, dtype=x.dtype, vmap_axes=()),
        opspec.arguments['q'],
    )

  def test_empty_tuple_shape(self):
    """Tests to make sure that empty tuple shapes throw ValueError."""

    hlo_instruction = hlo_pb2.HloInstructionProto()
    hlo_instruction.name = 'tuple'
    hlo_instruction.opcode = 'tuple'
    hlo_instruction.shape.element_type = 'TUPLE'
    self.assertRaises(
        ValueError, hlo_utils._parse_shapes, hlo_instruction.shape
    )

    hlo_instruction.shape.tuple_shapes.append(
        xla_data_pb2.ShapeProto(element_type=xla_data_pb2.F32, dimensions=[1])
    )
    self.assertNotEmpty(hlo_utils._parse_shapes(hlo_instruction.shape))


if __name__ == '__main__':
  absltest.main()
