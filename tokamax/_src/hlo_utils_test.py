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
import collections
import functools
import json
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import jax_triton as jt
from tokamax._src import batching
from tokamax._src import hlo_utils
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import numerics
from tokamax._src.ops import op
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.normalization import pallas_triton as pl_norm
from tokamax._src.ops.normalization import pallas_triton_vjp as pl_norm_vjp
from tokamax._src.ops.ragged_dot import pallas_triton as pl_ragged_dot
from tokamax.google.ops.gated_linear_unit import mosaic_gpu as mosaic_glu
import triton
import triton.language as tl


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

  def test_gated_linear_unit_mosaic_gpu(
      self,
  ):

    if not mgpu_lib.has_mosaic_gpu_support():
      self.skipTest("Can't run mosaic gpu. Maybe you are not on h100")

    dtype = jnp.bfloat16
    config = dict(input_shape=(64, 128), out_size=128)

    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0), 2)
    input_shape = config['input_shape']
    n = config['out_size']
    (_, k) = input_shape
    weight_shape = (k, 2, n)

    x = jax.random.normal(key=rng0, shape=input_shape, dtype=dtype)
    w = jax.random.normal(key=rng1, shape=weight_shape, dtype=dtype)

    @jax.jit
    def swiglu(x, w):
      return mosaic_glu.MosaicGpuGatedLinearUnit()(
          x=x, weights=w, activation=jax.nn.swish
      )

    kernels = hlo_utils.get_kernel_info(
        swiglu.lower(x, w), include_xla_kernels=False
    )

    self.assertLen(kernels, 1)
    kernel = kernels[0]
    self.assertIsInstance(kernel, hlo_utils.MosaicGpuKernelInfo)

    self.assertEqual(
        kernel.inputs,
        (
            jax.ShapeDtypeStruct(shape=input_shape, dtype=dtype),
            jax.ShapeDtypeStruct(shape=weight_shape, dtype=dtype),
        ),
    )

    # Note: currently a bug in swiglu that produces two outputs. The second
    # one is spurious and will be removed.
    self.assertEqual(
        kernel.outputs, (jax.ShapeDtypeStruct(shape=(64, 128), dtype=dtype),)
    )

  def test_jax_triton_simple(self):

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

    # TODO: Find a better place to put the kernel spec rather than
    #  embedded in the name of the op.

    # Create a string of Tokamax ops in Jax, lower it to HLO, and extract the
    # kernel spec from the name of the kernel.
    (key1, key2, key3, key4) = jax.random.split(jax.random.PRNGKey(0), 4)
    pt_normalization = functools.partial(
        pl_norm.PallasTritonNormalization(), axis=(-1)
    )
    x_shape = (64, 128)
    param_shape = (x_shape[-1],)
    x = jax.random.normal(key=key1, shape=x_shape, dtype=jnp.bfloat16)
    scale = jax.random.normal(key=key2, shape=param_shape, dtype=jnp.bfloat16)
    offset = jax.random.normal(key=key3, shape=param_shape, dtype=jnp.bfloat16)
    weights = jax.random.normal(
        key=key4, shape=(128, 2, 128), dtype=jnp.bfloat16
    )

    def norm_and_glu(x, scale, offset):
      normalized_x = pt_normalization(x, scale, offset)
      glu_x = mosaic_glu.MosaicGpuGatedLinearUnit()(
          x=normalized_x, weights=weights, activation=jax.nn.swish
      )
      return jnp.sum(glu_x)

    f_lowered = jax.jit(norm_and_glu).lower(x, scale, offset)

    op_specs = hlo_utils.get_opspecs(f_lowered)

    # Golden BatchedShapeDtype values.
    bs_128_bf16 = batching.BatchedShapeDtype(
        shape=param_shape, dtype=jnp.bfloat16, vmap_axes=()
    )
    bs_128x2x128_bf16 = batching.BatchedShapeDtype(
        shape=(128, 2, 128), dtype=jnp.bfloat16, vmap_axes=()
    )
    bs_64x1_fp32 = batching.BatchedShapeDtype(
        shape=(64, 1), dtype=jnp.float32, vmap_axes=()
    )
    bs_64x128_bf16 = batching.BatchedShapeDtype(
        shape=(64, 128), dtype=jnp.bfloat16, vmap_axes=()
    )
    self.assertLen(op_specs, 2)
    self.assertIsInstance(op_specs[0].op, pl_norm.PallasTritonNormalization)
    self.assertEqual(op_specs[0].arguments['x'], bs_64x128_bf16)
    self.assertEqual(op_specs[0].arguments['scale'], bs_128_bf16)
    self.assertEqual(op_specs[0].arguments['offset'], bs_128_bf16)
    self.assertEqual(op_specs[0].arguments['axis'], -1)

    self.assertIsInstance(op_specs[1].op, mosaic_glu.MosaicGpuGatedLinearUnit)
    self.assertEqual(op_specs[1].arguments['x'], bs_64x128_bf16)
    self.assertEqual(op_specs[1].arguments['weights'], bs_128x2x128_bf16)
    self.assertIsNone(op_specs[1].arguments['precision'])
    self.assertFalse(op_specs[1].arguments['return_residuals'])

    # Test VJP ops.
    def norm_vjp(x, scale, offset):
      norm_x = pt_normalization(x, scale, offset)
      return jnp.sum(norm_x)

    norm_lowered = jax.jit(jax.value_and_grad(norm_vjp)).lower(x, scale, offset)
    op_specs = hlo_utils.get_opspecs(norm_lowered, include_xla_kernels=False)
    self.assertLen(op_specs, 2)
    self.assertIsInstance(op_specs[0].op, pl_norm.PallasTritonNormalization)
    self.assertIsInstance(
        op_specs[1].op, pl_norm_vjp.PallasTritonNormalizationVjp
    )
    self.assertEqual(
        op_specs[1].arguments['residuals'], (bs_64x1_fp32, bs_64x1_fp32)
    )
    self.assertEqual(op_specs[1].arguments['out'], bs_64x128_bf16)
    self.assertEqual(op_specs[1].arguments['dout'], bs_64x128_bf16)

    # Lastly, test a regular jax function. This should not return any op specs.
    def sin_cos(x):
      return jnp.sin(x), jnp.cos(x)

    sin_cos_lowered = jax.jit(sin_cos).lower(x)
    op_specs = hlo_utils.get_opspecs(sin_cos_lowered)
    self.assertEmpty(op_specs)

  def test_opspecs_round_trip(self):

    if jax.default_backend() != 'gpu':
      self.skipTest('This test only runs on GPU.')

    # TODO: Add a test for vmap.

    normalization_bound_args = op.BoundArguments(
        op=pl_norm.PallasTritonNormalization(),
        arguments=collections.OrderedDict({
            'x': batching.BatchedShapeDtype(
                (128, 256), jnp.bfloat16, vmap_axes=()
            ),
            'scale': batching.BatchedShapeDtype(
                (256,), jnp.bfloat16, vmap_axes=()
            ),
            'offset': batching.BatchedShapeDtype(
                (256,), jnp.bfloat16, vmap_axes=()
            ),
        }),
    )

    def test_fn(kwargs):
      return normalization_bound_args.op(**kwargs)

    normalization_concrete_args = numerics.random_initialize(
        normalization_bound_args.arguments
    )

    normalization_fn_lowered = jax.jit(test_fn).lower(
        normalization_concrete_args
    )
    output = normalization_fn_lowered.compile()(normalization_concrete_args)
    op_specs = hlo_utils.get_opspecs(
        normalization_fn_lowered, include_xla_kernels=False
    )
    self.assertLen(op_specs, 1)
    self.assertIsInstance(op_specs[0].op, pl_norm.PallasTritonNormalization)
    self.assertEqual(
        op_specs[0].arguments['x'], normalization_bound_args.arguments['x']
    )
    self.assertEqual(
        op_specs[0].arguments['scale'],
        normalization_bound_args.arguments['scale'],
    )
    self.assertEqual(
        op_specs[0].arguments['offset'],
        normalization_bound_args.arguments['offset'],
    )

    # Run the op_spec again and get the output.
    def second_normalization_fn(kwargs):
      return op_specs[0].op(**kwargs)

    output_2 = (
        jax.jit(second_normalization_fn)
        .lower(normalization_concrete_args)
        .compile()(normalization_concrete_args)
    )

    # This should match the output of the original op.
    diff_summary = numerics.array_diff_summary(output, output_2)
    self.assertGreater(diff_summary.percent_close * 100, 99.99)

    partial_const = functools.partial(
        numerics.const_initializer, jnp.array([128] * 8, jnp.uint32)
    )
    # Do a second run with ragged dot
    initializer = numerics.InitializableArray(
        value=batching.BatchedShapeDtype((8,), jnp.uint32, vmap_axes=()),
        initializer=partial_const,
    )

    ragged_dot_bound_args = op.BoundArguments(
        op=pl_ragged_dot.PallasTritonRaggedDot(),
        arguments=collections.OrderedDict({
            'lhs': batching.BatchedShapeDtype(
                (1024, 128), jnp.bfloat16, vmap_axes=()
            ),
            'rhs': batching.BatchedShapeDtype(
                (8, 128, 256), jnp.bfloat16, vmap_axes=()
            ),
            'group_sizes': initializer,
        }),
    )

    def test_fn_ragged_dot(kwargs):
      return ragged_dot_bound_args.op(**kwargs)

    ragged_dot_concrete_args = numerics.random_initialize(
        ragged_dot_bound_args.arguments
    )
    test_ragged_dot_lowered = jax.jit(test_fn_ragged_dot).lower(
        ragged_dot_concrete_args
    )
    ragged_dot_output = test_ragged_dot_lowered.compile()(
        ragged_dot_concrete_args
    )
    op_specs = hlo_utils.get_opspecs(
        test_ragged_dot_lowered, include_xla_kernels=False
    )
    self.assertLen(op_specs, 1)
    self.assertIsInstance(op_specs[0].op, pl_ragged_dot.PallasTritonRaggedDot)
    self.assertEqual(
        op_specs[0].arguments['lhs'], ragged_dot_bound_args.arguments['lhs']
    )
    self.assertEqual(
        op_specs[0].arguments['rhs'], ragged_dot_bound_args.arguments['rhs']
    )
    self.assertEqual(
        op_specs[0].arguments['group_sizes'].value,
        ragged_dot_bound_args.arguments['group_sizes'].value,
    )

    def second_ragged_dot_fn(kwargs):
      return op_specs[0].op(**kwargs)

    ragged_dot_output_2 = (
        jax.jit(second_ragged_dot_fn)
        .lower(ragged_dot_concrete_args)
        .compile()(ragged_dot_concrete_args)
    )

    diff_summary = numerics.array_diff_summary(
        ragged_dot_output, ragged_dot_output_2
    )
    self.assertGreater(diff_summary.percent_close * 100, 99.99)

  def test_empty_opspecs_from_triton_kernel(self):

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
        implementation in ('mosaic', 'triton', None)
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


if __name__ == '__main__':
  absltest.main()
