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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import export
from jax.extend import backend
import jax.numpy as jnp
from tokamax import autotuning
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import shape as shape_lib
from tokamax._src.ops.attention import api


_CUDNN_CUSTOM_CALL_TARGET = 'custom_call_target="__cudnn'


class DotProductAttentionTest(parameterized.TestCase):
  IMPL = None

  # Tests derived from JAX `nn_test`.
  # pylint: disable=invalid-name
  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float16],
      group_num=[1, 2, 4],
      use_vmap=[False, True],
  )
  def test_dot_product_attention(self, dtype, group_num, use_vmap):

    B, S, T, N, H, G = 2, 128, 128, 4, 64, group_num
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    Q = jax.random.normal(keys[0], (B, T, N, H), dtype)
    K = jax.random.normal(keys[1], (B, S, N // G, H), dtype)
    V = jax.random.normal(keys[2], (B, S, N // G, H), dtype)
    grad = jax.random.normal(keys[3], (B, T, N, H), dtype)
    bias, mask = None, None

    sdpa_ref = functools.partial(
        jax.nn.dot_product_attention, implementation='xla'
    )
    sdpa_ans = functools.partial(
        api.dot_product_attention, implementation=self.IMPL
    )
    if use_vmap:
      sdpa_ans = jax.vmap(sdpa_ans, in_axes=(0, 0, 0, None, None), out_axes=0)

    # For testing purposes, we call the non-GQA version without vmap in the
    # reference code
    with shape_lib.upcast_broadcast():
      K_ref = jnp.repeat(K, G, axis=2)
      V_ref = jnp.repeat(V, G, axis=2)

    out_ref, sdpa_vjp_ref = jax.vjp(sdpa_ref, Q, K_ref, V_ref, bias, mask)
    out_ans, sdpa_vjp_ans = jax.vjp(sdpa_ans, Q, K, V, bias, mask)

    dQ_ref, dK_ref, dV_ref = sdpa_vjp_ref(grad)[:3]
    dQ_ans, dK_ans, dV_ans = sdpa_vjp_ans(grad)[:3]
    dK_ref = dK_ref.reshape(B, S, N // G, G, H).sum(axis=3)
    dV_ref = dV_ref.reshape(B, S, N // G, G, H).sum(axis=3)

    chex.assert_trees_all_close(out_ans, out_ref, atol=0.01, rtol=0.01)
    chex.assert_trees_all_close(dQ_ans, dQ_ref, rtol=0.01, atol=0.01)
    chex.assert_trees_all_close(dK_ans, dK_ref, rtol=0.01, atol=0.01)
    chex.assert_trees_all_close(dV_ans, dV_ref, rtol=0.01, atol=0.01)

    args = autotuning.get_bound_args(sdpa_ans, Q, K, V, bias, mask)
    self.assertLen(args, 1)

    if self.IMPL is not None:
      impl = api.IMPLEMENTATIONS[self.IMPL]
      self.assertIsInstance(args[0].op, impl.__class__)

    array_type = (
        batching.BatchedShapeDtype if use_vmap else jax.ShapeDtypeStruct
    )
    args = args[0].arguments
    self.assertIsInstance(args['q'], array_type)
    self.assertIsInstance(args['k'], array_type)
    self.assertIsInstance(args['v'], array_type)

  def test_symbolic_export(self):
    if self.IMPL != 'xla':
      self.skipTest('Symbolic export only supported for XLA.')

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 64), jnp.bfloat16)
    batch, seq_len, num_heads, num_channels = x.shape
    bias = jax.random.normal(
        jax.random.PRNGKey(1), (batch, num_heads, 1, seq_len), jnp.bfloat16
    )

    @jax.jit
    def f(x, bias):
      return api.dot_product_attention(x, x, x, bias=bias, implementation='xla')

    out = f(x, bias)

    b_symbolic, seq_symbolic = export.symbolic_shape('b,s')

    x_shape = jax.ShapeDtypeStruct(
        (b_symbolic, seq_symbolic, num_heads, num_channels), x.dtype
    )
    b_shape = jax.ShapeDtypeStruct(
        (b_symbolic, num_heads, 1, seq_symbolic), bias.dtype
    )

    # Disable jaxtyping to due to
    # https://github.com/patrick-kidger/jaxtyping/issues/338
    with jaxtyping.disable_jaxtyping():
      exported = export.export(f)(x_shape, b_shape)

    serialized = exported.serialize()
    f_roundtrip = export.deserialize(serialized)
    out_roundtrip = jax.jit(f_roundtrip.call)(x, bias)
    chex.assert_trees_all_close(out, out_roundtrip)

  @parameterized.product(
      mask_mode=[
          'bias',
          'causal',
          'padding',
          'custom',
          ('causal', 'padding'),
          ('custom', 'padding'),
          ('bias', 'causal'),
          ('causal', 'sliding_window'),
      ],
  )
  def testDotProductAttentionMask(self, mask_mode):
    # TODO: Fix test for 'xla_chunked' on TPU.
    if jax.default_backend() == 'tpu' and self.IMPL in ('xla_chunked',):
      self.skipTest(f'{self.IMPL} not supported on TPU')
    if isinstance(mask_mode, str):
      mask_mode = (mask_mode,)

    dtype = jnp.bfloat16
    cudnn_bias = self.IMPL == 'cudnn' and 'bias' in mask_mode
    B, S, T, N, H = (1 if cudnn_bias else 2), 256, 256, 4, 64
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    Q = jax.random.normal(keys[0], (B, T, N, H), dtype)
    K = jax.random.normal(keys[1], (B, S, N, H), dtype)
    V = jax.random.normal(keys[2], (B, S, N, H), dtype)
    grad = jax.random.normal(keys[3], (B, T, N, H), dtype)
    bias, mask = None, None
    q_seqlen, kv_seqlen = None, None
    window_size = None

    is_causal = 'causal' in mask_mode
    if 'padding' in mask_mode:
      q_seqlen = jnp.array([T // 2, T // 4], dtype=jnp.int32)
      kv_seqlen = jnp.array([S // 4, S // 2], dtype=jnp.int32)
    if 'custom' in mask_mode:
      # Use a generated causal mask as the custom mask.
      custom_mask = jnp.tril(jnp.ones((T, S), dtype=jnp.bool_))
      mask = custom_mask[None, None, :, :]
    if 'bias' in mask_mode:
      bias = jax.random.normal(keys[4], (1, N, T, S), dtype)
    if 'sliding_window' in mask_mode:
      window_size = (3, 2) if is_causal else (3, 0)

    sdpa_ref = functools.partial(
        jax.nn.dot_product_attention, is_causal=is_causal, implementation='xla'
    )
    sdpa_ans = functools.partial(
        api.dot_product_attention, is_causal=is_causal, implementation=self.IMPL
    )

    args = (Q, K, V, bias, mask)

    # Convert the kargs to positional args for the jax.vjp.
    fn_ref = lambda q, k, v, b, m, qs, kvs: sdpa_ref(
        q,
        k,
        v,
        b,
        m,
        query_seq_lengths=qs,
        key_value_seq_lengths=kvs,
        local_window_size=window_size,
    )

    def fn_ans(q, k, v, b, m, qs, kvs):
      out = sdpa_ans(
          q,
          k,
          v,
          b,
          m,
          query_seq_lengths=qs,
          key_value_seq_lengths=kvs,
          local_window_size=window_size,
      )
      # The JAX implementation zeroes output rows in the padding region.
      if qs is not None:
        mask = jnp.arange(0, T)[None, :] < q_seqlen[:, None]
        out *= mask[:, :, None, None]
      return out

    out_ref, sdpa_vjp_ref = jax.vjp(fn_ref, *args, q_seqlen, kv_seqlen)
    out_ans, sdpa_vjp_ans = jax.vjp(fn_ans, *args, q_seqlen, kv_seqlen)
    dQ_ref, dK_ref, dV_ref, dbias_ref = sdpa_vjp_ref(grad)[:4]
    dQ_ans, dK_ans, dV_ans, dbias_ans = sdpa_vjp_ans(grad)[:4]

    chex.assert_trees_all_close(out_ans, out_ref, atol=0.01, rtol=0.01)
    chex.assert_trees_all_close(dQ_ans, dQ_ref, rtol=0.02, atol=0.02)
    chex.assert_trees_all_close(dK_ans, dK_ref, rtol=0.02, atol=0.02)
    chex.assert_trees_all_close(dV_ans, dV_ref, rtol=0.01, atol=0.01)
    chex.assert_trees_all_close(dbias_ans, dbias_ref, rtol=0.05, atol=0.05)

  @parameterized.product(batch_size=[1, 16], use_vmap=[False, True])
  def test_dot_product_attention_bias_gradient(self, batch_size, use_vmap):
    # TODO: Fix test for 'xla_chunked' on TPU.
    if jax.default_backend() == 'tpu' and self.IMPL in ('xla_chunked',):
      self.skipTest(f'{self.IMPL} not supported on TPU')

    dtype = jnp.bfloat16
    B, S, N, H = batch_size, 128, 4, 64
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.normal(keys[0], (B, S, N, H), dtype)
    bias = jax.random.normal(keys[1], (B, N, S, S), dtype=dtype)
    mask = jnp.ones((1, 1, S), dtype=jnp.bool_)

    def attention(impl, x, bias, mask):
      return impl(
          query=x,
          key=x,
          value=x,
          bias=bias,
          mask=mask,
          is_causal=False,
      )

    attn_ref = functools.partial(
        attention,
        functools.partial(jax.nn.dot_product_attention, implementation='xla'),
    )
    attn_ans = functools.partial(
        attention,
        functools.partial(api.dot_product_attention, implementation=self.IMPL),
    )
    if use_vmap:
      attn_batched_ref = jax.vmap(attn_ref, in_axes=(0, 0, None))
      attn_batched_ans = jax.vmap(attn_ans, in_axes=(0, 0, None))
    else:
      attn_batched_ref = attn_ref
      attn_batched_ans = attn_ans

    fwd_ref = jax.jit(attn_batched_ref)
    fwd_ans = jax.jit(attn_batched_ans)
    y_ref = fwd_ref(x, bias, mask)
    y_ans = fwd_ans(x, bias, mask)
    chex.assert_trees_all_close(y_ans, y_ref, atol=0.01, rtol=0.01)

    @jax.jit
    def bwd_ref(x, bias, mask):
      _, f_vjp = jax.vjp(attn_ref, x, bias, mask)
      return f_vjp(x)

    @jax.jit
    def bwd_ans(x, bias, mask):
      _, f_vjp = jax.vjp(attn_ans, x, bias, mask)
      return f_vjp(x)

    _, dbias_ref, _ = bwd_ref(x, bias, mask)
    _, dbias_ans, _ = bwd_ans(x, bias, mask)
    chex.assert_trees_all_close(dbias_ans, dbias_ref, rtol=0.25, atol=0.25)


# pylint: enable=invalid-name


class DotProductAttentionMosaicTest(DotProductAttentionTest):
  IMPL = 'mosaic'

  def setUp(self):
    super().setUp()
    if not gpu_utils.has_mosaic_gpu_support() or gpu_utils.is_sm100():
      self.skipTest(
          'Skip test. Mosaic implementation is not supported on this platform.'
      )


class DotProductAttentionTritonTest(DotProductAttentionTest):
  IMPL = 'triton'

  def setUp(self):
    super().setUp()
    if not gpu_utils.has_triton_support():
      self.skipTest('Triton not supported on this platform.')


class DotProductAttentionCudnnTest(DotProductAttentionTest):
  IMPL = 'cudnn'

  def setUp(self):
    super().setUp()
    if jax.default_backend() != 'gpu':
      self.skipTest(f'cuDNN only supported on GPU, not {jax.default_backend()}')

  def test_impl_in_hlo(self):
    fn = functools.partial(api.dot_product_attention, implementation=self.IMPL)
    x = jnp.empty((2, 256, 4, 64), dtype=jnp.bfloat16)
    lowered = jax.jit(fn).lower(x, x, x)
    hlo_text = lowered.compiler_ir(dialect='hlo').as_hlo_text()
    self.assertIn(_CUDNN_CUSTOM_CALL_TARGET, hlo_text)


class DotProductAttentionXlaTest(DotProductAttentionTest):
  IMPL = 'xla'

  def test_precision(self):
    if jax.default_backend() == 'cpu':
      self.skipTest('XLA:CPU does not properly respect precision.')

    x = jax.random.normal(jax.random.PRNGKey(0), (1, 16, 2, 16), jnp.float32)

    @functools.partial(jax.jit, static_argnames=['precision'])
    def f(x, precision):
      return api.dot_product_attention(
          x, x, x, implementation='xla', precision=precision
      )

    out_1 = f(x, jax.lax.Precision.HIGHEST)
    out_2 = f(x, jax.lax.DotAlgorithmPreset.BF16_BF16_F32)
    equal = bool(jnp.array_equal(out_1, out_2))
    self.assertFalse(equal)


class DotProductAttentionXlaChunkedTest(DotProductAttentionTest):
  IMPL = 'xla_chunked'


if __name__ == '__main__':
  absltest.main()
