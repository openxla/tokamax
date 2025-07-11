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
import jax.numpy as jnp
from tokamax._src.ops.attention import api


# TODO: Check that the requested implementation is used.
class DotProductAttentionTest(parameterized.TestCase):
  IMPL = None
  SUPPORTS_VMAP = True

  # Tests derived from JAX `nn_test`.
  # pylint: disable=invalid-name
  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float16],
      group_num=[1, 2, 4],
      use_vmap=[False, True],
  )
  def testDotProductAttention(self, dtype, group_num, use_vmap):
    if use_vmap and not self.SUPPORTS_VMAP:
      self.skipTest('vmap not supported')

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
    if isinstance(mask_mode, str):
      mask_mode = (mask_mode,)

    dtype = jnp.bfloat16
    B, S, T, N, H = 2, 128, 128, 4, 64
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
      # Tokamax calculates dbias in f32, so use an f32 bias to reduce ref error.
      bias = jax.random.normal(keys[4], (1, N, T, S), jnp.float32)
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
  def testDotProductAttentionBiasGradient(self, batch_size, use_vmap):
    if use_vmap and not self.SUPPORTS_VMAP:
      self.skipTest('vmap not supported')

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
  SUPPORTS_VMAP = False

  def testDotProductAttentionMask6(self):
    self.skipTest('"mosaic" implementation does not support bias and vmap')

  def testDotProductAttentionBiasGradient1(self):
    self.skipTest('"mosaic" implementation does not support bias and vmap')

  def testDotProductAttentionBiasGradient3(self):
    self.skipTest('"mosaic" implementation does not support bias and vmap')


class DotProductAttentionTritonTest(DotProductAttentionTest):
  IMPL = 'triton'


class DotProductAttentionCudnnTest(DotProductAttentionTest):
  IMPL = 'cudnn'

  def testMemoryScaling(self):
    self.skipTest('Memory scaling guarantees not made for this implementation')


class DotProductAttentionXlaTest(DotProductAttentionTest):
  IMPL = 'xla'


class DotProductAttentionXlaChunkedTest(DotProductAttentionTest):
  IMPL = 'xla_chunked'


if __name__ == '__main__':
  absltest.main()
