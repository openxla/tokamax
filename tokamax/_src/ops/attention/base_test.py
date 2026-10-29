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

import contextlib

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import linen as nn
import jax
from jax import export
import jax.numpy as jnp
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import test_base


class MaskTest(absltest.TestCase):

  def test_as_array(self):
    mask = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(1, 2, 3))
    chex.assert_trees_all_equal(base.Mask(mask).as_array(2, 3), mask)

  def test_as_array_symbolic_args_specs(self):
    input_specs = export.symbolic_args_specs(
        {"mask": jax.random.bernoulli(jax.random.PRNGKey(0), shape=(1, 2, 3))},
        {"mask": "(b, q_len, k_len)"},
    )

    def f(mask):
      return base.Mask(mask).as_array(mask.shape[-2], mask.shape[-1])

    _ = jax.eval_shape(f, **input_specs)

  def test_take(self):
    mask_array = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(32, 32))
    mask, is_causal = base.Mask(mask_array).take("is_causal")
    chex.assert_trees_all_equal(mask.as_array(32, 32), mask_array[None, ...])
    self.assertFalse(is_causal)

    causal_mask = base.Mask(mask_array, is_causal=True)
    causal_mask, is_causal = causal_mask.take("is_causal")
    actual = causal_mask.as_array(32, 32)
    chex.assert_trees_all_equal(actual, mask_array[None, ...])
    self.assertTrue(is_causal)

    causal_mask = base.Mask(mask_array, causal_mask=base.CausalMask(42))
    causal_mask, causal = causal_mask.take("causal_mask")
    actual = causal_mask.as_array(32, 32)
    chex.assert_trees_all_equal(actual, mask_array[None, ...])
    self.assertEqual(causal, base.CausalMask(42))

  def test_k_range(self):
    k_start = jnp.array([0, 0, 1, 2])
    k_end = jnp.array([1, 2, 3, 4])
    actual = base.Mask(k_start=k_start, k_end=k_end).as_array(4, 3)
    expected = jnp.array(
        [[[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

  def test_q_range(self):
    q_end = jnp.array([1, 3, 0])
    actual = base.Mask(q_end=q_end).as_array(4, 3)
    expected = jnp.array(
        [[[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

  def test_causal(self):
    actual = base.Mask(is_causal=True).as_array(4, 4)
    expected = jnp.array(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

    actual = base.Mask(
        causal_mask=base.CausalMask(offset=1, window_size=2)
    ).as_array(4, 4)
    expected = jnp.array(
        [[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

    actual = base.Mask(
        causal_mask=base.CausalMask(
            offset=jnp.array(-1), window_size=jnp.array(2)
        )
    ).as_array(4, 4)
    expected = jnp.array(
        [[[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

    with self.assertRaises(ValueError):
      base.Mask(causal_mask=base.CausalMask(), is_causal=True)

  def test_intersection(self):
    mask0 = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(32, 32))
    mask1 = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(32, 32))
    actual = (base.Mask(mask0) & base.Mask(mask1)).as_array(32, 32)
    chex.assert_trees_all_equal(actual, (mask0 & mask1)[None, ...])

    q_start0 = jnp.array([0, 1, 2, 3])
    q_start1 = jnp.array([2, 1, 0, 4])
    mask = base.Mask(q_start=q_start0) & base.Mask(q_start=q_start1)
    chex.assert_trees_all_equal(mask.q_start, jnp.array([2, 1, 2, 4]))

    q_end0 = jnp.array([0, 1, 2, 3])
    q_end1 = jnp.array([2, 1, 0, 4])
    mask = base.Mask(q_end=q_end0) & base.Mask(q_end=q_end1)
    chex.assert_trees_all_equal(mask.q_end, jnp.array([0, 1, 0, 3]))

    k_start0 = jnp.array([0, 1, 2, 3])
    k_start1 = jnp.array([2, 1, 0, 4])
    mask = base.Mask(k_start=k_start0) & base.Mask(k_start=k_start1)
    chex.assert_trees_all_equal(mask.k_start, jnp.array([2, 1, 2, 4]))

    k_end0 = jnp.array([0, 1, 2, 3])
    k_end1 = jnp.array([2, 1, 0, 4])
    mask = base.Mask(k_end=k_end0) & base.Mask(k_end=k_end1)
    chex.assert_trees_all_equal(mask.k_end, jnp.array([0, 1, 0, 3]))

    for a in [True, False]:
      for b in [True, False]:
        mask = base.Mask(is_causal=a) & base.Mask(is_causal=b)
        self.assertEqual(mask.is_causal, a or b)

    causal0 = base.CausalMask(offset=2, window_size=5)
    causal1 = base.CausalMask(offset=0, window_size=4)
    combined = base.Mask(causal_mask=causal0) & causal1
    self.assertEqual(
        combined.causal_mask, base.CausalMask(offset=0, window_size=3)
    )
    arr0 = base.Mask(causal_mask=causal0).as_array(8, 8)
    arr1 = base.Mask(causal_mask=causal1).as_array(8, 8)
    assert arr0 is not None and arr1 is not None
    chex.assert_trees_all_equal(combined.as_array(8, 8), arr0 & arr1)

    two_sequence_mask = jnp.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool
    )
    actual = (base.Mask(is_causal=True) & two_sequence_mask).as_array(4, 4)
    expected = jnp.array(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

  def test_mask_as_array_gqa_per_kv_head(self):
    q = jnp.zeros((2, 8, 8, 16))
    k = jnp.zeros((2, 8, 2, 16))
    k_start = jnp.array([[0], [2]])
    k_end = jnp.array([[6], [8]])
    mask = base.Mask(k_start=k_start, k_end=k_end)
    arr = base.mask_as_array(mask, q, k, None, None)
    assert arr is not None
    self.assertEqual(arr.shape, (2, 1, 8))
    arr_rep = base.repeat_kv_heads(arr, -3, 8, 2)
    self.assertEqual(arr_rep.shape, (8, 1, 8))

    q_start = jnp.arange(8)[:, None]
    combined = base.Mask(q_start=q_start, k_start=k_start, k_end=k_end)
    combined_arr = base.mask_as_array(combined, q, k, None, None)
    expected = base.Mask(
        q_start=q_start,
        k_start=jnp.repeat(k_start, 4, axis=-2),
        k_end=jnp.repeat(k_end, 4, axis=-2),
    ).as_array(8, 8)
    chex.assert_trees_all_equal(combined_arr, expected)

  def test_find_low_precision_reductions_detector(self):
    x = jax.ShapeDtypeStruct((4, 8), jnp.bfloat16)
    b = jax.ShapeDtypeStruct((8,), jnp.bfloat16)

    def bad_bwd(x, b):
      _, vjp_fn = jax.vjp(lambda x, b: x + b[None, :], x, b)
      return vjp_fn(x)

    allow_f16_ctx = (
        jax.allow_f16_reductions(True)
        if hasattr(jax, "allow_f16_reductions")
        else contextlib.nullcontext()
    )
    with allow_f16_ctx:
      bad_hlo = jax.jit(bad_bwd).lower(x, b).as_text("hlo")
    self.assertNotEmpty(test_base.find_low_precision_reductions(bad_hlo))

    max_hlo = jax.jit(lambda x: jnp.max(x, axis=0)).lower(x).as_text("hlo")
    self.assertEmpty(test_base.find_low_precision_reductions(max_hlo))

    while_and_variadic_hlo = """
HloModule test
%sum_bf16 (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  ROOT %add = bf16[] add(bf16[] %lhs, bf16[] %rhs)
}
%sum_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}
%variadic_red (a0: f32[], b0: bf16[], a1: f32[], b1: bf16[]) -> (f32[], bf16[]) {
  %m = f32[] maximum(f32[] %a0, f32[] %a1)
  %s = bf16[] add(bf16[] %b0, bf16[] %b1)
  ROOT %t = (f32[], bf16[]) tuple(%m, %s)
}
%while_body (arg_tuple: (s32[], bf16[4,8], bf16[1,8], f32[4,8])) -> (s32[]) {
  %idx = s32[] get-tuple-element(%arg_tuple), index=0
  %x_nontrivial = bf16[4,8]{1,0} get-tuple-element(%arg_tuple), index=1
  %x_trivial = bf16[1,8]{1,0} get-tuple-element(%arg_tuple), index=2
  %y_f32 = f32[4,8]{1,0} get-tuple-element(%arg_tuple), index=3
  %z = bf16[] constant(0)
  %zf = f32[] constant(0)
  %r_bad = bf16[8]{0} reduce(%x_nontrivial, %z), dimensions={0}, to_apply=%sum_bf16
  %r_ok = bf16[8]{0} reduce(%x_trivial, %z), dimensions={0}, to_apply=%sum_bf16
  %r_tpu_f32_accum = bf16[8]{0} reduce(%y_f32, %zf), dimensions={0}, to_apply=%sum_f32
  %r_f32_out_bf16_accum = f32[8]{0} reduce(%x_nontrivial, %z), dimensions={0}, to_apply=%sum_bf16
  %r_var = (f32[8]{0}, bf16[8]{0}) reduce(%y_f32, %x_nontrivial, %zf, %z), dimensions={0}, to_apply=%variadic_red
  ROOT %out = s32[] add(%idx, %idx)
}
"""
    detected = test_base.find_low_precision_reductions(while_and_variadic_hlo)
    self.assertLen(detected, 3)
    self.assertTrue(any("%r_bad =" in line for line in detected))
    self.assertTrue(any("%r_f32_out_bf16_accum =" in line for line in detected))
    self.assertTrue(any("%r_var =" in line for line in detected))
    self.assertFalse(any("%r_tpu_f32_accum =" in line for line in detected))

  def test_bind_head_validation(self):
    op = base.DotProductAttention()
    q = jax.ShapeDtypeStruct((2, 16, 8, 32), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((2, 16, 2, 32), jnp.bfloat16)
    v_mismatch = jax.ShapeDtypeStruct((2, 16, 4, 32), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "same number of heads"):
      op.bind(q, k, v_mismatch)
    k_zero = jax.ShapeDtypeStruct((2, 16, 0, 32), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "divisible"):
      op.bind(q, k_zero, k_zero)
    q_rank2 = jax.ShapeDtypeStruct((16, 32), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "at least 3 dimensions"):
      op.bind(q_rank2, k, k)


class DotProductAttentionTest(test_base.AttentionTestBase):

  def __init__(self, *args, vjp=None):
    super().__init__(*args, attention_fn=base.DotProductAttention(vjp=vjp))

  def _run_test(self, q_shape, *args, dtype=jnp.float32, **kwargs):
    if q_shape[1] >= 16384:
      self.skipTest("XLA seems to fail for so long sequences (b/384038935)")

    impl_kwargs = kwargs.pop("impl_kwargs", {})
    if dtype == jnp.float32:
      impl_kwargs["precision"] = jax.lax.Precision.HIGHEST
    super()._run_test(
        q_shape, *args, dtype=dtype, impl_kwargs=impl_kwargs, **kwargs
    )

  @parameterized.product(
      num_heads_kv=(1, 2),
      dtype=(jnp.float32, jnp.bfloat16),
  )
  def test_mqa_and_gqa_with_masks_and_bias(self, num_heads_kv, dtype):
    num_heads_q = 8
    repeats = num_heads_q // num_heads_kv
    seq_q, seq_k = 64, 96
    k_start = jnp.arange(num_heads_kv, dtype=jnp.int32)[:, None] * 4
    k_end = jnp.full((num_heads_kv, 1), seq_k - 4, dtype=jnp.int32)
    q_start = jnp.arange(num_heads_q, dtype=jnp.int32)[:, None]
    mask = base.Mask(
        is_causal=True, q_start=q_start, k_start=k_start, k_end=k_end
    )
    ref_mask = base.Mask(
        is_causal=True,
        q_start=q_start,
        k_start=jnp.repeat(k_start, repeats, axis=-2),
        k_end=jnp.repeat(k_end, repeats, axis=-2),
    ).as_array(seq_q, seq_k)

    def ref_impl(q, k, v, **kwargs):
      k = jnp.repeat(k, repeats, axis=-2)
      v = jnp.repeat(v, repeats, axis=-2)
      return nn.dot_product_attention(q, k, v, **kwargs)

    atol = {jnp.float32: 2e-6, jnp.bfloat16: 1.5e-2}[dtype]
    atol_grads = {jnp.float32: 1e-5, jnp.bfloat16: 4e-2}[dtype]
    self._run_test(
        q_shape=(2, seq_q, num_heads_q, 32),
        kv_shape=(2, seq_k, num_heads_kv, 32),
        bias_shape=(1, num_heads_q, 1, seq_k),
        dtype=dtype,
        impl_kwargs=dict(mask=mask),
        ref_kwargs=dict(mask=ref_mask),
        ref_impl=ref_impl,
        atol=atol,
        atol_grads=atol_grads,
    )

  @parameterized.parameters(1, 2)
  def test_mqa_gqa_no_logits_transpose_or_kv_repeat_in_hlo(self, num_heads_kv):
    q = jax.ShapeDtypeStruct((2, 64, 8, 32), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((2, 96, num_heads_kv, 32), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((2, 96, num_heads_kv, 32), jnp.bfloat16)
    hlo_text = jax.jit(self._attention_fn).lower(q, k, v).as_text("hlo")
    for line in hlo_text.splitlines():
      if "transpose(" in line:
        has_q_dim = any(d in line for d in ("64", "256", "512"))
        self.assertFalse(
            has_q_dim and "96" in line,
            msg=f"Unexpected quadratic logits/weights transpose: {line}",
        )
      self.assertNotIn("bf16[2,96,8,32]", line)

  @parameterized.product(
      num_heads_kv=(1, 2, 8),
      logits_dtype=(base.AUTO, jnp.bfloat16),
      normalize_output=(True, False),
  )
  def test_no_bf16_reductions_in_bwd_hlo(
      self, num_heads_kv, logits_dtype, normalize_output
  ):
    bias_shape = {
        1: (1, 1, 1, 48),
        2: (1, 8, 32, 1),
        8: (1, 1, 48),
    }[num_heads_kv]
    q = jax.ShapeDtypeStruct((2, 32, 8, 32), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((2, 48, num_heads_kv, 32), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((2, 48, num_heads_kv, 32), jnp.bfloat16)
    bias = jax.ShapeDtypeStruct(bias_shape, jnp.bfloat16)
    dout = jax.ShapeDtypeStruct((2, 32, 8, 32), jnp.bfloat16)

    def bwd(q, k, v, bias, dout):
      fn = lambda q, k, v, b: self._attention_fn(
          q,
          k,
          v,
          bias=b,
          is_causal=True,
          logits_dtype=logits_dtype,
          normalize_output=normalize_output,
      )
      _, vjp_fn = jax.vjp(fn, q, k, v, bias)
      return vjp_fn(dout)

    lowered = jax.jit(bwd).lower(q, k, v, bias, dout)
    self.assertEmpty(
        test_base.find_low_precision_reductions(lowered.as_text("hlo"))
    )
    # XLA:TPU's BFloat16Propagation demotes f32 reductions between bf16 converts
    # because TPU hardware accumulates bf16 reductions in f32.
    if jax.default_backend() != "tpu":
      compiled_hlo = lowered.compile().as_text()
      if compiled_hlo is not None:
        self.assertEmpty(test_base.find_low_precision_reductions(compiled_hlo))

  @parameterized.parameters(
      ((0, None, None, None),),
      ((None, 0, 0, None),),
      ((None, None, None, 0),),
  )
  def test_no_bf16_reductions_in_vmap_bwd_hlo(self, in_axes):
    vmap_size = 3
    q_shape = (2, 32, 8, 32)
    kv_shape = (2, 32, 2, 32)
    bias_shape = (1, 8, 1, 32)
    q = jax.ShapeDtypeStruct(
        (vmap_size, *q_shape) if in_axes[0] == 0 else q_shape, jnp.bfloat16
    )
    k = jax.ShapeDtypeStruct(
        (vmap_size, *kv_shape) if in_axes[1] == 0 else kv_shape, jnp.bfloat16
    )
    v = jax.ShapeDtypeStruct(
        (vmap_size, *kv_shape) if in_axes[2] == 0 else kv_shape, jnp.bfloat16
    )
    bias = jax.ShapeDtypeStruct(
        (vmap_size, *bias_shape) if in_axes[3] == 0 else bias_shape,
        jnp.bfloat16,
    )
    dout = jax.ShapeDtypeStruct((vmap_size, *q_shape), jnp.bfloat16)

    def bwd(q, k, v, bias, dout):
      fn = jax.vmap(
          lambda q, k, v, b: self._attention_fn(
              q, k, v, bias=b, is_causal=True, logits_dtype=jnp.bfloat16
          ),
          in_axes=in_axes,
      )
      _, vjp_fn = jax.vjp(fn, q, k, v, bias)
      return vjp_fn(dout)

    lowered = jax.jit(bwd).lower(q, k, v, bias, dout)
    self.assertEmpty(
        test_base.find_low_precision_reductions(lowered.as_text("hlo"))
    )
    if jax.default_backend() != "tpu":
      compiled_hlo = lowered.compile().as_text()
      if compiled_hlo is not None:
        self.assertEmpty(test_base.find_low_precision_reductions(compiled_hlo))


class DotProductAttentionWithExplicitVjpTest(DotProductAttentionTest):

  def __init__(self, *args):
    super().__init__(*args, vjp=base.DotProductAttentionVjp())


if __name__ == "__main__":
  absltest.main()
