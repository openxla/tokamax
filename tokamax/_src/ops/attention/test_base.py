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
"""Tests for attention."""

import contextlib
import functools
import itertools
import sys
from unittest import mock

import pytest

from absl.testing import parameterized
import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import qwix
from tokamax._src import numerics
from tokamax._src import quantization
from tokamax._src import test_utils
from tokamax._src.ops.attention import arg_specs
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import xla_chunked
ARG_SPECS = arg_specs.ARG_SPECS


def _create_inputs(
    *,
    q_shape,
    kv_shape=None,
    bias_shape=None,
    mask_shape=None,
    dtype=jnp.float32,
):
  """Creates random inputs for the test."""
  if kv_shape is None:
    kv_shape = q_shape

  rngs = list(jax.random.split(jax.random.PRNGKey(0), 6))
  q = jax.random.normal(rngs.pop(), q_shape, dtype=dtype)
  k = jax.random.normal(rngs.pop(), kv_shape, dtype=dtype)
  v = jax.random.normal(rngs.pop(), kv_shape, dtype=dtype)

  if bias_shape is None:
    bias = None
  else:
    bias = jax.random.normal(rngs.pop(), bias_shape, dtype=dtype)

  if mask_shape is None:
    mask = None
  else:
    mask = jax.random.bernoulli(rngs.pop(), shape=mask_shape)

  return q, k, v, bias, mask, rngs.pop()


def _run_test(
    q,
    k,
    v,
    *,
    bias=None,
    impl,
    impl_kwargs=None,
    impl_supports_precisions,
    ref_impl=nn.dot_product_attention,
    ref_kwargs=None,
    atol=1e-6,
    atol_grads=None,
    test_deterministic=True,
    test_vjp=True,
    test_vjp_deterministic=True,
    rng=None,
    **kwargs,
):
  """Runs an attention test with given inputs against the given reference."""
  if impl_kwargs is None:
    impl_kwargs = {}

  if impl_supports_precisions:
    default_precision = (
        jax.lax.Precision.HIGHEST
        if q.dtype == jnp.float32
        else jax.lax.Precision.DEFAULT
    )
    impl_kwargs.setdefault("precision", default_precision)

  if ref_kwargs is None:
    ref_kwargs = {}
  ref_kwargs.setdefault("precision", jax.lax.Precision.HIGHEST)

  if isinstance(mask := (impl_kwargs | kwargs).get("mask"), base.Mask):
    q_indices = impl_kwargs.get("q_indices", None)
    k_indices = impl_kwargs.get("k_indices", None)
    q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
    k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
    mask = mask.as_array(q_len_or_indices, k_len_or_indices)

  if mask is not None:
    is_fully_masked = jnp.all(jnp.swapaxes(mask, -2, -3) == 0, axis=-1)
    output_mask = jnp.where(is_fully_masked, 0.0, 1.0)[..., None]

    def apply_output_mask(impl):
      return lambda *args, **kwargs: impl(*args, **kwargs) * output_mask

    impl = apply_output_mask(impl)
    ref_impl = apply_output_mask(ref_impl)

  impl = functools.partial(impl, **kwargs, **impl_kwargs)
  ref_impl = functools.partial(ref_impl, **kwargs, **ref_kwargs)

  wrap = lambda f: lambda q, k, v, bias: f(q, k, v, bias=bias)
  impl, ref_impl = map(wrap, (impl, ref_impl))

  def as_f32(x):
    if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
      return x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return x

  ref_inputs = jax.tree.map(as_f32, (q, k, v, bias))

  # Forwards inference.
  actual = jax.jit(impl)(q, k, v, bias)
  expected = jax.jit(ref_impl)(*ref_inputs)
  min_atol = atol if isinstance(atol, float) else min(atol.values())
  rtol = max(min_atol / 10, 1e-6)
  test_utils.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  if test_deterministic:
    chex.assert_trees_all_equal(actual, jax.jit(impl)(q, k, v, bias))

  if not test_vjp:
    return

  del actual, expected  # Free device memory.

  # Forwards training.
  actual, vjp_fn = jax.vjp(impl, q, k, v, bias)
  expected, ref_vjp_fn = jax.vjp(ref_impl, *ref_inputs)
  chex.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  if rng is None:
    rng = jax.random.PRNGKey(42)
  dout = jax.random.normal(rng, expected.shape, dtype=actual.dtype)
  ref_dout = dout.astype(expected.dtype)
  del expected, ref_kwargs, ref_impl, ref_inputs  # Free device memory.

  # Backwards.
  if atol_grads is None:
    atol_grads = max(2 * atol, 5e-6)

  grad_names = ("dq", "dk", "dv", "dbias")
  actual_grads = dict(zip(grad_names, vjp_fn(dout), strict=True))
  expected_grads = dict(zip(grad_names, ref_vjp_fn(ref_dout), strict=True))
  rtol_grads = max(atol_grads / 10, 1e-6)
  chex.assert_trees_all_close(
      actual_grads, expected_grads, atol=atol_grads, rtol=rtol_grads
  )
  del expected_grads, vjp_fn, ref_vjp_fn  # Free device memory.

  if test_vjp_deterministic:
    actual2, vjp_fn = jax.vjp(impl, q, k, v, bias)
    actual_grads2 = dict(zip(grad_names, vjp_fn(dout)))
    chex.assert_trees_all_equal(actual, actual2)
    chex.assert_trees_all_equal(actual_grads, actual_grads2)


def override_test_args(**kwargs):
  orig_run_test = _run_test
  run_test = lambda *a, **kw: orig_run_test(*a, **(kw | kwargs))
  return mock.patch.object(sys.modules[__name__], "_run_test", run_test)


def _ref_impl_tanh(
    q,
    k,
    v,
    *,
    logits_soft_cap=None,
    bias=None,
    precision=None,
    mask=None,
):
  """Reference implementation for tanh attention."""
  q /= jnp.sqrt(q.shape[-1])
  logits = jnp.einsum("...qhd,...khd->...hqk", q, k, precision=precision)
  if bias is not None:
    logits += bias
  if logits_soft_cap is not None:
    logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)
  if mask is not None:
    logits = jnp.where(mask, logits, float(jnp.finfo(logits.dtype).min))
  weights = jax.nn.softmax(logits)
  return jnp.einsum("...hqk,...khd->...qhd", weights, v, precision=precision)


NAMED_ARG_SPECS = {
    s.full_name: s.args for s in ARG_SPECS if "primary" in s.tags
}


# pylint: disable=missing-function-docstring
class AttentionTestBase(parameterized.TestCase):
  """Base class for attention tests."""

  def __init__(
      self,
      *args,
      attention_fn,
      supports_bias=True,
      supports_mask=True,
      supports_indices=True,
      supports_tanh_clipping=True,
      supports_dropout=True,
      supports_vjp=True,
      supports_cross_attention=True,
      supports_precisions=True,
      supports_logits_dtype=True,
      supports_vmap=True,
      supports_is_causal=True,
  ):
    super().__init__(*args)
    self._attention_fn = attention_fn
    self._supports_bias = supports_bias
    self._supports_mask = supports_mask
    self._supports_indices = supports_indices
    self._supports_tanh_clipping = supports_tanh_clipping
    self._supports_dropout = supports_dropout
    self._supports_vjp = supports_vjp
    self._supports_cross_attention = supports_cross_attention
    self._supports_precisions = supports_precisions
    self._supports_logits_dtype = supports_logits_dtype
    self._supports_vmap = supports_vmap
    self._supports_is_causal = supports_is_causal

  def _run_test(
      self,
      q_shape,
      *,
      kv_shape=None,
      bias_shape=None,
      mask_shape=None,
      dtype=jnp.float32,
      **kwargs,
  ):
    q, k, v, bias, mask, rng = _create_inputs(
        q_shape=q_shape,
        kv_shape=kv_shape,
        bias_shape=bias_shape,
        mask_shape=mask_shape,
        dtype=dtype,
    )

    if mask is not None:
      kwargs["mask"] = mask

    self._run_test_with_inputs(q, k, v, bias=bias, rng=rng, **kwargs)

  def _run_test_with_inputs(self, *args, expect_supported=True, **kwargs):
    kwargs.setdefault("impl", self._attention_fn)
    kwargs.setdefault("impl_supports_precisions", self._supports_precisions)
    kwargs.setdefault("test_vjp", self._supports_vjp)

    with contextlib.ExitStack() as stack:
      if not expect_supported:
        stack.enter_context(self.assertRaises(Exception))

      _run_test(*args, **kwargs)

  @parameterized.parameters(jnp.float32, jnp.bfloat16, jnp.float16)
  def test_self_attention(self, dtype):
    if (dtype == jnp.float16) and (jax.default_backend().lower() == "tpu"):
      self.skipTest("float16 is not supported on TPU.")

    atol = {jnp.float32: 1e-6, jnp.bfloat16: 1e-2, jnp.float16: 5e-4}[dtype]
    self._run_test((2, 1024, 4, 64), dtype=dtype, atol=atol)

  @parameterized.parameters(False, True)
  def test_cross_attention(self, has_bias):
    self._run_test(
        q_shape=(2, 1024, 4, 64),
        kv_shape=(2, 1536, 4, 64),
        bias_shape=(2, 4, 1024, 1536) if has_bias else None,
        expect_supported=(
            self._supports_cross_attention
            and (self._supports_bias or not has_bias)
        ),
        atol=3e-6 if has_bias else 1e-6,
    )

  @pytest.mark.long
  @parameterized.product(input_dim=(24, 128), output_dim=(64, 112))
  def test_different_output_head_dim(self, input_dim, output_dim):
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(rng0, (2, 1024, 4, input_dim))
    k = jax.random.normal(rng1, (2, 1024, 4, input_dim))
    v = jax.random.normal(rng2, (2, 1024, 4, output_dim))
    self._run_test_with_inputs(q, k, v)

  def test_multi_query_attention(self):
    def ref_impl(q, k, v, **kwargs):
      k = jnp.broadcast_to(k, (*k.shape[:-2], q.shape[-2], k.shape[-1]))
      v = jnp.broadcast_to(v, (*v.shape[:-2], q.shape[-2], v.shape[-1]))
      return nn.dot_product_attention(q, k, v, **kwargs)

    self._run_test(
        q_shape=(2, 1024, 4, 64), kv_shape=(2, 1024, 1, 64), ref_impl=ref_impl
    )

  def test_grouped_query_attention(self):

    def ref(q, k, v, **kwargs):
      repeats = q.shape[-2] // k.shape[-2]
      k = jnp.repeat(k, repeats, axis=-2)
      v = jnp.repeat(v, repeats, axis=-2)
      return nn.dot_product_attention(q, k, v, **kwargs)

    self._run_test((2, 1024, 16, 64), kv_shape=(2, 1024, 4, 64), ref_impl=ref)

  def test_no_batch_dim(self):
    self._run_test((1024, 4, 64))

  def test_multiple_batch_dims(self):
    self._run_test((2, 3, 4, 1024, 4, 64))

  def test_non_power_of_two_q_seq_len(self):
    self._run_test(
        (2, 112, 4, 64),
        kv_shape=(2, 1024, 4, 64),
        expect_supported=self._supports_cross_attention,
    )

  def test_non_power_of_two_head_dim(self):
    self._run_test((2, 1024, 4, 48))

  @pytest.mark.long
  @parameterized.parameters(
      ((2, 4, 1024, 1024),),
      ((2, 4, 1024, 1),),
      ((2, 4, 1, 1024),),
      ((2, 1, 1024, 1024),),
      ((1, 4, 1024, 1024),),
      ((4, 1024, 1024),),
  )
  def test_bias(self, bias_shape):
    self._run_test(
        (2, 1024, 4, 64),
        bias_shape=bias_shape,
        atol=5e-6,
        expect_supported=self._supports_bias,
    )

  @pytest.mark.long
  @parameterized.parameters(
      ((2, 4, 1024, 1024),),
      ((2, 4, 1024, 1),),
      ((2, 4, 1, 1024),),
      ((2, 1, 1024, 1024),),
      ((1, 4, 1024, 1024),),
      ((2, 1, 1024, 1),),
      ((1, 4, 1024, 1),),
      ((2, 1, 1, 1024),),
      ((1, 4, 1, 1024),),
  )
  def test_mask(self, mask_shape):
    self._test_mask(mask_shape)

  def _test_mask(self, mask_shape):
    self._run_test(
        (2, 1024, 4, 64),
        mask_shape=mask_shape,
        expect_supported=self._supports_mask,
    )

  def test_causal_mask(self):
    mask = nn.make_causal_mask(jax.ShapeDtypeStruct((2, 1024), jnp.float32))
    self._run_test(
        (2, 1024, 4, 64),
        impl_kwargs=dict(is_causal=True),
        ref_kwargs=dict(mask=mask),
    )

  @pytest.mark.long
  @parameterized.parameters(512, 539)
  def test_causal_mask_cross_attention(self, seq_len_k):
    mask = jnp.tri(1024, seq_len_k, dtype=bool)
    self._run_test(
        (2, 1024, 4, 64),
        kv_shape=(2, seq_len_k, 4, 64),
        impl_kwargs=dict(is_causal=True),
        ref_kwargs=dict(mask=mask),
        expect_supported=self._supports_cross_attention,
        atol=1e-5,
    )

  def test_causal_mask_q_indices(self):
    rng = jax.random.PRNGKey(0)
    q_indices = jax.random.permutation(rng, jnp.arange(512), independent=True)
    mask = jnp.tri(512, dtype=bool)[q_indices]
    self._run_test(
        (2, 512, 4, 64),
        impl_kwargs=dict(is_causal=True, q_indices=q_indices[None, :]),
        ref_kwargs=dict(mask=mask),
        expect_supported=self._supports_mask and self._supports_indices,
    )

  def test_causal_mask_k_indices(self):
    rng = jax.random.PRNGKey(0)
    k_indices = jax.random.permutation(rng, jnp.arange(512), independent=True)
    mask = jnp.tri(512, dtype=bool)[:, k_indices]
    self._run_test(
        (2, 512, 4, 64),
        impl_kwargs=dict(is_causal=True, k_indices=k_indices[None, :]),
        ref_kwargs=dict(mask=mask),
        expect_supported=self._supports_mask and self._supports_indices,
    )

  def test_q_start_end_indices(self):
    rng = jax.random.PRNGKey(0)
    q_indices = jax.random.permutation(rng, jnp.arange(512), independent=True)
    q_start = jnp.array([42])
    q_end = q_start + 256
    mask = base.Mask(q_start=q_start, q_end=q_end)

    self._run_test(
        (2, 512, 4, 64),
        impl_kwargs=dict(mask=mask, q_indices=q_indices[None, :]),
        ref_kwargs=dict(mask=mask.as_array(q_indices, 512)),
        expect_supported=self._supports_mask and self._supports_indices,
    )

  def test_padding_mask(self):
    q_valid_len = jnp.array([[456], [987]])
    k_valid_len = jnp.array([[123], [999]])
    mask = base.Mask(q_end=q_valid_len[..., None], k_end=k_valid_len[..., None])
    q_valid = jnp.arange(1024) < q_valid_len
    k_valid = jnp.arange(1024) < k_valid_len
    ref_mask = nn.make_attention_mask(q_valid, k_valid, dtype=bool)

    self._run_test(
        (2, 1024, 4, 64),
        impl_kwargs=dict(mask=mask),
        ref_kwargs=dict(mask=ref_mask),
        expect_supported=self._supports_mask,
    )

  def test_padding_mask_with_nans(self):
    q_valid = jnp.arange(1024) < jnp.array([[456], [987]])
    k_valid = jnp.arange(1024) < jnp.array([[123], [999]])
    mask = nn.make_attention_mask(q_valid, k_valid, dtype=bool)

    def impl(q, k, v, *, mask, **kwargs):
      out = self._attention_fn(
          # Fill the padded region with NaNs to check they don't leak.
          jnp.where(q_valid[:, :, None, None], q, float("nan")),
          jnp.where(k_valid[:, :, None, None], k, float("nan")),
          jnp.where(k_valid[:, :, None, None], v, jnp.finfo(jnp.float32).max),
          mask=mask,
          **kwargs,
      )
      return jnp.where(q_valid[:, :, None, None], out, 999.0)

    def ref_impl(q, k, v, **kwargs):
      out = nn.dot_product_attention(q, k, v, **kwargs)
      return jnp.where(q_valid[:, :, None, None], out, 999.0)

    self._run_test(
        (2, 1024, 4, 64),
        mask=mask,
        impl=impl,
        ref_impl=ref_impl,
        test_vjp=False,
        expect_supported=self._supports_mask,
    )

  def test_local_attention_mask(self):
    window_size = 128
    k_start = jnp.maximum(0, jnp.arange(1024) + 1 - window_size)
    ref_mask = jnp.tri(1024, dtype=bool)
    ref_mask &= ~jnp.tri(1024, k=-window_size, dtype=bool)
    self._run_test(
        (2, 1024, 4, 64),
        impl_kwargs=dict(mask=base.Mask(k_start=k_start, is_causal=True)),
        ref_kwargs=dict(mask=ref_mask),
        expect_supported=self._supports_mask,
    )

  @pytest.mark.long
  @parameterized.parameters(
      dict(is_causal=True),  # Explicit causal
      dict(k_end=range(1, 1024 + 1)),  # Lower tri (implicit causal)
      dict(k_start=range(1024)),  # Higher tri using k_start
      dict(q_start=range(1024)),  # Lower tri using q_start
      dict(q_start=range(1023, -1, -1)),  # Lower tri mirrored along vertical
      dict(q_end=range(1, 1024 + 1)),  # Higher tri using q_end
      dict(q_end=range(1024, 0, -1)),  # Higher tri mirrored along vertical
      dict(
          # Triangular hourglass-like shape
          k_start=(range(512), range(512 - 1, -1, -1)),
          k_end=(range(1024, 512, -1), range(512 + 1, 1024 + 1)),
      ),
      dict(
          # Triangular hourglass-like shape. Offset by 3 per head (max_offs=12).
          k_start=(
              range(512 - 12),
              (512 - 12 - 1,) * 2 * 12,
              range(512 - 12 - 1, -1, -1),
          ),
          k_end=(
              range(1024, 512 + 12, -1),
              (1024 - 512 + 12 + 1,) * 2 * 12,
              range(512 + 12 + 1, 1024 + 1),
          ),
          offset_per_head=3,
      ),
      dict(
          # Transposed triangular hourglass-like shape
          q_start=(range(512), range(512 - 1, -1, -1)),
          q_end=(range(1024, 512, -1), range(512 + 1, 1024 + 1)),
      ),
      dict(k_start=512, is_causal=True),  # Lower tri (explicit causal) right
      dict(k_end=(512 + 1), is_causal=True),  # Lower tri (explicit causal) left
      # Lower tri (explicit causal) truncated left and right
      dict(k_start=256, k_end=(1024 - 256 + 1), is_causal=True),
      dict(q_start=17, k_start=range(1024)),
      dict(q_start=128, k_end=range(1, 1024 + 1)),
      dict(q_end=1007, k_start=range(1024)),
      dict(q_end=896, k_end=range(1, 1024 + 1)),
      dict(k_start=576),
  )
  def test_mask_api(self, **kwargs):
    self._test_mask_api(**kwargs)

  def _test_mask_api(self, **kwargs):
    num_heads = 4
    offset_per_head = kwargs.pop("offset_per_head", None)

    def _maybe_array(x):
      if isinstance(x, bool):
        return x

      if isinstance(x, tuple):
        arr = jnp.array(list(itertools.chain(*x)))
      elif isinstance(x, range):
        arr = jnp.array(x)
      elif isinstance(x, int):
        arr = jnp.array([x])
      else:
        raise ValueError(f"Unsupported type: {type(x)}")
      arr = arr[None, :]  # Add num_heads dimension

      if offset_per_head is not None:
        offsets = [offset_per_head * i for i in range(num_heads)]
        arr += jnp.array(offsets)[:, None]
      return arr

    kwargs = {k: _maybe_array(v) for k, v in kwargs.items()}
    mask = base.Mask(**kwargs)
    seq_len = 1024

    self._run_test(
        (2, seq_len, num_heads, 64),
        impl_kwargs=dict(mask=mask),
        ref_kwargs=dict(mask=mask.as_array(seq_len, seq_len)),
        expect_supported=(
            self._supports_mask
            or (self._supports_is_causal and kwargs == dict(is_causal=True))
        ),
    )

  @parameterized.parameters(True, False)
  def test_tanh_clipping(self, use_bias):
    if use_bias and not self._supports_bias:
      self.skipTest("Bias not supported")
    self._run_test(
        (2, 1024, 4, 64),
        # Check we perform clipping after bias.
        bias_shape=(2, 4, 1024, 1024) if use_bias else None,
        ref_impl=_ref_impl_tanh,
        expect_supported=self._supports_tanh_clipping,
        logits_soft_cap=10.0,
        atol=2e-6,
    )

  def test_dropout(self):
    dropout_rng = jax.random.PRNGKey(42)
    dropout_mask = jax.random.bernoulli(dropout_rng, shape=(2, 4, 1024, 1024))

    self._run_test(
        (2, 1024, 4, 64),
        dropout_rate=0.5,
        impl_kwargs=dict(dropout_mask=dropout_mask),
        ref_kwargs=dict(broadcast_dropout=False, dropout_rng=dropout_rng),
        expect_supported=self._supports_dropout,
    )

  @parameterized.parameters(
      dict(vmap_in_axes=((0, 0, 0, 0, 0),)),
      dict(vmap_in_axes=((0, None, None, 0, 0),)),
      dict(vmap_in_axes=((0, 0, 0, None, 0),)),
      dict(vmap_in_axes=((0, 0, 0, 0, None),)),
      dict(vmap_in_axes=((0, 0, 0, 0, 0), (0, 0, 0, 0, 0))),
      dict(vmap_in_axes=((None, 0, 0, 0, 0), (None, 0, 0, 0, 0))),
      dict(vmap_in_axes=((0, None, None, 0, 0), (0, None, None, 0, 0))),
      dict(vmap_in_axes=((0, None, None, None, 0), (None, 0, 0, 0, None))),
  )
  def test_vmap(self, vmap_in_axes):
    self._test_vmap(vmap_in_axes)

  def _test_vmap(self, vmap_in_axes):
    self.skipTest("Too slow for OSS")
    def vmap_impl(impl):
      def vmapped(q, k, v, *, bias, mask, **kwargs):
        def f(q, k, v, bias, mask):
          return impl(q, k, v, bias=bias, mask=mask, **kwargs)

        for in_axes in vmap_in_axes:
          f = jax.vmap(f, in_axes=in_axes)
        return f(q, k, v, bias, mask)

      return vmapped

    q_shape = [2, 3, 1024, 4, 64]
    kv_shape = [2, 3, 1024, 4, 64]
    bias_shape = [2, 3, 4, 1024, 1024]
    mask_shape = [2, 3, 4, 1024, 1024]

    for i, in_axes in enumerate(vmap_in_axes[::-1]):
      q_axis, k_axis, _, bias_axis, mask_axis = in_axes
      if q_axis is None:
        del q_shape[len(vmap_in_axes) - i - 1]
      if k_axis is None:
        del kv_shape[len(vmap_in_axes) - i - 1]
      if bias_axis is None:
        del bias_shape[len(vmap_in_axes) - i - 1]
      if mask_axis is None:
        del mask_shape[len(vmap_in_axes) - i - 1]

    self._run_test(
        q_shape=q_shape,
        kv_shape=kv_shape,
        bias_shape=bias_shape,
        mask_shape=mask_shape,
        impl=vmap_impl(self._attention_fn),
        ref_impl=vmap_impl(nn.dot_product_attention),
        atol=3e-6,
        expect_supported=(
            self._supports_bias and self._supports_mask and self._supports_vmap
        ),
    )

  @pytest.mark.long
  @parameterized.parameters(
      dict(q_shape=(4, 1024, 2, 64), bias_shape=(4, 2, 1024, 512)),  # bias
      dict(q_shape=(4, 1024, 2, 64), mask_shape=(4, 2, 512, 1024)),  # mask
  )
  def test_invalid_shapes(self, **kwargs):
    self._test_invalid_shapes(**kwargs)

  def _test_invalid_shapes(self, **kwargs):
    self.skipTest("Too slow for OSS")
    self._run_test(**kwargs, expect_supported=False)

  @pytest.mark.long
  @parameterized.product(
      channelwise_axes=(
          (0, 1, 2),
      ),
      quantize_q=(True, False),
  )
  def test_quantized_int8(self, channelwise_axes, quantize_q):
    self._test_quantized_int8(channelwise_axes, quantize_q)

  def _test_quantized_int8(self, channelwise_axes, quantize_q):
    quantize = functools.partial(
        qwix.quantize, qtype=jnp.int8, channelwise_axes=channelwise_axes
    )
    quant_dequant = lambda x: qwix.dequantize(quantize(x))

    def impl(q, k, v, **kwargs):
      q = quantize(q) if quantize_q else q
      k = quantize(k)
      v = quantize(v)
      return self._attention_fn(q, k, v, **kwargs)

    def ref_impl(q, k, v, **kwargs):
      q = quant_dequant(q) if quantize_q else q
      k = quant_dequant(k)
      v = quant_dequant(v)
      return nn.dot_product_attention(q, k, v, **kwargs)

    self._run_test(
        (2, 1024, 4, 64), impl=impl, ref_impl=ref_impl, test_vjp=False
    )

  @pytest.mark.long
  @parameterized.parameters(64, 128, 256)
  def test_quantized_int4(self, subchannel_size):
    self._test_quantized_int4(subchannel_size)

  def _test_quantized_int4(self, subchannel_size):
    tiled_axes = {0: 1, 1: 1, 2: 1, 3: subchannel_size}
    quantize = lambda x: qwix.quantize(x, jnp.int4, tiled_axes=tiled_axes)
    quant_dequant = lambda x: qwix.dequantize(quantize(x))

    def impl(q, k, v, **kwargs):
      k, v = map(quantize, (k, v))
      return self._attention_fn(q, k, v, **kwargs)

    def ref_impl(q, k, v, **kwargs):
      k, v = map(quant_dequant, (k, v))
      return nn.dot_product_attention(q, k, v, **kwargs)

    self._run_test(
        (2, 1024, 4, 256), impl=impl, ref_impl=ref_impl, test_vjp=False
    )

  @pytest.mark.long
  @parameterized.parameters("bfloat16", "float16")
  def test_logits_dtype(self, dtype):
    self._run_test(
        (2, 1024, 4, 64),
        impl_kwargs=dict(logits_dtype=dtype),
        dtype=dtype,
        atol=({"float16": 1e-2, "bfloat16": 1e-2})[dtype],
        expect_supported=self._supports_logits_dtype,
    )

  def test_normalize_output(self):

    def ref_impl(q, k, v, *, bias, mask=None, precision):
      assert bias is None
      assert mask is None
      einsum = functools.partial(jnp.einsum, precision=precision)
      s = einsum("...qhd,...khd->...hqk", q, k)
      p_unnormalized = jnp.exp(s - jnp.max(s, axis=-1, keepdims=True))
      return einsum("...hqk,...khd->...qhd", p_unnormalized, v)

    self._run_test(
        (2, 1024, 4, 64),
        impl_kwargs=dict(logits_scale=1.0, normalize_output=False),
        ref_impl=ref_impl,
        atol=1e-5,
    )

  def test_partial_attention(self):
    def partial_impl(q, k, v, **kwargs):
      num_chunks = 2
      chunk_size = k.shape[-3] // num_chunks

      def run_partial_attention(i):
        chunk = lambda x: x[..., i * chunk_size : (i + 1) * chunk_size, :, :]
        o, (m, l) = self._attention_fn(q, chunk(k), chunk(v), **kwargs)
        return o, m.mT[..., None], l.mT[..., None]

      outs = [run_partial_attention(i) for i in range(num_chunks)]
      outs, maxes, sums = tuple(zip(*outs))

      max_ = jnp.max(jnp.stack(maxes), axis=0)
      out_ = sum([jnp.exp(m - max_) * o for (o, m) in zip(outs, maxes)])
      sum_ = sum([jnp.exp(m - max_) * s for (s, m) in zip(sums, maxes)])

      return jnp.where(sum_ == 0.0, 0.0, out_ / sum_)

    self._run_test(
        (2, 1024, 4, 64),
        impl=partial_impl,
        impl_kwargs=dict(return_residuals=True, normalize_output=False),
        test_vjp=False,
        expect_supported=self._supports_cross_attention,
    )

  @pytest.mark.long
  @parameterized.named_parameters(NAMED_ARG_SPECS.items())
  def test_bench(self, spec):
    self._test_bench(spec)

  def _test_bench(self, spec):
    self.skipTest("Awaiting Bug Fixes")

    spec = dict(spec)  # We need to take a copy to avoid modifying other tests.
    q, k, v, bias, mask = numerics.random_initialize((
        spec.pop("q"),
        spec.pop("k"),
        spec.pop("v"),
        spec.pop("bias", None),
        spec.pop("mask", None),
    ))
    is_causal = spec.pop("is_causal", False)
    logits_soft_cap = spec.get("logits_soft_cap")
    unnormalized = not spec.get("normalize_output", True)

    if unnormalized:
      # ensure that we return residuals to normalize the output
      spec["return_residuals"] = True

    def impl(*args, **kwargs):
      return_residuals = kwargs.get("return_residuals", False)
      logits_scale = kwargs.get("logits_scale", base.AUTO)

      q, *rest = args
      if logits_scale != base.AUTO:
        q_dtype = q.dtype
        q = (q / logits_scale) / jnp.sqrt(q.shape[-1])
        q = q.astype(q_dtype)
      out = self._attention_fn(q, *rest, **kwargs)
      attn = out[0] if return_residuals else out
      if unnormalized:
        _, (_, l) = out
        attn /= jnp.swapaxes(l, -1, -2)[..., None]
      return attn

    if k.shape[-3] > (32 * 1024):
      ref_impl = xla_chunked.XlaChunkedDotProductAttention()
      ref_kwargs = dict(precision=jax.lax.DotAlgorithmPreset.F32_F32_F32)
    elif logits_soft_cap:
      ref_impl = _ref_impl_tanh
      ref_kwargs = dict(logits_soft_cap=logits_soft_cap)
    else:
      ref_impl = nn.dot_product_attention
      ref_kwargs = dict()

    def ref_impl_(*args, **kwargs):
      q, k, v, *rest = args
      q, k, v = map(quantization.as_array, (q, k, v))
      mask = kwargs.pop("mask", None)
      is_causal = kwargs.pop("is_causal", None)

      *_, q_seq, q_heads, _ = q.shape
      *_, k_seq, k_heads, _ = k.shape

      if isinstance(mask, base.Mask):
        mask = mask.as_array(q_seq, k_seq)
      if is_causal:
        causal_mask = jnp.tri(q_seq, k_seq, dtype=bool)
        mask = (mask & causal_mask) if mask is not None else causal_mask
      if q_heads != k_heads:
        k, v = map(lambda x: jnp.repeat(x, q_heads // k_heads, axis=-2), (k, v))

      return ref_impl(q, k, v, *rest, mask=mask, **kwargs)

    expect_supported = (
        (self._supports_cross_attention or q.shape[-3] == k.shape[-3])
        and (self._supports_tanh_clipping or logits_soft_cap is None)
        and (self._supports_mask or mask is None)
        and (self._supports_bias or bias is None)
    )

    is_quantized = any(isinstance(x, qwix.QArray) for x in (q, k, v))
    test_vjp = (
        self._supports_vjp and not is_quantized and q.shape[-2] == k.shape[-2]
    )

    atol = {jnp.float32: 2e-6, jnp.bfloat16: 2e-2}[q.dtype.type]
    atol_grads_bias = {jnp.float32: 2e-5, jnp.bfloat16: 8e-2}[q.dtype.type]
    try:
      self._run_test_with_inputs(
          q,
          k,
          v,
          bias=bias,
          mask=mask,
          is_causal=is_causal,
          impl=impl,
          impl_kwargs=spec,
          ref_impl=ref_impl_,
          ref_kwargs=ref_kwargs,
          atol=atol,
          atol_grads=None if bias is None else atol_grads_bias,
          expect_supported=expect_supported,
          test_vjp=test_vjp,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      if "RESOURCE_EXHAUSTED: Out of memory while trying to allocate" in str(e):
        self.skipTest("Out of memory")  # TODO: Use XLA chunked?
      raise


class AttentionManualPartitioningTestBase(parameterized.TestCase):
  """Base class for attention manual partitioning tests."""

  # TODO: Add support and tests for partitioning with:
  # - Cross attention
  # - Different k/v num_heads
  # - Causal mask + seq_q sharding (self- and cross-)
  # - No batch/multiple batches
  # - Different sharding for q/kv heads
  # - Non-broadcasted multi-query
  # - Add test for vmap of vmap.

  _TEST_SHAPE = (16, 256, 16, 32)
  _ATTENTION_AXES = ("batch", "seq_q", "heads", "head_dim")
  _BIAS_MASK_AXES = ("batch", "heads", "seq_q", "seq_k")
  _PARTITION_AXES = _ATTENTION_AXES[:-1]
  _BIAS_MASK_SHAPES = (
      (1, 16, 256, 256),
      (16, 1, 256, 256),
      (16, 16, 1, 256),
      # Partitioning along seq_k is not supported: no need to test seq_k == 1.
      (16, 16, 256, 256),
  )

  def __init__(
      self,
      *args,
      attention_fn,
      supports_bias=True,
      supports_mask=True,
      supports_vjp=True,
      supports_precisions=True,
      supports_causal_sequence_sharding=True,
      supports_num_heads_sharding=True,
  ):
    super().__init__(*args)
    self._attention_fn = attention_fn
    self._supports_bias = supports_bias
    self._supports_mask = supports_mask
    self._supports_vjp = supports_vjp
    self._supports_precisions = supports_precisions
    self._supports_causal_sequence_sharding = supports_causal_sequence_sharding
    self._support_num_heads_sharding = supports_num_heads_sharding

  def _run_test(
      self,
      *,
      q_shape=None,
      kv_shape=None,
      bias_shape=None,
      mask_shape=None,
      impl=None,
      ref_impl=None,
      partition_axes,
      expect_supported=True,
      spec=None,
      **kwargs,
  ):
    num_axes = len(partition_axes)
    if num_axes == 1:
      devices = jax.devices()
    elif num_axes == 2:
      if jax.device_count() == 1:
        devices = np.array(jax.devices()).reshape((1, 1))
      else:
        devices = np.array(jax.devices()).reshape((2, -1))
    else:
      raise ValueError("Unsupported number of partition axes.")

    mesh = jax.sharding.Mesh(devices, partition_axes)
    if spec is None:
      spec = jax.sharding.PartitionSpec(
          *(a if a in partition_axes else None for a in self._ATTENTION_AXES)
      )

    impl_kwargs = kwargs.get("impl_kwargs", {})
    impl_kwargs |= dict(q_sharding=jax.sharding.NamedSharding(mesh, spec))

    if q_shape is None:
      q_shape = self._TEST_SHAPE
    if impl is None:
      impl = self._attention_fn
    if ref_impl is None:
      ref_impl = self._attention_fn

    kwargs |= dict(
        ref_impl=ref_impl,
        impl=impl,
        test_vjp=self._supports_vjp,
        impl_kwargs=impl_kwargs,
        test_deterministic=True,
        atol=0.0,
        impl_supports_precisions=self._supports_precisions,
    )

    q, k, v, bias, mask, rng = _create_inputs(
        q_shape=q_shape,
        kv_shape=kv_shape,
        bias_shape=bias_shape,
        mask_shape=mask_shape,
    )

    if mask is not None:
      kwargs["mask"] = mask

    expect_supported &= (
        self._support_num_heads_sharding
        or mesh.shape.get(self._ATTENTION_AXES[-2], 1) == 1
    )

    with contextlib.ExitStack() as stack:
      if not expect_supported:
        stack.enter_context(self.assertRaises(Exception))

      _run_test(q, k, v, bias=bias, rng=rng, **kwargs)

  @parameterized.parameters(*_PARTITION_AXES)
  def test_self_attention_one_axis(self, partition_axis):
    self._run_test(partition_axes=(partition_axis,))

  @parameterized.parameters(*_PARTITION_AXES)
  def test_self_attention_one_axis_shorter_spec(self, partition_axis):
    axis_idx = self._ATTENTION_AXES.index(partition_axis)
    spec_tuple = (None,) * axis_idx + (partition_axis,)

    self._run_test(
        partition_axes=(partition_axis,),
        spec=jax.sharding.PartitionSpec(*spec_tuple),
    )

  @parameterized.parameters(*_PARTITION_AXES)
  def test_causal_attention_one_axis(self, partition_axis):
    self._run_test(
        partition_axes=(partition_axis,),
        is_causal=True,
        expect_supported=(
            partition_axis != "seq_q" or self._supports_causal_sequence_sharding
        ),
    )

  @parameterized.parameters(
      ((_PARTITION_AXES[0], _PARTITION_AXES[1]),),
      ((_PARTITION_AXES[1], _PARTITION_AXES[2]),),
  )
  def test_self_attention_two_axes(self, partition_axes):
    self._run_test(partition_axes=partition_axes)

  @parameterized.product(bias_shape=_BIAS_MASK_SHAPES)
  def test_self_attention_bias(self, bias_shape):
    try:
      partition_axis = self._BIAS_MASK_AXES[bias_shape.index(1)]
    except ValueError:
      partition_axis = self._BIAS_MASK_AXES[0]

    self._run_test(
        partition_axes=(partition_axis,),
        expect_supported=self._supports_bias,
        bias_shape=bias_shape,
    )

  @parameterized.product(mask_shape=_BIAS_MASK_SHAPES)
  def test_self_attention_mask(self, mask_shape):
    self._test_self_attention_mask(mask_shape)

  def _test_self_attention_mask(self, mask_shape):
    try:
      partition_axis = self._BIAS_MASK_AXES[mask_shape.index(1)]
    except ValueError:
      partition_axis = self._BIAS_MASK_AXES[0]

    self._run_test(
        partition_axes=(partition_axis,),
        expect_supported=self._supports_mask,
        mask_shape=mask_shape,
    )

  @parameterized.parameters(
      ((_PARTITION_AXES[0],),),
      ((_PARTITION_AXES[-1],),),
      ((_PARTITION_AXES[0], _PARTITION_AXES[-1]),),
  )
  def test_self_attention_mask_api_local(self, partition_axes):
    window_size, window_offset = 64, 32
    seq_len = self._TEST_SHAPE[-3]
    seq_range = jnp.arange(seq_len) + 1
    k_start = jnp.maximum(0, seq_range - window_size) + window_offset
    k_end = k_start + jnp.minimum(seq_range, window_size)
    mask = base.Mask(k_start=k_start, k_end=k_end)
    self._run_test(
        partition_axes=partition_axes,
        impl_kwargs=dict(mask=mask),
        ref_kwargs=dict(mask=mask.as_array(seq_len, seq_len)),
    )

  @parameterized.parameters(*zip(_PARTITION_AXES))
  def test_broadcasted_multi_query_attention(self, partition_axis):
    kv_shape = self._TEST_SHAPE[:-2] + (1,) + self._TEST_SHAPE[-1:]
    self._run_test(partition_axes=(partition_axis,), kv_shape=kv_shape)

  @parameterized.product(
      vmap_in_axes=(
          ((0, 0, 0, 0, 0),),
          ((0, None, None, 0, 0),),
          ((0, 0, 0, None, None),),
      ),
      partition_axis=_PARTITION_AXES,
  )
  def test_vmap(self, vmap_in_axes, partition_axis):
    def vmap_impl(impl):
      def vmapped(q, k, v, *, bias, mask, **kwargs):
        def f(q, k, v, bias, mask):
          return impl(q, k, v, bias=bias, mask=mask, **kwargs)

        for in_axes in vmap_in_axes:
          f = jax.vmap(f, in_axes=in_axes)
        return f(q, k, v, bias, mask)

      return vmapped

    q_shape = [2, 4, 1024, 4, 64]
    kv_shape = [2, 4, 1024, 4, 64]
    bias_shape = [2, 4, 4, 1024, 1024]
    mask_shape = [2, 4, 4, 1024, 1024]

    for i, in_axes in enumerate(vmap_in_axes[::-1]):
      q_axis, k_axis, _, bias_axis, mask_axis = in_axes
      if q_axis is None:
        del q_shape[len(vmap_in_axes) - i - 1]
      if k_axis is None:
        del kv_shape[len(vmap_in_axes) - i - 1]
      if bias_axis is None:
        del bias_shape[len(vmap_in_axes) - i - 1]
      if mask_axis is None:
        del mask_shape[len(vmap_in_axes) - i - 1]

    self._run_test(
        partition_axes=(partition_axis,),
        q_shape=q_shape,
        kv_shape=kv_shape,
        bias_shape=bias_shape,
        mask_shape=mask_shape,
        impl=vmap_impl(self._attention_fn),
        ref_impl=vmap_impl(self._attention_fn),
        expect_supported=self._supports_bias and self._supports_mask,
    )

  @parameterized.product(
      partition_axis=_PARTITION_AXES,
      channelwise_axes=((0, 1, 2), (0, 2, 3), (0,)),
      quantize_q=(True, False),
  )
  def test_quantized_int8(self, partition_axis, channelwise_axes, quantize_q):
    quantize = functools.partial(
        qwix.quantize, qtype=jnp.int8, channelwise_axes=channelwise_axes
    )

    def impl(q, k, v, **kwargs):
      q = quantize(q) if quantize_q else q
      k = quantize(k)
      v = quantize(v)
      return self._attention_fn(q, k, v, **kwargs)

    self._run_test(
        partition_axes=(partition_axis,),
        impl=impl,
        ref_impl=impl,
        test_vjp=False,
    )


# pylint: enable=missing-function-docstring
