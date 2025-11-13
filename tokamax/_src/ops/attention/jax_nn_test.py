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

import pytest
from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src.ops.attention import jax_nn
from tokamax._src.ops.attention import test_base
from typing_extensions import override

_CUDNN_CUSTOM_CALL_TARGET = 'custom_call_target="__cudnn'


class JaxNnDotProductAttentionTest(test_base.AttentionTestBase):

  def __init__(self, *args, supports_vjp=True, implementation=None):
    attn = jax_nn.JaxNnDotProductAttention(implementation=implementation)
    super().__init__(
        *args,
        attention_fn=attn,
        supports_tanh_clipping=False,
        supports_dropout=False,
        supports_vjp=supports_vjp,
    )

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def _run_test_with_inputs(self, *args, **kwargs):
    impl_kwargs = kwargs.get("impl_kwargs", {})
    # pytype: disable=attribute-error
    if (
        (impl_kwargs.get("logits_dtype") not in (None, jnp.float32))
        or not impl_kwargs.get("normalize_output", True)
        or impl_kwargs.get("return_residuals", False)
    ):
      # pytype: enable=attribute-error
      kwargs["expect_supported"] = False
    super()._run_test_with_inputs(*args, **kwargs)


@pytest.mark.skip(reason="Too slow for OSS regression tests.")
class JaxNnDotProductAttentionXlaTest(JaxNnDotProductAttentionTest):

  def __init__(self, *args):
    super().__init__(*args, implementation="xla")


@pytest.mark.skip(reason="Too slow for OSS regression tests.")
class JaxNnDotProductAttentionCudnnTest(JaxNnDotProductAttentionTest):

  def __init__(self, *args):
    # TODO: Vjp has many restrictions. Work around and enable tests.
    super().__init__(*args, supports_vjp=False, implementation="cudnn")

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def _run_test_with_inputs(self, *args, **kwargs):
    # CuDNN doesn't support f32 inputs.
    orig_impl = kwargs.get("impl", self._attention_fn)

    def impl(q, k, v, *, bias, **kwargs):
      cast = lambda x: None if x is None else x.astype(jnp.bfloat16)
      return orig_impl(cast(q), cast(k), cast(v), bias=cast(bias), **kwargs)

    kwargs["impl"] = impl
    kwargs["atol"] = 0.025 if "bias" in kwargs else 0.015
    if args[0].shape[-1] > 128:
      kwargs["expect_supported"] = False
    super()._run_test_with_inputs(*args, **kwargs)

  def test_padding_mask_with_nans(self):
    self.skipTest("Unsupported.")

  def test_bench_veo3_veo3(self):
    self.skipTest("Numerical issue in CuDNN for head_dim=256.")

  @override
  def _test_vmap(self, vmap_in_axes):
    self.skipTest("Too slow for OSS")
    for in_axes in vmap_in_axes:
      if any(x is None for x in in_axes[:3]):
        self.skipTest("Unsupported replication for qkv.")
    super()._test_vmap(vmap_in_axes)

  def test_impl_in_hlo(self):
    x = jnp.empty((2, 256, 4, 64), dtype=jnp.bfloat16)
    lowered = jax.jit(self._attention_fn).lower(x, x, x)
    hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    self.assertIn(_CUDNN_CUSTOM_CALL_TARGET, hlo_text)


if __name__ == "__main__":
  absltest.main()
