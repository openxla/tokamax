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
"""Tests for scaling Splash Attention logits after QK accumulation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib


_SEQ_LEN = 128
_HEAD_DIM = 128
_SOFTMAX_SCALE = 192**-0.5


def _inputs():
  q_key, k_key, v_key, do_key = jax.random.split(jax.random.key(7), 4)
  shape = (1, _SEQ_LEN, _HEAD_DIM)
  q = jax.random.normal(q_key, shape, dtype=jnp.bfloat16) * 0.25
  k = jax.random.normal(k_key, shape, dtype=jnp.bfloat16) * 0.25
  v = jax.random.normal(v_key, shape, dtype=jnp.bfloat16)
  do = jax.random.normal(do_key, shape, dtype=jnp.bfloat16)
  return q, k, v, do


def _reference(q, k, v):
  logits = jnp.einsum(
      "hsd,htd->hst",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
  )
  logits *= jnp.float32(_SOFTMAX_SCALE)
  probabilities = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum("hst,htd->hsd", probabilities, v.astype(jnp.float32))


def _kernel(*, use_base2_exp, save_residuals=False):
  config = splash.SplashConfig(
      block_q=_SEQ_LEN,
      block_kv=_SEQ_LEN,
      block_q_dkv=_SEQ_LEN,
      block_kv_dkv=_SEQ_LEN,
      softmax_scale=_SOFTMAX_SCALE,
      use_base2_exp=use_base2_exp,
      interpret=True,
  )
  mask = mask_lib.FullMask((_SEQ_LEN, _SEQ_LEN))
  return splash.make_splash_mha_single_device(
      mask, config=config, save_residuals=save_residuals
  )


@pytest.mark.parametrize("use_base2_exp", [False, True])
def test_softmax_scale_is_applied_to_fp32_attention_logits(use_base2_exp):
  q, k, v, _ = _inputs()
  attention = _kernel(use_base2_exp=use_base2_exp, save_residuals=True)

  output, stats = attention(q, k, v)
  expected = _reference(q, k, v)
  expected_logits = jnp.einsum(
      "hsd,htd->hst",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
  ) * jnp.float32(_SOFTMAX_SCALE)

  np.testing.assert_allclose(
      output.astype(jnp.float32), expected, atol=1e-3, rtol=1e-2
  )
  np.testing.assert_allclose(
      stats["max_logits"],
      jnp.max(expected_logits, axis=-1),
      atol=1e-6,
      rtol=1e-6,
  )
  np.testing.assert_allclose(
      stats["logsumexp"],
      jax.nn.logsumexp(expected_logits, axis=-1),
      atol=1e-6,
      rtol=1e-6,
  )


@pytest.mark.parametrize("use_base2_exp", [False, True])
def test_softmax_scale_is_included_in_query_and_key_gradients(use_base2_exp):
  q, k, v, do = _inputs()
  attention = _kernel(use_base2_exp=use_base2_exp)

  output, pullback = jax.vjp(attention, q, k, v)
  expected, reference_pullback = jax.vjp(_reference, q, k, v)
  dq, dk, dv = pullback(do)
  dq_expected, dk_expected, dv_expected = reference_pullback(do.astype(expected.dtype))

  np.testing.assert_allclose(
      output.astype(jnp.float32), expected, atol=1e-3, rtol=1e-2
  )
  np.testing.assert_allclose(
      dq.astype(jnp.float32),
      dq_expected.astype(jnp.float32),
      atol=1e-3,
      rtol=1e-2,
  )
  np.testing.assert_allclose(
      dk.astype(jnp.float32),
      dk_expected.astype(jnp.float32),
      atol=1e-3,
      rtol=1e-2,
  )
  np.testing.assert_allclose(
      dv.astype(jnp.float32),
      dv_expected.astype(jnp.float32),
      atol=1e-3,
      rtol=1e-2,
  )
