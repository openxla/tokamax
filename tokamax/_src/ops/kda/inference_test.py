# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Native inference equivalence without a training residual tape."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import inference, api
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as f

interpret_on_cpu = f.interpret_on_cpu


@pytest.mark.parametrize(
    "lengths", [(64, 64), (17, 79), (1, 62, 2), (0, 0), (1,) * 9]
)
@pytest.mark.parametrize("states", [False, True])
def test_inference(lengths, states):
  h, b, t, d = 2, 1, 128, 128
  keys = jax.random.split(jax.random.key(19), 6)
  q, k, v, g = [
      jax.random.normal(key, (b, t, h, d)).astype(jnp.bfloat16)
      for key in keys[:4]
  ]
  beta = jnp.full((b, t, h), 0.5, jnp.bfloat16)
  ids = np.zeros((b, t), np.int32)
  start = 0
  for i, length in enumerate(lengths):
    ids[:, start : start + length] = i + 1
    start += length
  ids = jnp.array(ids)
  n = len(lengths) + 1
  h0 = jax.random.normal(keys[4], (b, n, h, d, d)) * 0.1 if states else None
  alog = jnp.zeros((h,), jnp.float32)
  bias = jnp.full((h * d,), -2.0, jnp.float32)
  actual = inference.kimi_delta_attention_inference(
      q,
      k,
      v,
      g,
      beta,
      segment_ids=ids,
      a_log=alog,
      delta_time_bias=bias,
      initial_state=h0,
      output_final_state=states,
      use_qk_l2norm_in_kernel=True,
      use_gate_in_kernel=True,
      lower_bound=-5.0,
      max_num_segments=n,
  )
  # Finish interpreter callbacks before starting the CPU reference; nested
  # CPU dispatch from both can otherwise deadlock the interpreter callback.
  jax.block_until_ready(actual)
  expected = api.kimi_delta_attention(
      *[x.transpose(2, 0, 1, 3) for x in (q, k, v, g)],
      beta.transpose(2, 0, 1),
      segment_ids=ids,
      a_log=alog,
      delta_time_bias=bias,
      initial_state=h0,
      output_final_state=states,
      use_qk_l2norm=True,
      use_gate_in_kernel=True,
      lower_bound=-5.0,
      max_num_segments=n,
      implementation="xla",
  )
  f._assert_close(
      (actual[0].transpose(2, 0, 1, 3), actual[1]), expected, tolerance=0.05
  )
  # A scale-relative check prevents the absolute BF16 tolerance from hiding
  # missing low-magnitude outputs (for example an omitted middle segment).
  for a, e in zip(
      jax.tree.leaves((actual[0].transpose(2, 0, 1, 3), actual[1])),
      jax.tree.leaves(expected),
      strict=True,
  ):
    a, e = np.asarray(a, np.float32), np.asarray(e, np.float32)
    assert np.linalg.norm(a - e) <= 0.02 * np.linalg.norm(e) + 1e-6
  np.testing.assert_array_equal(np.asarray(actual[0])[:, start:], 0)


@pytest.mark.parametrize(
    "invalid",
    ["dtype", "normalization", "gate", "bound", "segments", "state_dtype"],
)
def test_reject_unsupported_options(invalid):
  q = jnp.zeros((1, 64, 2, 128), jnp.bfloat16)
  beta = jnp.ones((1, 64, 2), jnp.bfloat16)
  kwargs = dict(
      segment_ids=jnp.ones((1, 64), jnp.int32),
      a_log=jnp.zeros((2,)),
      delta_time_bias=jnp.zeros((256,)),
      use_qk_l2norm_in_kernel=True,
      use_gate_in_kernel=True,
      lower_bound=-5.0,
      max_num_segments=1,
  )
  if invalid == "dtype":
    q = q.astype(jnp.float32)
  if invalid == "normalization":
    kwargs["use_qk_l2norm_in_kernel"] = False
  if invalid == "gate":
    kwargs["use_gate_in_kernel"] = False
  if invalid == "bound":
    kwargs["lower_bound"] = -6.0
  if invalid == "segments":
    kwargs["max_num_segments"] = 0
  if invalid == "state_dtype":
    kwargs["initial_state"] = jnp.zeros((1, 1, 2, 128, 128), jnp.bfloat16)
  with pytest.raises(ValueError):
    inference.kimi_delta_attention_inference(q, q, q, q, beta, **kwargs)


@pytest.mark.parametrize("batch", [1, 3])
def test_inference_shared_segment_map(batch):
  """A 1D [T] segment_ids must work for every batch size.

  Regression test: the wrapper used to expand [T] to [1, T], so B>1 inputs
  failed the shape check even though the native kernel broadcasts the map.
  """
  h, t, d = 2, 128, 128
  keys = jax.random.split(jax.random.key(53), 6)
  q, k, v, g = [
      jax.random.normal(key, (batch, t, h, d)).astype(jnp.bfloat16)
      for key in keys[:4]
  ]
  beta = jnp.full((batch, t, h), 0.5, jnp.bfloat16)
  lengths = (64, 64)
  ids_1d = np.zeros((t,), np.int32)
  start = 0
  for i, length in enumerate(lengths):
    ids_1d[start : start + length] = i + 1
    start += length
  n = len(lengths) + 1
  alog = jnp.zeros((h,), jnp.float32)
  bias = jnp.full((h * d,), -2.0, jnp.float32)
  actual = inference.kimi_delta_attention_inference(
      q,
      k,
      v,
      g,
      beta,
      segment_ids=jnp.array(ids_1d),
      a_log=alog,
      delta_time_bias=bias,
      use_qk_l2norm_in_kernel=True,
      use_gate_in_kernel=True,
      lower_bound=-5.0,
      max_num_segments=n,
  )
  jax.block_until_ready(actual)
  expected = api.kimi_delta_attention(
      *[x.transpose(2, 0, 1, 3) for x in (q, k, v, g)],
      beta.transpose(2, 0, 1),
      # The XLA reference only accepts the documented [B, T] layout, so
      # broadcast the shared map here; the inference wrapper under test must
      # handle the raw 1D form.
      segment_ids=jnp.broadcast_to(ids_1d[None], (batch, t)),
      a_log=alog,
      delta_time_bias=bias,
      use_qk_l2norm=True,
      use_gate_in_kernel=True,
      lower_bound=-5.0,
      max_num_segments=n,
      implementation="xla",
  )
  f._assert_close(
      (actual[0].transpose(2, 0, 1, 3), actual[1]), expected, tolerance=0.05
  )
