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
"""Tests for ring attention."""

import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import ring_attention_kernel
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_test_utils as test_utils

P = jax.sharding.PartitionSpec
partial = functools.partial

jax.config.parse_flags_with_absl()


class PallasBaseTest(test_utils.SplashAttentionTestCase):
  INTERPRET = False

  def setUp(self):
    super().setUp()
    if not test_utils.test_device_matches(["tpu"]):
      self.skipTest("Test requires TPU.")

    if len(jax.devices()) < 4:
      self.skipTest("This test requires at least 4 devices.")


class RingAttentionTest(PallasBaseTest):

  def setUp(self):
    self.skipTest("no sharding on runners")
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.product(
      topology=[(1, 4)],  # Ring attention requires >= 2 devices on ring axis
      batch_size=[1, 2],
      num_heads=[1],
      head_dim=[128, 256],
      dtype=[jnp.bfloat16],
      is_mqa=[False, True],
      padding_factor=[0, 1],
      mask=[None, "FULL CAUSAL"],
      fused_bwd=[True],
  )
  def test_ring_attention_mha_fwd_bwd(
      self,
      topology,
      batch_size,
      num_heads,
      head_dim,
      dtype,
      is_mqa,
      padding_factor,
      mask,
      fused_bwd,
  ):
    head_shards, q_seq_shards = topology
    num_devices = math.prod(topology)
    if head_shards > num_heads:
      self.skipTest(
          f"This test requires {num_heads} heads, but has only"
          f" {head_shards} head shards available."
      )
    if mask == "FULL CAUSAL" and padding_factor > 0:
      self.skipTest("Full causal mask not supported with padding yet.")

    # Mesh Creation and Input Generation
    devices = np.asarray(jax.devices()[:num_devices]).reshape(
        head_shards, q_seq_shards
    )
    mesh = jax.sharding.Mesh(devices, ("heads", "ring"))
    seq_shard = 256
    seq_shard_pad = padding_factor * 128
    seq_len = (seq_shard + seq_shard_pad) * q_seq_shards
    if len(jax.devices()) < num_devices:
      self.skipTest(
          f"This test requires {num_devices} devices, but has only"
          f" {len(jax.devices())} devices available."
      )
    k1, k2, k3, k4 = random.split(random.key(0), 4)
    q = random.uniform(
        k1, (batch_size, num_heads, seq_len, head_dim), dtype=dtype
    )
    if is_mqa:
      k = random.uniform(k2, (batch_size, seq_len, head_dim), dtype=dtype)
      v = random.uniform(k3, (batch_size, seq_len, head_dim), dtype=dtype)
    else:
      k = random.uniform(
          k2, (batch_size, num_heads, seq_len, head_dim), dtype=dtype
      )
      v = random.uniform(
          k3, (batch_size, num_heads, seq_len, head_dim), dtype=dtype
      )
    do = random.uniform(k4, q.shape, dtype=dtype)
    mask = mask_lib.FullMask(_shape=(q.shape[2], k.shape[2]))
    if mask == "FULL CAUSAL":
      mask = mask_lib.make_causal_mask((q.shape[2], k.shape[2]))
    local_segment_ids = None
    local_q_global_kv_segment_ids = None
    if padding_factor > 0:
      local_token_idx = jax.lax.broadcasted_iota(
          jnp.int32, (seq_shard + seq_shard_pad,), 0
      )
      local_segment_tokens = (local_token_idx < seq_shard).astype(jnp.int32)
      local_segment_ids = splash.SegmentIds(
          q=local_segment_tokens, kv=local_segment_tokens
      )

      global_kv_segment_tokens = jnp.tile(local_segment_tokens, q_seq_shards)
      local_q_global_kv_segment_ids = splash.SegmentIds(
          q=local_segment_tokens, kv=global_kv_segment_tokens
      )

    # For ring attention, sequence dimension is sharded over 'ring' axis
    q_spec = P(
        None,  # batch
        "heads" if head_shards > 1 else None,
        "ring",
        None,
    )
    # K and V must also be sharded along the 'ring' axis for ring attention
    if is_mqa:
      kv_spec = P(
          None,  # batch
          "ring",
          None,
      )
    else:
      kv_spec = q_spec

    # Splash Config and initialize splash kernel for reference kernel
    splash_config = splash.SplashConfig(
        block_q=128,
        block_kv=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        use_fused_bwd_kernel=fused_bwd,
        use_base2_exp=False,
        # fuse_reciprocal=False,
        # TODO: Change fuse_reciprocal behavior for ring attention
        # so we do the reciprocal after ring
    )
    make_splash = splash.make_splash_mqa if is_mqa else splash.make_splash_mha
    seq_parallel_kernel = make_splash(
        mask, config=splash_config, q_seq_shards=1
    )

    # Shardmap reference kernel for splash attention
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(q_spec, P(), P()),
        out_specs=q_spec,
        check_vma=False,
    )
    def shardmap_fn_ref(q, k, v):
      vmapped_kernel = jax.vmap(seq_parallel_kernel, in_axes=(0, 0, 0, None))
      o = vmapped_kernel(q, k, v, local_q_global_kv_segment_ids)
      return o

    # Ring kernel for splash attention
    ring_kernel = ring_attention_kernel.make_ring_attention(
        mask,
        is_mqa=is_mqa,
        ring_axis="ring",
        config=splash_config,
        save_residuals=False,
    )

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
        ),
        out_specs=q_spec,
        check_vma=False,
    )
    def shardmap_fn(q, k, v):
      vmapped_kernel = jax.vmap(ring_kernel, in_axes=(0, 0, 0, None))
      o = vmapped_kernel(q, k, v, local_segment_ids)
      jax.block_until_ready(o)
      return o

    # Run forward pass and assert close to reference kernel
    a = shardmap_fn(q, k, v)
    jax.block_until_ready(a)
    b = shardmap_fn_ref(q, k, v)
    jax.block_until_ready(b)
    self._assert_allclose(a, b, rtol=5e-3, atol=3e-3)

    # Ring attention backward pass wrapped in shardmap
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            q_spec,
        ),
        out_specs=(q_spec, q_spec, kv_spec, kv_spec),
        check_vma=False,
    )
    def backward_fn(q, k, v, do):
      def vmap_vjp(q, k, v, do):
        out_original, out_vjp_original = jax.vjp(ring_kernel, q, k, v)
        out_vjp_res = out_vjp_original(do)
        return out_original, out_vjp_res[0], out_vjp_res[1], out_vjp_res[2]

      vmap_vjp_res = jax.vmap(
          vmap_vjp, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0)
      )
      out_original, dq, dk, dv = vmap_vjp_res(q, k, v, do)
      return out_original, dq, dk, dv

    # Reference splash attention backward pass wrapped in shardmap

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(), P(), P(), P()),  # k  # v
        out_specs=(
            P(),  # out_original
            P(),  # dq
            P(),  # dk
            P(),  # dv
        ),
        check_vma=False,
    )
    def backward_fn_ref(q, k, v, do):
      def vmap_vjp(q, k, v, do):
        out_original, out_vjp_original = jax.vjp(seq_parallel_kernel, q, k, v)
        out_vjp_res = out_vjp_original(do)
        return out_original, out_vjp_res[0], out_vjp_res[1], out_vjp_res[2]

      vmap_vjp_res = jax.vmap(
          vmap_vjp, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0)
      )
      out_original, dq, dk, dv = vmap_vjp_res(q, k, v, do)
      return out_original, dq, dk, dv

    # Run backward pass and assert close to reference backward pass
    o_original, dq, dk, dv = backward_fn(q, k, v, do.astype(jnp.bfloat16))

    o_splash, dq_splash_ref, dk_splash_ref, dv_splash_ref = backward_fn_ref(
        q, k, v, do.astype(jnp.bfloat16)
    )
    jax.block_until_ready(dq_splash_ref)
    jax.block_until_ready(dk_splash_ref)
    jax.block_until_ready(dv_splash_ref)
    self._assert_allclose(o_splash, o_original, rtol=7e-2, atol=7e-2)
    self._assert_allclose(dq, dq_splash_ref, rtol=7e-2, atol=7e-2)
    self._assert_allclose(dk, dk_splash_ref, rtol=7e-2, atol=7e-2)
    self._assert_allclose(dv, dv_splash_ref, rtol=7e-2, atol=7e-2)


if __name__ == "__main__":
  absltest.main()
