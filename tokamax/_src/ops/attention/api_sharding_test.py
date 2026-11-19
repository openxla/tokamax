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
import dataclasses
import functools
import typing

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops.attention import api
from tokamax._src.ops.attention import base as attn_base

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec


class ApiShardingTest(parameterized.TestCase):

  @parameterized.product(
      implementation=typing.get_args(api.Implementation.__value__),
      shard_axis=[0, 2],
  )
  def test_dot_product_attention_sharding(self, implementation, shard_axis):
    if (
        implementation == 'mosaic_gpu'
        and not gpu_utils.has_mosaic_gpu_support()
    ):
      self.skipTest('Mosaic not supported on this platform.')
    if implementation == 'triton' and not gpu_utils.has_triton_support():
      self.skipTest('Triton not supported on this platform.')
    if implementation == 'cudnn' and jax.default_backend() != 'gpu':
      self.skipTest('cuDNN only supported on GPU')
    if implementation == 'mosaic_tpu' and jax.default_backend() != 'tpu':
      self.skipTest('Mosaic TPU only supported on TPU')

    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices for sharding.')

    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names='x')
    spec: list[str | None] = [None, None, None, None]
    spec[shard_axis] = 'x'
    sharding = NamedSharding(mesh, PartitionSpec(*spec))

    dtype = jnp.bfloat16
    b, s, t, n, h = jax.device_count(), 512, 512, jax.device_count() * 2, 64
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(keys[0], (b, t, n, h), dtype)
    k = jax.random.normal(keys[1], (b, s, n, h), dtype)
    v = jax.random.normal(keys[2], (b, s, n, h), dtype)

    sdpa_ref = functools.partial(
        jax.nn.dot_product_attention, implementation='xla'
    )
    sdpa_ans = functools.partial(
        api.dot_product_attention, implementation=implementation
    )

    out_ref = sdpa_ref(q, k, v)

    @functools.partial(
        jax.jit,
        in_shardings=(sharding, sharding, sharding),
        out_shardings=sharding,
    )
    def f(q, k, v):
      return sdpa_ans(q, k, v, q_sharding=sharding)

    q_sharded = jax.device_put(q, sharding)
    k_sharded = jax.device_put(k, sharding)
    v_sharded = jax.device_put(v, sharding)
    out_ans = f(q_sharded, k_sharded, v_sharded)
    self.assertEqual(out_ans.sharding, sharding)
    chex.assert_trees_all_close(out_ans, out_ref, atol=0.01, rtol=0.01)

  @parameterized.parameters(
      ('causal', True, None),
      ('sliding_window', True, (64, 0)),
      ('full', False, None),
  )
  def test_tpu_q_seq_and_k_sharding_vjp(self, _, is_causal, local_window_size):
    if jax.default_backend() != 'tpu' or jax.device_count() < 2:
      self.skipTest('Requires at least 2 TPU devices.')

    num_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names='x')
    q_sharding = NamedSharding(mesh, PartitionSpec(None, 'x', None, None))
    k_sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None))
    mlv_sharding = NamedSharding(mesh, PartitionSpec(None, None))

    dtype = jnp.bfloat16
    b, t, s, n_q, n_kv, h = 2, num_devices * 128, num_devices * 128, 4, 2, 64
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    q = jax.random.normal(keys[0], (b, t, n_q, h), dtype)
    k = jax.random.normal(keys[1], (b, s, n_kv, h), dtype)
    v = jax.random.normal(keys[2], (b, s, n_kv, h), dtype)
    dout = jax.random.normal(keys[3], (b, t, n_q, h), dtype)
    mlv = jnp.broadcast_to(
        jnp.array([8.0, 9.0, 10.0, 11.0], dtype=jnp.float32), (b, n_q)
    )

    def ref_fn(q_, k_, v_):
      k_rep = jnp.repeat(k_, n_q // n_kv, axis=2)
      v_rep = jnp.repeat(v_, n_q // n_kv, axis=2)
      return jax.nn.dot_product_attention(
          q_,
          k_rep,
          v_rep,
          is_causal=is_causal,
          local_window_size=local_window_size,
          implementation='xla',
      )

    out_ref, vjp_ref = jax.vjp(ref_fn, q, k, v)
    dq_ref, dk_ref, dv_ref = vjp_ref(dout)

    @functools.partial(
        jax.jit,
        in_shardings=(
            q_sharding,
            k_sharding,
            k_sharding,
            q_sharding,
            mlv_sharding,
        ),
        out_shardings=(q_sharding, (q_sharding, k_sharding, k_sharding)),
    )
    def sharded_fn(q_, k_, v_, dout_, mlv_):
      out_, vjp_fn = jax.vjp(
          lambda qq, kk, vv: api.dot_product_attention(
              qq,
              kk,
              vv,
              is_causal=is_causal,
              local_window_size=local_window_size,
              max_logit_value=mlv_,
              implementation='mosaic_tpu',
              q_sharding=q_sharding,
              k_sharding=k_sharding,
          ),
          q_,
          k_,
          v_,
      )
      return out_, vjp_fn(dout_)

    out_ans, (dq_ans, dk_ans, dv_ans) = sharded_fn(
        jax.device_put(q, q_sharding),
        jax.device_put(k, k_sharding),
        jax.device_put(v, k_sharding),
        jax.device_put(dout, q_sharding),
        jax.device_put(mlv, mlv_sharding),
    )
    chex.assert_trees_all_close(out_ans, out_ref, atol=0.02, rtol=0.02)
    chex.assert_trees_all_close(dq_ans, dq_ref, atol=0.04, rtol=0.04)
    chex.assert_trees_all_close(dk_ans, dk_ref, atol=0.04, rtol=0.04)
    chex.assert_trees_all_close(dv_ans, dv_ref, atol=0.03, rtol=0.03)

    # Head-sharded non-shift-invariant check: verify per-(batch, head)
    # `max_logit_value` alignment across TPU head shards.
    head_mesh = Mesh(mesh_utils.create_device_mesh((2,)), axis_names='x')
    q_head_sharding = NamedSharding(
        head_mesh, PartitionSpec(None, None, 'x', None)
    )
    k_head_sharding = NamedSharding(
        head_mesh, PartitionSpec(None, None, 'x', None)
    )
    mlv_head_sharding = NamedSharding(head_mesh, PartitionSpec(None, 'x'))
    mlv_bad = jnp.array(
        [[8.0, 9.0, -100.0, 11.0], [-100.0, 9.0, 10.0, 11.0]],
        dtype=jnp.float32,
    )

    @functools.partial(
        jax.jit,
        in_shardings=(
            q_head_sharding,
            k_head_sharding,
            k_head_sharding,
            mlv_head_sharding,
        ),
        out_shardings=q_head_sharding,
    )
    def head_sharded_fn(q_, k_, v_, mlv_):
      return api.dot_product_attention(
          q_,
          k_,
          v_,
          is_causal=is_causal,
          local_window_size=local_window_size,
          max_logit_value=mlv_,
          implementation='mosaic_tpu',
          q_sharding=q_head_sharding,
          k_sharding=k_head_sharding,
      )

    out_bad = head_sharded_fn(
        jax.device_put(q, q_head_sharding),
        jax.device_put(k, k_head_sharding),
        jax.device_put(v, k_head_sharding),
        jax.device_put(mlv_bad, mlv_head_sharding),
    )
    self.assertFalse(bool(jnp.all(jnp.isfinite(out_bad[0, :, 2, :]))))
    self.assertFalse(bool(jnp.all(jnp.isfinite(out_bad[1, :, 0, :]))))
    chex.assert_trees_all_close(
        out_bad[0, :, jnp.array([0, 1, 3]), :],
        out_ref[0, :, jnp.array([0, 1, 3]), :],
        atol=0.02,
        rtol=0.02,
    )
    chex.assert_trees_all_close(
        out_bad[1, :, jnp.array([1, 2, 3]), :],
        out_ref[1, :, jnp.array([1, 2, 3]), :],
        atol=0.02,
        rtol=0.02,
    )

  def test_k_sharding_without_q_sharding_raises(self):
    devices = jax.devices()[:1]
    mesh = Mesh(devices, ('x',))
    k_sharding = NamedSharding(mesh, PartitionSpec('x', None, None, None))
    q = jnp.ones((2, 128, 4, 16), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, 'q_sharding.*must be specified'):
      api.dot_product_attention(q, q, q, k_sharding=k_sharding)

  def test_k_sharding_uses_k_batch_axes(self):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices for sharding.')

    num_devices = jax.device_count()
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names='x')
    q_sharding = NamedSharding(mesh, PartitionSpec('x', None, None, None))
    k_sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None))
    num_pages = 3  # Intentionally indivisible by `num_devices`.

    @dataclasses.dataclass(frozen=True)
    class _PagedBatchShardingOp(
        attn_base.DotProductAttention[None, typing.Any]
    ):
      expected_q_batch: int = 1
      expected_k_batch: int = num_pages

      @typing.override
      def _fwd(
          self,
          q: jax.Array | attn_base.QArray,
          k: jax.Array | attn_base.QArray,
          v: jax.Array | attn_base.QArray,
          *,
          precision: tuple[
              attn_base.CanonicalPrecision, attn_base.CanonicalPrecision
          ],
          logits_dtype: jnp.dtype,
          logits_scale: float,
          bias: jax.Array | None,
          logits_soft_cap: float | None,
          max_logit_value: float | jax.Array | None = None,
          mask: attn_base.Mask,
          dropout_mask: jax.Array | None,
          dropout_rate: float,
          paging_info: attn_base.PagingInfo | None,
          q_indices: jax.Array | None,
          k_indices: jax.Array | None,
          normalize_output: bool,
          return_residuals: bool,
          config: None,
      ) -> tuple[jax.Array, attn_base.Residuals | None]:
        if (
            q.shape[0] != self.expected_q_batch
            or k.shape[0] != self.expected_k_batch
            or v.shape[0] != self.expected_k_batch
            or k_indices is None
            or k_indices.shape[0] != self.expected_k_batch
            or paging_info is None
            or paging_info.lengths.shape[0] != self.expected_k_batch
        ):
          raise AssertionError(
              f'Unexpected local shapes: q={q.shape}, k={k.shape}, '
              f'v={v.shape},'
              f' k_indices={None if k_indices is None else k_indices.shape}, '
              f'lengths={None if paging_info is None else paging_info.lengths.shape}'
          )
        page_idx = paging_info.active_page_indices[:, 0]
        return attn_base.DotProductAttention._fwd(
            self,
            q,
            k[page_idx],
            v[page_idx],
            precision=precision,
            logits_dtype=logits_dtype,
            logits_scale=logits_scale,
            bias=bias,
            logits_soft_cap=logits_soft_cap,
            max_logit_value=max_logit_value,
            mask=mask,
            dropout_mask=dropout_mask,
            dropout_rate=dropout_rate,
            paging_info=None,
            q_indices=q_indices,
            k_indices=k_indices[page_idx],
            normalize_output=normalize_output,
            return_residuals=return_residuals,
            config=config,
        )

    op_ref = _PagedBatchShardingOp(expected_q_batch=num_devices)
    op_sharded = _PagedBatchShardingOp(expected_q_batch=1)
    keys = jax.random.split(jax.random.PRNGKey(11), 3)
    seq_len, n_q, n_kv, head_dim = 16, 4, 4, 8
    q = jax.random.normal(
        keys[0], (num_devices, seq_len, n_q, head_dim), jnp.float32
    )
    k = jax.random.normal(
        keys[1], (num_pages, seq_len, n_kv, head_dim), jnp.float32
    )
    v = jax.random.normal(
        keys[2], (num_pages, seq_len, n_kv, head_dim), jnp.float32
    )
    k_start = jnp.zeros((num_devices, n_kv, seq_len), dtype=jnp.int32)
    mask = attn_base.Mask(k_start=k_start)
    k_indices = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, None, :],
        (num_pages, n_kv, seq_len),
    )
    paging_info = attn_base.PagingInfo(
        num_active_pages=jnp.ones((num_devices,), dtype=jnp.int32),
        active_page_indices=(
            jnp.arange(num_devices, dtype=jnp.int32) % num_pages
        )[:, None],
        lengths=jnp.full((num_pages,), seq_len, dtype=jnp.int32),
    )

    out_ref = op_ref(
        q,
        k,
        v,
        mask=mask,
        paging_info=paging_info,
        k_indices=k_indices,
    )

    mask_sharded = attn_base.Mask(
        k_start=jax.device_put(
            k_start, NamedSharding(mesh, PartitionSpec('x', None, None))
        )
    )
    paging_sharded = attn_base.PagingInfo(
        num_active_pages=jax.device_put(
            paging_info.num_active_pages,
            NamedSharding(mesh, PartitionSpec('x')),
        ),
        active_page_indices=jax.device_put(
            paging_info.active_page_indices,
            NamedSharding(mesh, PartitionSpec('x', None)),
        ),
        lengths=jax.device_put(
            paging_info.lengths, NamedSharding(mesh, PartitionSpec(None))
        ),
    )
    k_indices_sharding = NamedSharding(mesh, PartitionSpec(None, None, None))

    @functools.partial(jax.jit, out_shardings=q_sharding)
    def run_sharded(q_, k_, v_, mask_, paging_info_, k_indices_):
      return op_sharded(
          q_,
          k_,
          v_,
          mask=mask_,
          paging_info=paging_info_,
          k_indices=k_indices_,
          q_sharding=q_sharding,
          k_sharding=k_sharding,
      )

    out_ans = run_sharded(
        jax.device_put(q, q_sharding),
        jax.device_put(k, k_sharding),
        jax.device_put(v, k_sharding),
        mask_sharded,
        paging_sharded,
        jax.device_put(k_indices, k_indices_sharding),
    )
    chex.assert_trees_all_close(out_ans, out_ref, atol=1e-5, rtol=1e-5)

    # Also verify the reverse case: replicated `q` with batch-sharded `k`.
    q_rep_sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None))
    k_batch_sharding = NamedSharding(mesh, PartitionSpec('x', None, None, None))
    op_k_sharded = _PagedBatchShardingOp(
        expected_q_batch=num_pages, expected_k_batch=1
    )
    q_rep = jax.device_put(
        jax.random.normal(
            keys[0], (num_pages, seq_len, n_q, head_dim), jnp.float32
        ),
        q_rep_sharding,
    )
    k_sh = jax.device_put(
        jax.random.normal(
            keys[1], (num_devices, seq_len, n_kv, head_dim), jnp.float32
        ),
        k_batch_sharding,
    )
    v_sh = jax.device_put(
        jax.random.normal(
            keys[2], (num_devices, seq_len, n_kv, head_dim), jnp.float32
        ),
        k_batch_sharding,
    )
    paging_k_sharded = attn_base.PagingInfo(
        num_active_pages=jax.device_put(
            jnp.ones((num_pages,), dtype=jnp.int32),
            NamedSharding(mesh, PartitionSpec(None)),
        ),
        active_page_indices=jax.device_put(
            jnp.zeros((num_pages, 1), dtype=jnp.int32),
            NamedSharding(mesh, PartitionSpec(None, None)),
        ),
        lengths=jax.device_put(
            jnp.full((num_devices,), seq_len, dtype=jnp.int32),
            NamedSharding(mesh, PartitionSpec('x')),
        ),
    )
    k_indices_sh = jax.device_put(
        jnp.broadcast_to(
            jnp.arange(seq_len, dtype=jnp.int32)[None, None, :],
            (num_devices, n_kv, seq_len),
        ),
        NamedSharding(mesh, PartitionSpec('x', None, None)),
    )
    out_rev = jax.jit(
        lambda q_, k_, v_, p_, ki_: op_k_sharded(
            q_,
            k_,
            v_,
            paging_info=p_,
            k_indices=ki_,
            q_sharding=q_rep_sharding,
            k_sharding=k_batch_sharding,
        ),
        out_shardings=q_rep_sharding,
    )(q_rep, k_sh, v_sh, paging_k_sharded, k_indices_sh)
    self.assertEqual(out_rev.shape, q_rep.shape)

  def test_tpu_fold_q_seq_heads_q_seq_sharding_vjp(self):
    if jax.default_backend() != 'tpu' or jax.device_count() < 2:
      self.skipTest('Requires at least 2 TPU devices.')

    num_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names='x')
    q_sharding = NamedSharding(mesh, PartitionSpec(None, 'x', None, None))
    k_sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None))

    tpu_impl: typing.Any = api.IMPLEMENTATIONS['mosaic_tpu']
    seq_minor = tpu_impl.vjp.q_layout.__class__.SEQ_MINOR
    folded_cfg = tpu_impl.config_cls(
        block_q=256,
        block_kv=128,
        block_kv_compute=128,
        q_layout=seq_minor,
        k_layout=seq_minor,
        v_layout=seq_minor,
        use_experimental_scheduler=True,
        fold_q_seq_heads=True,
    )
    folded_impl = tpu_impl.replace(config=folded_cfg)

    dtype = jnp.bfloat16
    b, t, s, n_q, n_kv, h = 2, num_devices * 128, num_devices * 128, 4, 2, 64
    keys = jax.random.split(jax.random.PRNGKey(13), 4)
    q = jax.random.normal(keys[0], (b, t, n_q, h), dtype)
    k = jax.random.normal(keys[1], (b, s, n_kv, h), dtype)
    v = jax.random.normal(keys[2], (b, s, n_kv, h), dtype)
    dout = jax.random.normal(keys[3], (b, t, n_q, h), dtype)

    def ref_fn(q_, k_, v_):
      k_rep = jnp.repeat(k_, n_q // n_kv, axis=2)
      v_rep = jnp.repeat(v_, n_q // n_kv, axis=2)
      return jax.nn.dot_product_attention(
          q_,
          k_rep,
          v_rep,
          is_causal=True,
          local_window_size=(64, 0),
          implementation='xla',
      )

    out_ref, vjp_ref = jax.vjp(ref_fn, q, k, v)
    dq_ref, dk_ref, dv_ref = vjp_ref(dout)

    @functools.partial(
        jax.jit,
        in_shardings=(q_sharding, k_sharding, k_sharding, q_sharding),
        out_shardings=(q_sharding, (q_sharding, k_sharding, k_sharding)),
    )
    def sharded_fn(q_, k_, v_, dout_):
      out_, vjp_fn = jax.vjp(
          lambda qq, kk, vv: api.dot_product_attention(
              qq,
              kk,
              vv,
              is_causal=True,
              local_window_size=(64, 0),
              max_logit_value=10.0,
              implementation=folded_impl,
              q_sharding=q_sharding,
              k_sharding=k_sharding,
          ),
          q_,
          k_,
          v_,
      )
      return out_, vjp_fn(dout_)

    out_ans, (dq_ans, dk_ans, dv_ans) = sharded_fn(
        jax.device_put(q, q_sharding),
        jax.device_put(k, k_sharding),
        jax.device_put(v, k_sharding),
        jax.device_put(dout, q_sharding),
    )
    chex.assert_trees_all_close(out_ans, out_ref, atol=0.02, rtol=0.02)
    chex.assert_trees_all_close(dq_ans, dq_ref, atol=0.04, rtol=0.04)
    chex.assert_trees_all_close(dk_ans, dk_ref, atol=0.04, rtol=0.04)
    chex.assert_trees_all_close(dv_ans, dv_ref, atol=0.03, rtol=0.03)


if __name__ == '__main__':
  absltest.main()
