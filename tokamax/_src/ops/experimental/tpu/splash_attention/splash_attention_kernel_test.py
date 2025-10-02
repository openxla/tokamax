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

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import math
from typing import Any, TypeVar

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_test_utils as test_utils
from tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask_info import process_mask


jax.config.parse_flags_with_absl()

hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=15,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile(name="deterministic")

partial = functools.partial
Draw = TypeVar("Draw", bound=Callable[[hps.SearchStrategy[Any]], Any])

LOG2E = math.log2(math.e)


@hps.composite
def segment_ids_strategy(draw, seq_len: int) -> splash.SegmentIds:
  boundaries = hps.sets(hps.integers(1, seq_len - 1), min_size=1, max_size=4)
  bounds = sorted(draw(boundaries))
  ids_array = np.empty((seq_len,), dtype=np.int32)
  for i, (start, end) in enumerate(zip((0, *bounds), (*bounds, seq_len))):
    # Not sure why, but short segments can trip things up
    if end - start < 2:
      end = start + 2
    ids_array[start:end] = i
  return splash.SegmentIds(ids_array, ids_array)


def seed_strategy() -> hps.SearchStrategy[int]:
  return hps.integers(min_value=0, max_value=4)


class Mask:

  def get_mask(self) -> mask_lib.Mask:
    raise NotImplementedError()


def full_mask_strategy(
    q_seq_len: int, kv_seq_len: int
) -> hps.SearchStrategy[Mask]:
  return hps.just(FullMask(q_seq_len, kv_seq_len))


@dataclasses.dataclass
class FullMask(Mask):
  q_seq_len: int
  kv_seq_len: int

  def get_mask(self) -> mask_lib.Mask:
    return mask_lib.FullMask((self.q_seq_len, self.kv_seq_len))


def causal_mask_strategy(
    q_seq_len: int, kv_seq_len: int
) -> hps.SearchStrategy[Mask]:
  return hps.just(CausalMask(q_seq_len, kv_seq_len))


@dataclasses.dataclass
class CausalMask(Mask):
  q_seq_len: int
  kv_seq_len: int

  def get_mask(self) -> mask_lib.Mask:
    return mask_lib.CausalMask((self.q_seq_len, self.kv_seq_len))


@dataclasses.dataclass
class LocalAttentionMask(Mask):
  seq_len: int
  left: int | None
  right: int | None
  offset: int

  def get_mask(self) -> mask_lib.Mask:
    mask = mask_lib.LocalMask(
        (self.seq_len, self.seq_len),
        (self.left, self.right),
        offset=self.offset,
    )
    # Make sure that no row is full of zeros as this is leads to undefined
    # softmax.
    diagonal = mask_lib.NumpyMask(np.identity(self.seq_len, dtype=np.bool_))
    return mask | diagonal


@hps.composite
def local_attention_mask_strategy(draw: Draw, seq_len: int) -> Mask:
  left_window = draw(
      hps.one_of(hps.none(), hps.integers(min_value=0, max_value=seq_len))
  )
  right_window = draw(
      hps.one_of(hps.none(), hps.integers(min_value=0, max_value=seq_len))
  )
  offset = draw(hps.integers(min_value=-seq_len, max_value=seq_len - 1))
  return LocalAttentionMask(seq_len, left_window, right_window, offset=offset)


@dataclasses.dataclass
class RandomMask(Mask):
  q_seq_len: int
  kv_seq_len: int
  sparsity: float
  seed: int

  def get_mask(self) -> mask_lib.Mask:
    mask = mask_lib.make_random_mask(
        (self.q_seq_len, self.kv_seq_len), self.sparsity, self.seed
    )
    # Make sure that no row is full of zeros as this is leads to undefined
    # softmax.
    mask[:, 0] = True

    return mask_lib.NumpyMask(mask)


@hps.composite
def random_mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int) -> Mask:
  rand = draw(hps.randoms())
  seed = rand.randint(0, 2**32 - 1)
  sparsity = rand.uniform(0.01, 0.5)
  return RandomMask(q_seq_len, kv_seq_len, sparsity, seed)


@dataclasses.dataclass
class ComposeMask(Mask):
  left: Mask
  right: Mask
  op: Callable[[mask_lib.Mask, mask_lib.Mask], mask_lib.Mask]

  def get_mask(self) -> mask_lib.Mask:
    return self.op(self.left.get_mask(), self.right.get_mask())


@hps.composite
def compose_mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int) -> Mask:
  mask1 = draw(mask_strategy(q_seq_len, kv_seq_len))
  mask2 = draw(mask_strategy(q_seq_len, kv_seq_len))
  op = draw(
      hps.one_of(hps.just(mask_lib.LogicalOr), hps.just(mask_lib.LogicalAnd))
  )
  return ComposeMask(mask1, mask2, op)


@hps.composite
def mask_strategy(draw: Draw, q_seq_len: int, kv_seq_len: int) -> Mask:
  oneof = [
      causal_mask_strategy(q_seq_len, kv_seq_len),
      full_mask_strategy(q_seq_len, kv_seq_len),
      random_mask_strategy(q_seq_len, kv_seq_len),
      # TODO Composing masks creates masks that produce minor numerical
      # differences. We should investigate this in the future.
      # compose_mask_strategy(q_seq_len, kv_seq_len),
  ]

  if q_seq_len == kv_seq_len:
    oneof.append(local_attention_mask_strategy(q_seq_len))

  return draw(hps.one_of(oneof))


@hps.composite
def sequence_length_strategy(draw: Draw) -> tuple[int, int]:
  q_seq_len = draw(hps.sampled_from([1024, 2048, 4096]))
  kv_seq_len = draw(hps.sampled_from([1024, 2048, 4096]))
  return q_seq_len, kv_seq_len


@hps.composite
def attention_strategy(draw: Draw) -> tuple[int, int, int, int, np.dtype]:
  q_seq_len, kv_seq_len = draw(sequence_length_strategy())
  head_dim_qk, head_dim_v = draw(
      hps.sampled_from([(128, 128), (256, 256), (192, 128)])
  )
  if q_seq_len >= 4096 and kv_seq_len >= 4096:
    # Do not draw bfloat16 on longer sequence lengths, as this increases
    # the risk of numerical precision errors causing false positives in
    # tests.
    dtype = np.dtype("float32")
  else:
    dtype = draw(hps.sampled_from([np.dtype("float32"), np.dtype(jnp.bfloat16)]))
  return q_seq_len, kv_seq_len, head_dim_qk, head_dim_v, dtype


@hps.composite
def mha_strategy(draw: Draw) -> tuple[int, int, int, int, int, int, np.dtype]:
  q_seq_len, kv_seq_len, head_dim_qk, head_dim_v, dtype = draw(
      attention_strategy()
  )
  num_q_heads, num_kv_heads = draw(
      hps.sampled_from([(1, 1), (2, 2), (4, 1), (8, 4), (6, 2)])
  )
  return (
      q_seq_len,
      kv_seq_len,
      num_q_heads,
      num_kv_heads,
      head_dim_qk,
      head_dim_v,
      dtype,
  )


@hps.composite
def block_sizes_strategy(
    draw: Draw,
    q_seq_len: int,
    kv_seq_len: int,
    include_bwd_blocks: bool = False,
    use_fused_bwd_kernel: bool = False,
) -> splash.SplashConfig:
  all_block_shapes = [128, 256, 512]
  q_layout = draw(hps.sampled_from(splash.QKVLayout))
  k_layout = draw(hps.sampled_from(splash.QKVLayout))
  v_layout = draw(hps.sampled_from(splash.QKVLayout))
  layouts = dict(q_layout=q_layout, k_layout=k_layout, v_layout=v_layout)
  q_valid_block_shapes = [bs for bs in all_block_shapes if bs <= q_seq_len]
  kv_valid_block_shapes = [bs for bs in all_block_shapes if bs <= kv_seq_len]
  bq, bkv = (
      draw(hps.sampled_from(q_valid_block_shapes)),
      draw(hps.sampled_from(kv_valid_block_shapes)),
  )
  bkv_compute = draw(
      hps.sampled_from([None, *[b for b in kv_valid_block_shapes if b <= bkv]])
  )
  if not include_bwd_blocks:
    return splash.SplashConfig(
        block_q=bq, block_kv=bkv, block_kv_compute=bkv_compute, **layouts
    )
  all_block_shapes = [128, 256]
  q_valid_block_shapes = [bs for bs in all_block_shapes if bs <= q_seq_len]
  kv_valid_block_shapes = [bs for bs in all_block_shapes if bs <= kv_seq_len]
  bq_dkv, bkv_dkv = (
      draw(hps.sampled_from(q_valid_block_shapes)),
      draw(hps.sampled_from(kv_valid_block_shapes)),
  )
  if use_fused_bwd_kernel:
    bq_dq, bkv_dq = None, None
  else:
    bq_dq = draw(hps.sampled_from(q_valid_block_shapes))
    bkv_dq = draw(hps.sampled_from(kv_valid_block_shapes))
  block_kv_dkv_compute = draw(
      hps.sampled_from(
          [None, *[b for b in kv_valid_block_shapes if b <= bkv_dkv]]
      )
  )
  return splash.SplashConfig(
      block_q=bq,
      block_kv=bkv,
      block_kv_compute=bkv_compute,
      block_q_dkv=bq_dkv,
      block_kv_dkv=bkv_dkv,
      block_kv_dkv_compute=block_kv_dkv_compute,
      block_q_dq=bq_dq,
      block_kv_dq=bkv_dq,
      use_fused_bwd_kernel=use_fused_bwd_kernel,
      **layouts,
  )


def attn_logits_soft_cap_strategy() -> hps.SearchStrategy[float | None]:
  return hps.one_of(hps.just(None), hps.floats(min_value=1.0, max_value=50.0))


@test_utils.thread_unsafe_test_class()  # hypothesis is not thread safe
class SplashAttentionTest(test_utils.SplashAttentionTestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.product(
      is_mqa=(False, True),
      is_segmented=(False, True),
      is_dynamic_mask=(False, True),
  )
  @hp.given(hps.data())
  def test_splash_attention(self, is_mqa, is_segmented, is_dynamic_mask, data):
    # TODO: Re-enable once dynamic masks are fixed.
    if is_dynamic_mask:
      self.skipTest("Dynamic masks not supported.")

    seed = data.draw(seed_strategy())
    key = random.key(seed)
    k1, k2, k3 = random.split(key, 3)

    (
        q_seq_len,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        dtype,
    ) = data.draw(mha_strategy())

    # Avoid segment ids for rectangular matrices, as its hard to enforce
    # valid masks (non-0 rows).
    hp.assume(q_seq_len == kv_seq_len or not is_segmented)

    q = random.uniform(k1, (num_q_heads, q_seq_len, head_dim_qk), dtype=dtype)
    if is_mqa:
      k = random.uniform(k2, (kv_seq_len, head_dim_qk), dtype=dtype)
      v = random.uniform(k3, (kv_seq_len, head_dim_v), dtype=dtype)
    else:
      k = random.uniform(
          k2, (num_kv_heads, kv_seq_len, head_dim_qk), dtype=dtype
      )
      v = random.uniform(
          k3, (num_kv_heads, kv_seq_len, head_dim_v), dtype=dtype
      )

    segment_ids = None
    if is_segmented:
      assert q_seq_len == kv_seq_len
      segment_ids = data.draw(segment_ids_strategy(q_seq_len))

    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask = data.draw(mask_strategy(q_seq_len, kv_seq_len)).get_mask()
    if is_dynamic_mask:
      mask = jnp.array(mask[:, :])
    config = data.draw(block_sizes_strategy(q_seq_len, kv_seq_len))
    config = dataclasses.replace(
        config,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=self.INTERPRET,
    )

    attn_ref = partial(splash.attention_reference, is_mqa=is_mqa)
    if is_mqa:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mqa_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mqa
    else:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mha_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mha

    make_mask_fn = partial(make_mask_fn, config=config)
    attn = make_mask_fn(mask)

    o = attn(q, k, v, segment_ids)
    o_ref = attn_ref(
        q.astype(np.float32),
        k.astype(np.float32),
        v.astype(np.float32),
        jnp.array(mask[:, :]),
        segment_ids,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    self._assert_allclose(o, o_ref, atol=6e-3, rtol=3e-3)

  @parameterized.product(
      is_mqa=(False, True),
      is_segmented=(False, True),
      is_dynamic_mask=(False, True),
      use_base2_exp=(False, True),
      use_max_logit_estimate=(None, "const", "value_1d", "value_2d"),
      fuse_reciprocal=(True, False),
  )
  @hp.given(hps.data())
  def test_splash_attention_fwd(self, is_mqa, is_segmented, is_dynamic_mask,
                                use_base2_exp, use_max_logit_estimate,
                                fuse_reciprocal, data):
    # TODO: Re-enable once dynamic masks are fixed.
    if is_dynamic_mask:
      self.skipTest("Dynamic masks not supported.")

    seed = data.draw(seed_strategy())
    key = random.key(seed)
    k1, k2, k3 = random.split(key, 3)

    (
        q_seq_len,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        dtype,
    ) = data.draw(mha_strategy())

    # Avoid segment ids for rectangular matrices, as its hard to enforce
    # valid masks (non-0 rows).
    hp.assume(q_seq_len == kv_seq_len or not is_segmented)

    q = random.uniform(k1, (num_q_heads, q_seq_len, head_dim_qk), dtype=dtype)
    if is_mqa:
      k = random.uniform(k2, (kv_seq_len, head_dim_qk), dtype=dtype)
      v = random.uniform(k3, (kv_seq_len, head_dim_v), dtype=dtype)
    else:
      k = random.uniform(
          k2, (num_kv_heads, kv_seq_len, head_dim_qk), dtype=dtype
      )
      v = random.uniform(
          k3, (num_kv_heads, kv_seq_len, head_dim_v), dtype=dtype
      )

    segment_ids = None
    if is_segmented:
      assert q_seq_len == kv_seq_len
      segment_ids = data.draw(segment_ids_strategy(q_seq_len))
    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask = data.draw(mask_strategy(q_seq_len, kv_seq_len)).get_mask()
    if is_dynamic_mask:
      mask = jnp.array(mask[:, :])
    config = data.draw(block_sizes_strategy(q_seq_len, kv_seq_len))
    if is_mqa:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mqa_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mqa
    else:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mha_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mha

    config = dataclasses.replace(
        config,
        fuse_reciprocal=fuse_reciprocal,
        attn_logits_soft_cap=attn_logits_soft_cap,
        use_base2_exp=use_base2_exp,
        interpret=self.INTERPRET,
    )

    max_logit_value, max_val = None, 30.0
    if use_max_logit_estimate == "const":
      config = dataclasses.replace(config, max_logit_const=max_val)
    elif use_max_logit_estimate == "value_1d":
      max_logit_value = max_val * jnp.ones((1,), dtype=jnp.bfloat16)
    elif use_max_logit_estimate == "value_2d":
      max_logit_value = max_val * jnp.ones((num_q_heads,), dtype=jnp.bfloat16)

    make_mask_fn = partial(
        make_mask_fn, config=config, save_residuals=True
    )
    attn = make_mask_fn(mask)
    attn_ref = partial(
        splash.attention_reference,
        is_mqa=is_mqa,
        save_residuals=True,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    o, (logsumexp, max_logits) = attn(
        q, k, v, segment_ids, max_logit_value=max_logit_value
    )

    o_ref, (logsumexp_ref, max_logits_ref) = attn_ref(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        jnp.array(mask[:, :]),
        segment_ids,
    )
    if use_base2_exp:
      logsumexp_ref *= LOG2E
      max_logits_ref *= LOG2E

    if (use_base2_exp or use_max_logit_estimate is not None
        or not fuse_reciprocal):
      o_rtol, res_tol = dict(atol=8e-3, rtol=3e-3), dict(atol=1e-3, rtol=3e-3)
    else:
      o_rtol, res_tol = dict(atol=4e-3, rtol=3e-3), dict(atol=1e-3, rtol=3e-3)

    self._assert_allclose(o, o_ref, **o_rtol)
    self._assert_allclose(logsumexp, logsumexp_ref, **res_tol)
    if use_max_logit_estimate is None:
      self._assert_allclose(max_logits, max_logits_ref, **res_tol)

  @parameterized.product(
      is_mqa=(False, True),
      is_segmented=(False, True),
      downcast_smem_data=(False, True),
      use_fused_bwd_kernel=(False, True),
      is_dynamic_mask=(False, True),
      use_base2_exp=(False, True),
      use_max_logit_estimate=(None, "const", "value_1d", "value_2d"),
      fuse_reciprocal=(True, False),
  )
  @hp.given(hps.data())
  def test_splash_attention_bwd(
      self,
      is_mqa,
      is_segmented,
      downcast_smem_data,
      use_fused_bwd_kernel,
      is_dynamic_mask,
      use_base2_exp,
      use_max_logit_estimate,
      fuse_reciprocal,
      data,
  ):
    # TODO: Re-enable once dynamic masks are fixed.
    if is_dynamic_mask:
      self.skipTest("Dynamic masks not supported.")
    seed = data.draw(seed_strategy())
    key = random.key(seed)
    k1, k2, k3, k4 = random.split(key, 4)

    (
        q_seq_len,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        dtype,
    ) = data.draw(mha_strategy())

    # Avoid segment ids for rectangular matrices, as it's hard to enforce
    # valid masks (non-0 rows).
    hp.assume(q_seq_len == kv_seq_len or not is_segmented)

    q = random.uniform(k1, (num_q_heads, q_seq_len, head_dim_qk), dtype=dtype)
    if is_mqa:
      k = random.uniform(k2, (kv_seq_len, head_dim_qk), dtype=dtype)
      v = random.uniform(k3, (kv_seq_len, head_dim_v), dtype=dtype)
    else:
      k = random.uniform(
          k2, (num_kv_heads, kv_seq_len, head_dim_qk), dtype=dtype
      )
      v = random.uniform(
          k3, (num_kv_heads, kv_seq_len, head_dim_v), dtype=dtype
      )

    segment_ids = None
    if is_segmented:
      assert q_seq_len == kv_seq_len
      segment_ids = data.draw(segment_ids_strategy(q_seq_len))
    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask = data.draw(mask_strategy(q_seq_len, kv_seq_len)).get_mask()
    if is_dynamic_mask:
      mask = jnp.array(mask[:, :])
    config = data.draw(
        block_sizes_strategy(q_seq_len, kv_seq_len, include_bwd_blocks=True,
                             use_fused_bwd_kernel=use_fused_bwd_kernel)
    )

    config = dataclasses.replace(
        config,
        fuse_reciprocal=fuse_reciprocal,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=self.INTERPRET,
        use_base2_exp=use_base2_exp,
    )
    if is_mqa:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mqa_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mqa
    else:
      if not is_dynamic_mask:
        make_mask_fn = splash.make_splash_mha_single_device
      else:
        make_mask_fn = splash.make_dynamic_splash_mha

    max_logit_value, max_val = None, 30.0
    if use_max_logit_estimate == "const":
      config = dataclasses.replace(config, max_logit_const=max_val)
    elif use_max_logit_estimate == "value_1d":
      max_logit_value = max_val * jnp.ones((1,), dtype=jnp.bfloat16)
    elif use_max_logit_estimate == "value_2d":
      max_logit_value = max_val * jnp.ones((num_q_heads,), dtype=jnp.bfloat16)

    make_mask_fn = partial(
        make_mask_fn, config=config, downcast_smem_data=downcast_smem_data
    )
    attn = make_mask_fn(mask)

    o, attn_vjp = jax.vjp(partial(attn, max_logit_value=max_logit_value),
                          q, k, v, segment_ids)
    q32, k32, v32 = jax.tree.map(lambda x: x.astype(jnp.float32), (q, k, v))
    o_ref, (logsumexp, max_logits) = splash.attention_reference(
        q32,
        k32,
        v32,
        jnp.array(mask[:, :]),
        segment_ids,
        is_mqa=is_mqa,
        save_residuals=True,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    del max_logits
    atol = 1e-2 if (use_base2_exp or max_logit_value is not None) else 5e-3
    self._assert_allclose(o, o_ref, atol=atol, rtol=5e-3)

    do = random.uniform(k4, o.shape, dtype=o.dtype)
    dq, dk, dv, _ = attn_vjp(do)

    def bwd(
        mask, q, k, v, segment_ids, o, logsumexp, do
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      attn_ref = partial(
          splash._attention_reference_custom_bwd,
          backward_impl="flash",
          attn_logits_soft_cap=attn_logits_soft_cap,
      )
      dq, dk, dv, _, _ = attn_ref(do, q, k, v, mask, segment_ids, o, logsumexp)
      return dq, dk, dv

    is_grouped = not is_mqa and num_kv_heads < num_q_heads
    assert num_q_heads % num_kv_heads == 0
    head_multiplier = num_q_heads // num_kv_heads
    if is_mqa:
      bwd = jax.vmap(bwd, in_axes=(None, 0, None, None, None, 0, 0, 0))
    else:
      bwd = jax.vmap(bwd, in_axes=(None, 0, 0, 0, None, 0, 0, 0))
      # Interleave the KV heads to match the corresponding Q heads.
      if is_grouped:
        k32 = jnp.repeat(k32, head_multiplier, axis=0)
        v32 = jnp.repeat(v32, head_multiplier, axis=0)

    dq_ref, dk_ref, dv_ref = bwd(
        mask[:, :],
        q32,
        k32,
        v32,
        segment_ids,
        o.astype(jnp.float32),
        logsumexp,
        do.astype(jnp.float32),
    )
    if is_mqa:
      dk_ref, dv_ref = dk_ref.sum(axis=0), dv_ref.sum(axis=0)
    elif is_grouped:
      # Perform the sum reduction across the head_multiplier dimension only.
      # So that the output still has KV heads.
      dk_ref = dk_ref.reshape(num_kv_heads, head_multiplier, *dk_ref.shape[1:])
      dv_ref = dv_ref.reshape(num_kv_heads, head_multiplier, *dv_ref.shape[1:])

      dk_ref, dv_ref = dk_ref.sum(axis=1), dv_ref.sum(axis=1)

    dq_atol = 5e-2 if use_base2_exp else 2e-2
    dk_atol = 7e-2 if use_base2_exp else 2e-2
    dv_atol = 2e-2 if use_base2_exp else 2e-2
    self._assert_allclose(dq, dq_ref, atol=dq_atol, rtol=3e-2)
    self._assert_allclose(dk, dk_ref, atol=dk_atol, rtol=3e-2)
    self._assert_allclose(dv, dv_ref, atol=dv_atol, rtol=3e-2)


if __name__ == "__main__":
  absltest.main()
