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

from collections.abc import Callable
import dataclasses
import functools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_test_utils as test_utils


jax.config.parse_flags_with_absl()


hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=15,
    print_blob=False,
    verbosity=hp.Verbosity.normal,
)
hp.settings.load_profile(name="deterministic")

partial = functools.partial
type Draw = hps.DrawFn


@dataclasses.dataclass
class ModelConfig:
  q_seq_len: int
  kv_seq_len: int
  num_q_heads: int
  num_kv_heads: int
  head_dim_qk: int
  head_dim_v: int
  dtype: np.dtype


@hps.composite
def segment_ids_strategy(draw, seq_len: int) -> base.SegmentIds:
  boundaries = hps.sets(hps.integers(1, seq_len - 1), min_size=1, max_size=4)
  bounds = sorted(draw(boundaries))
  ids_array = np.empty((seq_len,), dtype=np.int32)
  for i, (start, end) in enumerate(zip((0, *bounds), (*bounds, seq_len))):
    # Not sure why, but short segments can trip things up
    if end - start < 2:
      end = start + 2
    ids_array[start:end] = i
  return base.SegmentIds(jnp.asarray(ids_array), jnp.asarray(ids_array))


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
class SplitMask(Mask):
  q_seq_len: int
  kv_seq_len: int

  def get_mask(self) -> mask_lib.Mask:
    mask = np.ones((self.q_seq_len, self.kv_seq_len)).astype(np.bool_)
    mask[:, mask.shape[1] // 2 :] = False
    return mask_lib.NumpyMask(mask)


def split_mask_strategy(
    q_seq_len: int, kv_seq_len: int
) -> hps.SearchStrategy[Mask]:
  return hps.just(SplitMask(q_seq_len, kv_seq_len))


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
      split_mask_strategy(q_seq_len, kv_seq_len),
      random_mask_strategy(q_seq_len, kv_seq_len),
      # TODO Composing masks creates masks that produce minor numerical
      # differences. We should investigate this in the future.
      # compose_mask_strategy(q_seq_len, kv_seq_len),
  ]

  if q_seq_len == kv_seq_len:
    oneof.append(local_attention_mask_strategy(q_seq_len))

  return draw(hps.one_of(oneof))


@hps.composite
def model_config_strategy(draw: Draw) -> ModelConfig:
  q_seq_len = draw(hps.sampled_from([1024, 2048, 4096]))
  kv_seq_len = draw(hps.sampled_from([1024, 2048, 4096]))
  head_dim_qk, head_dim_v = draw(
      hps.sampled_from(
          [(64, 128), (64, 64), (128, 128), (256, 256), (192, 128)]
      )
  )
  if q_seq_len >= 4096 and kv_seq_len >= 4096:
    dtype = np.dtype("float32")
  else:
    dtype = draw(
        hps.sampled_from([np.dtype("float32"), np.dtype(jnp.bfloat16)])
    )

  num_q_heads, num_kv_heads = draw(
      hps.sampled_from([(1, 1), (2, 2), (4, 1), (8, 4), (6, 2)])
  )
  return ModelConfig(
      q_seq_len,
      kv_seq_len,
      num_q_heads,
      num_kv_heads,
      head_dim_qk,
      head_dim_v,
      dtype,
  )


def check_mask_no_empty_rows(
    mask: mask_lib.Mask, segment_ids: splash.SegmentIds | None
):
  effective_mask = np.array(mask[:, :])

  if segment_ids is not None:
    segment_mask = segment_ids.q[:, None] == segment_ids.kv[None, :]
    effective_mask = effective_mask & segment_mask

  hp.assume(np.all(np.any(effective_mask, axis=1)))


@hps.composite
def block_sizes_strategy(
    draw: Draw,
    q_seq_len: int,
    kv_seq_len: int,
    include_bwd_blocks: bool = False,
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
      **layouts,
  )


def _generate_inputs(
    data,
    config: ModelConfig,
    is_mqa: bool,
    is_segmented: bool,
    use_sinks: bool = False,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    splash.SegmentIds | None,
    jax.Array,
]:
  seed = data.draw(seed_strategy())
  key = random.key(seed)
  k1, k2, k3, k_sinks, k_do = random.split(key, 5)

  q_shape = (config.num_q_heads, config.q_seq_len, config.head_dim_qk)
  if is_mqa:
    k_shape = (config.kv_seq_len, config.head_dim_qk)
    v_shape = (config.kv_seq_len, config.head_dim_v)
  else:
    k_shape = (config.num_kv_heads, config.kv_seq_len, config.head_dim_qk)
    v_shape = (config.num_kv_heads, config.kv_seq_len, config.head_dim_v)

  q = random.uniform(k1, q_shape, dtype=config.dtype)
  k = random.uniform(k2, k_shape, dtype=config.dtype)
  v = random.uniform(k3, v_shape, dtype=config.dtype)

  sinks = None
  if use_sinks:
    sinks = random.uniform(k_sinks, (config.num_q_heads,), dtype=config.dtype)

  segment_ids = None
  if is_segmented:
    hp.assume(config.q_seq_len == config.kv_seq_len)
    segment_ids = data.draw(segment_ids_strategy(config.q_seq_len))

  o_shape = (config.num_q_heads, config.q_seq_len, config.head_dim_v)
  do = random.uniform(k_do, o_shape, dtype=config.dtype)
  return (q, k, v, sinks, segment_ids, do)


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
    model_config = data.draw(model_config_strategy())
    q_seq_len, kv_seq_len = model_config.q_seq_len, model_config.kv_seq_len
    q, k, v, _, segment_ids, _ = _generate_inputs(
        data, model_config, is_mqa, is_segmented
    )
    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask = data.draw(mask_strategy(q_seq_len, kv_seq_len)).get_mask()
    check_mask_no_empty_rows(mask, segment_ids)
    if is_dynamic_mask:
      mask = jnp.array(mask[:, :])
    config = data.draw(block_sizes_strategy(q_seq_len, kv_seq_len))
    config = dataclasses.replace(
        config,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=self.INTERPRET,
    )

    attn_ref = partial(base.attention_reference, is_mqa=is_mqa)
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

    attn = make_mask_fn(mask, config=config)

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
      use_sinks=(False, True),
  )
  @hp.given(hps.data())
  def test_splash_attention_fwd(self, is_mqa, is_segmented, is_dynamic_mask,
                                use_base2_exp, use_max_logit_estimate,
                                fuse_reciprocal, use_sinks, data):
    model_config = data.draw(model_config_strategy())
    q_seq_len, kv_seq_len = model_config.q_seq_len, model_config.kv_seq_len
    q, k, v, sinks, segment_ids, _ = _generate_inputs(
        data, model_config, is_mqa, is_segmented, use_sinks
    )
    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask = data.draw(mask_strategy(q_seq_len, kv_seq_len)).get_mask()
    check_mask_no_empty_rows(mask, segment_ids)
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

    q_heads_per_kv_head = model_config.num_q_heads // model_config.num_kv_heads
    if (
        sinks is None
        and use_max_logit_estimate is None
        and q_heads_per_kv_head % 2 == 0
    ):
      num_stacked_q_heads = 2
    else:
      num_stacked_q_heads = 1

    config = dataclasses.replace(
        config,
        fuse_reciprocal=fuse_reciprocal,
        attn_logits_soft_cap=attn_logits_soft_cap,
        use_base2_exp=use_base2_exp,
        interpret=self.INTERPRET,
        num_stacked_q_heads=num_stacked_q_heads,
    )

    max_logit_value, max_val = None, 30.0
    if use_max_logit_estimate == "const":
      config = dataclasses.replace(config, max_logit_const=max_val)
    elif use_max_logit_estimate == "value_1d":
      max_logit_value = max_val * jnp.ones((1,), dtype=jnp.bfloat16)
    elif use_max_logit_estimate == "value_2d":
      max_logit_value = max_val * jnp.ones(
          (model_config.num_q_heads,), dtype=jnp.bfloat16
      )
    attn = make_mask_fn(mask, config=config, save_residuals=True)
    attn_ref = partial(
        base.attention_reference,
        is_mqa=is_mqa,
        save_residuals=True,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    o, stats = attn(
        q, k, v, segment_ids, sinks, max_logit_value=max_logit_value
    )

    o_ref, stats_ref = attn_ref(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        jnp.array(mask[:, :]),
        segment_ids,
        sinks,
    )

    lse_tol = dict(atol=1e-3, rtol=3e-3)
    max_logits_tol = dict(atol=1e-3, rtol=4e-3)
    if use_sinks:
      o_tol = dict(atol=8e-3, rtol=5e-3)
      lse_tol["rtol"] = 5e-3
    elif (use_base2_exp or use_max_logit_estimate is not None
          or not fuse_reciprocal):
      o_tol = dict(atol=8e-3, rtol=3e-3)
    else:
      o_tol = dict(atol=4e-3, rtol=3e-3)

    self._assert_allclose(o, o_ref, **o_tol)
    self._assert_allclose(stats["logsumexp"],
                          stats_ref["logsumexp"], **lse_tol)
    if use_max_logit_estimate is None:
      self._assert_allclose(stats["max_logits"],
                            stats_ref["max_logits"], **max_logits_tol)

  @parameterized.product(
      is_mqa=(False, True),
      is_segmented=(False, True),
      is_dynamic_mask=(False, True),
      # use_max_logit_estimate=(None, "const", "value_1d", "value_2d"),
      use_max_logit_estimate=(None,),
      use_sinks=(False, True),
      dq_reduction_steps=(None, 3),
      save_residuals=(False, True),
  )
  @hp.given(hps.data())
  def test_splash_attention_bwd(
      self,
      is_mqa,
      is_segmented,
      is_dynamic_mask,
      use_max_logit_estimate,
      dq_reduction_steps,
      use_sinks,
      save_residuals,
      data,
  ):
    downcast_smem_data = data.draw(hp.strategies.booleans())
    fuse_reciprocal = data.draw(hp.strategies.booleans())
    use_base2_exp = data.draw(hp.strategies.booleans())

    model_config = data.draw(model_config_strategy())
    q_seq_len, kv_seq_len = model_config.q_seq_len, model_config.kv_seq_len
    q, k, v, sinks, segment_ids, do = _generate_inputs(
        data, model_config, is_mqa, is_segmented, use_sinks=use_sinks
    )
    attn_logits_soft_cap = data.draw(attn_logits_soft_cap_strategy())
    mask = data.draw(mask_strategy(q_seq_len, kv_seq_len)).get_mask()
    check_mask_no_empty_rows(mask, segment_ids)
    if is_dynamic_mask:
      mask = jnp.array(mask[:, :])
    config = data.draw(
        block_sizes_strategy(q_seq_len, kv_seq_len, include_bwd_blocks=True)
    )

    config = dataclasses.replace(
        config,
        fuse_reciprocal=fuse_reciprocal,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=self.INTERPRET,
        use_base2_exp=use_base2_exp,
        dq_reduction_steps=dq_reduction_steps,
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
      max_logit_value = max_val * jnp.ones(
          (model_config.num_q_heads,), dtype=jnp.bfloat16
      )

    attn = make_mask_fn(
        mask,
        config=config,
        downcast_smem_data=downcast_smem_data,
        save_residuals=save_residuals,
    )

    if save_residuals:
      (o, stats), attn_vjp = jax.vjp(
          partial(attn, max_logit_value=max_logit_value),
          q,
          k,
          v,
          segment_ids,
          sinks,
      )
      cotangents = (do, jax.tree.map(jnp.zeros_like, stats))
    else:
      o, attn_vjp = jax.vjp(
          partial(attn, max_logit_value=max_logit_value),
          q,
          k,
          v,
          segment_ids,
          sinks,
      )
      cotangents = do
    q32, k32, v32 = jax.tree.map(lambda x: x.astype(jnp.float32), (q, k, v))
    o_ref, stats_ref = base.attention_reference(
        q32,
        k32,
        v32,
        jnp.array(mask[:, :]),
        segment_ids,
        sinks,
        is_mqa=is_mqa,
        save_residuals=True,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    if use_sinks:
      o_tol = dict(atol=1e-2, rtol=1e-2)
    elif (use_base2_exp or use_max_logit_estimate is not None
          or not fuse_reciprocal):
      o_tol = dict(atol=8e-3, rtol=1e-2)
    else:
      o_tol = dict(atol=4e-3, rtol=3e-3)
    self._assert_allclose(o, o_ref, **o_tol)

    dq, dk, dv, _, dsinks = attn_vjp(cotangents)
    dq_ref, dk_ref, dv_ref, dsinks_ref = base.attention_reference_vjp(
        do.astype(jnp.float32),
        q32,
        k32,
        v32,
        jnp.array(mask[:, :]),
        segment_ids,
        sinks,
        o.astype(jnp.float32),
        stats_ref["logsumexp"],
        is_mqa=is_mqa,
        backward_impl="flash",
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    dq_atol = 8e-2 if use_base2_exp else 2e-2
    dk_atol = 7e-2 if use_base2_exp else 2e-2
    dv_atol = 2e-2 if use_base2_exp else 2e-2
    self._assert_allclose(dq, dq_ref, atol=dq_atol, rtol=3e-2)
    self._assert_allclose(dk, dk_ref, atol=dk_atol, rtol=3e-2)
    self._assert_allclose(dv, dv_ref, atol=dv_atol, rtol=3e-2)
    if use_sinks:
      self._assert_allclose(dsinks, dsinks_ref, atol=4e-3, rtol=6e-3)

  @parameterized.product(
      mode=("forward", "backward"),
      qk_diag_grid=(2, 4),
      head_dim_qk=(128, 192),
  )
  def test_qk_diag_skip_bit_exact(self, mode, qk_diag_grid, head_dim_qk):
    """`qk_diag_skip` must be BIT-EXACT vs the stock (`qk_diag_skip=False`) path.

    The skip fills `mask_value` on fully-masked (kv > q) diagonal sub-tiles, which the
    softmax's `jnp.where` overwrites — so the output is identical for any `qk_diag_grid`.
    Covers forward `O` and backward `dQ/dK/dV` on a causal mask with square, power-of-2
    blocks, at two head dims (incl. the DS-v3 192/128 shape).
    """
    seq_len, num_heads, block = 512, 2, 256
    k1, k2, k3, k4 = random.split(random.key(0), 4)
    q = (random.normal(k1, (num_heads, seq_len, head_dim_qk)) * 0.5).astype(jnp.bfloat16)
    k = (random.normal(k2, (num_heads, seq_len, head_dim_qk)) * 0.5).astype(jnp.bfloat16)
    v = (random.normal(k3, (num_heads, seq_len, 128)) * 0.5).astype(jnp.bfloat16)
    do = (random.normal(k4, (num_heads, seq_len, 128)) * 0.5).astype(jnp.bfloat16)
    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))

    def build(qk_diag_skip):
      config = splash.SplashConfig(
          block_q=block, block_kv=block, block_kv_compute=block,
          block_q_dkv=block, block_kv_dkv=block, block_kv_dkv_compute=block,
          use_fused_bwd_kernel=True, residual_checkpoint_name="context",
          qk_diag_skip=qk_diag_skip, qk_diag_grid=qk_diag_grid,
          interpret=self.INTERPRET,
      )
      attn = splash.make_splash_mha_single_device(mask, config=config)
      if mode == "forward":
        return jax.jit(lambda q, k, v: attn(q, k, v))

      def bwd(q, k, v, do):
        _, vjp = jax.vjp(lambda q, k, v: attn(q, k, v), q, k, v)
        return vjp(do)

      return jax.jit(bwd)

    args = (q, k, v) if mode == "forward" else (q, k, v, do)
    ref = jax.tree.leaves(build(False)(*args))
    opt = jax.tree.leaves(build(True)(*args))
    for r, o in zip(ref, opt):
      # Bit-exact: the skip removes matmul work but must not change a single bit.
      np.testing.assert_array_equal(np.asarray(o), np.asarray(r))

  def test_qk_diag_skip_preconditions_raise(self):
    """`qk_diag_skip` must fail loud (never silently corrupt) on unsupported configs."""
    block = 256
    square = dict(
        block_q=block, block_kv=block, block_kv_compute=block,
        block_q_dkv=block, block_kv_dkv=block, block_kv_dkv_compute=block,
        use_fused_bwd_kernel=True, residual_checkpoint_name="context",
        interpret=self.INTERPRET,
    )
    # Non-square forward blocks -> raises in SplashConfig.__post_init__.
    with self.assertRaisesRegex(ValueError, "square forward blocks"):
      splash.SplashConfig(
          **{**square, "block_kv": block // 2, "block_kv_compute": block // 2},
          qk_diag_skip=True,
      )
    # Non-power-of-2 grid -> raises in __post_init__.
    with self.assertRaisesRegex(ValueError, "power of 2"):
      splash.SplashConfig(**square, qk_diag_skip=True, qk_diag_grid=3)
    # Non-causal mask -> raises in make_splash_mha (the skip assumes kv > q is masked).
    seq_len = 512
    config = splash.SplashConfig(**square, qk_diag_skip=True)
    local = mask_lib.LocalMask(
        shape=(seq_len, seq_len), window_size=(128, 0), offset=0
    )
    with self.assertRaisesRegex(ValueError, "CausalMask"):
      splash.make_splash_mha_single_device(local, config=config)

  @parameterized.named_parameters(
      dict(testcase_name="grid2", qk_diag_grid=2, head_dim_qk=128),
      dict(testcase_name="grid4", qk_diag_grid=4, head_dim_qk=192),
  )
  def test_sv_diag_skip_equivalence(self, qk_diag_grid, head_dim_qk):
    """`sv_diag_skip` must be equivalent to the stock (`sv_diag_skip=False`) path."""
    seq_len, num_heads, block = 512, 2, 256
    k1, k2, k3 = random.split(random.key(0), 3)
    q = (random.normal(k1, (num_heads, seq_len, head_dim_qk)) * 0.5).astype(
        jnp.bfloat16
    )
    k = (random.normal(k2, (num_heads, seq_len, head_dim_qk)) * 0.5).astype(
        jnp.bfloat16
    )
    v = (random.normal(k3, (num_heads, seq_len, 128)) * 0.5).astype(
        jnp.bfloat16
    )
    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))

    def build(sv_diag_skip):
      config = splash.SplashConfig(
          block_q=block,
          block_kv=block,
          block_kv_compute=block,
          block_q_dkv=block,
          block_kv_dkv=block,
          block_kv_dkv_compute=block,
          use_fused_bwd_kernel=True,
          residual_checkpoint_name="context",
          sv_diag_skip=sv_diag_skip,
          qk_diag_grid=qk_diag_grid,
          interpret=self.INTERPRET,
      )
      attn = splash.make_splash_mha_single_device(mask, config=config)
      return jax.jit(attn)

    ref = jax.tree.leaves(build(False)(q, k, v))
    opt = jax.tree.leaves(build(True)(q, k, v))
    for r, o in zip(ref, opt):
      self._assert_allclose(np.asarray(o), np.asarray(r), atol=3e-3, rtol=1e-2)

  def test_sv_diag_skip_preconditions_raise(self):
    """`sv_diag_skip` must fail loud (never silently corrupt) on unsupported configs."""
    block = 256
    square = dict(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        use_fused_bwd_kernel=True,
        residual_checkpoint_name="context",
        interpret=self.INTERPRET,
    )
    # Non-square forward blocks -> raises in SplashConfig.__post_init__.
    with self.assertRaisesRegex(ValueError, "square forward blocks"):
      splash.SplashConfig(
          **{**square, "block_kv": block // 2, "block_kv_compute": block // 2},
          sv_diag_skip=True,
      )
    # Non-power-of-2 grid -> raises in __post_init__.
    with self.assertRaisesRegex(ValueError, "power of 2"):
      splash.SplashConfig(**square, sv_diag_skip=True, qk_diag_grid=3)
    # Non-causal mask -> raises in make_splash_mha (the skip assumes kv > q
    # is masked).
    seq_len = 512
    config = splash.SplashConfig(**square, sv_diag_skip=True)
    local = mask_lib.LocalMask(
        shape=(seq_len, seq_len), window_size=(128, 0), offset=0
    )
    with self.assertRaisesRegex(ValueError, "CausalMask"):
      splash.make_splash_mha_single_device(local, config=config)


def _rel_l2(x: jax.Array, y: jax.Array) -> float:
  """Relative L2 error ‖x - y‖ / ‖y‖."""
  x_np = np.asarray(x, np.float64)
  y_np = np.asarray(y, np.float64)
  return float(np.linalg.norm(x_np - y_np) / (np.linalg.norm(y_np) + 1e-12))


def _get_dropout_mask_kernel(
    prng_key_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    bq: int,
    bkv_compute: int,
    canonical_q: int,
    canonical_kv: int,
    dropout_rate: float,
):
  # pylint: disable-next=protected-access
  out_ref[...] = splash._dropout_mask_tile(
      prng_key_ref,
      head_idx=pl.program_id(0),
      q_block_idx=pl.program_id(1),
      kv_block_idx=pl.program_id(2),
      q_block_size=bq,
      kv_block_size=bkv_compute,
      canonical_q=canonical_q,
      canonical_kv=canonical_kv,
      dropout_rate=dropout_rate,
  )


def _get_dropout_mask(
    num_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    config: splash.SplashConfig,
    prng_key: jax.Array,
    bq: int | None = None,
    bkv_compute: int | None = None,
) -> jax.Array:
  """Materializes the dropout mask the attention kernels generate internally.

  Test-only: the attention kernels never build this array, they regenerate one
  (bq, block_kv_compute) tile at a time. The `prng_key` must be the same one
  passed to the kernel, and `config` the same config, or the masks will not
  correspond.

  `bq`/`bkv_compute` choose the tiling this materialization walks the matrix
  with, defaulting to the canonical dropout grid. The result must not depend on
  them --- that is the whole point of keying the mask on canonical coordinates
  --- so passing them is how the tiling-invariance is tested.

  Returns:
    A [num_heads, q_seq_len, kv_seq_len] bool array; True means "dropped".
  """
  prng_key = pltpu.to_pallas_key(prng_key)
  canonical_q, canonical_kv = config.dropout_block_q, config.dropout_block_kv
  assert canonical_q is not None and canonical_kv is not None
  bq = canonical_q if bq is None else bq
  bkv_compute = canonical_kv if bkv_compute is None else bkv_compute
  grid = (num_heads, q_seq_len // bq, kv_seq_len // bkv_compute)

  kernel_name = "get_dropout_mask"
  with jax.named_scope(kernel_name):
    return pl.pallas_call(
        partial(
            _get_dropout_mask_kernel,
            bq=bq,
            bkv_compute=bkv_compute,
            canonical_q=canonical_q,
            canonical_kv=canonical_kv,
            dropout_rate=config.dropout_rate,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[],
            out_specs=pl.BlockSpec(
                (None, bq, bkv_compute), lambda h, i, j, *_: (h, i, j)
            ),
            grid=grid,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (num_heads, q_seq_len, kv_seq_len), jnp.bool_
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
        name=kernel_name,
        interpret=config.interpret,
    )(prng_key)


def _restated_dropout_mask_kernel(
    prng_key_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    bq: int,
    bkv: int,
    dropout_rate: float,
):
  """Restates the derivation `_generate_blockwise_dropout_mask` implements."""
  key_h = random.fold_in(prng_key_ref[...], pl.program_id(0))
  for i in range(out_ref.shape[0] // bq):
    key_q = random.fold_in(key_h, i)
    for j in range(out_ref.shape[1] // bkv):
      out_ref[i * bq : (i + 1) * bq, j * bkv : (j + 1) * bkv] = (
          random.bernoulli(random.fold_in(key_q, j), dropout_rate, (bq, bkv))
      )


def _restated_dropout_mask(
    num_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    config: splash.SplashConfig,
    prng_key: jax.Array,
) -> jax.Array:
  """Second statement of the dropout derivation, for `_get_dropout_mask`.

  `_get_dropout_mask` calls the kernel's own `_generate_blockwise_dropout_mask`,
  so on its own it can only show that the forward and backward passes agree with
  each other --- it cannot show that they agree with the *intended* scheme. This
  spells the scheme out a second time: fold head, then canonical q block, then
  canonical kv block into the key, in that order, and draw a
  (dropout_block_q, dropout_block_kv) tile.

  The two differ structurally on purpose. Here the grid is over heads alone and
  the block loop is unrolled inside the kernel with Python ints, so the block
  coordinates never come from `pl.program_id` and never pass through a
  `BlockSpec` index map. A q/kv transposition in either of those --- which both
  sides of a same-grid comparison would make identically --- shows up as a
  mismatch.

  Returns:
    A [num_heads, q_seq_len, kv_seq_len] bool array; True means "dropped".
  """
  prng_key = pltpu.to_pallas_key(prng_key)
  bq, bkv = config.dropout_block_q, config.dropout_block_kv
  assert bq is not None and bkv is not None

  return pl.pallas_call(
      partial(
          _restated_dropout_mask_kernel,
          bq=bq,
          bkv=bkv,
          dropout_rate=config.dropout_rate,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[],
          out_specs=pl.BlockSpec(
              (None, q_seq_len, kv_seq_len), lambda h, *_: (h, 0, 0)
          ),
          grid=(num_heads,),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (num_heads, q_seq_len, kv_seq_len), jnp.bool_
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      name="restated_dropout_mask",
      interpret=config.interpret,
  )(prng_key)


@test_utils.thread_unsafe_test_class()
class SplashAttentionDropoutTest(test_utils.SplashAttentionTestCase):
  """Attention dropout: the mask is generated in-kernel, never materialized.

  The kernel regenerates each (head, q block, kv block) tile of the mask from
  `prng_key`, in both the forward and the backward pass. These tests pin that
  down against a dense reference fed with the same mask, obtained from the
  test-only `_get_dropout_mask` above.
  """

  NUM_HEADS = 2
  SEQ_LEN = 512
  HEAD_DIM = 128
  BLOCK = 128

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def _config(self, dropout_rate: float, **kwargs) -> splash.SplashConfig:
    block = self.BLOCK
    if dropout_rate > 0:
      kwargs.setdefault("dropout_block_q", block)
      kwargs.setdefault("dropout_block_kv", block)
    return splash.SplashConfig(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        dropout_rate=dropout_rate,
        use_base2_exp=False,
        interpret=self.INTERPRET,
        **kwargs,
    )

  def _inputs(self, is_mqa: bool = False, seed: int = 0):
    k1, k2, k3, k4 = random.split(random.key(seed), 4)
    seq_len, head_dim = self.SEQ_LEN, self.HEAD_DIM
    q_shape = (self.NUM_HEADS, seq_len, head_dim)
    kv_shape = (seq_len, head_dim) if is_mqa else q_shape
    return (
        random.uniform(k1, q_shape, dtype=jnp.float32),
        random.uniform(k2, kv_shape, dtype=jnp.float32),
        random.uniform(k3, kv_shape, dtype=jnp.float32),
        random.uniform(k4, q_shape, dtype=jnp.float32),  # do
    )

  def _mask(self, is_causal: bool) -> mask_lib.Mask:
    shape = (self.SEQ_LEN, self.SEQ_LEN)
    return mask_lib.CausalMask(shape) if is_causal else mask_lib.FullMask(shape)

  def _attn(self, mask, config, is_mqa, is_dynamic_mask=False):
    if is_dynamic_mask:
      make_fn = (
          splash.make_dynamic_splash_mqa
          if is_mqa
          else splash.make_dynamic_splash_mha
      )
      return make_fn(jnp.array(mask[:, :]), config=config)
    make_fn = (
        splash.make_splash_mqa_single_device
        if is_mqa
        else splash.make_splash_mha_single_device
    )
    return make_fn(mask, config=config)

  @parameterized.product(
      dropout_rate=(0.1, 0.5),
      is_mqa=(False, True),
      is_causal=(False, True),
      is_dynamic_mask=(False, True),
  )
  def test_dropout_fwd(self, dropout_rate, is_mqa, is_causal, is_dynamic_mask):
    q, k, v, _ = self._inputs(is_mqa)
    mask = self._mask(is_causal)
    config = self._config(dropout_rate)
    attn = self._attn(mask, config, is_mqa, is_dynamic_mask)
    prng_key = random.key(1234)

    o = jax.jit(partial(attn, prng_key=prng_key))(q, k, v)
    dropout_mask = jax.jit(
        partial(
            _get_dropout_mask,
            self.NUM_HEADS,
            self.SEQ_LEN,
            self.SEQ_LEN,
            config,
        )
    )(prng_key)
    o_ref = base.attention_reference(
        q,
        k,
        v,
        jnp.array(mask[:, :]),
        dropout_mask=dropout_mask,
        is_mqa=is_mqa,
        dropout_rate=dropout_rate,
    )
    self._assert_allclose(o, o_ref, atol=2e-2, rtol=2e-2)

    # The mask is a Bernoulli draw per element, so over 2 * 512 * 512 elements
    # the realized rate is within a fraction of a percent of the target.
    self.assertAlmostEqual(
        float(jnp.mean(dropout_mask)), dropout_rate, delta=5e-3
    )

  @parameterized.product(is_mqa=(False, True), is_dynamic_mask=(False, True))
  def test_dropout_bwd(self, is_mqa, is_dynamic_mask):
    """dq/dk/dv against autodiff of the dense reference under the same mask.

    Compared with a relative L2 norm rather than elementwise `allclose`: the
    kernel recomputes `ds = (dp - di) * p` from the saved logsumexp, so it
    disagrees with dense autodiff by a large *relative* amount on the handful
    of dq/dk entries that are near zero. A rate=0 run of this same comparison
    already shows such outliers (~1% of entries beyond atol=rtol=5e-2), so an
    elementwise bound here would be measuring the kernel's flash-vs-dense
    error, not dropout. The thresholds below sit ~2x above the measured
    error, which is itself the same order as the rate=0 kernel's.
    """
    dropout_rate = 0.25
    q, k, v, do = self._inputs(is_mqa)
    mask = self._mask(is_causal=True)
    dense_mask = jnp.array(mask[:, :])
    config = self._config(dropout_rate)
    attn = self._attn(mask, config, is_mqa, is_dynamic_mask)
    prng_key = random.key(1234)
    dropout_mask = jax.jit(
        partial(
            _get_dropout_mask,
            self.NUM_HEADS,
            self.SEQ_LEN,
            self.SEQ_LEN,
            config,
        )
    )(prng_key)

    def loss(fn):
      return lambda q, k, v: jnp.sum(fn(q, k, v) * do)

    grads = jax.jit(
        jax.grad(loss(partial(attn, prng_key=prng_key)), argnums=(0, 1, 2))
    )(q, k, v)
    grads_ref = jax.jit(
        jax.grad(
            loss(
                lambda q, k, v: base.attention_reference(
                    q,
                    k,
                    v,
                    dense_mask,
                    dropout_mask=dropout_mask,
                    is_mqa=is_mqa,
                    dropout_rate=dropout_rate,
                )
            ),
            argnums=(0, 1, 2),
        )
    )(q, k, v)
    # dv is exactly linear in the dropout mask, so it is held to a much tighter
    # bound than dq/dk: any disagreement about *which* weights were dropped
    # shows up here first.
    tolerances = dict(dq=2e-2, dk=2e-2, dv=1e-3)
    for name, g, g_ref in zip(("dq", "dk", "dv"), grads, grads_ref):
      with self.subTest(name):
        self.assertTrue(jnp.isfinite(g).all())
        self.assertTupleEqual(g.shape, g_ref.shape)
        self.assertLess(_rel_l2(g, g_ref), tolerances[name])

  def test_dropout_rate_zero_is_a_no_op(self):
    """rate=0 must leave the kernel bit-identical, key or no key."""
    q, k, v, _ = self._inputs()
    mask = self._mask(is_causal=True)
    attn = self._attn(mask, self._config(0.0), is_mqa=False)
    o_no_key = jax.jit(attn)(q, k, v)
    o_with_key = jax.jit(partial(attn, prng_key=random.key(1234)))(q, k, v)
    self._assert_array_equal(o_no_key, o_with_key)

    # ... and it must differ from a run that actually drops.
    dropout_attn = self._attn(mask, self._config(0.25), is_mqa=False)
    o_dropout = jax.jit(partial(dropout_attn, prng_key=random.key(1234)))(
        q, k, v
    )
    self.assertGreater(float(jnp.abs(o_dropout - o_no_key).max()), 1e-2)

  def test_dropout_is_deterministic_in_the_key(self):
    """Same key -> bit-identical; different key -> a different mask."""
    q, k, v, _ = self._inputs()
    attn = self._attn(self._mask(is_causal=True), self._config(0.25), False)
    run = lambda key: jax.jit(partial(attn, prng_key=key))(q, k, v)
    self._assert_array_equal(run(random.key(1234)), run(random.key(1234)))
    self.assertGreater(
        float(jnp.abs(run(random.key(1234)) - run(random.key(4321))).max()),
        1e-2,
    )

  def test_dropout_fwd_and_bwd_use_the_same_mask(self):
    """dv is the cleanest probe: dv = sum_i pr_ij do_i uses the dropped weights.

    If the backward regenerated a different mask than the forward, dv would be
    wrong by O(rate) while still looking plausible, so compare it against the
    reference at a tight tolerance.
    """
    dropout_rate = 0.5
    q, k, v, do = self._inputs()
    mask = self._mask(is_causal=True)
    config = self._config(dropout_rate)
    attn = self._attn(mask, config, is_mqa=False)
    prng_key = random.key(7)
    dropout_mask = jax.jit(
        partial(
            _get_dropout_mask,
            self.NUM_HEADS,
            self.SEQ_LEN,
            self.SEQ_LEN,
            config,
        )
    )(prng_key)

    _, vjp = jax.vjp(partial(attn, prng_key=prng_key), q, k, v)
    _, _, dv = vjp(do)
    dense_mask = jnp.array(mask[:, :])
    _, vjp_ref = jax.vjp(
        lambda q, k, v: base.attention_reference(
            q,
            k,
            v,
            dense_mask,
            dropout_mask=dropout_mask,
            is_mqa=False,
            dropout_rate=dropout_rate,
        ),
        q,
        k,
        v,
    )
    _, _, dv_ref = vjp_ref(do)
    self._assert_allclose(dv, dv_ref, atol=5e-3, rtol=5e-3)

  def test_dropout_without_a_key_raises(self):
    q, k, v, _ = self._inputs()
    attn = self._attn(self._mask(is_causal=True), self._config(0.25), False)
    with self.assertRaisesRegex(ValueError, "prng_key is required"):
      jax.jit(attn)(q, k, v)

  def test_invalid_dropout_rate_raises(self):
    for rate in (-0.1, 1.0, 1.5):
      with self.assertRaisesRegex(ValueError, "dropout_rate must be in"):
        self._config(rate)

  def _mismatched_config(self, dropout_rate, fwd_mult, bwd_mult, **kwargs):
    block = self.BLOCK
    if dropout_rate > 0:
      default_canonical = min(fwd_mult, bwd_mult) * block
      kwargs.setdefault("dropout_block_q", default_canonical)
      kwargs.setdefault("dropout_block_kv", default_canonical)
    return splash.SplashConfig(
        block_q=fwd_mult * block,
        block_kv=fwd_mult * block,
        block_kv_compute=fwd_mult * block,
        block_q_dkv=bwd_mult * block,
        block_kv_dkv=bwd_mult * block,
        block_kv_dkv_compute=bwd_mult * block,
        dropout_rate=dropout_rate,
        use_base2_exp=False,
        interpret=self.INTERPRET,
        **kwargs,
    )

  def test_dropout_mask_is_invariant_to_the_tiling(self):
    """The same (head, q, kv) element draws the same bit at any tile size.

    This is the property that lets the two passes disagree about block sizes:
    the mask lives on the canonical grid, and a tile is only ever an assembly of
    whole canonical blocks. Walked here at four tilings coarser than the
    canonical 128x128 grid --- two of them non-square, so a q/kv transposition
    in the assembly shows up --- against the canonical walk. Bit-exact: the
    draws are literally the same draws, just grouped differently.

    Materialized over 4x the test's sequence length rather than over SEQ_LEN.
    The extent is independent of any attention shape here, and at SEQ_LEN the
    coarsest tiling would be a single tile spanning the whole matrix: both
    block indices are then 0, so scaling them into canonical units is a no-op
    and that tiling cannot tell a correct scaling from a missing one.
    """
    config = self._config(0.5)
    extent = 4 * self.SEQ_LEN
    args = (self.NUM_HEADS, extent, extent, config)
    key = random.key(3)
    reference = jax.jit(partial(_get_dropout_mask, *args))(key)
    block = self.BLOCK
    for bq_mult, bkv_mult in ((2, 1), (1, 2), (2, 2), (4, 4)):
      with self.subTest(f"{bq_mult}x{bkv_mult}"):
        tiled = jax.jit(
            partial(
                _get_dropout_mask,
                *args,
                bq=bq_mult * block,
                bkv_compute=bkv_mult * block,
            )
        )(key)
        self._assert_array_equal(tiled, reference)

  @parameterized.parameters((1, 2), (2, 1), (2, 4), (4, 2))
  def test_mismatched_fwd_bwd_blocks_agree(self, fwd_mult, bwd_mult):
    """End-to-end: fwd and bwd tiled differently still drop the same weights.

    Same dv probe as `test_dropout_fwd_and_bwd_use_the_same_mask` --- dv is
    exactly linear in the mask, so a backward that regenerated a *different*
    realization would be wrong by O(rate) here --- but with the two passes
    deliberately tiled 2x apart in both directions.
    """
    dropout_rate = 0.5
    q, k, v, do = self._inputs()
    mask = self._mask(is_causal=True)
    config = self._mismatched_config(dropout_rate, fwd_mult, bwd_mult)
    attn = self._attn(mask, config, is_mqa=False)
    prng_key = random.key(7)
    dropout_mask = jax.jit(
        partial(
            _get_dropout_mask,
            self.NUM_HEADS,
            self.SEQ_LEN,
            self.SEQ_LEN,
            config,
        )
    )(prng_key)

    _, vjp = jax.vjp(partial(attn, prng_key=prng_key), q, k, v)
    _, _, dv = vjp(do)
    dense_mask = jnp.array(mask[:, :])
    _, vjp_ref = jax.vjp(
        lambda q, k, v: base.attention_reference(
            q,
            k,
            v,
            dense_mask,
            dropout_mask=dropout_mask,
            is_mqa=False,
            dropout_rate=dropout_rate,
        ),
        q,
        k,
        v,
    )
    _, _, dv_ref = vjp_ref(do)
    self._assert_allclose(dv, dv_ref, atol=5e-3, rtol=5e-3)

  # Forward and backward block sizes from the tuned sweep, which picks them
  # per pass and never lands on the same ones: (block_q, block_kv,
  # block_kv_compute) forward, then the dkv trio. Under the old scheme every
  # one of these was un-runnable with dropout on.
  TUNED_BLOCKS = (
      ("s4k_to_s32k", (512, 1024, 512), (2048, 2048, 512)),
      ("s128k_d128", (512, 1024, 512), (1024, 4096, 512)),
      ("s128k_d192", (512, 1024, 512), (2048, 1024, 1024)),
      ("s128k_d256", (512, 1024, 512), (1024, 2048, 256)),
  )

  def test_tuned_block_sizes_are_expressible_with_dropout(self):
    """The shipped block-size pairs must all admit a canonical dropout grid."""
    for label, (bq, bkv, bkv_c), (bq_d, bkv_d, bkv_dc) in self.TUNED_BLOCKS:
      with self.subTest(label):
        gq = min(bq, bq_d)
        gkv = min(bkv_c, bkv_dc)
        # If this does not raise ValueError, they are valid multiples
        splash.SplashConfig(
            block_q=bq,
            block_kv=bkv,
            block_kv_compute=bkv_c,
            block_q_dkv=bq_d,
            block_kv_dkv=bkv_d,
            block_kv_dkv_compute=bkv_dc,
            dropout_rate=0.1,
            dropout_block_q=gq,
            dropout_block_kv=gkv,
        )

  def test_tiles_not_a_multiple_of_the_dropout_grid_raise(self):
    """A tile that splits a canonical block would break the invariance."""
    block = self.BLOCK
    with self.assertRaisesRegex(
        ValueError, "block_q_dkv=192 must be a multiple"
    ):
      splash.SplashConfig(
          block_q=block,
          block_kv=block,
          block_kv_compute=block,
          block_q_dkv=block + block // 2,
          block_kv_dkv=block,
          block_kv_dkv_compute=block,
          dropout_rate=0.25,
          dropout_block_q=block,
          dropout_block_kv=block,
      )
    with self.assertRaisesRegex(
        ValueError, "block_kv_compute=128 must be a multiple"
    ):
      self._mismatched_config(0.25, 1, 1, dropout_block_kv=48)
    with self.assertRaisesRegex(ValueError, "dropout_block_q must be positive"):
      self._mismatched_config(0.25, 1, 1, dropout_block_q=0)

  def test_get_dropout_mask_is_blockwise_and_head_dependent(self):
    """Different (head, q block, kv block) tiles must get independent draws."""
    config = self._config(0.5)
    mask = jax.jit(
        partial(
            _get_dropout_mask,
            self.NUM_HEADS,
            self.SEQ_LEN,
            self.SEQ_LEN,
            config,
        )
    )(random.key(0))
    self.assertEqual(mask.shape, (self.NUM_HEADS, self.SEQ_LEN, self.SEQ_LEN))
    self.assertEqual(mask.dtype, jnp.bool_)
    block = self.BLOCK
    tile = lambda h, i, j: mask[
        h, i * block : (i + 1) * block, j * block : (j + 1) * block
    ]
    self.assertFalse(bool(jnp.array_equal(tile(0, 0, 0), tile(1, 0, 0))))
    self.assertFalse(bool(jnp.array_equal(tile(0, 0, 0), tile(0, 1, 0))))
    self.assertFalse(bool(jnp.array_equal(tile(0, 0, 0), tile(0, 0, 1))))

  def test_dropout_mask_matches_a_restated_derivation(self):
    """The mask is the documented function of the key and block coordinates.

    Pins the derivation itself against `_restated_dropout_mask`: bit-exact,
    since both draw from the same key with the same rate. Changing the order
    the coordinates are folded in, or the shape of a tile, breaks this test
    without breaking any of the ones above, which only require the forward and
    backward passes to agree with each other.
    """
    for rate in (0.25, 0.5):
      with self.subTest(rate=rate):
        config = self._config(rate)
        args = (self.NUM_HEADS, self.SEQ_LEN, self.SEQ_LEN, config)
        key = random.key(7)
        actual = jax.jit(partial(_get_dropout_mask, *args))(key)
        expected = jax.jit(partial(_restated_dropout_mask, *args))(key)
        self._assert_array_equal(actual, expected)

  def test_reference_vjp_matches_autodiff_under_dropout(self):
    """Covers the hand-written backward in `base.attention_reference_vjp`.

    Reference against reference, no kernel involved: a manual backward has to
    agree with autodiff of the forward it claims to differentiate, so the bound
    here is float32 reassociation noise rather than flash-vs-dense error.

    Matmul precision is pinned to "highest" because the manual backward
    *recomputes* the logits. Under TPU's default bf16-pass f32 matmul the two
    q@k products differ by ~1e-3 relative, which propagates into `p` and swamps
    the thing being tested; the alternative --- a 1e-2 tolerance --- would pass
    for a backward that had the dropout rescaling wrong.
    """
    dropout_rate = 0.25
    q, k, v, do = self._inputs()
    dense_mask = jnp.array(self._mask(is_causal=True)[:, :])
    dropout_mask = jax.jit(
        partial(
            _get_dropout_mask,
            self.NUM_HEADS,
            self.SEQ_LEN,
            self.SEQ_LEN,
            self._config(dropout_rate),
        )
    )(random.key(11))

    fwd = lambda q, k, v: base.attention_reference(
        q,
        k,
        v,
        dense_mask,
        dropout_mask=dropout_mask,
        is_mqa=False,
        dropout_rate=dropout_rate,
        save_residuals=True,
    )
    with jax.default_matmul_precision("highest"):
      o, stats = fwd(q, k, v)
      dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
          do,
          q,
          k,
          v,
          dense_mask,
          None,
          None,
          o,
          stats["logsumexp"],
          dropout_mask,
          is_mqa=False,
          backward_impl="flash",
          dropout_rate=dropout_rate,
      )

      grads = jax.grad(
          lambda q, k, v: jnp.sum(fwd(q, k, v)[0] * do), argnums=(0, 1, 2)
      )(q, k, v)
    for name, g, g_ref in zip(
        ("dq", "dk", "dv"), grads, (dq_ref, dk_ref, dv_ref)
    ):
      with self.subTest(name):
        self.assertLess(_rel_l2(g, g_ref), 3e-5)


if __name__ == "__main__":
  absltest.main()
