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
"""Benchmarks for attention implementations."""

# pylint: disable=line-too-long

import dataclasses
from functools import partial  # pylint: disable=g-importing-member
import pprint
from typing import Any, Callable

from absl import app
from absl import flags
import jax
from jax import random  # pylint: disable=g-importing-member
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib
import tune_jax


QKVLayout = splash.QKVLayout

# pytype: disable=attribute-error

flags.DEFINE_integer("num_samples", int(1e30), "Tuning samples upper bound.")
flags.DEFINE_integer("q_seq_len", 4096, "Number of q sequence length.")
flags.DEFINE_integer("kv_seq_len", 4096, "Number of kv sequence length.")
flags.DEFINE_integer("q_heads", 128, "Number of q heads.")
flags.DEFINE_integer("kv_heads", 128, "Number of kv heads.")
flags.DEFINE_integer("qk_head_dim", 192, "Dimension of qk heads.")
flags.DEFINE_integer("v_head_dim", 128, "Dimension of kv heads.")
flags.DEFINE_integer("batch_size", 8, "Batch size.")
flags.DEFINE_string("mode", "fwd", "'fwd' or 'bwd'.")
flags.DEFINE_string("dtype", "bfloat16", "Data type to use.")
flags.DEFINE_string("mask_name", "causal_mask", "Mask type to use.")
flags.DEFINE_bool(
    "tune_pallas_only",
    False,
    "Whether to only time Pallas kernels when tuning (no XLA work).",
)


@dataclasses.dataclass(frozen=True, slots=True)
class ArgSpec:
  q_shape: tuple[int, ...]
  k_shape: tuple[int, ...]
  v_shape: tuple[int, ...]
  mask_shape: tuple[int, ...]
  mask_name: str
  dtype: jnp.dtype


_mask_name_to_fn = {
    "full_mask": lambda shape, offset=0: mask_lib.FullMask(shape[-2:]),
    "causal_mask": lambda shape, offset=0: mask_lib.CausalMask(
        shape[-2:], offset=offset
    ),
}


def get_benchmark_func(mask, config, mode, mqa: bool) -> Callable:
  config = dataclasses.replace(config, use_base2_exp=True)
  if mqa:
    kernel = splash.make_splash_mqa_single_device(mask, config=config)

    @jax.jit
    def f(q, k, v):
      q = q.reshape(q.shape[:-3] + (k.shape[-3], -1) + q.shape[-2:])
      kernel_ = jax.vmap(kernel, in_axes=(0, 0, 0))  # batch vmap
      kernel_ = jax.vmap(kernel_, in_axes=(0, 0, 0))  # mqa vmap
      return kernel_(q, k, v)

  else:
    kernel = splash.make_splash_mha_single_device(mask, config=config)
    f = jax.jit(jax.vmap(kernel, in_axes=(0, 0, 0)))

  if mode == "fwd":
    return f
  if mode == "bwd":
    return jax.grad(lambda q, k, v: f(q, k, v).mean(), argnums=(0, 1, 2))
  raise ValueError(f"Invalid mode: {mode}")


def randn_init(key, shape, dtype):
  """Use numpy for initialization because it's faster than XLA compilation."""
  seed = int(
      np.sum(
          [int(x) << (32 * i) for i, x in enumerate(jax.random.key_data(key))]
      )
  )
  r = jnp.array(np.random.default_rng(seed).normal(size=shape)).astype(dtype)
  return r / (r.shape[-1] ** 0.5)


def run_benchmark(
    name: str,
    spec: ArgSpec,
    mode: str,
    collect_paccheto_trace: bool = False,
    hyperparams_override: dict[str, Any] | None = None,
    event_filter_regex: str | None = None,
    num_samples: int = int(1e30),
):
  bmark_name = f"{name}_{mode}"

  k1, k2, k3 = random.split(random.key(0), 3)
  # q = jax.random.uniform(k1, spec.q_shape, dtype=spec.dtype)
  # k = jax.random.uniform(k2, spec.k_shape, dtype=spec.dtype)
  # v = jax.random.uniform(k3, spec.v_shape, dtype=spec.dtype)
  q = randn_init(k1, spec.q_shape, dtype=spec.dtype)
  k = randn_init(k2, spec.k_shape, dtype=spec.dtype)
  v = randn_init(k3, spec.v_shape, dtype=spec.dtype)
  # pick offset for causal masks for a "representative" slice of the causal
  # attention mask
  offset = (
      0 if q.shape[-2] == v.shape[-2] else (v.shape[-2] // 2 - q.shape[-2] // 2)
  )
  mask = _mask_name_to_fn[spec.mask_name](spec.mask_shape, offset=offset)

  def attention_fn(
      *args,
      block_q: int,
      block_kv: int,
      block_kv_compute: int,
      block_q_dkv: int | None,
      block_kv_dkv: int | None,
      block_kv_dkv_compute: int | None,
      block_q_dq: int | None,
      block_kv_dq: int | None,
      q_layout: QKVLayout,
      k_layout: QKVLayout,
      v_layout: QKVLayout,
      mask: mask_lib.Mask,
      mode: str,
      use_experimental_scheduler: bool,
  ):
    config = splash.SplashConfig(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
        use_experimental_scheduler=use_experimental_scheduler,
    )

    f = get_benchmark_func(
        mask, config, mode, mqa=spec.q_shape[-3] != spec.k_shape[-3]
    )
    return f(*args)

  attention_fn = partial(attention_fn, mask=mask, mode=mode)

  tiles = [256, 512, 1024, 2048, 4096, 8192]
  layouts = [QKVLayout.HEAD_DIM_MINOR, QKVLayout.SEQ_MINOR]
  hyperparams = {
      "block_q": tiles,
      "block_kv": tiles,
      "block_kv_compute": tiles,
      "block_q_dkv": [None],
      "block_kv_dkv": [None],
      "block_kv_dkv_compute": [None],
      "block_q_dq": [None],
      "block_kv_dq": [None],
      "q_layout": layouts,
      "k_layout": layouts,
      "v_layout": layouts,
      "use_experimental_scheduler": [True, False],
  }

  if mode == "bwd":
    hyperparams["block_q_dkv"] = tiles
    hyperparams["block_kv_dkv"] = tiles
    hyperparams["block_kv_dkv_compute"] = tiles

  # incorporate any potentially previously tuned hyperparameters
  hyperparams_override = hyperparams_override or {}
  hyperparams = dict(hyperparams, **hyperparams_override)

  tune_jax.logger.setLevel("INFO")
  tune_jax.CONFIG.allow_fallback_timing = False
  splash_fn = jax.jit(
      attention_fn,
      static_argnames=(
          "block_q",
          "block_kv",
          "block_kv_compute",
          "block_q_dkv",
          "block_kv_dkv",
          "block_kv_dkv_compute",
          "block_q_dq",
          "block_kv_dq",
          "q_layout",
          "k_layout",
          "v_layout",
          "use_experimental_scheduler",
      ),
  )
  tuned_splash = tune_jax.tune(
      splash_fn,
      hyperparams=hyperparams,
      event_filter_regex=event_filter_regex,
      sample_num=num_samples,
  )
  tuned_splash(q, k, v)

  # Collect an xprof trace using the best block sizes.
  with jax.profiler.trace("/tmp/splash_attention_benchmark"):
    for _ in range(3):
      jax.block_until_ready(tuned_splash(q, k, v))

  print("-------------------------------------------------------------------")
  print(f"Benchmarking [{bmark_name}]...")
  print(f"Tuning spec: {pprint.pformat(spec)}")
  print("Run `xprof --logdir /tmp/splash_attention_benchmark` to get the xprof url.")
  print("Autotuning results:")
  print(tune_jax.tabulate(tuned_splash))
  print("Optimal hyperparameters:")
  print(tuned_splash.optimal_hyperparams)  # pytype: disable=attribute-error
  print("-------------------------------------------------------------------\n")

  return tuned_splash.optimal_hyperparams  # pytype: disable=attribute-error


def _pallas_call_hlo_pattern(mode: str, mqa: bool) -> str:
  if mode not in ["fwd", "bwd", "combined"]:
    raise ValueError(f"Invalid mode: {mode}, select either 'fwd' or 'bwd'.")
  mha_or_mqa = "mqa" if mqa else "mha"
  suffix = {"fwd": "fwd", "bwd": "dkv", "combined": ""}.get(mode, "")
  return f"splash_{mha_or_mqa}_{suffix}"


def main(argv):
  del argv  # Unused.

  bs = flags.FLAGS.batch_size
  q_seq_len = flags.FLAGS.q_seq_len
  kv_seq_len = flags.FLAGS.kv_seq_len
  q_heads = flags.FLAGS.q_heads
  kv_heads = flags.FLAGS.kv_heads
  qk_head_dim = flags.FLAGS.qk_head_dim
  v_head_dim = flags.FLAGS.v_head_dim
  mode = flags.FLAGS.mode
  dtype = flags.FLAGS.dtype
  mask_name = flags.FLAGS.mask_name

  # when timing, we can either pay attention to pallas calls only or the total
  # end2end time (inclusive of XLA work)
  if flags.FLAGS.tune_pallas_only:
    event_filter_regex = _pallas_call_hlo_pattern(mode, q_heads != kv_heads)
  else:
    event_filter_regex = None

  spec = ArgSpec(
      q_shape=(bs, q_heads, q_seq_len, qk_head_dim),
      k_shape=(bs, kv_heads, kv_seq_len, qk_head_dim),
      v_shape=(bs, kv_heads, kv_seq_len, v_head_dim),
      mask_shape=(q_seq_len, kv_seq_len),
      mask_name=mask_name,
      dtype=jnp.dtype(dtype),
  )

  # always tune the fwd kernel to discover the optimal forward hyperparameters
  if mode == "fwd":
    run_benchmark(
        "fwd",
        spec,
        mode,
        collect_paccheto_trace=False,
        num_samples=flags.FLAGS.num_samples,
        event_filter_regex=event_filter_regex,
    )
  elif mode == "bwd":
    # don't tune fwd only hyperparams
    hyperparams_override = dict(
        block_q=min(512, q_seq_len),
        block_kv=min(1024, kv_seq_len),
        block_kv_compute=min(512, kv_seq_len),
    )
    run_benchmark(
        "bwd",
        spec,
        mode,
        collect_paccheto_trace=False,
        hyperparams_override=hyperparams_override,
        num_samples=flags.FLAGS.num_samples,
        event_filter_regex=event_filter_regex,
    )
  elif mode == "combined":
    run_benchmark(
        "combined",
        spec,
        "bwd",  # mode combined
        collect_paccheto_trace=False,
        num_samples=flags.FLAGS.num_samples,
        event_filter_regex=None,  # for combined select kernels only
    )


if __name__ == "__main__":
  app.run(main)
