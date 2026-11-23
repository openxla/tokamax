# Copyright 2026 Google LLC
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
"""Tests of fused_out_proj: the GQA attention output projection."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jaxtyping import Array, BFloat16  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src.ops.batched_rpa.kernel import configs as rpa_configs
from tokamax._src.ops.batched_rpa.kernel import wrapper
from tokamax._src.ops.experimental.fused_qkv_proj import models
from tokamax._src.ops.experimental.fused_qkv_proj import reference
from tokamax._src.ops.experimental.fused_qkv_proj import test_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import kernel as gqa
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import kernel as out_proj

CASES = [
    (f"_{m.name}_T{t}", m.name, t)
    for m, t in itertools.product(models.MODELS, (32, 64, 128, 1024, 16384))
]
# test_accuracy: how far the kernel may fall below vLLM's SNR.
SNR_TOLERANCE_DB = 0.15
# test_accuracy: below this many tokens, the tolerance grows as 1/sqrt(T).
SNR_TOLERANCE_TOKENS = 128
# vLLM (torch) rounds every bf16 op; XLA must not elide those roundings.
strict_reference = jax.jit(
    reference.out_proj_reference,
    static_argnums=0,
    static_argnames=("num_tokens", "numerics"),
    compiler_options={"xla_allow_excess_precision": False},
)
Numerics = reference.Numerics


def run_kernel(
    model: models.Model,
    o: BFloat16[Array, "Hkv T_rpa q_per_kv/q_packing q_packing Dp"],
    weights: model_config.OutProjWeights,
    gate: BFloat16[Array, "T Hq*D"] | None,
    *,
    num_tokens: int | None = None,
) -> BFloat16[Array, "T K"]:
  """fused_out_proj on weights as loaded."""
  return out_proj.fused_out_proj(
      o,
      *weights,
      gate,
      **models.out_proj_kwargs(model.config),
      num_tokens=num_tokens,
  )


def full_attention_block(
    model: models.Model,
    t: int,
    qkv_args: models.PreparedInputs,
    out_weights: model_config.OutProjWeights,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    dist: jax.Array,
) -> tuple[BFloat16[Array, "T K"], jax.Array]:
  """fused_qkv -> batched_rpa (return_raw_output=True) -> fused_out_proj."""
  s = model.config
  kv_layout = rpa_configs.KVLayout.SEQ_ALONG_LANE
  x, residual, qkv_weights, cos, sin = qkv_args
  q, kv, gate, _ = gqa.fused_qkv(
      x,
      residual,
      *qkv_weights,
      cos,
      sin,
      **models.qkv_kwargs(s),
      kv_layout=kv_layout,
  )
  t_rpa = q.shape[1]
  qs = jax.ShapeDtypeStruct((t_rpa, s.num_q_heads, s.head_dim), s.q_dtype)
  kvs = jax.ShapeDtypeStruct((t_rpa, s.num_kv_heads, s.head_dim), s.kv_dtype)
  o_raw, kv_cache = wrapper.ragged_paged_attention(
      qs,
      kvs,
      kvs,
      kv_cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      dist,
      sm_scale=s.head_dim**-0.5,
      k_scale=s.k_scale,
      v_scale=s.v_scale,
      kv_layout=kv_layout,
      prepared_inputs=(q, kv),
      return_raw_output=True,
  )
  out = out_proj.fused_out_proj(
      o_raw,
      *out_weights,
      gate[:t] if gate is not None else None,
      **models.out_proj_kwargs(s),
      num_tokens=t,
  )
  return out, kv_cache


class FusedOutProjTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("TPU-only")

  def assert_exact(self, label: str, got: jax.Array, want: jax.Array) -> None:
    """Asserts got matches want bit for bit; prints the largest diff."""
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    self.assertEqual(g.shape, w.shape)
    diff = np.abs(g - w)
    first = np.unravel_index(diff.argmax(), diff.shape)
    print(
        f"{label}: max_abs_diff={diff.max():.3e} "
        f"mismatches={(diff > 0).sum()}/{diff.size} "
        f"worst@{first} got={g[first]} want={w[first]}",
        flush=True,
    )
    self.assertLess(diff.max(), 1e-5)

  @parameterized.named_parameters(*CASES)
  def test_kernel_matches_reference(self, name: str, t: int) -> None:
    """The kernel computes exactly Numerics.KERNEL."""
    model = models.BY_NAME[name]
    args = models.make_out_proj_inputs(model, t, exact=True)
    self.assert_exact(
        f"{name} T={t}",
        run_kernel(model, *args),
        strict_reference(model.config, *args),
    )

  @parameterized.named_parameters(*(
      (f"_{m.name}_T{t}_Trpa{t_rpa}", m.name, t, t_rpa)
      for m, (t, t_rpa) in itertools.product(
          models.MODELS, ((32, 128), (64, 128), (128, 256))
      )
  ))
  def test_padded_rpa_tokens(self, name: str, t: int, t_rpa: int) -> None:
    """When batched_rpa pads T to T_rpa, only the first T tokens are projected."""
    model = models.BY_NAME[name]
    args = models.make_out_proj_inputs(model, t, exact=True, t_rpa=t_rpa)
    self.assert_exact(
        f"{name} T={t} T_rpa={t_rpa}",
        run_kernel(model, *args, num_tokens=t),
        strict_reference(model.config, *args, num_tokens=t),
    )

  @parameterized.named_parameters(
      *((f"_{m.name}", m.name) for m in models.MODELS)
  )
  def test_kernel_numerics_match_vllm_when_exact(self, name: str) -> None:
    """Numerics.KERNEL and VLLM agree exactly on exact inputs."""
    model = models.BY_NAME[name]
    args = models.make_out_proj_inputs(model, 1024, exact=True)
    self.assert_exact(
        name,
        strict_reference(model.config, *args, numerics=Numerics.KERNEL),
        strict_reference(model.config, *args, numerics=Numerics.VLLM),
    )

  @parameterized.named_parameters(*CASES)
  def test_accuracy(self, name: str, t: int) -> None:
    """On random data, the kernel's SNR vs. UNQUANTIZED exceeds vLLM's."""
    model = models.BY_NAME[name]
    args = models.make_out_proj_inputs(model, t)
    got = run_kernel(model, *args)
    vllm = strict_reference(model.config, *args, numerics=Numerics.VLLM)
    truth = strict_reference(model.config, *args, numerics=Numerics.UNQUANTIZED)
    tolerance = SNR_TOLERANCE_DB * max(1, (SNR_TOLERANCE_TOKENS / t) ** 0.5)
    snr_kernel, snr_vllm = test_util.snr_db(got, truth), test_util.snr_db(
        vllm, truth
    )
    print(
        f"{name} T={t}: SNR vs unquantized kernel={snr_kernel:.2f} dB"
        f" vllm={snr_vllm:.2f} dB",
        flush=True,
    )
    if model.config.fp8:
      self.assertGreaterEqual(snr_kernel, snr_vllm - tolerance)

  @parameterized.named_parameters(
      *((f"_{m.name}", m.name) for m in models.MODELS)
  )
  def test_no_relayout_from_rpa(self, name: str) -> None:
    """batched_rpa's raw output reaches fused_out_proj with no relayout."""
    model = models.BY_NAME[name]
    s = model.config
    t, t_rpa, seqs, page, pages = 64, 128, 8, 128, 64
    kv_layout = rpa_configs.KVLayout.SEQ_ALONG_LANE
    kv_cache = jnp.zeros(
        wrapper.get_kv_cache_shape(
            pages,
            page,
            s.num_kv_heads,
            s.head_dim,
            s.kv_dtype,
            kv_layout,
        ),
        s.kv_dtype,
    )
    q_lens = jnp.full((seqs,), t // seqs, jnp.int32)
    cu_q_lens = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(q_lens)])
    page_indices = jnp.arange(pages, dtype=jnp.int32)
    dist = jnp.array([0, 0, seqs], jnp.int32)
    x, residual, qkv_w, cos, sin = models.make_inputs(model, t_rpa)
    qkv_args = (x, residual, models.prepare_gqa_weights(s, qkv_w), cos, sin)
    _, out_w, _ = models.make_out_proj_inputs(model, t, t_rpa=t_rpa)
    hlo = (
        jax.jit(full_attention_block, static_argnums=(0, 1))
        .lower(
            model,
            t,
            qkv_args,
            out_w,
            kv_cache,
            q_lens,
            page_indices,
            cu_q_lens,
            dist,
        )
        .compile()
        .as_text()
    )
    assert hlo is not None

    lines, users = test_util.hlo_graph(hlo)
    rpa_mixed = [n for n in lines if n.startswith("RPAm")]
    self.assertLen(rpa_mixed, 1, hlo)
    out_proj_consumers, bad = set(), []
    frontier = [
        u
        for u in users[rpa_mixed[0]]
        if "index=0" in lines[u]
        or test_util.opcode(lines[u]) != "get-tuple-element"
    ]
    while frontier:
      n = frontier.pop()
      line = lines[n]
      op = test_util.opcode(line)
      if op in ("get-tuple-element", "bitcast", "copy-done") or (
          op == "copy-start" and test_util.is_same_layout_copy(line)
      ):
        frontier += users[n]
      elif op == "custom-call" and n.startswith("fused_out_proj"):
        out_proj_consumers.add(n)
      elif not line.lstrip().startswith("ROOT"):
        bad.append(line.strip()[:400])
    print(
        f"{name}: RPAm -> fused_out_proj consumers:", sorted(out_proj_consumers)
    )
    self.assertEmpty(bad, "relayouts between batched_rpa and fused_out_proj")
    self.assertNotEmpty(out_proj_consumers)
    self.assertNotIn("reshape_transpose_reshape", hlo)


if __name__ == "__main__":
  absltest.main()
