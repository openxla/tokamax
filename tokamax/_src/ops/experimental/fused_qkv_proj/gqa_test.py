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
"""Tests of fused_qkv: the GQA attention prologue."""

import dataclasses
import itertools
import re

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jaxtyping import Array, BFloat16  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import test_utils as jtu
from tokamax._src.ops.batched_rpa.kernel import configs as rpa_configs
from tokamax._src.ops.batched_rpa.kernel import wrapper
from tokamax._src.ops.experimental.fused_qkv_proj import models
from tokamax._src.ops.experimental.fused_qkv_proj import reference
from tokamax._src.ops.experimental.fused_qkv_proj import test_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import kernel as gqa

CASES = [
    (f"_{m.name}_T{t}", m.name, t)
    for m, t in itertools.product(models.MODELS, (32, 64, 128, 1024, 16384))
]
KV_LAYOUTS = (
    rpa_configs.KVLayout.SEQ_ALONG_LANE,
    rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
)
LAYOUT_CASES = [
    (f"{case}_{layout.symbol}", *args, layout)
    for (case, *args), layout in itertools.product(CASES, KV_LAYOUTS)
]
OUTPUTS = ("q", "kv", "gate", "residual")
# test_accuracy: how far the kernel may fall below vLLM's SNR.
SNR_TOLERANCE_DB = 0.15
# test_accuracy: below this many tokens, the tolerance grows as 1/sqrt(T).
SNR_TOLERANCE_TOKENS = 128
# vLLM (torch) rounds every bf16 op; XLA must not elide those roundings.
strict_reference = jax.jit(
    reference.gqa_reference,
    static_argnums=0,
    static_argnames=("kv_layout", "numerics", "page_size"),
    compiler_options={"xla_allow_excess_precision": False},
)
Numerics = reference.Numerics


def run_kernel(
    model: models.Model,
    x: BFloat16[Array, "T K"],
    residual: BFloat16[Array, "T K"] | None,
    weights: models.QkvWeights,
    cos: BFloat16[Array, "T n_pairs"] | None,
    sin: BFloat16[Array, "T n_pairs"] | None,
    *,
    kv_layout: rpa_configs.KVLayout = rpa_configs.KVLayout.SEQ_ALONG_LANE,
    page_size: int = 128,
    tile_size: int | None = None,
) -> tuple[jax.Array | None, ...]:
  """fused_qkv on weights as loaded."""
  prepared = models.prepare_gqa_weights(model.config, weights)
  return gqa.fused_qkv(
      x,
      residual,
      *prepared,
      cos,
      sin,
      **models.qkv_kwargs(model.config),
      kv_layout=kv_layout,
      page_size=page_size,
      tile_size=tile_size,
  )


def fused_qkv_rpa(
    model: models.Model,
    kv_layout: rpa_configs.KVLayout,
    args: models.PreparedInputs,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    dist: jax.Array,
) -> tuple[jax.Array, jax.Array, BFloat16[Array, "T Hq*D"] | None]:
  """fused_qkv feeding batched_rpa directly (no prepare_inputs)."""
  s = model.config
  x, residual, weights, cos, sin = args
  q, kv, gate, _ = gqa.fused_qkv(
      x,
      residual,
      *weights,
      cos,
      sin,
      **models.qkv_kwargs(s),
      kv_layout=kv_layout,
  )
  t = x.shape[0]
  qs = jax.ShapeDtypeStruct((t, s.num_q_heads, s.head_dim), s.q_dtype)
  kvs = jax.ShapeDtypeStruct((t, s.num_kv_heads, s.head_dim), s.kv_dtype)
  out, kv_cache = wrapper.ragged_paged_attention(
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
  )
  return out, kv_cache, gate


def two_layers(
    model: models.Model,
    args: models.PreparedInputs,
    attn: BFloat16[Array, "T K"],
    mlp_w: BFloat16[Array, "K K"],
) -> tuple[jax.Array, ...]:
  """fused_qkv -> vLLM's post-attention add + RMSNorm -> MLP -> fused_qkv.

  The residual stream as a decoder carries it: produced by an XLA op, updated
  in fused_qkv, then in the post-attention norm, then read by the next layer.
  attn stands in for the attention output and mlp_w for the MLP. The MLP's
  matmul matters: XLA tiles the norm (and so the residual) for it.
  """
  x, residual, weights, cos, sin = args
  assert residual is not None
  kw = models.qkv_kwargs(model.config)
  residual = x + residual
  _, kv1, _, residual = gqa.fused_qkv(x, residual, *weights, cos, sin, **kw)
  residual = (attn.astype(jnp.float32) + residual.astype(jnp.float32)).astype(
      attn.dtype
  )
  h = residual.astype(jnp.float32)
  x = (h * jax.lax.rsqrt(jnp.mean(h * h, -1, keepdims=True) + 1e-6)).astype(
      attn.dtype
  )
  x = jnp.dot(x, mlp_w, preferred_element_type=jnp.float32).astype(attn.dtype)
  q2, kv2, _, residual = gqa.fused_qkv(x, residual, *weights, cos, sin, **kw)
  return kv1, q2, kv2, residual


class FusedQkvTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu" or not jtu.is_device_tpu_at_least(
        version=7
    ):
      self.skipTest("Expect TPUv7+")

  def assert_exact(
      self,
      label: str,
      got: tuple[jax.Array | None, ...],
      want: tuple[jax.Array | None, ...],
  ) -> None:
    """Asserts each output matches bit for bit; prints the largest diff."""
    for out, g, w in zip(OUTPUTS, got, want):
      self.assertEqual(g is None, w is None, out)
      if g is None:
        continue
      g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
      self.assertEqual(g.shape, w.shape, out)
      diff = np.abs(g - w)
      first = np.unravel_index(diff.argmax(), diff.shape)
      print(
          f"{label} {out}: max_abs_diff={diff.max():.3e} "
          f"mismatches={(diff > 0).sum()}/{diff.size} "
          f"worst@{first} got={g[first]} want={w[first]}",
          flush=True,
      )
      self.assertLess(diff.max(), 1e-5, out)

  @parameterized.named_parameters(*LAYOUT_CASES)
  def test_kernel_matches_reference(
      self, name: str, t: int, kv_layout: rpa_configs.KVLayout
  ) -> None:
    """The kernel computes exactly Numerics.KERNEL."""
    model = models.BY_NAME[name]
    args = models.make_exact_inputs(model, t)
    self.assert_exact(
        f"{name} T={t} {kv_layout}",
        run_kernel(model, *args, kv_layout=kv_layout),
        strict_reference(model.config, *args, kv_layout=kv_layout),
    )

  @parameterized.named_parameters(
      *((f"_{m.name}", m.name) for m in models.MODELS)
  )
  def test_kernel_matches_reference_bf16_rope(self, name: str) -> None:
    """With rope_precision=bf16, the kernel still computes Numerics.KERNEL."""
    model = models.BY_NAME[name]
    if model.config.rope is None:
      self.skipTest("no RoPE")
    model = dataclasses.replace(
        model,
        config=dataclasses.replace(
            model.config, rope_precision=model_config.RopePrecision.BF16
        ),
    )
    args = models.make_exact_inputs(model, 128)
    self.assert_exact(
        f"{name} bf16 RoPE",
        run_kernel(model, *args),
        strict_reference(model.config, *args),
    )

  @parameterized.named_parameters(
      *((f"_{m.name}", m.name) for m in models.MODELS if m.config.fp8)
  )
  def test_fp32_rope_at_least_as_accurate(self, name: str) -> None:
    """fp32 RoPE's q and gate are no further from UNQUANTIZED than bf16's."""
    model = models.BY_NAME[name]
    if model.config.rope is None:
      self.skipTest("no RoPE")
    bf16 = dataclasses.replace(
        model,
        config=dataclasses.replace(
            model.config, rope_precision=model_config.RopePrecision.BF16
        ),
    )
    args = models.make_inputs(model, 1024)
    truth = strict_reference(model.config, *args, numerics=Numerics.UNQUANTIZED)
    got32, got16 = run_kernel(model, *args), run_kernel(bf16, *args)
    for out, g32, g16, e in zip(OUTPUTS, got32, got16, truth):
      if g32 is None or g16 is None or out in ("residual", "kv"):
        continue
      snr32 = test_util.snr_db(g32, e)
      snr16 = test_util.snr_db(g16, e)
      print(f"{name} {out}: SNR fp32 RoPE={snr32:.3f} dB bf16={snr16:.3f} dB")
      self.assertGreaterEqual(snr32, snr16, out)

  @parameterized.named_parameters(*(
      (f"_{m.name}_T{t}_tile{tile}", m.name, t, tile)
      for m, (t, tile) in itertools.product(
          models.MODELS, ((64, 64), (128, 128), (384, 128))
      )
  ))
  def test_kv_padded_to_page_size(self, name: str, t: int, tile: int) -> None:
    """SEQ_ALONG_LANE KV is zero-padded to the page size past the grid."""
    model = models.BY_NAME[name]
    layout, page_size = rpa_configs.KVLayout.SEQ_ALONG_LANE, 256
    args = models.make_exact_inputs(model, t)
    self.assert_exact(
        f"{name} T={t} tile={tile} page_size={page_size}",
        run_kernel(
            model, *args, kv_layout=layout, page_size=page_size, tile_size=tile
        ),
        strict_reference(
            model.config, *args, kv_layout=layout, page_size=page_size
        ),
    )

  @parameterized.named_parameters(
      *((f"_{m.name}", m.name) for m in models.MODELS)
  )
  def test_kernel_numerics_match_vllm_when_exact(self, name: str) -> None:
    """Numerics.KERNEL and VLLM agree exactly on exact inputs.

    They differ only in where they round. With eps = 0, 1/rms is exactly 1 and
    nothing rounds, so a difference is a bug in the reference.
    """
    model = models.BY_NAME[name]
    config = dataclasses.replace(model.config, eps=0.0)
    args = models.make_exact_inputs(model, 1024)
    self.assert_exact(
        name,
        strict_reference(config, *args, numerics=Numerics.KERNEL),
        strict_reference(config, *args, numerics=Numerics.VLLM),
    )

  @parameterized.named_parameters(*CASES)
  def test_accuracy(self, name: str, t: int) -> None:
    """On random data, the kernel's SNR vs. UNQUANTIZED is close to vLLM's.

    The kernel's bf16 prep math costs up to ~0.1 dB, hence SNR_TOLERANCE_DB.
    Fewer tokens make the measured gap noisier (KV: -0.19 dB at T=32 vs -0.11
    dB at T>=128), so below SNR_TOLERANCE_TOKENS the tolerance grows as
    1/sqrt(T). bf16 configs have no activation quant, so they are only
    reported. The residual must match exactly.
    """
    model = models.BY_NAME[name]
    args = models.make_inputs(model, t)
    got = run_kernel(model, *args)
    vllm = strict_reference(model.config, *args, numerics=Numerics.VLLM)
    truth = strict_reference(model.config, *args, numerics=Numerics.UNQUANTIZED)
    tolerance = SNR_TOLERANCE_DB * max(1, (SNR_TOLERANCE_TOKENS / t) ** 0.5)
    for out, g, v, e in zip(OUTPUTS, got, vllm, truth):
      if g is None:
        continue
      if out == "residual":
        np.testing.assert_array_equal(
            np.asarray(g, np.float32), np.asarray(v, np.float32)
        )
        continue
      snr_kernel, snr_vllm = test_util.snr_db(g, e), test_util.snr_db(v, e)
      print(
          f"{name} T={t} {out}: SNR vs unquantized kernel={snr_kernel:.2f} dB"
          f" vllm={snr_vllm:.2f} dB",
          flush=True,
      )
      if model.config.fp8:
        self.assertGreaterEqual(snr_kernel, snr_vllm - tolerance, out)

  @parameterized.named_parameters(*(
      (f"_{m.name}_{layout.symbol}", m.name, layout)
      for m, layout in itertools.product(models.MODELS, KV_LAYOUTS)
  ))
  def test_no_relayout_into_rpa(
      self, name: str, kv_layout: rpa_configs.KVLayout
  ) -> None:
    """fused_qkv's q and kv reach batched_rpa with no relayout in between."""
    model = models.BY_NAME[name]
    s = model.config
    t, seqs, page, pages = 1024, 8, 128, 64
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
    x, residual, weights, cos, sin = models.make_inputs(model, t)
    args = (x, residual, models.prepare_gqa_weights(s, weights), cos, sin)
    hlo = (
        jax.jit(fused_qkv_rpa, static_argnums=(0, 1))
        .lower(
            model,
            kv_layout,
            args,
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
    producers = [n for n in lines if n.startswith("fused_qkv")]
    self.assertLen(producers, 1, hlo)
    rpa_consumers, bad, frontier = set(), [], list(users[producers[0]])
    while frontier:
      n = frontier.pop()
      line = lines[n]
      op = test_util.opcode(line)
      if op in ("get-tuple-element", "bitcast", "copy-done") or (
          op == "copy-start" and test_util.is_same_layout_copy(line)
      ):
        frontier += users[n]
      elif op == "custom-call" and n.startswith(("RPA", "rpa_prefetch_start")):
        rpa_consumers.add(n)
      elif not line.lstrip().startswith("ROOT"):
        bad.append(line.strip()[:400])
    print(f"{name}: fused_qkv -> RPA consumers:", sorted(rpa_consumers))
    self.assertEmpty(bad, "relayouts between fused_qkv and RPA")
    self.assertNotEmpty(rpa_consumers)

  @parameterized.named_parameters(*(
      (f"_{m.name}_T{t}", m.name, t)
      for m, t in itertools.product(models.MODELS, (64, 1024))
      if m.config.residual
  ))
  def test_no_relayout_of_residual(self, name: str, t: int) -> None:
    """The residual stream crosses fused_qkv in one tiling: no copies."""
    model = models.BY_NAME[name]
    x, residual, weights, cos, sin = models.make_inputs(model, t)
    args = (
        x,
        residual,
        models.prepare_gqa_weights(model.config, weights),
        cos,
        sin,
    )
    hlo = (
        jax.jit(two_layers, static_argnums=0)
        .lower(
            model,
            args,
            jnp.full_like(x, 0.5),
            jnp.full((x.shape[1], x.shape[1]), 0.01, x.dtype),
        )
        .compile()
        .as_text()
    )
    assert hlo is not None
    lines, _ = test_util.hlo_graph(hlo)
    stream = re.escape(f"bf16[{t},{x.shape[1]}]") + r"\{1,0:(T\(\d+,128\))"
    # (residual in, residual out) tiling of each fused_qkv, in program order.
    tiles = []
    for n, line in lines.items():
      if not n.startswith("fused_qkv"):
        continue
      result, operands = line.split(" = ", 1)[1].split(" custom-call(", 1)
      residual_in = operands.split(",")[1].split("*/")[-1].strip().lstrip("%")
      tiling_in = re.search(stream, lines[residual_in].split(" = ", 1)[1])
      # The residual is the last output (gate may have the same shape).
      tiling_out = re.findall(stream, result)
      assert tiling_in is not None and tiling_out, line
      print(
          f"{name} T={t} {n}: residual in {tiling_in[1]} out {tiling_out[-1]}"
      )
      tiles.append((tiling_in[1], tiling_out[-1]))
    self.assertLen(tiles, 2)
    # Layer 1's residual reaches layer 2 (through the norm) in one tiling.
    self.assertEqual(tiles[0][1], tiles[1][0], "residual tiling across layers")
    copies = [
        line.strip()[:300]
        for line in lines.values()
        if re.search(r" copy(-start)?\(%", line)
        and re.search(stream, line)
        and not test_util.is_same_layout_copy(line)
    ]
    self.assertEmpty(copies, "relayouts of the residual stream")


if __name__ == "__main__":
  absltest.main()
