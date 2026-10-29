## Supported Ops and Hardware

<style>
table.ops-table {
  display: table !important;
  width: 100% !important;
  table-layout: fixed !important;
  border-collapse: collapse !important;
}
table.ops-table th,
table.ops-table td {
  border: 1px solid #dadce0 !important;
  padding: 8px 12px;
  vertical-align: middle !important;
  overflow-wrap: break-word;
  word-wrap: break-word;
  word-break: break-word;
}
table.ops-table code {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}
</style>

### Attention Variants

<table class="ops-table fixed" width="100%" style="display: table; width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col width="10%" style="width: 10%;">
    <col width="18%" style="width: 18%;">
    <col width="10%" style="width: 10%;">
    <col width="10%" style="width: 10%;">
    <col width="10%" style="width: 10%;">
    <col width="42%" style="width: 42%;">
  </colgroup>
  <thead>
    <tr>
      <th colspan="2" width="28%" align="left" style="width: 28%; text-align: left;">Op</th>
      <th width="10%" align="left" style="width: 10%; text-align: left;">HW</th>
      <th width="10%" align="center" style="width: 10%; text-align: center;">PyTorch TPU API</th>
      <th width="10%" align="center" style="width: 10%; text-align: center;">Bwd Pass</th>
      <th width="42%" align="left" style="width: 42%; text-align: left;">Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2" width="28%"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/splash_attention"><code>splash_attention</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Computes block-sparse Flash Attention (Splash Attention) for long-context Transformer training and inference.</td>
    </tr>
    <tr>
      <td colspan="2"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/batched_rpa"><code>batched_rpa</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Computes <a href="https://arxiv.org/abs/2309.06180">paged attention</a> (<code>batched_ragged_paged_attention</code>) over packed variable-length sequences and a paged KV cache for LLM inference serving.</td>
    </tr>
    <tr>
      <td colspan="2"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/causal_conv1d_gated_delta_rule"><code>causal_conv1d_gated_delta_rule</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Fuses causal 1D convolution with <a href="https://arxiv.org/abs/2412.06464">Gated DeltaNet (GDN)</a> or Kimi Delta Attention (KDA) linear attention state updates for inference in hybrid linear-attention models (e.g., Qwen 3.5, Kimi Linear).</td>
    </tr>
    <tr>
      <td rowspan="10" width="10%" align="center" valign="middle" style="width: 10%; vertical-align: middle; text-align: center;"><a href="https://arxiv.org/abs/2512.02556">DeepSeek Sparse Attention (DSA)</a></td>
      <td width="18%" style="width: 18%;"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/deepseek_rope"><code>deepseek_rope</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Applies interleaved <a href="https://arxiv.org/abs/2104.09864">Rotary Position Embedding (RoPE)</a> (optionally fused with RMSNorm or quantization) to queries and keys in DeepSeek-V4 attention layers.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/proj_and_save_state"><code>proj_and_save_state</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Projects hidden states into KV and score states and writes them to a paged state cache as the first stage of KV cache compression in DeepSeek-V4.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/dsv4_compressor"><code>dsv4_compressor</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Compresses windows of cached token states and writes the compressed representations into the KV cache for DeepSeek-V4.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/lightning_indexer"><code>lightning_indexer</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Computes lightweight query-key relevance scores over a paged KV cache and selects the top-k KV positions per query token (invoking <code>topk</code> at the end) in GLM-5.2, DeepSeek-V3.2, and DeepSeek-V4.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/topk"><code>topk</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Selects the top-k indices per row (<code>top_k</code>) across variable-length score rows on TPU SparseCore; invoked as the selection stage of <code>lightning_indexer</code> and also available standalone.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/csa_gather"><code>csa_gather</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Gathers and repacks compressed KV cache entries on TPU SparseCore for selected top-k token indices; invoked as the gather stage of <code>sparse_mla</code> and also available standalone.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/sparse_mla"><code>sparse_mla</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Computes Sparse Multi-Head Latent Attention (SparseMLA) over selected top-k compressed KV entries (invoking <code>csa_gather</code> first to gather them) for DeepSeek-V4 inference.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/masked_dense_mla"><code>masked_dense_mla</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Computes Multi-Head Latent Attention (MLA) restricted to a selected top-k subset of KV positions via a dense mask, used for shorter sequences in GLM-5.2, DeepSeek-V3.2, and DeepSeek-V4.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/mla_swa"><code>mla_swa</code></a> (experimental)</td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Computes Multi-Head Latent Sliding Window Attention over a paged KV cache (optionally returning partial softmax accumulators to combine with <code>sparse_mla</code>) for DeepSeek-V4 inference.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/grouped_lowrank_proj"><code>grouped_lowrank_proj</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Fuses inverse RoPE rotation, quantization, and grouped low-rank output projection in the DeepSeek-V4 attention output path.</td>
    </tr>
    <tr>
      <td colspan="2"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/attention"><code>attention</code></a></td>
      <td>GPU, TPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Computes standard scaled dot-product attention (<a href="https://arxiv.org/abs/2205.14135">FlashAttention</a>) for Transformer training and inference.</td>
    </tr>
    <tr>
      <td colspan="2"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/kda"><code>kda</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Computes <a href="https://arxiv.org/abs/2510.26692">Kimi Delta Attention (KDA)</a> (<code>kimi_delta_attention</code>), a linear attention mechanism with delta-rule state updates used in Kimi Linear.</td>
    </tr>
    <tr>
      <td colspan="2"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/flex_attention"><code>flex_attention</code></a> (experimental)</td>
      <td>GPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Computes scaled dot-product attention with user-defined score modification and masking functions (<a href="https://pytorch.org/blog/flexattention/">FlexAttention</a>).</td>
    </tr>
    <tr>
      <td colspan="2"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/mla"><code>mla</code></a> (experimental)</td>
      <td>TPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;"></td>
      <td>Computes <a href="https://arxiv.org/abs/2405.04434">Multi-Head Latent Attention (MLA)</a> over compressed latent KV caches for DeepSeek-V2, DeepSeek-V3, and DeepSeek-R1 inference.</td>
    </tr>
    <tr>
      <td colspan="2"><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/nsa"><code>nsa</code></a> (experimental)</td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Hardware-aligned hierarchical sparse attention combining compressed-block, selected-block, and sliding-window branches, implementing <a href="https://arxiv.org/abs/2502.11089">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>.</td>
    </tr>
  </tbody>
</table>

### Mixture-of-Experts (MoE) Components

<table class="ops-table fixed" width="100%" style="display: table; width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col width="28%" style="width: 28%;">
    <col width="10%" style="width: 10%;">
    <col width="10%" style="width: 10%;">
    <col width="10%" style="width: 10%;">
    <col width="42%" style="width: 42%;">
  </colgroup>
  <thead>
    <tr>
      <th width="28%" align="left" style="width: 28%; text-align: left;">Op</th>
      <th width="10%" align="left" style="width: 10%; text-align: left;">HW</th>
      <th width="10%" align="center" style="width: 10%; text-align: center;">PyTorch TPU API</th>
      <th width="10%" align="center" style="width: 10%; text-align: center;">Bwd Pass</th>
      <th width="42%" align="left" style="width: 42%; text-align: left;">Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/ragged_dot"><code>ragged_dot</code></a></td>
      <td>GPU, TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Performs grouped matrix multiplication (<a href="https://arxiv.org/abs/2211.15841">MegaBlocks</a>; drop-in for <code>jax.lax.ragged_dot</code>, with <code>gmm_v2</code> as one of its TPU backends) to compute expert projections over routed token groups.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/ragged_gather"><code>ragged_gather</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Gathers and permutes token activations into expert-grouped order during MoE token dispatch.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/ragged_gather_reduce"><code>ragged_gather_reduce</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Gathers expert outputs and performs a weighted reduction per token during the MoE combine (unpermute) step.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/router_topk"><code>router_topk</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Selects the top-k expert weights and indices per token from router scores for MoE routing.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/fused_moe"><code>fused_moe</code></a> (experimental)</td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Executes a fused end-to-end expert-parallel MoE layer (routing, dispatch, expert projections, activation, and collectives) for LLM inference.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/fused_moe_rs"><code>fused_moe_rs</code></a> (experimental)</td>
      <td>TPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;"></td>
      <td>Fuses MoE expert computation with cross-device reduce-scatter into a single TPU kernel for inference.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/routing_moe"><code>routing_moe</code></a> (experimental)</td>
      <td>TPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;"></td>
      <td>Implements a pipelined MoE layer that overlaps token gather/scatter with expert matrix multiplication.</td>
    </tr>
  </tbody>
</table>

### Miscellaneous

<table class="ops-table fixed" width="100%" style="display: table; width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col width="28%" style="width: 28%;">
    <col width="10%" style="width: 10%;">
    <col width="10%" style="width: 10%;">
    <col width="10%" style="width: 10%;">
    <col width="42%" style="width: 42%;">
  </colgroup>
  <thead>
    <tr>
      <th width="28%" align="left" style="width: 28%; text-align: left;">Op</th>
      <th width="10%" align="left" style="width: 10%; text-align: left;">HW</th>
      <th width="10%" align="center" style="width: 10%; text-align: center;">PyTorch TPU API</th>
      <th width="10%" align="center" style="width: 10%; text-align: center;">Bwd Pass</th>
      <th width="42%" align="left" style="width: 42%; text-align: left;">Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/linear_softmax_cross_entropy_loss"><code>linear_softmax_xent_loss</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Fuses the final linear projection with softmax cross-entropy loss (<code>linear_softmax_cross_entropy_loss</code>) to avoid materializing the full vocabulary logits tensor in memory.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/mhc"><code>mhc</code></a></td>
      <td>TPU</td>
      <td align="center" style="text-align: center;">✅</td>
      <td align="center" style="text-align: center;"></td>
      <td>Implements <a href="https://arxiv.org/abs/2512.24880">Manifold-Constrained Hyper-Connections (mHC)</a> (<code>mhc_pre</code>, <code>mhc_post</code>, and <code>mhc_fused_post_pre</code>) for mixing multiple residual streams across sublayers in DeepSeek-V4.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/gated_linear_unit"><code>gated_linear_unit</code></a></td>
      <td>GPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Computes a fused <a href="https://arxiv.org/abs/2002.05202">Gated Linear Unit</a> (e.g., SwiGLU, GeGLU) for Transformer feed-forward networks.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/normalization"><code>normalization</code></a></td>
      <td>GPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Computes <a href="https://arxiv.org/abs/1607.06450">Layer Normalization</a> (LayerNorm) or <a href="https://arxiv.org/abs/1910.07467">Root Mean Square Normalization</a> (RMSNorm).</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/triangle_multiplication"><code>triangle_multiplication</code></a></td>
      <td>GPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Computes the triangle multiplicative update on 2D pair representations for structure prediction models such as <a href="https://www.nature.com/articles/s41586-021-03819-2">AlphaFold 2</a> and AlphaFold 3.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/fused_fp8_matmul"><code>fused_fp8_matmul</code></a> (experimental)</td>
      <td>TPU</td>
      <td align="center" style="text-align: center;"></td>
      <td align="center" style="text-align: center;">✅</td>
      <td>Performs FP8 matrix multiplication with fused dynamic activation quantization and weight scaling.</td>
    </tr>
  </tbody>
</table>
