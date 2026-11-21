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
"""Single-kernel Pallas/Mosaic TPU forward for the scalar-gate GDN delta rule.

Each grid step `(batch, chunk)` computes the chunk's WY transform, its
triangular inverse, the output and the next recurrent state for all heads,
keeping the state in VMEM across chunks. Inputs and output stay in the
token-major `[B, T, ...]` layout; query/key heads may be fewer than value heads.
With residuals, the kernel also writes each chunk's entering state `h` and WY
inverse `akk` in the layouts of `GdnResiduals`, so the GDN backward consumes
them unchanged.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.kda import common
from tokamax._src.ops.kda import utils
from tokamax._src.ops.kda.pallas_mosaic_tpu_types import GdnResiduals

# Dimensions.
#   H_QK: query/key heads; H: value heads (a multiple of H_QK)
#   B: batch size; T: sequence length (a multiple of BT); BT: chunk size
#   K: key/query head dimension; V: value head dimension; NT: T // BT

CHUNK_SIZE = 64

_NT = (((2,), (2,)), ((0,), (0,)))  # [H, M, D] x [H, N, D] -> [H, M, N]
_NN = (((2,), (1,)), ((0,), (0,)))  # [H, M, D] x [H, D, N] -> [H, M, N]
_TN = (((1,), (1,)), ((0,), (0,)))  # [H, D, M] x [H, D, N] -> [H, M, N]


def _bmm(a, b, dims):
  """Head-batched bf16 matmul with fp32 accumulation."""
  return jax.lax.dot_general(
      a.astype(jnp.bfloat16),
      b.astype(jnp.bfloat16),
      dims,
      preferred_element_type=jnp.float32,
  )


def _split_heads(x, num_heads):
  """`[BT, N * D]` -> `[N, BT, D]` via lane-aligned slices."""
  d = x.shape[-1] // num_heads
  return jnp.stack([x[:, i * d : (i + 1) * d] for i in range(num_heads)])


def _fwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    gc_ref,
    beta_ref,
    o_ref,
    *refs,
    num_heads,
    num_qk_heads,
    scale,
    use_qk_l2norm,
):
  """One chunk of the GDN forward for all heads."""
  if len(refs) == 3:
    h_ref, akk_ref, h_scr = refs
  else:
    (h_scr,) = refs
    h_ref = akk_ref = None

  @pl.when(pl.program_id(1) == 0)
  def _():
    h_scr[...] = jnp.zeros_like(h_scr)

  q = _split_heads(q_ref[0], num_qk_heads)  # [H_QK, BT, K]
  k = _split_heads(k_ref[0], num_qk_heads)
  if use_qk_l2norm:
    q = utils.l2norm_fwd(q)[0]
    k = utils.l2norm_fwd(k)[0]
  repeats = num_heads // num_qk_heads
  if repeats > 1:
    q = jnp.repeat(q, repeats, axis=0)
    k = jnp.repeat(k, repeats, axis=0)
  v = _split_heads(v_ref[0], num_heads)  # [H, BT, V]
  gc = gc_ref[0].T  # [H, BT] log2-space chunk cumsum
  beta = beta_ref[0].T[..., None]  # [H, BT, 1]
  bt, v_dim = v.shape[1], v.shape[2]

  k_beta = k * beta
  akk_qk = _bmm(jnp.concatenate([k_beta, q], axis=1), k, _NT)
  row = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  causal = row >= col
  decay = jnp.exp2(jnp.where(causal, gc[:, :, None] - gc[:, None, :], -jnp.inf))
  akk = common.tri_inv_xor_dnc(
      jnp.where(row > col, akk_qk[:, :bt] * decay, 0.0)
  )  # (I + L)^-1
  if akk_ref is not None:
    akk_ref[:, 0] = akk.astype(akk_ref.dtype)

  eg = jnp.exp2(gc)[..., None]
  uw = _bmm(akk, jnp.concatenate([v * beta, k_beta * eg], axis=2), _NN)
  h = h_scr[...]  # state entering this chunk, [H, K, V]
  if h_ref is not None:
    h_ref[:, 0, 0] = h.astype(h_ref.dtype)
  wh_qh = _bmm(jnp.concatenate([uw[..., v_dim:], q * eg], axis=1), h, _NN)
  v_new = uw[..., :v_dim] - wh_qh[:, :bt]
  aqk = jnp.where(causal, akk_qk[:, bt:] * decay, 0.0)
  o = (wh_qh[:, bt:] + _bmm(aqk, v_new, _NN)) * scale
  o_ref[0] = jnp.concatenate(list(o), axis=-1).astype(o_ref.dtype)

  g_last = gc[:, bt - 1 :]
  h_scr[...] = jnp.exp2(g_last)[..., None] * h + _bmm(
      k * jnp.exp2(g_last - gc)[..., None], v_new, _TN
  )


def gdn_fwd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    gate: jax.Array,
    beta: jax.Array,
    *,
    scale: float,
    use_qk_l2norm: bool,
    use_beta_sigmoid_in_kernel: bool,
    return_residuals: bool,
    save_for_backward: bool,
) -> tuple[tuple[jax.Array, None], GdnResiduals | None]:
  """Scalar-gate GDN forward in one Pallas kernel.

  Args:
    query: `[B, T, H_QK, K]` queries.
    key: `[B, T, H_QK, K]` keys.
    value: `[B, T, H, V]` values; `H` is a multiple of `H_QK`.
    gate: `[B, T, H]` log-space decay.
    beta: `[B, T, H]` delta-rule learning rate (pre-sigmoid with
      `use_beta_sigmoid_in_kernel`).
    scale: Query scale.
    use_qk_l2norm: Whether to L2-normalize queries and keys in the kernel.
    use_beta_sigmoid_in_kernel: Whether to apply a sigmoid to `beta`.
    return_residuals: Whether to return `GdnResiduals`.
    save_for_backward: Whether the residuals carry `h`, `akk` and `g_cumsum`.

  Returns:
    `(([B, T, H, V] output, None), residuals or None)`, as
    `chunk_gdn_fwd_custom`.
  """
  b, t, h_qk, k_dim = query.shape
  _, _, h, v_dim = value.shape
  bt = CHUNK_SIZE
  nt = t // bt
  beta_act = (
      jax.nn.sigmoid(beta.astype(jnp.float32))
      if use_beta_sigmoid_in_kernel
      else beta
  )
  g_cumsum = utils.chunk_local_cumsum_vector(
      gate.transpose(2, 0, 1)[..., None], bt, scale=utils.RCP_LN2
  )  # [H, B, T, 1], as `GdnResiduals.g_cumsum`
  store = return_residuals and save_for_backward

  def tokens_spec(width):
    return pl.BlockSpec((1, bt, width), lambda i, s: (i, s, 0))

  out_shape = [jax.ShapeDtypeStruct((b, t, h * v_dim), query.dtype)]
  out_specs = [tokens_spec(h * v_dim)]
  if store:
    out_shape += [
        jax.ShapeDtypeStruct((h, b, nt, k_dim, v_dim), query.dtype),
        jax.ShapeDtypeStruct((h, b, t, bt), query.dtype),
    ]
    out_specs += [
        pl.BlockSpec((h, 1, 1, k_dim, v_dim), lambda i, s: (0, i, s, 0, 0)),
        pl.BlockSpec((h, 1, bt, bt), lambda i, s: (0, i, s, 0)),
    ]
  outs = pl.pallas_call(
      functools.partial(
          _fwd_kernel,
          num_heads=h,
          num_qk_heads=h_qk,
          scale=scale,
          use_qk_l2norm=use_qk_l2norm,
      ),
      out_shape=out_shape,
      grid=(b, nt),
      in_specs=[
          tokens_spec(h_qk * k_dim),
          tokens_spec(h_qk * k_dim),
          tokens_spec(h * v_dim),
          tokens_spec(h),
          tokens_spec(h),
      ],
      out_specs=out_specs,
      scratch_shapes=[pltpu.VMEM((h, k_dim, v_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=common.get_vmem_limit_bytes(),
      ),
      interpret=utils.get_interpret(),
      name="gdn_fused_fwd",
  )(
      query.reshape(b, t, h_qk * k_dim),
      key.reshape(b, t, h_qk * k_dim),
      value.reshape(b, t, h * v_dim),
      g_cumsum[..., 0].transpose(1, 2, 0),
      beta_act,
  )
  output = (outs[0].reshape(b, t, h, v_dim), None)
  if not return_residuals:
    return output, None
  residuals = GdnResiduals(
      qkv_conv=None,
      q=query if store else None,
      k=key if store else None,
      v=value if store else None,
      beta=beta_act.transpose(2, 0, 1),
      g_cumsum=g_cumsum if store else None,
      akk=outs[2] if store else None,
      h=outs[1] if store else None,
      initial_state=None,
      g_org=None,
      a_log=None,
      delta_time_bias=None,
      qkv_raw=None,
      conv_weight=None,
      conv_bias=None,
      segment_ids=None,
      g_dtype_marker=jnp.zeros((), dtype=gate.dtype),
      chunk_size=bt,
      beta_dtype=jnp.dtype(beta.dtype).name,
  )
  return output, residuals
