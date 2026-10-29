# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Base operator and JAX reference for Raven Gated Slot Attention (GSA)."""

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension
#   tt     tile extent of T (chunk_size)
#   nt     grid extent along T: nt = T // tt

import dataclasses
from typing import Any, TypeAlias, TypeVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from typing_extensions import override

_CHUNK_SIZE = 64
_Config = TypeVar("_Config")
State: TypeAlias = (
    Float[Array, "B H K M"]
    | tuple[
        Float[Array, "B H K M"] | None,
        Float[Array, "B H M V"] | None,
    ]
    | None
)
Output: TypeAlias = tuple[
    Float[Array, "B T H D"],
    State,
]
Residuals: TypeAlias = Any


def _chunkify(x: jax.Array, B: int, nt: int, tt: int) -> jax.Array:
  # [B, T, H, ...] -> [B, H, nt, tt, ...]
  _, _, H, *rest = x.shape
  x = x.reshape((B, nt, tt, H, *rest))
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _move_nt_front(x: jax.Array) -> jax.Array:
  return jnp.moveaxis(x, 2, 0)


def _stage1_scan_step(
    s_state: jax.Array,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
  q_i, g_i, y_diag_i, state_contrib_i, g_last_i = inputs
  y_off = jnp.matmul(
      q_i,
      s_state,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) * jnp.exp(g_i)
  y = y_diag_i + y_off
  s_new = s_state * jnp.exp(g_last_i)[None, :] + state_contrib_i
  return s_new.astype(s_state.dtype), y.astype(y_diag_i.dtype)


def _stage1_intra_one_plain(
    q_i: jax.Array, k_i: jax.Array, s_i: jax.Array, g_i: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  tt, _ = q_i.shape
  mask_upper = jnp.triu(jnp.ones((tt, tt), dtype=bool), k=1)
  qk = jnp.where(
      mask_upper,
      0.0,
      jnp.matmul(
          q_i,
          k_i.T,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      ),
  )
  diff = jnp.where(
      mask_upper[:, :, None], 0.0, g_i[:, None, :] - g_i[None, :, :]
  )
  decay = jnp.exp(diff)
  y_diag = jnp.sum(qk[:, :, None] * decay * s_i[None, :, :], axis=1)
  g_last = g_i[-1]
  mask_last = (jnp.arange(tt) == (tt - 1))[:, None]
  decay_tail = jnp.exp(jnp.where(mask_last, 0.0, g_last[None, :] - g_i))
  state_contrib = jnp.matmul(
      k_i.T,
      s_i * decay_tail,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return y_diag, state_contrib, g_last[None, :]


def raven_stage1_reference(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    s: Float[Array, "B T H M"],
    g: Float[Array, "B T H M"],
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: Float[Array, "B H K M"] | None = None,
    output_final_state: bool = False,
) -> tuple[Float[Array, "B T H M"], Float[Array, "B H K M"] | None]:
  """Stage 1 reference: decay on slot output (M) axis."""
  B, T, H, K = q.shape
  _, _, _, M = s.shape
  tt = chunk_size
  nt = T // tt

  if scale is None:
    scale = float(K**-0.5)
  q_scaled = (q.astype(jnp.float32) * scale).astype(q.dtype)

  q_c = _chunkify(q_scaled.astype(jnp.float32), B, nt, tt)
  k_c = _chunkify(k.astype(jnp.float32), B, nt, tt)
  s_c = _chunkify(s.astype(jnp.float32), B, nt, tt)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  )

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_stage1_intra_one_plain)))
  y_diag, state_contrib, g_last_block = vmap_intra(q_c, k_c, s_c, g_c)
  g_last = g_last_block[..., 0, :]

  q_s, g_s, y_diag_s, state_contrib_s, g_last_s = map(
      _move_nt_front, (q_c, g_c, y_diag, state_contrib, g_last)
  )

  scan_body_batched = jax.vmap(jax.vmap(_stage1_scan_step))

  if initial_state is None:
    initial_recurrent_state = jnp.zeros((B, H, K, M), dtype=jnp.float32)
  else:
    initial_recurrent_state = initial_state.astype(jnp.float32)

  s_final, o_stacked = jax.lax.scan(
      scan_body_batched,
      initial_recurrent_state,
      (q_s, g_s, y_diag_s, state_contrib_s, g_last_s),
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(B, T, H, M).astype(s.dtype)
  final_state = s_final.astype(q.dtype) if output_final_state else None
  return o, final_state


def _stage2_scan_step(
    s_state: jax.Array,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
  qv_i, g_i, y_diag_i, state_contrib_i, g_last_i = inputs
  y_off = jnp.matmul(
      qv_i * jnp.exp(g_i),
      s_state,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  y = y_diag_i + y_off
  s_new = s_state * jnp.exp(g_last_i)[:, None] + state_contrib_i
  return s_new.astype(s_state.dtype), y.astype(y_diag_i.dtype)


def _stage2_intra_one_plain(
    qv_i: jax.Array, s_i: jax.Array, v_i: jax.Array, g_i: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  tt, _ = qv_i.shape
  mask_upper = jnp.triu(jnp.ones((tt, tt), dtype=bool), k=1)
  diff = jnp.where(
      mask_upper[:, :, None], 0.0, g_i[:, None, :] - g_i[None, :, :]
  )
  decay = jnp.exp(diff)
  aqk_raw = jnp.einsum(
      "tm,nm,tnm->tn",
      qv_i,
      s_i,
      decay,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  aqk = jnp.where(mask_upper, 0.0, aqk_raw)
  y_diag = jnp.matmul(
      aqk,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  g_last = g_i[-1]
  mask_last = (jnp.arange(tt) == (tt - 1))[:, None]
  decay_tail = jnp.exp(jnp.where(mask_last, 0.0, g_last[None, :] - g_i))
  state_contrib = jnp.matmul(
      (s_i * decay_tail).T,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return y_diag, state_contrib, g_last[None, :]


def raven_stage2_reference(
    qv: Float[Array, "B T H M"],
    s: Float[Array, "B T H M"],
    v: Float[Array, "B T H V"],
    g: Float[Array, "B T H M"],
    *,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: Float[Array, "B H M V"] | None = None,
    output_final_state: bool = False,
) -> tuple[Float[Array, "B T H V"], Float[Array, "B H M V"] | None]:
  """Stage 2 reference: decay on slot input (M) axis, output on value (V) axis."""
  B, T, H, M = qv.shape
  _, _, _, V = v.shape
  tt = chunk_size
  nt = T // tt

  qv_c = _chunkify(qv.astype(jnp.float32), B, nt, tt)
  s_c = _chunkify(s.astype(jnp.float32), B, nt, tt)
  v_c = _chunkify(v.astype(jnp.float32), B, nt, tt)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  )

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_stage2_intra_one_plain)))
  y_diag, state_contrib, g_last_block = vmap_intra(qv_c, s_c, v_c, g_c)
  g_last = g_last_block[..., 0, :]

  qv_s, g_s, y_diag_s, state_contrib_s, g_last_s = map(
      _move_nt_front, (qv_c, g_c, y_diag, state_contrib, g_last)
  )

  scan_body_batched = jax.vmap(jax.vmap(_stage2_scan_step))

  if initial_state is None:
    initial_recurrent_state = jnp.zeros((B, H, M, V), dtype=jnp.float32)
  else:
    initial_recurrent_state = initial_state.astype(jnp.float32)

  s_final, o_stacked = jax.lax.scan(
      scan_body_batched,
      initial_recurrent_state,
      (qv_s, g_s, y_diag_s, state_contrib_s, g_last_s),
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(B, T, H, V).astype(v.dtype)
  final_state = s_final.astype(v.dtype) if output_final_state else None
  return o, final_state


@jaxtyping.jaxtyped
def raven_gsa_reference(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    s: Float[Array, "B T H M"],
    g: Float[Array, "B T H M"],
    v: Float[Array, "B T H V"] | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: State = None,
    output_final_state: bool = False,
) -> Output:
  """Pure JAX reference for Raven Gated Slot Attention (Stage 1 or full 2-stage)."""
  if v is None:
    if isinstance(initial_state, tuple):
      s1_init = initial_state[0]
    else:
      s1_init = initial_state
    return raven_stage1_reference(
        q,
        k,
        s,
        g,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=s1_init,
        output_final_state=output_final_state,
    )

  if isinstance(initial_state, tuple):
    s1_init, s2_init = initial_state
  else:
    s1_init, s2_init = initial_state, None

  ok, s1_final = raven_stage1_reference(
      q,
      k,
      s,
      g,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=s1_init,
      output_final_state=output_final_state,
  )
  qv = jax.nn.softmax(ok.astype(jnp.float32), axis=-1).astype(v.dtype)
  out, s2_final = raven_stage2_reference(
      qv,
      s,
      v,
      g,
      chunk_size=chunk_size,
      initial_state=s2_init,
      output_final_state=output_final_state,
  )
  final_state = (s1_final, s2_final) if output_final_state else None
  return out, final_state


@dataclasses.dataclass(frozen=True, kw_only=True)
class RavenGsaVjp[C](op.Op[Any, dict[str, Any], None, C, None]):
  """Tokamax base VJP operator for Raven Gated Slot Attention (GSA)."""

  supports_symbolic_shapes = False

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      residuals: Residuals,
      out: Output,
      dout: Output,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      s: Float[Array, "B T H M"],
      g: Float[Array, "B T H M"],
      v: Float[Array, "B T H V"] | None = None,
      *,
      chunk_size: int = _CHUNK_SIZE,
      scale: float | None = None,
      initial_state: State = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[dict[str, Any], None]:
    del residuals, out, return_residuals, config

    def _ref_fn(q_, k_, s_, g_, v_, init_state_):
      return raven_gsa_reference(
          q_,
          k_,
          s_,
          g_,
          v_,
          chunk_size=chunk_size,
          scale=scale,
          initial_state=init_state_,
          output_final_state=output_final_state,
      )

    primals_out, vjp_fn = jax.vjp(_ref_fn, q, k, s, g, v, initial_state)
    grad_out, grad_final_state = (
        dout if isinstance(dout, tuple) else (dout, None)
    )
    if not output_final_state:
      cotangent_state = None
    elif grad_final_state is None:
      cotangent_state = jax.tree.map(jnp.zeros_like, primals_out[1])
    else:
      cotangent_state = grad_final_state

    dq, dk, ds, dg, dv, ds0 = vjp_fn((grad_out, cotangent_state))
    grads: dict[str, Any] = {"q": dq, "k": dk, "s": ds, "g": dg}
    if v is not None:
      grads["v"] = dv
    if initial_state is not None:
      grads["initial_state"] = ds0
    return grads, None


@dataclasses.dataclass(frozen=True, kw_only=True)
class RavenGsa[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Raven Gated Slot Attention (GSA).

  Supports both standalone Stage 1 slot-logit computation (when ``v is None``)
  and full two-stage Gated Slot Attention (Stage 1 -> slot softmax -> Stage 2
  when ``v`` is provided).
  """

  supports_symbolic_shapes = False

  def __post_init__(self) -> None:
    if self.vjp is None:
      object.__setattr__(self, "vjp", RavenGsaVjp())

  @override
  def bind(
      self,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      s: Float[Array, "B T H M"],
      g: Float[Array, "B T H M"],
      v: Float[Array, "B T H V"] | None = None,
      *,
      chunk_size: int = _CHUNK_SIZE,
      scale: float | None = None,
      initial_state: State = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if q.ndim != 4:
      raise ValueError(
          f"Expected 4D q (batch, seq_len, heads, dim), got {q.shape}."
      )
    B, T, H, K = q.shape
    if k.shape != (B, T, H, K):
      raise ValueError(f"Expected k shape {(B, T, H, K)}, got {k.shape}.")
    if s.ndim != 4 or s.shape[:3] != (B, T, H):
      raise ValueError(
          f"Expected 4D s with leading shape {(B, T, H)}, got {s.shape}."
      )
    M = s.shape[3]
    if g.shape != (B, T, H, M):
      raise ValueError(f"Expected g shape {(B, T, H, M)}, got {g.shape}.")
    if v is not None and (v.ndim != 4 or v.shape[:3] != (B, T, H)):
      raise ValueError(
          f"Expected 4D v with leading shape {(B, T, H)}, got {v.shape}."
      )
    if isinstance(initial_state, tuple):
      if len(initial_state) != 2:
        raise ValueError(
            f"Expected 2-tuple initial_state, got length {len(initial_state)}."
        )
      s1_init, s2_init = initial_state
      if s1_init is not None and s1_init.shape != (B, H, K, M):
        raise ValueError(
            f"Expected initial_state[0] shape {(B, H, K, M)}, got"
            f" {s1_init.shape}."
        )
      if (
          s2_init is not None
          and v is not None
          and s2_init.shape != (B, H, M, v.shape[3])
      ):
        raise ValueError(
            f"Expected initial_state[1] shape {(B, H, M, v.shape[3])}, got"
            f" {s2_init.shape}."
        )
    elif initial_state is not None:
      if initial_state.shape != (B, H, K, M):
        raise ValueError(
            f"Expected initial_state shape {(B, H, K, M)}, got"
            f" {initial_state.shape}."
        )

    if chunk_size <= 0:
      raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if T % chunk_size != 0:
      raise ValueError(
          f"seq_len ({T}) must be divisible by chunk_size ({chunk_size})."
      )

    if scale is None:
      scale = float(K**-0.5)

    return super().bind(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      s: Float[Array, "B T H M"],
      g: Float[Array, "B T H M"],
      v: Float[Array, "B T H V"] | None = None,
      *,
      chunk_size: int = _CHUNK_SIZE,
      scale: float | None = None,
      initial_state: State = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del config, return_residuals
    out = raven_gsa_reference(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return out, None
