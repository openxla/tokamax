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
"""MultiHeadLatentAttention operator definition."""

from typing import Any, TypeVar
import jax
import jax.numpy as jnp
from tokamax._src.ops import op
from tokamax._src.ops.experimental.mla import reference
from tokamax._src.ops.experimental.mla import utils
from typing_extensions import override

_Config = TypeVar("_Config")


class MultiHeadLatentAttention(op.Op[Any, Any, None, _Config, Any]):
  """Tokamax operator for Multi-Head Latent Attention."""

  def bind(
      self,
      ql_nope: jax.Array,
      q_pe: jax.Array,
      new_kv_c: jax.Array,
      new_k_pe: jax.Array,
      cache_kv: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      debug_mode: bool = False,
      return_residuals: bool = False,
  ):

    utils.static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        debug_mode=debug_mode,
    )

    return super().bind(
        ql_nope=ql_nope,
        q_pe=q_pe,
        new_kv_c=new_kv_c,
        new_k_pe=new_k_pe,
        cache_kv=cache_kv,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        debug_mode=debug_mode,
        return_residuals=return_residuals,
    )

  @override
  def _fwd(
      self,
      ql_nope: jax.Array,
      q_pe: jax.Array,
      new_kv_c: jax.Array,
      new_k_pe: jax.Array,
      cache_kv: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      debug_mode: bool = False,
      return_residuals: bool = False,
      config: _Config | None = None,
  ):

    return (
        reference.mla_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        ),
        None,
    )
