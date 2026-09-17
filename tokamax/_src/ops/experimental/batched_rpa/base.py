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
"""Base class and interface for Batched Ragged Paged Attention (bRPA)."""

import dataclasses
from typing import Any, ClassVar, TypeVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.batched_rpa import reference
from typing_extensions import override

_Config = TypeVar("_Config")

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray


@dataclasses.dataclass(frozen=True)
class BatchedRpa(op.Op[Any, tuple[jax.Array, jax.Array], None, _Config, Any]):
  """Batched Ragged Paged Attention (bRPA) base operator.

  Supports batched multi-head and grouped-query attention for mixed prefill and
  decode workloads with ragged sequence lengths and paged KV cache.
  """

  supports_batched_args_capture: ClassVar[bool] = False

  @jaxtyping.jaxtyped
  def bind(
      self,
      queries: Float[Array | AbstractArray, "total_q_tokens num_q_heads head_dim"],
      keys: Float[Array | AbstractArray, "total_q_tokens num_kv_heads head_dim"],
      values: Float[Array | AbstractArray, "total_q_tokens num_kv_heads head_dim"],
      kv_cache: Shaped[Array | AbstractArray, "total_pages page_size num_kv_heads_x2 head_dim_aligned"],
      kv_lens: Int[Array | AbstractArray, "max_num_seqs"],
      page_indices: Int[Array | AbstractArray, "total_page_indices"],
      cu_q_lens: Int[Array | AbstractArray, "max_num_seqs_plus_one"],
      distribution: Int[Array | AbstractArray, "3"],
      *,
      use_causal_mask: bool = True,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      out_dtype: Any = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Validates structural dimensions and binds input arguments."""
    total_q_tokens, num_q_heads, head_dim = queries.shape
    _, num_kv_heads, kv_head_dim = keys.shape
    assert head_dim == kv_head_dim, f"Head dim mismatch: {head_dim} vs {kv_head_dim}"
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    assert keys.shape == values.shape, f"Keys {keys.shape} != Values {values.shape}"
    assert cu_q_lens.shape[0] == kv_lens.shape[0] + 1, "cu_q_lens must have len(kv_lens) + 1"

    return super().bind(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        use_causal_mask=use_causal_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        out_dtype=out_dtype,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      queries: jax.Array,
      keys: jax.Array,
      values: jax.Array,
      kv_cache: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      *,
      use_causal_mask: bool = True,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      out_dtype: Any = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    """Invokes standard reference implementation."""
    del config, return_residuals  # Base class always executes pure JAX reference.
    del mask_value, v_scale  # Reference impl does not use these.
    return (
        reference.batched_ragged_paged_attention_reference(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            use_causal_mask=use_causal_mask,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            out_dtype=out_dtype,
            q_scale=q_scale,
            k_scale=k_scale,
        ),
        None,
    )
