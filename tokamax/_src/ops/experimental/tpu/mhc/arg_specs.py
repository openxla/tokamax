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
"""mHC benchmark argument specifications."""

from typing import Any, Final

import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct

# DeepSeek-V4 shapes and gate constants (from its config.json; hc_post_alpha=2.0
# is hardcoded in the model).
_HC_MULT = 4
_HIDDEN_SIZE = 7168
_GATE_CONSTANTS: Final[dict[str, Any]] = dict(
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
)
# Decode batch, a mid-size batch and a prefill chunk.
_NUM_TOKENS = (16, 512, 8192)
_TAGS: Final[tuple[arg_spec.Tag, ...]] = ("primary", "ci_tests", "forward_only")


def _pre_args(num_tokens: int) -> dict[str, Any]:
  m, h = _HC_MULT, _HIDDEN_SIZE
  return dict(
      residual=ShapeDtype((num_tokens, m * h), jnp.bfloat16),
      fn=ShapeDtype((m * (m + 2), m * h), jnp.float32),
      hc_scale=ShapeDtype((3,), jnp.float32),
      hc_base=ShapeDtype((m * (m + 2),), jnp.float32),
      **_GATE_CONSTANTS,
  )


def _post_args(num_tokens: int) -> dict[str, Any]:
  m, h = _HC_MULT, _HIDDEN_SIZE
  return dict(
      x=ShapeDtype((num_tokens, h), jnp.bfloat16),
      residual=ShapeDtype((num_tokens, m * h), jnp.bfloat16),
      post_layer_mix=ShapeDtype((num_tokens, m), jnp.float32),
      comb_res_mix=ShapeDtype((num_tokens, m, m), jnp.float32),
  )


def _fused_args(num_tokens: int) -> dict[str, Any]:
  pre_args = _pre_args(num_tokens)
  del pre_args["residual"]
  return _post_args(num_tokens) | pre_args


def _name(num_tokens: int) -> str:
  return f"dsv4_t{num_tokens}_m{_HC_MULT}_h{_HIDDEN_SIZE}"


PRE_ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = tuple(
    arg_spec.ArgSpec(
        args=_pre_args(t), project="inference", name=_name(t), tags=_TAGS
    )
    for t in _NUM_TOKENS
)

POST_ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = tuple(
    arg_spec.ArgSpec(
        args=_post_args(t), project="inference", name=_name(t), tags=_TAGS
    )
    for t in _NUM_TOKENS
)

FUSED_POST_PRE_ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = tuple(
    arg_spec.ArgSpec(
        args=_fused_args(t), project="inference", name=_name(t), tags=_TAGS
    )
    for t in _NUM_TOKENS
)
