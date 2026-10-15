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
"""Ragged gather reduce benchmark argument specifications."""

from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _make_argspec(
    *,
    name: str,
    project: str,
    input_size: int,
    hidden_size: int,
    reduce_group_size: int,
    num_rows: int | None = None,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    indices_dtype: jax.typing.DTypeLike = jnp.int32,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Make argspec for ragged gather reduce.

  Args:
    name: The name of the argument specification.
    project: The project the argument specification comes from.
    input_size: The number of routes, `num_tokens * reduce_group_size`.
    hidden_size: The hidden size.
    reduce_group_size: The number of routes summed into one token (the top-k).
    num_rows: The number of rows of `x`, the expert outputs. Defaults to
      `input_size`. Larger when the expert outputs are padded.
    dtype: The dtype of `x` and `topk_weights`.
    indices_dtype: The dtype of `indices`.
    tags: Tags for the argument specification.

  Returns:
    The argument specification.
  """
  num_rows = input_size if num_rows is None else num_rows
  return arg_spec.ArgSpec(
      args={
          "x": ShapeDtype((num_rows, hidden_size), dtype),
          "indices": numerics.RangedArrayInitializer(
              (input_size,), indices_dtype, 0, num_rows
          ),
          "topk_weights": ShapeDtype((input_size,), dtype),
          "valid_rows_mask": ShapeDtype((input_size,), jnp.bool_),
          "reduce_group_size": reduce_group_size,
      },
      project=project,
      name=name,
      tags=tags,
  )


def _qwen3_5_397b_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """The MoE output combine of Qwen3.5-397B on one device with TP8 and EP.

  Shapes are from the per-device MoE combine in vLLM xprofs (ISL 8192, OSL 1024,
  DP1TP8_EP) linked in
  `google/benchmarking/components/torchtpu_fused_moe_v2/fused_moe_benchmark.py`.
  With TP8 every device sees all tokens, so there are `num_tokens * 10` routes.
  The expert outputs are `(num_rows, 4096)` bf16, the GMM output, reduced to
  `(num_tokens, 4096)`. DP8TP1 does the combine inside `fused_ep_moe_v2`, so it
  has no ragged gather reduce.
  """
  top_k = 10
  hidden_size = 4096
  # name: (num_tokens, num_rows, tags). `num_rows` is the number of GMM output
  # rows; for 4 decode tokens the 40 routes are padded to 48.
  shapes: dict[str, tuple[int, int | None, tuple[arg_spec.Tag, ...]]] = {
      "tp8_decode_4": (4, 48, ("forward_only",)),
      "tp8_decode_8": (8, None, ("forward_only",)),
      "tp8_decode_16": (16, None, ("forward_only",)),
      "tp8_decode_32": (32, None, ("primary", "forward_only")),
      "tp8_decode_64": (64, None, ("primary", "ci_tests", "forward_only")),
      "tp8_decode_128": (128, None, ("primary", "forward_only")),
      "tp8_prefill_4096": (4096, None, ("forward_only",)),
      "tp8_prefill_8192": (8192, None, ("primary", "ci_tests", "forward_only")),
      "tp8_prefill_16384": (16384, None, ("primary", "forward_only")),
  }
  return tuple(
      _make_argspec(
          name=name,
          project="qwen3_5_397b",
          input_size=num_tokens * top_k,
          hidden_size=hidden_size,
          reduce_group_size=top_k,
          num_rows=num_rows,
          tags=tags,
      )
      for name, (num_tokens, num_rows, tags) in shapes.items()
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="8192x4096_group8_bf16",
        project="inference",
        input_size=8192,
        hidden_size=4096,
        reduce_group_size=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_argspec(
        name="1024x512_group4_bf16",
        project="inference",
        input_size=1024,
        hidden_size=512,
        reduce_group_size=4,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_argspec(
        name="16384x7168_group8_bf16",
        project="inference",
        input_size=16384,
        hidden_size=7168,
        reduce_group_size=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
) + _qwen3_5_397b_specs()
