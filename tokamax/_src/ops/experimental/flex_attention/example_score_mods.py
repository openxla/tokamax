# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Example score modifications for FlexAttention."""

import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src.ops.experimental.flex_attention import api

ScoreModInput = Float[Array, "B H T t"]


# Used by eg. Gemma 2: Improving Open Language Models at a Practical Size,
# https://arxiv.org/abs/2408.00118.
def generate_tanh_softcap(soft_cap: float) -> api.ScoreMod:
  """Returns a soft-cap score modification function."""

  def tanh_softcap(score: ScoreModInput) -> ScoreModInput:
    return soft_cap * jnp.tanh(score / soft_cap)

  return tanh_softcap
