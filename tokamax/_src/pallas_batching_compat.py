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
"""Batching rule for jax's `with_memory_space_constraint`, so kernels using it can be `vmap`ed.

The v2 ragged-dot kernels pin `group_sizes`/`group_offset` into SMEM with
`pltpu.with_memory_space_constraint`. jax declares that primitive with only `def_impl`,
`def_abstract_eval` and an identity MLIR lowering, so any `jax.vmap` reaching it dies with:

    NotImplementedError: Batching rule for 'with_memory_space_constraint' not implemented

The primitive carries an annotation and moves no data -- its abstract eval is
`x.update(memory_space=...)` and its lowering is `return [x]` -- so batching it is simply "apply it
to the batched array and leave the batch dimension where it is".

SCOPE, and it matters: this is only HALF of what `vmap` over these kernels needs. Behind it sits
`mpmd_map` (which `pl.kernel` lowers to), whose batching rule is registered but rejects any batch
dimension != 1:

    NotImplementedError: mpmd_map only supports batching with a batch dimension of 1, got 4

Measured on v7x / jax 0.11.1 by pinning tokamax at 64435bb, which predates the
`with_memory_space_constraint` calls and therefore behaves as if this rule already existed: `vmap`
still fails, one layer down. So this module does not by itself make the kernels `vmap`able -- it
removes the shallower of two independent blockers, and becomes useful once `mpmd_map` gains real
batching support.

One consequence worth stating: under `vmap` the array pinned into SMEM grows by the batch size
(`[num_groups]` -> `[batch, num_groups]`). SMEM is small, so a large enough batch will not fit. The
rule does not police that; the failure surfaces as a Mosaic SMEM allocation error rather than
silently misbehaving.
"""

from typing import Any

from jax._src.interpreters import batching
from jax._src.pallas import core as pallas_core

_PRIM = pallas_core.with_memory_space_constraint_p


def _with_memory_space_constraint_batching_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], *, memory_space: Any
):
  """Applies the constraint to the batched operand, leaving the batch dimension in place."""
  (x,), (batch_dim,) = args, dims
  return _PRIM.bind(x, memory_space=memory_space), batch_dim


def _has_rule() -> bool:
  """Whether jax already provides a rule. The registries are proxies, so probe by lookup."""
  for registry in (
      batching.primitive_batchers,
      getattr(batching, "fancy_primitive_batchers", {}),
  ):
    try:
      if registry.get(_PRIM) is not None:
        return True
    except AttributeError:
      try:
        if _PRIM in dict(registry):
          return True
      except Exception:  # pylint: disable=broad-except
        pass
  return False


def register() -> bool:
  """Installs the rule unless jax has one. Returns whether we installed it.

  Idempotent, and deliberately yields to jax: once a jax release ships its own rule this becomes a
  no-op rather than shadowing it.
  """
  if _has_rule():
    return False
  batching.primitive_batchers[_PRIM] = _with_memory_space_constraint_batching_rule
  return True
