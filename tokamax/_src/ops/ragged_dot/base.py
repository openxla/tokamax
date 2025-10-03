# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Ragged dot base class."""

from collections.abc import Callable, Sequence
import dataclasses
import types
from typing import Any, TypeVar

import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np
from pydantic_core import core_schema as cs
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
Residuals = types.NoneType
QuantizedArray = quantization.QuantizedArray
CanonicalPrecision = precision_lib.CanonicalPrecision


DEFAULT_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [1]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)


_STATIC = dataclasses.field(metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GroupSizes:
  """A group sizes array with representative values.

  `ragged_dot` performance is sensitive to the distribution of the group sizes,
  but we cannot serialize the actual values (as they are runtime determined, and
  will vary from one step to the next). Instead, we serialize a representative
  distribution of group sizes. This allows `ragged_dot` to be benchmarked /
  autotuned with representative data.
  """
  value: jax.Array
  representative_value: tuple[int, ...] = _STATIC

  def __post_init__(self):
    (num_groups,) = self.value.shape
    if len(self.representative_value) != num_groups:
      raise ValueError(
          "Representative value must have the same length as the group sizes."
      )

    if not isinstance(self.value, jax.Array):
      value = np.asarray(self.representative_value, np.int32)
      object.__setattr__(self, "value", value)

    if not np.issubdtype(self.value.dtype, np.integer):
      raise ValueError("Group sizes must be integers.")

  def __jax_array__(self):
    return self.value

  def __eq__(self, other) -> bool:
    return isinstance(other, GroupSizes) and (
        self.representative_value == other.representative_value
    )

  def __hash__(self) -> int:
    return hash(self.representative_value)

  @classmethod
  def __get_pydantic_core_schema__(cls, source, handler):
    del handler  # Unused.
    assert source is cls
    serialize = lambda x: x.representative_value
    validate = lambda x: cls(jax.ShapeDtypeStruct([len(x)], jnp.int32), x)  # pytype: disable=wrong-arg-types
    from_ints_schema = cs.chain_schema([
        cs.tuple_schema([cs.int_schema()], variadic_item_index=0),
        cs.no_info_plain_validator_function(validate),
    ])
    instance_schema = cs.is_instance_schema(cls)
    return cs.json_or_python_schema(
        json_schema=from_ints_schema,
        python_schema=cs.union_schema([instance_schema, from_ints_schema]),
        serialization=cs.plain_serializer_function_ser_schema(serialize),
    )


class RaggedDot(op.Op[Any, jax.Array, Residuals, _Config, _Key]):
  """Ragged dot base class.

  For use in MegaBlocks-style models: https://arxiv.org/abs/2211.15841.
  """

  def bind(
      self,
      lhs: jax.Array | QuantizedArray,
      rhs: jax.Array | QuantizedArray,
      *,
      group_sizes: jax.Array | GroupSizes | Sequence[int],
      ragged_dot_dimension_numbers: (
          jax.lax.RaggedDotDimensionNumbers | None
      ) = None,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if ragged_dot_dimension_numbers is None:
      # TODO: Support batch dims on LHS and/or RHS?
      ragged_dot_dimension_numbers = DEFAULT_RAGGED_DOT_DIM_NUMS

    if isinstance(group_sizes, (tuple, list)):
      group_sizes = tuple(group_sizes)
      group_sizes = GroupSizes(jnp.array(group_sizes, jnp.int32), group_sizes)

    # TODO: Create representative values for other ragged dot dim numbers.
    if ragged_dot_dimension_numbers == DEFAULT_RAGGED_DOT_DIM_NUMS:
      if not isinstance(group_sizes, GroupSizes):
        representative_sizes = (lhs.shape[0] // rhs.shape[0],) * rhs.shape[0]
        group_sizes = GroupSizes(group_sizes, representative_sizes)

      # Checkify requires fix in JAX 0.8.0 to work with `custom_vjp`.
      if jax.version.__version_info__ >= (0, 8, 0):
        gs = group_sizes.value
        dbg_check = checkify.debug_check
        dbg_check(jnp.all(gs >= 0), "Negative group size.")
        dbg_check(jnp.sum(gs) <= lhs.shape[0], "Group size sum > num rows.")

    if preferred_element_type is not None:
      preferred_element_type = jnp.dtype(preferred_element_type)
    return super().bind(
        lhs,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        precision=precision_lib.canonicalize_precision(precision),
        preferred_element_type=preferred_element_type,
        return_residuals=return_residuals,
    )

  def _fwd(
      self,
      lhs: jax.Array | QuantizedArray,
      rhs: jax.Array | QuantizedArray,
      *,
      group_sizes: jax.Array | GroupSizes,
      ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
      precision: CanonicalPrecision,
      preferred_element_type: jnp.dtype | None,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[jax.Array, Residuals]:
    del config  # Unused.

    if isinstance(lhs, QuantizedArray):
      lhs = lhs.recompose()

    if isinstance(rhs, QuantizedArray):
      rhs = rhs.recompose()

    if isinstance(group_sizes, GroupSizes):
      group_sizes = jnp.array(group_sizes)

    out = jax.lax.ragged_dot_general(
        lhs,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    return out, None


def vjp(
    residuals: Residuals,
    out: jax.Array,
    dout: jax.Array,
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    precision: CanonicalPrecision,
    preferred_element_type: jnp.dtype | None,
    dlhs_ragged_dot: Callable[..., jax.Array] = RaggedDot(),
    drhs_ragged_dot: Callable[..., jax.Array] = RaggedDot(),
) -> tuple[jax.Array, jax.Array]:
  """Ragged dot VJP."""
  del out, preferred_element_type  # Unused.
  assert residuals is None

  dot_dim_nums = ragged_dot_dimension_numbers.dot_dimension_numbers
  lhs_ragged = ragged_dot_dimension_numbers.lhs_ragged_dimensions
  rhs_group = ragged_dot_dimension_numbers.rhs_group_dimensions
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dot_dim_nums
  lhs_not_kept = (*lhs_batch, *lhs_contract)
  rhs_not_kept = (*rhs_batch, *rhs_contract, *rhs_group)
  lhs_kept = [i for i in range(lhs.ndim) if i not in lhs_not_kept]
  rhs_kept = [i for i in range(rhs.ndim) if i not in rhs_not_kept]

  assert len(lhs_batch) == len(rhs_batch)
  assert dout.ndim == len(lhs_batch) + len(lhs_kept) + len(rhs_kept)
  dout_batch = list(range(len(lhs_batch)))
  dout_lhs_kept = list(range(len(lhs_batch), len(lhs_batch) + len(lhs_kept)))
  dout_rhs_kept = list(range(len(lhs_batch) + len(lhs_kept), dout.ndim))

  dot_dim_nums = ((dout_rhs_kept, rhs_kept), (dout_batch, rhs_batch))
  dout_ragged = [(len(dout_batch) + lhs_kept.index(d)) for d in lhs_ragged]
  dlhs = dlhs_ragged_dot(
      dout,
      rhs,
      group_sizes=group_sizes,
      ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
          dot_dimension_numbers=dot_dim_nums,
          lhs_ragged_dimensions=dout_ragged,
          rhs_group_dimensions=rhs_group,
      ),
      precision=precision,
      preferred_element_type=lhs.dtype,
  )

  dot_dim_nums = ((lhs_kept, dout_lhs_kept), (lhs_batch, dout_batch))
  drhs = drhs_ragged_dot(
      lhs,
      dout,
      group_sizes=group_sizes,
      ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
          dot_dimension_numbers=dot_dim_nums,
          lhs_ragged_dimensions=lhs_ragged,
          rhs_group_dimensions=[],
      ),
      precision=precision,
      preferred_element_type=rhs.dtype,
  )
  return dlhs, drhs
