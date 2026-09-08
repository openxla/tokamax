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
import dataclasses
from typing import Any, Tuple, Mapping

from tokamax._src.ops import op as op_lib
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.attention import arg_specs as attention
from tokamax._src.ops.attention import base as attention_base
from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import arg_specs as mla
from tokamax._src.ops.experimental.mla import base as mla_base
from tokamax._src.ops.flex_attention import api as flex_attention_api
from tokamax._src.ops.flex_attention import arg_specs as flex_attention
from tokamax._src.ops.flex_attention import base as flex_attention_base
from tokamax._src.ops.gated_linear_unit import api as gated_linear_unit_api
from tokamax._src.ops.gated_linear_unit import arg_specs as gated_linear_unit
from tokamax._src.ops.gated_linear_unit import base as gated_linear_unit_base
from tokamax._src.ops.linear_softmax_cross_entropy_loss import api as linear_softmax_cross_entropy_loss_api
from tokamax._src.ops.linear_softmax_cross_entropy_loss import arg_specs as linear_softmax_cross_entropy_loss
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base as linear_softmax_cross_entropy_loss_base
from tokamax._src.ops.normalization import api as normalization_api
from tokamax._src.ops.normalization import arg_specs as normalization
from tokamax._src.ops.normalization import base as normalization_base
from tokamax._src.ops.ragged_dot import api as ragged_dot_api
from tokamax._src.ops.ragged_dot import arg_specs as ragged_dot
from tokamax._src.ops.ragged_dot import base as ragged_dot_base
from tokamax._src.ops.ragged_gather import api as ragged_gather_api
from tokamax._src.ops.ragged_gather import arg_specs as ragged_gather
from tokamax._src.ops.ragged_gather import base as ragged_gather_base
from tokamax._src.ops.ragged_gather_reduce import api as ragged_gather_reduce_api
from tokamax._src.ops.ragged_gather_reduce import arg_specs as ragged_gather_reduce
from tokamax._src.ops.ragged_gather_reduce import base as ragged_gather_reduce_base
from tokamax._src.ops.ragged_scatter import api as ragged_scatter_api
from tokamax._src.ops.ragged_scatter import arg_specs as ragged_scatter
from tokamax._src.ops.ragged_scatter import base as ragged_scatter_base
from tokamax._src.ops.triangle_multiplication import api as triangle_multiplication_api
from tokamax._src.ops.triangle_multiplication import arg_specs as triangle_multiplication
from tokamax._src.ops.triangle_multiplication import base as triangle_multiplication_base

@dataclasses.dataclass(frozen=True)
class RegisteredOp:
    name: str
    base_class: Any
    implementations: Mapping[str, Any]
    external_arg_specs: Tuple[Any, ...]
    numerical_reference: str | None = None
    has_vjp: bool = False

@dataclasses.dataclass(frozen=True)
class AutotuneArgSpecs:
  """Autotuning argument specs for an op."""
  attention: Tuple[Any, ...] = ()
  attention_vjp: Tuple[Any, ...] = ()
  flex_attention: Tuple[Any, ...] = ()
  gated_linear_unit: Tuple[Any, ...] = ()
  linear_softmax_cross_entropy_loss: Tuple[Any, ...] = ()
  linear_softmax_cross_entropy_loss_vjp: Tuple[Any, ...] = ()
  normalization: Tuple[Any, ...] = ()
  normalization_vjp: Tuple[Any, ...] = ()
  ragged_dot: Tuple[Any, ...] = ()
  ragged_gather: Tuple[Any, ...] = ()
  ragged_gather_reduce: Tuple[Any, ...] = ()
  ragged_scatter: Tuple[Any, ...] = ()
  triangle_multiplication: Tuple[Any, ...] = ()
  mla: Tuple[Any, ...] = ()

@dataclasses.dataclass(frozen=True)
class Implementations:
  """Implementations for all ops."""
  attention: Tuple[attention_base.DotProductAttention, ...] = ()
  attention_vjp: Tuple[attention_base.DotProductAttentionVjp, ...] = ()
  flex_attention: Tuple[flex_attention_base.FlexAttention, ...] = ()
  gated_linear_unit: Tuple[gated_linear_unit_base.GatedLinearUnit, ...] = ()
  linear_softmax_cross_entropy_loss: Tuple[linear_softmax_cross_entropy_loss_base.LinearSoftmaxCrossEntropyLoss, ...] = ()
  linear_softmax_cross_entropy_loss_vjp: Tuple[linear_softmax_cross_entropy_loss_base.LinearSoftmaxCrossEntropyLossVjp, ...] = ()
  normalization: Tuple[normalization_base.Normalization, ...] = ()
  normalization_vjp: Tuple[normalization_base.NormalizationVjp, ...] = ()
  ragged_dot: Tuple[ragged_dot_base.RaggedDot, ...] = ()
  ragged_gather: Tuple[ragged_gather_base.RaggedGather, ...] = ()
  ragged_gather_reduce: Tuple[ragged_gather_reduce_base.RaggedGatherReduce, ...] = ()
  ragged_scatter: Tuple[ragged_scatter_base.RaggedScatter, ...] = ()
  triangle_multiplication: Tuple[triangle_multiplication_base.TriangleMultiplication, ...] = ()
  mla: Tuple[mla_base.MultiHeadLatentAttention, ...] = ()

OPS = (
    RegisteredOp("attention", attention_base.DotProductAttention, attention_api.IMPLEMENTATIONS, attention.ARG_SPECS, numerical_reference="xla_chunked", has_vjp=True),
    RegisteredOp("flex_attention", flex_attention_base.FlexAttention, flex_attention_api.IMPLEMENTATIONS, flex_attention.ARG_SPECS),
    RegisteredOp("gated_linear_unit", gated_linear_unit_base.GatedLinearUnit, gated_linear_unit_api.IMPLEMENTATIONS, gated_linear_unit.ARG_SPECS),
    RegisteredOp("linear_softmax_cross_entropy_loss", linear_softmax_cross_entropy_loss_base.LinearSoftmaxCrossEntropyLoss, linear_softmax_cross_entropy_loss_api.IMPLEMENTATIONS, linear_softmax_cross_entropy_loss.ARG_SPECS, has_vjp=True),
    RegisteredOp("normalization", normalization_base.Normalization, normalization_api.IMPLEMENTATIONS, normalization.ARG_SPECS, has_vjp=True),
    RegisteredOp("ragged_dot", ragged_dot_base.RaggedDot, ragged_dot_api.IMPLEMENTATIONS, ragged_dot.ARG_SPECS, has_vjp=True),
    RegisteredOp("ragged_gather", ragged_gather_base.RaggedGather, ragged_gather_api.IMPLEMENTATIONS, ragged_gather.ARG_SPECS),
    RegisteredOp("ragged_gather_reduce", ragged_gather_reduce_base.RaggedGatherReduce, ragged_gather_reduce_api.IMPLEMENTATIONS, ragged_gather_reduce.ARG_SPECS),
    RegisteredOp("ragged_scatter", ragged_scatter_base.RaggedScatter, ragged_scatter_api.IMPLEMENTATIONS, ragged_scatter.ARG_SPECS),
    RegisteredOp("triangle_multiplication", triangle_multiplication_base.TriangleMultiplication, triangle_multiplication_api.IMPLEMENTATIONS, triangle_multiplication.ARG_SPECS),
    RegisteredOp("mla", mla_base.MultiHeadLatentAttention, mla_api.IMPLEMENTATIONS, mla.ARG_SPECS),
)
