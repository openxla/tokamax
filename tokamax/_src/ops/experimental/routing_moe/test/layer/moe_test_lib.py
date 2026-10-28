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
"""Layouts, input sharding, and runner for the MoE layer tests."""

import dataclasses
import functools
import math
from typing import Literal, Self

import jax
from jax import sharding
import jax.numpy as jnp
from tokamax._src import test_utils
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.layer import modular
from tokamax._src.ops.experimental.routing_moe.layer import monolithic
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan_lib
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs
from tokamax._src.ops.experimental.routing_moe.test.common import test_reference
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes
from tokamax._src.ops.experimental.routing_moe.test.common import test_util

P = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh

Mode = Literal["modular", "monolithic"]


@dataclasses.dataclass(frozen=True)
class WeightLayout:
  """How one layer weight is split over the mesh.

  Attributes:
    use_ep: Shard the experts (expert parallel) instead of the intermediate
      dimension (tensor parallel).
    use_sp: Shard the tokens on input and output (sequence parallel).
    fold_shared_expert: Append the shared expert to the routed experts and route
      to it with a sigmoid gate. Only valid under TP, and only for the entries
      that take an external top-k.
    num_cores: Devices in the mesh. If None, all available devices are used.
  """

  use_ep: bool = False
  use_sp: bool = False
  fold_shared_expert: bool = False
  num_cores: int | None = None

  @property
  def tp_axis(self) -> str | None:
    """Mesh axis for tensor parallelism, or None under EP."""
    return None if self.use_ep else test_inputs.MESH_AXIS

  @property
  def ep_axis(self) -> str | None:
    """Mesh axis for expert parallelism, or None under TP."""
    return test_inputs.MESH_AXIS if self.use_ep else None

  @property
  def reduction_axis(self) -> str:
    """Mesh axis over which per-shard partial outputs are summed."""
    return test_inputs.MESH_AXIS


def _interleave_gate_up(w: jax.Array, n_shards: int) -> jax.Array:
  """Reorders `[.., gate | up]` columns so each TP shard gets both halves."""
  gate, up = jnp.split(w, 2, axis=-1)
  shards = w.shape[:-1] + (n_shards, -1)
  interleaved = jnp.concat([gate.reshape(shards), up.reshape(shards)], axis=-1)
  return interleaved.reshape(w.shape)


def _shard_inputs(
    moe_in: test_inputs.MoEInputs, mesh: Mesh, layout: WeightLayout
) -> test_inputs.MoEInputs:
  """Places `moe_in` on `mesh` as the layer expects for `layout`."""
  m = test_inputs.MESH_AXIS
  n_shards = mesh.shape[m]
  if not layout.use_ep:
    moe_in = dataclasses.replace(
        moe_in,
        w1=_interleave_gate_up(moe_in.w1, n_shards),
        w1_bias=None
        if moe_in.w1_bias is None
        else _interleave_gate_up(moe_in.w1_bias, n_shards),
    )
  if moe_in.w1_shared is not None:
    moe_in = dataclasses.replace(
        moe_in, w1_shared=_interleave_gate_up(moe_in.w1_shared, n_shards)
    )

  def _maybe_put[T](x: T, *spec: str | None) -> T:
    return jax.tree.map(
        lambda v: jax.device_put(v, NamedSharding(mesh, P(*spec))), x
    )

  act = (m, None) if layout.use_sp else (None, None)
  if layout.use_ep:
    w1 = w2 = w1_scale = w2_scale = w1_bias = w2_bias = (m,)
  else:
    w1, w2 = (None, None, m), (None, m, None)
    w1_scale = (None, None, None, m)
    # Because w2 is sharded along its contracting dimension while per-channel
    # quantization has a single scale across all TP shards, the scale is
    # replicated unless sub-channel blocks are present.
    w2_scale = (None, None)
    if moe_in.w2_scale is not None and moe_in.w2_scale.shape[1] > 1:
      w2_scale = (None, m)
    w1_bias, w2_bias = (None, None, m), (None, None)

  return test_inputs.MoEInputs(
      hidden_states=_maybe_put(moe_in.hidden_states, *act),
      gating_output=_maybe_put(moe_in.gating_output, *act),
      w1=_maybe_put(moe_in.w1, *w1),
      w2=_maybe_put(moe_in.w2, *w2),
      w1_scale=_maybe_put(moe_in.w1_scale, *w1_scale),
      w2_scale=_maybe_put(moe_in.w2_scale, *w2_scale),
      w1_bias=_maybe_put(moe_in.w1_bias, *w1_bias),
      w2_bias=_maybe_put(moe_in.w2_bias, *w2_bias),
      w1_shared=_maybe_put(moe_in.w1_shared, None, None, m),
      w2_shared=_maybe_put(moe_in.w2_shared, None, m, None),
      w1_shared_scale=_maybe_put(moe_in.w1_shared_scale, None, None, None, m),
      w2_shared_scale=_maybe_put(moe_in.w2_shared_scale, None, None),
      w_shared_gate=_maybe_put(moe_in.w_shared_gate, None, None),
  )


def _align_intermediate(
    shape: test_shapes.MoEShape, layout: WeightLayout, num_cores: int
) -> test_shapes.MoEShape:
  """Aligns `intermediate_size` to a multiple of 128 per TP shard."""
  lane_size = 128
  n_shards = 1 if layout.use_ep else num_cores
  per_shard_intermediate = shape.intermediate_size // n_shards
  aligned_per_shard = math.ceil(per_shard_intermediate / lane_size) * lane_size
  aligned_intermediate = aligned_per_shard * n_shards
  return dataclasses.replace(shape, intermediate_size=aligned_intermediate)


def _make_raw_inputs(
    shape: test_shapes.MoEShape,
    num_tokens: int,
    layout: WeightLayout,
    *,
    seed: int,
    active_expert_ratio: float = 1.0,
) -> test_inputs.MoEInputs:
  """Generates, folds, and quantizes the unsharded inputs of one layer call."""
  if layout.fold_shared_expert:
    assert shape.has_shared_expert and not layout.use_ep
  moe = test_inputs.MoEInputs.generate(
      shape, num_tokens, seed=seed, active_expert_ratio=active_expert_ratio
  )
  if layout.fold_shared_expert:
    moe = moe.fold_shared_expert()
  if shape.w_dtype is not None:
    moe = moe.quantize(shape.w_dtype, shape.quant_block_size)
  return moe


def _shared_expert(
    moe: test_inputs.MoEInputs,
    *,
    mesh: Mesh,
    layout: WeightLayout,
    activation: str,
    reduce_output: bool,
) -> jax.Array:
  """Computes the TP-sharded shared expert output on `mesh`."""
  assert moe.w1_shared is not None
  assert moe.w2_shared is not None
  assert moe.w_shared_gate is not None
  axis = layout.reduction_axis
  act_spec = P(axis, None) if layout.use_sp else P(None, None)
  if reduce_output and not layout.use_sp:
    out_spec = P(None, None)
  else:
    out_spec = P(axis, None)

  w1_scale_spec = (
      None if moe.w1_shared_scale is None else P(None, None, None, axis)
  )
  w2_scale_spec = None if moe.w2_shared_scale is None else P(None, None)

  def _local_fn(
      x: jax.Array,
      w1: jax.Array,
      w2: jax.Array,
      w1_scale: jax.Array | None,
      w2_scale: jax.Array | None,
      w_gate: jax.Array,
  ) -> jax.Array:
    if layout.use_sp:
      x = jax.lax.all_gather(x, axis_name=axis, axis=0, tiled=True)
    w1_wrap = arg.make_weight_wrapper(w1, w1_scale, fused_gate_up=True)
    w2_wrap = arg.make_weight_wrapper(w2, w2_scale)
    group_sizes = jnp.array([x.shape[0]], dtype=jnp.int32)
    down = test_reference.reference_gmm(
        x, (w1_wrap, w2_wrap), group_sizes, fuse_act=activation
    )
    gate_logits = jnp.matmul(x, w_gate, preferred_element_type=jnp.float32)
    out = down * jax.nn.sigmoid(gate_logits).astype(x.dtype)
    if reduce_output:
      if layout.use_sp:
        out = jax.lax.psum_scatter(
            out, axis_name=axis, scatter_dimension=0, tiled=True
        )
      else:
        out = jax.lax.psum(out, axis_name=axis)
    return out

  return jax.shard_map(
      _local_fn,
      mesh=mesh,
      in_specs=(
          act_spec,
          P(None, None, axis),
          P(None, axis, None),
          w1_scale_spec,
          w2_scale_spec,
          P(None, None),
      ),
      out_specs=out_spec,
      check_vma=False,
  )(
      moe.hidden_states,
      moe.w1_shared,
      moe.w2_shared,
      moe.w1_shared_scale,
      moe.w2_shared_scale,
      moe.w_shared_gate,
  )


def _reduce(out: jax.Array, *, mesh: Mesh, layout: WeightLayout) -> jax.Array:
  """Sums the per-shard partial outputs, scattered by token under SP."""
  axis = layout.reduction_axis
  if layout.use_sp:
    reduce_fn = functools.partial(
        jax.lax.psum_scatter, axis_name=axis, scatter_dimension=0, tiled=True
    )
    out_spec = P(axis)
  else:
    reduce_fn = functools.partial(jax.lax.psum, axis_name=axis)
    out_spec = P(None)
  return jax.shard_map(
      reduce_fn, mesh=mesh, in_specs=P(axis), out_specs=out_spec
  )(out)


@jax.jit(
    static_argnames=(
        "mode",
        "shape",
        "layout",
        "mesh",
        "remote_scatter",
        "reduce_output",
    )
)
def _run_moe(
    moe: test_inputs.MoEInputs,
    *,
    mode: Mode,
    shape: test_shapes.MoEShape,
    layout: WeightLayout,
    mesh: Mesh,
    remote_scatter: bool = False,
    reduce_output: bool = True,
) -> jax.Array:
  """Runs execution mode on routed and optional shared experts, then reduces."""
  routed_kwargs = dict(
      w1=moe.w1,
      w2=moe.w2,
      w1_scale=moe.w1_scale,
      w2_scale=moe.w2_scale,
      w1_bias=moe.w1_bias,
      w2_bias=moe.w2_bias,
      mesh=mesh,
      activation=shape.activation,
      tp_axis=layout.tp_axis,
      ep_axis=layout.ep_axis,
      use_sp=layout.use_sp,
      reduce_output=reduce_output,
  )
  moe_plan = moe_plan_lib.RoutingMoEPlan(remote_scatter=remote_scatter)

  match mode:
    case "monolithic":
      assert not layout.fold_shared_expert
      out = monolithic.gated_routing_moe(
          moe.hidden_states,
          moe.gating_output,
          topk=shape.top_k,
          renormalize=shape.renormalize,
          scoring_fn="softmax",
          moe_plan=moe_plan,
          **routed_kwargs,
      )
    case "modular":
      topk_weights, topk_indices = test_reference.compute_topk(
          moe, shape, fold_shared_expert=layout.fold_shared_expert
      )
      out = modular.routing_moe(
          moe.hidden_states,
          topk_weights,
          topk_indices,
          moe_plan=moe_plan,
          **routed_kwargs,
      )

  if moe.w1_shared is not None:
    out += _shared_expert(
        moe,
        mesh=mesh,
        layout=layout,
        activation=shape.activation,
        reduce_output=reduce_output,
    )
  if not reduce_output:
    out = _reduce(out, mesh=mesh, layout=layout)
  return out


@dataclasses.dataclass(frozen=True)
class MoECase:
  """Prepared inputs, mesh, and runner for one MoE layer test case.

  Attributes:
    raw: Unsharded layer inputs for the XLA reference.
    sharded: Mesh-sharded layer inputs for `_run_moe`.
    shape: Model shape with `intermediate_size` aligned per TP shard.
    layout: Sharding layout with `num_cores` resolved.
    mesh: Device mesh for `sharded`.
  """

  raw: test_inputs.MoEInputs
  sharded: test_inputs.MoEInputs
  shape: test_shapes.MoEShape
  layout: WeightLayout
  mesh: Mesh

  def run(
      self,
      mode: Mode,
      *,
      remote_scatter: bool = False,
      reduce_output: bool = True,
  ) -> jax.Array:
    """Runs `mode` on `sharded` and blocks until the output is ready."""
    out = _run_moe(
        self.sharded,
        mode=mode,
        shape=self.shape,
        layout=self.layout,
        mesh=self.mesh,
        remote_scatter=remote_scatter,
        reduce_output=reduce_output,
    )
    return jax.block_until_ready(out)

  def assert_close(self, actual: jax.Array) -> None:
    """Asserts that `actual` matches `reference_moe` on `raw`."""
    expected = test_reference.reference_moe(
        self.raw, self.shape, fold_shared_expert=self.layout.fold_shared_expert
    )
    rtol = 2e-2 if self.shape.w_dtype is None else 5e-2
    atol = rtol * float(jnp.abs(expected).max())
    test_utils.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  @classmethod
  def make_case(
      cls,
      case: test_cases.Case,
      layout: WeightLayout = WeightLayout(),
      *,
      seed: int = 1234,
  ) -> Self:
    """Builds and shards the layer inputs of `case` for `layout`."""
    num_devices = jax.device_count()
    num_cores = layout.num_cores or (
        case.ep_size if case.ep_size > 1 else num_devices
    )
    layout = dataclasses.replace(layout, num_cores=num_cores)
    aligned = _align_intermediate(case.shape, layout, num_cores)
    tp_shards = 1 if layout.use_ep else num_cores
    test_util.check_moe_supported(
        case,
        per_shard_intermediate=aligned.intermediate_size // tp_shards,
        num_cores=num_cores,
    )
    mesh = test_inputs.make_mesh((1, num_cores))
    raw = _make_raw_inputs(
        aligned,
        case.num_tokens,
        layout,
        seed=seed,
        active_expert_ratio=case.active_expert_ratio,
    )
    sharded = jax.block_until_ready(_shard_inputs(raw, mesh, layout))
    return cls(
        raw=raw, sharded=sharded, shape=aligned, layout=layout, mesh=mesh
    )
