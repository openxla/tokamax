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
from typing import Literal, Self

import jax
from jax import sharding
import jax.numpy as jnp
from tokamax._src import test_utils
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan_lib
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
AxisSpec = str | tuple[str, ...] | None


@dataclasses.dataclass(frozen=True)
class WeightLayout:
  """How one layer's weights and activations are sharded across a mesh.

  Attributes:
    sharding: Data-, expert-, and tensor-parallel shard counts.
    use_sp: Whether tokens are sharded on input and output (sequence parallel).
    fold_shared_expert: Whether to append the shared expert to the routed
      experts and route to it with a sigmoid gate. Only valid when `sharding.ep
      == 1` and for entries that take an external top-k.
  """

  sharding: test_cases.MoESharding = test_cases.MoESharding()
  use_sp: bool = False
  fold_shared_expert: bool = False


def _interleave_gate_up(w: jax.Array, n_shards: int) -> jax.Array:
  """Reorders `[.., gate | up]` columns so each TP shard gets both halves."""
  gate, up = jnp.split(w, 2, axis=-1)
  shards = w.shape[:-1] + (n_shards, -1)
  interleaved = jnp.concat([gate.reshape(shards), up.reshape(shards)], axis=-1)
  return interleaved.reshape(w.shape)


def _scale_block_axis(scale: jax.Array | None, axis: AxisSpec) -> AxisSpec:
  """Returns `axis` when `scale` has multiple intermediate blocks, else None."""
  # Because w2 is sharded along its contracting dimension while per-channel
  # quantization has a single scale across all TP shards, the scale is
  # replicated unless sub-channel blocks are present.
  if scale is None or scale.shape[1] == 1:
    return None
  return axis


def _shard_inputs(
    moe_in: test_inputs.MoEInputs, mesh: Mesh, layout: WeightLayout
) -> test_inputs.MoEInputs:
  """Places `moe_in` on `mesh` as the layer expects for `layout`."""
  sharding_spec = layout.sharding
  if sharding_spec.tp > 1:
    moe_in = dataclasses.replace(
        moe_in,
        w1=_interleave_gate_up(moe_in.w1, sharding_spec.tp),
        w1_bias=None
        if moe_in.w1_bias is None
        else _interleave_gate_up(moe_in.w1_bias, sharding_spec.tp),
    )
  if moe_in.w1_shared is not None and sharding_spec.num_cores > 1:
    moe_in = dataclasses.replace(
        moe_in,
        w1_shared=_interleave_gate_up(
            moe_in.w1_shared, sharding_spec.num_cores
        ),
    )

  def _maybe_put[T](x: T, spec: tuple[AxisSpec, ...]) -> T:
    return jax.tree.map(
        lambda v: jax.device_put(v, NamedSharding(mesh, P(*spec))), x
    )

  ep = layout.sharding.ep_axis
  tp = layout.sharding.tp_axis
  red = layout.sharding.reduction_axis
  act = (red, None) if layout.use_sp else (None, None)
  w2_block = _scale_block_axis(moe_in.w2_scale, tp)
  w2_shared_block = _scale_block_axis(moe_in.w2_shared_scale, red)

  return test_inputs.MoEInputs(
      hidden_states=_maybe_put(moe_in.hidden_states, act),
      gating_output=_maybe_put(moe_in.gating_output, act),
      w1=_maybe_put(moe_in.w1, (ep, None, tp)),
      w2=_maybe_put(moe_in.w2, (ep, tp, None)),
      w1_scale=_maybe_put(moe_in.w1_scale, (ep, None, None, tp)),
      w2_scale=_maybe_put(moe_in.w2_scale, (ep, w2_block, None, None)),
      w1_bias=_maybe_put(moe_in.w1_bias, (ep, None, tp)),
      w2_bias=_maybe_put(moe_in.w2_bias, (ep, None, None)),
      w1_shared=_maybe_put(moe_in.w1_shared, (None, None, red)),
      w2_shared=_maybe_put(moe_in.w2_shared, (None, red, None)),
      w1_shared_scale=_maybe_put(
          moe_in.w1_shared_scale, (None, None, None, red)
      ),
      w2_shared_scale=_maybe_put(
          moe_in.w2_shared_scale, (None, w2_shared_block, None, None)
      ),
      w_shared_gate=_maybe_put(moe_in.w_shared_gate, (None, None)),
  )


def _make_raw_inputs(
    shape: test_shapes.MoEShape,
    num_tokens: int,
    layout: WeightLayout,
    *,
    seed: int,
    active_expert_ratio: float = 1.0,
    max_to_mean_load: float = 1.0,
) -> test_inputs.MoEInputs:
  """Generates, folds, and quantizes the unsharded inputs of one layer call."""
  if layout.fold_shared_expert and (
      not shape.has_shared_expert or layout.sharding.ep > 1
  ):
    raise ValueError(
        "fold_shared_expert requires has_shared_expert and sharding.ep == 1."
    )
  moe = test_inputs.MoEInputs.generate(
      shape,
      num_tokens,
      seed=seed,
      active_expert_ratio=active_expert_ratio,
      max_to_mean_load=max_to_mean_load,
  )
  if layout.fold_shared_expert:
    moe = moe.fold_shared_expert()
  if shape.w_dtype is not None:
    moe = moe.quantize(shape.w_dtype, shape.w_quant_block_size)
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
  axis = layout.sharding.reduction_axis
  act_spec = P(axis, None) if layout.use_sp else P(None, None)
  if reduce_output and not layout.use_sp:
    out_spec = P(None, None)
  else:
    out_spec = P(axis, None)

  w1_scale_spec = (
      None if moe.w1_shared_scale is None else P(None, None, None, axis)
  )
  w2_scale_spec = None
  if moe.w2_shared_scale is not None:
    block_axis = _scale_block_axis(moe.w2_shared_scale, axis)
    w2_scale_spec = P(None, block_axis, None, None)

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
    out = test_reference.shared_expert(
        x, w1, w2, w1_scale, w2_scale, w_gate, activation=activation
    )
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
  axis = layout.sharding.reduction_axis
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
        "gmm_plan",
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
    gmm_plan: gmm_plan_lib.RoutingGmmPlan | None = None,
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
      tp_axis=layout.sharding.tp_axis,
      ep_axis=layout.sharding.ep_axis,
      use_sp=layout.use_sp,
      reduce_output=reduce_output,
      gmm_plan=gmm_plan,
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
    layout: Sharding layout of the layer weights and activations.
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
      gmm_plan: gmm_plan_lib.RoutingGmmPlan | None = None,
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
        gmm_plan=gmm_plan,
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
      layout: WeightLayout | None = None,
      *,
      remote_scatter: bool = False,
      seed: int = 1234,
  ) -> Self:
    """Builds and shards the layer inputs of `case` for `layout`."""
    if layout is None:
      layout = WeightLayout(case.sharding)
    else:
      case = dataclasses.replace(case, sharding=layout.sharding)
    test_util.check_moe_supported(case, remote_scatter=remote_scatter)
    aligned = dataclasses.replace(
        case.shape,
        intermediate_size=case.local_intermediate_size * layout.sharding.tp,
    )
    mesh = test_inputs.make_mesh(layout.sharding)
    raw = _make_raw_inputs(
        aligned,
        case.num_tokens,
        layout,
        seed=seed,
        active_expert_ratio=case.active_expert_ratio,
        max_to_mean_load=case.max_to_mean_load,
    )
    sharded = jax.block_until_ready(_shard_inputs(raw, mesh, layout))
    return cls(
        raw=raw, sharded=sharded, shape=aligned, layout=layout, mesh=mesh
    )
