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
"""CPU-only AOT compilation tests for routing MoE layer helpers."""

import dataclasses
import functools
import math
import re

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import sharding
from jax.experimental import mesh_utils
from jax.experimental import topologies
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import out_format
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.layer import modular

jax.config.parse_flags_with_absl()


@dataclasses.dataclass(frozen=True)
class _MeshSpec:
  topology_name: str
  axis_name: tuple[str, ...]
  mesh_shape: tuple[int, ...]

  def __post_init__(self) -> None:
    assert len(self.axis_name) == len(self.mesh_shape)


def _extract_hlo_instructions(hlo_text: str) -> list[tuple[bool, str, str]]:
  """Extracts `(is_root, shape, opcode)` across all computations in HLO text."""
  # Matches "%name = <shape> <opcode>(...)", with an optional "ROOT" prefix.
  matches = re.findall(
      r"^\s*(ROOT )?%\S+ = (.+?) ([\w-]+)\(", hlo_text, re.MULTILINE
  )
  return [(bool(root), shape, op) for root, shape, op in matches]


def _call_all_gather_activation(
    x_data: jax.Array,
    *,
    local_spec: spec.ActivationSpec,
    mesh: sharding.Mesh,
    axis_name: tuple[str, ...],
) -> arg.ActivationWrapper[jax.Array]:
  """Shards `x_data` across `axis_name` and runs `_all_gather_activation`."""

  def _body(x_local: jax.Array) -> arg.ActivationWrapper[jax.Array]:
    x_compact = arg.ActivationWrapper(data=x_local, scale=None, spec=local_spec)
    # pylint: disable-next=protected-access  # Unit testing private helper.
    return modular._all_gather_activation(x_compact, axis_name=axis_name)

  return jax.shard_map(
      _body,
      mesh=mesh,
      in_specs=sharding.PartitionSpec(axis_name),
      out_specs=sharding.PartitionSpec(),
      check_vma=False,
  )(x_data)


class MoEAotUnitTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    backend = jax.default_backend()
    if backend not in ("cpu", "tpu"):
      self.skipTest(
          f"TPU AOT compilation is not supported on {backend} backend."
      )

  @parameterized.product(
      shape=[(2, 512), (128, 4096)],
      act_format=[
          out_format.ResolvedActivationFormat(
              dtype=jnp.bfloat16.dtype, quant=None
          ),
          out_format.ResolvedActivationFormat(
              dtype=jnp.float8_e4m3fn.dtype, quant=None
          ),
          out_format.ResolvedActivationFormat(
              dtype=jnp.float8_e4m3fn.dtype,
              quant=out_format.ResolvedLaneStrided(
                  block_col_size=out_format.tile_size
              ),
          ),
          out_format.ResolvedActivationFormat(
              dtype=jnp.float32.dtype, quant=None
          ),
      ],
      mesh_spec=[
          _MeshSpec("v6e=2x2", ("x",), (4,)),
          _MeshSpec("v6e=2x2", ("x", "y"), (2, 2)),
          _MeshSpec("tpu7x=2x2x1", ("x",), (8,)),
          _MeshSpec("tpu7x=2x2x1", ("x", "y"), (4, 2)),
      ],
  )
  def test_all_gather_activation_has_no_reshape(
      self,
      shape: tuple[int, int],
      act_format: out_format.ResolvedActivationFormat,
      mesh_spec: _MeshSpec,
  ):
    local_rows, cols = shape
    topo = topologies.get_topology_desc(mesh_spec.topology_name, platform="tpu")
    mesh = sharding.Mesh(
        mesh_utils.create_device_mesh(mesh_spec.mesh_shape, topo.devices),
        mesh_spec.axis_name,
        (sharding.AxisType.Auto,) * len(mesh_spec.axis_name),
    )

    dtype = act_format.dtype
    quant = out_format.to_quant_spec(act_format.quant, tile_col_size=cols)
    local_spec = spec.ActivationSpec(local_rows, cols, dtype, quant)
    global_rows = local_rows * len(topo.devices)
    global_shape = (global_rows, *local_spec.physical_compact_shape[1:])
    x_data = jax.ShapeDtypeStruct(
        global_shape,
        dtype,
        sharding=sharding.NamedSharding(
            mesh, sharding.PartitionSpec(mesh_spec.axis_name)
        ),
    )

    lowered = jax.jit(
        functools.partial(
            _call_all_gather_activation,
            local_spec=local_spec,
            mesh=mesh,
            axis_name=mesh_spec.axis_name,
        )
    ).lower(x_data)
    expected_spec = dataclasses.replace(local_spec, row_size=global_rows)
    self.assertEqual(lowered.out_info.spec, expected_spec)
    self.assertEqual(lowered.out_info.data.shape, global_shape)

    hlo_text = lowered.compile().as_text()
    assert hlo_text is not None

    # Compact and 128-lane large layouts share the same physical arrangement,
    # so XLA lowers both reshapes around all-gather to zero-cost bitcasts.
    entries = _extract_hlo_instructions(hlo_text)
    opcodes = [op for _, _, op in entries]
    non_root_opcodes = [op for is_root, _, op in entries if not is_root]
    all_gather_shapes = [
        shape for _, shape, op in entries if op.startswith("all-gather")
    ]
    self.assertLen(all_gather_shapes, 1, msg=hlo_text)
    expected_large_dims = f"[{math.prod(global_shape[:-1])},{global_shape[-1]}]"
    self.assertIn(expected_large_dims, all_gather_shapes[0], msg=hlo_text)
    self.assertIn("bitcast", opcodes, msg=hlo_text)
    self.assertNotIn("reshape", opcodes, msg=hlo_text)
    self.assertNotIn("transpose", opcodes, msg=hlo_text)
    self.assertNotIn("copy", non_root_opcodes, msg=hlo_text)


if __name__ == "__main__":
  absltest.main()
