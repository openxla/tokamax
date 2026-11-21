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

"""Autotuning argument spec."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Literal

import jax
import numpy as np
from tokamax._src import numerics

# Tags are used to quickly identify different workloads for the same op.
# forward_only models are models that only require forward passes - meaning no
# vjp tuning is required.
# Primary arg specs support models that are considered important for key
# users. These are used for internal nightly benchmarking.
# 'ci_tests' arg specs are used for tests that run as part of the CI flow.
# 'primary' arg specs are a superset of 'ci_tests' arg specs.
type Tag = Literal['primary', 'forward_only', 'ci_tests', 'json']


@dataclasses.dataclass(frozen=True, slots=True)
class ArgSpec:
  """Argument specification for an op with metadata.

  Attributes:
    args: The argument specification.
    project: The project the argument specification comes from.
    name: The name of the argument specification.
    tags: Tags for the argument specification.
    excluded_platforms: Platforms to exclude from tuning for this argument
      specification. Each entry is either a platform (e.g. 'gpu', 'tpu'),
      excluding all devices of that platform, or an exact device kind (e.g.
      'TPU v5 lite'), excluding only that device kind.
  """

  args: dict[str, Any]
  project: str = ''
  name: str = ''
  tags: tuple[Tag, ...] = ()
  excluded_platforms: tuple[str, ...] = ()

  @property
  def full_name(self) -> str:
    """The full name (including project name)."""
    return '_'.join(filter(bool, (self.project, self.name)))

  def is_excluded_on(self, device: jax.Device) -> bool:
    """Returns whether this argument specification is excluded on `device`."""
    return any(
        p.lower() == device.platform or p == device.device_kind
        for p in self.excluded_platforms
    )

  @property
  def num_devices(self) -> int:
    """The number of devices required, i.e. the size of the largest mesh.

    Meshes are found in `args`, either directly or in `jax.NamedSharding`s of
    array-like arguments. Arg specs without a mesh require a single device.
    """
    sizes = [1]
    for x in jax.tree.leaves(self.args):
      sharding = getattr(x, 'sharding', None)
      if isinstance(x, (jax.sharding.Mesh, jax.sharding.AbstractMesh)):
        sizes.append(x.size)
      elif isinstance(sharding, jax.NamedSharding):
        sizes.append(sharding.mesh.size)
    return max(sizes)


def concretize_mesh(
    abstract_mesh: jax.sharding.AbstractMesh,
    devices: Sequence[jax.Device] | None = None,
) -> jax.sharding.Mesh:
  """Materializes an `AbstractMesh` over the first `mesh.size` devices."""
  devices = jax.devices() if devices is None else devices
  if len(devices) < abstract_mesh.size:
    raise ValueError(
        f'{abstract_mesh} requires {abstract_mesh.size} devices, but only'
        f' {len(devices)} are available.'
    )
  mesh_devices = np.asarray(devices[: abstract_mesh.size])
  return jax.sharding.Mesh(
      mesh_devices.reshape(abstract_mesh.axis_sizes),
      abstract_mesh.axis_names,
      axis_types=abstract_mesh.axis_types,
  )


def _get_concrete_mesh(
    abstract_mesh: jax.sharding.AbstractMesh,
    devices: Sequence[jax.Device],
    meshes: dict[jax.sharding.AbstractMesh, jax.sharding.Mesh],
) -> jax.sharding.Mesh:
  if abstract_mesh not in meshes:
    meshes[abstract_mesh] = concretize_mesh(abstract_mesh, devices)
  return meshes[abstract_mesh]


def _concretize_leaf(
    leaf: Any,
    devices: Sequence[jax.Device],
    meshes: dict[jax.sharding.AbstractMesh, jax.sharding.Mesh],
) -> Any:
  """Replaces any abstract mesh on a single argument leaf."""
  if isinstance(leaf, jax.sharding.AbstractMesh):
    return _get_concrete_mesh(leaf, devices, meshes)
  if isinstance(leaf, (jax.ShapeDtypeStruct, numerics.ArrayInitializer)):
    sharding = getattr(leaf, 'sharding', None)
    if isinstance(sharding, jax.NamedSharding) and isinstance(
        sharding.mesh, jax.sharding.AbstractMesh
    ):
      mesh = _get_concrete_mesh(sharding.mesh, devices, meshes)
      concrete_sharding = jax.NamedSharding(
          mesh, sharding.spec, memory_kind=sharding.memory_kind
      )
      if isinstance(leaf, numerics.ArrayInitializer):
        return leaf.with_sharding(concrete_sharding)
      return leaf.update(sharding=concrete_sharding)
  return leaf


def concretize_meshes(
    args: Mapping[str, Any], devices: Sequence[jax.Device] | None = None
) -> dict[str, Any]:
  """Replaces abstract meshes (and shardings using them) with concrete ones.

  Arg specs use `jax.sharding.AbstractMesh`es, as they are defined without
  access to devices. Each is replaced by a `jax.sharding.Mesh` over the first
  `mesh.size` devices, so arrays are initialized with the requested sharding.

  Args:
    args: The arguments.
    devices: The devices to use. Defaults to `jax.devices()`.

  Returns:
    A copy of `args` with concrete meshes.
  """
  devices = jax.devices() if devices is None else devices
  meshes: dict[jax.sharding.AbstractMesh, jax.sharding.Mesh] = {}
  is_leaf = lambda x: isinstance(
      x,
      (
          jax.sharding.AbstractMesh,
          jax.ShapeDtypeStruct,
          numerics.ArrayInitializer,
      ),
  )
  return jax.tree.map(
      lambda x: _concretize_leaf(x, devices, meshes),
      dict(args),
      is_leaf=is_leaf,
  )
