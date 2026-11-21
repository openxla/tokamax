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

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import inspect
import math
from typing import Any, Literal

import immutabledict
import jax
import numpy as np
from tokamax._src import numerics
from tokamax._src import utils

# Tags are used to quickly identify different workloads for the same op.
# forward_only models are models that only require forward passes - meaning no
# vjp tuning is required.
# Primary arg specs support models that are considered important for key
# users. These are used for internal nightly benchmarking.
# 'ci_tests' arg specs are used for tests that run as part of the CI flow.
# 'primary' arg specs are a superset of 'ci_tests' arg specs.
type Tag = Literal['primary', 'forward_only', 'ci_tests', 'json']


@dataclasses.dataclass(frozen=True, slots=True)
class ShardMapSpec:
  """Execution wrapper for `shard_map`-native multi-device arg specs.

  `ArgSpec.args` keeps per-shard argument shapes so `op.bind(**spec.args)` and
  autotuning cache keys match what the op sees inside a caller's `shard_map`.
  `ShardMapSpec` describes how to lift those per-shard args onto `mesh` and wrap
  the op call in `jax.shard_map` for benchmarking and autotuning.

  Attributes:
    mesh: Abstract mesh over which the op executes.
    in_specs: Per-argument `PartitionSpec` mapping. Array-like arguments not
      listed default to replicated (`PartitionSpec()`).
    out_specs: Output `PartitionSpec` (or pytree of specs) passed to
      `jax.shard_map`.
    check_vma: Passed through to `jax.shard_map`.
  """

  mesh: jax.sharding.AbstractMesh
  in_specs: Mapping[str, jax.sharding.PartitionSpec | None] = (
      immutabledict.immutabledict()
  )
  out_specs: Any = jax.sharding.PartitionSpec()
  check_vma: bool = False

  def __post_init__(self):
    if not isinstance(self.in_specs, immutabledict.immutabledict):
      object.__setattr__(
          self, 'in_specs', immutabledict.immutabledict(self.in_specs)
      )


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
    shard_map: Optional `shard_map` execution wrapper for ops that run inside a
      caller's `jax.shard_map` rather than taking an explicit `mesh` argument.
  """

  args: dict[str, Any]
  project: str = ''
  name: str = ''
  tags: tuple[Tag, ...] = ()
  excluded_platforms: tuple[str, ...] = ()
  shard_map: ShardMapSpec | None = None

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

    Meshes are found in `shard_map` and `args`, either directly or in
    `jax.NamedSharding`s of array-like arguments. Arg specs without a mesh
    require a single device.
    """
    sizes = [1]
    if self.shard_map is not None:
      sizes.append(self.shard_map.mesh.size)
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


def _axis_size(
    axis: str | tuple[str, ...] | None, mesh: jax.sharding.Mesh
) -> int:
  if axis is None:
    return 1
  if isinstance(axis, str):
    return mesh.shape[axis]
  return math.prod(mesh.shape[a] for a in axis)


def _global_shape(
    local_shape: tuple[int, ...],
    pspec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh,
) -> tuple[int, ...]:
  return tuple(
      dim * _axis_size(pspec[i] if i < len(pspec) else None, mesh)
      for i, dim in enumerate(local_shape)
  )


def _lift_leaf(
    leaf: Any,
    pspec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh,
) -> Any:
  if not isinstance(leaf, (jax.ShapeDtypeStruct, numerics.ArrayInitializer)):
    return leaf
  global_shape = _global_shape(leaf.shape, pspec, mesh)
  sharding = jax.NamedSharding(mesh, pspec)
  if isinstance(leaf, numerics.ArrayInitializer):
    return leaf.with_shape_and_sharding(global_shape, sharding)
  return leaf.update(shape=global_shape, sharding=sharding)


def lift_shard_map_args(
    args: Mapping[str, Any],
    shard_map: ShardMapSpec,
    devices: Sequence[jax.Device] | None = None,
) -> dict[str, Any]:
  """Lifts per-shard abstract args to global sharded args on `shard_map.mesh`."""
  mesh = concretize_mesh(shard_map.mesh, devices)
  is_leaf = lambda x: isinstance(
      x, (jax.ShapeDtypeStruct, numerics.ArrayInitializer)
  )
  lifted = {}
  for name, val in args.items():
    pspec = shard_map.in_specs.get(name)
    leaf_pspec = jax.sharding.PartitionSpec() if pspec is None else pspec
    lifted[name] = jax.tree.map(
        lambda x, p=leaf_pspec: _lift_leaf(x, p, mesh),
        val,
        is_leaf=is_leaf,
    )
  return lifted


_ARRAY_TYPES = (jax.Array, np.ndarray, jax.ShapeDtypeStruct)


def wrap_shard_map(
    fn: Callable[..., Any],
    shard_map: ShardMapSpec,
    devices: Sequence[jax.Device] | None = None,
) -> Callable[..., Any]:
  """Wraps `fn` in `jax.shard_map` while preserving its call signature."""
  mesh = concretize_mesh(shard_map.mesh, devices)
  sig = fn.signature if hasattr(fn, 'signature') else inspect.signature(fn)

  def wrapped(*args, **kwargs):
    ba = sig.bind(*args, **kwargs)
    ba.apply_defaults()
    param_names = list(ba.arguments)
    arg_values = [ba.arguments[name] for name in param_names]
    arg_treedefs = []
    flat_values = []
    flat_pspecs = []
    for name, val in zip(param_names, arg_values, strict=True):
      leaves, treedef = jax.tree.flatten(val)
      pspec = shard_map.in_specs.get(name)
      leaf_pspec = jax.sharding.PartitionSpec() if pspec is None else pspec
      arg_treedefs.append((name, treedef, len(leaves)))
      flat_values.extend(leaves)
      flat_pspecs.extend([leaf_pspec] * len(leaves))

    is_array = lambda x: isinstance(x, _ARRAY_TYPES)
    arrays, other, merge = utils.split_merge(is_array, flat_values)
    in_specs = tuple(
        p for x, p in zip(flat_values, flat_pspecs, strict=True) if is_array(x)
    )

    def inner(*local_arrays):
      merged = merge(local_arrays, other)
      local_kwargs = {}
      offset = 0
      for name, treedef, count in arg_treedefs:
        local_kwargs[name] = treedef.unflatten(merged[offset : offset + count])
        offset += count
      local_ba = sig.bind(**local_kwargs)
      return fn(*local_ba.args, **local_ba.kwargs)

    return jax.shard_map(
        inner,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=shard_map.out_specs,
        check_vma=shard_map.check_vma,
    )(*arrays)

  wrapped.__signature__ = sig  # pyrefly: ignore[missing-attribute]
  return wrapped
