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
"""Board topology of the De-Dup dispatch and its static gateway tables.

De-Dup targets meshes whose devices are grouped into boards: devices of one
board are connected by fast intra-board links, and boards are connected by a
limited number of inter-board links. A token that several devices of a remote
board need crosses to that board once, to a gateway device, which forwards it
to the board's other devices that need it.

The caller describes the topology with `BoardTopology`: the device groups
(boards) as positions on the layer's expert-parallel mesh axis, and
optionally the gateway of every (source device, remote board) pair. By
default a source's gateway on a remote board is the device with the same
local index on that board ("mirror" gateways), which needs no knowledge of
the inter-board links.

Everything here is host numpy, built once per topology, and closed over as
compile-time constants by the on-device plan (`dedup_plan.py`).

Terms used by the tables:
  * ep devices, nb boards of board_size devices, board(d) = d // board_size.
  * gateway `gw[s, b]` of source s on remote board b.
  * served(G): the sources whose stage rows land on gateway G, in the order G
    forwards them (expected arrival order, see `_served_order_key`);
    sigma[s, G] = rank of s in served(G) ("stage slot"), S = max_G |served(G)|.
"""

from collections.abc import Sequence
import dataclasses
import functools

import numpy as np

# The largest board the on-device plan supports: the stage plan packs each
# token's per-board target mask into 8 bits and the per-target first waves
# into one int32 (4 bits per target).
MAX_BOARD_SIZE = 8
# The board sizes the layer is tested with. Other sizes up to MAX_BOARD_SIZE
# are refused until they are covered by tests.
SUPPORTED_BOARD_SIZES = (4, 8)


@dataclasses.dataclass(frozen=True, eq=False)
class TopologyTables:
  """Static De-Dup tables for one board topology.

  All arrays are int32 numpy; -1 marks "none".

  Attributes:
    num_devices: ep.
    board_size: Devices per board.
    num_boards: nb = ep // board_size.
    num_slots: S = max_G |served(G)|, at least 1 (stage slots per gateway).
    board: [ep] board(d) = d // board_size.
    gw: [ep, nb] gateway of source s on board b; -1 for b == board(s).
    sigma: [ep, ep] sigma[s, G] = served rank of s at gateway G, or -1.
    sigma_at_gw: [ep, nb] sigma[s, gw[s, b]]; -1 for b == board(s).
    served: [ep, S] served[G, sigma] = source device, -1 padded.
    n_served: [ep] |served(G)|.
  """

  num_devices: int
  board_size: int
  num_boards: int
  num_slots: int
  board: np.ndarray
  gw: np.ndarray
  sigma: np.ndarray
  sigma_at_gw: np.ndarray
  served: np.ndarray
  n_served: np.ndarray

  @property
  def max_served(self) -> int:
    """Stage slots per gateway (S)."""
    return self.num_slots

  def describe(self) -> str:
    """Returns a one-line summary of the tables."""
    return (
        f"ep={self.num_devices}, board_size={self.board_size},"
        f" nb={self.num_boards}, S={self.num_slots}, served per gateway"
        f" {sorted(set(self.n_served.tolist()))}"
    )


def _served_order_key(
    gateway: int, src: int, *, num_boards: int, board_size: int
) -> tuple[int, int]:
  """Forward order of the sources a gateway serves: expected arrival order.

  Every source issues its stage runs to the remote boards in rotated order
  (board(s) + 1, board(s) + 2, ...), so the source whose issue rank for the
  gateway's board is lowest lands first. Ties break by source.

  Args:
    gateway: Gateway device G.
    src: Served source device s.
    num_boards: nb.
    board_size: Devices per board.

  Returns:
    The sort key ((board(G) - board(s)) mod nb, s).
  """
  rel = (gateway // board_size - src // board_size) % num_boards
  return (rel, src)


def _as_int_tuples(
    rows: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
  return tuple(tuple(int(v) for v in row) for row in rows)


def _check_groups(groups: tuple[tuple[int, ...], ...]) -> None:
  """Validates the device groups of a `BoardTopology`.

  Args:
    groups: The device groups.

  Raises:
    ValueError: If the groups are not supported.
  """
  if len(groups) < 2:
    raise ValueError(
        f"De-Dup needs at least 2 device groups (boards); got {len(groups)}"
    )
  size = len(groups[0])
  if any(len(g) != size for g in groups):
    raise ValueError(
        "device groups must have equal sizes; got sizes"
        f" {[len(g) for g in groups]}"
    )
  if size not in SUPPORTED_BOARD_SIZES:
    raise ValueError(
        f"board size {size} is not supported; the De-Dup dispatch is"
        f" tested with board sizes {SUPPORTED_BOARD_SIZES}"
    )
  bad = [
      b
      for b, group in enumerate(groups)
      if group != tuple(range(b * size, (b + 1) * size))
  ]
  if bad:
    b = bad[0]
    raise ValueError(
        f"device group {b} is {groups[b]}; group b must be the contiguous mesh"
        f" positions [{size} * b, {size} * (b + 1)). Order the mesh axis so"
        " that the devices of one board are adjacent."
    )


def _check_gateways(
    gateways: tuple[tuple[int, ...], ...],
    groups: tuple[tuple[int, ...], ...],
) -> None:
  """Validates a gateway map against the (validated) device groups.

  Args:
    gateways: [ep][nb] gateway of every (source, board) pair.
    groups: The device groups.

  Raises:
    ValueError: If the map has the wrong shape or a gateway is off its board.
  """
  size, nb = len(groups[0]), len(groups)
  ep = size * nb
  if len(gateways) != ep or any(len(r) != nb for r in gateways):
    raise ValueError(f"gateways must be [{ep}][{nb}]")
  for s, row in enumerate(gateways):
    _check_gateway_row(s, row, groups)


def _check_gateway_row(
    s: int, row: tuple[int, ...], groups: tuple[tuple[int, ...], ...]
) -> None:
  """Validates source `s`'s gateways: each must be on its board."""
  own = s // len(groups[0])
  for b, g in enumerate(row):
    if b != own and g not in groups[b]:
      raise ValueError(
          f"gateway {g} of source {s} for board {b} is not on board {b}"
      )


@dataclasses.dataclass(frozen=True)
class BoardTopology:
  """Devices of the expert-parallel mesh axis grouped into boards.

  Attributes:
    device_groups: The boards, each a tuple of positions on the layer's mesh
      axis. The groups must have equal sizes in SUPPORTED_BOARD_SIZES, there
      must be at least 2 of them, and group b must hold the contiguous positions
      [b * board_size, (b + 1) * board_size): order the mesh axis so that the
      devices of one board are adjacent.
    gateways: Optional [ep][nb] gateway of every (source device, board) pair:
      the device on board b that receives the source's de-duplicated rows for
      that board. Entries for a source's own board are ignored. None uses mirror
      gateways (the device with the source's local index on board b).
  """

  device_groups: tuple[tuple[int, ...], ...]
  gateways: tuple[tuple[int, ...], ...] | None = None

  def __post_init__(self):
    groups = _as_int_tuples(self.device_groups)
    object.__setattr__(self, "device_groups", groups)
    if self.gateways is not None:
      object.__setattr__(self, "gateways", _as_int_tuples(self.gateways))
    _check_groups(groups)
    if self.gateways is not None:
      _check_gateways(self.gateways, groups)

  @classmethod
  def uniform(
      cls,
      num_devices: int,
      board_size: int,
      gateways: Sequence[Sequence[int]] | None = None,
  ) -> "BoardTopology":
    """Boards of `board_size` consecutive devices.

    Args:
      num_devices: Devices on the mesh axis (ep); a multiple of board_size.
      board_size: Devices per board.
      gateways: Optional gateway map, as the attribute.

    Returns:
      The topology.

    Raises:
      ValueError: If num_devices is not a multiple of board_size.
    """
    if board_size < 1 or num_devices % board_size:
      raise ValueError(
          f"num_devices={num_devices} is not a multiple of"
          f" board_size={board_size}"
      )
    groups = tuple(
        tuple(range(b * board_size, (b + 1) * board_size))
        for b in range(num_devices // board_size)
    )
    return cls(groups, None if gateways is None else _as_int_tuples(gateways))

  @property
  def board_size(self) -> int:
    """Devices per board."""
    return len(self.device_groups[0])

  @property
  def num_boards(self) -> int:
    """Number of boards."""
    return len(self.device_groups)

  @property
  def num_devices(self) -> int:
    """Devices on the mesh axis (ep)."""
    return self.board_size * self.num_boards

  def gateway(self, src: int, board: int) -> int:
    """Returns the gateway of `src` on remote board `board`."""
    if self.gateways is not None:
      return self.gateways[src][board]
    return board * self.board_size + src % self.board_size

  def tables(self) -> TopologyTables:
    """Returns the static De-Dup tables of this topology (cached)."""
    return build_topology_tables(self)


@functools.cache
def build_topology_tables(topology: BoardTopology) -> TopologyTables:
  """Builds the static De-Dup tables of a topology.

  Args:
    topology: The board topology.

  Returns:
    The tables.
  """
  ep = topology.num_devices
  bs = topology.board_size
  nb = topology.num_boards
  board = np.arange(ep, dtype=np.int32) // bs
  gw = np.full((ep, nb), -1, dtype=np.int32)
  served_lists: list[list[int]] = [[] for _ in range(ep)]
  for s in range(ep):
    for b in range(nb):
      if b == board[s]:
        continue
      g = topology.gateway(s, b)
      gw[s, b] = g
      served_lists[g].append(s)
  for g, lst in enumerate(served_lists):
    lst.sort(
        key=lambda s, g=g: _served_order_key(g, s, num_boards=nb, board_size=bs)
    )
  num_slots = max([1] + [len(lst) for lst in served_lists])
  served = np.full((ep, num_slots), -1, dtype=np.int32)
  sigma = np.full((ep, ep), -1, dtype=np.int32)
  for g, lst in enumerate(served_lists):
    for rank, s in enumerate(lst):
      served[g, rank] = s
      sigma[s, g] = rank
  sigma_at_gw = np.full((ep, nb), -1, dtype=np.int32)
  for s in range(ep):
    for b in range(nb):
      if gw[s, b] >= 0:
        sigma_at_gw[s, b] = sigma[s, gw[s, b]]
  n_served = np.array([len(lst) for lst in served_lists], dtype=np.int32)
  return TopologyTables(
      num_devices=ep,
      board_size=bs,
      num_boards=nb,
      num_slots=num_slots,
      board=board,
      gw=gw,
      sigma=sigma,
      sigma_at_gw=sigma_at_gw,
      served=served,
      n_served=n_served,
  )
