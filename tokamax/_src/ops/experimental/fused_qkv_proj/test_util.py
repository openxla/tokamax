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
"""Helpers shared by the fused_qkv tests: SNR, test values and HLO graphs."""

import re

import jax
import jax.numpy as jnp
import numpy as np


def snr_db(got: jax.Array | np.ndarray, want: jax.Array | np.ndarray) -> float:
  """10 log10(|want|^2 / |got - want|^2), in f64."""
  got64, want64 = np.asarray(got, np.float64), np.asarray(want, np.float64)
  return 10 * np.log10(np.sum(want64**2) / np.sum((got64 - want64) ** 2))


def distinct(
    shape: tuple[int, ...], dtype: jax.typing.DTypeLike = jnp.float32
) -> jax.Array:
  """Random values: a misplaced row, lane or head is a mismatch."""
  key = jax.random.key(np.prod(shape))
  return jax.random.normal(key, shape, jnp.float32).astype(dtype)


def assert_equal(
    got: jax.Array | np.ndarray, want: jax.Array | np.ndarray
) -> None:
  """Asserts got == want exactly, compared in f32."""
  np.testing.assert_array_equal(
      np.asarray(got, np.float32), np.asarray(want, np.float32)
  )


def hlo_graph(hlo: str) -> tuple[dict[str, str], dict[str, list[str]]]:
  """Returns ({name: line}, {name: [user names]}) for compiled HLO text."""
  lines = {}
  for line in hlo.splitlines():
    if m := re.match(r"\s*(ROOT )?%([\w.\-]+) = ", line):
      lines[m[2]] = line
  users = {n: [] for n in lines}
  for n, line in lines.items():
    for operand in re.findall(r"%([\w.\-]+)", _rhs(line)):
      if operand in users:
        users[operand].append(n)
  return lines, users


def opcode(line: str) -> str:
  """The opcode of an HLO instruction line."""
  match = re.search(r" ([a-z][a-z\-]*)\(%", line)
  assert match is not None, line
  return match[1]


def is_same_layout_copy(line: str) -> bool:
  """Whether an HLO copy is a prefetch: only its memory space changes."""
  shapes = re.findall(r"\w+\[[\d,]*\]\{[^}]*\}", _rhs(line))[:2]
  return len({re.sub(r"S\(\d\)", "", s) for s in shapes}) == 1


def _rhs(line: str) -> str:
  return line.split(" = ", 1)[1]
