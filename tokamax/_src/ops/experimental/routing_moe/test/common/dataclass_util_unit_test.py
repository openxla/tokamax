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
"""Unit tests for dataclass_util."""

import dataclasses

from absl.testing import absltest
from tokamax._src.ops.experimental.routing_moe.common import dataclass_util


class DataclassUtilTest(absltest.TestCase):

  def test_require_none_default_rejects_missing_or_non_none_default(self):
    with self.assertRaisesRegex(ValueError, "must default to None"):

      @dataclass_util.require_none_default
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _BadUnresolved:  # pylint: disable=unused-variable
        x: int

  def test_require_no_default_rejects_inherited_default(self):
    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Base:
      x: int | None = None

    with self.assertRaisesRegex(ValueError, "must not have a default value"):

      @dataclass_util.require_no_default
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _BadResolved(_Base):  # pylint: disable=unused-variable
        x: int

  def test_resolves_from_accepts_matching_order_and_trailing_dropped(self):
    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Unresolved:
      a: int | None = None
      b: str | None = None
      hint: float | None = None

    @dataclass_util.resolves_from(_Unresolved, dropped=("hint",))
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Resolved:
      a: int
      b: str

    self.assertEqual(_Resolved(a=1, b="ok"), _Resolved(a=1, b="ok"))

  def test_resolves_from_rejects_reordered_or_mismatched_fields(self):
    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Unresolved:
      a: int | None = None
      b: str | None = None

    with self.assertRaisesRegex(ValueError, "must match _Unresolved fields"):

      @dataclass_util.resolves_from(_Unresolved)
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _Reordered:  # pylint: disable=unused-variable
        b: str
        a: int

    with self.assertRaisesRegex(ValueError, "must match _Unresolved fields"):

      @dataclass_util.resolves_from(_Unresolved)
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _MissingField:  # pylint: disable=unused-variable
        a: int

  def test_resolves_from_rejects_unknown_or_non_trailing_dropped(self):
    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Unresolved:
      hint: float | None = None
      a: int | None = None

    with self.assertRaisesRegex(ValueError, "Unknown dropped fields"):
      dataclass_util.resolves_from(_Unresolved, dropped=("nonexistent",))

    with self.assertRaisesRegex(ValueError, "must appear at the end"):
      dataclass_util.resolves_from(_Unresolved, dropped=("hint",))

  def test_resolves_from_enforces_default_invariants_on_both_classes(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _BadUnresolved:
      a: int = 1

    with self.assertRaisesRegex(ValueError, "must default to None"):
      dataclass_util.resolves_from(_BadUnresolved)

    @dataclass_util.require_none_default
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _GoodUnresolved:
      a: int | None = None

    with self.assertRaisesRegex(ValueError, "must not have a default value"):

      @dataclass_util.resolves_from(_GoodUnresolved)
      @dataclasses.dataclass(frozen=True, kw_only=True)
      class _BadResolved:  # pylint: disable=unused-variable
        a: int = 1


if __name__ == "__main__":
  absltest.main()
