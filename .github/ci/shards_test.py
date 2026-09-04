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
"""Tests for `shards.py`.

Mostly about what happens when someone edits a test file and forgets to edit
the shard table: the checks exist to turn that into a failed `build-matrix`
job instead of a test that quietly stops running.

Stdlib `unittest`, no third-party imports.

  python3 -m unittest discover -s .github/ci -p '*_test.py'
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
import os
import tempfile
import textwrap
from typing import NoReturn
import unittest
from unittest import mock

import shards

# Two private names, aliased once rather than reached for at each use: the
# catch-all shard's key, and the glob helper the coverage checks are defined
# in terms of.
# pylint: disable=protected-access
CATCH_ALL = shards._CATCH_ALL_SHARD
matches = shards._matches
# pylint: enable=protected-access


# A fake tree. `base_test.py` is the file split by node ID.
FILES = (
    'pkg/base_test.py',
    'pkg/api_test.py',
    'pkg/slow_test.py',
    'pkg/test_base.py',  # matched by IGNORED_GLOBS
)
CLASSES = {'pkg/base_test.py': ['MaskTest', 'DpaTest', 'VjpTest']}


def is_noise(error: str, noise: Iterable[str]) -> bool:
  """Returns whether an error is about the fixture, not the case under test.

  Args:
    error: One string from `check_consistency`.
    noise: Substrings that mark an error as being about the fixture.

  Returns:
    True if `error` contains any of them.
  """
  return any(n in error for n in noise)


def raise_on_read(path: str) -> NoReturn:
  """Stands in for `class_reader` where it must not be called.

  Args:
    path: The file `check_consistency` tried to read.

  Raises:
    AssertionError: Always.
  """
  raise AssertionError(f'should not have read {path}')


def check(
    shard_map: Mapping[str, tuple[str, ...] | shards.Spec],
    files: Collection[str] = FILES,
    classes: Mapping[str, list[str]] | None = None,
) -> list[str]:
  """Runs `check_consistency` over a fake tree.

  Args:
    shard_map: Shard name to either a full spec or, for the common case, just
      its `paths`.
    files: The fake tree, defaulting to `FILES`.
    classes: File to its class names, standing in for the AST reader. Defaults
      to `CLASSES`.

  Returns:
    The errors, minus the ones the fake tree provokes about itself.
  """
  spec = {
      n: dict(p) if isinstance(p, dict) else dict(paths=p)
      for n, p in shard_map.items()
  }
  ignored = matches(files, shards.IGNORED_GLOBS)
  # `list(...)` is redundant at runtime -- `check_consistency` returns a list.
  # It is here because `import shards` is a sibling import that resolves only
  # when this file is run from `.github/ci`, so a type checker rooted at the
  # repo sees the call as untyped and reports the comprehension below as
  # possibly unbound. Wrapping pins the type without a suppression comment.
  errors = list(
      shards.check_consistency(
          spec,
          files,
          set(),
          ignored,
          class_reader=(classes or CLASSES).get,
      )
  )
  # Errors about the fake tree rather than about the case under test: the
  # fixtures use short names like `s1`, so the theme rule has its own tests
  # below rather than firing in every one of these.
  noise = ('EXCLUDED_TESTS', 'no theme in THEMES')
  return [e for e in errors if not is_noise(e, noise)]


class NodeIdSplitTest(unittest.TestCase):
  """The failure the node-ID split introduces, and the ones it must not."""

  FULL = {
      's1': ('pkg/base_test.py::MaskTest', 'pkg/base_test.py::DpaTest'),
      's2': ('pkg/base_test.py::VjpTest',),
      's3': ('pkg/api_test.py', 'pkg/slow_test.py'),
  }

  def test_a_complete_split_is_clean(self) -> None:
    self.assertEqual(check(self.FULL), [])

  def test_new_class_in_a_split_file_is_caught(self) -> None:
    # The whole reason `declared_test_classes` exists. Every path in the table
    # is still valid and the file is still covered; without this check the new
    # class simply never runs.
    classes = {'pkg/base_test.py': [*CLASSES['pkg/base_test.py'], 'NewTest']}
    errors = check(self.FULL, classes=classes)
    self.assertEqual(len(errors), 1, errors)
    self.assertIn('no shard names NewTest', errors[0])

  def test_renamed_class_is_caught_from_both_sides(self) -> None:
    classes = {'pkg/base_test.py': ['MaskTest', 'DpaTest', 'VjpTestRenamed']}
    errors = ' | '.join(check(self.FULL, classes=classes))
    self.assertIn('no shard names VjpTestRenamed', errors)
    self.assertIn('does not declare: VjpTest', errors)

  def test_deleted_class_is_caught(self) -> None:
    classes = {'pkg/base_test.py': ['MaskTest', 'DpaTest']}
    errors = check(self.FULL, classes=classes)
    self.assertIn('does not declare: VjpTest', ' '.join(errors))

  def test_whole_file_and_node_id_double_runs(self) -> None:
    errors = check({
        's1': ('pkg/base_test.py',),
        's2': ('pkg/base_test.py::MaskTest',),
        's3': ('pkg/api_test.py', 'pkg/slow_test.py'),
    })
    self.assertIn('claimed both whole and by node ID', ' '.join(errors))

  def test_same_node_id_in_two_shards(self) -> None:
    shard_map = dict(self.FULL)
    shard_map['s2'] = ('pkg/base_test.py::VjpTest', 'pkg/base_test.py::DpaTest')
    self.assertIn('claimed more than once', ' '.join(check(shard_map)))

  def test_same_file_in_two_shards(self) -> None:
    errors = check({
        's1': ('pkg/api_test.py',),
        's2': ('pkg/api_test.py',),
        's3': ('pkg/base_test.py', 'pkg/slow_test.py'),
    })
    self.assertIn('run by more than one shard', ' '.join(errors))

  def test_unsplit_file_never_reads_source(self) -> None:
    # `class_reader` would raise on a file it does not know. Calling it for an
    # unsplit file would be a needless read of every test file in the repo.
    shards.check_consistency(
        {
            's': dict(
                paths=(
                    'pkg/api_test.py',
                    'pkg/base_test.py',
                    'pkg/slow_test.py',
                )
            )
        },
        FILES,
        set(),
        matches(FILES, shards.IGNORED_GLOBS),
        class_reader=raise_on_read,
    )


class ShardTableTest(unittest.TestCase):
  """Editing the table itself."""

  def test_new_test_file_lands_in_the_catch_all(self) -> None:
    # Adding a test without touching SHARDS is not fatal: it runs, visibly,
    # in the catch-all. This is what makes "add an op" a one-line change.
    table = {
        's1': dict(paths=('pkg/api_test.py',)),
        CATCH_ALL: dict(paths=None),
    }
    with mock.patch.dict(shards.SHARDS, table, clear=True):
      resolved, catch_all, _, _ = shards.resolve_shards(
          [*FILES, 'pkg/brand_new_test.py']
      )
    self.assertIn('pkg/brand_new_test.py', catch_all)
    self.assertIn(CATCH_ALL, resolved)

  def test_empty_catch_all_is_dropped_not_emitted(self) -> None:
    # An empty `paths` would reach the workflow as no arguments, and pytest
    # with no arguments collects the whole repository.
    table = {'s1': dict(paths=FILES[:3]), CATCH_ALL: dict(paths=None)}
    with mock.patch.dict(shards.SHARDS, table, clear=True):
      resolved, catch_all, _, _ = shards.resolve_shards(FILES)
    self.assertEqual(catch_all, [])
    self.assertNotIn(CATCH_ALL, resolved)

  def test_directory_target_is_rejected(self) -> None:
    errors = check({'s1': ('pkg/',), 's2': FILES[:3]})
    self.assertIn('not test files: pkg/', ' '.join(errors))

  def test_selection_flag_in_paths_is_rejected(self) -> None:
    errors = check({
        's1': ('pkg/api_test.py', '-k', 'foo'),
        's2': ('pkg/base_test.py', 'pkg/slow_test.py'),
    })
    self.assertIn('passes flags in paths', ' '.join(errors))

  def test_renamed_target_fails_loudly(self) -> None:
    errors = ' '.join(check({'s1': ('pkg/gone_test.py',), 's2': FILES[:3]}))
    self.assertIn('not test files: pkg/gone_test.py', errors)
    self.assertIn('collects no test files', errors)

  def test_unclaimed_file_is_reported(self) -> None:
    errors = check({'s1': ('pkg/api_test.py',)})
    self.assertIn('test files no shard runs', ' '.join(errors))

  def test_leftover_devices_key_is_rejected(self) -> None:
    # Every shard runs on every runner, so `devices` pins nothing. A stale key
    # has to fail rather than be ignored, or the table reads as pinned when it
    # is not.
    errors = check({'s1': dict(paths=FILES[:3], devices=('tpu',))})
    self.assertIn('has a `devices` key', ' '.join(errors))

  def test_matrix_is_one_job_per_shard_and_runner(self) -> None:
    table = {'s1': dict(paths=FILES[:2]), 's2': dict(paths=FILES[2:3])}
    combos = shards.build_matrix(table)
    self.assertEqual(len(combos), 2 * len(shards.RUNNERS))
    self.assertTrue(all(c['test_paths'] for c in combos))

  def test_matrix_covers_every_runner_once_per_shard(self) -> None:
    table = {'s1': dict(paths=FILES[:2])}
    combos = shards.build_matrix(table)
    self.assertCountEqual([c['runner'] for c in combos], list(shards.RUNNERS))
    # The device is the pip extra the job installs, and comes from `RUNNERS`.
    self.assertCountEqual(
        [c['device'] for c in combos],
        [device for device, _ in shards.RUNNERS.values()],
    )


class DeclaredTestClassesTest(unittest.TestCase):
  """The AST reader, which decides what counts as a class needing a shard."""

  def classes(self, source: str) -> list[str]:
    """Reads the classes of a throwaway module.

    Args:
      source: Module source, dedented before it is written out.

    Returns:
      What `declared_test_classes` finds in it.
    """
    with tempfile.NamedTemporaryFile(
        'w', suffix='_test.py', delete=False, encoding='utf-8'
    ) as f:
      f.write(textwrap.dedent(source))
      path = f.name
    self.addCleanup(os.unlink, path)
    return shards.declared_test_classes(path)

  def test_plain_and_prefixed_classes(self) -> None:
    self.assertEqual(
        self.classes("""
            class MaskTest(absltest.TestCase): pass
            class TestSomething: pass
            class Bare: pass
        """),
        ['MaskTest', 'TestSomething', 'Bare'],
    )

  def test_private_helper_is_exempt(self) -> None:
    self.assertEqual(self.classes('class _Helper: pass\n'), [])

  def test_private_test_case_is_not_exempt(self) -> None:
    # pytest collects TestCase subclasses whatever they are called, so a
    # leading underscore does not stop this one running.
    self.assertEqual(
        self.classes("""
            class _Sneaky(parameterized.TestCase): pass
            class _AlsoSneaky(test_base.AttentionTestBase): pass
        """),
        ['_Sneaky', '_AlsoSneaky'],
    )

  def test_class_under_a_backend_guard_counts(self) -> None:
    # Still a module attribute, so pytest still collects it.
    self.assertEqual(
        self.classes("""
            if backend == 'tpu':
              class TpuTest(absltest.TestCase): pass
            else:
              class FallbackTest(absltest.TestCase): pass
        """),
        ['TpuTest', 'FallbackTest'],
    )

  def test_class_in_a_function_does_not_count(self) -> None:
    # Not a module attribute, so pytest does not collect it, and flagging it
    # would be a false positive nothing but a rename could silence.
    self.assertEqual(
        self.classes("""
            def f():
              class LocalTest(absltest.TestCase): pass
        """),
        [],
    )


class ThemeTest(unittest.TestCase):
  """Grouping, which is carried by the shard name."""

  def test_theme_is_the_prefix(self) -> None:
    self.assertEqual(shards.shard_theme('attention-base-vjp')[1], 'attention')
    self.assertEqual(shards.shard_theme('ragged-dot-misc')[1], 'ragged-dot')

  def test_a_theme_can_be_a_whole_name(self) -> None:
    self.assertEqual(shards.shard_theme('catch-all')[1], 'catch-all')

  def test_prefix_must_end_at_a_hyphen(self) -> None:
    # `opsomething` is not in the `ops` theme. Without this the check would
    # accept a name that only looks grouped.
    self.assertIsNone(shards.shard_theme('opsomething'))

  def test_untethered_name_is_an_error(self) -> None:
    errors = check({'s1': FILES[:3]})
    joined = ' '.join(
        shards.check_consistency(
            {'s1': dict(paths=FILES[:3])},
            FILES,
            set(),
            matches(FILES, shards.IGNORED_GLOBS),
            class_reader=CLASSES.get,
        )
    )
    self.assertNotIn('no theme', ' '.join(errors))  # filtered by `check`
    self.assertIn('no theme in THEMES: s1', joined)

  def test_order_is_longest_first_across_themes(self) -> None:
    table = {
        'core-utils': dict(paths=(), minutes=2),
        'attention-slow': dict(paths=(), minutes=30),
        'attention-unmeasured': dict(paths=()),
        'attention-fast': dict(paths=(), minutes=1),
    }
    self.assertEqual(
        [n for n, _ in sorted(table.items(), key=shards.shard_order)],
        # Unmeasured sorts last overall: absent is not zero.
        [
            'attention-slow',
            'core-utils',
            'attention-fast',
            'attention-unmeasured',
        ],
    )

  def test_theme_breaks_ties_between_equal_length_shards(self) -> None:
    table = {
        'gmm-a': dict(paths=(), minutes=5),
        'splash-b': dict(paths=(), minutes=5),
    }
    self.assertEqual(
        [n for n, _ in sorted(table.items(), key=shards.shard_order)],
        ['splash-b', 'gmm-a'],  # splash precedes gmm in THEMES
    )

  def test_catch_all_always_sorts_first(self) -> None:
    table = {
        'core-utils': dict(paths=(), minutes=2),
        'attention-slow': dict(paths=(), minutes=30),
        'catch-all': dict(paths=(), minutes=100),
    }
    self.assertEqual(
        [n for n, _ in sorted(table.items(), key=shards.shard_order)],
        ['catch-all', 'attention-slow', 'core-utils'],
    )
    # catch-all still sorts first even if another shard has more minutes
    table_other = {
        'attention-longest': dict(paths=(), minutes=200),
        'catch-all': dict(paths=(), minutes=100),
    }
    self.assertEqual(
        [n for n, _ in sorted(table_other.items(), key=shards.shard_order)],
        ['catch-all', 'attention-longest'],
    )

  def test_every_real_shard_has_a_theme(self) -> None:
    strays = [n for n in shards.SHARDS if shards.shard_theme(n) is None]
    self.assertEqual(strays, [])


class RealRepositoryTest(unittest.TestCase):
  """The table as it actually is, which is what CI runs."""

  def test_check_passes(self) -> None:
    files = shards.all_test_files()
    resolved, _, excluded, ignored = shards.resolve_shards(files)
    self.assertEqual(
        shards.check_consistency(resolved, files, excluded, ignored), []
    )

  def test_a_new_test_file_needs_no_edit_to_run(self) -> None:
    # Deliberately not `assertEqual(catch_all, [])`. The catch-all being empty
    # is the goal, but asserting it here would turn "added a test file" into a
    # red build, which is the opposite of what the catch-all is for. `check`
    # and `matrix` both print its contents; that is the visibility.
    files = [*shards.all_test_files(), 'tokamax/_src/ops/new_thing_test.py']
    resolved, catch_all, excluded, ignored = shards.resolve_shards(files)
    self.assertIn('tokamax/_src/ops/new_thing_test.py', catch_all)
    self.assertEqual(
        shards.check_consistency(resolved, files, excluded, ignored), []
    )

  def test_every_matrix_entry_has_paths(self) -> None:
    resolved, *_ = shards.resolve_shards()
    for combo in shards.build_matrix(resolved):
      self.assertTrue(combo['test_paths'], combo)


if __name__ == '__main__':
  unittest.main()
