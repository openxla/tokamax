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

import ast
import collections
from collections.abc import Collection, Iterable, Iterator, Mapping
import contextlib
import io
import os
import re
import shlex
import subprocess
import tempfile
import textwrap
from typing import NoReturn
import unittest
from unittest import mock

import deps
import shards

# Private names, aliased once rather than reached for at each use: the
# catch-all shard's key, the glob helper the coverage checks are defined in
# terms of, and the reader that takes the table out of a version of the file.
# pylint: disable=protected-access
CATCH_ALL = shards._CATCH_ALL_SHARD
assigns_shards = shards._assigns_shards
matches = shards._matches
split_shards_table = shards._split_shards_table
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
    pyproject: str | None = None,
) -> list[str]:
  """Runs `check_consistency` over a fake tree.

  Args:
    shard_map: Shard name to either a full spec or, for the common case, just
      its `paths`.
    files: The fake tree, defaulting to `FILES`.
    classes: File to its class names, standing in for the AST reader. Defaults
      to `CLASSES`.
    pyproject: Path to pyproject.toml to verify JAX floor against, or None.

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
          pyproject=pyproject,
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

  def _jobs_on_latest_jax(
      self, table: Mapping[str, shards.Spec]
  ) -> list[shards.MatrixEntry]:
    return [
        c
        for c in shards.build_matrix(table)
        if c['jax_pin'] == shards.latest_jax()
    ]

  def test_matrix_is_one_job_per_shard_and_runner(self) -> None:
    table = {'s1': dict(paths=FILES[:2]), 's2': dict(paths=FILES[2:3])}
    combos = self._jobs_on_latest_jax(table)
    self.assertEqual(len(combos), 2 * len(shards.RUNNERS))
    self.assertTrue(all(c['test_paths'] for c in combos))

  def test_matrix_covers_every_runner_once_per_shard(self) -> None:
    combos = self._jobs_on_latest_jax({'s1': dict(paths=FILES[:2])})
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


class JaxFloorTest(unittest.TestCase):

  def _jax_floor(self, body: str) -> str:
    with tempfile.NamedTemporaryFile(
        'w', suffix='.toml', delete=False, encoding='utf-8'
    ) as f:
      f.write(textwrap.dedent(body))
      path = f.name
    self.addCleanup(os.unlink, path)
    return shards.jax_floor(path)

  def test_disagreeing_floors_raise(self) -> None:
    with self.assertRaisesRegex(ValueError, 'more than one JAX floor'):
      self._jax_floor("""
          [project]
          dependencies = ["jax>=0.11.1"]
          [project.optional-dependencies]
          tpu = ["jax[tpu]>=0.11.0"]
      """)

  def test_no_jax_requirement_raises(self) -> None:
    with self.assertRaisesRegex(ValueError, 'no .jax>=. requirement'):
      self._jax_floor("""
          [project]
          dependencies = ["numpy>=2.1"]
      """)

  def test_floor_extracted_from_dependencies_and_extras(self) -> None:
    floor = self._jax_floor("""
        [project]
        dependencies = [
            "jax>=0.11.0",
            "jaxlib>=0.11.0",
            "jaxtyping>=0.3",
        ]
        [project.optional-dependencies]
        tpu = ["jax[tpu]>=0.11.0"]
        cuda = ["jax[cuda13]>=0.11.0"]
    """)
    self.assertEqual(floor, '0.11.0')

  def test_unrelated_packages_with_jax_name_not_matched(self) -> None:
    floor = self._jax_floor("""
        [project]
        dependencies = [
            "jax>=0.11.0",
            "cuequivariance-jax>=0.10.0",
        ]
    """)
    self.assertEqual(floor, '0.11.0')


class JaxVersionTest(unittest.TestCase):

  def test_version_sorting_is_numerical(self) -> None:
    versions = ('0.11.2', '0.11.10', '0.12.0', '0.11.0')
    with mock.patch.object(shards, 'JAX_VERSIONS', versions):
      self.assertEqual(
          shards.sorted_jax_versions(),
          ('0.12.0', '0.11.10', '0.11.2', '0.11.0'),
      )
      self.assertEqual(shards.latest_jax(), '0.12.0')
      self.assertEqual(shards.older_jaxs(), ('0.11.10', '0.11.2', '0.11.0'))
      self.assertEqual(shards.oldest_jax(), '0.11.0')

  def test_bench_jax_pin_matches_latest_jax(self) -> None:
    pyproject_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'pyproject.toml'
    )
    with open(pyproject_path, 'rb') as f:
      bench_deps = shards.tomllib.load(f)['project']['optional-dependencies'][
          'bench'
      ]
    self.assertIn(
        f'jax=={shards.latest_jax()}',
        bench_deps,
        msg=(
            '[project.optional-dependencies].bench in pyproject.toml is out of'
            ' sync with JAX_VERSIONS in shards.py. Update `jax==...` under'
            ' `bench` in pyproject.toml to match `shards.latest_jax()`.'
        ),
    )


class CompatMatrixTest(unittest.TestCase):

  def resolved(self) -> shards.ShardMap:
    resolved, *_ = shards.resolve_shards()
    return resolved

  def test_primary_jobs_are_on_the_latest_jax_only(self) -> None:
    pins = {e['jax_pin'] for e in shards.build_matrix(self.resolved())}
    self.assertEqual(pins, {shards.latest_jax()})

  def test_every_older_version_runs_every_bin_on_every_compat_runner(
      self,
  ) -> None:
    resolved = self.resolved()
    bins = shards.compat_bins(resolved)
    jobs = shards.build_compat_matrix(resolved)
    self.assertCountEqual(
        [(j['runner'], j['jax_pin'], j['shard_name']) for j in jobs],
        [
            (runner, version, f'compat-{i}')
            for runner in shards.COMPAT_RUNNERS
            for version in shards.older_jaxs()
            for i in range(1, len(bins) + 1)
        ],
    )

  def test_every_shard_is_in_exactly_one_bin(self) -> None:
    resolved = self.resolved()
    binned = [n for names in shards.compat_bins(resolved) for n in names]
    self.assertCountEqual(binned, list(resolved))

  def test_a_bin_runs_all_of_its_shards_paths(self) -> None:
    resolved = self.resolved()
    for job in shards.build_compat_matrix(resolved):
      expected = [
          p for n in job['shards'].split() for p in resolved[n]['paths']
      ]
      self.assertEqual(shlex.split(job['test_paths']), expected)

  def test_job_names_are_unique(self) -> None:
    resolved = self.resolved()
    jobs = [
        (prefix, e['runner_short'], e['shard_name'])
        for prefix, matrix in (
            ('shard-tests', shards.build_matrix(resolved)),
            ('compat-tests', shards.build_compat_matrix(resolved)),
        )
        for e in matrix
    ]
    duplicates = [
        job for job, count in collections.Counter(jobs).items() if count > 1
    ]
    self.assertEqual(duplicates, [])

  def test_every_job_preserves_runner_device(self) -> None:
    resolved = self.resolved()
    for entry in [
        *shards.build_matrix(resolved),
        *shards.build_compat_matrix(resolved),
    ]:
      expected_device = shards.RUNNERS[entry['runner']][0]
      self.assertEqual(entry['device'], expected_device)

  def test_total_matrix_job_count(self) -> None:
    resolved = self.resolved()
    self.assertEqual(
        len(shards.build_matrix(resolved)), len(resolved) * len(shards.RUNNERS)
    )
    self.assertEqual(
        len(shards.build_compat_matrix(resolved)),
        len(shards.compat_bins(resolved))
        * len(shards.older_jaxs())
        * len(shards.COMPAT_RUNNERS),
    )

  def test_matrix_with_only_filters_all_versions(self) -> None:
    resolved = self.resolved()
    only = {'core-api'}
    jobs = shards.build_matrix(resolved, only=only)
    self.assertTrue(all(j['shard_name'] == 'core-api' for j in jobs))
    self.assertEqual(len(jobs), len(shards.RUNNERS))
    compat = shards.build_compat_matrix(resolved, only=only)
    self.assertTrue(all(j['shards'] == 'core-api' for j in compat))
    self.assertEqual(
        len(compat), len(shards.older_jaxs()) * len(shards.COMPAT_RUNNERS)
    )


class CompatBinTest(unittest.TestCase):
  """Packing shards into compat bins, against a made-up table."""

  def bins(self, **sizes: float | None) -> list[list[str]]:
    table = {
        name: (
            dict(paths=(f'tokamax/{name}_test.py',))
            | ({} if minutes is None else dict(minutes=minutes))
        )
        for name, minutes in sizes.items()
    }
    return shards.compat_bins(table)

  def test_small_shards_share_a_bin(self) -> None:
    with mock.patch.object(shards, 'COMPAT_BIN_MINUTES', 20):
      self.assertEqual(self.bins(a=5, b=5, c=5), [['a', 'b', 'c']])

  def test_no_bin_exceeds_the_budget_unless_one_shard_does(self) -> None:
    with mock.patch.object(shards, 'COMPAT_BIN_MINUTES', 20):
      bins = self.bins(big=45, a=12, b=9, c=8, d=7, e=3)
    self.assertEqual(bins[0], ['big'])
    sizes = dict(big=45, a=12, b=9, c=8, d=7, e=3)
    for names in bins[1:]:
      self.assertLessEqual(sum(sizes[n] for n in names), 20)

  def test_an_unmeasured_shard_runs_alone(self) -> None:
    # Absent is an unknown, not a zero: sharing a bin, it could be the job
    # that sets the compat wall clock.
    with mock.patch.object(shards, 'COMPAT_BIN_MINUTES', 20):
      bins = self.bins(unknown=None, a=1, b=1)
    self.assertIn(['unknown'], bins)

  def test_compat_minutes_wins_over_minutes(self) -> None:
    table = {
        'slow': dict(
            paths=('tokamax/slow_test.py',), minutes=40, compat_minutes=2
        ),
        'fast': dict(paths=('tokamax/fast_test.py',), minutes=3),
    }
    with mock.patch.object(shards, 'COMPAT_BIN_MINUTES', 20):
      self.assertEqual(shards.compat_bins(table), [['fast', 'slow']])

  def test_fullest_bin_comes_first(self) -> None:
    with mock.patch.object(shards, 'COMPAT_BIN_MINUTES', 20):
      bins = self.bins(a=4, b=18, c=19)
    self.assertEqual(bins[0], ['c'])

  def test_bad_compat_minutes_is_rejected(self) -> None:
    errors = check({'s1': dict(paths=FILES, compat_minutes=0)})
    self.assertIn('compat_minutes=0', ' '.join(errors))


class JaxConsistencyTest(unittest.TestCase):

  def test_support_floor_mismatch_is_caught(self) -> None:
    with tempfile.NamedTemporaryFile(
        'w', suffix='.toml', delete=False, encoding='utf-8'
    ) as f:
      f.write(textwrap.dedent("""
          [project]
          dependencies = ["jax>=0.11.1"]
      """))
      path = f.name
    self.addCleanup(os.unlink, path)

    with mock.patch.object(shards, 'JAX_VERSIONS', ('0.11.1', '0.11.0')):
      errors = check(
          {'s1': ('pkg/api_test.py',)},
          files=('pkg/api_test.py',),
          pyproject=path,
      )
      self.assertTrue(
          any(
              'the JAX lower bound and the oldest JAX version the CI tests'
              ' against have to be the same.'
              in e
              for e in errors
          )
      )

  def test_not_equal_to_two_jax_versions_is_caught(self) -> None:
    with mock.patch.object(shards, 'JAX_VERSIONS', ('0.11.0',)):
      errors = check({'s1': ('pkg/api_test.py',)}, files=('pkg/api_test.py',))
      self.assertTrue(
          any(
              'tokamax supports 2 latest JAX versions for backward'
              ' compatibility.'
              in e
              for e in errors
          )
      )

  def test_duplicated_jax_version_is_caught(self) -> None:
    with mock.patch.object(shards, 'JAX_VERSIONS', ('0.11.0', '0.11.0')):
      errors = check({'s1': ('pkg/api_test.py',)}, files=('pkg/api_test.py',))
      self.assertTrue(any('Duplicated JAX version found' in e for e in errors))

  def test_invalid_jax_version_format_is_caught(self) -> None:
    with mock.patch.object(
        shards, 'JAX_VERSIONS', ('0.11.1', 'invalid-version')
    ):
      errors = check({'s1': ('pkg/api_test.py',)}, files=('pkg/api_test.py',))
      self.assertTrue(
          any('Invalid JAX version found in JAX_VERSIONS' in e for e in errors)
      )

  def test_unknown_compat_runner_is_caught(self) -> None:
    with mock.patch.object(shards, 'COMPAT_RUNNERS', ('non-existent-runner',)):
      errors = check({'s1': ('pkg/api_test.py',)}, files=('pkg/api_test.py',))
      self.assertTrue(
          any(
              "COMPAT_RUNNERS 'non-existent-runner' is not in RUNNERS" in e
              for e in errors
          )
      )


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
    for combo in [
        *shards.build_matrix(resolved),
        *shards.build_compat_matrix(resolved),
    ]:
      self.assertTrue(combo['test_paths'], combo)

  def test_whole_matrix_fits_githubs_job_limit(self) -> None:
    # GitHub caps one job's matrix at 256 entries; a matrix past it fails to
    # start rather than running a subset.
    resolved, *_ = shards.resolve_shards()
    self.assertLessEqual(len(shards.build_matrix(resolved)), 256)
    self.assertLessEqual(len(shards.build_compat_matrix(resolved)), 256)


class SelectionTest(unittest.TestCase):
  """Narrowing the matrix to the shards a change can reach.

  The asymmetry under test throughout: a shard wrongly run costs runner
  minutes, a shard wrongly skipped reports a change green that was never
  tested. Every uncertain path here has to fail open.
  """

  def _changed(self, *paths: str) -> str:
    """Writes a NUL-separated change list and returns its path."""
    handle, name = tempfile.mkstemp()
    with os.fdopen(handle, 'w') as f:
      f.write(''.join(f'{p}\0' for p in paths))
    self.addCleanup(os.unlink, name)
    return name

  def test_no_list_runs_everything(self) -> None:
    resolved, *_ = shards.resolve_shards()
    only, reason = shards.select(resolved, shards.all_test_files(), None)
    self.assertIsNone(only)
    self.assertIn('every shard runs', reason)

  def test_empty_list_runs_everything(self) -> None:
    # Distinct from "reaches no test": an empty diff is a diff that failed to
    # produce anything, and guessing that it means "nothing to run" is how a
    # broken `git diff` would silently skip the whole suite.
    resolved, *_ = shards.resolve_shards()
    only, reason = shards.select(
        resolved, shards.all_test_files(), self._changed()
    )
    self.assertIsNone(only)
    self.assertIn('empty', reason)

  def test_a_change_no_shard_owns_runs_everything(self) -> None:
    # Two files that reach no shard, for unrelated reasons: `conftest.py` is
    # inside the package and changes what every test collects, and
    # `pyproject.toml` is outside it, so `deps` cannot trace it to a test at
    # all. Neither may be read as "nothing to run".
    resolved, *_ = shards.resolve_shards()
    files = shards.all_test_files()
    for path in ('tokamax/conftest.py', 'pyproject.toml'):
      with self.subTest(path=path):
        only, _ = shards.select(resolved, files, self._changed(path))
        self.assertIsNone(only)

  def test_documentation_selects_nothing(self) -> None:
    # An empty set is a real answer, not a failure: `None` means "run
    # everything" and these two must never be confused.
    resolved, *_ = shards.resolve_shards()
    for path in ('README.md', 'docs/tokamax_ops.ipynb'):
      with self.subTest(path=path):
        only, _ = shards.select(
            resolved, shards.all_test_files(), self._changed(path)
        )
        self.assertEqual(only, set())

  def test_a_test_file_selects_its_own_shard(self) -> None:
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    target = 'tokamax/_src/ops/attention/base_test.py'
    self.assertIn(target, files)
    only, _ = shards.select(resolved, files, self._changed(target))
    self.assertTrue(only)
    for name in only:
      self.assertIn(name, resolved)
    # Whichever shards those are, they are the ones that collect the file.
    owning = {
        n
        for n, s in resolved.items()
        if target in shards.collected_by(s['paths'], files)
    }
    self.assertEqual(only & owning, owning)

  def test_selection_covers_every_affected_file(self) -> None:
    # The property that matters: nothing affected is left unrun. Checked
    # against a high fan-in module, so the selected set is large but not all.
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    affected, reason = deps.affected_tests(['tokamax/_src/config.py'])
    self.assertIsNone(reason)
    only = shards.shards_for_tests(resolved, files, affected)
    ran = set()
    for name in only:
      ran |= shards.collected_by(resolved[name]['paths'], files)
    # Minus the files pytest never collects: `deps` counts `test_base.py` as
    # affected because a real test imports it, and that real test is in `ran`.
    self.assertEqual(
        affected - ran - matches(affected, shards.IGNORED_GLOBS), set()
    )

  def test_an_unclaimed_affected_file_forces_a_full_run(self) -> None:
    # `check_consistency` should make this unreachable. If the two ever
    # disagree, the safe reading is to run everything, not to drop the file.
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    stripped = {n: dict(s) for n, s in resolved.items()}
    victim = 'tokamax/_src/ops/attention/base_test.py'
    for spec in stripped.values():
      spec['paths'] = tuple(
          p for p in spec['paths'] if shards.split_node_id(p)[0] != victim
      )
    only, reason = shards.select(stripped, files, self._changed(victim))
    self.assertIsNone(only)
    self.assertIn('no shard', reason)

  def test_ignored_files_do_not_force_a_full_run(self) -> None:
    # `test_base.py` is a test file to `deps` (something imports it) and not
    # one to pytest (`IGNORED_GLOBS`). Treating that as an unclaimed file
    # would make every change to a base class a full run.
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    self.assertEqual(
        shards.unclaimed_tests(
            resolved, files, {'tokamax/_src/ops/attention/test_base.py'}
        ),
        set(),
    )

  def _without(self, resolved: shards.ShardMap, victim: str) -> shards.ShardMap:
    """The shard table with `victim` taken out of every shard's paths.

    Excluding a test means removing it from its shard as well as listing it:
    `check_consistency` subtracts `excluded` from the files a shard has to
    cover, so the two go together and neither half alone is the real state.
    """
    stripped = {}
    for name, spec in resolved.items():
      stripped[name] = dict(spec)
      stripped[name]['paths'] = tuple(
          p for p in spec['paths'] if shards.split_node_id(p)[0] != victim
      )
    return stripped

  def test_excluded_files_do_not_force_a_full_run(self) -> None:
    files = shards.all_test_files()
    victim = files[0]
    stripped = self._without(shards.resolve_shards(files)[0], victim)
    self.assertEqual(
        shards.unclaimed_tests(stripped, files, {victim}), {victim}
    )
    with mock.patch.object(shards, 'EXCLUDED_TESTS', (victim,)):
      self.assertEqual(shards.resolve_shards(files)[2], {victim})
      self.assertEqual(shards.unclaimed_tests(stripped, files, {victim}), set())

  def test_an_excluded_file_is_labelled_in_the_explanation(self) -> None:
    files = shards.all_test_files()
    victim = files[0]
    changed = self._changed(victim)
    stripped = self._without(shards.resolve_shards(files)[0], victim)
    with mock.patch.object(shards, 'EXCLUDED_TESTS', (victim,)):
      out = '\n'.join(shards.explain(stripped, files, changed))
    self.assertIn('in EXCLUDED_TESTS, selects nothing', out)
    self.assertNotIn('NO SHARD', out)

  def test_matrix_narrows_to_the_named_shards(self) -> None:
    resolved, *_ = shards.resolve_shards()
    picked = sorted(resolved)[:2]
    combos = shards.build_matrix(resolved, only=picked)
    self.assertEqual({c['shard_name'] for c in combos}, set(picked))
    self.assertEqual(len(combos), len(picked) * len(shards.RUNNERS))
    # The compat check is narrowed by the same selection.
    binned = {
        name
        for names in shards.compat_bins(resolved, only=picked)
        for name in names
    }
    self.assertEqual(binned, set(picked))

  def test_matrix_with_no_shards_is_empty_not_everything(self) -> None:
    resolved, *_ = shards.resolve_shards()
    self.assertEqual(shards.build_matrix(resolved, only=set()), [])
    self.assertNotEqual(shards.build_matrix(resolved, only=None), [])
    self.assertEqual(shards.build_compat_matrix(resolved, only=set()), [])
    self.assertNotEqual(shards.build_compat_matrix(resolved, only=None), [])

  def test_reading_a_change_list_drops_the_trailing_empty(self) -> None:
    self.assertEqual(
        shards.read_changed(self._changed('a.py', 'b.py')), ['a.py', 'b.py']
    )


class ExplainTest(unittest.TestCase):
  """The log that says why each selected shard was selected.

  Names shards by their formatted column rather than by substring: several
  shard names are prefixes of others -- `attention-base` of
  `attention-base-vjp` -- so a substring check would pass on the wrong row.
  """

  def _changed(self, *paths: str) -> str:
    handle, name = tempfile.mkstemp()
    with os.fdopen(handle, 'w') as f:
      f.write(''.join(f'{p}\0' for p in paths))
    self.addCleanup(os.unlink, name)
    return name

  def test_a_node_id_split_file_says_what_each_shard_runs(self) -> None:
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    target = 'tokamax/_src/ops/attention/base_test.py'
    body = '\n'.join(shards.explain(resolved, files, self._changed(target)))

    split = [ln for ln in body.splitlines() if '[runs ::' in ln]
    self.assertEqual(len(split), 2, body)
    # Disjoint class lists, which is the invariant `check_consistency`
    # enforces on a split file and the reason running both is not duplication.
    self.assertTrue(
        any('::DotProductAttentionWithExplicitVjpTest' in l for l in split)
    )
    self.assertTrue(any('::MaskTest' in l for l in split))
    # A shard that names whole files must not grow the annotation.
    for line in body.splitlines():
      if 'tokamax_test.py' in line:
        self.assertNotIn('[runs ::', line)

  def test_the_shard_clock_is_reported_next_to_the_file_count(self) -> None:
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    target = 'tokamax/_src/ops/attention/base_test.py'
    body = '\n'.join(shards.explain(resolved, files, self._changed(target)))
    minutes = resolved['attention-base-vjp']['minutes']
    header = f'  {"attention-base-vjp":32s} {f"{minutes}m":>4}'
    self.assertIn(header, body)

  def test_names_every_selected_shard_and_no_other(self) -> None:
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    target = 'tokamax/_src/ops/attention/base_test.py'
    body = '\n'.join(shards.explain(resolved, files, self._changed(target)))

    affected, _ = deps.affected_tests([target])
    picked = shards.shards_for_tests(resolved, files, affected)
    self.assertTrue(picked)
    self.assertNotEqual(picked, set(resolved))
    for name in picked:
      self.assertIn(f'  {name:32s} ', body)
    for name in set(resolved) - picked:
      self.assertNotIn(f'  {name:32s} ', body)
    self.assertIn(f'      {target}', body)

  def test_separates_the_edited_tests_from_the_fallout(self) -> None:
    source = 'tokamax/_src/ops/attention/base.py'
    test = 'tokamax/_src/ops/attention/base_test.py'
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    changed = self._changed(source, test)
    body = '\n'.join(shards.explain(resolved, files, changed))

    self.assertIn(f'      edited  {test}', body)
    self.assertNotIn(f'      import  {test}', body)
    # The source is listed as changed, but is not itself a test file, so it
    # is never tagged in the per-shard block.
    self.assertIn(f'      {source}', body)
    self.assertIn('2 changed file(s), 1 of them a test file:', body)
    # Something has to have come in through the graph, or the tags are not
    # telling the two cases apart at all.
    self.assertIn('      import  ', body)

  def test_a_never_collected_file_is_labelled_not_orphaned(self) -> None:
    files = shards.all_test_files()
    resolved, *_ = shards.resolve_shards(files)
    body = '\n'.join(
        shards.explain(
            resolved,
            files,
            self._changed('tokamax/_src/ops/attention/test_base.py'),
        )
    )
    self.assertIn('never collected, selects nothing', body)
    self.assertNotIn('NO SHARD', body)

  def test_an_empty_change_list_says_every_shard_runs(self) -> None:
    resolved, *_ = shards.resolve_shards()
    self.assertEqual(
        shards.explain(resolved, shards.all_test_files(), self._changed()),
        ['changed-file list is empty: every shard runs'],
    )


def _find_resplit(table: shards.ShardMap) -> tuple[str, str, str] | None:
  """Finds `(src, node_id, dst)` for a file split by node ID across shards."""
  split = collections.defaultdict(list)
  for name, spec in sorted(table.items()):
    if name == CATCH_ALL:
      continue
    for path in spec['paths']:
      file, selector = shards.split_node_id(path)
      if selector is not None:
        split[file].append((name, path))
  for entries in split.values():
    names = sorted({n for n, _ in entries})
    for src in names:
      held = [p for n, p in entries if n == src]
      dst = next((n for n in names if n != src), None)
      if dst is not None and len(held) > 1 and len(table[src]['paths']) > 1:
        return src, held[0], dst
  return None


class TableDiffTest(unittest.TestCase):
  """Selection for an edit to `shards.py` itself.

  Every case is an edit to the real file, resolved against the real tree,
  because the thing under test is how real edits to this table read: the base
  is the file as it is, the head is the file as someone might change it. git
  is never run; the two versions are handed in or patched over the readers.

  Table edits are made by shard name, on specs read back out of the file, and
  the shards they touch are picked by what they hold rather than by name. So a
  routine edit to `SHARDS` -- a retime, a new shard, a moved file -- does not
  break these tests; only a table with no shard of the needed shape would, and
  that fails `setUpClass` saying which shape is missing.
  """

  # A test file no shard names, so that resolving against a tree with it in
  # leaves something in the catch-all.
  UNCLAIMED = 'tokamax/_src/zz_unclaimed_test.py'

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    with open(shards.__file__, encoding='utf-8') as f:
      cls.base = f.read()
    cls.files = shards.all_test_files()
    cls.resolved = shards.resolve_shards(cls.files)[0]
    cls.table = split_shards_table(cls.base)[1]
    # The tree without whatever the live catch-all holds, which is not always
    # nothing. The catch-all cases pick its contents themselves, from this and
    # `UNCLAIMED`, rather than inherit them from the repository.
    caught = set(cls.resolved.get(CATCH_ALL, {}).get('paths', ()))
    cls.claimed = [f for f in cls.files if f not in caught]

    files = set(cls.files)
    # Shards whose every path is a whole test file, the ones a file can be
    # moved into or out of without touching a node ID split.
    plain = sorted(
        name
        for name, spec in cls.table.items()
        if name != CATCH_ALL
        and all('::' not in p and p in files for p in spec['paths'])
    )
    sources = [n for n in plain if len(cls.table[n]['paths']) > 1]
    if not sources or len(plain) < 3:
      raise AssertionError('SHARDS needs three whole-file shards, one with 2+')
    # Moves the last file of one shard into another: both are selected.
    cls.move_from = sources[0]
    cls.move_to = next(n for n in plain if n != cls.move_from)
    cls.moved = cls.table[cls.move_from]['paths'][-1]
    # A shard neither of those is, for the cases that drop or retime one.
    cls.other = next(n for n in plain if n not in (cls.move_from, cls.move_to))
    cls.resplit = _find_resplit(cls.table)

  def entries(self, source: str) -> dict[str, tuple[int, int]]:
    """Returns each shard's `[first, last)` line span in `source`."""
    tree = ast.parse(source)
    (statement,) = [node for node in tree.body if assigns_shards(node)]
    assert isinstance(statement.value, ast.Dict)
    return {
        ast.literal_eval(key): (key.lineno - 1, value.end_lineno)
        for key, value in zip(
            statement.value.keys, statement.value.values, strict=True
        )
    }

  def replace(
      self, name: str, entry: str | None, source: str | None = None
  ) -> str:
    """Returns `source` with shard `name`'s entry replaced, or dropped."""
    source = self.base if source is None else source
    first, last = self.entries(source)[name]
    lines = source.splitlines(keepends=True)
    lines[first:last] = [] if entry is None else [f'    {entry},\n']
    return ''.join(lines)

  def respec(
      self,
      name: str,
      source: str | None = None,
      rename: str | None = None,
      **changes: object,
  ) -> str:
    """Returns `source` with shard `name`'s spec updated by `changes`."""
    source = self.base if source is None else source
    spec = {**split_shards_table(source)[1][name], **changes}
    keywords = ', '.join(f'{k}={v!r}' for k, v in spec.items())
    return self.replace(name, f'{rename or name!r}: Spec({keywords})', source)

  def move(self, source: str | None = None) -> str:
    """Returns `source` with `moved` taken from `move_from` to `move_to`."""
    source = self.base if source is None else source
    table = split_shards_table(source)[1]
    head = self.respec(
        self.move_from,
        source,
        paths=tuple(
            p for p in table[self.move_from]['paths'] if p != self.moved
        ),
    )
    return self.respec(
        self.move_to,
        head,
        paths=(*table[self.move_to]['paths'], self.moved),
    )

  def edit(self, *pairs: tuple[str, str], source: str | None = None) -> str:
    """Returns the file with each `(old, new)` replaced, each exactly once."""
    head = self.base if source is None else source
    for old, new in pairs:
      self.assertEqual(head.count(old), 1, old)
      head = head.replace(old, new)
    return head

  def diff(
      self,
      head: str,
      base: str | None = None,
      files: Collection[str] | None = None,
      class_reader: shards.Callable[[str], list[str]] = (
          shards.declared_test_classes
      ),
  ) -> tuple[set[str] | None, str]:
    """Runs `shards_from_table_diff` against `self.files` and `self.base`."""
    return shards.shards_from_table_diff(
        self.files if files is None else files,
        self.base if base is None else base,
        head,
        class_reader=class_reader,
    )

  def _changed(self, *paths: str) -> str:
    handle, name = tempfile.mkstemp()
    with os.fdopen(handle, 'w') as f:
      f.write(''.join(f'{p}\0' for p in paths))
    self.addCleanup(os.unlink, name)
    return name

  @contextlib.contextmanager
  def versions(self, head: str, base: str | None = None) -> Iterator[None]:
    """Patches the readers `select` and `explain` go through."""
    with (
        mock.patch.object(
            shards,
            'read_base_shards',
            autospec=True,
            spec_set=True,
            return_value=self.base if base is None else base,
        ),
        mock.patch.object(
            shards,
            'read_head_shards',
            autospec=True,
            spec_set=True,
            return_value=head,
        ),
    ):
      yield

  def test_respec_round_trips(self) -> None:
    # The helpers the rest lean on: rewriting a spec to itself, in the helper's
    # own formatting, is not a change to the table.
    head = self.respec(self.other)
    self.assertNotEqual(head, self.base)
    self.assertEqual(split_shards_table(head), split_shards_table(self.base))

  def test_entries_accepts_an_unannotated_shards_assignment(self) -> None:
    plain = self.edit(('SHARDS: ShardMap = {', 'SHARDS = {'))
    self.assertEqual(
        split_shards_table(plain)[1], split_shards_table(self.base)[1]
    )
    self.assertEqual(set(self.entries(plain)), set(self.table))

  def test_find_resplit_skips_single_shard_node_ids(self) -> None:
    # A file with 2+ node IDs in one shard and none in any other must not
    # raise StopIteration.
    only_one = {
        'a': {'paths': ('pkg/t_test.py::A', 'pkg/t_test.py::B')},
        CATCH_ALL: {'paths': None},
    }
    self.assertIsNone(_find_resplit(only_one))

  def test_no_edit_selects_nothing(self) -> None:
    names, reason = self.diff(self.base)
    self.assertEqual(names, set())
    self.assertIn('table-only change; 0 of', reason)

  def test_a_minutes_only_edit_selects_nothing(self) -> None:
    spec = self.table[self.other]
    head = self.respec(
        self.other,
        minutes=spec.get('minutes', 1) + 1,
        compat_minutes=spec.get('compat_minutes', 1) + 1,
    )
    self.assertEqual(self.diff(head)[0], set())

  def test_a_renamed_shard_with_the_same_paths_selects_nothing(self) -> None:
    head = self.respec(self.other, rename=f'{self.other}-renamed')
    self.assertEqual(self.diff(head)[0], set())

  def test_a_comment_or_formatting_edit_selects_nothing(self) -> None:
    # Comments are not in the syntax tree, so even one inside a function is
    # not a change to what the function does.
    head = self.edit(
        ('# Sibling module, same directory', '# Sibling module, same dir'),
        (
            "    if '__pycache__' in dirpath:\n",
            "    if '__pycache__' in dirpath:  # bytecode\n",
        ),
    )
    self.assertEqual(self.diff(head)[0], set())

  def test_moving_a_file_selects_both_shards(self) -> None:
    names, reason = self.diff(self.move())
    # The shard it moved to gained a path, and the one it left lost one.
    self.assertEqual(names, {self.move_from, self.move_to})
    self.assertIn('table-only change; 2 of', reason)

  def test_move_preserves_earlier_edits_in_source(self) -> None:
    dropped = self.replace(self.other, None)
    names, _ = self.diff(self.move(dropped), files=self.claimed)
    self.assertEqual(names, {self.move_from, self.move_to, CATCH_ALL})

  def test_a_new_shard_selects_it(self) -> None:
    # 3f41997, replayed: the files were in the catch-all, and a new shard
    # claims them. The catch-all is left empty and dropped from the head, so
    # it is not selected.
    names, _ = self.diff(
        self.base, base=self.replace(self.other, None), files=self.claimed
    )
    self.assertEqual(names, {self.other})

  def test_a_new_shard_selects_a_catch_all_it_leaves_nonempty(self) -> None:
    # The same, with another file still in the catch-all: it is in the head,
    # and it lost the new shard's files, so it is selected too.
    files = [*self.claimed, self.UNCLAIMED]
    names, _ = self.diff(
        self.base, base=self.replace(self.other, None), files=files
    )
    self.assertEqual(names, {self.other, CATCH_ALL})

  def test_an_unchanged_nonempty_catch_all_selects_nothing(self) -> None:
    files = [*self.claimed, self.UNCLAIMED]
    self.assertEqual(self.diff(self.base, files=files)[0], set())

  def test_dropping_a_shard_selects_the_catch_all(self) -> None:
    # Its files fall into the catch-all, which is the only head shard whose
    # paths changed.
    names, _ = self.diff(self.replace(self.other, None), files=self.claimed)
    self.assertEqual(names, {CATCH_ALL})

  def test_a_head_table_without_a_proper_catch_all_runs_everything(
      self,
  ) -> None:
    edits = {
        'missing': self.replace(CATCH_ALL, None),
        'with paths': self.respec(CATCH_ALL, paths=(self.moved,)),
    }
    for what, head in edits.items():
      with self.subTest(what=what):
        names, reason = self.diff(head)
        self.assertIsNone(names)
        self.assertIn('could not resolve the SHARDS table', reason)
        self.assertIn('paths=None', reason)

  def test_a_class_resplit_selects_both_sides(self) -> None:
    # Always exercise splitting a whole-file entry across two shards by
    # `::Class`, plus the live table's node-ID split when one is present.
    kept = tuple(
        p for p in self.table[self.move_from]['paths'] if p != self.moved
    )
    head = self.respec(
        self.move_from, paths=(*kept, f'{self.moved}::FirstTest')
    )
    head = self.respec(
        self.move_to,
        head,
        paths=(*self.table[self.move_to]['paths'], f'{self.moved}::SecondTest'),
    )
    reader = lambda f: (
        ['FirstTest', 'SecondTest']
        if f == self.moved
        else shards.declared_test_classes(f)
    )
    names, _ = self.diff(head, class_reader=reader)
    self.assertEqual(names, {self.move_from, self.move_to})

    if self.resplit is not None:
      src, path, dst = self.resplit
      head = self.respec(
          src, paths=tuple(p for p in self.table[src]['paths'] if p != path)
      )
      head = self.respec(dst, head, paths=(*self.table[dst]['paths'], path))
      names, _ = self.diff(head)
      self.assertEqual(names, {src, dst})

  def test_a_diff_over_max_diff_shards_runs_everything(self) -> None:
    with mock.patch.object(shards, 'MAX_DIFF_SHARDS', 1):
      names, reason = self.diff(self.move())
    self.assertIsNone(names)
    self.assertIn(
        f'table diff touches 2 of {len(self.resolved)} shards, over the'
        ' diff-aware bound',
        reason,
    )

  def test_a_diff_over_max_diff_fraction_runs_everything(self) -> None:
    # Two shards is over the fraction, and well under the count.
    with mock.patch.object(
        shards, 'MAX_DIFF_FRACTION', 1.5 / len(self.resolved)
    ):
      names, reason = self.diff(self.move())
    self.assertIsNone(names)
    self.assertIn('over the diff-aware bound', reason)

  def test_the_bound_is_inclusive(self) -> None:
    with (
        mock.patch.object(shards, 'MAX_DIFF_SHARDS', 2),
        mock.patch.object(shards, 'MAX_DIFF_FRACTION', 2 / len(self.resolved)),
    ):
      self.assertEqual(
          self.diff(self.move())[0], {self.move_from, self.move_to}
      )

  def test_an_edit_outside_the_table_runs_everything(self) -> None:
    minutes = shards.COMPAT_BIN_MINUTES
    runner = next(iter(shards.RUNNERS.values()))
    edits = {
        'function': (
            "    if '__pycache__' in dirpath:\n",
            "    if '__pycache__' in dirpath or '.git' in dirpath:\n",
        ),
        'selection flag': (
            "    '--ignore-glob=*/test_utils.py',\n",
            "    '--ignore-glob=*/test_utils.py',\n    '-x',\n",
        ),
        'constant': (
            f'COMPAT_BIN_MINUTES = {minutes}',
            f'COMPAT_BIN_MINUTES = {minutes + 10}',
        ),
        'runner': (repr(runner), repr((runner[0], f'{runner[1]}-renamed'))),
    }
    for what, pair in edits.items():
      with self.subTest(what=what):
        names, reason = self.diff(self.edit(pair))
        self.assertIsNone(names)
        self.assertIn('changes outside the SHARDS table', reason)

  def test_a_table_edit_with_a_function_edit_runs_everything(self) -> None:
    head = self.edit(
        ('return sorted_jax_versions()[0]', 'return sorted_jax_versions()[-1]'),
        source=self.move(),
    )
    self.assertIsNone(self.diff(head)[0])

  def test_read_base_shards_queries_the_merge_base(self) -> None:
    def fake_run(
        cmd: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
      out = 'abc123\n' if cmd[1] == 'merge-base' else self.base
      return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr='')

    cases = (
        ({'BASE_REF': 'release', 'GITHUB_BASE_REF': 'pr-base'}, 'release'),
        ({'BASE_REF': '', 'GITHUB_BASE_REF': 'pr-base'}, 'pr-base'),
        ({'BASE_REF': '', 'GITHUB_BASE_REF': ''}, 'main'),
    )
    for env, branch in cases:
      with self.subTest(branch=branch):
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                shards.subprocess,
                'run',
                autospec=True,
                spec_set=True,
                side_effect=fake_run,
            ) as run,
        ):
          self.assertEqual(shards.read_base_shards(), self.base)
        self.assertEqual(
            [c.args[0] for c in run.call_args_list],
            [
                ['git', 'merge-base', f'origin/{branch}', 'HEAD'],
                ['git', 'show', f'abc123:{shards.SHARDS_PATH}'],
            ],
        )

  def test_a_base_version_that_fails_to_load_runs_everything(self) -> None:
    error = subprocess.CalledProcessError(128, ['git', 'show'])
    with mock.patch.object(
        shards,
        'read_base_shards',
        autospec=True,
        spec_set=True,
        side_effect=error,
    ):
      names, reason = shards.shards_from_table_diff(
          self.files, head_source=self.base
      )
    self.assertIsNone(names)
    self.assertIn('could not load the base version', reason)

  def test_a_head_version_that_fails_to_load_runs_everything(self) -> None:
    with mock.patch.object(
        shards,
        'read_head_shards',
        autospec=True,
        spec_set=True,
        side_effect=OSError('unreadable'),
    ):
      names, reason = shards.shards_from_table_diff(
          self.files, base_source=self.base
      )
    self.assertIsNone(names)
    self.assertIn('could not load the head version', reason)

  def test_a_missing_git_runs_everything(self) -> None:
    with mock.patch.object(
        shards.subprocess,
        'run',
        autospec=True,
        spec_set=True,
        side_effect=FileNotFoundError('git'),
    ):
      names, reason = shards.shards_from_table_diff(
          self.files, head_source=self.base
      )
    self.assertIsNone(names)
    self.assertIn('could not load the base version', reason)

  def test_a_base_version_that_does_not_parse_runs_everything(self) -> None:
    cases = {
        'syntax error': ('SHARDS = {', 'could not read the SHARDS table'),
        'missing SHARDS': ('X = 1\n', '0 assignments to SHARDS, expected 1'),
        'not a dict display': (
            'SHARDS = dict(a=1)\n',
            'SHARDS is not a dict display',
        ),
        'dict unpacking': (
            'SHARDS = {**OTHER}\n',
            'SHARDS unpacks another mapping',
        ),
        'non-string key': (
            "SHARDS = {1: Spec(paths=('a_test.py',))}\n",
            'shard name is not a string',
        ),
        'unhashable key': (
            "SHARDS = {['a']: Spec(paths=('a_test.py',))}\n",
            'shard name is not a string',
        ),
        'repeated keyword': (
            (
                "SHARDS = {'a': Spec(paths=('a_test.py',), minutes=1,"
                ' minutes=2)}\n'
            ),
            'a shard spec repeats a keyword',
        ),
        'unhashable spec literal': (
            "SHARDS = {'a': Spec(paths=({['x']: 1},))}\n",
            'not a literal shard spec',
        ),
        'missing paths': (
            "SHARDS = {'a': Spec(minutes=1)}\n",
            'shard spec has no paths',
        ),
        'non-tuple paths': (
            "SHARDS = {'a': Spec(paths=['a_test.py'])}\n",
            'shard paths is not a tuple of strings',
        ),
        'non-catch-all paths=None': (
            "SHARDS = {'a': Spec(paths=None), 'catch-all': Spec(paths=None)}\n",
            "shard 'a' must not have paths=None",
        ),
    }
    for what, (base, expected) in cases.items():
      with self.subTest(what=what):
        names, reason = self.diff(self.base, base=base)
        self.assertIsNone(names)
        self.assertIn(expected, reason)

  def test_a_shard_declared_twice_runs_everything(self) -> None:
    # Python would keep the second silently, hiding the first from every check.
    head = self.respec(self.other, rename=self.move_to)
    names, reason = self.diff(head)
    self.assertIsNone(names)
    self.assertIn('declared twice', reason)

  def test_a_spec_that_is_not_a_literal_runs_everything(self) -> None:
    paths = self.table[self.other]['paths']
    head = self.replace(
        self.other, f'{self.other!r}: Spec(paths={paths!r}, minutes=max(1, 2))'
    )
    names, reason = self.diff(head)
    self.assertIsNone(names)
    self.assertIn('could not read the SHARDS table', reason)

  def test_a_head_table_failing_check_consistency_runs_everything(
      self,
  ) -> None:
    gone = 'tokamax/_src/gone_test.py'
    head = self.respec(
        self.other, paths=(*self.table[self.other]['paths'], gone)
    )
    names, reason = self.diff(head)
    self.assertIsNone(names)
    self.assertIn('head table fails check_consistency', reason)
    self.assertIn(gone, reason)

  def test_select_runs_only_the_table_diff(self) -> None:
    with self.versions(self.move()):
      only, reason = shards.select(
          self.resolved, self.files, self._changed(shards.SHARDS_PATH)
      )
    self.assertEqual(only, {self.move_from, self.move_to})
    self.assertIn('table-only change; 2 of', reason)

  def test_select_after_a_minutes_only_edit_runs_nothing(self) -> None:
    minutes = self.table[self.other].get('minutes', 1) + 1
    with self.versions(self.respec(self.other, minutes=minutes)):
      only, _ = shards.select(
          self.resolved, self.files, self._changed(shards.SHARDS_PATH)
      )
    self.assertEqual(only, set())

  def test_select_fails_open_with_the_reason(self) -> None:
    error = subprocess.CalledProcessError(128, ['git', 'show'])
    with mock.patch.object(
        shards,
        'read_base_shards',
        autospec=True,
        spec_set=True,
        side_effect=error,
    ):
      only, reason = shards.select(
          self.resolved, self.files, self._changed(shards.SHARDS_PATH)
      )
    self.assertIsNone(only)
    self.assertEqual(
        reason, 'every shard runs: shards.py: could not load the base version'
    )

  def test_select_fails_open_when_table_diff_names_an_unknown_shard(
      self,
  ) -> None:
    without_move_to = {
        k: v for k, v in self.resolved.items() if k != self.move_to
    }
    with self.versions(self.move()):
      only, reason = shards.select(
          without_move_to, self.files, self._changed(shards.SHARDS_PATH)
      )
    self.assertIsNone(only)
    self.assertEqual(
        reason,
        'every shard runs: shards.py: table diff names unknown shard'
        f' {self.move_to}',
    )

  def test_select_unions_the_table_diff_with_the_rest_of_the_change(
      self,
  ) -> None:
    test = 'tokamax/tokamax_test.py'
    reached, _ = shards.select(self.resolved, self.files, self._changed(test))
    self.assertTrue(reached)
    with self.versions(self.move()):
      only, _ = shards.select(
          self.resolved, self.files, self._changed(shards.SHARDS_PATH, test)
      )
    self.assertEqual(only, {self.move_from, self.move_to} | reached)

  def test_any_other_path_outside_the_package_still_runs_everything(
      self,
  ) -> None:
    for other in ('.github/ci/deps.py', 'pyproject.toml'):
      with self.subTest(other=other):
        with self.versions(self.move()):
          only, reason = shards.select(
              self.resolved,
              self.files,
              self._changed(shards.SHARDS_PATH, other),
          )
        self.assertIsNone(only)
        self.assertIn(f'{other} is outside tokamax/', reason)

  def test_explain_says_why(self) -> None:
    with self.versions(self.move()):
      body = '\n'.join(
          shards.explain(
              self.resolved, self.files, self._changed(shards.SHARDS_PATH)
          )
      )
    self.assertIn(
        f"shards.py: table-only change; 2 of {len(self.resolved)} shards'"
        ' test paths changed',
        body,
    )
    for name in (self.move_from, self.move_to):
      self.assertRegex(
          body,
          rf'(?m)^  {re.escape(name)}\s+.*\n'
          r'      table   test paths changed in \.github/ci/shards\.py$',
      )
    self.assertNotIn(f'  {self.other:32s} ', body)

  def test_explain_fails_open_with_the_reason(self) -> None:
    minutes = shards.COMPAT_BIN_MINUTES
    with self.versions(
        self.edit((
            f'COMPAT_BIN_MINUTES = {minutes}',
            f'COMPAT_BIN_MINUTES = {minutes + 10}',
        ))
    ):
      body = shards.explain(
          self.resolved, self.files, self._changed(shards.SHARDS_PATH)
      )
    self.assertEqual(
        body, ['every shard runs: shards.py: changes outside the SHARDS table']
    )


class DryRunTest(unittest.TestCase):
  """`--dry-run`, which is how selection is landed without it deciding.

  The mode exists so the selection can be read against real pull requests
  before it gates one. What it must guarantee is therefore narrow and exact:
  the report is the one selection would really have produced, and the matrix
  is untouched.
  """

  def _matrix(self, *argv: str) -> tuple[dict[str, str], str]:
    """Runs `matrix` and returns its `key=value` outputs and its stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
      shards.main(['matrix', *argv])
    outputs = dict(
        line.split('=', 1)
        for line in out.getvalue().splitlines()
        if '=' in line
    )
    return outputs, err.getvalue()

  def _changed(self, *paths: str) -> str:
    handle, name = tempfile.mkstemp()
    with os.fdopen(handle, 'w') as f:
      f.write(''.join(f'{p}\0' for p in paths))
    self.addCleanup(os.unlink, name)
    return name

  def test_dry_run_reports_the_selection_and_still_runs_everything(
      self,
  ) -> None:
    changed = self._changed('tokamax/_src/ops/attention/base_test.py')
    live, _ = self._matrix('--changed-from', changed)
    dry, log = self._matrix('--changed-from', changed, '--dry-run')
    full, _ = self._matrix()

    # Worth asserting that the selection is real before asserting that the
    # dry run ignores it: if `select` returned everything, the interesting
    # half of this test would pass for the wrong reason.
    self.assertNotEqual(live['include'], full['include'])
    self.assertEqual(dry['include'], full['include'])
    self.assertNotEqual(live['compat_include'], full['compat_include'])
    self.assertEqual(dry['compat_include'], full['compat_include'])
    # The reason reaches the run summary through this line only.
    self.assertIn('DRY RUN', log.partition('\n')[0])
    self.assertIn('would select', log)

  def test_dry_run_without_a_change_list_changes_nothing(self) -> None:
    dry, log = self._matrix('--dry-run')
    full, _ = self._matrix()
    self.assertEqual(dry['include'], full['include'])
    # Nothing was narrowed, so there is no dry run to announce.
    self.assertNotIn('DRY RUN', log)

  def test_dry_run_leaves_the_other_outputs_alone(self) -> None:
    changed = self._changed('README.md')
    dry, _ = self._matrix('--changed-from', changed, '--dry-run')
    full, _ = self._matrix()
    self.assertEqual(dry['pytest_flags'], full['pytest_flags'])
    self.assertEqual(dry['catch_all'], full['catch_all'])
    # A documentation-only change selects nothing, and `include` being `[]`
    # is what skips `shard-tests`. A dry run must not trip that path.
    self.assertNotEqual(dry['include'], '[]')
    self.assertNotEqual(dry['compat_include'], '[]')

  def test_documentation_selects_no_compat_bin_either(self) -> None:
    live, _ = self._matrix('--changed-from', self._changed('README.md'))
    self.assertEqual(live['include'], '[]')
    self.assertEqual(live['compat_include'], '[]')


if __name__ == '__main__':
  unittest.main()
