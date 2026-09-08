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
"""Tests for `deps.py`, run against a fake package of fake changes.

The fake package is named `tokamax` and built in a temp directory, so these
exercise the real code paths -- real `ast` parsing, the real `PACKAGE_ROOT`
check -- rather than a version of them parameterised for testing.

Stdlib `unittest`, no pytest and no third-party imports, because this has to
run in the `shard-matrix` job, which is the job that runs before anything is
installed.

  python3 -m unittest discover -s .github/ci -p '*_test.py'
"""

import os
import tempfile
import textwrap
import unittest

import deps

# A small package with one of each edge that has ever been a problem: a chain
# to walk transitively, an import guarded by a backend check, an import inside
# a function, a relative import, a cycle, a file that only the package
# `__init__` reaches, and a source file no test imports at all.
FAKE_PACKAGE = {
    'tokamax/__init__.py': 'from tokamax import api\n',
    'tokamax/conftest.py': 'import pytest\n',
    'tokamax/api.py': 'from tokamax._src import op\n',
    'tokamax/data/tuning.json': '{}\n',
    # core <- util <- op <- op_test, the chain the reverse closure must walk.
    'tokamax/_src/__init__.py': '',
    'tokamax/_src/core.py': 'import jax\n',
    'tokamax/_src/util.py': 'from tokamax._src import core\n',
    'tokamax/_src/op.py': 'from tokamax._src import util\n',
    'tokamax/_src/op_test.py': 'from tokamax._src import op\n',
    'tokamax/_src/core_test.py': 'from tokamax._src import core\n',
    # The `ops/attention/api.py` shape: the implementation is imported only
    # when the backend has it, so the edge exists only inside an `if`.
    'tokamax/_src/ops/__init__.py': '',
    'tokamax/_src/ops/backend.py': (
        'HAVE_CUDA = False\n'
        'if HAVE_CUDA:\n'
        '  from tokamax._src.ops import triton\n'
    ),
    'tokamax/_src/ops/triton.py': 'from tokamax._src import core\n',
    'tokamax/_src/ops/backend_test.py': (
        'from tokamax._src.ops import backend\n'
    ),
    # An import inside a function body, and a relative import.
    'tokamax/_src/ops/lazy.py': (
        'def f():\n  from tokamax._src.ops import triton\n  return triton\n'
    ),
    'tokamax/_src/ops/relative.py': 'from . import lazy\n',
    'tokamax/_src/ops/relative_test.py': (
        'from tokamax._src.ops import relative\n'
    ),
    # A cycle. Neither file is reachable from a test; the point is that the
    # walk terminates.
    'tokamax/_src/cycle_a.py': 'from tokamax._src import cycle_b\n',
    'tokamax/_src/cycle_b.py': 'from tokamax._src import cycle_a\n',
    # Imported by nothing: changing it must select no test, not every test.
    'tokamax/_src/orphan.py': 'import jax\n',
}

# Kept out of the package above and written only by the tests that want it.
# An unparseable file anywhere makes every selection a full run, so leaving
# one in the fixture would quietly turn the rest of these tests into
# assertions about the fail-open path instead of about the graph.
UNPARSEABLE = 'def f(\n'


class DepsTestCase(unittest.TestCase):
  """Builds FAKE_PACKAGE in a temp dir and runs from inside it."""

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    for path, text in FAKE_PACKAGE.items():
      full = os.path.join(tmp.name, path)
      os.makedirs(os.path.dirname(full), exist_ok=True)
      with open(full, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent(text))

    cwd = os.getcwd()
    self.addCleanup(os.chdir, cwd)
    os.chdir(tmp.name)

    self.files = deps.python_files()
    self.edges, self.unparsed = deps.import_graph(self.files)

  def add(self, path, text):
    """Writes a file into the fake package and rebuilds the graph."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
      f.write(textwrap.dedent(text))
    self.files = deps.python_files()
    self.edges, self.unparsed = deps.import_graph(self.files)

  def affected(self, *changed):
    tests, reason = deps.affected_tests(
        list(changed), self.edges, self.unparsed, self.files
    )
    return tests, reason


class GraphTest(DepsTestCase):

  def test_walks_a_transitive_chain(self):
    # core <- util <- op <- op_test. The chain is the reason a glob cannot do
    # this: nothing about op_test.py's path mentions core.py.
    tests, _ = self.affected('tokamax/_src/core.py')
    self.assertIn('tokamax/_src/op_test.py', tests)
    self.assertIn('tokamax/_src/core_test.py', tests)

  def test_does_not_select_unrelated_tests(self):
    # The value of the whole exercise. If this ever fails open, the graph is
    # correct but useless.
    tests, _ = self.affected('tokamax/_src/ops/relative.py')
    self.assertEqual(tests, {'tokamax/_src/ops/relative_test.py'})

  def test_changed_test_file_runs_itself(self):
    tests, _ = self.affected('tokamax/_src/core_test.py')
    self.assertIn('tokamax/_src/core_test.py', tests)

  def test_follows_import_guarded_by_a_backend_check(self):
    # `ops/attention/api.py` imports pallas_triton only under `if cuda`. A
    # scan of module-level statements would miss it and drop a real edge.
    tests, _ = self.affected('tokamax/_src/ops/triton.py')
    self.assertIn('tokamax/_src/ops/backend_test.py', tests)

  def test_follows_import_inside_a_function(self):
    tests, _ = self.affected('tokamax/_src/ops/triton.py')
    self.assertIn('tokamax/_src/ops/relative_test.py', tests)

  def test_resolves_relative_imports(self):
    tests, _ = self.affected('tokamax/_src/ops/lazy.py')
    self.assertIn('tokamax/_src/ops/relative_test.py', tests)

  def test_importing_a_submodule_depends_on_the_package_init(self):
    # `from tokamax._src import core` executes `_src/__init__.py`, so editing
    # the `__init__` has to select everything under it.
    tests, _ = self.affected('tokamax/_src/__init__.py')
    self.assertIn('tokamax/_src/core_test.py', tests)

  def test_cycles_terminate(self):
    tests, _ = self.affected('tokamax/_src/cycle_a.py')
    self.assertEqual(tests, set())

  def test_file_no_test_reaches_selects_nothing(self):
    # Empty, not None: no test covers it, so there is nothing to run. This is
    # a coverage fact, and `--graph` reports it rather than hiding it.
    tests, reason = self.affected('tokamax/_src/orphan.py')
    self.assertEqual(tests, set())
    self.assertIsNone(reason)


class FailOpenTest(DepsTestCase):
  """Every case where the graph cannot answer must select the full suite."""

  def assert_full_run(self, path):
    tests, reason = self.affected(path)
    self.assertIsNone(tests, f'{path} should force a full run')
    self.assertIn(os.path.basename(path), reason)

  def test_change_outside_the_package(self):
    self.assert_full_run('pyproject.toml')
    self.assert_full_run('.github/workflows/ci-build.yml')
    self.assert_full_run('.github/ci/shards.py')

  def test_conftest(self):
    self.assert_full_run('tokamax/conftest.py')

  def test_package_data(self):
    # The autotuning JSON decides which kernel configuration runs and is read
    # by path, so no import edge covers it.
    self.assert_full_run('tokamax/data/tuning.json')

  def test_documentation_does_not_force_a_full_run(self):
    # The one carve-out from "not Python means full run": a README cannot
    # change a kernel result, and a typo fix should not cost an accelerator.
    tests, reason = self.affected('tokamax/README.md')
    self.assertEqual(tests, set())
    self.assertIsNone(reason)

  def test_documentation_outside_the_package_is_inert_too(self):
    # Regression: the inert-suffix rule used to sit below the "outside
    # tokamax/" rule, so it matched only the one markdown file that happens to
    # live inside the package, and every README or docs/ change cost a full
    # run of every shard. Documentation is the common case for a no-test PR,
    # so getting this backwards was the expensive way round.
    for path in ('README.md', 'docs/index.md', 'docs/ops/kda/index.md'):
      with self.subTest(path=path):
        tests, reason = self.affected(path)
        self.assertEqual(tests, set())
        self.assertIsNone(reason)

  def test_documentation_next_to_data_still_yields_to_the_data(self):
    tests, _ = self.affected('tokamax/README.md', 'tokamax/data/tuning.json')
    self.assertIsNone(tests)

  def test_inert_files_do_not_exempt_the_rest_of_the_changeset(self):
    # Skipping an inert path must not skip the loop. The docs-only case is
    # cheap precisely because it is only reached when nothing else is there.
    tests, reason = self.affected('README.md', 'pyproject.toml')
    self.assertIsNone(tests)
    self.assertIn('pyproject.toml', reason)

  def test_unparseable_file(self):
    self.add('tokamax/_src/broken.py', UNPARSEABLE)
    self.assert_full_run('tokamax/_src/broken.py')

  def test_an_unparseable_file_anywhere_forces_a_full_run(self):
    # The rule is a property of the graph, not of the changeset. A file that
    # does not parse has no outgoing edges, so paths through it are missing
    # and no narrowing is sound -- including for a change to a file that
    # parses. The realistic cause is a parser older than the syntax in the
    # tree, where the unparsed files are infrastructure nobody is editing:
    # at python 3.11 that would be 53 of tokamax's 278 files.
    self.add('tokamax/_src/broken.py', UNPARSEABLE)
    tests, reason = self.affected('tokamax/_src/core.py')
    self.assertIsNone(tests)
    self.assertIn('broken.py', reason)

  def test_a_prebuilt_graph_without_unparsed_still_fails_open(self):
    # `edges` and `unparsed` are two halves of one answer, and a caller that
    # passes only the first used to get a narrowed selection with the
    # unreadable-file rule silently switched off -- the one failure mode this
    # module exists to prevent, reachable by omitting an argument.
    self.add('tokamax/_src/broken.py', UNPARSEABLE)
    tests, reason = deps.affected_tests(
        ['tokamax/_src/core.py'], self.edges, None, self.files
    )
    self.assertIsNone(tests)
    self.assertIn('broken.py', reason)

  def test_one_unanalysable_file_taints_the_whole_changeset(self):
    tests, _ = self.affected('tokamax/_src/core.py', 'tokamax/data/tuning.json')
    self.assertIsNone(tests)

  def test_deleted_file(self):
    # The one case that looks answerable and is not. The graph is built from
    # the tree as it is now, so a deleted file has no node and no dependents:
    # without this rule it selects nothing, while every file that imported it
    # is broken.
    self.assert_full_run('tokamax/_src/deleted.py')

  def test_renamed_file_is_a_delete_and_an_add(self):
    # git reports a rename as both paths. The old one is gone, which is what
    # forces the full run; the new one on its own would not.
    tests, _ = self.affected(
        'tokamax/_src/core.py', 'tokamax/_src/core_renamed.py'
    )
    self.assertIsNone(tests)


class NewFileTest(DepsTestCase):
  """Adding a file, which is where a change-detection scheme usually leaks."""

  def test_new_test_file_selects_itself(self):
    self.add('tokamax/_src/ops/new_test.py', 'from tokamax._src import util\n')
    tests, _ = self.affected('tokamax/_src/ops/new_test.py')
    self.assertEqual(tests, {'tokamax/_src/ops/new_test.py'})

  def test_new_test_file_is_selected_by_what_it_imports(self):
    # The property that makes adding a test safe without touching this file:
    # the new test is in the graph the moment it exists, so a later edit to
    # `core.py` selects it with no configuration anywhere.
    self.add('tokamax/_src/ops/new_test.py', 'from tokamax._src import core\n')
    tests, _ = self.affected('tokamax/_src/core.py')
    self.assertIn('tokamax/_src/ops/new_test.py', tests)

  def test_new_source_file_selects_its_importers_tests(self):
    self.add('tokamax/_src/helper.py', 'import jax\n')
    self.add(
        'tokamax/_src/util.py',
        'from tokamax._src import core\nfrom tokamax._src import helper\n',
    )
    tests, _ = self.affected('tokamax/_src/helper.py')
    self.assertIn('tokamax/_src/op_test.py', tests)

  def test_new_untested_source_file_selects_nothing(self):
    # Honest rather than safe-looking: nothing imports it, so no test covers
    # it, and running the full suite would not cover it either.
    self.add('tokamax/_src/brand_new.py', 'import jax\n')
    tests, reason = self.affected('tokamax/_src/brand_new.py')
    self.assertEqual(tests, set())
    self.assertIsNone(reason)


class ClosureInvariantTest(DepsTestCase):
  """The property the whole module exists for, checked by brute force.

  `affected_tests` walks forward from each test and inverts. This recomputes
  the same answer by fixpoint relaxation over the edge set -- a different
  algorithm, so agreement is evidence rather than a restatement.
  """

  def naive_reach(self):
    reach = {f: set(self.edges.get(f, ())) for f in self.files}
    changed = True
    while changed:
      changed = False
      for f in self.files:
        grown = set(reach[f]).union(*(reach.get(g, set()) for g in reach[f]))
        if grown != reach[f]:
          reach[f] = grown
          changed = True
    return reach

  def test_no_affected_test_is_ever_missed(self):
    reach = self.naive_reach()
    tests = [f for f in self.files if deps.is_test_file(f)]
    for source in self.files:
      if source in self.unparsed:
        continue
      expected = {t for t in tests if source in reach[t] or t == source}
      actual, reason = self.affected(source)
      if actual is None:
        continue  # fail-open is always allowed to be broader
      self.assertEqual(
          actual, expected, f'wrong affected set for {source} ({reason})'
      )


if __name__ == '__main__':
  unittest.main()
