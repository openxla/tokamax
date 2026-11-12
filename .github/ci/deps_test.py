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

import ast
import os
import re
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
    # An op the autotuning cache serves, and its table. The table is named
    # after the class, as `cache.py` names it, so module and table disagree
    # the way they do in the real package: nothing about `ragged.py` says
    # `pallas_mosaic_tpu_ragged_dot`.
    'tokamax/_src/ops/ragged.py': (
        'import jax\n\n\nclass PallasMosaicTpuRaggedDot(Op):\n  pass\n'
    ),
    'tokamax/_src/ops/ragged_test.py': 'from tokamax._src.ops import ragged\n',
    'tokamax/data/autotuning/tpu7x/pallas_mosaic_tpu_ragged_dot.json': '{}\n',
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

  def remove(self, path):
    """Deletes a file from the fake package and rebuilds the graph.

    The graph is rebuilt from what is left, which is the situation the CI job
    is in: it checks out the merge result and never sees the old tree.
    """
    os.remove(path)
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

  def test_importing_a_submodule_records_an_edge_to_the_package_init(self):
    # `from tokamax._src import core` executes `_src/__init__.py`, and the
    # graph does record that edge -- the package imported *from* is named by
    # the statement. Asserted against the plain graph rather than through
    # `affected_tests`, which also adds the edges no statement names; this one
    # has to be there without them.
    index = deps.dependents(self.edges, self.files)
    self.assertIn(
        'tokamax/_src/core_test.py', index['tokamax/_src/__init__.py']
    )

  def test_follows_lazy_submodules_declared_in_package_all(self):
    self.add('tokamax/ops/__init__.py', '__all__ = ("tpu",)\n')
    self.add('tokamax/ops/tpu.py', 'from tokamax._src.ops import ragged\n')
    self.add('tokamax/ops/lazy_test.py', 'from tokamax import ops\n')
    tests, _ = self.affected('tokamax/_src/ops/ragged.py')
    self.assertIn('tokamax/ops/lazy_test.py', tests)

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
    for path in (
        'README.md',
        'docs/index.md',
        'docs/ops/kda/index.md',
        'docs/tokamax_ops.ipynb',
    ):
      with self.subTest(path=path):
        tests, reason = self.affected(path)
        self.assertEqual(tests, set())
        self.assertIsNone(reason)

  def test_documentation_next_to_data_still_yields_to_the_data(self):
    tests, _ = self.affected('tokamax/README.md', 'tokamax/data/tuning.json')
    self.assertIsNone(tests)

  def test_workflows_other_than_ci_build_are_inert(self):
    # CI-Build runs none of them, so every shard ran without testing them.
    for path in (
        '.github/workflows/ci-nightly.yml',
        '.github/workflows/nightly_benchmarks.yml',
        '.github/workflows/publish.yml',
        '.github/workflows/run_benchmarks.yml',
        '.github/workflows/a_new_workflow.yml',
    ):
      with self.subTest(path=path):
        tests, reason = self.affected(path)
        self.assertEqual(tests, set())
        self.assertIsNone(reason)

  def test_ci_build_and_its_scripts_still_force_a_full_run(self):
    self.assert_full_run('.github/workflows/ci-build.yml')
    self.assert_full_run('.github/ci/install_deps.sh')

  def test_other_workflows_do_not_exempt_the_rest_of_the_changeset(self):
    tests, reason = self.affected(
        '.github/workflows/ci-nightly.yml', '.github/ci/install_deps.sh'
    )
    self.assertIsNone(tests)
    self.assertIn('install_deps.sh', reason)

  def test_any_file_under_docs_is_inert(self):
    # The site's table of contents is YAML, which no suffix rule covers, and
    # every docs change that touched it used to run every shard.
    for path in ('docs/_toc.yaml', 'docs/ops/kda/images/diagram.png'):
      with self.subTest(path=path):
        tests, reason = self.affected(path)
        self.assertEqual(tests, set())
        self.assertIsNone(reason)

  def test_only_the_top_level_docs_directory_is_inert(self):
    # Package data that happens to sit in a `docs` directory is still package
    # data, and a sibling whose name merely starts with `docs` is not docs.
    self.assert_full_run('tokamax/docs/table.yaml')
    self.assert_full_run('docs_build/config.yaml')

  def test_docs_do_not_exempt_the_rest_of_the_changeset(self):
    tests, reason = self.affected('docs/_toc.yaml', 'pyproject.toml')
    self.assertIsNone(tests)
    self.assertIn('pyproject.toml', reason)

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

  def test_deleted_package_init(self):
    # The one deletion the referrer scan is not trusted with: `module_name`
    # collapses an `__init__.py` to the package, so deleting it re-resolves
    # every sibling, and what is left behind is a namespace package rather
    # than an import error anyone would notice.
    self.remove('tokamax/_src/ops/__init__.py')
    tests, reason = self.affected('tokamax/_src/ops/__init__.py')
    self.assertIsNone(tests)
    self.assertIn('deleted', reason)

  def test_package_init_without_the_file_list(self):
    # Without `known`, an edited `__init__.py` cannot be told apart from a
    # deleted one, so it has to take the deleted one's answer.
    reason = deps.full_run_reason(['tokamax/_src/ops/__init__.py'])
    self.assertIsNotNone(reason)
    self.assertIn('__init__.py', reason)

  def test_a_file_merely_named_init_elsewhere_is_not_special(self):
    self.add('tokamax/_src/__init___helper.py', 'import jax\n')
    tests, reason = self.affected('tokamax/_src/__init___helper.py')
    self.assertEqual(tests, set())
    self.assertIsNone(reason)


class PackageInitTest(DepsTestCase):
  """An edited or added `__init__.py` selects the tests that run it."""

  ALL_TESTS = frozenset({
      'tokamax/_src/op_test.py',
      'tokamax/_src/core_test.py',
      'tokamax/_src/ops/backend_test.py',
      'tokamax/_src/ops/relative_test.py',
      'tokamax/_src/ops/ragged_test.py',
  })
  OPS_TESTS = frozenset({
      'tokamax/_src/ops/backend_test.py',
      'tokamax/_src/ops/relative_test.py',
      'tokamax/_src/ops/ragged_test.py',
  })

  def test_the_package_root_init_runs_for_every_test(self):
    self.add('tokamax/__init__.py', 'from tokamax import api\nX = 1\n')
    tests, reason = self.affected('tokamax/__init__.py')
    self.assertIsNone(reason)
    self.assertEqual(tests, self.ALL_TESTS)

  def test_a_subpackage_init_runs_only_for_tests_inside_it(self):
    # Nothing outside `_src/ops/` imports from it, so `op_test.py` and
    # `core_test.py` never run `_src/ops/__init__.py`.
    self.add('tokamax/_src/ops/__init__.py', 'X = 1\n')
    tests, reason = self.affected('tokamax/_src/ops/__init__.py')
    self.assertIsNone(reason)
    self.assertEqual(tests, self.OPS_TESTS)

  def test_a_test_runs_every_enclosing_init(self):
    # `relative_test.py` imports nothing from `_src` by that name, but pytest
    # imports it as `tokamax._src.ops.relative_test`, which runs
    # `_src/__init__.py` first.
    tests, _ = self.affected('tokamax/_src/__init__.py')
    self.assertIn('tokamax/_src/ops/relative_test.py', tests)

  def test_a_new_package_selects_only_its_own_tests(self):
    # The new-op case: a new directory with an `__init__.py`, its modules and
    # their tests. Nothing that existed before can import it.
    self.add('tokamax/_src/ops/moba/__init__.py', '')
    self.add('tokamax/_src/ops/moba/base.py', 'from tokamax._src import core\n')
    self.add(
        'tokamax/_src/ops/moba/base_test.py',
        'from tokamax._src.ops.moba import base\n',
    )
    tests, reason = self.affected(
        'tokamax/_src/ops/moba/__init__.py',
        'tokamax/_src/ops/moba/base.py',
        'tokamax/_src/ops/moba/base_test.py',
    )
    self.assertIsNone(reason)
    self.assertEqual(tests, {'tokamax/_src/ops/moba/base_test.py'})

  def test_follows_an_enclosing_init_that_imports_the_package(self):
    # `core_test.py` never names `kern`, but `_src/__init__.py` imports it, and
    # importing `tokamax._src.core` runs `_src/__init__.py` first.
    self.add('tokamax/_src/ops/kern/__init__.py', 'X = 1\n')
    self.add('tokamax/_src/__init__.py', 'from tokamax._src.ops import kern\n')
    tests, _ = self.affected('tokamax/_src/ops/kern/__init__.py')
    self.assertIn('tokamax/_src/core_test.py', tests)

  def test_an_init_does_not_widen_the_rest_of_the_changeset(self):
    # It used to force a full run for everything changed alongside it.
    tests, reason = self.affected(
        'tokamax/_src/orphan.py', 'tokamax/_src/ops/__init__.py'
    )
    self.assertIsNone(reason)
    self.assertEqual(tests, self.OPS_TESTS)

  def test_ordinary_edits_ignore_the_enclosing_init_edges(self):
    # Selected over the plain graph, as before: `tokamax/__init__.py` imports
    # `api`, which reaches `_src/op.py`, and every test runs
    # `tokamax/__init__.py` -- over the execution graph, every test would
    # select itself for any change to `op.py`.
    tests, _ = self.affected('tokamax/_src/op.py')
    self.assertEqual(tests, {'tokamax/_src/op_test.py'})

  def test_enclosing_inits(self):
    known = set(self.files)
    self.assertEqual(
        deps.enclosing_inits('tokamax/_src/ops/relative.py', known),
        {
            'tokamax/__init__.py',
            'tokamax/_src/__init__.py',
            'tokamax/_src/ops/__init__.py',
        },
    )
    # An `__init__.py` is its package: only the packages above it run first.
    self.assertEqual(
        deps.enclosing_inits('tokamax/_src/ops/__init__.py', known),
        {'tokamax/__init__.py', 'tokamax/_src/__init__.py'},
    )
    # A namespace package has no `__init__.py` to run.
    self.assertEqual(
        deps.enclosing_inits('tokamax/data/autotuning/x.py', known),
        {'tokamax/__init__.py'},
    )


class DeletedFileTest(DepsTestCase):
  """Deletions, which the graph built from the surviving tree cannot see.

  A deleted file is not a node, and `import_graph` drops the edges that
  pointed at it, so `dependents` finds nothing for it -- an answer identical
  to the one a genuinely inert change gets. What is recovered instead is the
  set of `import` statements that still spell its name; each file holding one
  is broken exactly as if it had been edited.
  """

  def test_a_file_nothing_imports_selects_nothing(self):
    # Empty and correct, not empty and lucky: `orphan.py` is named by no
    # import in the tree, so removing it cannot break a test.
    self.remove('tokamax/_src/orphan.py')
    tests, reason = self.affected('tokamax/_src/orphan.py')
    self.assertEqual(tests, set())
    self.assertIsNone(reason)

  def test_a_file_still_imported_selects_its_dependents(self):
    # The case that actually goes red: `util.py` and `core_test.py` still say
    # `from tokamax._src import core`, so both fail at collection. Reached
    # through the surviving import statement, then up the graph as usual --
    # `op_test.py` is three edges away and has to run.
    self.remove('tokamax/_src/core.py')
    tests, reason = self.affected('tokamax/_src/core.py')
    self.assertIsNone(reason)
    self.assertIn('tokamax/_src/core_test.py', tests)
    self.assertIn('tokamax/_src/op_test.py', tests)

  def test_a_deleted_chain_reaches_the_surviving_test(self):
    # Two links of a chain deleted at once. Neither is a node any more, so the
    # walk cannot be transitive -- it works because every link is itself in the
    # changeset and gets its own turn, and the last one is named by a file that
    # still exists.
    self.remove('tokamax/_src/util.py')
    self.remove('tokamax/_src/op.py')
    tests, reason = self.affected('tokamax/_src/util.py', 'tokamax/_src/op.py')
    self.assertIsNone(reason)
    self.assertIn('tokamax/_src/op_test.py', tests)

  def test_a_deleted_test_file_runs_nothing(self):
    self.remove('tokamax/_src/core_test.py')
    tests, reason = self.affected('tokamax/_src/core_test.py')
    self.assertIsNone(reason)
    self.assertNotIn('tokamax/_src/core_test.py', tests)

  def test_a_completed_rename_does_not_force_a_full_run(self):
    # The change this was built for. `git diff --no-renames` reports both
    # paths, and previously the old one alone cost the whole matrix. `op.py` is
    # named by exactly one file, so a finished rename -- new name in place, its
    # one importer updated -- leaves the old name spelled nowhere, and the
    # selection is as narrow as an ordinary edit of the same module.
    self.remove('tokamax/_src/op.py')
    self.add('tokamax/_src/op_renamed.py', 'from tokamax._src import util\n')
    self.add('tokamax/_src/op_test.py', 'from tokamax._src import op_renamed\n')
    tests, reason = self.affected(
        'tokamax/_src/op.py',
        'tokamax/_src/op_renamed.py',
        'tokamax/_src/op_test.py',
    )
    self.assertIsNone(reason)
    self.assertEqual(tests, {'tokamax/_src/op_test.py'})

  def test_a_half_finished_rename_selects_the_importers_that_were_missed(self):
    # The dangerous version of the same change, and the reason the scan reads
    # raw import names rather than graph edges. `core.py` is named by three
    # files and none of them were updated, so all three break at collection --
    # including `triton.py`, which is reached only through an import inside an
    # `if` and a second inside a function body, and is exactly the importer a
    # human doing the rename would miss.
    self.remove('tokamax/_src/core.py')
    self.add('tokamax/_src/core_renamed.py', 'import jax\n')
    tests, reason = self.affected(
        'tokamax/_src/core.py', 'tokamax/_src/core_renamed.py'
    )
    self.assertIsNone(reason)
    self.assertIn('tokamax/_src/core_test.py', tests)
    self.assertIn('tokamax/_src/op_test.py', tests)
    self.assertIn('tokamax/_src/ops/backend_test.py', tests)
    self.assertIn('tokamax/_src/ops/relative_test.py', tests)


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


class AutotuningTableTest(DepsTestCase):

  TABLE = 'tokamax/data/autotuning/tpu7x/pallas_mosaic_tpu_ragged_dot.json'

  def assert_full_run(self, path, *fragments):
    tests, reason = self.affected(path)
    self.assertIsNone(tests, f'{path} should force a full run')
    for fragment in fragments:
      self.assertIn(fragment, reason)

  def test_a_table_reaches_only_its_ops_tests(self):
    tests, reason = self.affected(self.TABLE)
    self.assertIsNone(reason)
    self.assertEqual(tests, {'tokamax/_src/ops/ragged_test.py'})

  def test_a_table_is_matched_by_class_name_not_module_name(self):
    self.assert_full_run(
        'tokamax/data/autotuning/tpu7x/ragged.json', 'ragged.json', "'ragged'"
    )

  def test_a_table_for_no_op_runs_everything(self):
    self.add(
        'tokamax/ops.py',
        'from tokamax._src.ops.ragged import PallasMosaicTpuRaggedDot as'
        ' PallasTritonRaggedDot\n',
    )
    self.assert_full_run(
        'tokamax/data/autotuning/tpu7x/pallas_triton_ragged_dot.json',
        "'pallas_triton_ragged_dot'",
        'no class',
    )

  def test_other_package_data_still_runs_everything(self):
    for path in (
        'tokamax/benchmarks/benchmark_registry.pbtxt',
        'tokamax/data/autotuning/pallas_mosaic_tpu_ragged_dot.json',
        'tokamax/data/autotuning/tpu7x/v2/pallas_mosaic_tpu_ragged_dot.json',
        'tokamax/data/autotuning/tpu7x/pallas_mosaic_tpu_ragged_dot.txt',
    ):
      with self.subTest(path=path):
        self.assert_full_run(path, os.path.basename(path), 'package data')

  def test_a_deleted_table_selects_the_same_tests_as_an_edited_one(self):
    self.remove(self.TABLE)
    tests, reason = self.affected(self.TABLE)
    self.assertIsNone(reason)
    self.assertEqual(tests, {'tokamax/_src/ops/ragged_test.py'})

  def test_a_class_name_defined_twice_reaches_both(self):
    self.add(
        'tokamax/_src/ops/ragged_copy.py',
        'class PallasMosaicTpuRaggedDot(Op):\n  pass\n',
    )
    self.add(
        'tokamax/_src/ops/ragged_copy_test.py',
        'from tokamax._src.ops import ragged_copy\n',
    )
    tests, reason = self.affected(self.TABLE)
    self.assertIsNone(reason)
    self.assertEqual(
        tests,
        {
            'tokamax/_src/ops/ragged_test.py',
            'tokamax/_src/ops/ragged_copy_test.py',
        },
    )

  def test_a_table_is_walked_through_the_graph_like_its_module(self):
    self.add('tokamax/_src/wrapper.py', 'from tokamax._src.ops import ragged\n')
    self.add(
        'tokamax/_src/wrapper_test.py', 'from tokamax._src import wrapper\n'
    )
    tests, _ = self.affected(self.TABLE)
    self.assertIn('tokamax/_src/wrapper_test.py', tests)

  def test_a_table_does_not_exempt_the_rest_of_the_changeset(self):
    tests, reason = self.affected(self.TABLE, 'tokamax/data/tuning.json')
    self.assertIsNone(tests)
    self.assertIn('tuning.json', reason)

  def test_without_the_op_index_a_table_is_ordinary_package_data(self):
    reason = deps.full_run_reason([self.TABLE], self.unparsed, self.files)
    self.assertIn('package data', reason)

  def test_an_unparseable_file_still_comes_first(self):
    self.add('tokamax/_src/broken.py', UNPARSEABLE)
    tests, reason = self.affected(self.TABLE)
    self.assertIsNone(tests)
    self.assertIn('broken.py', reason)

  def test_a_table_also_selects_the_cache_test(self):
    self.add(deps.AUTOTUNING_CACHE_TEST, 'from tokamax._src import core\n')
    tests, reason = self.affected(self.TABLE)
    self.assertIsNone(reason)
    self.assertEqual(
        tests,
        {'tokamax/_src/ops/ragged_test.py', deps.AUTOTUNING_CACHE_TEST},
    )

  def test_a_table_whose_op_reaches_no_test_runs_everything(self):
    self.add(deps.AUTOTUNING_CACHE_TEST, 'from tokamax._src import core\n')
    self.remove('tokamax/_src/ops/ragged_test.py')
    self.assert_full_run(self.TABLE, 'reaches no test')

  def test_non_op_classes_are_not_indexed(self):
    self.remove('tokamax/_src/ops/ragged.py')
    self.add(
        'tokamax/_src/core.py',
        'class PallasMosaicTpuRaggedDot(Op):\n  pass\n',
    )
    self.add(
        'tokamax/_src/ops/ragged_test.py',
        'class PallasMosaicTpuRaggedDot(Op):\n  pass\n',
    )
    self.add(
        'tokamax/_src/ops/config.py',
        'class PallasMosaicTpuRaggedDot:\n  pass\n',
    )
    self.assert_full_run(self.TABLE, 'no class')


class OpsListTest(DepsTestCase):

  def setUp(self):
    super().setUp()
    # Like the real test, it walks the ops tree rather than importing the ops.
    self.add(deps.OPS_LIST_TEST, 'import pathlib\n')

  def test_a_new_op_selects_the_ops_list_test(self):
    self.add('tokamax/_src/ops/new_op/base.py', 'import jax\n')
    tests, reason = self.affected('tokamax/_src/ops/new_op/base.py')
    self.assertIsNone(reason)
    self.assertEqual(tests, {deps.OPS_LIST_TEST})

  def test_an_op_edit_also_selects_the_ops_list_test(self):
    tests, reason = self.affected('tokamax/_src/ops/ragged.py')
    self.assertIsNone(reason)
    self.assertEqual(
        tests, {'tokamax/_src/ops/ragged_test.py', deps.OPS_LIST_TEST}
    )

  def test_a_deleted_op_file_selects_the_ops_list_test(self):
    self.remove('tokamax/_src/ops/triton.py')
    tests, _ = self.affected('tokamax/_src/ops/triton.py')
    self.assertIn(deps.OPS_LIST_TEST, tests)

  def test_a_public_ops_export_change_selects_the_ops_list_test(self):
    self.add('tokamax/ops/experimental/tpu.py', 'import jax\n')
    tests, _ = self.affected('tokamax/ops/experimental/tpu.py')
    self.assertIn(deps.OPS_LIST_TEST, tests)

  def test_a_doc_under_ops_does_not_select_it(self):
    tests, reason = self.affected('tokamax/_src/ops/ragged/README.md')
    self.assertIsNone(reason)
    self.assertEqual(tests, set())

  def test_a_change_outside_ops_does_not_select_it(self):
    self.add('tokamax/_src/helper.py', 'import jax\n')
    tests, reason = self.affected('tokamax/_src/helper.py')
    self.assertIsNone(reason)
    self.assertNotIn(deps.OPS_LIST_TEST, tests)


class CiBuildWorkflowTest(unittest.TestCase):
  """Keeps `deps.CI_BUILD_WORKFLOW` the only workflow CI-Build depends on."""

  def test_ci_build_calls_no_local_workflow_or_action(self):
    # Other workflows are inert only because CI-Build never runs them. A local
    # `uses: ./...` -- a reusable workflow or a composite action -- would make
    # one part of CI-Build, and a change to it would have to run every shard.
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    path = os.path.join(root, deps.CI_BUILD_WORKFLOW)
    with open(path, encoding='utf-8') as f:
      local = re.findall(r'^\s*(?:-\s*)?uses:\s*[\'"]?\./.*$', f.read(), re.M)
    self.assertEqual(local, [], f'{deps.CI_BUILD_WORKFLOW} calls local files')


class LoaderAgreementTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    cls.pkg = os.path.join(root, deps.PACKAGE_ROOT)
    if not os.path.isdir(cls.pkg):
      cls.pkg = root
    path = os.path.join(cls.pkg, '_src', 'autotuning', 'cache.py')
    with open(path, encoding='utf-8') as f:
      cls.tree = ast.parse(f.read(), filename=path)

  def constant(self, name):
    for node in ast.walk(self.tree):
      if (
          isinstance(node, ast.AnnAssign)
          and isinstance(node.target, ast.Name)
          and node.target.id == name
      ):
        return node.value
    self.fail(f'cache.py no longer defines {name}')

  def test_the_table_directory(self):
    self.assertEqual(
        f'{deps.PACKAGE_ROOT}/{ast.literal_eval(self.constant("CACHE_PATH"))}',
        deps.AUTOTUNING_DATA,
    )

  def test_the_cache_test_exists(self):
    rel = deps.AUTOTUNING_CACHE_TEST.removeprefix(f'{deps.PACKAGE_ROOT}/')
    self.assertTrue(os.path.isfile(os.path.join(self.pkg, rel)))

  def test_the_ops_list_test_exists(self):
    rel = deps.OPS_LIST_TEST.removeprefix(f'{deps.PACKAGE_ROOT}/')
    self.assertTrue(os.path.isfile(os.path.join(self.pkg, rel)))

  def test_every_resolved_table_reaches_an_op_test(self):
    cwd = os.getcwd()
    self.addCleanup(os.chdir, cwd)
    os.chdir(os.path.dirname(self.pkg))
    files = deps.python_files()
    edges, unparsed = deps.import_graph(files)
    self.assertFalse(unparsed)
    index = deps.dependents(edges, files)
    ops = deps.op_modules(files)
    tables = sorted(
        os.path.join(d, f).replace(os.sep, '/')
        for d, _, fs in os.walk(deps.AUTOTUNING_DATA)
        for f in fs
    )
    self.assertTrue(tables)
    for table in tables:
      name = deps.autotuning_op_name(table)
      self.assertIsNotNone(name, table)
      if modules := ops.get(name):
        op_tests = set().union(*(index.get(m, set()) for m in modules))
        self.assertTrue(op_tests, f'{table} reaches no op test')

  def test_the_class_name_to_op_name_pattern(self):
    call = self.constant('_CAMEL_TO_SNAKE_PATTERN')
    self.assertEqual(
        ast.literal_eval(call.args[0]), deps._CAMEL_TO_SNAKE.pattern
    )

  def test_the_op_name_is_taken_from_the_op_class_name(self):
    subs = [
        (ast.literal_eval(node.args[0]), ast.unparse(node.args[1]))
        for node in ast.walk(self.tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == '_CAMEL_TO_SNAKE_PATTERN'
        and node.func.attr == 'sub'
    ]
    self.assertIn((r'_\1', 'type(self.op).__name__'), subs)

  def test_the_table_path_is_device_then_op(self):
    joins = [
        ', '.join(ast.unparse(arg) for arg in node.args[1:])
        for node in ast.walk(self.tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, 'attr', None) == 'joinpath'
    ]
    self.assertIn("device_kind, f'{op_name}.json'", joins)


class ClosureInvariantTest(DepsTestCase):
  """The property the whole module exists for, checked by brute force.

  `affected_tests` walks forward from each test and inverts. This recomputes
  the same answer by fixpoint relaxation over the edge set -- a different
  algorithm, so agreement is evidence rather than a restatement.
  """

  def naive_reach(self, edges):
    reach = {f: set(edges.get(f, ())) for f in self.files}
    changed = True
    while changed:
      changed = False
      for f in self.files:
        grown = set(reach[f]).union(*(reach.get(g, set()) for g in reach[f]))
        if grown != reach[f]:
          reach[f] = grown
          changed = True
    return reach

  def naive_execution_edges(self):
    # Every package `__init__.py` whose directory encloses the file, computed
    # by prefix rather than by walking up as `enclosing_inits` does.
    inits = [f for f in self.files if os.path.basename(f) == '__init__.py']
    edges = {f: set(self.edges.get(f, ())) for f in self.files}
    for f in self.files:
      for init in inits:
        package = os.path.dirname(init) + '/'
        if f != init and f.startswith(package):
          edges[f].add(init)
    return edges

  def test_no_affected_test_is_ever_missed(self):
    # A changed `__init__.py` is selected over the graph with the enclosing
    # package edges added; everything else over the plain graph. A nested
    # package that imports a sibling gives the walk more than one level.
    self.add(
        'tokamax/_src/ops/kern/__init__.py',
        'from tokamax._src.ops import triton\n',
    )
    reach = self.naive_reach(self.edges)
    run_by = self.naive_reach(self.naive_execution_edges())
    tests = [f for f in self.files if deps.is_test_file(f)]
    for source in self.files:
      if source in self.unparsed:
        continue
      closure = run_by if os.path.basename(source) == '__init__.py' else reach
      expected = {t for t in tests if source in closure[t] or t == source}
      actual, reason = self.affected(source)
      if actual is None:
        continue  # fail-open is always allowed to be broader
      self.assertEqual(
          actual, expected, f'wrong affected set for {source} ({reason})'
      )


if __name__ == '__main__':
  unittest.main()
