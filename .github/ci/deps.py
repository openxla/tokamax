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
"""Which tests a change can affect, from the import graph.

`.github/ci/shards.py` decides what the shards *are*; this decides which of
them a pull request has to run. A test has to run if it imports a changed
module, directly or transitively -- so this builds the import graph of the
`tokamax` package with `ast` and runs the corresponding tests.

What the graph cannot see is handled by triggering a full run.

Usage:
  python3 .github/ci/deps.py tokamax/_src/pallas/block.py  # explain a change
  python3 .github/ci/deps.py --graph                       # fan-in summary
"""

import argparse
import ast
import collections
from collections.abc import Collection, Mapping, Sequence
import os
import sys
from typing import Final

# A file and the package files it imports directly. Keyed and valued by
# repo-relative paths.
type ImportEdges = Mapping[str, set[str]]

PACKAGE_ROOT: Final[str] = 'tokamax'

# Files that select every shard when changed, because no import edge would
# reveal what they affect.
#
#   conftest.py    a session-scoped autouse fixture that sets JAX and XLA
#                  flags for the whole suite. pytest loads it by location, so
#                  nothing imports it and it has no dependents to find.
#   data/**.json   the autotuning cache, read at runtime through
#                  `resources.files("tokamax")` in `_src/autotuning/cache.py`.
#                  It decides which kernel configuration a test actually runs,
#                  so it is a behavioural input with no import edge at all.
#
# Both are named by the rules in `full_run_reason` rather than listed here;
# this comment is what they are.

# `all_test_files` in shards.py matches pytest's `python_files`; the same
# universe, kept in one place there and mirrored here rather than imported, so
# `deps.py` stays usable on its own.
TEST_FILE_SUFFIX: Final[str] = '_test.py'
TEST_FILE_PREFIX: Final[str] = 'test_'

# Non-Python files that cannot change what a test does, and so do not force a
# full run.
INERT_SUFFIXES: Final[tuple[str, ...]] = ('.md', '.rst', '.pdf')


def python_files(root: str = PACKAGE_ROOT) -> list[str]:
  """Every `.py` file in the package, repo-relative and slash-separated.

  Args:
    root: Directory to walk. Repo-relative, and the returned paths keep that
      prefix, so they compare directly against `git diff --name-only` output.

  Returns:
    Sorted paths, `/`-separated on every platform. `__pycache__` is skipped:
    a stale `.pyc` beside a deleted source would otherwise become a node.
  """
  return sorted(
      os.path.join(dirpath, name).replace(os.sep, '/')
      for dirpath, _, filenames in os.walk(root)
      for name in filenames
      if name.endswith('.py') and '__pycache__' not in dirpath
  )


def is_test_file(path: str) -> bool:
  """Whether pytest would collect `path` as a test module.

  Args:
    path: Repo-relative path to a Python file.

  Returns:
    True if the basename matches `TEST_FILE_SUFFIX` or `TEST_FILE_PREFIX`,
    the same pair pytest's `python_files` setting uses.
  """
  name = os.path.basename(path)
  return name.endswith(TEST_FILE_SUFFIX) or name.startswith(TEST_FILE_PREFIX)


def module_name(path: str) -> str:
  """`tokamax/_src/ops/op.py` -> `tokamax._src.ops.op`.

  A package's `__init__.py` is the package itself, so that importing
  `tokamax._src.ops` and importing `tokamax._src.ops.op` both resolve to a
  file: the first to the `__init__`, the second to the module. Importing the
  second executes the first, which is why `imported_modules` records both.

  Args:
    path: Repo-relative path to a `.py` file under the package root.

  Returns:
    The dotted module name, with a trailing `.__init__` stripped so that a
    package and its `__init__.py` are one node in the graph rather than two.
  """
  mod = path[: -len('.py')].replace('/', '.')
  return mod.removesuffix('.__init__') if mod.endswith('.__init__') else mod


def imported_modules(path: str) -> set[str] | None:
  """Absolute module names `path` imports, or `None` if it does not parse.

  Names are emitted without checking that they exist. `import_graph` keeps
  only the ones that resolve to a file in the package, so third-party and
  stdlib imports fall away there rather than needing a list here.

  Args:
    path: Repo-relative path to a `.py` file.

  Returns:
    Absolute module names, including ones that resolve to nothing --
    `import_graph` is what discards those. `None` if the file did not parse,
    which is distinct from an empty set: empty means it imports nothing.
  """
  try:
    with open(path, encoding='utf-8') as f:
      tree = ast.parse(f.read(), filename=path)
  except (SyntaxError, ValueError, UnicodeDecodeError):
    return None

  own = module_name(path)
  package = own if path.endswith('__init__.py') else own.rpartition('.')[0]

  modules = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      modules.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      if node.level:
        parts = package.split('.')
        base = '.'.join(parts[: len(parts) - node.level + 1])
        target = f'{base}.{node.module}' if node.module else base
      else:
        target = node.module or ''
      # `from a.b import c` reaches `a.b`, and `a.b.c` if that is a module.
      modules.add(target)
      modules.update(f'{target}.{alias.name}' for alias in node.names)
  return modules


def import_graph(
    files: Sequence[str] | None = None,
) -> tuple[dict[str, set[str]], set[str]]:
  """Builds the forward import graph of the package.

  Args:
    files: Files to parse. Defaults to every `.py` file in the package.

  Returns:
    `(edges, unparsed)`. `edges` maps each file to the package files it
    imports directly; imports that do not resolve to a file in the package are
    dropped, which is how third-party and stdlib names fall away. `unparsed`
    is the files `ast` could not read, which `full_run_reason` turns into a
    full run when one of them is itself in the changeset.
  """
  files = python_files() if files is None else files
  by_module = {module_name(f): f for f in files}

  edges = collections.defaultdict(set)
  unparsed = set()
  for path in files:
    modules = imported_modules(path)
    if modules is None:
      unparsed.add(path)
      continue
    edges[path] = {
        by_module[m] for m in modules if m in by_module and by_module[m] != path
    }
  return edges, unparsed


def _reachable(start: str, edges: ImportEdges) -> set[str]:
  """Every file `start` imports, transitively.

  Args:
    start: File to walk out from.
    edges: Forward import graph, as built by `import_graph`.

  Returns:
    The files reachable from `start`, not including `start` itself unless a
    cycle leads back to it. Cycle-safe: each file is pushed at most once.
  """
  seen, stack = set(), [start]
  while stack:
    for nxt in edges.get(stack.pop(), ()):
      if nxt not in seen:
        seen.add(nxt)
        stack.append(nxt)
  return seen


def dependents(
    edges: ImportEdges, files: Sequence[str] | None = None
) -> dict[str, set[str]]:
  """Inverts the graph: file -> the test files that transitively import it.

  Args:
    edges: Forward import graph, as built by `import_graph`.
    files: Files to index. Defaults to every `.py` file in the package.

  Returns:
    A mapping from file to the test files that reach it. A test file maps to
    itself as well: editing a test has to run it. Files no test reaches are
    absent rather than mapped to an empty set.
  """
  files = python_files() if files is None else files
  index = collections.defaultdict(set)
  for test in (f for f in files if is_test_file(f)):
    # A test depends on itself: editing it must run it.
    for dep in _reachable(test, edges) | {test}:
      index[dep].add(test)
  return index


def full_run_reason(
    changed: Collection[str],
    unparsed: Collection[str] = (),
    known: Collection[str] | None = None,
) -> str | None:
  """Why `changed` requires every shard, or `None` if it does not.

  The exception is `INERT_SUFFIXES`, checked before anything else: those files
  cannot affect a test wherever they live (think documentation).

  `known` is the file set the graph was built from. This handles the case of 
  a file being deleted, which has no import edges and so would be ignored
  otherwise.

  `unparsed` is not checked against `changed`: it is a property of the graph,
  and forces a full run if the collection is non-empty

  Args:
    changed: Repo-relative paths of the changed files.
    unparsed: Files `ast` could not read, from `import_graph`. Any at all makes
      the whole graph untrustworthy.
    known: The file set the graph was built from. `None` switches the
      deleted-file rule off, which is only safe when the caller already knows
      every changed path still exists.

  Returns:
    A human-readable reason to run every shard, or `None` if the changeset can
    be narrowed. The string is what the workflow prints, so it names the file
    that forced the decision.
  """
  unparsed = set(unparsed)
  known = None if known is None else set(known)

  if unparsed:
    first, *rest = sorted(unparsed)
    more = f' and {len(rest)} other files' if rest else ''
    return (
        f'{first}{more} did not parse under python'
        f' {sys.version.split()[0]}, so the import graph is incomplete'
    )

  for path in sorted(changed):
    if path.endswith(INERT_SUFFIXES):
      continue
    if not path.startswith(f'{PACKAGE_ROOT}/'):
      return f'{path} is outside {PACKAGE_ROOT}/'
    if os.path.basename(path) == 'conftest.py':
      return f'{path} configures every test in the package'
    if not path.endswith('.py'):
      return f'{path} is package data, which no import edge covers'
    if known is not None and path not in known:
      return f'{path} was deleted or renamed, so its dependents are unknown'
  return None


def affected_tests(
    changed: Collection[str],
    edges: ImportEdges | None = None,
    unparsed: Collection[str] | None = None,
    files: Sequence[str] | None = None,
) -> tuple[set[str] | None, str | None]:
  """Test files a change can affect, or `None` meaning "run everything".

  Any `None` outputs will be documented with a `full_run_reason`.

  Args:
    changed: Repo-relative paths of the changed files. Iterated more than once,
      so a generator will not do.
    edges: Forward import graph. Built from `files` if omitted.
    unparsed: Files that did not parse. Built from `files` if omitted --
      independently of `edges`, so that supplying one without the other cannot
      quietly cost the fail-open rule.
    files: Files the graph covers. Defaults to every `.py` file in the package.

  Returns:
    `(tests, reason)`. `tests` is the affected test files, or `None` when
    every shard has to run, in which case `reason` says why. Exactly one of
    the two is ever `None`.
  """
  files = python_files() if files is None else files
  if edges is None or unparsed is None:
    built_edges, built_unparsed = import_graph(files)
    edges = built_edges if edges is None else edges
    unparsed = built_unparsed if unparsed is None else unparsed
  if reason := full_run_reason(changed, unparsed, files):
    return None, reason

  index = dependents(edges, files)
  tests = set()
  for path in changed:
    tests |= index.get(path, set())
  return tests, None


def main(argv: Sequence[str] | None = None) -> int:
  """Command-line entry point.

  Args:
    argv: Argument list. Defaults to `sys.argv[1:]`.

  Returns:
    A process exit code, always 0. This is a reporting tool: the full runs and
    coverage gaps it describes are findings, not failures of its own.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('changed', nargs='*', help='changed file paths')
  parser.add_argument(
      '--graph', action='store_true', help='summarise fan-in instead'
  )
  args = parser.parse_args(argv)

  files = python_files()
  edges, unparsed = import_graph(files)

  if args.graph:
    index = dependents(edges, files)
    tests = [f for f in files if is_test_file(f)]
    ranked = sorted(index.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    print(
        f'{len(files)} files, {sum(len(v) for v in edges.values())} edges,'
        f' {len(tests)} test files, parsed by python'
        f' {sys.version.split()[0]}'
    )
    if unparsed:
      # Expected to be empty - will cause full runs until fixed.
      print(f'unparsed ({len(unparsed)}, forcing a full run for any change):')
      for path in sorted(unparsed):
        print(f'  {path}')
    print('highest fan-in:')
    for path, ts in ranked[:10]:
      print(f'  {len(ts):3d}/{len(tests)}  {path}')
    untested = sorted(
        f for f in files if not is_test_file(f) and not index.get(f)
    )
    print(f'reached by no test ({len(untested)}): changing one runs nothing')
    for path in untested:
      print(f'  {path}')
    return 0

  if not args.changed:
    parser.error('give at least one changed path, or --graph')

  tests, reason = affected_tests(args.changed, edges, unparsed, files)
  if tests is None:
    print(f'full run: {reason}')
    return 0
  print(f'{len(tests)} affected test files')
  for path in sorted(tests):
    print(f'  {path}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
