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
"""Selects affected test files from the package import graph.

Builds a static AST import graph of the `tokamax` package to identify test
files that directly or transitively import changed modules. Changes that cannot
be safely narrowed trigger a full test run.

Usage:
  python3 .github/ci/deps.py <changed_path>...
  python3 .github/ci/deps.py --graph
"""

import argparse
import ast
import collections
from collections.abc import Collection, Mapping, Sequence
import os
import re
import sys
from typing import Final

# Forward import edges: file path -> directly imported package file paths.
type ImportEdges = Mapping[str, set[str]]

PACKAGE_ROOT: Final[str] = 'tokamax'

# Changes that cannot be traced via import edges (conftest.py, package data,
# deleted __init__.py) force a full test run via full_run_reason().

# Matches pytest's default python_files naming patterns.
TEST_FILE_SUFFIX: Final[str] = '_test.py'
TEST_FILE_PREFIX: Final[str] = 'test_'

# Non-Python files that do not affect test execution.
INERT_SUFFIXES: Final[tuple[str, ...]] = ('.md', '.rst', '.pdf', '.ipynb')

# Workflows other than ci-build.yml are inert: CI-Build never calls them, and
# `deps_test.py` checks that it calls no local workflow or action.
WORKFLOWS_DIR: Final[str] = '.github/workflows/'
CI_BUILD_WORKFLOW: Final[str] = f'{WORKFLOWS_DIR}ci-build.yml'

# `_src/autotuning/cache.py` reads `<device_kind>/<op_name>.json`, where
# `<op_name>` is the op's snake-cased class name. Selection narrows by op only,
# not by device.
AUTOTUNING_DATA: Final[str] = f'{PACKAGE_ROOT}/data/autotuning'
AUTOTUNING_CACHE_TEST: Final[str] = (
    f'{PACKAGE_ROOT}/_src/autotuning/cache_test.py'
)
_OPS_ROOT: Final[str] = f'{PACKAGE_ROOT}/_src/ops/'
_CAMEL_TO_SNAKE: Final[re.Pattern[str]] = re.compile(r'(?!^)([A-Z])')


def python_files(root: str = PACKAGE_ROOT) -> list[str]:
  """Returns sorted repo-relative paths of all `.py` files under `root`.

  Args:
    root: Directory to walk, defaulting to PACKAGE_ROOT.

  Returns:
    Sorted repo-relative paths with `/` separators.
  """
  found = []
  for dirpath, _, filenames in os.walk(root):
    if '__pycache__' in dirpath:
      continue
    for name in filenames:
      if name.endswith('.py'):
        found.append(os.path.join(dirpath, name).replace(os.sep, '/'))
  return sorted(found)


def is_test_file(path: str) -> bool:
  """Returns True if `path` matches pytest test file naming conventions.

  Args:
    path: Path to a Python file.

  Returns:
    True if the filename matches pytest test file patterns.
  """
  name = os.path.basename(path)
  return name.endswith(TEST_FILE_SUFFIX) or name.startswith(TEST_FILE_PREFIX)


def _is_package_init(path: str) -> bool:
  """Returns True if `path` is an __init__.py file."""
  return os.path.basename(path) == '__init__.py'


def module_name(path: str) -> str:
  """`tokamax/_src/ops/op.py` -> `tokamax._src.ops.op`.

  A package's `__init__.py` maps to the package name, so the package and its
  `__init__.py` are one node in the graph.

  Args:
    path: Repo-relative path to a `.py` file under the package root.

  Returns:
    The dotted module name, with any trailing `.__init__` stripped.
  """
  mod = path[: -len('.py')].replace('/', '.')
  return mod.removesuffix('.__init__') if mod.endswith('.__init__') else mod


def _parse(path: str) -> ast.Module | None:
  """Parses `path` with `ast`, or returns None on error."""
  try:
    with open(path, encoding='utf-8') as f:
      return ast.parse(f.read(), filename=path)
  except (SyntaxError, ValueError, UnicodeDecodeError):
    return None


def imported_modules(path: str) -> set[str] | None:
  """Returns absolute module names `path` imports, or None if unparseable.

  Args:
    path: Repo-relative path to a `.py` file.

  Returns:
    Absolute module names imported by `path`, or None if parsing fails.
  """
  if (tree := _parse(path)) is None:
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
      # 'from a.b import c' may import package a.b or submodule a.b.c.
      modules.add(target)
      modules.update(f'{target}.{alias.name}' for alias in node.names)
    elif (
        path.endswith('__init__.py')
        and isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets
        )
        and isinstance(node.value, (ast.Tuple, ast.List))
    ):
      modules.update(
          f'{package}.{elt.value}'
          for elt in node.value.elts
          if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
      )
  return modules


def import_graph(
    files: Sequence[str] | None = None,
) -> tuple[dict[str, set[str]], set[str]]:
  """Builds the forward import graph of the package.

  Args:
    files: Files to parse. Defaults to all package `.py` files.

  Returns:
    A tuple of (edges, unparsed), where edges maps each file to directly
    imported package files, and unparsed contains unparseable files.
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


def module_referrers(
    files: Sequence[str] | None = None,
) -> dict[str, set[str]]:
  """Maps each imported module name to the package files that import it.

  Unlike `import_graph`, this keeps names that resolve to no file, so
  `affected_tests` can find files that still import a deleted or renamed module.

  Args:
    files: Files to parse. Defaults to all package `.py` files.

  Returns:
    Mapping from imported module name to files that import it.
  """
  files = python_files() if files is None else files
  index = collections.defaultdict(set)
  for path in files:
    for module in imported_modules(path) or ():
      index[module].add(path)
  return index


def autotuning_op_name(path: str) -> str | None:
  """`tokamax/data/autotuning/tpu7x/ragged_dot.json` -> `ragged_dot`."""
  rest = path.removeprefix(f'{AUTOTUNING_DATA}/')
  if rest == path:
    return None
  device, _, name = rest.partition('/')
  if not device or '/' in name or not name.endswith('.json'):
    return None
  return name.removesuffix('.json') or None


def op_name(class_name: str) -> str:
  """`PallasMosaicTpuRaggedDot` -> `pallas_mosaic_tpu_ragged_dot`."""
  return _CAMEL_TO_SNAKE.sub(r'_\1', class_name).lower()


def op_modules(files: Sequence[str] | None = None) -> dict[str, set[str]]:
  """Maps snake_case op class names to the `_src/ops/` files defining them."""
  files = python_files() if files is None else files
  index = collections.defaultdict(set)
  for path in files:
    if (
        not path.startswith(_OPS_ROOT)
        or is_test_file(path)
        or (tree := _parse(path)) is None
    ):
      continue
    for node in ast.walk(tree):
      if isinstance(node, ast.ClassDef) and node.bases:
        index[op_name(node.name)].add(path)
  return index


def _reachable(start: str, edges: ImportEdges) -> set[str]:
  """Returns all package files transitively imported by `start`.

  Args:
    start: File path to walk from.
    edges: Forward import graph, as built by `import_graph`.

  Returns:
    All package files reachable from `start`.
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
  """Inverts the import graph to map each file to test files reaching it.

  Test files map to themselves so editing a test runs it.

  Args:
    edges: Forward import graph, as built by `import_graph`.
    files: Files to index. Defaults to all package `.py` files.

  Returns:
    Mapping from each file to test files that transitively import it.
  """
  files = python_files() if files is None else files
  index = collections.defaultdict(set)
  for test in (f for f in files if is_test_file(f)):
    for dep in _reachable(test, edges) | {test}:
      index[dep].add(test)
  return index


def enclosing_inits(path: str, known: Collection[str]) -> set[str]:
  """Returns enclosing package `__init__.py` files present in `known`.

  Importing a submodule executes parent package `__init__.py` files first.

  Args:
    path: Repo-relative path to a `.py` file.
    known: Collection of known existing file paths.

  Returns:
    Enclosing package `__init__.py` files in `known`, up to the package root.
  """
  directory = os.path.dirname(path)
  if _is_package_init(path):
    directory = os.path.dirname(directory)
  inits = set()
  while directory:
    if (init := f'{directory}/__init__.py') in known:
      inits.add(init)
    directory = os.path.dirname(directory)
  return inits


def execution_edges(
    edges: ImportEdges, files: Sequence[str] | None = None
) -> dict[str, set[str]]:
  """Returns `edges` plus edges from each file to its enclosing `__init__.py`s.

  Used only when an `__init__.py` changes, to find tests that execute it without
  over-selecting tests on ordinary edits.

  Args:
    edges: Forward import graph, as built by `import_graph`.
    files: Files to cover. Defaults to all package `.py` files.

  Returns:
    A new forward import graph including edges to enclosing `__init__.py` files.
  """
  files = python_files() if files is None else files
  known = set(files)
  return {
      path: set(edges.get(path, ())) | enclosing_inits(path, known)
      for path in files
  }


def full_run_reason(
    changed: Collection[str],
    unparsed: Collection[str] = (),
    known: Collection[str] | None = None,
    ops: Mapping[str, Collection[str]] | None = None,
) -> str | None:
  """Returns why `changed` requires running every shard, or None.

  Files with an `INERT_SUFFIXES` suffix and workflows other than
  `CI_BUILD_WORKFLOW` never force a full run. Any unparsed file does, whether
  or not it changed. Of the deleted `.py` files (changed paths missing from
  `known`), only an `__init__.py` forces one. With `known=None`, every changed
  `__init__.py` forces one, since a deletion cannot be ruled out.

  Args:
    changed: Repo-relative paths of changed files.
    unparsed: Files that failed AST parsing, from `import_graph`.
    known: Known package files, or None if deletions cannot be verified.
    ops: Mapping from op names to defining files, from `op_modules`.

  Returns:
    Reason string to run every shard, or None if tests can be narrowed.
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
    if path.startswith(WORKFLOWS_DIR) and path != CI_BUILD_WORKFLOW:
      continue
    if not path.startswith(f'{PACKAGE_ROOT}/'):
      return f'{path} is outside {PACKAGE_ROOT}/'
    if os.path.basename(path) == 'conftest.py':
      return f'{path} configures every test in the package'
    if ops is not None and (name := autotuning_op_name(path)):
      if not ops.get(name):
        return (
            f'{path} is the autotuning table for {name!r}, which no class'
            f' under {_OPS_ROOT} is named after, so no op is known to read it'
        )
      continue
    if not path.endswith('.py'):
      return f'{path} is package data, which no import edge covers'
    # Deleting __init__.py converts the package to a namespace package,
    # affecting how sibling imports resolve.
    if _is_package_init(path):
      if known is None:
        return (
            f'{path} is a package __init__.py, and without the file list it'
            ' cannot be told apart from a deleted one'
        )
      if path not in known:
        return f'{path} was deleted, which re-resolves the whole package'
  return None


def affected_tests(
    changed: Collection[str],
    edges: ImportEdges | None = None,
    unparsed: Collection[str] | None = None,
    files: Sequence[str] | None = None,
) -> tuple[set[str] | None, str | None]:
  """Returns `(tests, reason)` for `changed`; `tests` is None for a full run.

  Args:
    changed: Repo-relative paths of changed files.
    edges: Forward import graph. Built from `files` if omitted.
    unparsed: Files that failed AST parsing. Built from `files` if omitted.
    files: Files to cover. Defaults to all package `.py` files.

  Returns:
    A tuple of (tests, reason). `tests` is affected test files, or None for
    a full run with `reason` explaining why.
  """
  files = python_files() if files is None else files
  if edges is None or unparsed is None:
    built_edges, built_unparsed = import_graph(files)
    edges = built_edges if edges is None else edges
    unparsed = built_unparsed if unparsed is None else unparsed
  tables = {p: n for p in sorted(changed) if (n := autotuning_op_name(p))}
  ops = op_modules(files) if tables else {}
  if reason := full_run_reason(
      changed, unparsed, files, ops if tables else None
  ):
    return None, reason

  index = dependents(edges, files)
  known = set(files)
  tests = set()
  # Non-Python files that pass full_run_reason have no module name.
  deleted = [p for p in changed if p not in known and p.endswith('.py')]
  for path in changed:
    tests |= index.get(path, set())
  # Deleted __init__.py files trigger a full run above, so surviving ones exist.
  if inits := [p for p in changed if _is_package_init(p)]:
    run_by = dependents(execution_edges(edges, files), files)
    for path in inits:
      tests |= run_by.get(path, set())
  for path, name in tables.items():
    op_tests = set().union(*(index.get(m, set()) for m in ops[name]))
    if not op_tests:
      return (
          None,
          f'{path} is the autotuning table for {name!r}, which reaches no'
          ' test',
      )
    tests |= op_tests
  if tables:
    tests |= index.get(AUTOTUNING_CACHE_TEST, set())

  if deleted:
    # Surviving files that still import a deleted module, and their dependents.
    referrers = module_referrers(files)
    for path in deleted:
      for referrer in referrers.get(module_name(path), ()):
        tests |= index.get(referrer, set())
  return tests, None


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point for querying affected test files.

  Args:
    argv: Command-line arguments, defaulting to `sys.argv[1:]`.

  Returns:
    Process exit code, always 0.
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
