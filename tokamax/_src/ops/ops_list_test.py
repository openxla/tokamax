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
"""Tests for ops_list drift and consistency."""

import ast
import pathlib
from typing import Any, cast

from absl.testing import absltest
from tokamax._src.ops import ops_list


def _find_tokamax_root() -> pathlib.Path:
  real_path = pathlib.Path(__file__)
  for parent_dir in (real_path, *real_path.parents):
    if parent_dir.name == "tokamax" and (parent_dir / "_src" / "ops").is_dir():
      return parent_dir
  return pathlib.Path(__file__).parents[2]


_TOKAMAX_ROOT = _find_tokamax_root()
_NON_OP_ALLOWLIST = frozenset(("experimental", "__pycache__", "testdata"))


def _parse_source_file(file_path: pathlib.Path) -> ast.Module:
  with open(file_path, "r", encoding="utf-8") as source_file:
    return ast.parse(source_file.read(), filename=str(file_path))


def _op_subdirs_with_py_files(parent_dir: pathlib.Path) -> set[str]:
  subdirs: set[str] = set()
  for entry in parent_dir.iterdir():
    if not entry.is_dir() or entry.name in _NON_OP_ALLOWLIST:
      continue
    py_files = [
        candidate_file
        for candidate_file in entry.rglob("*.py")
        if candidate_file.name != "__init__.py"
        and "__pycache__" not in candidate_file.parts
    ]
    if py_files:
      subdirs.add(entry.name)
  return subdirs


def _all_op_dirs() -> list[str]:
  dirs: list[str] = []
  for op_entry in ops_list.OPS:
    dirs.extend(op_entry.dirs)
  return dirs


def _str_constants(nodes: list[ast.expr]) -> set[str]:
  return {
      node.value
      for node in nodes
      if isinstance(node, ast.Constant) and isinstance(node.value, str)
  }


def _assignment(node: ast.AST) -> tuple[list[ast.expr], ast.expr | None]:
  """Returns the (targets, value) of an assignment, or ([], None)."""
  if isinstance(node, ast.Assign):
    return node.targets, node.value
  if isinstance(node, ast.AnnAssign):
    return [node.target], node.value
  return [], None


def _extract_dunder_all(tree: ast.Module) -> frozenset[str]:
  for node in tree.body:
    targets, value = _assignment(node)
    if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
      continue
    if isinstance(value, (ast.Tuple, ast.List)):
      return frozenset(_str_constants(value.elts))
  return frozenset()


def _extract_op_api_imports(tree: ast.Module) -> frozenset[str]:
  names: set[str] = set()
  for node in ast.walk(tree):
    if (
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("tokamax._src.ops.")
        and node.module.endswith(".api")
    ):
      for alias in node.names:
        names.add(alias.asname or alias.name)
  return frozenset(names)


def _extract_explicit_reexports(tree: ast.Module) -> frozenset[str]:
  reexports: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, (ast.ImportFrom, ast.Import)):
      for alias in node.names:
        if alias.asname is not None:
          reexports.add(alias.asname)
  return frozenset(reexports)


_OPS_NON_ENTRYPOINT_EXPORTS = frozenset({
    "DotProductAttentionImplementation",
    "experimental",
    "gpu",
    "op_fusion",
    "tpu",
})
_EXP_NON_ENTRYPOINT_EXPORTS = frozenset({"tpu"})


def _extract_dict_literal_keys(expression: ast.AST) -> set[str]:
  keys: set[str] = set()
  if isinstance(expression, ast.Dict):
    for key_node in expression.keys:
      if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
        keys.add(key_node.value)
  elif isinstance(expression, ast.Call):
    for keyword in expression.keywords:
      if keyword.arg:
        keys.add(keyword.arg)
    for argument in expression.args:
      keys.update(_extract_dict_literal_keys(argument))
  return keys


def _is_impl_dict_name(name: str) -> bool:
  upper_name = name.upper()
  return (
      upper_name.endswith("IMPLEMENTATIONS")
      and not upper_name.startswith("DEFAULT_")
      and not upper_name.startswith("_DEFAULT_")
  )


def _impl_keys_from_target(
    target: ast.expr, value: ast.expr | None
) -> set[str]:
  """Keys set by `IMPLEMENTATIONS = {...}` or `IMPLEMENTATIONS["k"] = ...`."""
  if isinstance(target, ast.Name):
    if _is_impl_dict_name(target.id) and value is not None:
      return _extract_dict_literal_keys(value)
    return set()
  if (
      isinstance(target, ast.Subscript)
      and isinstance(target.value, ast.Name)
      and _is_impl_dict_name(target.value.id)
  ):
    return _str_constants([target.slice])
  return set()


def _implementation_literal_keys(tree: ast.Module) -> set[str]:
  """Keys from `type Implementation = Literal[...]`."""
  keys: set[str] = set()
  for node in tree.body:
    if not (
        isinstance(node, ast.TypeAlias)
        and node.name.id == "Implementation"
        and isinstance(node.value, ast.Subscript)
    ):
      continue
    slice_node = node.value.slice
    if isinstance(slice_node, ast.Tuple):
      keys |= _str_constants(slice_node.elts)
    else:
      keys |= _str_constants([slice_node])
  return keys


def _extract_implementation_keys(tree: ast.Module) -> frozenset[str]:
  keys: set[str] = set()
  for node in ast.walk(tree):
    targets, value = _assignment(node)
    for target in targets:
      keys |= _impl_keys_from_target(target, value)
  return frozenset(keys or _implementation_literal_keys(tree))


def _module_op_dir(module: str) -> str | None:
  for root in ("_src/ops/", "google/ops/"):
    if module.startswith(root):
      parts = module.removeprefix(root).split("/")
      if parts[0] == "experimental" and len(parts) > 1:
        return f"experimental/{parts[1]}"
      return parts[0]
  return None


class OpsListTest(absltest.TestCase):
  """Tests verifying ops_list manifest integrity and consistency."""

  def test_impl_modules_exist(self):
    google_dir_exists = (_TOKAMAX_ROOT / "google").is_dir()
    for op_entry in ops_list.OPS:
      if (_TOKAMAX_ROOT / f"_src/ops/{op_entry.op_dir}/base.py").is_file():
        self.assertEqual(
            op_entry.impls[0].backend,
            ops_list.XLA,
            f"Op '{op_entry.name}' has base.py but first impl is not XLA",
        )
      for impl in op_entry.impls:
        file_path = impl.module
        if file_path.startswith("google/") and not google_dir_exists:
          continue
        full_path = _TOKAMAX_ROOT / file_path
        self.assertTrue(
            full_path.is_file(),
            f"Path '{file_path}' for op '{op_entry.name}' impl"
            f" '{impl.name}' does not exist under tokamax"
            " root",
        )

  def test_api_symbols_exported(self):
    root_init_path = _TOKAMAX_ROOT / "__init__.py"
    ops_init_path = _TOKAMAX_ROOT / "ops" / "__init__.py"
    ops_exp_init_path = _TOKAMAX_ROOT / "ops" / "experimental" / "__init__.py"

    self.assertTrue(
        root_init_path.is_file(),
        f"Missing root __init__.py at {root_init_path}",
    )
    self.assertTrue(
        ops_init_path.is_file(),
        f"Missing ops __init__.py at {ops_init_path}",
    )
    self.assertTrue(
        ops_exp_init_path.is_file(),
        f"Missing ops/experimental __init__.py at {ops_exp_init_path}",
    )

    root_tree = _parse_source_file(root_init_path)
    ops_tree = _parse_source_file(ops_init_path)
    ops_exp_tree = _parse_source_file(ops_exp_init_path)

    root_reexports = _extract_explicit_reexports(root_tree)
    ops_all = _extract_dunder_all(ops_tree)
    ops_exp_all = _extract_dunder_all(ops_exp_tree)

    all_apis: list[str] = []
    for op_entry in ops_list.OPS:
      all_apis.extend(op_entry.apis)
    self.assertCountEqual(all_apis, set(all_apis))

    recorded_exp_symbols: set[str] = set()
    recorded_ops_symbols: set[str] = set()
    recorded_root_symbols: set[str] = set()

    for op_entry in ops_list.OPS:
      for api_path in op_entry.apis:
        if api_path.startswith("tokamax.ops.experimental."):
          symbol_name = api_path.removeprefix("tokamax.ops.experimental.")
          self.assertTrue(
              symbol_name,
              f"Empty symbol name in '{api_path}' on op '{op_entry.name}'",
          )
          self.assertNotIn(
              ".",
              symbol_name,
              f"Dotted symbol '{symbol_name}' in '{api_path}' on op"
              f" '{op_entry.name}'",
          )
          self.assertIn(
              symbol_name,
              ops_exp_all,
              f"Symbol '{symbol_name}' from api '{api_path}' on op"
              f" '{op_entry.name}' not exported in"
              " ops/experimental/__init__.py",
          )
          recorded_exp_symbols.add(symbol_name)
        elif api_path.startswith("tokamax.ops."):
          symbol_name = api_path.removeprefix("tokamax.ops.")
          self.assertTrue(
              symbol_name,
              f"Empty symbol name in '{api_path}' on op '{op_entry.name}'",
          )
          self.assertNotIn(
              ".",
              symbol_name,
              f"Dotted symbol '{symbol_name}' in '{api_path}' on op"
              f" '{op_entry.name}'",
          )
          self.assertIn(
              symbol_name,
              ops_all,
              f"Symbol '{symbol_name}' from api '{api_path}' on op"
              f" '{op_entry.name}' not exported in ops/__init__.py",
          )
          recorded_ops_symbols.add(symbol_name)
        elif api_path.startswith("tokamax."):
          symbol_name = api_path.removeprefix("tokamax.")
          self.assertTrue(
              symbol_name,
              f"Empty symbol name in '{api_path}' on op '{op_entry.name}'",
          )
          self.assertNotIn(
              ".",
              symbol_name,
              f"Dotted symbol '{symbol_name}' in '{api_path}' on op"
              f" '{op_entry.name}'",
          )
          self.assertIn(
              symbol_name,
              root_reexports,
              f"Symbol '{symbol_name}' from api '{api_path}' on op"
              f" '{op_entry.name}' not re-exported in root __init__.py",
          )
          self.assertIn(
              symbol_name,
              ops_all,
              f"Symbol '{symbol_name}' from api '{api_path}' on op"
              f" '{op_entry.name}' not exported in ops/__init__.py",
          )
          recorded_root_symbols.add(symbol_name)
        else:
          self.fail(f"Invalid api path '{api_path}' on op '{op_entry.name}'")

    expected_ops_entrypoints = (
        ops_all | _extract_op_api_imports(ops_tree)
    ) - _OPS_NON_ENTRYPOINT_EXPORTS
    self.assertEqual(
        recorded_ops_symbols | recorded_root_symbols, expected_ops_entrypoints
    )
    self.assertEqual(
        ops_all - _OPS_NON_ENTRYPOINT_EXPORTS,
        _extract_op_api_imports(ops_tree) - _OPS_NON_ENTRYPOINT_EXPORTS,
        "Symbols in ops/__init__.py __all__ and imports must match",
    )

    expected_exp_entrypoints = (
        ops_exp_all | _extract_op_api_imports(ops_exp_tree)
    ) - _EXP_NON_ENTRYPOINT_EXPORTS
    self.assertEqual(recorded_exp_symbols, expected_exp_entrypoints)
    self.assertEqual(
        ops_exp_all - _EXP_NON_ENTRYPOINT_EXPORTS,
        _extract_op_api_imports(ops_exp_tree) - _EXP_NON_ENTRYPOINT_EXPORTS,
        "Symbols in ops/experimental/__init__.py __all__ and imports must"
        " match",
    )

    expected_root_entrypoints = (
        _extract_op_api_imports(root_tree) - _OPS_NON_ENTRYPOINT_EXPORTS
    )
    self.assertEqual(recorded_root_symbols, expected_root_entrypoints)

  def test_api_implementations_keys_appear_in_impls(self):
    for op_entry in ops_list.OPS:
      api_rel_path = f"_src/ops/{op_entry.op_dir}/api.py"
      api_full_path = _TOKAMAX_ROOT / api_rel_path
      if op_entry.apis:
        self.assertTrue(
            api_full_path.is_file(),
            f"Op '{op_entry.name}' defines APIs {op_entry.apis} but lacks"
            f" {api_rel_path}",
        )
      if not api_full_path.is_file():
        continue

      all_names = frozenset(impl.name for impl in op_entry.impls)
      oss_names = frozenset(
          impl.name
          for impl in op_entry.impls
          if not impl.module.startswith("google/")
      )
      api_tree = _parse_source_file(api_full_path)
      api_keys = _extract_implementation_keys(api_tree)
      self.assertContainsSubset(
          oss_names,
          api_keys,
          f"OSS impl names not registered in api.py for op '{op_entry.name}'"
          f" in '{api_rel_path}'",
      )
      self.assertContainsSubset(
          api_keys,
          all_names,
          f"Declared impl names mismatch for op '{op_entry.name}' in"
          f" '{api_rel_path}'",
      )

  def test_op_directories_covered(self):
    ops_dir = _TOKAMAX_ROOT / "_src" / "ops"
    all_dirs = frozenset(_all_op_dirs())
    covered_standard = frozenset(
        d for d in all_dirs if not d.startswith("experimental/")
    )
    for name in _op_subdirs_with_py_files(ops_dir):
      self.assertIn(
          name,
          covered_standard,
          f"Directory _src/ops/{name} contains python files but is not covered"
          " by any standard Op in ops_list",
      )

    exp_ops_dir = ops_dir / "experimental"
    covered_experimental = frozenset(
        d.removeprefix("experimental/")
        for d in all_dirs
        if d.startswith("experimental/")
    )
    if exp_ops_dir.is_dir():
      for name in _op_subdirs_with_py_files(exp_ops_dir):
        self.assertIn(
            name,
            covered_experimental,
            f"Directory _src/ops/experimental/{name} contains python files but"
            " is not covered by any experimental Op in ops_list",
        )

    google_ops_dir = _TOKAMAX_ROOT / "google" / "ops"
    if google_ops_dir.is_dir():
      for name in _op_subdirs_with_py_files(google_ops_dir):
        self.assertIn(
            name,
            covered_standard,
            f"Directory google/ops/{name} contains python files but is not"
            " covered by any standard Op in ops_list",
        )

    exp_google_ops_dir = google_ops_dir / "experimental"
    if exp_google_ops_dir.is_dir():
      for name in _op_subdirs_with_py_files(exp_google_ops_dir):
        self.assertIn(
            name,
            covered_experimental,
            f"Directory google/ops/experimental/{name} contains python files"
            " but is not covered by any experimental Op in ops_list",
        )

  def test_names_are_non_empty_strings(self):
    for op_entry in ops_list.OPS:
      impl_names = [impl.name for impl in op_entry.impls]
      self.assertCountEqual(
          impl_names,
          set(impl_names),
          f"Impl names must be unique within op '{op_entry.name}'",
      )
      for name in (op_entry.name, *impl_names):
        self.assertIsInstance(name, str)
        self.assertTrue(name and name == name.strip())

    for bad_name in ("", cast(Any, None)):
      with self.subTest(bad_name=bad_name):
        with self.assertRaises(ValueError):
          ops_list.Backend(name=bad_name, platforms=frozenset())
        with self.assertRaises(ValueError):
          ops_list.Impl(
              name=bad_name,
              backend=ops_list.XLA,
              module="_src/ops/attention/base.py",
          )
        with self.assertRaises(ValueError):
          ops_list.Op(name=bad_name, apis=(), impls=())

  def test_unique_op_names_and_dirs(self):
    op_names = [op.name for op in ops_list.OPS]
    self.assertCountEqual(
        op_names,
        set(op_names),
        "Op names in ops_list.OPS must be unique",
    )
    op_dirs = _all_op_dirs()
    self.assertCountEqual(
        op_dirs,
        set(op_dirs),
        "Op directories in ops_list.OPS must be unique",
    )

  def test_kernel_dirs(self):
    for op in ops_list.OPS:
      for kernel_dir in op.dirs[1:]:
        self.assertRegex(
            kernel_dir,
            r"^(experimental/)?[a-z0-9_]+$",
            f"kernel_dir '{kernel_dir}' on op '{op.name}' must name a"
            " top-level op directory",
        )
        name = kernel_dir.rpartition("/")[2]
        self.assertTrue(
            any(
                (path := _TOKAMAX_ROOT / root / "ops" / kernel_dir).is_dir()
                and name in _op_subdirs_with_py_files(path.parent)
                for root in ("_src", "google")
            ),
            f"kernel_dir '{kernel_dir}' on op '{op.name}' has no python files",
        )

  def test_experimental_ops_and_impl_modules_consistency(self):
    for op in ops_list.OPS:
      self.assertNotEmpty(
          op.impls, f"Op '{op.name}' must have at least one implementation"
      )
      modules = [impl.module for impl in op.impls]
      self.assertCountEqual(
          modules,
          set(modules),
          f"Implementation modules must be unique within op '{op.name}'",
      )
      for impl in op.impls:
        impl_op_dir = _module_op_dir(impl.module)
        self.assertIsNotNone(
            impl_op_dir,
            f"Impl module '{impl.module}' on op '{op.name}' must start with"
            " '_src/ops/' or 'google/ops/'",
        )
        self.assertEqual(
            op.op_dir,
            impl_op_dir,
            f"Impl {impl.module} resolves to {impl_op_dir}, expected"
            f" {op.op_dir}",
        )


if __name__ == "__main__":
  absltest.main()
