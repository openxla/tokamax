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
"""Verifies that routing_moe modules declare no module-level constants.

Constants and tuning parameters belong on a frozen dataclass (e.g., Config/Plan)
or as local variables inside the function that owns them. At module scope, value
assignments are restricted to zero-field frozen dataclass unit sentinels (e.g.,
`auto_quant = AutoQuant()`).
"""

import ast
import pathlib

from absl.testing import absltest
from absl.testing import parameterized


def _is_frozen_dataclass_decorator(node: ast.expr) -> bool:
  """Returns whether `node` is a `@dataclass(..., frozen=True)` call."""
  if not isinstance(node, ast.Call) or node.args:
    return False
  func = node.func
  is_dataclass = (
      isinstance(func, ast.Name) and func.id == "dataclass"
  ) or (isinstance(func, ast.Attribute) and func.attr == "dataclass")
  if not is_dataclass:
    return False
  return any(
      kw.arg == "frozen"
      and isinstance(kw.value, ast.Constant)
      and isinstance(kw.value.value, bool)
      and kw.value.value
      for kw in node.keywords
  )


def _is_unit_body_stmt(stmt: ast.stmt) -> bool:
  """Returns whether `stmt` is allowed inside a zero-field unit dataclass."""
  if isinstance(stmt, (ast.Pass, ast.FunctionDef)):
    return True
  return (
      isinstance(stmt, ast.Expr)
      and isinstance(stmt.value, ast.Constant)
      and isinstance(stmt.value.value, (str, type(...)))
  )


def _is_unit_dataclass(node: ast.ClassDef) -> bool:
  """Returns whether `node` is a zero-field, inheritance-free frozen dataclass."""
  if node.bases or node.keywords or len(node.decorator_list) != 1:
    return False
  if not _is_frozen_dataclass_decorator(node.decorator_list[0]):
    return False
  return all(_is_unit_body_stmt(stmt) for stmt in node.body)


def _is_unit_sentinel(value: ast.expr, unit_types: set[str]) -> bool:
  """Returns whether `value` instantiates a unit dataclass sentinel."""
  return (
      isinstance(value, ast.Call)
      and isinstance(value.func, ast.Name)
      and value.func.id in unit_types
      and not value.args
      and not value.keywords
  )


def find_module_scope_violations(source: str) -> list[tuple[int, str]]:
  """Returns `(lineno, first_line)` for disallowed top-level statements."""
  tree = ast.parse(source)
  unit_types = {
      stmt.name
      for stmt in tree.body
      if isinstance(stmt, ast.ClassDef) and _is_unit_dataclass(stmt)
  }
  violations: list[tuple[int, str]] = []
  for idx, stmt in enumerate(tree.body):
    if (
        idx == 0
        and isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    ):
      continue
    if isinstance(
        stmt,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.ClassDef,
            ast.TypeAlias,
        ),
    ):
      continue
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and _is_unit_sentinel(stmt.value, unit_types)
    ):
      continue
    segment = ast.get_source_segment(source, stmt) or ""
    first_line = segment.splitlines()[0] if segment else type(stmt).__name__
    violations.append((stmt.lineno, first_line))
  return violations


class ModuleScopeTest(parameterized.TestCase):

  def test_routing_moe_modules_have_no_disallowed_globals(self):
    # Use `.absolute()` rather than `.resolve()` because test runfiles symlink
    # leaf files into a flat CAS directory outside the package tree.
    routing_moe_dir = pathlib.Path(__file__).absolute().parents[2]
    packages = ("common", "gmm", "io_relayout", "layer")
    unexpected_dirs = sorted(
        p.name
        for p in routing_moe_dir.iterdir()
        if p.is_dir()
        and p.name not in (*packages, "test", "__pycache__")
        and any(p.glob("*.py"))
    )
    self.assertEmpty(
        unexpected_dirs,
        msg=(
            f"Add {unexpected_dirs} `:py_srcs` to `module_scope_unit_test`"
            " data and `packages`."
        ),
    )

    violations: list[str] = []
    for pkg in packages:
      py_files = sorted((routing_moe_dir / pkg).glob("*.py"))
      self.assertNotEmpty(py_files, msg=f"No .py files staged for {pkg}/.")
      for path in py_files:
        for lineno, line in find_module_scope_violations(path.read_text()):
          violations.append(f"{pkg}/{path.name}:{lineno}: {line}")

    self.assertEmpty(
        violations,
        msg=(
            "Disallowed module-level statements in routing_moe:\n  "
            + "\n  ".join(violations)
            + "\nHow to fix:\n"
            "  - Constants / tuning parameters: move onto a frozen dataclass"
            " (e.g., Config/Plan), or keep local inside the function that owns"
            " them.\n"
            "  - Imported symbol shorthands (e.g.,"
            " `P = sharding.PartitionSpec`): reference the qualified symbol"
            " directly (`sharding.PartitionSpec(...)`), not at module"
            " scope.\n"
            "  - Type-only aliases (e.g., `GridIndices = tuple[...]`): use"
            " `type GridIndices = tuple[...]` (annotation-only, not callable"
            " at runtime).\n"
            "Only `type` statements and zero-field `@dataclass(frozen=True)`"
            " unit sentinels are permitted at module scope."
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="unit_sentinel",
          source=(
              '"""Module docstring."""\n'
              "import dataclasses\n"
              "@dataclasses.dataclass(frozen=True)\n"
              "class AutoQuant:\n"
              '  """Doc."""\n'
              "auto_quant = AutoQuant()\n"
          ),
      ),
      dict(
          testcase_name="unit_sentinel_with_kw_only_and_pass",
          source=(
              "from dataclasses import dataclass\n"
              "@dataclass(frozen=True, kw_only=True)\n"
              "class TileSize:\n"
              "  pass\n"
              "tile_size = TileSize()\n"
          ),
      ),
      dict(
          testcase_name="type_alias_statement",
          source="type QuantFormat = int | None\n",
      ),
  )
  def test_allows_permitted_top_level_constructs(self, source: str):
    self.assertEmpty(find_module_scope_violations(source))

  @parameterized.named_parameters(
      dict(
          testcase_name="private_int_constant",
          source="_DEFAULT_NUM_BUFFERS = 2\n",
      ),
      dict(
          testcase_name="arithmetic_constant",
          source="_SCOPED_VMEM_RESERVED_BYTES = 64 * 1024\n",
      ),
      dict(
          testcase_name="annotated_float_constant",
          source="_MAX_SCRATCH_VMEM_RATIO: float = 0.75\n",
      ),
      dict(
          testcase_name="lowercase_constant",
          source="default_tile_rows = 512\n",
      ),
      dict(
          testcase_name="unfrozen_dataclass_sentinel",
          source=(
              "import dataclasses\n"
              "@dataclasses.dataclass\n"
              "class AutoQuant:\n"
              "  pass\n"
              "auto_quant = AutoQuant()\n"
          ),
      ),
      dict(
          testcase_name="explicitly_unfrozen_dataclass_sentinel",
          source=(
              "import dataclasses\n"
              "@dataclasses.dataclass(frozen=False)\n"
              "class AutoQuant:\n"
              "  pass\n"
              "auto_quant = AutoQuant()\n"
          ),
      ),
      dict(
          testcase_name="dataclass_with_field",
          source=(
              "import dataclasses\n"
              "@dataclasses.dataclass(frozen=True)\n"
              "class Config:\n"
              "  num_buffers: int = 2\n"
              "config = Config()\n"
          ),
      ),
      dict(
          testcase_name="dataclass_with_base_class",
          source=(
              "import dataclasses\n"
              "@dataclasses.dataclass(frozen=True)\n"
              "class Sub(Base):\n"
              "  pass\n"
              "sub = Sub()\n"
          ),
      ),
      dict(
          testcase_name="imported_class_alias",
          source="from jax import sharding\nP = sharding.PartitionSpec\n",
      ),
      dict(
          testcase_name="imported_dtype_or_value_attribute",
          source="import jax.numpy as jnp\nDefaultDtype = jnp.bfloat16\n",
      ),
      dict(
          testcase_name="imported_upper_case_constant",
          source="import os\nSeekSet = os.SEEK_SET\n",
      ),
      dict(
          testcase_name="assignment_style_type_alias",
          source="GridIndices = tuple[int, int, int]\n",
      ),
      dict(
          testcase_name="top_level_control_flow",
          source="if True:\n  x = 1\n",
      ),
  )
  def test_rejects_disallowed_top_level_constructs(self, source: str):
    self.assertLen(find_module_scope_violations(source), 1)


if __name__ == "__main__":
  absltest.main()
