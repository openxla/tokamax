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
"""Renders and checks `docs/supported_ops.md` from `ops_list.py` and manifest.

Stdlib-only so it runs before any dependencies are installed.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import difflib
import html
import pathlib
import sys
import textwrap
import tomllib
import types
from typing import Any

GITHUB_OPS_URL = 'https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/'
RENDER_COMMAND = 'python3 .github/ci/ops_table.py render'
HARDWARE: tuple[str, ...] = ('TPU', 'GPU')

_MANIFEST_OP_KEYS: frozenset[str] = frozenset(
    {'section', 'summary', 'family', 'listed', 'bwd_pass', 'torch_tpu'}
)
_CHECK = '✅'

_STYLE = textwrap.dedent("""\
    <style>
    table.ops-table {
      display: table !important;
      width: 100% !important;
      table-layout: fixed !important;
      border-collapse: collapse !important;
    }
    table.ops-table th,
    table.ops-table td {
      border: 1px solid #dadce0 !important;
      padding: 8px 12px;
      vertical-align: middle !important;
      overflow-wrap: break-word;
      word-wrap: break-word;
      word-break: break-word;
    }
    table.ops-table code {
      white-space: normal !important;
      overflow-wrap: anywhere !important;
      word-break: break-word !important;
    }
    </style>""")


@dataclasses.dataclass(frozen=True)
class Layout:
  """Paths in either the GitHub (`tokamax/_src/...`) or internal layout."""

  root: pathlib.Path
  package: pathlib.Path

  @property
  def ops_dir(self) -> pathlib.Path:
    return self.package / '_src' / 'ops'

  @property
  def ops_list_file(self) -> pathlib.Path:
    return self.ops_dir / 'ops_list.py'

  @property
  def torch_ops_init(self) -> pathlib.Path:
    return self.package / 'experimental' / 'torch_tpu' / 'ops' / '__init__.py'

  @property
  def manifest(self) -> pathlib.Path:
    return self.root / 'docs' / 'ops_manifest.toml'

  @property
  def page(self) -> pathlib.Path:
    return self.root / 'docs' / 'supported_ops.md'


def find_layout(root: pathlib.Path) -> Layout:
  """Detects the GitHub or internal repo layout under `root`."""
  if (root / 'tokamax' / '_src' / 'ops' / 'ops_list.py').is_file():
    return Layout(root, root / 'tokamax')
  if (root / '_src' / 'ops' / 'ops_list.py').is_file():
    return Layout(root, root)
  raise SystemExit(
      f'No tokamax/_src/ops/ops_list.py or _src/ops/ops_list.py under {root}.'
  )


@dataclasses.dataclass(frozen=True)
class Op:
  """Derived metadata and manifest fields for one table row."""

  path: str
  experimental: bool
  backends: tuple[str, ...]
  hardware: tuple[str, ...]
  torch_tpu: bool
  bwd_pass: bool
  section: str
  summary: str
  family: str | None = None
  listed: bool = True

  @property
  def name(self) -> str:
    return self.path.rsplit('/', 1)[-1]


@dataclasses.dataclass(frozen=True)
class Manifest:
  """Parsed `docs/ops_manifest.toml`."""

  sections: tuple[str, ...]
  not_ops: tuple[str, ...]
  families: Mapping[str, str]
  ops: Mapping[str, Mapping[str, Any]]


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
  return tuple(dict.fromkeys(items))


def strip_copybara_internal(source: str) -> str:
  """Removes Copybara internal-only strip blocks and single-line strips."""
  # Split so Copybara's scrubber doesn't treat these literals as markers.
  marker = '# copybara:' + 'strip'
  begin, end, single = marker + '_begin', marker + '_end', marker + '('
  kept: list[str] = []
  depth = 0
  for line in source.splitlines(keepends=True):
    comment = line.strip()
    if comment.startswith(begin):
      depth += 1
      continue
    if comment.startswith(end) and depth > 0:
      depth -= 1
      continue
    if depth == 0 and single not in line:
      kept.append(line)
  return ''.join(kept)


def load_public_ops_list(ops_list_path: pathlib.Path) -> types.ModuleType:
  """Loads `ops_list.py` with Copybara internal blocks stripped."""
  stripped = strip_copybara_internal(ops_list_path.read_text(encoding='utf-8'))
  mod_name = 'tokamax_public_ops_list'
  module = types.ModuleType(mod_name)
  sys.modules[mod_name] = module
  exec(compile(stripped, str(ops_list_path), 'exec'), module.__dict__)  # pylint: disable=exec-used
  return module


def public_impls_by_dir(ops_list_module: Any) -> dict[str, tuple[Any, ...]]:
  """Maps `_src/ops/`-relative op paths to their public `Impl` tuples."""
  grouped: dict[str, list[Any]] = {}
  for raw_op in ops_list_module.OPS:
    op_dir = raw_op.op_dir
    if not op_dir or op_dir.startswith('google/'):
      continue
    impls = [
        impl for impl in raw_op.impls if not impl.module.startswith('google/')
    ]
    grouped.setdefault(op_dir, []).extend(impls)
  return {op_dir: tuple(impls) for op_dir, impls in grouped.items()}


def _subdirs(directory: pathlib.Path) -> list[str]:
  if not directory.is_dir():
    return []
  return sorted(
      entry.name
      for entry in directory.iterdir()
      if entry.is_dir() and not entry.name.startswith(('.', '_'))
  )


def discover_op_dirs(layout: Layout) -> list[str]:
  """Returns every physical op directory under `_src/ops/`."""
  found = [name for name in _subdirs(layout.ops_dir) if name != 'experimental']
  found += [
      f'experimental/{name}'
      for name in _subdirs(layout.ops_dir / 'experimental')
  ]
  return found


def torch_tpu_wrapper_names(init_file: pathlib.Path) -> frozenset[str]:
  """Extracts wrapped op names imported in `torch_tpu/ops/__init__.py`."""
  if not init_file.is_file():
    return frozenset()
  tree = ast.parse(init_file.read_text(encoding='utf-8'))
  prefix = 'tokamax.experimental.torch_tpu.ops.'
  names: set[str] = set()
  for node in ast.walk(tree):
    if not (isinstance(node, ast.ImportFrom) and node.module):
      continue
    if not node.module.startswith(prefix):
      continue
    submod = node.module.removeprefix(prefix).split('.', 1)[0]
    if submod:
      names.add(submod)
    for alias in node.names:
      if alias.asname:
        names.add(alias.asname)
  return frozenset(names)


def _impl_hardware(impls: Sequence[Any]) -> tuple[str, ...]:
  """Extracts sorted accelerator hardware ('TPU', 'GPU') from impls."""
  found: set[str] = set()
  for impl in impls:
    for platform in impl.accelerator_platforms:
      found.add(str(platform).upper())
  return tuple(hw for hw in HARDWARE if hw in found)


def load_manifest(path: pathlib.Path) -> Manifest:
  """Loads and parses `docs/ops_manifest.toml`."""
  with path.open('rb') as f:
    data = tomllib.load(f)
  return Manifest(
      sections=tuple(data.get('sections', ())),
      not_ops=tuple(data.get('not_ops', ())),
      families={
          key: value.get('title', key)
          for key, value in data.get('families', {}).items()
      },
      ops=dict(data.get('ops', {})),
  )


def _validate_manifest_ops(
    manifest: Manifest,
    known_paths: set[str],
    disk_dirs: set[str],
) -> list[str]:
  """Checks 1-to-1 coverage between manifest ops, `ops_list`, and disk."""
  errors: list[str] = []
  for rel_dir in sorted(disk_dirs):
    if rel_dir not in known_paths and rel_dir not in manifest.not_ops:
      errors.append(
          f'`_src/ops/{rel_dir}` is not in `ops_list.py` or `not_ops`.'
      )
  for op_path in sorted(known_paths):
    if op_path not in manifest.ops:
      errors.append(
          f'`{op_path}` from `ops_list.py` is missing from'
          ' docs/ops_manifest.toml.'
      )
  for listed_dir in manifest.ops:
    if listed_dir.startswith('google/'):
      errors.append(
          'docs/ops_manifest.toml must not contain internal path'
          f' `{listed_dir}`.'
      )
    elif listed_dir not in known_paths:
      errors.append(
          f'docs/ops_manifest.toml lists `{listed_dir}`, which is not a'
          ' public op in `ops_list.py`.'
      )
  for listed_dir in manifest.not_ops:
    if listed_dir in manifest.ops:
      errors.append(
          f'docs/ops_manifest.toml lists `{listed_dir}` in both `ops` and'
          ' `not_ops`.'
      )
    elif listed_dir.startswith('google/'):
      errors.append(
          'docs/ops_manifest.toml must not contain internal path'
          f' `{listed_dir}`.'
      )
    elif disk_dirs and listed_dir not in disk_dirs:
      errors.append(
          f'docs/ops_manifest.toml lists `{listed_dir}` in `not_ops`, but'
          f' `_src/ops/{listed_dir}` does not exist.'
      )
  return errors


def _validate_section_families(
    section: str, manifest_ops: Mapping[str, Mapping[str, Any]]
) -> list[str]:
  """Ensures ops belonging to a family appear contiguously within `section`."""
  rows = [
      entry.get('family')
      for entry in manifest_ops.values()
      if entry.get('section') == section and entry.get('listed', True)
  ]
  seen: set[str] = set()
  errors: list[str] = []
  for i, family in enumerate(rows):
    if family is None or (i > 0 and rows[i - 1] == family):
      continue
    if family in seen:
      errors.append(
          f'Family {family!r} is split in section {section!r}; keep its ops'
          ' adjacent.'
      )
    seen.add(family)
  return errors


def _validate_manifest_entries(manifest: Manifest) -> list[str]:
  """Checks per-entry keys, section/family validity, and adjacent families."""
  errors: list[str] = []
  for op_path, entry in manifest.ops.items():
    where = f'[ops."{op_path}"]'
    unknown = set(entry) - _MANIFEST_OP_KEYS
    if unknown:
      errors.append(f'{where}: unknown keys {sorted(unknown)}.')
    if entry.get('section') not in manifest.sections:
      errors.append(f'{where}: section must be one of `sections`.')
    if not isinstance(entry.get('summary'), str) or not entry['summary']:
      errors.append(f'{where}: needs a non-empty `summary`.')
    if 'family' in entry and entry['family'] not in manifest.families:
      errors.append(f'{where}: family {entry["family"]!r} is not defined.')
    for flag in ('listed', 'bwd_pass', 'torch_tpu'):
      if flag in entry and not isinstance(entry[flag], bool):
        errors.append(f'{where}: `{flag}` must be true or false.')
  for section in manifest.sections:
    errors += _validate_section_families(section, manifest.ops)
  return errors


def validate_manifest(
    manifest: Manifest,
    public_paths: Sequence[str],
    disk_dirs: Sequence[str] = (),
) -> list[str]:
  """Returns problems that would make the generated page incomplete or wrong."""
  errors: list[str] = []
  if not manifest.sections:
    errors.append('`sections` is empty.')
  errors += _validate_manifest_ops(manifest, set(public_paths), set(disk_dirs))
  errors += _validate_manifest_entries(manifest)
  return errors


def build_ops(
    layout: Layout,
    manifest: Manifest,
    impls_by_dir: Mapping[str, Sequence[Any]],
) -> list[Op]:
  """Builds `Op` rows in section and manifest order."""
  torch_wrappers = torch_tpu_wrapper_names(layout.torch_ops_init)
  ops: list[Op] = []
  for section in manifest.sections:
    for op_path, entry in manifest.ops.items():
      if op_path not in impls_by_dir or entry.get('section') != section:
        continue
      impls = impls_by_dir[op_path]
      fwd_impls = tuple(impl for impl in impls if not getattr(impl, 'vjp', ()))
      has_vjp = any(getattr(impl, 'vjp', ()) for impl in impls)
      basename = op_path.rsplit('/', 1)[-1]
      ops.append(
          Op(
              path=op_path,
              experimental=op_path.startswith('experimental/'),
              backends=_dedupe(str(impl.name) for impl in fwd_impls),
              hardware=_impl_hardware(fwd_impls),
              torch_tpu=(
                  basename in torch_wrappers or entry.get('torch_tpu', False)
              ),
              bwd_pass=has_vjp or entry.get('bwd_pass', False),
              section=section,
              summary=entry.get('summary', ''),
              family=entry.get('family'),
              listed=entry.get('listed', True),
          )
      )
  return ops


def _op_cell(op: Op) -> str:
  label = (
      f'<a href="{GITHUB_OPS_URL}{html.escape(op.path)}">'
      f'<code>{html.escape(op.name)}</code></a>'
  )
  return label + (' (experimental)' if op.experimental else '')


def _centered(text: str) -> str:
  return f'<td align="center" style="text-align: center;">{text}</td>'


def _row_cells(
    row: Op,
    families: Mapping[str, str],
    colspan: str,
    run: int,
    row_index: int,
) -> list[str]:
  """Formats the `<tr>...</tr>` HTML lines for one table row."""
  lines = ['    <tr>']
  if row.family and row_index == 0:
    lines.append(
        f'      <td rowspan="{run}" align="center" valign="middle">'
        f'{families[row.family]}</td>'
    )
  if row.family:
    lines.append(f'      <td>{_op_cell(row)}</td>')
  else:
    lines.append(f'      <td{colspan}>{_op_cell(row)}</td>')
  hardware = ', '.join(row.hardware)
  if not hardware:
    hardware = 'XLA only' if 'xla' in row.backends else '—'
  lines.append(f'      <td>{hardware}</td>')
  backends = ', '.join(
      f'<code>{html.escape(item)}</code>' for item in row.backends
  )
  lines.append(f'      <td>{backends}</td>')
  lines.append('      ' + _centered(_CHECK if row.torch_tpu else ''))
  lines.append('      ' + _centered(_CHECK if row.bwd_pass else ''))
  lines.append(f'      <td>{row.summary}</td>')
  lines.append('    </tr>')
  return lines


def _table(ops: Sequence[Op], families: Mapping[str, str]) -> list[str]:
  """Renders one section table."""
  has_family = any(op.family for op in ops)
  headers = (
      ('HW', 'left'),
      ('Backends', 'left'),
      ('PyTorch TPU API', 'center'),
      ('Bwd Pass', 'center'),
      ('Summary', 'left'),
  )
  lines = ['<table class="ops-table">', '  <thead>', '    <tr>']
  colspan = ' colspan="2"' if has_family else ''
  lines.append(f'      <th{colspan} align="left">Op</th>')
  for header, align in headers:
    lines.append(f'      <th align="{align}">{html.escape(header)}</th>')
  lines += ['    </tr>', '  </thead>', '  <tbody>']
  i = 0
  while i < len(ops):
    run = 1
    if ops[i].family:
      while i + run < len(ops) and ops[i + run].family == ops[i].family:
        run += 1
    for j in range(run):
      lines += _row_cells(ops[i + j], families, colspan, run, j)
    i += run
  lines += ['  </tbody>', '</table>']
  return lines


def render_page(ops: Sequence[Op], manifest: Manifest) -> str:
  """Renders the checked-in `docs/supported_ops.md` markdown document."""
  intro = textwrap.dedent("""\
      This page is generated from `tokamax/_src/ops/ops_list.py` and
      `docs/ops_manifest.toml`. **HW**, **Backends**, and **Bwd Pass** come
      from the public implementations registered in `ops_list.py`. **PyTorch
      TPU API** means `tokamax/experimental/torch_tpu/ops/` wraps the op.""")
  lines = [
      '<!--',
      '  Generated by .github/ci/ops_table.py from docs/ops_manifest.toml and',
      '  tokamax/_src/ops/ops_list.py. Do not edit by hand: run',
      f'  `{RENDER_COMMAND}`.',
      '-->',
      '',
      '## Supported Ops and Hardware',
      '',
      _STYLE,
      '',
      intro,
  ]
  for section in manifest.sections:
    rows = [op for op in ops if op.section == section and op.listed]
    if not rows:
      continue
    lines += ['', f'### {section}', '']
    lines += _table(rows, manifest.families)
  return '\n'.join(lines) + '\n'


def load_all(
    root: pathlib.Path, *, check_disk_dirs: bool = True
) -> tuple[Layout, Manifest, list[Op], list[str]]:
  """Loads layout, manifest, public ops, and validation errors."""
  layout = find_layout(root)
  ops_list_mod = load_public_ops_list(layout.ops_list_file)
  impls_by_dir = public_impls_by_dir(ops_list_mod)
  manifest = load_manifest(layout.manifest)
  disk_dirs = discover_op_dirs(layout) if check_disk_dirs else []
  errors = validate_manifest(manifest, sorted(impls_by_dir), disk_dirs)
  ops = build_ops(layout, manifest, impls_by_dir)
  return layout, manifest, ops, errors


def check(root: pathlib.Path, *, check_disk_dirs: bool = True) -> int:
  """Verifies manifest and checked-in `supported_ops.md`."""
  layout, manifest, ops, errors = load_all(
      root, check_disk_dirs=check_disk_dirs
  )
  if not errors:
    expected = render_page(ops, manifest)
    actual = (
        layout.page.read_text(encoding='utf-8') if layout.page.is_file() else ''
    )
    if expected != actual:
      diff = difflib.unified_diff(
          actual.splitlines(keepends=True),
          expected.splitlines(keepends=True),
          'docs/supported_ops.md (checked in)',
          'docs/supported_ops.md (rendered)',
      )
      sys.stdout.writelines(list(diff)[:200])
      errors.append(
          f'docs/supported_ops.md is stale; run `{RENDER_COMMAND}` and commit'
          ' the result.'
      )
  for error in errors:
    sys.stdout.write(f'::error::{error}\n')
  if not errors:
    sys.stdout.write(
        f'docs/supported_ops.md is up to date ({len(ops)} ops).\n'
    )
  return 1 if errors else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  """Parses CLI arguments for `ops_table.py`."""
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
      '--root',
      type=pathlib.Path,
      default=pathlib.Path(__file__).resolve().parents[2],
      help='Repository root (GitHub) or package directory (internal).',
  )
  sub = parser.add_subparsers(dest='command', required=True)
  sub.add_parser('render', help='Rewrite docs/supported_ops.md.')
  sub.add_parser('check', help='Fail if the page or manifest is stale.')
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point for `ops_table.py`."""
  args = _parse_args(argv)
  if args.command == 'check':
    return check(args.root)
  layout, manifest, ops, errors = load_all(args.root)
  if errors:
    for error in errors:
      sys.stderr.write(f'::error::{error}\n')
    return 1
  layout.page.write_text(render_page(ops, manifest), encoding='utf-8')
  sys.stdout.write(f'Wrote {layout.page} ({len(ops)} ops).\n')
  return 0


if __name__ == '__main__':
  sys.exit(main())
