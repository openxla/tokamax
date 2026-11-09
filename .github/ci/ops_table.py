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
"""Builds the supported-ops page from `_src/ops/ops_list.py` and the manifest.

Renders `docs/supported_ops.md` from two sources:

- `docs/ops_manifest.toml` holds presentation decisions: section ordering,
  optional family groupings, one-line summaries, whether an op is listed, and
  any shared non-op directories under `_src/ops/`.
- `tokamax/_src/ops/ops_list.py` gives the ops, backends, and accelerator
  hardware, with Copybara
  `google_internal` stripped sections excluded so internal entries never leak
  into the public page.

Stdlib-only so it runs before any dependencies are installed.
"""

from __future__ import annotations

import argparse
import ast
import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import datetime
import difflib
import html
import importlib.util
import json
import pathlib
import sys
import tomllib
import types
from typing import Any
import xml.etree.ElementTree as ET

GITHUB_OPS_URL = 'https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/'

RENDER_COMMAND = 'python3 .github/ci/ops_table.py render'

CANONICAL_BACKENDS: tuple[str, ...] = (
    'xla',
    'mosaic_tpu',
    'mosaic_gpu',
    'triton',
    'cutedsl',
    'cudnn',
)

HARDWARE: tuple[str, ...] = ('TPU', 'GPU')

CHECKS: tuple[tuple[str, str], ...] = (
    ('api', 'Has an `api.py` entry point.'),
    ('ops_list', 'Is registered in `_src/ops/ops_list.py`.'),
    ('xla', 'Has an XLA implementation to test against.'),
    ('kernel', 'Has at least one accelerator kernel.'),
    ('canonical', 'Every implementation name is a canonical backend.'),
    ('tests', 'Has tests.'),
    ('arg_specs', 'Has `arg_specs.py`, used by benchmarks and autotuning.'),
    ('public', 'Is exported from `tokamax` or `tokamax.ops`.'),
)

_MANIFEST_OP_KEYS: frozenset[str] = frozenset(
    {'section', 'summary', 'family', 'listed'}
)

_JUNIT_RUNNER_SEP = '--'

_CHECK = '✅'

_STYLE = """\
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
</style>"""


@dataclasses.dataclass(frozen=True)
class Layout:
  """Where things live in either the GitHub or the internal layout."""

  root: pathlib.Path
  package: pathlib.Path

  @property
  def ops_dir(self) -> pathlib.Path:
    return self.package / '_src' / 'ops'

  @property
  def ops_list_file(self) -> pathlib.Path:
    return self.ops_dir / 'ops_list.py'

  @property
  def torch_ops_dir(self) -> pathlib.Path:
    return self.package / 'experimental' / 'torch_tpu' / 'ops'

  @property
  def manifest(self) -> pathlib.Path:
    return self.root / 'docs' / 'ops_manifest.toml'

  @property
  def page(self) -> pathlib.Path:
    return self.root / 'docs' / 'supported_ops.md'

  @property
  def hw_support(self) -> pathlib.Path:
    return self.package / 'google' / 'hw_support.bzl'

  def public_api_files(self) -> list[pathlib.Path]:
    files = [self.package / '__init__.py']
    files += sorted((self.package / 'ops').glob('*.py'))
    files += sorted((self.package / 'ops' / 'experimental').glob('*.py'))
    return [item for item in files if item.is_file()]


def find_layout(root: pathlib.Path) -> Layout:
  if (root / 'tokamax' / '_src' / 'ops' / 'ops_list.py').is_file():
    return Layout(root, root / 'tokamax')
  if (root / '_src' / 'ops' / 'ops_list.py').is_file():
    return Layout(root, root)
  raise SystemExit(
      f'No tokamax/_src/ops/ops_list.py or _src/ops/ops_list.py under {root}.'
  )


@dataclasses.dataclass
class Op:
  """Derived metadata and manifest fields for one op."""

  path: str
  experimental: bool
  backends: tuple[str, ...]
  hardware: tuple[str, ...]
  torch_tpu: bool
  checks: dict[str, bool]
  non_canonical: tuple[str, ...]
  section: str = ''
  summary: str = ''
  family: str | None = None
  listed: bool = True

  @property
  def name(self) -> str:
    return self.path.rsplit('/', 1)[-1]

  @property
  def dotted(self) -> str:
    return self.path.replace('/', '.')


@dataclasses.dataclass
class Manifest:
  """Parsed `docs/ops_manifest.toml`."""

  sections: list[str]
  not_ops: list[str]
  families: dict[str, str]
  ops: dict[str, dict[str, Any]]


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
  return tuple(dict.fromkeys(items))


def split_backend(name: str) -> tuple[str, str] | None:
  """`mosaic_tpu_v2` -> (`mosaic_tpu`, `v2`); None if not canonical."""
  for backend in sorted(CANONICAL_BACKENDS, key=len, reverse=True):
    if name == backend:
      return backend, ''
    if name.startswith(backend + '_'):
      return backend, name[len(backend) + 1 :]
  return None


def strip_copybara_internal(source: str) -> str:
  """Removes Copybara internal-only strip blocks."""
  # Split so Copybara's scrubber doesn't treat these literals as markers.
  marker = '# copybara:' + 'strip'
  begin, end = marker + '_begin', marker + '_end'
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
    if depth == 0:
      kept.append(line)
  return ''.join(kept)


def _call_kwarg(call: ast.Call, name: str) -> ast.AST | None:
  for kw in call.keywords:
    if kw.arg == name:
      return kw.value
  return None


def _const_str(node: ast.AST | None) -> str | None:
  if isinstance(node, ast.Constant) and isinstance(node.value, str):
    return node.value
  return None


def _impl_call_name(node: ast.AST) -> str | None:
  if not (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id == 'Impl'
  ):
    return None
  arg = node.args[0] if node.args else _call_kwarg(node, 'name')
  return _const_str(arg)


def _op_call_impl_names(call: ast.Call) -> tuple[str | None, set[str]]:
  """Extracts (op_name, impl_names) from an Op(...) AST call."""
  if not (isinstance(call.func, ast.Name) and call.func.id == 'Op'):
    return None, set()
  op_name = _const_str(_call_kwarg(call, 'name'))
  if not op_name:
    return None, set()
  impls_node = _call_kwarg(call, 'impls')
  if not isinstance(impls_node, (ast.Tuple, ast.List)):
    return op_name, set()
  names = {_impl_call_name(item) for item in impls_node.elts}
  return op_name, {name for name in names if name is not None}


def _extract_public_allowlist(source: str) -> dict[str, set[str]]:
  """Parses stripped `ops_list.py` AST to find `{op_name: {impl_names}}`."""
  tree = ast.parse(strip_copybara_internal(source))
  allowed: dict[str, set[str]] = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      op_name, impl_names = _op_call_impl_names(node)
      if op_name:
        allowed[op_name] = impl_names
  return allowed


def load_public_ops_list(
    ops_list_path: pathlib.Path,
) -> tuple[types.ModuleType, dict[str, set[str]]]:
  """Loads `ops_list.py` via importlib alongside Copybara allowlist."""
  source = ops_list_path.read_text(encoding='utf-8')
  allowed = _extract_public_allowlist(source)
  mod_name = 'tokamax_public_ops_list'
  spec = importlib.util.spec_from_file_location(mod_name, ops_list_path)
  if spec is None or spec.loader is None:
    raise ImportError(f'Could not load {ops_list_path}')
  module = importlib.util.module_from_spec(spec)
  sys.modules[mod_name] = module
  spec.loader.exec_module(module)
  return module, allowed


def _is_public_impl(impl: Any, allowed_impls: set[str] | None) -> bool:
  if allowed_impls is not None and impl.name not in allowed_impls:
    return False
  return not impl.module.startswith('google/')


def public_ops_by_dir(
    ops_list_module: Any, allowed: Mapping[str, set[str]] | None = None
) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
  """Maps `_src/ops/`-relative op paths to `(raw_op, public_impls)`."""
  raw_ops_by_dir: dict[str, Any] = {}
  public_impls_by_dir: dict[str, tuple[Any, ...]] = {}
  for raw_op in ops_list_module.OPS:
    if allowed is not None and raw_op.name not in allowed:
      continue
    allowed_impls = allowed.get(raw_op.name) if allowed is not None else None
    impls = tuple(
        impl
        for impl in raw_op.impls
        if _is_public_impl(impl, allowed_impls)
    )
    op_dir = raw_op.op_dir
    if not op_dir or op_dir.startswith('google/'):
      continue
    raw_ops_by_dir[op_dir] = raw_op
    public_impls_by_dir[op_dir] = impls
  return raw_ops_by_dir, public_impls_by_dir


def _subdirs(directory: pathlib.Path) -> list[str]:
  if not directory.is_dir():
    return []
  return sorted(
      entry.name
      for entry in directory.iterdir()
      if entry.is_dir() and not entry.name.startswith(('.', '_'))
  )


def discover_op_dirs(layout: Layout) -> list[str]:
  """Every physical op directory on disk relative to `_src/ops/`."""
  found = [name for name in _subdirs(layout.ops_dir) if name != 'experimental']
  found += [
      f'experimental/{name}'
      for name in _subdirs(layout.ops_dir / 'experimental')
  ]
  return found


def _torch_wrappers_from_init(init_file: pathlib.Path) -> set[str]:
  """Extracts wrapped op directory names imported in `torch_tpu/ops/__init__.py`."""
  if not init_file.is_file():
    return set()
  tree = ast.parse(init_file.read_text(encoding='utf-8'))
  prefix = 'tokamax.experimental.torch_tpu.ops.'
  wrappers: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module:
      if node.module.startswith(prefix):
        wrapper = node.module.removeprefix(prefix).split('.', 1)[0]
        if wrapper:
          wrappers.add(wrapper)
  return wrappers


def _match_torch_wrapper(
    wrapper: str, by_name: Mapping[str, list[str]]
) -> list[str]:
  if wrapper in by_name:
    return list(by_name[wrapper])
  extending: list[str] = []
  for name, paths in by_name.items():
    if name.startswith(wrapper + '_'):
      extending.extend(paths)
  return extending if len(extending) == 1 else []


def torch_tpu_ops(layout: Layout, op_paths: Sequence[str]) -> set[str]:
  """Op paths wrapped under `experimental/torch_tpu/ops/`."""
  wrappers = _torch_wrappers_from_init(layout.torch_ops_dir / '__init__.py')
  wrappers.update(_subdirs(layout.torch_ops_dir))
  if not wrappers:
    return set()
  by_name: dict[str, list[str]] = collections.defaultdict(list)
  for op_path in op_paths:
    by_name[op_path.rsplit('/', 1)[-1]].append(op_path)
  matched: set[str] = set()
  for wrapper in wrappers:
    matched.update(_match_torch_wrapper(wrapper, by_name))
  return matched


def _is_test_file(path: pathlib.Path) -> bool:
  return (
      path.stem.endswith('_test')
      or path.stem.startswith('test_')
      or path.stem == 'test_base'
  )


def _impl_hardware(impls: Sequence[Any]) -> tuple[str, ...]:
  """Extracts sorted accelerator hardware ('TPU', 'GPU') from impls."""
  found: set[str] = set()
  for impl in impls:
    backend = impl.backend
    if backend is None or backend.name == 'xla':
      continue
    for platform in backend.platforms:
      upper = str(platform).upper()
      if upper in HARDWARE:
        found.add(upper)
  return tuple(hw for hw in HARDWARE if hw in found)


def analyze_op(
    layout: Layout,
    path: str,
    raw_op: Any,
    public_impls: Sequence[Any],
    *,
    torch_ops: set[str],
    public_api_text: str,
) -> Op:
  """Builds an `Op` purely from public `ops_list` data and local wrappers."""
  backends = _dedupe(str(impl.name) for impl in public_impls)
  non_canonical = tuple(
      name for name in backends if split_backend(name) is None
  )
  hardware = _impl_hardware(public_impls)
  op_dir = layout.ops_dir / path
  has_xla = any(
      impl.backend is not None and impl.backend.name == 'xla'
      for impl in public_impls
  )
  has_dir = op_dir.is_dir()
  dotted = path.replace('/', '.')
  has_tests = has_dir and any(
      _is_test_file(item) for item in op_dir.rglob('*.py')
  )
  is_public = (
      f'tokamax._src.ops.{dotted}.' in public_api_text
      or f'tokamax._src.ops.{dotted} ' in public_api_text
  )
  checks = {
      'api': bool(raw_op.apis) or (op_dir / 'api.py').is_file(),
      'ops_list': True,
      'xla': has_xla or (op_dir / 'reference.py').is_file(),
      'kernel': bool(hardware),
      'canonical': bool(backends) and not non_canonical,
      'tests': has_tests,
      'arg_specs': (op_dir / 'arg_specs.py').is_file(),
      'public': is_public,
  }
  return Op(
      path=path,
      experimental=path.startswith('experimental/'),
      backends=backends,
      hardware=hardware,
      torch_tpu=path in torch_ops,
      checks=checks,
      non_canonical=non_canonical,
  )


def load_manifest(path: pathlib.Path) -> Manifest:
  with path.open('rb') as f:
    data = tomllib.load(f)
  return Manifest(
      sections=list(data.get('sections', [])),
      not_ops=list(data.get('not_ops', [])),
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
  for listed_dir in list(manifest.ops) + manifest.not_ops:
    if listed_dir.startswith('google/'):
      errors.append(
          'docs/ops_manifest.toml must not contain internal path'
          f' `{listed_dir}`.'
      )
    elif listed_dir in manifest.ops and listed_dir not in known_paths:
      errors.append(
          f'docs/ops_manifest.toml lists `{listed_dir}`, which is not a'
          ' public op in `ops_list.py`.'
      )
    elif (
        listed_dir in manifest.not_ops
        and disk_dirs
        and listed_dir not in disk_dirs
    ):
      errors.append(
          f'docs/ops_manifest.toml lists `{listed_dir}` in `not_ops`, but'
          f' `_src/ops/{listed_dir}` does not exist.'
      )
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
    if not isinstance(entry.get('listed', True), bool):
      errors.append(f'{where}: `listed` must be true or false.')
  for section in manifest.sections:
    errors += _validate_section_families(section, manifest.ops)
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


def read_hw_support(path: pathlib.Path) -> dict[str, tuple[str, ...]] | None:
  """Reads `google/hw_support.bzl`'s op tuples, or None if it is absent."""
  if not path.is_file():
    return None
  out: dict[str, tuple[str, ...]] = {}
  keys = ('SHARED_OPS', 'TPU_ONLY_OPS', 'GPU_ONLY_OPS')
  for node in ast.parse(path.read_text(encoding='utf-8')).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      if isinstance(target, ast.Name) and target.id in keys:
        out[target.id] = tuple(ast.literal_eval(node.value))
  return out


def hw_support_errors(
    hw_support: Mapping[str, Sequence[str]], ops: Sequence[Op]
) -> list[str]:
  """Cross-checks public ops against `google/hw_support.bzl`."""
  expected = {
      'SHARED_OPS': ('TPU', 'GPU'),
      'TPU_ONLY_OPS': ('TPU',),
      'GPU_ONLY_OPS': ('GPU',),
  }
  reverse_expected = {value: key for key, value in expected.items()}
  by_path = {op.path: op for op in ops}
  seen_in_bzl: set[str] = set()
  errors: list[str] = []
  for group, paths in hw_support.items():
    for op_path in paths:
      seen_in_bzl.add(op_path)
      op = by_path.get(op_path)
      if op is None or not op.hardware:
        continue
      if op.hardware != expected[group]:
        errors.append(
            f'hw_support.bzl lists `{op_path}` in {group}, but ops_list'
            f' derives {"/".join(op.hardware)}.'
        )
  for op in ops:
    if op.hardware and op.path not in seen_in_bzl:
      want = reverse_expected[op.hardware]
      errors.append(
          f'`{op.path}` has {"/".join(op.hardware)} kernels in ops_list, but'
          f' is missing from hw_support.bzl ({want}).'
      )
  return errors


def build_ops(
    layout: Layout,
    manifest: Manifest,
    raw_ops: Mapping[str, Any],
    public_impls_by_dir: Mapping[str, Sequence[Any]],
) -> list[Op]:
  """Analyzes every op in the manifest, in section and manifest order."""
  paths = [op_path for op_path in manifest.ops if op_path in raw_ops]
  torch_ops = torch_tpu_ops(layout, paths)
  public_api_text = '\n'.join(
      item.read_text(errors='replace') for item in layout.public_api_files()
  )
  ops: list[Op] = []
  for section in manifest.sections:
    for op_path in paths:
      entry = manifest.ops[op_path]
      if entry.get('section') != section:
        continue
      op = analyze_op(
          layout,
          op_path,
          raw_ops[op_path],
          public_impls_by_dir[op_path],
          torch_ops=torch_ops,
          public_api_text=public_api_text,
      )
      op.section = section
      op.summary = entry.get('summary', '')
      op.family = entry.get('family')
      op.listed = entry.get('listed', True)
      ops.append(op)
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
    extra_cells: Mapping[str, Sequence[str]] | None,
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
  for cell in (extra_cells or {}).get(row.path, ()):
    lines.append('      ' + _centered(cell))
  lines.append(f'      <td>{row.summary}</td>')
  lines.append('    </tr>')
  return lines


def _table(
    ops: Sequence[Op],
    families: Mapping[str, str],
    extra_headers: Sequence[str] = (),
    extra_cells: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
  """Renders one section table. `extra_*` add columns before Summary."""
  has_family = any(op.family for op in ops)
  headers = [
      ('HW', 'left'),
      ('Backends', 'left'),
      ('PyTorch TPU API', 'center'),
      *((item, 'center') for item in extra_headers),
      ('Summary', 'left'),
  ]
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
      lines += _row_cells(ops[i + j], families, colspan, run, j, extra_cells)
    i += run
  lines += ['  </tbody>', '</table>']
  return lines


def render_page(ops: Sequence[Op], manifest: Manifest) -> str:
  """Renders the checked-in `docs/supported_ops.md` markdown document."""
  intro = (
      'This page is generated from `tokamax/_src/ops/ops_list.py` and'
      ' `docs/ops_manifest.toml`. **HW** and **Backends** come from the'
      ' public implementations registered in `ops_list.py`. **PyTorch TPU'
      ' API** means `tokamax/experimental/torch_tpu/ops/` wraps the op.'
  )
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


def op_to_json(op: Op) -> dict[str, Any]:
  return {
      'path': op.path,
      'name': op.name,
      'experimental': op.experimental,
      'listed': op.listed,
      'section': op.section,
      'family': op.family,
      'hardware': list(op.hardware),
      'backends': list(op.backends),
      'non_canonical_implementations': list(op.non_canonical),
      'torch_tpu': op.torch_tpu,
      'checks': op.checks,
  }


@dataclasses.dataclass
class Outcomes:
  """Per-op nightly test outcome counts."""

  passed: int = 0
  failed: int = 0
  skipped: int = 0

  def cell(self) -> str:
    if self.failed:
      return f'❌ {self.failed} failed'
    if self.passed:
      return f'{_CHECK} {self.passed}'
    if self.skipped:
      return '⏭ skipped'
    return ''


def op_for_test(classname: str, op_paths: Sequence[str]) -> str | None:
  """`tokamax._src.ops.experimental.mla_swa.api_test.T` -> `experimental/mla_swa`."""
  marker = '_src.ops.'
  if marker not in classname:
    return None
  parts = classname.split(marker, 1)[1].split('.')
  known = set(op_paths)
  for segs in (2, 1):
    candidate = '/'.join(parts[:segs])
    if len(parts) > segs and candidate in known:
      return candidate
  return None


def _record_case_outcome(outcome: Outcomes, case: ET.Element) -> None:
  if case.find('failure') is not None or case.find('error') is not None:
    outcome.failed += 1
  elif case.find('skipped') is not None:
    outcome.skipped += 1
  else:
    outcome.passed += 1


def read_junit(
    junit_dir: pathlib.Path, op_paths: Sequence[str]
) -> dict[str, dict[str, Outcomes]]:
  """Maps `runner label -> op path -> Outcomes` across JUnit XML files."""
  results: dict[str, dict[str, Outcomes]] = collections.defaultdict(
      lambda: collections.defaultdict(Outcomes)
  )
  for xml_file in sorted(junit_dir.rglob('*.xml')):
    runner = xml_file.stem.split(_JUNIT_RUNNER_SEP, 1)[0]
    for case in ET.parse(xml_file).iter('testcase'):
      op_path = op_for_test(case.get('classname', ''), op_paths)
      if op_path is not None:
        _record_case_outcome(results[runner][op_path], case)
  return results


_RUNNERS: dict[str, Any] | None = None


def _load_shards_runners() -> dict[str, Any]:
  """Loads and caches runner metadata from shards.py."""
  global _RUNNERS
  if _RUNNERS is not None:
    return _RUNNERS
  shards_path = pathlib.Path(__file__).with_name('shards.py')
  if not shards_path.is_file():
    _RUNNERS = {}
    return _RUNNERS
  mod_name = 'tokamax_ci_shards'
  shards_mod = sys.modules.get(mod_name)
  if shards_mod is None:
    spec = importlib.util.spec_from_file_location(mod_name, shards_path)
    if spec is None or spec.loader is None:
      _RUNNERS = {}
      return _RUNNERS
    shards_mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = shards_mod
    spec.loader.exec_module(shards_mod)
  runners = getattr(shards_mod, 'RUNNERS', {})
  _RUNNERS = runners if isinstance(runners, dict) else {}
  return _RUNNERS


def runner_display_name(label: str) -> str:
  """Returns the display name for a CI runner label from shards.py."""
  runners = _load_shards_runners()
  info = runners.get(label)
  return info[1] if isinstance(info, tuple) and len(info) > 1 else label


def _render_readiness_table(ops: Sequence[Op]) -> list[str]:
  """Renders the HTML readiness checklist section for the static site."""
  desc = (
      '<p>What an op is expected to have. Experimental ops are not held to'
      ' it; ops not on the page above are included.</p>'
  )
  lines = [
      '<h2>Readiness</h2>',
      desc,
      '<ul>',
  ]
  for key, description in CHECKS:
    lines.append(f'<li><code>{key}</code>: {html.escape(description)}</li>')
  lines += ['</ul>', '<table class="ops-table"><thead><tr><th>Op</th>']
  lines += [f'<th>{key}</th>' for key, _ in CHECKS]
  lines.append('<th>Non-canonical names</th></tr></thead><tbody>')
  for op in ops:
    lines.append(f'<tr><td>{_op_cell(op)}</td>')
    lines += [
        _centered(_CHECK if op.checks[key] else '❌') for key, _ in CHECKS
    ]
    names = ', '.join(
        f'<code>{html.escape(item)}</code>' for item in op.non_canonical
    )
    lines.append(f'<td>{names}</td></tr>')
  lines.append('</tbody></table>')
  return lines


def render_site(
    ops: Sequence[Op],
    manifest: Manifest,
    junit: Mapping[str, Mapping[str, Outcomes]],
    *,
    commit: str,
    run_url: str,
    now: datetime.datetime,
) -> str:
  """Renders the standalone GitHub Pages HTML with nightly runner columns."""
  runners = sorted(junit)
  headers = [f'Nightly {runner_display_name(runner)}' for runner in runners]
  cells = {
      op.path: [
          junit[runner].get(op.path, Outcomes()).cell() for runner in runners
      ]
      for op in ops
  }
  stamp = now.strftime('%Y-%m-%d %H:%M UTC')
  source = html.escape(commit[:12]) if commit else 'a local checkout'
  if run_url:
    source += f' after <a href="{html.escape(run_url)}">this nightly run</a>'
  body = [
      '<h1>Tokamax ops</h1>',
      f'<p>Generated {stamp} from {source}. Nightly columns count the tests'
      ' under each op directory that passed on that runner.</p>',
  ]
  for section in manifest.sections:
    rows = [op for op in ops if op.section == section and op.listed]
    if rows:
      body.append(f'<h2>{html.escape(section)}</h2>')
      body += _table(rows, manifest.families, headers, cells)
  body += _render_readiness_table(ops)
  return '\n'.join([
      '<!doctype html>',
      '<html lang="en"><head><meta charset="utf-8">',
      '<title>Tokamax ops</title>',
      '<style>body { font-family: sans-serif; margin: 2em; }</style>',
      _STYLE,
      '</head><body>',
      *body,
      '</body></html>',
      '',
  ])


def load_all(
    root: pathlib.Path,
) -> tuple[Layout, Manifest, list[Op], list[str]]:
  """Loads layout, manifest, public ops, and validation errors."""
  layout = find_layout(root)
  ops_list_mod, allowed = load_public_ops_list(layout.ops_list_file)
  raw_ops, public_impls_by_dir = public_ops_by_dir(ops_list_mod, allowed)
  manifest = load_manifest(layout.manifest)
  disk_dirs = discover_op_dirs(layout)
  errors = validate_manifest(manifest, sorted(raw_ops), disk_dirs)
  ops = build_ops(layout, manifest, raw_ops, public_impls_by_dir)
  hw_support = read_hw_support(layout.hw_support)
  if hw_support is not None:
    errors += hw_support_errors(hw_support, ops)
  return layout, manifest, ops, errors


def check(root: pathlib.Path) -> int:
  """Verifies manifest, `hw_support.bzl`, and checked-in `supported_ops.md`."""
  layout, manifest, ops, errors = load_all(root)
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


def _write_site(
    args: argparse.Namespace, ops: Sequence[Op], manifest: Manifest
) -> None:
  """Writes standalone HTML (`index.html`) and JSON (`ops.json`) site output."""
  junit = (
      read_junit(args.junit_dir, [op.path for op in ops])
      if args.junit_dir is not None
      else {}
  )
  args.out.mkdir(parents=True, exist_ok=True)
  now = datetime.datetime.now(datetime.timezone.utc)
  (args.out / 'index.html').write_text(
      render_site(
          ops,
          manifest,
          junit,
          commit=args.commit,
          run_url=args.run_url,
          now=now,
      ),
      encoding='utf-8',
  )
  payload = {
      'generated': now.isoformat(),
      'commit': args.commit,
      'run_url': args.run_url,
      'ops': [op_to_json(op) for op in ops],
      'nightly': {
          runner: {
              op_path: dataclasses.asdict(outcomes)
              for op_path, outcomes in by_op.items()
          }
          for runner, by_op in junit.items()
      },
  }
  (args.out / 'ops.json').write_text(
      json.dumps(payload, indent=2) + '\n', encoding='utf-8'
  )
  sys.stdout.write(
      f'Wrote {args.out}/index.html and ops.json ({len(ops)} ops).\n'
  )


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
  sub.add_parser('json', help='Print the derived table as JSON.')
  site = sub.add_parser('site', help='Write a standalone page and ops.json.')
  site.add_argument('--out', type=pathlib.Path, required=True)
  site.add_argument('--junit-dir', type=pathlib.Path)
  site.add_argument('--commit', default='')
  site.add_argument('--run-url', default='')
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  if args.command == 'check':
    return check(args.root)
  layout, manifest, ops, errors = load_all(args.root)
  if errors:
    for error in errors:
      sys.stderr.write(f'::error::{error}\n')
    return 1
  if args.command == 'render':
    layout.page.write_text(render_page(ops, manifest), encoding='utf-8')
    sys.stdout.write(f'Wrote {layout.page} ({len(ops)} ops).\n')
  elif args.command == 'json':
    json.dump([op_to_json(op) for op in ops], sys.stdout, indent=2)
    sys.stdout.write('\n')
  elif args.command == 'site':
    _write_site(args, ops, manifest)
  return 0


if __name__ == '__main__':
  sys.exit(main())
