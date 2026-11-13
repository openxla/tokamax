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
"""Unit tests for `ops_table.py` using synthetic on-disk repositories."""

import contextlib
import io
import json
import pathlib
import tempfile
import textwrap
import unittest

import ops_table

_MANIFEST_HEAD = """\
sections = ['Attention', 'Other']
not_ops = []

[families.fam]
title = 'Family'
"""

_OPS_LIST_PREAMBLE = """\
import dataclasses
import enum

class Platform(enum.StrEnum):
  CPU = 'cpu'
  GPU = 'gpu'
  TPU = 'tpu'

@dataclasses.dataclass(frozen=True)
class Backend:
  name: str
  platforms: frozenset[Platform]

XLA = Backend('xla', frozenset({Platform.CPU, Platform.GPU, Platform.TPU}))
MOSAIC_TPU = Backend('mosaic_tpu', frozenset({Platform.TPU}))
MOSAIC_GPU = Backend('mosaic_gpu', frozenset({Platform.GPU}))
TRITON = Backend('triton', frozenset({Platform.GPU}))

@dataclasses.dataclass(frozen=True)
class Impl:
  name: str
  backend: Backend
  module: str
  kernel_dirs: tuple[str, ...] = ()

@dataclasses.dataclass(frozen=True)
class Op:
  name: str
  apis: tuple[str, ...] = ()
  impls: tuple[Impl, ...] = ()

  @property
  def op_dir(self) -> str:
    for impl in self.impls:
      if impl.module.startswith('_src/ops/'):
        parts = impl.module.removeprefix('_src/ops/').split('/')
        if parts[0] == 'experimental' and len(parts) > 1:
          return f'experimental/{parts[1]}'
        return parts[0]
    return self.name
"""


class _Repo:
  """A GitHub-layout synthetic repository in a temporary directory."""

  def __init__(self, root: pathlib.Path):
    self.root = root
    self.ops = root / 'tokamax' / '_src' / 'ops'
    self.ops.mkdir(parents=True)
    (root / 'docs').mkdir()

  def write(self, relpath: str, text: str = '') -> None:
    target = self.root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(text), encoding='utf-8')

  def write_ops_list(self, ops_body: str) -> None:
    self.write(
        'tokamax/_src/ops/ops_list.py',
        _OPS_LIST_PREAMBLE + textwrap.dedent(ops_body),
    )

  def manifest(self, body: str) -> None:
    self.write('docs/ops_manifest.toml', _MANIFEST_HEAD + textwrap.dedent(body))


class OpsTableTestCase(unittest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.repo = _Repo(pathlib.Path(tmp.name))

  def run_main(self, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
      code = ops_table.main(['--root', str(self.repo.root), *argv])
    return code, out.getvalue()


class CopybaraStripTest(unittest.TestCase):

  def test_strips_internal_blocks(self):
    source = textwrap.dedent("""\
        a = 1
        b = 3
        """)
    stripped = ops_table.strip_copybara_internal(source)
    self.assertEqual(stripped, 'a = 1\nb = 3\n')

  def test_split_backend(self):
    self.assertEqual(ops_table.split_backend('mosaic_tpu'), ('mosaic_tpu', ''))
    self.assertEqual(
        ops_table.split_backend('mosaic_tpu_v2'), ('mosaic_tpu', 'v2')
    )
    self.assertEqual(ops_table.split_backend('xla_chunked'), ('xla', 'chunked'))
    self.assertIsNone(ops_table.split_backend('mosaic'))
    self.assertIsNone(ops_table.split_backend('chunked_xla'))


class DeriveOpsTest(OpsTableTestCase):

  def build(self, ops_body: str, manifest_body: str) -> dict[str, ops_table.Op]:
    self.repo.write_ops_list(ops_body)
    self.repo.manifest(manifest_body)
    layout = ops_table.find_layout(self.repo.root)
    ops_mod, allowed = ops_table.load_public_ops_list(layout.ops_list_file)
    raw_ops, public_impls_by_dir = ops_table.public_ops_by_dir(ops_mod, allowed)
    manifest = ops_table.load_manifest(layout.manifest)
    return {
        op.path: op
        for op in ops_table.build_ops(
            layout, manifest, raw_ops, public_impls_by_dir
        )
    }

  def test_derives_hardware_backends_and_torch_tpu(self):
    self.repo.write(
        'tokamax/experimental/torch_tpu/ops/__init__.py',
        'from tokamax.experimental.torch_tpu.ops.demo.tpu import Demo\n',
    )
    ops_by_path = self.build(
        """\
        OPS = (
            Op(
                name='demo',
                apis=('tokamax.ops.demo',),
                impls=(
                    Impl('xla', XLA, '_src/ops/demo/base.py'),
                    Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/demo/tpu.py'),
                ),
            ),
            Op(
                name='shared_op',
                apis=('tokamax.ops.shared_op',),
                impls=(
                    Impl('mosaic_gpu', MOSAIC_GPU, '_src/ops/experimental/shared_op/gpu.py'),
                    Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/experimental/shared_op/tpu.py'),
                ),
            ),
            Op(
                name='flex',
                apis=('tokamax.ops.flex',),
                impls=(Impl('xla', XLA, '_src/ops/flex/base.py'),),
            ),
        )
        """,
        """\
        [ops."demo"]
        section = 'Attention'
        summary = 'Demo op.'

        [ops."experimental/shared_op"]
        section = 'Attention'
        summary = 'Shared op.'

        [ops."flex"]
        section = 'Other'
        summary = 'XLA-only op.'
        """,
    )
    demo = ops_by_path['demo']
    self.assertEqual(demo.hardware, ('TPU',))
    self.assertEqual(demo.backends, ('xla', 'mosaic_tpu'))
    self.assertTrue(demo.torch_tpu)

    shared = ops_by_path['experimental/shared_op']
    self.assertEqual(shared.hardware, ('TPU', 'GPU'))
    self.assertTrue(shared.experimental)

    flex = ops_by_path['flex']
    self.assertEqual(flex.hardware, ())
    self.assertEqual(flex.backends, ('xla',))

  def test_copybara_stripped_internal_ops_and_backends_never_appear(self):
    ops_by_path = self.build(
        """\
        OPS = (
            Op(
                name='norm',
                apis=('tokamax.ops.norm',),
                impls=(
                    Impl('xla', XLA, '_src/ops/norm/base.py'),
                    Impl('triton', TRITON, '_src/ops/norm/triton.py'),
                ),
            ),
        )
        """,
        """\
        [ops."norm"]
        section = 'Other'
        summary = 'Norm op.'
        """,
    )
    self.assertEqual(set(ops_by_path), {'norm'})
    self.assertEqual(ops_by_path['norm'].backends, ('xla', 'triton'))


class RenderAndCheckTest(OpsTableTestCase):

  def setUp(self):
    super().setUp()
    self.repo.write_ops_list("""\
        OPS = (
            Op(
                name='x1',
                apis=('tokamax.ops.x1',),
                impls=(Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/x1/tpu.py'),),
            ),
            Op(
                name='x2',
                apis=('tokamax.ops.x2',),
                impls=(Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/x2/tpu.py'),),
            ),
            Op(
                name='solo',
                apis=('tokamax.ops.solo',),
                impls=(
                    Impl('xla', XLA, '_src/ops/solo/base.py'),
                    Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/solo/tpu.py'),
                ),
            ),
            Op(
                name='hidden',
                apis=('tokamax.ops.hidden',),
                impls=(Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/hidden/tpu.py'),),
            ),
        )
        """)
    self.repo.manifest("""\
        [ops."x1"]
        section = 'Attention'
        family = 'fam'
        summary = 'X one.'

        [ops."x2"]
        section = 'Attention'
        family = 'fam'
        summary = 'X <b>two</b>.'

        [ops."hidden"]
        section = 'Attention'
        listed = false
        summary = 'Hidden.'

        [ops."solo"]
        section = 'Other'
        summary = 'Solo.'
        """)

  def test_render_then_check_and_detect_drift(self):
    code, out = self.run_main('check')
    self.assertEqual(code, 1)
    self.assertIn('docs/supported_ops.md is stale', out)

    self.assertEqual(self.run_main('render')[0], 0)
    code, out = self.run_main('check')
    self.assertEqual(code, 0, out)

    page = self.repo.root / 'docs' / 'supported_ops.md'
    text = page.read_text(encoding='utf-8')
    self.assertIn('rowspan="2"', text)
    self.assertIn('X <b>two</b>.', text)
    self.assertNotIn('Hidden.', text)
    self.assertLess(text.index('### Attention'), text.index('### Other'))

    page.write_text(text.replace('Solo.', 'Hand edit.'), encoding='utf-8')
    self.assertEqual(self.run_main('check')[0], 1)

  def test_check_fails_on_unlisted_disk_directory(self):
    self.run_main('render')
    self.repo.write('tokamax/_src/ops/new_op/api.py', '')
    code, out = self.run_main('check')
    self.assertEqual(code, 1)
    self.assertIn('`_src/ops/new_op` is not in `ops_list.py` or `not_ops`', out)

  def test_hw_support_disagreement_fails(self):
    self.run_main('render')
    self.repo.write(
        'tokamax/google/hw_support.bzl',
        (
            'SHARED_OPS = ()\n'
            'TPU_ONLY_OPS = ("x1", "x2", "hidden")\n'
            'GPU_ONLY_OPS = ("solo",)\n'
        ),
    )
    code, out = self.run_main('check')
    self.assertEqual(code, 1)
    self.assertIn('lists `solo` in GPU_ONLY_OPS, but ops_list derives TPU', out)

  def test_json_and_site_with_nightly_junit(self):
    code, out = self.run_main('json')
    self.assertEqual(code, 0)
    by_path = {item['path']: item for item in json.loads(out)}
    self.assertEqual(set(by_path), {'x1', 'x2', 'solo', 'hidden'})
    self.assertFalse(by_path['hidden']['listed'])
    self.assertEqual(by_path['solo']['hardware'], ['TPU'])

    junit = self.repo.root / 'junit'
    junit.mkdir()
    (junit / 'linux-x86-ct6e-44-1tpu--ops.xml').write_text(
        textwrap.dedent("""\
            <testsuites><testsuite>
              <testcase classname="tokamax._src.ops.x1.api_test.T" name="a"/>
              <testcase classname="tokamax._src.ops.x1.api_test.T" name="b"/>
              <testcase classname="tokamax._src.ops.solo.api_test.T" name="c">
                <skipped/>
              </testcase>
              <testcase classname="tokamax._src.other_test.T" name="d"/>
            </testsuite></testsuites>
            """),
        encoding='utf-8',
    )
    out_dir = self.repo.root / 'site'
    code, out = self.run_main(
        'site',
        '--out',
        str(out_dir),
        '--junit-dir',
        str(junit),
        '--commit',
        'abcdef1234567890',
        '--run-url',
        'https://example/run',
    )
    self.assertEqual(code, 0, out)
    page = (out_dir / 'index.html').read_text(encoding='utf-8')
    self.assertIn('Nightly', page)
    self.assertIn('✅ 2', page)
    self.assertIn('⏭ skipped', page)
    self.assertIn('abcdef123456', page)
    self.assertIn('<h2>Readiness</h2>', page)
    payload = json.loads((out_dir / 'ops.json').read_text(encoding='utf-8'))
    runner = payload['nightly']['linux-x86-ct6e-44-1tpu']
    self.assertEqual(runner['x1'], {'passed': 2, 'failed': 0, 'skipped': 0})
    self.assertEqual(runner['solo']['skipped'], 1)


class JunitTest(unittest.TestCase):

  def test_op_for_test_prefers_experimental_path(self):
    paths = ['mla', 'experimental/mla']
    self.assertEqual(
        ops_table.op_for_test(
            'tokamax._src.ops.experimental.mla.api_test.T', paths
        ),
        'experimental/mla',
    )
    self.assertEqual(
        ops_table.op_for_test('tokamax._src.ops.mla.base_test.T', paths), 'mla'
    )
    self.assertIsNone(
        ops_table.op_for_test('tokamax._src.ops.op_test.T', paths)
    )

  def test_outcome_cells(self):
    self.assertEqual(ops_table.Outcomes(passed=3).cell(), '✅ 3')
    self.assertEqual(
        ops_table.Outcomes(passed=3, failed=1).cell(), '❌ 1 failed'
    )
    self.assertEqual(ops_table.Outcomes(skipped=2).cell(), '⏭ skipped')
    self.assertEqual(ops_table.Outcomes().cell(), '')


if __name__ == '__main__':
  unittest.main()
