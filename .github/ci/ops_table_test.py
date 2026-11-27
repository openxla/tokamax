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
import pathlib
import tempfile
import textwrap
import unittest

import ops_table

_STRIP = '# copybara:' + 'strip'
_STRIP_BEGIN = f'{_STRIP}_begin(google_internal)'
_STRIP_END = f'{_STRIP}_end'

_MANIFEST_HEAD = textwrap.dedent("""\
    sections = ['Attention', 'Other']
    not_ops = []

    [families.fam]
    title = 'Family'
""")

_OPS_LIST_PREAMBLE = (
    textwrap.dedent("""\
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
    """)
    + f'{_STRIP_BEGIN}\n'
    + "TRANSFORMER_ENGINE = Backend('transformer_engine',"
    + ' frozenset({Platform.GPU}))\n'
    + f'{_STRIP_END}\n'
    + textwrap.dedent("""\

        @dataclasses.dataclass(frozen=True)
        class Impl:
          name: str
          backend: Backend
          module: str
          kernel_dirs: tuple[str, ...] = ()
          vjp: tuple[str, ...] = ()

          @property
          def accelerator_platforms(self) -> frozenset[Platform]:
            if self.backend == XLA:
              return frozenset()
            return self.backend.platforms & frozenset({Platform.GPU, Platform.TPU})

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
    """)
)


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

  def test_strips_internal_blocks_nested_and_single_line(self):
    source = '\n'.join([
        'a = 1',
        _STRIP_BEGIN,
        'secret = 2',
        _STRIP_BEGIN,
        'nested = 3',
        _STRIP_END,
        'still_secret = 4',
        _STRIP_END,
        f'inline_secret = 5  {_STRIP}(google_internal)',
        'b = 6',
        '',
    ])
    stripped = ops_table.strip_copybara_internal(source)
    self.assertEqual(stripped, 'a = 1\nb = 6\n')


class DeriveOpsTest(OpsTableTestCase):

  def build(self, ops_body: str, manifest_body: str) -> dict[str, ops_table.Op]:
    self.repo.write_ops_list(ops_body)
    self.repo.manifest(manifest_body)
    layout = ops_table.find_layout(self.repo.root)
    ops_mod = ops_table.load_public_ops_list(layout.ops_list_file)
    impls_by_dir = ops_table.public_impls_by_dir(ops_mod)
    manifest = ops_table.load_manifest(layout.manifest)
    return {
        op.path: op
        for op in ops_table.build_ops(layout, manifest, impls_by_dir)
    }

  def test_derives_hardware_backends_and_torch_tpu(self):
    self.repo.write(
        'tokamax/experimental/torch_tpu/ops/__init__.py',
        textwrap.dedent("""\
            from tokamax.experimental.torch_tpu.ops.demo.tpu import Demo
            from tokamax.experimental.torch_tpu.ops.xent.tpu import Xent as linear_xent_loss
        """),
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
                name='linear_xent_loss',
                apis=('tokamax.ops.linear_xent_loss',),
                impls=(
                    Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/linear_xent_loss/tpu.py'),
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

        [ops."linear_xent_loss"]
        section = 'Other'
        summary = 'Aliased torch_tpu op.'

        [ops."experimental/shared_op"]
        section = 'Attention'
        torch_tpu = true
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

    self.assertTrue(ops_by_path['linear_xent_loss'].torch_tpu)

    shared = ops_by_path['experimental/shared_op']
    self.assertEqual(shared.hardware, ('TPU', 'GPU'))
    self.assertTrue(shared.experimental)
    self.assertTrue(shared.torch_tpu)

    flex = ops_by_path['flex']
    self.assertEqual(flex.hardware, ())
    self.assertEqual(flex.backends, ('xla',))
    self.assertFalse(flex.torch_tpu)

  def test_copybara_stripped_and_vjp_impls_never_appear(self):
    ops_by_path = self.build(
        f"""\
        OPS = (
            Op(
                name='norm',
                apis=('tokamax.ops.norm',),
                impls=(
                    Impl('xla', XLA, '_src/ops/norm/base.py'),
                    Impl('triton', TRITON, '_src/ops/norm/triton.py'),
                    Impl('xla_vjp', XLA, '_src/ops/norm/base.py', vjp=('xla',)),
                    {_STRIP_BEGIN}
                    Impl('transformer_engine', TRANSFORMER_ENGINE, 'google/ops/norm/te.py'),
                    {_STRIP_END}
                ),
            ),
            {_STRIP_BEGIN}
            Op(
                name='secret_op',
                impls=(Impl('mosaic_tpu', MOSAIC_TPU, 'google/ops/experimental/secret_op/tpu.py'),),
            ),
            {_STRIP_END}
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
    self.assertTrue(ops_by_path['norm'].bwd_pass)

  def test_multiple_ops_sharing_op_dir_and_manifest_bwd_pass(self):
    ops_by_path = self.build(
        """\
        OPS = (
            Op(
                name='mhc_pre',
                apis=('tokamax.ops.mhc_pre',),
                impls=(
                    Impl('xla', XLA, '_src/ops/mhc/base.py'),
                    Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/mhc/tpu.py'),
                ),
            ),
            Op(
                name='mhc_post',
                apis=('tokamax.ops.mhc_post',),
                impls=(
                    Impl('xla', XLA, '_src/ops/mhc/base.py'),
                    Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/mhc/tpu.py'),
                    Impl('xla_vjp', XLA, '_src/ops/mhc/base.py', vjp=('xla',)),
                ),
            ),
            Op(
                name='ragged_dot',
                apis=('tokamax.ragged_dot',),
                impls=(Impl('mosaic_tpu', MOSAIC_TPU, '_src/ops/ragged_dot/tpu.py'),),
            ),
        )
        """,
        """\
        [ops."mhc"]
        section = 'Other'
        summary = 'mHC.'

        [ops."ragged_dot"]
        section = 'Other'
        bwd_pass = true
        summary = 'Ragged dot.'
        """,
    )
    self.assertEqual(ops_by_path['mhc'].backends, ('xla', 'mosaic_tpu'))
    self.assertEqual(ops_by_path['mhc'].hardware, ('TPU',))
    self.assertTrue(ops_by_path['mhc'].bwd_pass)
    self.assertTrue(ops_by_path['ragged_dot'].bwd_pass)


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
    self.assertIn('<th align="center">Bwd Pass</th>', text)
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

  def test_check_reports_manifest_errors_without_crashing(self):
    self.repo.write(
        'docs/ops_manifest.toml',
        textwrap.dedent("""\
            sections = ['Attention']
            not_ops = ['x1']

            [ops."x1"]
            section = 'Attention'
            family = 'missing_family'
            summary = ''
            bogus = 1
        """),
    )
    code, out = self.run_main('check')
    self.assertEqual(code, 1)
    self.assertIn("family 'missing_family' is not defined", out)
    self.assertIn('both `ops` and `not_ops`', out)
    self.assertIn('unknown keys', out)
    self.assertIn('non-empty `summary`', out)


if __name__ == '__main__':
  unittest.main()
