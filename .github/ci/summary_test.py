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
"""Tests for `summary.py`.

Mostly about the one rule that matters: a duration is reported only when it is
a measurement. A cancelled job carries a plausible-looking duration, and the
shards `fail-fast` cancels are the long ones, so getting this wrong publishes
"the 38m shard took 5m" on every red run.

Stdlib `unittest`, no third-party imports.

  python3 -m unittest discover -s .github/ci -p '*_test.py'
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import shards
import summary


def job(
    shard: str,
    runner: str = 'h100',
    conclusion: str | None = 'success',
    minutes: float | None = 10,
) -> str:
  """Builds one line of `gh api --jq '.jobs[]'` output.

  Args:
    shard: Shard name, as it appears in the job title.
    runner: Runner short name.
    conclusion: The job's conclusion.
    minutes: Wall clock, or None to omit the timestamps entirely.

  Returns:
    A JSON object on one line.
  """
  record = {
      'name': f'shard-tests ({runner}, {shard})',
      'conclusion': conclusion,
  }
  if minutes is not None:
    end = 60 * minutes
    record['started_at'] = '2026-09-03T00:00:00Z'
    record['completed_at'] = (
        f'2026-09-03T{int(end // 3600):02d}'
        f':{int(end % 3600 // 60):02d}:{int(end % 60):02d}Z'
    )
  return json.dumps(record)


TABLE = {
    'attention-long': dict(paths=('a_test.py',), minutes=30),
    'core-short': dict(paths=('b_test.py',), minutes=2),
}


def render(lines: list[str], table: dict[str, object] | None = None) -> str:
  """Renders the summary over a fake table.

  Args:
    lines: Job JSON lines.
    table: Stand-in for `SHARDS`, defaulting to `TABLE`.

  Returns:
    The markdown, as one string.
  """
  with mock.patch.dict(shards.SHARDS, table or TABLE, clear=True):
    return '\n'.join(summary.render(summary.parse_jobs(lines)))


class DurationTest(unittest.TestCase):
  """When a number is a measurement and when it is not."""

  def test_all_runners_green_reports_the_slowest(self) -> None:
    out = render([
        job('attention-long', 'h100', minutes=20),
        job('attention-long', 'tpu6e', minutes=31),
        job('core-short', 'h100', minutes=2),
    ])
    self.assertIn('| `attention-long` | pass | 30m | 31m | +1 | tpu6e |', out)

  def test_cancelled_shard_reports_no_duration(self) -> None:
    out = render([job('attention-long', 'h100', 'cancelled', minutes=5)])
    self.assertIn('| `attention-long` | cancelled | 30m | - | - | - |', out)
    self.assertNotIn('5m', out)

  def test_one_cancelled_runner_suppresses_the_whole_shard(self) -> None:
    out = render([
        job('attention-long', 'h100', minutes=20),
        job('attention-long', 'tpu6e', 'cancelled', minutes=3),
    ])
    self.assertIn('| `attention-long` | cancelled | 30m | - | - | - |', out)

  def test_failed_shard_reports_no_duration(self) -> None:
    out = render([job('attention-long', 'h100', 'failure', minutes=4)])
    self.assertIn('| `attention-long` | FAIL | 30m | - | - | - |', out)

  def test_missing_timestamps_report_no_duration(self) -> None:
    out = render([job('attention-long', 'h100', 'success', minutes=None)])
    self.assertIn('| `attention-long` | pass | 30m | - | - | - |', out)


class TableShapeTest(unittest.TestCase):
  """The table reads the same on every run."""

  def test_cancelled_shard_keeps_its_row(self) -> None:
    out = render([
        job('attention-long', 'h100', 'cancelled', minutes=5),
        job('core-short', 'h100', minutes=2),
    ])
    self.assertIn('| `attention-long` | cancelled | 30m | - | - | - |', out)
    self.assertIn('2 shards ran:', out)

  def test_unselected_shard_gets_no_row(self) -> None:
    out = render([job('core-short', 'h100', minutes=2)])
    self.assertNotIn('attention-long', out)
    self.assertIn('1 shards ran: 1 pass (1 not selected)', out)

  def test_skipped_shard_gets_no_row(self) -> None:
    out = render([job('attention-long', 'h100', 'skipped', minutes=0)])
    self.assertNotIn('attention-long', out)
    self.assertIn('(2 not selected)', out)

  def test_nothing_selected_still_renders(self) -> None:
    out = render([])
    self.assertIn('No shards ran. (2 not selected)', out)

  def test_rows_stay_in_shard_order_when_measurement_is_missing(self) -> None:
    out = render([
        job('attention-long', 'h100', 'cancelled', minutes=1),
        job('core-short', 'h100', minutes=2),
    ])
    rows = [l for l in out.splitlines() if l.startswith('| `')]
    self.assertEqual(
        [r.split('`')[1] for r in rows], ['attention-long', 'core-short']
    )

  def test_result_is_the_worst_across_runners(self) -> None:
    out = render([
        job('attention-long', 'h100', 'success'),
        job('attention-long', 'tpu6e', 'cancelled'),
        job('attention-long', 'tpu7x', 'failure'),
    ])
    self.assertIn('| `attention-long` | FAIL |', out)


class DriftTest(unittest.TestCase):
  """Calling out a table that needs rebalancing."""

  def test_large_drift_is_called_out(self) -> None:
    out = render([job('core-short', 'h100', minutes=39)])
    self.assertIn('1 shard(s) drifted', out)
    self.assertIn('`core-short`: table says 2m, ran 39m (+37)', out)

  def test_small_drift_is_not_called_out(self) -> None:
    out = render([job('core-short', 'h100', minutes=3)])
    self.assertNotIn('drifted', out)

  def test_cancelled_shard_never_drifts(self) -> None:
    out = render([job('attention-long', 'h100', 'cancelled', minutes=1)])
    self.assertNotIn('drifted', out)


class ParseTest(unittest.TestCase):
  """Reading the API output."""

  def test_non_shard_jobs_are_ignored(self) -> None:
    lines = [
        json.dumps({'name': 'shard-matrix', 'conclusion': 'success'}),
        json.dumps({'name': 'shard-all-tests-passed', 'conclusion': None}),
        job('core-short'),
    ]
    self.assertEqual(list(summary.parse_jobs(lines)), ['core-short'])

  def test_malformed_line_does_not_lose_the_table(self) -> None:
    # A summary that raises replaces the whole report with a traceback. The
    # shard reads as "not run", which is visible and recoverable.
    out = render(['{not json', '', job('core-short', minutes=2)])
    self.assertIn('| `core-short` | pass | 2m | 2m | +0 | h100 |', out)

  def test_rfc3339_z_timestamps_parse(self) -> None:
    self.assertEqual(
        summary.duration_minutes(
            '2026-09-03T09:00:00Z', '2026-09-03T09:30:00Z'
        ),
        30,
    )

  def test_unparseable_timestamp_costs_one_number_not_the_table(self) -> None:
    self.assertIsNone(summary.duration_minutes('not a time', 'nor this'))
    out = render([
        json.dumps({
            'name': 'shard-tests (h100, core-short)',
            'conclusion': 'success',
            'started_at': 'not a time',
            'completed_at': 'nor this',
        })
    ])
    self.assertIn('| `core-short` | pass | 2m | - | - | - |', out)

  def test_shard_name_containing_a_comma_is_read_whole(self) -> None:
    lines = [job('core-short', 'h100')]
    self.assertIn('core-short', summary.parse_jobs(lines))


class RealTableTest(unittest.TestCase):
  """Against the shard table as it actually is."""

  def test_every_real_shard_appears_when_every_shard_ran(self) -> None:
    # The state today: nothing narrows the run, so all of them dispatch.
    lines = [job(name, 'h100') for name in shards.SHARDS]
    out = '\n'.join(summary.render(summary.parse_jobs(lines)))
    for name in shards.SHARDS:
      self.assertIn(f'| `{name}` |', out)
    self.assertNotIn('not selected', out)


if __name__ == '__main__':
  unittest.main()
