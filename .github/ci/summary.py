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
"""Renders the per-shard result and timing table for the run summary.
"""

from __future__ import annotations

from collections.abc import Sequence
import datetime
import json
import re
import sys

import shards

JOB_NAME = re.compile(r'shard-tests \((?P<runner>[^,]+), (?P<shard>.+)\)')

# Worst-first, so a shard's row reports its worst runner. `success` last means
# a shard is only green when every runner it ran on was.
SEVERITY = ('failure', 'timed_out', 'cancelled', 'skipped', 'success')

# What each conclusion is called in the table. `None` is a job that exists but
# has not finished, which `if: always()` makes possible if a runner is still
# winding down as the aggregate job starts.
RESULT_TEXT = {
    'success': 'pass',
    'failure': 'FAIL',
    'timed_out': 'TIMEOUT',
    'cancelled': 'cancelled',
    'skipped': 'skipped',
    None: 'running',
}

# Drift past this many minutes is worth re-adjusting.
REBALANCE_MINUTES = 5


def duration_minutes(started: object, completed: object) -> float | None:
  """Wall clock between two API timestamps.

  Args:
    started: `started_at`, or None.
    completed: `completed_at`, or None.

  Returns:
    Minutes, or None if either timestamp is absent or unparseable.
  """
  if not isinstance(started, str) or not isinstance(completed, str):
    return None
  try:
    delta = datetime.datetime.fromisoformat(
        completed.replace('Z', '+00:00')
    ) - datetime.datetime.fromisoformat(started.replace('Z', '+00:00'))
  except ValueError:
    # Same rule as a malformed line: a timestamp this does not understand
    # costs that shard its number, not the report its table.
    return None
  return delta.total_seconds() / 60


def parse_jobs(lines: Sequence[str]) -> dict[str, dict[str, dict[str, object]]]:
  """Groups shard jobs by shard and runner.

  Args:
    lines: JSON objects, one job per line, as `gh api --jq '.jobs[]'` emits.

  Returns:
    Shard name to runner short name to `{'conclusion', 'minutes'}`. `minutes`
    is None when the job did not report both timestamps.
  """
  jobs: dict[str, dict[str, dict[str, object]]] = {}
  for line in lines:
    line = line.strip()
    if not line:
      continue
    try:
      job = json.loads(line)
    except json.JSONDecodeError:
      # One malformed line must not cost the whole summary. Skipping it makes
      # its shard read as "not run", which is visible; raising would replace
      # the table with a stack trace.
      continue
    match = JOB_NAME.fullmatch(job.get('name', ''))
    if not match:
      continue
    jobs.setdefault(match['shard'], {})[match['runner']] = {
        'conclusion': job.get('conclusion'),
        'minutes': duration_minutes(
            job.get('started_at'), job.get('completed_at')
        ),
    }
  return jobs


def dispatched(runners: dict[str, dict[str, object]]) -> bool:
  """Whether this run actually started the shard.

  Args:
    runners: Runner short name to that job's record; empty if no job exists.

  Returns:
    True if any of the shard's jobs got past being skipped.
  """
  return any(r['conclusion'] != 'skipped' for r in runners.values())


def shard_result(runners: dict[str, dict[str, object]]) -> str | None:
  """Reduces a shard's per-runner conclusions to one.

  Args:
    runners: Runner short name to that job's record.

  Returns:
    The worst conclusion across the shard's runners.
  """
  seen = [r['conclusion'] for r in runners.values()]
  for conclusion in SEVERITY:
    if conclusion in seen:
      return conclusion
  return None


def measured(runners: dict[str, dict[str, object]]) -> tuple[float, str] | None:
  """Returns a shard's wall clock, or None if it cannot be trusted.

  Args:
    runners: Runner short name to that job's record.

  Returns:
    `(minutes, slowest runner)`, or None if any job did not succeed or is
    missing a duration.
  """
  if not runners:
    return None
  if any(r['conclusion'] != 'success' for r in runners.values()):
    return None
  if any(r['minutes'] is None for r in runners.values()):
    return None
  slowest = max(runners, key=lambda n: runners[n]['minutes'])
  return runners[slowest]['minutes'], slowest


def render(jobs: dict[str, dict[str, dict[str, object]]]) -> list[str]:
  """Builds the markdown summary.

  Args:
    jobs: As `parse_jobs` returns.

  Returns:
    Lines of markdown.
  """
  out = ['## Shard results and timings', '']
  rows = []
  counts = {}
  drifted = []
  absent = 0
  for name, spec in sorted(shards.SHARDS.items(), key=shards.shard_order):
    runners = jobs.get(name, {})
    if not dispatched(runners):
      absent += 1
      continue
    result = shard_result(runners)
    counts[result] = counts.get(result, 0) + 1
    claimed = spec.get('minutes')
    actual = measured(runners)
    if actual is None:
      rows.append((name, result, claimed, None, None, None))
      continue
    minutes, slowest = actual
    # Rounded the way the table records it, so the drift is the number that
    # would change if `minutes` were updated from this run, not a fraction.
    rounded = max(1, round(minutes))
    drift = None if claimed is None else rounded - claimed
    rows.append((name, result, claimed, rounded, drift, slowest))
    if drift is not None and abs(drift) >= REBALANCE_MINUTES:
      drifted.append((name, claimed, rounded, drift))

  tally = ', '.join(
      f'{n} {RESULT_TEXT.get(r, r)}' for r, n in sorted(counts.items())
  )
  headline = f'{len(rows)} shards ran: {tally}' if rows else 'No shards ran.'
  # The shards this run did not touch are a count, not rows. They are still
  # worth stating: it is the one number that says selection narrowed, and a
  # reader who expected their shard to run needs to see that it did not.
  if absent:
    headline += f' ({absent} not selected)'
  out += [headline, '']

  if drifted:
    out += [
        (
            f'{len(drifted)} shard(s) drifted at least'
            f' {REBALANCE_MINUTES}m from `shards.py`. Worth rebalancing:'
        ),
        '',
    ]
    for name, claimed, rounded, drift in sorted(
        drifted, key=lambda d: -abs(d[3])
    ):
      out.append(
          f'- `{name}`: table says {claimed}m, ran {rounded}m ({drift:+d})'
      )
    out.append('')

  out += [
      '| shard | result | table | actual | drift | slowest |',
      '| --- | --- | --- | --- | --- | --- |',
  ]
  for name, result, claimed, rounded, drift, slowest in rows:
    # `-` wherever the run cannot answer
    out.append(
        f'| `{name}` | {RESULT_TEXT.get(result, result)}'
        f' | {"-" if claimed is None else f"{claimed}m"}'
        f' | {"-" if rounded is None else f"{rounded}m"}'
        f' | {"-" if drift is None else f"{drift:+d}"}'
        f' | {slowest or "-"} |'
    )
  out += [
      '',
      (
          '`table` is the hand-measured `minutes` in `.github/ci/shards.py`,'
          ' which orders the job list. `actual` is this run, and is blank for'
          ' any shard whose jobs did not all succeed: a cancelled job stopped'
          ' early, so its duration is how far it got, not what the shard costs.'
      ),
  ]
  return out


def main(argv: Sequence[str] | None = None) -> int:
  """Entry point.

  Args:
    argv: Unused; present so this matches the other scripts here.

  Returns:
    Zero. This reports and must never be the reason a run goes red: the
    verdict is the `Check results` step, and a summary that failed the job
    would turn a green run red over a table.
  """
  del argv
  print('\n'.join(render(parse_jobs(sys.stdin.readlines()))))
  return 0


if __name__ == '__main__':
  sys.exit(main())
