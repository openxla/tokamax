# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

"""Stamps release metadata into `tokamax/_src/version.py` before building.

Run by `.github/workflows/publish.yml`. Edits the working tree only; nothing is
committed, so the default branch keeps its stable version.

Environment variables:
  NIGHTLY: "true" for a nightly build. Anything else builds the stable version
    declared in `version.py`.
  TESTED_SHA: the commit CI-Nightly tested; empty on a manual release. When set,
    it must match HEAD or the build is aborted.
"""

import datetime
import os
import pathlib
import re
import subprocess
import sys

_VERSION_FILE = (
    pathlib.Path(__file__).resolve().parents[2] / "tokamax/_src/version.py"
)
_VERSION_RE = re.compile(r'TOKAMAX_VERSION: Final\[str\] = "([^"]+)"')
_UNSTAMPED_REVISION = 'TOKAMAX_GIT_REVISION: Final[str] = ""'


def _next_dev_version(current: str) -> str:
  """Returns `<next version>.dev<YYYYMMDD>` for `current`.

  PEP 440 sorts a dev release before its own base version, so the base must be
  the *next* version for a nightly to sort after the current release.
  """
  major, minor, patch = (int(part) for part in current.split("."))
  stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
  return f"{major}.{minor}.{patch + 1}.dev{stamp}"


def main() -> None:
  head = subprocess.run(
      ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
  ).stdout.strip()

  tested = os.environ.get("TESTED_SHA", "")
  if tested and head != tested:
    sys.exit(f"building {head} but CI-Nightly tested {tested}")

  src = _VERSION_FILE.read_text()
  match = _VERSION_RE.search(src)
  if match is None:
    sys.exit(f"TOKAMAX_VERSION not found in {_VERSION_FILE}")
  current = match.group(1)

  if os.environ.get("NIGHTLY") == "true":
    version = _next_dev_version(current)
    src = src.replace(
        f'TOKAMAX_VERSION: Final[str] = "{current}"',
        f'TOKAMAX_VERSION: Final[str] = "{version}"',
        1,
    )
    print(f"version  -> {version} (was {current})")
  else:
    print(f"version  -> {current} (stable, unchanged)")

  stamped = src.replace(
      _UNSTAMPED_REVISION, f'TOKAMAX_GIT_REVISION: Final[str] = "{head}"', 1
  )
  if stamped == src:
    sys.exit(f"{_UNSTAMPED_REVISION!r} not found in {_VERSION_FILE}")

  print(f"revision -> {head}")
  print(f"tested   -> {tested or '(manual dispatch)'}")

  _VERSION_FILE.write_text(stamped)


if __name__ == "__main__":
  main()
