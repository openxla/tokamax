#!/bin/bash
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
# Installs Tokamax and its test dependencies into the CI container's Python.
#
# Shared by every job that runs Tokamax tests in the CI container, so that the
# install, and how it copes with a flaky network, is defined in one place.
#
# Usage: install_deps.sh <device extra> <jax version>

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <device extra> <jax version>" >&2
  exit 2
fi
DEVICE="$1"
JAX_PIN="$2"

# The uv shipped in the CI image (0.5.31) only retries a reset connection. A
# PyPI connection that drops mid-request ("stream closed because of a broken
# pipe") fails the install on the first try, and `fail-fast` then cancels every
# other shard in the run. Later releases retry broken pipes and any HTTP/2
# error, so install a pinned newer uv first.
UV_VERSION=0.12.18
readonly UV_VERSION

# Runs a command that downloads from PyPI, retrying it if it fails. This is a
# backstop for drops that even a recent uv gives up on, such as a short PyPI
# outage. Installs take seconds, so a retry costs little.
retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    if (( attempt < 3 )); then
      echo "::warning::Attempt ${attempt}/3 failed, retrying: $*"
      sleep $(( attempt * 10 ))
    fi
  done
  echo "::error::Failed after 3 attempts: $*"
  return 1
}

echo "Installing dependencies on device ${DEVICE}, jax ${JAX_PIN}"

# The uv cache and the container's /usr are on different filesystems, so uv
# cannot hardlink and warns on every install. Copying is what it falls back to.
export UV_LINK_MODE=copy

# This download still goes through the old uv, hence the retry.
retry python3.12 -m uv pip install "uv==${UV_VERSION}"
got_uv=$(python3.12 -m uv --version)
[[ "${got_uv}" == "uv ${UV_VERSION}"* ]] || {
  echo "::error::asked for uv ${UV_VERSION}, got ${got_uv}"; exit 1; }
echo "Using ${got_uv}"

# Pin `jax` only. Jax has some sort of metadata that can find the corresponding
# jaxlib and either libtpu or CUDA plugin, and will install the
# appropriate version for us, rather than us having to pin it separately.
printf 'jax==%s\n' "${JAX_PIN}" > /tmp/jax-pin.txt
retry python3.12 -m uv pip install --upgrade pip
retry python3.12 -m uv pip install --constraint /tmp/jax-pin.txt \
  -e ".[${DEVICE},test]"
python3.12 -m uv pip freeze
got=$(python3.12 -c 'import jax; print(jax.__version__)')
[[ "${got}" == "${JAX_PIN}" ]] || {
  echo "::error::asked for jax ${JAX_PIN}, got ${got}"; exit 1; }
