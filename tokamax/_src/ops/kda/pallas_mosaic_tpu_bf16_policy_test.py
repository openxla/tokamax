# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Acceptance rules shared with the internal BF16 backward tests."""

import jax.numpy as jnp
import pytest
from tokamax._src.ops.kda.pallas_mosaic_tpu_fwd_fused_test import _assert_close


def test_historical_one_ulp_difference():
  a = jnp.array([-1.0390625], jnp.bfloat16)
  b = jnp.array([-1.046875], jnp.bfloat16)
  _assert_close(a, b, tolerance=0.002, bf16_internal=True)
  _assert_close(a, b, tolerance=0, bf16_ulp_only=True)
  with pytest.raises(AssertionError):
    _assert_close(a, b, tolerance=0)


def test_ulp_only_rejects_two_ulps():
  with pytest.raises(AssertionError):
    _assert_close(
        jnp.array([1.015625], jnp.bfloat16),
        jnp.array([1.0], jnp.bfloat16),
        tolerance=0,
        bf16_ulp_only=True,
    )


def test_backward_policy_rejects_excess_error():
  with pytest.raises(AssertionError):
    _assert_close(
        jnp.array([1.25], jnp.bfloat16),
        jnp.array([1.0], jnp.bfloat16),
        tolerance=0.002,
        bf16_internal=True,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_is_rejected(value):
  with pytest.raises(AssertionError):
    _assert_close(
        jnp.array([value], jnp.bfloat16),
        jnp.array([value], jnp.bfloat16),
        tolerance=0.002,
        bf16_internal=True,
    )


def test_fp32_rule_is_unchanged():
  with pytest.raises(AssertionError):
    _assert_close(
        jnp.array([1.01], jnp.float32),
        jnp.array([1.0], jnp.float32),
        tolerance=0.002,
        bf16_internal=True,
    )
