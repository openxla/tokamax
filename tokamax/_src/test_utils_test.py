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

import functools
import random
import unittest
from unittest import mock
from absl.testing import absltest
import jax
from tokamax._src import test_utils


@test_utils.with_config(jax_numpy_dtype_promotion="strict")
class _DummySkipTestCase(absltest.TestCase):
  __test__ = False

  def setUp(self):
    super().setUp()
    self.skipTest("Intentionally skipped to test cleanup")

  def test_dummy(self):
    pass


class TestUtilsTest(absltest.TestCase):

  def test_assert_trees_all_close(self):
    epsilon = 1e-6
    strict_atol = 1e-5
    relaxed_atol = 1e-4
    relaxed_fraction = 0.01

    assert_trees_all_close = functools.partial(
        test_utils.assert_trees_all_close,
        rtol=0.0,
        atol={(1.0 - relaxed_fraction): strict_atol, 1.0: relaxed_atol},
    )

    x = jax.random.normal(jax.random.PRNGKey(42), shape=(256, 256))
    y = x + strict_atol - epsilon
    z = x - strict_atol + epsilon

    # Generate 1% of random indices. Use a set to avoid duplicate indices.
    idx = set()
    while len(idx) < int(x.size * relaxed_fraction):
      idx.add((random.randint(0, 255), random.randint(0, 255)))

    for i in idx:
      y = y.at[*i].set(y[*i] - strict_atol + relaxed_atol)
      z = z.at[*i].set(z[*i] + strict_atol - relaxed_atol)

    assert_trees_all_close(x, y, z)

    # Add another value within relaxed tolerance to make it >1% and fail.
    while True:
      rand_item = (random.randint(0, 255), random.randint(0, 255))
      if rand_item not in idx:
        break

    y = y.at[*rand_item].set(y[*rand_item] - strict_atol + relaxed_atol)
    z = z.at[*rand_item].set(z[*rand_item] + strict_atol - relaxed_atol)

    with self.assertRaises(AssertionError):
      assert_trees_all_close(x, y, z)

  def test_get_tpu_version_mocked(self):
    cases = (
        ("TPU v4", 4),
        ("TPU v5e", 5),
        ("TPU v5p", 5),
        ("TPU v6e", 6),
        ("TPU7x", 7),
        ("TPU8i", 8),
        ("TPU v5e lite", 5),
        ("TPU6e", 6),
        ("NVIDIA A100", -1),
        ("CPU", -1),
    )
    for kind, expected_version in cases:
      with self.subTest(kind=kind):
        mock_device = mock.MagicMock()
        mock_device.device_kind = kind
        with mock.patch.object(jax, "devices", return_value=[mock_device]):
          self.assertEqual(test_utils.get_tpu_version(), expected_version)

  def test_with_config_restores_on_skip(self):
    original_val = getattr(jax.config, "jax_numpy_dtype_promotion", "standard")
    suite = unittest.TestLoader().loadTestsFromTestCase(_DummySkipTestCase)
    result = unittest.TestResult()
    suite.run(result)
    self.assertEqual(len(result.skipped), 1)
    current_val = getattr(jax.config, "jax_numpy_dtype_promotion", "standard")
    self.assertEqual(current_val, original_val)


if __name__ == "__main__":
  absltest.main()
