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

import sys
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from tokamax._src import config


enum_option = config._ConfigOption(
    flags.DEFINE_enum(
        "enum_option", "optA", ("optA", "optB", "optC"), "Enum option."
    )
)


def _host_flag_values() -> flags.FlagValues:
  """Returns flags mimicking a host that has absl's logging flags registered."""
  flag_values = flags.FlagValues()
  flags.DEFINE_integer(
      "verbosity", 0, "Host verbosity.", short_name="v", flag_values=flag_values
  )
  flags.DEFINE_bool(
      "tokamax_test_option", False, "Tokamax option.", flag_values=flag_values
  )
  return flag_values


class ConfigTest(parameterized.TestCase):

  def test_config_option_scope(self):
    self.assertEqual(enum_option.value, "optA")
    with enum_option("optB"):
      self.assertEqual(enum_option.value, "optB")
      with enum_option("optC"):
        self.assertEqual(enum_option.value, "optC")
      self.assertEqual(enum_option.value, "optB")
    self.assertEqual(enum_option.value, "optA")

  def test_config_option_validation(self):
    self.assertEqual(enum_option.value, "optA")
    with self.assertRaisesRegex(ValueError, "Invalid value"):
      with enum_option("optD"):
        pass

  @parameterized.named_parameters(
      ("pytest_args", ["-s", "-v", "-x", "tests/foo_test.py"], []),
      (
          "vllm_args",
          ["serve", "model", "-tp", "8", "-O3"],
          [],
      ),
      (
          "value_in_same_arg",
          ["-v", "-x", "--tokamax_cross_compile=true"],
          ["--tokamax_cross_compile=true"],
      ),
      (
          "value_in_next_arg",
          ["-x", "--tokamax_autotuning_cache_miss_fallback", "error", "foo.py"],
          ["--tokamax_autotuning_cache_miss_fallback", "error"],
      ),
      (
          "consecutive_bool_options",
          [
              "--tokamax_cross_compile",
              "--tokamax_ignore_autotuning_cache",
              "-v",
          ],
          ["--tokamax_cross_compile", "--tokamax_ignore_autotuning_cache"],
      ),
      (
          "negated_bool_option",
          ["-v", "-x", "--notokamax_cross_compile"],
          ["--notokamax_cross_compile"],
      ),
  )
  def test_tokamax_argv(self, args, expected):
    with mock.patch.object(sys, "argv", ["prog", *args]):
      self.assertEqual(config._tokamax_argv(), ["prog", *expected])

  def test_option_value_with_unparsed_host_flags(self):
    # `sys.argv` belongs to the host program, which may pass arguments that absl
    # misreads as its own: here absl takes `-x` as the value of `--verbosity`
    # flag that `absl.logging` registers, and fails to parse it as an integer.
    argv = ["prog", "-s", "-v", "-x", "--tokamax_test_option=true"]
    with self.assertRaises(flags.IllegalFlagValueError):
      _host_flag_values()(argv, known_only=True)

    flag_values = _host_flag_values()
    with mock.patch.object(config.flags, "FLAGS", flag_values):
      with mock.patch.object(sys, "argv", argv):
        _ = enum_option.value  # Must not raise.
    self.assertTrue(flag_values.is_parsed())
    self.assertTrue(flag_values.tokamax_test_option)


if __name__ == "__main__":
  absltest.main()
