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

from absl import flags
from absl.testing import absltest
from tokamax._src import config


enum_option = config._ConfigOption(
    flags.DEFINE_enum(
        "enum_option", "optA", ("optA", "optB", "optC"), "Enum option."
    )
)


class ConfigTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()
