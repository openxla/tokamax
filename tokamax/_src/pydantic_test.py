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
from absl.testing import absltest
import pydantic
from tokamax._src import pydantic as pydantic_lib


class PydanticTest(absltest.TestCase):

  def test_power_of_two(self):
    pydantic.TypeAdapter(pydantic_lib.PowerOfTwo).validate_python(1)
    pydantic.TypeAdapter(pydantic_lib.PowerOfTwo).validate_python(2)
    pydantic.TypeAdapter(pydantic_lib.PowerOfTwo).validate_python(64)

    with self.assertRaises(pydantic.ValidationError):
      pydantic.TypeAdapter(pydantic_lib.PowerOfTwo).validate_python(0)

    with self.assertRaises(pydantic.ValidationError):
      pydantic.TypeAdapter(pydantic_lib.PowerOfTwo).validate_python(3)

    with self.assertRaises(pydantic.ValidationError):
      pydantic.TypeAdapter(pydantic_lib.PowerOfTwo).validate_python(-1)


if __name__ == "__main__":
  absltest.main()
