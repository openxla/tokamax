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
"""Tokamax test utilities."""

from absl.testing import parameterized


def get_names_and_params(
    test_case: type[parameterized.TestCase], test_name: str
) -> list[tuple[str, str]]:
  """Returns the names and parameters of a parameterized test case.

  In Tokamax's test suite, this is used to override a test's behaviour from a
  child test, while keeping the same arguments and names of the original test.

  Args:
    test_case: Parent test case inheriting from `parameterized.TestCase`.
    test_name: The original name of the test.

  Example:
  ```
  class ExampleTestBase(parameterized.TestCase):
    @parameterized.parameters(1, 2, 3)
    def test_foo(self, foo):
      ...
    @parameterized.parameters("a", "b", "c")
    def test_bar(self, bar):
      ...

  if __name__ == "__main__":
    assert test_utils.get_names_and_params(ExampleTestBase, "test_foo") == [
        ("test_foo0", "(1)"),
        ("test_foo1", "(2)"),
        ("test_foo2", "(3)"),
    ]
  ```
  """
  if not hasattr(test_case, "_test_params_reprs"):
    raise ValueError(
        f'Test case {test_case} does not have "_test_params_reprs" attribute.'
    )
  return [
      (k, v)
      for k, v in test_case._test_params_reprs.items()  # pylint: disable=protected-access
      if k.startswith(test_name) and k[len(test_name)].isdigit()
  ]
