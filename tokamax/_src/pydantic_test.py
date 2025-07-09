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
from absl.testing import parameterized
import pydantic
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.ops.attention import base as attn_base
from tokamax._src.ops.attention import bench_arg_specs as attn_arg_specs


class PydanticTest(parameterized.TestCase):

  def test_power_of_two(self):
    pow2 = pydantic.TypeAdapter(pydantic_lib.PowerOfTwo)
    pow2.validate_python(1)
    pow2.validate_python(2)
    pow2.validate_python(64)

    with self.assertRaises(pydantic.ValidationError):
      pow2.validate_python(0)

    with self.assertRaises(pydantic.ValidationError):
      pow2.validate_python(3)

    with self.assertRaises(pydantic.ValidationError):
      pow2.validate_python(-1)

  # TODO(cjfj): Add tests for other ops.
  @parameterized.named_parameters(
      ("attention", attn_base.DotProductAttention, attn_arg_specs.ARG_SPECS),
  )
  def test_arg_specs_roundtrip(self, op_cls, arg_specs):
    model = pydantic_lib.get_arg_spec_model("ArgSpec", op_cls().signature)
    for name, spec in arg_specs.items():
      with self.subTest(name):
        spec = model(**(spec() if callable(spec) else spec))
        self.assertEqual(spec, model.model_validate(spec.model_dump()))


if __name__ == "__main__":
  absltest.main()
