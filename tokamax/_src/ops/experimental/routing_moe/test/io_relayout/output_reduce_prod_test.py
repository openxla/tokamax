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
"""Production-shape numerics and benchmarks for output_reduce_unpack."""

import dataclasses

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from tokamax._src.ops.experimental.routing_moe.io_relayout import output_reduce as kernel
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_reference
from tokamax._src.ops.experimental.routing_moe.test.common import test_util
from tokamax._src.ops.experimental.routing_moe.test.io_relayout import output_reduce_test_lib

jax.config.parse_flags_with_absl()


class OutputReduceProdTest(parameterized.TestCase):

  def setUp(self):
    test_util.skip_unless_supported(self)
    super().setUp()

  @parameterized.product(case=test_cases.PROD_CASES, ep_size=[1, 8])
  def test_prod_cases(self, case: test_cases.Case, ep_size: int):
    case_inputs = output_reduce_test_lib.ReduceCase.make_case(
        dataclasses.replace(case, ep_size=ep_size), seed=54321
    )
    xla = test_util.Timed.create(
        test_reference.xla_output_reduce_unpack,
        case_inputs.rows_clean_compact,
        case_inputs.topk_weights,
        case_inputs.topk_indices,
        case_inputs.group_offset.reshape(()),
        num_local_experts=case_inputs.num_local_experts,
    )
    timed_kernel = test_util.Timed.create(
        kernel.output_reduce_unpack,
        case_inputs.rows_compact,
        case_inputs.topk_weights,
        case_inputs.topk_indices,
        case_inputs.group_offset,
        num_local_experts=case_inputs.num_local_experts,
    )
    case_inputs.assert_close(timed_kernel.fn(timed_kernel.args))
    logging.info("xla: %.1f us", xla.latency_us())
    logging.info(
        "kernel: %.1f us %s", timed_kernel.latency_us(), timed_kernel.trace()
    )


if __name__ == "__main__":
  absltest.main()
