# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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

import os
from typing import Literal
from typing import Sequence

from absl import logging
from tensorboardX import writer

SummaryWriter = writer.SummaryWriter


def write_tensorboard_logs(
    tensorboard_output: str, value: float | Sequence[float], metric_tag: str
) -> None:
  """Writes TensorBoard logs.

  Args:
    tensorboard_output: The TensorBoard `tensorboard_output_env_var` environment
      variable.
    value: The value to write to TensorBoard.
    metric_tag: The metric tag to use for TensorBoard.
  """

  tblog_dir = os.environ.get(tensorboard_output)
  value = [value] if isinstance(value, float) else value

  if tblog_dir:
    try:
      tb_writer = SummaryWriter(log_dir=tblog_dir)
      for i, value in enumerate(value):
        tb_writer.add_scalar(metric_tag, value, global_step=i)

      tb_writer.close()
    except (OSError, IOError):
      logging.exception('Error writing TensorBoard logs')
  else:
    logging.info(
        'metric_tag: %s, value: %s',
        metric_tag,
        str(value),
    )
