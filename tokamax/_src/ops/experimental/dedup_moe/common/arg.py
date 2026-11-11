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
"""Weight formats accepted by the De-Dup MoE layer.

The weight-format records (`WeightFormat`, `WeightForm`) are shared with
`fused_moe.host`, so both layers describe a format the same way. This layer
supports the subset that its fused kernel implements: fp8 W8A8 (fp8 weights
and token rows, per-channel weight scales, fp8 wire rows) and bf16 W16A16
(unquantized weights, bf16 token rows and wire rows).
"""

from tokamax._src.ops.experimental.fused_moe import host as fused_host

FP8 = fused_host.FP8
FP8_MAX = fused_host.FP8_MAX
BF16 = fused_host.BF16

WeightFormat = fused_host.WeightFormat
WeightForm = fused_host.WeightForm

# The formats this layer's kernel implements, in `fused_host.WEIGHT_FORMS`
# order. Other `WeightFormat` members are refused by name.
SUPPORTED_WEIGHT_FORMATS = (WeightFormat.FP8, WeightFormat.BF16)
WEIGHT_FORMS = {
    wf: fused_host.WEIGHT_FORMS[wf] for wf in SUPPORTED_WEIGHT_FORMATS
}
WEIGHT_FORMAT_NAMES = tuple(WEIGHT_FORMS)


def weight_form(weight_format: WeightFormat | str) -> WeightForm:
  """Returns the `WeightForm` record of a supported weight format.

  Args:
    weight_format: A `WeightFormat` member or its string spelling.

  Returns:
    The `WeightForm` that describes the format's element types and scales.

  Raises:
    NotImplementedError: If this layer does not support the format; the
      message lists the supported formats.
  """
  try:
    return WEIGHT_FORMS[WeightFormat(weight_format)]
  except (KeyError, ValueError):
    raise NotImplementedError(
        "the De-Dup MoE kernel takes weight formats "
        f"{tuple(str(f) for f in WEIGHT_FORMAT_NAMES)}; got {weight_format!r}"
    ) from None


def weight_format_of_dtype(dtype) -> WeightFormat | None:
  """Returns the supported weight format whose weights have this dtype.

  Args:
    dtype: A weight element type, such as `jnp.float8_e4m3fn`.

  Returns:
    The matching `WeightFormat` member, or None if no supported format stores
    weights in `dtype`.
  """
  wf = fused_host.weight_format_of_dtype(dtype)
  return wf if wf in WEIGHT_FORMS else None
