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
"""Inactive backward chunks clear their carried state without reading operands."""
import inspect
from types import SimpleNamespace
import numpy as np
import pytest
from tokamax._src.ops.kda import pallas_mosaic_tpu_bwd_kernel as kernels

@pytest.mark.parametrize("poison", [np.nan, np.inf])
def test_inactive_chunk_clears_carry_and_outputs(monkeypatch, poison):
  kernel = kernels._fused_dhu_wy_intra_cumsum_kernel
  kwargs = {name: None for name, param in inspect.signature(kernel).parameters.items()
            if param.default is inspect.Parameter.empty}
  kwargs.update(step=SimpleNamespace(index=(0, 0, 0)), BT=64, K=128,
                V=128, NT=1, MB=1, scale=1.0, is_varlen=True)
  refs = [np.full((1, 1, 1, 64, 128), poison, np.float32) for _ in range(5)]
  kwargs.update(zip(("dq_ref", "dk_ref", "dv_ref", "db_ref", "dg_ref"), refs))
  carry = np.full((1, 128, 128), poison, np.float32)
  kwargs["dh_scratch"] = carry
  monkeypatch.setattr(kernels, "_chunk_segment_metadata",
                      lambda *args: (0, 0, False, False, False))
  monkeypatch.setattr(kernels.jax.lax, "cond",
                      lambda pred, yes, no, arg: (yes if pred else no)(arg))
  kernel(**kwargs)
  for ref in [carry, *refs]:
    np.testing.assert_array_equal(ref, 0)
