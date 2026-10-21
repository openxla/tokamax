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
"""TPU export coverage independent of the numerical interpreter tests."""

import jax
import jax.numpy as jnp
import pytest
from tokamax._src.ops.kda import inference


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("final", [False, True])
def test_tpu_export(initial, final, monkeypatch):
    """Exercise real TPU lowering on CPU, independently of interpretation."""
    monkeypatch.setenv("PALLAS_INTERPRET", "0")
    from jax import export

    q = jax.ShapeDtypeStruct((1, 128, 2, 128), jnp.bfloat16)
    beta = jax.ShapeDtypeStruct((1, 128, 2), jnp.bfloat16)
    ids = jax.ShapeDtypeStruct((1, 128), jnp.int32)
    h0 = jax.ShapeDtypeStruct((1, 3, 2, 128, 128), jnp.float32) if initial else None

    def forward(q, beta, ids, h0):
        return inference.kimi_delta_attention_inference(
            q,
            q,
            q,
            q,
            beta,
            segment_ids=ids,
            initial_state=h0,
            output_final_state=final,
            a_log=jnp.zeros(2),
            delta_time_bias=jnp.zeros(256),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            max_num_segments=3,
        )

    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("device",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 2, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        exported = export.export(jax.jit(forward), platforms=("tpu",))(q, beta, ids, h0)
    assert "tpu_custom_call" in exported.mlir_module()
