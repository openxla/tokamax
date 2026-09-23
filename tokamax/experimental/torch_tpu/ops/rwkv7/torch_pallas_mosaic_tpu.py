# Copyright 2026 Google LLC. All Rights Reserved.
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

"""PyTorch interface for Pallas TPU RWKV-7 operator."""

from typing import Any
import jax
import jax.numpy as jnp
import torch
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu as jax_pallas
from tokamax.experimental.torch_tpu.ops.rwkv7 import torch_base
from typing_extensions import override


class _PallasTpuRWKV7(torch_base._RWKV7):
  """PyTorch wrapper for Pallas TPU RWKV-7 operator."""

  def __init__(self):
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuRWKV7()
    self.jax_op_name = "torch_tpu_pallas_rwkv7"
    try:
      from torch_tpu._internal import pallas
      self._torch_tokamax_op = pallas.jax_op(self.jax_op_name, self.op_impl_call)
      self._setup_autograd(pallas)
    except Exception:
      pass

  def _setup_autograd(self, pallas_module):
    def bwd_jax(q, k, v, alpha, beta, gk, do, chunk_size=64):
      def loss_fn(q_, k_, v_, a_, b_, g_):
        (out, _), _ = self.op_impl_jax._fwd(
            q_[None], k_[None], v_[None], a_[None], b_[None], g_[None], chunk_size=chunk_size
        )
        return jnp.sum(out[0].astype(jnp.float32) * do.astype(jnp.float32))

      return jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, alpha, beta, gk)

    bwd_op = pallas_module.jax_op(f"{self.jax_op_name}_bwd", bwd_jax)

    def setup_context(ctx, inputs, output):
      del output
      ctx.save_for_backward(*inputs[:6])
      ctx.chunk_size = inputs[6] if len(inputs) > 6 else 64

    def backward(ctx, do):
      saved = ctx.saved_tensors
      return bwd_op(*saved, do, chunk_size=ctx.chunk_size)

    self._torch_tokamax_op.register_autograd(backward, setup_context=setup_context)


# Singleton instance
rwkv7 = _PallasTpuRWKV7()
