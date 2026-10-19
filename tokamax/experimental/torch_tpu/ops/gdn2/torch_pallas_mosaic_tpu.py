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

"""PyTorch interface for Pallas TPU GDN-2 operator."""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.gdn2 import pallas_mosaic_tpu as jax_pallas
from tokamax.experimental.torch_tpu.ops.gdn2 import torch_base


class _PallasTpuGatedDeltaNet2(torch_base._GatedDeltaNet2):
  """PyTorch wrapper for Pallas TPU GDN-2 operator."""

  def __init__(self):
    super().__init__()
    self.op_impl_jax = jax_pallas.PallasTpuGatedDeltaNet2()
    self.jax_op_name = "torch_tpu_pallas_gdn2"
    self._torch_tokamax_op = None
    try:
      from torch_tpu._internal import pallas
    except ImportError:
      pallas = None
    else:
      try:
        self._torch_tokamax_op = pallas.jax_op(self.jax_op_name, self.op_impl_call)
        self._setup_autograd(pallas)
      except (ImportError, AttributeError, RuntimeError):
        self._torch_tokamax_op = None

  def _setup_autograd(self, pallas_module):
    def bwd_jax(q, k, v, g, b, w, do, chunk_size=64):
      def loss_fn(q_, k_, v_, g_, b_, w_):
        (out, _), _ = self.op_impl_jax._fwd(
            q_, k_, v_, g_, b_, w_, chunk_size=chunk_size
        )
        return jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32))

      return jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, b, w)

    bwd_op = pallas_module.jax_op(f"{self.jax_op_name}_bwd", bwd_jax)

    def setup_context(ctx, inputs, output):
      del output
      ctx.save_for_backward(*inputs[:6])
      ctx.has_chunk_size_arg = len(inputs) > 6
      ctx.chunk_size = inputs[6] if ctx.has_chunk_size_arg else 64

    def backward(ctx, do):
      saved = ctx.saved_tensors
      grads = bwd_op(*saved, do, chunk_size=ctx.chunk_size)
      if ctx.has_chunk_size_arg:
        return (*grads, None)
      return grads

    self._torch_tokamax_op.register_autograd(backward, setup_context=setup_context)


# Singleton instance
gdn2 = _PallasTpuGatedDeltaNet2()
