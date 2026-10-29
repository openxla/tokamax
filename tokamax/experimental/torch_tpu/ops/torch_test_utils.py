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
"""Test-only Torch Tokamax utility functions."""

import jax
from jax import numpy as jnp
import numpy as np
import torch


def torch_to_jax(tensor: torch.Tensor) -> jax.Array:
  """Converts a PyTorch tensor to a JAX array via CPU copy (for test use only)."""
  if tensor.dtype == torch.bfloat16:
    return jnp.asarray(
        tensor.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
  return jnp.asarray(tensor.detach().cpu().numpy())


def jax_to_torch(array: jax.Array, device: str = "tpu") -> torch.Tensor:
  """Converts a JAX array to a PyTorch tensor on `device` via CPU copy (for test use only)."""
  if array.dtype == jnp.bfloat16:
    return torch.as_tensor(
        np.asarray(array, dtype=np.float32),
        dtype=torch.bfloat16,
        device=device,
    )
  return torch.as_tensor(np.asarray(array), device=device)
