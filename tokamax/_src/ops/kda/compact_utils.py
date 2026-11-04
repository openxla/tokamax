# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Small shape utilities shared by the native compact kernel port."""

import jax.numpy as jnp
from tokamax._src.ops.kda.utils import align_up, prepare_chunk_indices


def assert_shape(value, shape, name="value"):
  assert value.shape == tuple(
      shape
  ), f"{name}: expected {tuple(shape)}, got {value.shape}"


def assert_shape_or_none(value, shape, name="value"):
  if value is not None:
    assert_shape(value, shape, name)


def compute_padded_cu_seqlens(cu_seqlens, chunk_size):
  lengths = jnp.diff(cu_seqlens, axis=-1)
  padded = ((lengths + chunk_size - 1) // chunk_size) * chunk_size
  return jnp.concatenate(
      (jnp.zeros_like(cu_seqlens[..., :1]), jnp.cumsum(padded, axis=-1)),
      axis=-1,
  )
