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
"""Numerical tests for the experimental Pallas TPU KDA implementation."""

from collections.abc import Sequence
import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import api
from tokamax._src.ops.experimental.kda.cp_utils import CPContext


def compute_ulp(
    x: np.ndarray, dtype: jax.typing.DTypeLike
) -> np.ndarray:
  """Computes one unit in the last place for `dtype` at each value in `x`."""
  finfo = jnp.finfo(dtype)
  mantissa_bits = round(-math.log2(float(finfo.eps)))
  min_ulp = float(finfo.tiny) * float(finfo.eps)

  abs_x = np.abs(x).astype(np.float64)
  _, exponent = np.frexp(abs_x)
  ulp = np.ldexp(1.0, exponent.astype(np.int32) - mantissa_bits - 1)
  ulp = np.maximum(ulp, min_ulp)
  return np.where(abs_x == 0, min_ulp, ulp)


def compare_tensor(
    name: str,
    expected: jax.Array | np.ndarray | None,
    actual: jax.Array | np.ndarray | None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_ulp: int = 1,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    compare_dtype: jax.typing.DTypeLike = np.float64,
) -> bool:
  """Compares two tensors and prints focused numerical diagnostics."""
  if expected is None and actual is None:
    print(f"[{name}] Both are None. PASS.")
    return True
  if expected is None or actual is None:
    print(f"[{name}] One is None. FAIL.")
    return False

  expected_dtype = np.dtype(expected.dtype).name
  actual_dtype = np.dtype(actual.dtype).name
  if expected_dtype != actual_dtype:
    print(
        f"[{name}] Dtype mismatch: Left {expected.dtype} vs "
        f"Right {actual.dtype}. FAIL."
    )
    return False

  expected_np = np.asarray(expected).astype(compare_dtype)
  actual_np = np.asarray(actual).astype(compare_dtype)
  if expected_np.shape != actual_np.shape:
    print(
        f"[{name}] Shape mismatch: Left {expected_np.shape} vs "
        f"Right {actual_np.shape}. FAIL."
    )
    if expected_np.squeeze().shape != actual_np.squeeze().shape:
      return False
    expected_np = expected_np.squeeze()
    actual_np = actual_np.squeeze()
    print(f"  Comparing squeezed shape: {expected_np.shape}")

  diff = np.abs(expected_np - actual_np)
  max_diff = np.max(diff)
  max_value = np.max(np.abs(expected_np))
  max_relative_diff = np.max(diff / (np.abs(expected_np) + 1e-12))
  is_close = np.allclose(
      actual_np,
      expected_np,
      atol=atol,
      rtol=rtol,
      equal_nan=True,
  )

  if not is_close:
    ulp = compute_ulp(
        np.maximum(np.abs(expected_np), np.abs(actual_np)), dtype
    )
    tolerance = np.maximum(
        atol + rtol * np.abs(expected_np), max_ulp * ulp
    )
    is_close = bool(np.all(diff <= tolerance))

  print(f"[{name}] {'PASS' if is_close else 'FAIL'}")
  print(f"  Max Value        : {max_value:.6e}")
  print(f"  Max Abs Diff     : {max_diff:.6e}")
  print(f"  Max Rel Diff     : {max_relative_diff:.6e}")

  if not is_close:
    error_ratio = diff / (tolerance + 1e-12)
    index = np.unravel_index(np.argmax(error_ratio), error_ratio.shape)
    print(f"  Max Mismatch details at index {index}:")
    print(f"    Left (expected) = {expected_np[index]}")
    print(f"    Right (actual)  = {actual_np[index]}")
    print(f"    Diff            = {diff[index]}")
    print(f"    Tolerance       = {tolerance[index]}")
    print(f"    ULP diff        = {diff[index] / ulp[index]:.2f}")
    print(f"    Ratio           = {error_ratio[index]}")

  return is_close


@dataclasses.dataclass(frozen=True)
class TestConfig:
  __test__ = False

  name: str
  seq_len: int
  heads: int
  batch_size: int = 1
  seq_lens: tuple[int, ...] | None = None
  batch_seq_lens: tuple[tuple[int, ...], ...] | None = None
  n_max_override: int | None = None
  cp_size: int = 1
  key_dim: int = 128
  value_dim: int = 128
  dtype: jax.typing.DTypeLike = jnp.bfloat16
  # Test-only preset for the synthetic input distribution used by _make_inputs.
  # It changes numerical coverage, not the attention implementation under test.
  input_profile: str = "model"
  seed: int = 42
  scale: float | None = None
  gate_logit_normalizer: float = 1.0
  gate_mask_probability: float = 0.0
  use_initial_state: bool = False
  output_final_state: bool = False
  use_qk_l2norm_in_kernel: bool = False
  use_gate_in_kernel: bool = False
  safe_gate: bool = True
  lower_bound: float | None = None
  disable_recompute: bool = True
  forward_check: str = "reference"
  backward_check: str = "reference"
  forward_atol: float = 0.05
  forward_rtol: float = 0.05
  state_atol: float | None = None
  state_rtol: float | None = None
  backward_atol: float = 0.05
  backward_rtol: float = 0.05
  long: bool = False

  @property
  def layouts(self) -> tuple[tuple[int, ...], ...] | None:
    if self.seq_lens is not None and self.batch_seq_lens is not None:
      raise ValueError("Only one of seq_lens and batch_seq_lens may be set.")
    if self.batch_seq_lens is not None:
      return self.batch_seq_lens
    if self.seq_lens is not None:
      return (self.seq_lens,)
    return None

  @property
  def batch(self) -> int:
    return len(self.layouts) if self.layouts is not None else self.batch_size

  @property
  def n_max(self) -> int | None:
    if self.layouts is None:
      return None
    real_n_max = max(len(seq_lens) for seq_lens in self.layouts)
    return max(real_n_max, self.n_max_override or 0)

  @property
  def effective_scale(self) -> float:
    return self.scale if self.scale is not None else self.key_dim**-0.5


@dataclasses.dataclass(frozen=True)
class _Inputs:
  q: jax.Array
  k: jax.Array
  v: jax.Array
  g: jax.Array
  beta: jax.Array
  A_log: jax.Array
  dt_bias: jax.Array
  initial_state: jax.Array | None
  segment_ids: jax.Array | None
  dout: jax.Array
  dfinal_state: jax.Array | None


def make_test_config(
    name: str,
    *,
    T: int,
    H: int,
    B: int = 1,
    D: int | None = None,
    K: int | None = None,
    V: int | None = None,
    seq_lens: tuple[int, ...] | None = None,
    batch_seq_lens: tuple[tuple[int, ...], ...] | None = None,
    **kwargs,
) -> TestConfig:
  """Builds a case using the batch-first names from pallas-kernel tests."""
  if D is not None:
    K = D if K is None else K
    V = D if V is None else V
  if K is None or V is None:
    raise ValueError("Specify D or both K and V.")
  if batch_seq_lens is not None:
    B = len(batch_seq_lens)
  return TestConfig(
      name=name,
      seq_len=T,
      heads=H,
      batch_size=B,
      seq_lens=seq_lens,
      batch_seq_lens=batch_seq_lens,
      key_dim=K,
      value_dim=V,
      **kwargs,
  )


def _chunk_api_case(name: str, **cfg) -> TestConfig:
  """Ports one test_chunk_kda_tpu.py public-API configuration."""
  dtype = cfg.pop("dtype", jnp.float32)
  has_h0 = cfg.pop("has_h0", True)
  is_bf16 = dtype == jnp.bfloat16
  masked_gate = (
      cfg.get("gate_mask_probability", 0.0) > 0
      or cfg.get("gate_logit_normalizer", 1.0) != 1.0
  )
  return make_test_config(
      f"chunk_api_{name}",
      dtype=dtype,
      input_profile="chunk_api",
      use_initial_state=has_h0,
      output_final_state=True,
      safe_gate=cfg.pop("safe_gate", False),
      disable_recompute=cfg.pop("disable_recompute", False),
      forward_atol=0.05 if is_bf16 else 5e-3,
      forward_rtol=0.05 if is_bf16 else 5e-3,
      backward_atol=0.05 if is_bf16 else (2e-2 if masked_gate else 1e-2),
      backward_rtol=0.05 if is_bf16 else (2e-2 if masked_gate else 1e-2),
      **cfg,
  )

def _chunk_fwd_case(name: str, **cfg) -> TestConfig:
  """Ports a full-pipeline case from test_chunk_fwd_pallas.py."""
  dtype = cfg.pop("dtype", jnp.float32)
  is_bf16 = dtype == jnp.bfloat16
  return make_test_config(
      f"chunk_fwd_{name}",
      dtype=dtype,
      input_profile="chunk_fwd",
      safe_gate=False,
      disable_recompute=False,
      forward_atol=0.05 if is_bf16 else 2e-4,
      forward_rtol=0.05 if is_bf16 else 2e-4,
      backward_atol=0.05 if is_bf16 else 1e-3,
      backward_rtol=0.05 if is_bf16 else 1e-3,
      **cfg,
  )


def _varlen_case(
    name: str,
    *,
    seq_lens: tuple[int, ...] | None = None,
    batch_seq_lens: tuple[tuple[int, ...], ...] | None = None,
    T: int | None = None,
    H: int = 2,
    K: int = 128,
    V: int = 128,
    N_pad: int | None = None,
    dtype: jax.typing.DTypeLike = jnp.float32,
    use_initial_state: bool = False,
    input_profile: str = "model",
    **kwargs,
) -> TestConfig:
  """Builds a packed-sequence end-to-end case."""
  if (seq_lens is None) == (batch_seq_lens is None):
    raise ValueError("Specify exactly one sequence layout.")
  layouts = batch_seq_lens if batch_seq_lens is not None else (seq_lens,)
  assert layouts is not None
  if T is None:
    T = max(sum(layout) for layout in layouts)
  is_bf16 = dtype == jnp.bfloat16
  forward_atol = kwargs.pop("forward_atol", 0.05 if is_bf16 else 5e-4)
  forward_rtol = kwargs.pop("forward_rtol", 0.05 if is_bf16 else 5e-4)
  state_atol = kwargs.pop("state_atol", 0.05 if is_bf16 else 2e-3)
  state_rtol = kwargs.pop("state_rtol", 0.05 if is_bf16 else 2e-3)
  backward_atol = kwargs.pop("backward_atol", 0.05 if is_bf16 else 1e-2)
  backward_rtol = kwargs.pop("backward_rtol", 0.05 if is_bf16 else 5e-3)
  long = kwargs.pop("long", T >= 8192)
  return make_test_config(
      name,
      T=T,
      H=H,
      K=K,
      V=V,
      seq_lens=seq_lens,
      batch_seq_lens=batch_seq_lens,
      n_max_override=N_pad,
      dtype=dtype,
      input_profile=input_profile,
      use_initial_state=use_initial_state,
      output_final_state=True,
      forward_atol=forward_atol,
      forward_rtol=forward_rtol,
      state_atol=state_atol,
      state_rtol=state_rtol,
      backward_atol=backward_atol,
      backward_rtol=backward_rtol,
      long=long,
      **kwargs,
  )


def _cp_case(
    name: str,
    *,
    cp_size: int,
    T: int,
    H: int,
    seq_lens: tuple[int, ...] | None = None,
    batch_seq_lens: tuple[tuple[int, ...], ...] | None = None,
    dtype: jax.typing.DTypeLike = jnp.float32,
    input_profile: str = "cp",
    **kwargs,
) -> TestConfig:
  """Builds a context-parallel case from test_chunk_kda_cp.py."""
  if seq_lens is None and batch_seq_lens is None:
    seq_lens = (T,)
  layouts = batch_seq_lens if batch_seq_lens is not None else (seq_lens,)
  assert layouts is not None
  is_bf16 = dtype == jnp.bfloat16
  forward_atol = kwargs.pop("forward_atol", 0.05 if is_bf16 else 5e-4)
  forward_rtol = kwargs.pop("forward_rtol", 0.05 if is_bf16 else 5e-4)
  backward_atol = kwargs.pop("backward_atol", 0.05 if is_bf16 else 1e-3)
  backward_rtol = kwargs.pop("backward_rtol", backward_atol)
  long = kwargs.pop("long", T >= 8192)
  safe_gate = kwargs.pop("safe_gate", False)
  return make_test_config(
      name,
      T=T,
      H=H,
      D=128,
      seq_lens=seq_lens,
      batch_seq_lens=batch_seq_lens,
      n_max_override=max(len(layout) for layout in layouts),
      cp_size=cp_size,
      dtype=dtype,
      input_profile=input_profile,
      output_final_state=False,
      safe_gate=safe_gate,
      forward_atol=forward_atol,
      forward_rtol=forward_rtol,
      backward_atol=backward_atol,
      backward_rtol=backward_rtol,
      long=long,
      **kwargs,
  )

def _lengths_from_boundaries(boundaries: tuple[int, ...]) -> tuple[int, ...]:
  return tuple(right - left for left, right in zip(boundaries, boundaries[1:]))

def _jittered_lengths(T: int, n_segs: int, seed: int) -> tuple[int, ...]:
  rng = np.random.default_rng(seed)
  internal = sorted(
      int(x)
      for x in rng.choice(np.arange(1, T), size=n_segs - 1, replace=False)
  )
  return _lengths_from_boundaries((0, *internal, T))

# Active end-to-end configurations from test_chunk_kda_tpu.py. Commented-out
# gate-kernel cases in that file intentionally remain excluded.
_CASES = [
    _chunk_api_case("b1_t128_h2_d16_fp32", B=1, T=128, H=2, D=16),
    _chunk_api_case("b1_t256_h2_d32_fp32", B=1, T=256, H=2, D=32),
    _chunk_api_case("b1_t256_h2_d64_fp32", B=1, T=256, H=2, D=64),
    _chunk_api_case("b2_t128_h4_d32_fp32", B=2, T=128, H=4, D=32),
    _chunk_api_case("b4_t128_h4_d32_fp32", B=4, T=128, H=4, D=32),
    _chunk_api_case("b2_t64_h4_k16_v32_fp32", B=2, T=64, H=4, K=16, V=32),
    _chunk_api_case("b1_t128_h2_k128_v64_fp32", B=1, T=128, H=2, K=128, V=64),
    _chunk_api_case("b1_t128_h2_k64_v128_fp32", B=1, T=128, H=2, K=64, V=128),
    _chunk_api_case("single_chunk_fp32", B=1, T=64, H=2, D=64),
    _chunk_api_case("single_head_fp32", B=2, T=128, H=1, D=32),
    _chunk_api_case("many_chunks_fp32", B=1, T=512, H=2, D=32),
    _chunk_api_case("b1_t256_h2_d64_fp32_no_h0", B=1, T=256, H=2, D=64, has_h0=False),
    _chunk_api_case("b1_t64_h2_d64_fp32_no_h0", B=1, T=64, H=2, D=64, has_h0=False),
    _chunk_api_case("b1_t256_h2_d64_bf16", B=1, T=256, H=2, D=64, dtype=jnp.bfloat16),
    _chunk_api_case("b1_t128_h2_k128_v64_bf16", B=1, T=128, H=2, K=128, V=64, dtype=jnp.bfloat16),
    _chunk_api_case("normalizer8_mask03", B=1, T=256, H=2, D=64, gate_logit_normalizer=8.0, gate_mask_probability=0.3),
    _chunk_api_case("l2norm_in_kernel", B=1, T=256, H=2, D=64, scale=0.125, use_qk_l2norm_in_kernel=True),
    _chunk_api_case("b1_t8192_h16_d128_bf16", B=1, T=8192, H=16, D=128, scale=0.125, gate_mask_probability=0.3, dtype=jnp.bfloat16, long=True),
]

_CASES.extend(
    [
        _chunk_fwd_case(f"e2e_0", B=1, T=64, H=2, K=64, V=64, use_initial_state=True, output_final_state=True),
        _chunk_fwd_case(f"e2e_1", B=2, T=128, H=4, K=64, V=64, use_initial_state=True, output_final_state=True),
        _chunk_fwd_case(f"e2e_2", B=1, T=256, H=2, K=128, V=128, use_initial_state=True, output_final_state=True),
        _chunk_fwd_case(f"e2e_3", B=2, T=128, H=4, K=64, V=128, use_initial_state=True, output_final_state=True),
        _chunk_fwd_case(f"e2e_4", B=1, T=128, H=2, K=64, V=64, use_initial_state=True, output_final_state=True),
        _chunk_fwd_case(f"e2e_5", B=2, T=1024, H=8, K=128, V=128, use_initial_state=True, output_final_state=True),
        _chunk_fwd_case(f"e2e_6", B=1, T=256, H=2, K=64, V=64, use_initial_state=True, output_final_state=True),
        _chunk_fwd_case(f"e2e_7", B=1, T=64, H=1, D=32,use_initial_state=True, output_final_state=True, dtype=jnp.bfloat16),
        _chunk_fwd_case(f"e2e_8", B=1, T=128, H=2, D=64, use_initial_state=True, output_final_state=True, dtype=jnp.bfloat16),
        _chunk_fwd_case(f"e2e_9", B=1, T=256, H=4, D=128, use_initial_state=True, output_final_state=True, dtype=jnp.bfloat16),
        _chunk_fwd_case(f"e2e_10", B=1, T=512, H=8, D=256, use_initial_state=True, output_final_state=True, dtype=jnp.bfloat16),
        _chunk_fwd_case(f"e2e_11", B=1, T=1024, H=16, D=256, use_initial_state=True, output_final_state=True, dtype=jnp.bfloat16),
    ]
)


# End-to-end segment_ids coverage from test_segment_ids.py. The three
# segment_ids_to_cu_seqlens conversion tests are unit tests, so they are not
# ported here.
_CASES.extend([
    _varlen_case("segment_ids_recurrent_0", seq_lens=(64, 128), disable_recompute=False),
    _varlen_case("segment_ids_recurrent_1", seq_lens=(128, 64, 192), disable_recompute=False),
    _varlen_case("segment_ids_recurrent_2", seq_lens=(30, 50), T=128, disable_recompute=False),
    _varlen_case("segment_ids_recurrent_3", seq_lens=(45, 80, 20), T=256, disable_recompute=False),
    _varlen_case("segment_ids_recurrent_4", seq_lens=(64, 128), use_initial_state=True, disable_recompute=False),
    _varlen_case("segment_ids_recurrent_5", seq_lens=(45, 80, 20), T=256, use_initial_state=True, disable_recompute=False),
    _varlen_case("segment_ids_recurrent_6", seq_lens=(64, 128), dtype=jnp.bfloat16, disable_recompute=False),
    _varlen_case("segment_ids_recurrent_7", seq_lens=(45, 80, 20), T=256, dtype=jnp.bfloat16, use_initial_state=True, disable_recompute=False),
])
_CASES.extend([
    _varlen_case("segment_ids_batch_0_fp32", batch_seq_lens=((64, 128), (128, 64)), disable_recompute=False),
    _varlen_case("segment_ids_batch_1_fp32", batch_seq_lens=((64, 64), (128,)), disable_recompute=False),
    _varlen_case("segment_ids_batch_2_fp32", batch_seq_lens=((64, 64), (128,), (64, 64)), disable_recompute=False),
    _varlen_case("segment_ids_batch_3_fp32", batch_seq_lens=((30, 50), (45, 35)), disable_recompute=False),
    _varlen_case("segment_ids_batch_0_bf16", batch_seq_lens=((64, 128), (128, 64)), dtype=jnp.bfloat16, disable_recompute=False),
])


# Forward and backward public-wrapper cases from test_varlen_e2e.py.
_CASES.extend([
    _varlen_case("varlen_e2e_0", seq_lens=(64, 128)),
    _varlen_case("varlen_e2e_1", seq_lens=(128, 64, 192)),
    _varlen_case("varlen_e2e_2", seq_lens=(30, 50), T=128),
    _varlen_case("varlen_e2e_3", seq_lens=(45, 80, 20), T=256),
    _varlen_case("varlen_e2e_4", seq_lens=(100,), T=192),
    _varlen_case("varlen_e2e_5", seq_lens=(64, 128), T=256, N_pad=8),
    _varlen_case("varlen_e2e_6", seq_lens=(64, 128), use_initial_state=True),
    _varlen_case("varlen_e2e_7", seq_lens=(128, 64, 192), use_initial_state=True),
    _varlen_case("varlen_e2e_8", seq_lens=(30, 50), T=128, use_initial_state=True),
    _varlen_case("varlen_e2e_9", seq_lens=(45, 80, 20), T=256, use_initial_state=True),
    _varlen_case("varlen_e2e_10", seq_lens=(40, 55), T=256, N_pad=8, use_initial_state=True),
    _varlen_case("varlen_e2e_11", seq_lens=(64, 128), dtype=jnp.bfloat16),
    _varlen_case("varlen_e2e_12", seq_lens=(45, 80, 20), T=256, dtype=jnp.bfloat16),
    _varlen_case("varlen_e2e_13", seq_lens=(64, 128), T=256, N_pad=8, dtype=jnp.bfloat16),
    _varlen_case("varlen_e2e_14", seq_lens=(64, 128), dtype=jnp.bfloat16, use_initial_state=True),
    _varlen_case("varlen_e2e_15", seq_lens=(45, 80, 20), T=256, dtype=jnp.bfloat16, use_initial_state=True),
    _varlen_case("varlen_e2e_16", seq_lens=(450, 800, 200), T=8192, N_pad=64, dtype=jnp.bfloat16, use_initial_state=True),
    _varlen_case("varlen_e2e_17", seq_lens=(450, 800, 200), T=8192, H=8, N_pad=64, dtype=jnp.bfloat16, use_initial_state=True),
    _varlen_case("varlen_e2e_18", seq_lens=(40, 55), T=256, N_pad=8, dtype=jnp.bfloat16, use_initial_state=True),
    _varlen_case("varlen_e2e_19", seq_lens=(1800, 1500, 2000, 1200, 800), T=8192, H=16, N_pad=8, dtype=jnp.bfloat16, use_initial_state=True),
    _varlen_case("varlen_e2e_20", seq_lens=(1800, 1500, 2000, 1200, 800), T=8192, H=16, N_pad=5, dtype=jnp.bfloat16),
])

_CASES.extend([
    _varlen_case("varlen_batch_0", batch_seq_lens=((64, 128), (128, 64))),
    _varlen_case("varlen_batch_1", batch_seq_lens=((64, 128), (128, 64)), use_initial_state=True),
    _varlen_case("varlen_batch_2", batch_seq_lens=((64, 128), (64, 64, 64)), use_initial_state=True),
    _varlen_case("varlen_batch_3", batch_seq_lens=((30, 50), (45, 35)), T=128, use_initial_state=True),
    _varlen_case("varlen_batch_4", batch_seq_lens=((64, 128), (128, 64)), dtype=jnp.bfloat16, use_initial_state=True),
])


# Full-pipeline production-range cases from test_intra_varlen_prod.py. These
# feed already activated gates into the public wrapper, matching the source
# test's end-to-end call; the isolated intra-chunk tests are intentionally
# omitted.
_CASES.extend([
    _varlen_case("prod_preactivated_0", seq_lens=(64, 128), input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_1", seq_lens=(128, 64, 192), input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_2", seq_lens=(30, 50), T=128, input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_3", seq_lens=(45, 80, 20), T=256, input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_4", seq_lens=(64, 128), input_profile="prod", use_initial_state=True, safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_5", seq_lens=(64, 128), T=256, N_pad=8, input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_6", seq_lens=(64, 128), input_profile="prod", dtype=jnp.bfloat16, safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("prod_preactivated_7", seq_lens=(45, 80, 20), T=256, input_profile="prod", dtype=jnp.bfloat16, safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("prod_preactivated_8", seq_lens=(64, 128), input_profile="prod", dtype=jnp.bfloat16, use_initial_state=True, safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("prod_preactivated_9", seq_lens=(450, 800, 200), T=8192, N_pad=64, input_profile="prod", dtype=jnp.bfloat16, use_initial_state=True, safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("prod_preactivated_10", seq_lens=(64, 128), input_profile="low_gate", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_11", seq_lens=(64, 128), input_profile="low_gate", dtype=jnp.bfloat16, safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("prod_preactivated_12", seq_lens=(45, 80, 20), T=256, input_profile="low_gate", dtype=jnp.bfloat16, safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("prod_preactivated_13", seq_lens=(64, 128), input_profile="strong_gate", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_14", seq_lens=(64, 128), input_profile="strong_gate", dtype=jnp.bfloat16, safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("prod_preactivated_15", seq_lens=(45, 80, 20), T=256, input_profile="strong_gate", dtype=jnp.bfloat16, use_initial_state=True, safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
])

_CASES.extend([
    _varlen_case("prod_preactivated_batch_0", batch_seq_lens=((64, 128), (128, 64)), input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_batch_1", batch_seq_lens=((64, 128), (128, 64)), use_initial_state=True, input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_batch_2", batch_seq_lens=((64, 128), (64, 64, 64)), use_initial_state=True, input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=8e-4, forward_rtol=8e-4, state_atol=5e-3, state_rtol=5e-3, backward_atol=2e-2, backward_rtol=1e-2),
    _varlen_case("prod_preactivated_batch_3", batch_seq_lens=((64, 128), (128, 64)), dtype=jnp.bfloat16, use_initial_state=True, input_profile="prod", safe_gate=True, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
])


# Public chunk_kda fused-gate E2E cases from test_segment_e2e.py.
# The source uses per-sequence Pallas as its backward baseline. The unified E2E
# test compares against XLA, whose bf16 fused-gate accumulation differs more
# when the final-state cotangent is present.
_CASES.extend([
    _varlen_case("segment_fused_fwd_0", seq_lens=(64, 128), H=8, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_fwd_1", seq_lens=(128, 64, 192), H=8, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_fwd_2", seq_lens=(30, 50), T=128, H=8, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_fwd_3", seq_lens=(45, 80, 20), T=256, H=8, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_fwd_4", seq_lens=(64, 128), H=8, use_initial_state=True, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_fwd_5", seq_lens=(45, 80, 20, 1500, 300, 2000, 1200, 800, 1747, 500), T=8192, H=8, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_fwd_6", seq_lens=(64, 128), H=8, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.1, backward_rtol=0.1),
    _varlen_case("segment_fused_fwd_7", seq_lens=(128, 64, 192), H=8, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.1, backward_rtol=0.1),
    _varlen_case("segment_fused_fwd_8", seq_lens=(45, 80, 20), T=256, H=8, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.1, backward_rtol=0.1),
    _varlen_case("segment_fused_fwd_9", seq_lens=(64, 128), H=8, dtype=jnp.bfloat16, use_initial_state=True, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.1, backward_rtol=0.1),
])

_CASES.extend([
    _varlen_case("segment_fused_batch_0", batch_seq_lens=((64, 128), (128, 64)), H=8, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_batch_1", batch_seq_lens=((64, 128), (128, 64)), H=8, use_initial_state=True, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_batch_2", batch_seq_lens=((64, 128), (64, 64, 64)), H=8, use_initial_state=True, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4, state_atol=8e-3, state_rtol=8e-3, backward_atol=0.05, backward_rtol=0.05),
    _varlen_case("segment_fused_batch_3", batch_seq_lens=((64, 128), (128, 64)), H=8, dtype=jnp.bfloat16, use_initial_state=True, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.1, backward_rtol=0.1),
])

_CASES.extend([
    _cp_case("cp_fwd_cp2_single_fp32", cp_size=2, T=8192, H=32, seq_lens=(8192,), seed=2, dtype=jnp.float32),
    _cp_case("cp_fwd_cp2_single_bf16", cp_size=2, T=8192, H=32, seq_lens=(8192,), seed=2, dtype=jnp.bfloat16),
    _cp_case("cp_fwd_cp4_single_fp32", cp_size=4, T=8192, H=32, seq_lens=(8192,), seed=3, dtype=jnp.float32),
    _cp_case("cp_fwd_cp4_single_bf16", cp_size=4, T=8192, H=32, seq_lens=(8192,), seed=3, dtype=jnp.bfloat16),
    _cp_case("cp_fwd_cp2_multi_fp32", cp_size=2, T=8192, H=32, seq_lens=(1024, 7168), seed=11, dtype=jnp.float32),
    _cp_case("cp_fwd_cp2_multi_bf16", cp_size=2, T=8192, H=32, seq_lens=(1024, 7168), seed=11, dtype=jnp.bfloat16),
    _cp_case("cp_fwd_cp4_multi_fp32", cp_size=4, T=8192, H=32, seq_lens=_lengths_from_boundaries((0, 392, 777, 1024, 3241, 5120, 7777, 8192)), seed=13, dtype=jnp.float32),
    _cp_case("cp_fwd_cp4_multi_bf16", cp_size=4, T=8192, H=32, seq_lens=_lengths_from_boundaries((0, 392, 777, 1024, 3241, 5120, 7777, 8192)), seed=13, dtype=jnp.bfloat16),
])

_CASES.extend([
    _cp_case("cp_prod_cp2_sparse_3_bf16", cp_size=2, T=16384, H=32, seq_lens=(5000, 6000, 5384), seed=42, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05),
    _cp_case("cp_prod_cp2_medium_10_fp32", cp_size=2, T=16384, H=32, seq_lens=_lengths_from_boundaries((0, 700, 1800, 3500, 5200, 7000, 9000, 10500, 12500, 14000, 16384)), seed=42, dtype=jnp.float32, input_profile="prod", use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4),
    _cp_case("cp_prod_cp2_medium_10_bf16", cp_size=2, T=16384, H=32, seq_lens=_lengths_from_boundaries((0, 700, 1800, 3500, 5200, 7000, 9000, 10500, 12500, 14000, 16384)), seed=42, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05),
    _cp_case("cp_prod_cp2_dense_25_fp32", cp_size=2, T=16384, H=32, seq_lens=_jittered_lengths(16384, 25, 11), seed=42, dtype=jnp.float32, input_profile="prod", use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4),
    _cp_case("cp_prod_cp2_dense_25_bf16", cp_size=2, T=16384, H=32, seq_lens=_jittered_lengths(16384, 25, 11), seed=42, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05),
    _cp_case("cp_prod_cp2_max_45_fp32", cp_size=2, T=16384, H=32, seq_lens=_jittered_lengths(16384, 45, 12), seed=42, dtype=jnp.float32, input_profile="prod", use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4),
    _cp_case("cp_prod_cp2_max_45_bf16", cp_size=2, T=16384, H=32, seq_lens=_jittered_lengths(16384, 45, 12), seed=42, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05),
])
_CASES.extend([
    _cp_case("cp_prod_cp4_sparse_4_fp32", cp_size=4, T=32768, H=32, seq_lens=(8000, 8000, 8000, 8768), seed=99, dtype=jnp.float32, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4),
    _cp_case("cp_prod_cp4_sparse_4_bf16", cp_size=4, T=32768, H=32, seq_lens=(8000, 8000, 8000, 8768), seed=99, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05),
    _cp_case("cp_prod_cp4_medium_15_fp32", cp_size=4, T=32768, H=32, seq_lens=_lengths_from_boundaries((0, 1500, 5500, 10500, 15500, 20500, 26500, 30500, 32768)), seed=99, dtype=jnp.float32, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4),
    _cp_case("cp_prod_cp4_medium_15_bf16", cp_size=4, T=32768, H=32, seq_lens=_lengths_from_boundaries((0, 1500, 5500, 10500, 15500, 20500, 26500, 30500, 32768)), seed=99, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05),
    _cp_case("cp_prod_cp4_dense_40_fp32", cp_size=4, T=32768, H=32, seq_lens=_jittered_lengths(32768, 40, 21), seed=99, dtype=jnp.float32, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4),
    _cp_case("cp_prod_cp4_dense_40_bf16", cp_size=4, T=32768, H=32, seq_lens=_jittered_lengths(32768, 40, 21), seed=99, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05),
    _cp_case("cp_prod_cp4_max_85_fp32", cp_size=4, T=32768, H=32, seq_lens=_jittered_lengths(32768, 85, 20), seed=99, dtype=jnp.float32, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=8e-4, forward_rtol=8e-4),
    _cp_case("cp_prod_cp4_max_85_bf16", cp_size=4, T=32768, H=32, seq_lens=_jittered_lengths(32768, 85, 20), seed=99, dtype=jnp.bfloat16, input_profile="prod", use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, forward_atol=0.05, forward_rtol=0.05),
])


# The source backward suite uses all available devices. Tokamax keeps the
# explicit CP4 mesh used by the source forward tests so collection is stable
# on hosts exposing extra TPU devices.
_CASES.extend([
    _cp_case("cp_bwd_h2_single_fp32", cp_size=4, T=512, H=2),
    _cp_case("cp_bwd_h16_single_mask03_fp32", cp_size=4, T=512, H=16, gate_mask_probability=0.3, backward_atol=2e-3),
    _cp_case("cp_bwd_h2_split075_fp32", cp_size=4, T=512, H=2, seq_lens=(384, 128), backward_atol=2e-3),
    _cp_case("cp_bwd_h16_split05_mask03_fp32", cp_size=4, T=512, H=16, seq_lens=(256, 256), gate_mask_probability=0.3, backward_atol=2e-3),
    _cp_case("cp_bwd_h8_split0375_fp32", cp_size=4, T=512, H=8, seq_lens=(192, 320), backward_atol=5e-4),
    _cp_case("cp_bwd_h16_t32768_bf16", cp_size=4, T=32768, H=16, dtype=jnp.bfloat16, gate_mask_probability=0.3),
    _cp_case("cp_fwd_cp2_b2_single_fp32", cp_size=2, T=8192, H=32, batch_seq_lens=((8192,), (8192,)), seed=50, dtype=jnp.float32),
    _cp_case("cp_fwd_cp2_b2_single_bf16", cp_size=2, T=8192, H=32, batch_seq_lens=((8192,), (8192,)), seed=50, dtype=jnp.bfloat16),
    _cp_case("cp_fwd_cp2_b2_different_fp32", cp_size=2, T=8192, H=32, batch_seq_lens=((8192,), (1024, 7168)), seed=51, dtype=jnp.float32),
    _cp_case("cp_fwd_cp2_b2_different_bf16", cp_size=2, T=8192, H=32, batch_seq_lens=((8192,), (1024, 7168)), seed=51, dtype=jnp.bfloat16),
    _cp_case("cp_fwd_cp4_b2_multi_fp32", cp_size=4, T=8192, H=32, batch_seq_lens=(_lengths_from_boundaries((0, 392, 1024, 3241, 5120, 7777, 8192)), (2048, 2048, 2048, 2048)), seed=52, dtype=jnp.float32),
    _cp_case("cp_fwd_cp4_b2_multi_bf16", cp_size=4, T=8192, H=32, batch_seq_lens=(_lengths_from_boundaries((0, 392, 1024, 3241, 5120, 7777, 8192)), (2048, 2048, 2048, 2048)), seed=52, dtype=jnp.bfloat16),
    _cp_case("cp_fwd_cp2_b4_varlen_fp32", cp_size=2, T=8192, H=32, batch_seq_lens=((8192,), (3000, 5192), (1024, 3072, 4096), (512, 1536, 3072, 1880, 1192)), seed=53, dtype=jnp.float32),
    _cp_case("cp_fwd_cp2_b4_varlen_bf16", cp_size=2, T=8192, H=32, batch_seq_lens=((8192,), (3000, 5192), (1024, 3072, 4096), (512, 1536, 3072, 1880, 1192)), seed=53, dtype=jnp.bfloat16),
    _cp_case("cp_bwd_b2_single_fp32", cp_size=4, T=512, H=2, batch_seq_lens=((512,), (512,))),
    _cp_case("cp_bwd_b2_different_fp32", cp_size=4, T=512, H=2, batch_seq_lens=((512,), (384, 128)), backward_atol=2e-3),
    _cp_case("cp_bwd_b2_splits_fp32", cp_size=4, T=512, H=8, batch_seq_lens=((256, 256), (192, 320))),
    _cp_case("cp_bwd_b4_mixed_mask03_fp32", cp_size=4, T=512, H=2, batch_seq_lens=((512,), (256, 256), (384, 128), (512,)), gate_mask_probability=0.3, backward_atol=2e-3),
    make_test_config("tokamax_fixed_t8192", B=1, T=8192, H=16, D=128, dtype=jnp.bfloat16, input_profile="model", output_final_state=True, use_qk_l2norm_in_kernel=True, long=True),
    _varlen_case("tokamax_varlen_fused", seq_lens=(45, 80, 20), T=256, use_initial_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, lower_bound=-0.01, disable_recompute=False, forward_atol=0.05, forward_rtol=0.05, state_atol=0.05, state_rtol=0.05, backward_atol=0.05, backward_rtol=0.05),
    make_test_config("tokamax_fixed_unaligned_kv", B=1, T=64, H=1, K=129, V=127, dtype=jnp.bfloat16, input_profile="model"),
    _cp_case("tokamax_cp2_small", cp_size=2, T=128, H=2, dtype=jnp.float32),
    # Public varlen regressions from test_varlen_e2e.py.
    make_test_config("varlen_shallow_tail", T=256, H=2, K=128, V=128, seq_lens=(30, 50), dtype=jnp.float32, input_profile="model", backward_check="finite"),
    make_test_config("varlen_shallow_tail_large_head", T=256, H=2, K=256, V=256, seq_lens=(30, 50), dtype=jnp.float32, input_profile="model", backward_check="finite"),
    make_test_config("varlen_single_seq_tail", T=256, H=2, K=128, V=128, seq_lens=(45,), dtype=jnp.float32, input_profile="model", backward_check="finite"),
    make_test_config("varlen_padding_tail", T=128, H=2, K=128, V=128, seq_lens=(30, 50), dtype=jnp.float32, input_profile="model", forward_check="padding_tail"),
    _varlen_case("varlen_empty_states_h0", seq_lens=(64, 128), T=256, N_pad=8, use_initial_state=True, forward_check="empty_states"),
    _varlen_case("varlen_empty_states_no_h0", seq_lens=(64, 128), T=256, N_pad=8, forward_check="empty_states"),
])

def _case_params():
  return tuple(
      pytest.param(
          case,
          id=case.name,
          marks=(pytest.mark.long,) if case.long else (),
      )
      for case in _CASES
  )


def _l2_normalize(x: jax.Array) -> jax.Array:
  x_f32 = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(jnp.square(x_f32), axis=-1) + 1e-6)
  return (x_f32 * rstd[..., None]).astype(x.dtype)


def _make_segment_ids(
    layouts: tuple[tuple[int, ...], ...], seq_len: int
) -> jax.Array:
  ids = np.zeros((len(layouts), seq_len), dtype=np.int32)
  for batch_index, seq_lens in enumerate(layouts):
    offset = 0
    for segment_id, length in enumerate(seq_lens, start=1):
      ids[batch_index, offset : offset + length] = segment_id
      offset += length
  return jnp.asarray(ids)


def _make_inputs(case: TestConfig) -> _Inputs:
  if case.batch < 1:
    raise ValueError("Batch size must be positive.")
  if case.layouts is not None:
    if len(case.layouts) != case.batch:
      raise ValueError("The number of layouts must equal the batch size.")
    for batch_index, seq_lens in enumerate(case.layouts):
      if not seq_lens or any(length <= 0 for length in seq_lens):
        raise ValueError(f"Invalid sequence layout for batch {batch_index}.")
      if sum(seq_lens) > case.seq_len:
        raise ValueError(
            f"Batch {batch_index} has {sum(seq_lens)} real tokens, which "
            f"exceeds T={case.seq_len}."
        )
  if case.seq_len % case.cp_size != 0:
    raise ValueError("Global sequence length must be divisible by cp_size.")
  if case.cp_size > 1 and case.seq_len // case.cp_size % 64 != 0:
    raise ValueError("Each device must receive a multiple of 64 tokens.")
  if case.layouts is None and case.seq_len % 64 != 0:
    raise ValueError("Fixed-length cases must contain complete chunks.")

  keys = jax.random.split(jax.random.key(case.seed), 12)
  qk_shape = (case.heads, case.batch, case.seq_len, case.key_dim)
  v_shape = (case.heads, case.batch, case.seq_len, case.value_dim)
  beta_shape = (case.heads, case.batch, case.seq_len)
  segment_ids = (
      _make_segment_ids(case.layouts, case.seq_len)
      if case.layouts is not None
      else None
  )
  valid = (
      segment_ids > 0
      if segment_ids is not None
      else jnp.ones((case.batch, case.seq_len), dtype=jnp.bool_)
  )
  valid_4d = valid[None, :, :, None]
  valid_3d = valid[None, :, :]

  profile = case.input_profile
  if profile == "chunk_api":
    q = jax.random.uniform(keys[0], qk_shape, dtype=jnp.float32)
    k = jax.random.uniform(keys[1], qk_shape, dtype=jnp.float32)
    v = jax.random.uniform(keys[2], v_shape, dtype=jnp.float32)
    if not case.use_qk_l2norm_in_kernel:
      q = _l2_normalize(q)
      k = _l2_normalize(k)
    g_raw = jax.random.uniform(keys[3], qk_shape, dtype=jnp.float32)
    A_log = jax.random.normal(
        keys[5], (case.heads,), dtype=jnp.float32
    )
    dt_bias = jax.random.normal(
        keys[6], (case.heads * case.key_dim,), dtype=jnp.float32
    )
    if case.use_gate_in_kernel:
      g = g_raw
    else:
      g = jax.nn.log_sigmoid(g_raw) / case.gate_logit_normalizer
      if case.safe_gate:
        g = jnp.clip(g, -5.0, 0.0)
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], beta_shape, dtype=jnp.float32)
    )
    initial_scale = 1.0
    initial_dtype = jnp.float32
  elif profile == "chunk_fwd":
    q = 0.1 * jax.random.normal(keys[0], qk_shape, dtype=jnp.float32)
    k = 0.1 * jax.random.normal(keys[1], qk_shape, dtype=jnp.float32)
    v = 0.1 * jax.random.normal(keys[2], v_shape, dtype=jnp.float32)
    g = -0.05 * jnp.abs(
        jax.random.normal(keys[3], qk_shape, dtype=jnp.float32)
    )
    beta = 0.5 * jax.nn.sigmoid(
        jax.random.normal(keys[4], beta_shape, dtype=jnp.float32)
    )
    A_log = jnp.zeros((case.heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros(
        (case.heads * case.key_dim,), dtype=jnp.float32
    )
    initial_scale = 0.01
    initial_dtype = case.dtype
  elif profile == "cp":
    q = 0.1 * jax.random.uniform(keys[0], qk_shape, dtype=jnp.float32)
    k = 0.1 * jax.random.uniform(keys[1], qk_shape, dtype=jnp.float32)
    v = 0.1 * jax.random.uniform(keys[2], v_shape, dtype=jnp.float32)
    g = jax.nn.log_sigmoid(
        jax.random.uniform(keys[3], qk_shape, dtype=jnp.float32)
    ) / case.gate_logit_normalizer
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], beta_shape, dtype=jnp.float32)
    )
    A_log = jnp.zeros((case.heads,), dtype=jnp.float32)
    dt_bias = jnp.zeros(
        (case.heads * case.key_dim,), dtype=jnp.float32
    )
    initial_scale = 0.0
    initial_dtype = jnp.float32
  elif profile in ("model", "prod", "low_gate", "strong_gate"):
    q = jax.nn.silu(
        jax.random.normal(keys[0], qk_shape, dtype=jnp.float32)
    )
    k = jax.nn.silu(
        jax.random.normal(keys[1], qk_shape, dtype=jnp.float32)
    )
    if not case.use_qk_l2norm_in_kernel:
      q = _l2_normalize(q.astype(case.dtype))
      k = _l2_normalize(k.astype(case.dtype))

    if profile == "strong_gate":
      value_scale = 2.0
    elif profile in ("prod", "low_gate"):
      value_scale = 1.5
    else:
      value_scale = 1.0
    v = value_scale * jax.random.normal(
        keys[2], v_shape, dtype=jnp.float32
    )

    if profile == "model":
      A_log = jnp.log(
          jax.random.uniform(
              keys[5], (case.heads,), minval=1.0, maxval=16.0
          )
      )
      dt = jnp.exp(
          jax.random.uniform(keys[6], (case.heads * case.key_dim,))
          * (jnp.log(jnp.array(0.1)) - jnp.log(jnp.array(0.001)))
          + jnp.log(jnp.array(0.001))
      ).clip(min=1e-4)
      dt_bias = dt + jnp.log(-jnp.expm1(-dt))
      g_raw = jax.random.normal(keys[3], qk_shape, dtype=jnp.float32)
      beta = jax.nn.sigmoid(
          jax.random.normal(keys[4], beta_shape, dtype=jnp.float32)
      )
    elif profile == "prod":
      A_log = jax.random.uniform(
          keys[5], (case.heads,), minval=0.2, maxval=3.0
      )
      dt_bias = jax.random.uniform(
          keys[6],
          (case.heads * case.key_dim,),
          minval=-8.0,
          maxval=-1.5,
      )
      g_raw = jax.random.uniform(
          keys[3], qk_shape, minval=-4.5, maxval=4.5
      )
      beta = jax.random.uniform(
          keys[4], beta_shape, minval=0.05, maxval=0.95
      )
    elif profile == "low_gate":
      A_log = jnp.full((case.heads,), 2.5, dtype=jnp.float32)
      dt_bias = jnp.full(
          (case.heads * case.key_dim,), -10.0, dtype=jnp.float32
      )
      g_raw = jax.random.uniform(
          keys[3], qk_shape, minval=-1.0, maxval=1.0
      )
      beta = jax.random.uniform(
          keys[4], beta_shape, minval=0.3, maxval=0.95
      )
    else:
      A_log = jnp.full((case.heads,), 2.7, dtype=jnp.float32)
      dt_bias = jnp.full(
          (case.heads * case.key_dim,), -2.0, dtype=jnp.float32
      )
      g_raw = jax.random.uniform(
          keys[3], qk_shape, minval=2.0, maxval=4.5
      )
      beta = jax.random.uniform(
          keys[4], beta_shape, minval=0.05, maxval=0.95
      )

    if case.use_gate_in_kernel:
      g = g_raw
    else:
      gate_input = g_raw + dt_bias.reshape(
          case.heads, 1, 1, case.key_dim
      )
      A = jnp.exp(A_log).reshape(case.heads, 1, 1, 1)
      if profile in ("prod", "low_gate", "strong_gate"):
        g = -5.0 * jax.nn.sigmoid(A * gate_input)
      else:
        g = -A * jax.nn.softplus(gate_input)
    initial_scale = 0.1
    initial_dtype = jnp.float32
  else:
    raise ValueError(f"Unknown input profile: {profile}")

  if case.gate_mask_probability > 0:
    gate_mask = (
        jax.random.uniform(keys[10], qk_shape, dtype=jnp.float32)
        > case.gate_mask_probability
    )
    g = g * gate_mask

  q = jnp.where(valid_4d, q, 0).astype(case.dtype)
  k = jnp.where(valid_4d, k, 0).astype(case.dtype)
  v = jnp.where(valid_4d, v, 0).astype(case.dtype)
  g = jnp.where(valid_4d, g, 0).astype(case.dtype)
  beta = jnp.where(valid_3d, beta, 0).astype(case.dtype)

  state_count = case.n_max or 1
  initial_state = None
  if case.use_initial_state:
    initial_state = initial_scale * jax.random.normal(
        keys[7],
        (
            case.batch,
            state_count,
            case.heads,
            case.key_dim,
            case.value_dim,
        ),
        dtype=jnp.float32,
    ).astype(initial_dtype)

  dout = jax.random.normal(
      keys[8],
      (case.heads, case.batch, case.seq_len, case.value_dim),
      dtype=jnp.float32,
  ).astype(case.dtype)
  dfinal_state = None
  if case.output_final_state:
    dfinal_state = 0.1 * jax.random.normal(
        keys[9],
        (
            case.batch,
            state_count,
            case.heads,
            case.key_dim,
            case.value_dim,
        ),
        dtype=jnp.float32,
    )

  inputs = _Inputs(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      A_log=A_log,
      dt_bias=dt_bias,
      initial_state=initial_state,
      segment_ids=segment_ids,
      dout=dout,
      dfinal_state=dfinal_state,
  )
  if case.forward_check == "padding_tail":
    inputs = dataclasses.replace(
        inputs,
        q=inputs.q.at[:, 0, 0].set(10.0),
        k=inputs.k.at[:, 0, 0].set(5.0),
        v=inputs.v.at[:, 0, 0].set(3.0),
    )
  elif case.forward_check == "empty_states" and inputs.initial_state is not None:
    assert case.layouts is not None
    initial_state = inputs.initial_state
    for batch_index, seq_lens in enumerate(case.layouts):
      for state_index in range(len(seq_lens), case.n_max or 0):
        initial_state = initial_state.at[batch_index, state_index].set(
            float(state_index + 1)
        )
    inputs = dataclasses.replace(inputs, initial_state=initial_state)
  elif case.forward_check not in ("reference", "empty_states"):
    raise ValueError(f"Unknown forward check: {case.forward_check}")
  return inputs


def _attention_kwargs(case: TestConfig, inputs: _Inputs) -> dict[str, object]:
  return dict(
      A_log=inputs.A_log if case.use_gate_in_kernel else None,
      dt_bias=inputs.dt_bias if case.use_gate_in_kernel else None,
      scale=case.effective_scale,
      initial_state=inputs.initial_state,
      output_final_state=case.output_final_state,
      use_qk_l2norm_in_kernel=case.use_qk_l2norm_in_kernel,
      use_gate_in_kernel=case.use_gate_in_kernel,
      safe_gate=case.safe_gate,
      lower_bound=case.lower_bound,
      disable_recompute=case.disable_recompute,
      chunk_size=64,
      N_max=case.n_max,
  )


def _call_attention(
    implementation: api.Implementation | Sequence[api.Implementation] | None,
    case: TestConfig,
    inputs: _Inputs,
    *,
    cp_context: CPContext | None = None,
):
  return api.kimi_delta_attention(
      inputs.q,
      inputs.k,
      inputs.v,
      inputs.g,
      inputs.beta,
      segment_ids=inputs.segment_ids,
      cp_context=cp_context,
      implementation=implementation,
      **_attention_kwargs(case, inputs),
  )


def _cp_mesh(case: TestConfig) -> tuple[Mesh, CPContext]:
  devices = jax.devices()[: case.cp_size]
  mesh = Mesh(np.asarray(devices), ("context",))
  return mesh, CPContext(mesh=mesh, axis_name="context")


def _cp_forward(
    implementation: api.Implementation | Sequence[api.Implementation] | None,
    case: TestConfig,
    inputs: _Inputs,
) -> tuple[jax.Array, None]:
  mesh, cp_context = _cp_mesh(case)

  def local_forward(q, k, v, g, beta, segment_ids):
    local_inputs = dataclasses.replace(
        inputs,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        segment_ids=segment_ids,
    )
    output, _ = _call_attention(
        implementation,
        case,
        local_inputs,
        cp_context=cp_context,
    )
    return output

  # The CP pre-process accepts batched cu_seqlens, but its runtime annotation
  # currently describes only the unbatched fast path.
  with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
    forward = jax.jit(
        jax.shard_map(
            local_forward,
            mesh=mesh,
            in_specs=(P(None, None, "context", None),) * 4
            + (P(None, None, "context"), P(None, "context")),
            out_specs=P(None, None, "context", None),
            check_vma=False,
        )
    )
    output = forward(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.segment_ids,
    )
  return output, None


def _cp_backward(
    implementation: api.Implementation | Sequence[api.Implementation] | None,
    case: TestConfig,
    inputs: _Inputs,
) -> tuple[jax.Array, ...]:
  mesh, cp_context = _cp_mesh(case)

  def local_backward(q, k, v, g, beta, segment_ids, dout):
    local_inputs = dataclasses.replace(
        inputs,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        segment_ids=segment_ids,
        dout=dout,
    )

    def local_forward(q, k, v, g, beta):
      current_inputs = dataclasses.replace(
          local_inputs, q=q, k=k, v=v, g=g, beta=beta
      )
      output, _ = _call_attention(
          implementation, case, current_inputs, cp_context=cp_context
      )
      return output

    _, pullback = jax.vjp(local_forward, q, k, v, g, beta)
    return pullback(dout)

  qkv_spec = P(None, None, "context", None)
  beta_spec = P(None, None, "context")
  with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
    backward = jax.jit(
        jax.shard_map(
            local_backward,
            mesh=mesh,
            in_specs=(qkv_spec,) * 4
            + (beta_spec, P(None, "context"), qkv_spec),
            out_specs=(qkv_spec,) * 4 + (beta_spec,),
            check_vma=False,
        )
    )
    return backward(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.segment_ids,
        inputs.dout,
    )


def _direct_backward(
    implementation: api.Implementation | Sequence[api.Implementation] | None,
    case: TestConfig,
    inputs: _Inputs,
) -> tuple[jax.Array, ...]:
  def loss_fn(q, k, v, g, beta, initial_state, A_log, dt_bias):
    current_inputs = dataclasses.replace(
        inputs,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    output, final_state = _call_attention(
        implementation, case, current_inputs
    )
    loss = jnp.sum(output.astype(jnp.float32) * inputs.dout)
    if inputs.dfinal_state is not None:
      if final_state is None:
        raise ValueError("Expected a final state for the backward case.")
      loss += jnp.sum(final_state * inputs.dfinal_state)
    return loss

  argnums = [0, 1, 2, 3, 4]
  if case.use_initial_state:
    argnums.append(5)
  return jax.grad(loss_fn, argnums=tuple(argnums))(
      inputs.q,
      inputs.k,
      inputs.v,
      inputs.g,
      inputs.beta,
      inputs.initial_state,
      inputs.A_log,
      inputs.dt_bias,
  )


def _require_tpu(case: TestConfig) -> None:
  if jax.default_backend() != "tpu":
    pytest.skip("Pallas TPU KDA tests require a TPU backend.")
  if len(jax.devices()) < case.cp_size:
    pytest.skip(f"Case requires {case.cp_size} TPU devices.")


def _valid_gradient_values(
    name: str, gradient: jax.Array, case: TestConfig, inputs: _Inputs
) -> jax.Array | np.ndarray:
  """Selects the valid tokens/states used by the source varlen tests."""
  if case.layouts is None:
    return gradient
  gradient_np = np.asarray(gradient)
  if name == "dh0":
    state_mask = np.zeros((case.batch, case.n_max or 1), dtype=np.bool_)
    for batch_index, seq_lens in enumerate(case.layouts):
      state_mask[batch_index, : len(seq_lens)] = True
    mask = state_mask[:, :, None, None, None]
  else:
    assert inputs.segment_ids is not None
    token_mask = np.asarray(inputs.segment_ids) > 0
    mask = (
        token_mask[None, :, :, None]
        if gradient_np.ndim == 4
        else token_mask[None, :, :]
    )
  return gradient_np[np.broadcast_to(mask, gradient_np.shape)]


@pytest.mark.parametrize("case", _case_params())
def test_chunk_kda_forward(case: TestConfig):
  _require_tpu(case)
  inputs = _make_inputs(case)

  if case.cp_size > 1:
    actual_result = _cp_forward("pallas_tpu", case, inputs)
    expected_result = _cp_forward("xla", case, inputs)
  else:
    actual_result = _call_attention("pallas_tpu", case, inputs)
    expected_result = _call_attention("xla", case, inputs)
  jax.block_until_ready((actual_result, expected_result))

  for name, actual, expected in zip(
      ("output", "final_state"), actual_result, expected_result, strict=True
  ):
    is_state = name == "final_state"
    assert compare_tensor(
        name,
        expected,
        actual,
        atol=(case.state_atol if is_state and case.state_atol is not None
              else case.forward_atol),
        rtol=(case.state_rtol if is_state and case.state_rtol is not None
              else case.forward_rtol),
        dtype=actual.dtype if actual is not None else case.dtype,
    ), f"{name} mismatch"

  output, final_state = actual_result
  if case.forward_check == "padding_tail":
    assert case.layouts is not None and case.batch == 1
    real_seq_len = sum(case.layouts[0])
    output_padding = np.asarray(output[:, 0, real_seq_len:])
    output_token_zero = np.asarray(output[:, 0, :1])
    assert np.max(np.abs(output_padding - output_token_zero)) > 1e-3
    assert np.max(np.abs(output_padding)) < 50.0
  elif case.forward_check == "empty_states":
    assert case.layouts is not None and final_state is not None
    for batch_index, seq_lens in enumerate(case.layouts):
      real_state_count = len(seq_lens)
      for state_index in range(real_state_count, case.n_max or 0):
        expected = (
            inputs.initial_state[batch_index, state_index]
            if inputs.initial_state is not None
            else jnp.zeros_like(final_state[batch_index, state_index])
        )
        assert compare_tensor(
            f"empty_state_{batch_index}_{state_index}",
            expected,
            final_state[batch_index, state_index],
            atol=0.01,
            rtol=0.0,
            dtype=final_state.dtype,
        )
      assert bool(
          jnp.all(jnp.isfinite(final_state[batch_index, :real_state_count]))
      )


@pytest.mark.parametrize("case", _case_params())
def test_chunk_kda_backward(case: TestConfig):
  _require_tpu(case)
  inputs = _make_inputs(case)

  grads = (
      _cp_backward("pallas_tpu", case, inputs)
      if case.cp_size > 1
      else _direct_backward(None, case, inputs)
  )
  grad_names = ("dq", "dk", "dv", "dg", "dbeta")
  if case.use_initial_state:
    grad_names += ("dh0",)

  if case.backward_check == "finite":
    output, _ = (
        _cp_forward("pallas_tpu", case, inputs)
        if case.cp_size > 1
        else _call_attention("pallas_tpu", case, inputs)
    )
    jax.block_until_ready((grads, output))
    for name, grad in zip(grad_names, grads, strict=True):
      assert bool(jnp.all(jnp.isfinite(grad))), f"{name} contains NaN/Inf"
    assert bool(jnp.all(jnp.isfinite(output))), "output contains NaN/Inf"
    return
  if case.backward_check != "reference":
    raise ValueError(f"Unknown backward check: {case.backward_check}")

  reference_grads = (
      _cp_backward("xla", case, inputs)
      if case.cp_size > 1
      else _direct_backward("xla", case, inputs)
  )
  jax.block_until_ready((grads, reference_grads))

  for name, actual, expected in zip(
      grad_names, grads, reference_grads, strict=True
  ):
    actual = _valid_gradient_values(name, actual, case, inputs)
    expected = _valid_gradient_values(name, expected, case, inputs)
    # The source bf16 E2E tests compare gradients in float32. In particular,
    # the custom VJP accumulates dh0 in float32 while XLA returns the tangent
    # dtype of a bf16 initial state.
    if case.dtype == jnp.bfloat16:
      actual = actual.astype(np.float32)
      expected = expected.astype(np.float32)
    assert compare_tensor(
        name,
        expected,
        actual,
        atol=case.backward_atol,
        rtol=case.backward_rtol,
        dtype=actual.dtype,
    ), f"Gradient mismatch for {name}"


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
