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
"""Tokamax ops and kernel implementations."""

import dataclasses
import enum


class Platform(enum.StrEnum):
  CPU = "cpu"
  GPU = "gpu"
  TPU = "tpu"


def _check_name(owner: object, name: object) -> None:
  if not isinstance(name, str) or not name:
    raise ValueError(
        f"{type(owner).__name__}.name must be a non-empty str, got {name!r}"
    )


@dataclasses.dataclass(frozen=True)
class Backend:
  name: str
  platforms: frozenset[Platform]

  def __post_init__(self) -> None:
    _check_name(self, self.name)


XLA = Backend("xla", frozenset({Platform.CPU, Platform.GPU, Platform.TPU}))
MOSAIC_TPU = Backend("mosaic_tpu", frozenset({Platform.TPU}))
MOSAIC_GPU = Backend("mosaic_gpu", frozenset({Platform.GPU}))
TRITON = Backend("triton", frozenset({Platform.GPU}))
CUTEDSL = Backend("cutedsl", frozenset({Platform.GPU}))
CUDNN = Backend("cudnn", frozenset({Platform.GPU}))


@dataclasses.dataclass(frozen=True)
class Impl:
  name: str
  backend: Backend
  module: str
  # Extra op directories (relative to _src/ops/ or google/ops/), e.g. a raw
  # kernel package that this implementation wraps, that share its backend.
  kernel_dirs: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    _check_name(self, self.name)


@dataclasses.dataclass(frozen=True)
class Op:
  """Operator metadata."""

  name: str
  apis: tuple[str, ...]
  impls: tuple[Impl, ...]

  def __post_init__(self) -> None:
    _check_name(self, self.name)

  @property
  def op_dir(self) -> str:
    """Op directory relative to _src/ops/ or google/ops/."""
    for impl in self.impls:
      rel_path: str | None = None
      if impl.module.startswith("_src/ops/"):
        rel_path = impl.module.removeprefix("_src/ops/")
      if rel_path is not None:
        parts = rel_path.split("/")
        if parts[0] == "experimental" and len(parts) > 1:
          return f"experimental/{parts[1]}"
        return parts[0]
    return self.name

  @property
  def dirs(self) -> tuple[str, ...]:
    """op_dir followed by the kernel_dirs of all impls."""
    dirs = [self.op_dir]
    for impl in self.impls:
      dirs.extend(impl.kernel_dirs)
    return tuple(dict.fromkeys(dirs))

  @property
  def experimental(self) -> bool:
    return self.op_dir.startswith("experimental/")


OPS: tuple[Op, ...] = (
    Op(
        name="splash_attention",
        apis=("tokamax.ops.splash_attention",),
        impls=(
            Impl(
                name="base",
                backend=XLA,
                module="_src/ops/splash_attention/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/splash_attention/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="batched_rpa",
        apis=(
            "tokamax.ops.batched_ragged_paged_attention",
            "tokamax.ops.ragged_paged_attention",
        ),
        impls=(
            Impl(
                name="reference",
                backend=XLA,
                module="_src/ops/batched_rpa/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/batched_rpa/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="causal_conv1d_gated_delta_rule",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/causal_conv1d_gated_delta_rule/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/causal_conv1d_gated_delta_rule/"
                    "pallas_mosaic_tpu.py"
                ),
            ),
        ),
    ),
    Op(
        name="deepseek_rope",
        apis=("tokamax.ops.deepseek_rope",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/deepseek_rope/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/deepseek_rope/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="proj_and_save_state",
        apis=("tokamax.ops.proj_and_save_state",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/proj_and_save_state/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/proj_and_save_state/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="dsv4_compressor",
        apis=("tokamax.ops.dsv4_compressor",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/dsv4_compressor/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/dsv4_compressor/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="lightning_indexer",
        apis=("tokamax.ops.lightning_indexer",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/lightning_indexer/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/lightning_indexer/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="topk",
        apis=("tokamax.ops.top_k",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/topk/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/topk/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="csa_gather",
        apis=("tokamax.ops.csa_gather",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/csa_gather/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/csa_gather/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="sparse_mla",
        apis=("tokamax.ops.sparse_mla",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/sparse_mla/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/sparse_mla/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="masked_dense_mla",
        apis=("tokamax.ops.masked_dense_mla",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/masked_dense_mla/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/masked_dense_mla/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="mla_swa",
        apis=("tokamax.ops.experimental.mla_swa",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/mla_swa/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/mla_swa/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="grouped_lowrank_proj",
        apis=("tokamax.ops.grouped_lowrank_proj",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/grouped_lowrank_proj/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/grouped_lowrank_proj/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="attention",
        apis=("tokamax.dot_product_attention",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/attention/base.py",
            ),
            Impl(
                name="xla_chunked",
                backend=XLA,
                module="_src/ops/attention/xla_chunked.py",
            ),
            Impl(
                name="cudnn",
                backend=CUDNN,
                module="_src/ops/attention/jax_nn.py",
            ),
            Impl(
                name="mosaic_gpu",
                backend=MOSAIC_GPU,
                module="_src/ops/attention/pallas_mosaic_gpu.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/attention/pallas_mosaic_tpu.py",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/attention/triton.py",
            ),
        ),
    ),
    Op(
        name="kda",
        apis=("tokamax.ops.kimi_delta_attention",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/kda/base.py",
            ),
            Impl(
                name="xla_chunked",
                backend=XLA,
                module="_src/ops/kda/xla_chunked.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/kda/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="flex_attention",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/flex_attention/base.py",
            ),
        ),
    ),
    Op(
        name="mla",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/mla/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/mla/pallas_mosaic_tpu.py",
            ),
            Impl(
                name="mosaic_tpu_v3",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/mla/pallas_mosaic_tpu_v3.py",
            ),
        ),
    ),
    Op(
        name="moba",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/moba/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/moba/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="nsa",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/nsa/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/nsa/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="gdn2",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/gdn2/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/gdn2/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="rwkv7",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/rwkv7/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/rwkv7/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="raven",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/raven/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/raven/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="ragged_dot",
        apis=(
            "tokamax.ragged_dot",
            "tokamax.ragged_dot_general",
        ),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/ragged_dot/base.py",
            ),
            Impl(
                name="mosaic_gpu",
                backend=MOSAIC_GPU,
                module="_src/ops/ragged_dot/pallas_mosaic_gpu.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/ragged_dot/pallas_mosaic_tpu.py",
            ),
            Impl(
                name="mosaic_tpu_v2",
                backend=MOSAIC_TPU,
                module="_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/ragged_dot/triton.py",
            ),
        ),
    ),
    Op(
        name="ragged_gather",
        apis=("tokamax.ops.ragged_gather",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/ragged_gather/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/ragged_gather/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="ragged_gather_reduce",
        apis=("tokamax.ops.ragged_gather_reduce",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/ragged_gather_reduce/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/ragged_gather_reduce/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="router_topk",
        apis=("tokamax.ops.router_topk",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/router_topk/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/router_topk/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="fused_moe",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/fused_moe/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/fused_moe/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="fused_moe_rs",
        apis=(),
        impls=(
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/fused_moe_rs/fused_moe_rs.py",
            ),
        ),
    ),
    Op(
        name="routing_moe",
        apis=(),
        impls=(
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/routing_moe/layer/modular.py",
            ),
        ),
    ),
    Op(
        name="dedup_moe",
        apis=(),
        impls=(
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/dedup_moe/layer/modular.py",
            ),
        ),
    ),
    Op(
        name="fused_moe_decode",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/fused_moe_decode/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/experimental/fused_moe_decode/"
                    "pallas_mosaic_tpu.py"
                ),
            ),
        ),
    ),
    Op(
        name="gmm_v2",
        apis=(),
        impls=(
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/gmm_v2/gmm_v2.py",
            ),
        ),
    ),
    Op(
        name="linear_softmax_cross_entropy_loss",
        apis=("tokamax.linear_softmax_cross_entropy_loss",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/linear_softmax_cross_entropy_loss/base.py",
            ),
            Impl(
                name="chunked_xla",
                backend=XLA,
                module=(
                    "_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py"
                ),
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/linear_softmax_cross_entropy_loss/"
                    "pallas_mosaic_tpu.py"
                ),
            ),
        ),
    ),
    Op(
        name="mhc",
        apis=(
            "tokamax.ops.mhc_pre",
            "tokamax.ops.mhc_post",
            "tokamax.ops.mhc_fused_post_pre",
        ),
        impls=(
            Impl(name="xla", backend=XLA, module="_src/ops/mhc/base.py"),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/mhc/pallas_mosaic_tpu.py",
            ),
        ),
    ),
    Op(
        name="gated_linear_unit",
        apis=("tokamax.gated_linear_unit",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/gated_linear_unit/base.py",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_GPU,
                module="_src/ops/gated_linear_unit/pallas_mosaic_gpu.py",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/gated_linear_unit/triton.py",
            ),
            Impl(
                name="cutedsl",
                backend=CUTEDSL,
                module="_src/ops/gated_linear_unit/cutedsl.py",
            ),
        ),
    ),
    Op(
        name="normalization",
        apis=("tokamax.layer_norm",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/normalization/base.py",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/normalization/triton.py",
            ),
        ),
    ),
    Op(
        name="triangle_multiplication",
        apis=("tokamax.triangle_multiplication",),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/triangle_multiplication/base.py",
            ),
            Impl(
                name="mosaic_gpu",
                backend=MOSAIC_GPU,
                module="_src/ops/triangle_multiplication/pallas_mosaic_gpu.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/triangle_multiplication/pallas_mosaic_tpu.py",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/triangle_multiplication/triton.py",
            ),
        ),
    ),
    Op(
        name="fused_fp8_matmul",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/fused_fp8_matmul/base.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/experimental/fused_fp8_matmul/"
                    "pallas_mosaic_tpu.py"
                ),
            ),
        ),
    ),
    Op(
        name="mhc_lite",
        apis=(),
        impls=(
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/experimental/mhc_lite/mhc_xla.py",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/mhc_lite/mhc_kernels_fwd.py",
            ),
        ),
    ),
)
