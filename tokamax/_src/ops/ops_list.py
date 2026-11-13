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


def _check_cls(cls_name: object) -> None:
  if cls_name is not None and (
      not isinstance(cls_name, str) or not cls_name.isidentifier()
  ):
    raise ValueError(
        f"Impl.cls must be a non-empty identifier str, got {cls_name!r}"
    )


def _check_vjp_targets(owner: "Impl") -> None:
  for target in owner.vjp:
    if not isinstance(target, str) or not target:
      raise ValueError(
          f"Impl {owner.name!r} vjp target must be a non-empty str,"
          f" got {target!r}"
      )


@dataclasses.dataclass(frozen=True)
class Impl:
  """Kernel implementation metadata."""

  name: str
  backend: Backend
  module: str
  cls: str | None = None
  # Extra op directories (relative to _src/ops/ or google/ops/), e.g. a raw
  # kernel package that this implementation wraps, that share its backend.
  kernel_dirs: tuple[str, ...] = ()
  vjp: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    _check_name(self, self.name)
    _check_cls(self.cls)
    _check_vjp_targets(self)


def _check_op_impls(op_name: str, impls: tuple[Impl, ...]) -> None:
  """Validates impl name uniqueness and that vjp targets are forward impls."""
  by_name: dict[str, Impl] = {}
  for impl in impls:
    if impl.name in by_name:
      raise ValueError(
          f"Duplicate impl name {impl.name!r} in op {op_name!r}"
      )
    by_name[impl.name] = impl
  for impl in impls:
    for target in impl.vjp:
      target_impl = by_name.get(target)
      if target_impl is None:
        raise ValueError(
            f"Impl {impl.name!r} in op {op_name!r} references unknown"
            f" impl {target!r}"
        )
      if target_impl.vjp:
        raise ValueError(
            f"Impl {impl.name!r} in op {op_name!r} vjp target {target!r}"
            " must be a forward impl"
        )


@dataclasses.dataclass(frozen=True)
class Op:
  """Operator metadata."""

  name: str
  apis: tuple[str, ...]
  impls: tuple[Impl, ...]

  def __post_init__(self) -> None:
    _check_name(self, self.name)
    _check_op_impls(self.name, self.impls)

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
                cls="SplashAttention",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/splash_attention/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuSplashAttention",
            ),
            Impl(
                name="base_vjp",
                backend=XLA,
                module="_src/ops/splash_attention/base.py",
                cls="SplashAttentionVjp",
                vjp=("base", "mosaic_tpu"),
            ),
            Impl(
                name="mosaic_tpu_vjp",
                backend=MOSAIC_TPU,
                module="_src/ops/splash_attention/pallas_mosaic_tpu_vjp.py",
                cls="PallasMosaicTpuSplashAttentionVjp",
                vjp=("mosaic_tpu",),
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
                cls="BatchedRpa",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/batched_rpa/pallas_mosaic_tpu.py",
                cls="PallasTpuBatchedRpa",
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
                cls="CausalConv1dGatedDeltaRule",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/causal_conv1d_gated_delta_rule/"
                    "pallas_mosaic_tpu.py"
                ),
                cls="PallasMosaicTpuCausalConv1dGatedDeltaRule",
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
                cls="DeepseekRope",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/deepseek_rope/pallas_mosaic_tpu.py",
                cls="PallasTpuDeepseekRope",
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
                cls="ProjAndSaveState",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/proj_and_save_state/pallas_mosaic_tpu.py",
                cls="PallasTpuProjAndSaveState",
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
                cls="Dsv4Compressor",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/dsv4_compressor/pallas_mosaic_tpu.py",
                cls="PallasTpuDsv4Compressor",
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
                cls="LightningIndexer",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/lightning_indexer/pallas_mosaic_tpu.py",
                cls="PallasTpuLightningIndexer",
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
                cls="TopK",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/topk/pallas_mosaic_tpu.py",
                cls="PallasTpuTopK",
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
                cls="CsaGather",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/csa_gather/pallas_mosaic_tpu.py",
                cls="PallasTpuCsaGather",
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
                cls="SparseMla",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/sparse_mla/pallas_mosaic_tpu.py",
                cls="PallasTpuSparseMla",
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
                cls="MaskedDenseMla",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/masked_dense_mla/pallas_mosaic_tpu.py",
                cls="PallasTpuMaskedDenseMla",
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
                cls="MlaSwa",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/mla_swa/pallas_mosaic_tpu.py",
                cls="PallasTpuMlaSwa",
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
                cls="GroupedLowrankProj",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/grouped_lowrank_proj/pallas_mosaic_tpu.py",
                cls="PallasTpuGroupedLowrankProj",
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
                cls="DotProductAttention",
            ),
            Impl(
                name="xla_chunked",
                backend=XLA,
                module="_src/ops/attention/xla_chunked.py",
                cls="XlaChunkedDotProductAttention",
            ),
            Impl(
                name="cudnn",
                backend=CUDNN,
                module="_src/ops/attention/jax_nn.py",
                cls="JaxNnDotProductAttention",
            ),
            Impl(
                name="mosaic_gpu",
                backend=MOSAIC_GPU,
                module="_src/ops/attention/pallas_mosaic_gpu.py",
                cls="PallasMosaicGpuFlashAttention",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/attention/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuFlashAttention",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/attention/triton.py",
                cls="TritonFlashAttention",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/attention/base.py",
                cls="DotProductAttentionVjp",
                vjp=(
                    "xla",
                    "xla_chunked",
                    "cudnn",
                    "mosaic_gpu",
                    "mosaic_tpu",
                    "triton",
                ),
            ),
            Impl(
                name="mosaic_gpu_vjp",
                backend=MOSAIC_GPU,
                module="_src/ops/attention/pallas_mosaic_gpu_vjp.py",
                cls="PallasMosaicGpuFlashAttentionVjp",
                vjp=("mosaic_gpu",),
            ),
            Impl(
                name="mosaic_tpu_vjp",
                backend=MOSAIC_TPU,
                module="_src/ops/attention/pallas_mosaic_tpu_vjp.py",
                cls="PallasMosaicTpuFlashAttentionVjp",
                vjp=("mosaic_tpu",),
            ),
            Impl(
                name="triton_vjp",
                backend=TRITON,
                module="_src/ops/attention/triton_vjp.py",
                cls="TritonFlashAttentionVjp",
                vjp=("triton",),
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
                cls="KimiDeltaAttention",
            ),
            Impl(
                name="xla_chunked",
                backend=XLA,
                module="_src/ops/kda/xla_chunked.py",
                cls="XlaChunkedKimiDeltaAttention",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/kda/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuKimiDeltaAttention",
            ),
            Impl(
                name="xla_chunked_vjp",
                backend=XLA,
                module="_src/ops/kda/xla_chunked.py",
                cls="XlaChunkedKimiDeltaAttentionVjp",
                vjp=("xla_chunked",),
            ),
            Impl(
                name="mosaic_tpu_vjp",
                backend=MOSAIC_TPU,
                module="_src/ops/kda/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuKimiDeltaAttentionVjp",
                vjp=("mosaic_tpu",),
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
                cls="FlexAttention",
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
                cls="MultiHeadLatentAttention",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/mla/pallas_mosaic_tpu.py",
                cls="PallasTpuMultiHeadLatentAttention",
                kernel_dirs=(
                ),
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
                cls="MixtureOfBlockAttention",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/moba/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuMixtureOfBlockAttention",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/experimental/moba/base.py",
                cls="MixtureOfBlockAttentionVjp",
                vjp=("xla", "mosaic"),
            ),
            Impl(
                name="mosaic_vjp",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/moba/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuMixtureOfBlockAttentionVjp",
                vjp=("mosaic",),
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
                cls="NativeSparseAttention",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/nsa/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuNativeSparseAttention",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/experimental/nsa/base.py",
                cls="NativeSparseAttentionVjp",
                vjp=("xla", "mosaic"),
            ),
            Impl(
                name="mosaic_vjp",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/nsa/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuNativeSparseAttentionVjp",
                vjp=("mosaic",),
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
                cls="GatedDeltaNet2",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/gdn2/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuGatedDeltaNet2",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/experimental/gdn2/base.py",
                cls="GatedDeltaNet2Vjp",
                vjp=("xla", "mosaic"),
            ),
            Impl(
                name="mosaic_vjp",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/gdn2/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuGatedDeltaNet2Vjp",
                vjp=("mosaic",),
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
                cls="Rwkv7",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/rwkv7/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuRwkv7",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/experimental/rwkv7/base.py",
                cls="Rwkv7Vjp",
                vjp=("xla", "mosaic"),
            ),
            Impl(
                name="mosaic_vjp",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/rwkv7/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuRwkv7Vjp",
                vjp=("mosaic",),
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
                cls="RavenGsa",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/raven/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuRavenGsa",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/experimental/raven/base.py",
                cls="RavenGsaVjp",
                vjp=("xla", "mosaic"),
            ),
            Impl(
                name="mosaic_vjp",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/raven/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuRavenGsaVjp",
                vjp=("mosaic",),
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
                cls="RaggedDot",
            ),
            Impl(
                name="mosaic_gpu",
                backend=MOSAIC_GPU,
                module="_src/ops/ragged_dot/pallas_mosaic_gpu.py",
                cls="PallasMosaicGpuRaggedDot",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/ragged_dot/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuRaggedDot",
            ),
            Impl(
                name="mosaic_tpu_v2",
                backend=MOSAIC_TPU,
                module="_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py",
                cls="PallasMosaicTpuV2RaggedDot",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/ragged_dot/triton.py",
                cls="TritonRaggedDot",
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
                cls="RaggedGather",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_TPU,
                module="_src/ops/ragged_gather/pallas_mosaic_tpu.py",
                cls="PallasTpuRaggedGather",
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
                cls="RaggedGatherReduce",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/ragged_gather_reduce/pallas_mosaic_tpu.py",
                cls="PallasTpuRaggedGatherReduce",
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
                cls="RouterTopK",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/router_topk/pallas_mosaic_tpu.py",
                cls="PallasTpuRouterTopK",
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
                cls="FusedMoe",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/fused_moe/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuFusedMoe",
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
                cls="FusedMoeDecode",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/experimental/fused_moe_decode/"
                    "pallas_mosaic_tpu.py"
                ),
                cls="PallasMosaicTpuFusedMoeDecode",
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
                cls="LinearSoftmaxCrossEntropyLoss",
            ),
            Impl(
                name="chunked_xla",
                backend=XLA,
                module=(
                    "_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py"
                ),
                cls="ChunkedXlaLinearSoftmaxCrossEntropyLoss",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/linear_softmax_cross_entropy_loss/"
                    "pallas_mosaic_tpu.py"
                ),
                cls="PallasMosaicTpuLinearSoftmaxCrossEntropyLoss",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/linear_softmax_cross_entropy_loss/base.py",
                cls="LinearSoftmaxCrossEntropyLossVjp",
                vjp=("xla", "chunked_xla", "mosaic_tpu"),
            ),
            Impl(
                name="chunked_xla_vjp",
                backend=XLA,
                module=(
                    "_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py"
                ),
                cls="ChunkedXlaLinearSoftmaxCrossEntropyLossVjp",
                vjp=("chunked_xla",),
            ),
            Impl(
                name="mosaic_tpu_vjp",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/linear_softmax_cross_entropy_loss/"
                    "pallas_mosaic_tpu.py"
                ),
                cls="PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp",
                vjp=("mosaic_tpu",),
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
            Impl(
                name="xla",
                backend=XLA,
                module="_src/ops/mhc/base.py",
                cls="MhcPre",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/mhc/pallas_mosaic_tpu.py",
                cls="PallasTpuMhcPre",
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
                cls="GatedLinearUnit",
            ),
            Impl(
                name="mosaic",
                backend=MOSAIC_GPU,
                module="_src/ops/gated_linear_unit/pallas_mosaic_gpu.py",
                cls="PallasMosaicGpuGatedLinearUnit",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/gated_linear_unit/triton.py",
                cls="TritonGatedLinearUnit",
            ),
            Impl(
                name="cutedsl",
                backend=CUTEDSL,
                module="_src/ops/gated_linear_unit/cutedsl.py",
                cls="CuteDslGatedLinearUnit",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/gated_linear_unit/base.py",
                cls="GatedLinearUnitVjp",
                vjp=("xla", "mosaic", "triton", "cutedsl"),
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
                cls="Normalization",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/normalization/triton.py",
                cls="TritonNormalization",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/normalization/base.py",
                cls="NormalizationVjp",
                vjp=(
                    "xla",
                    "triton",
                ),
            ),
            Impl(
                name="triton_vjp",
                backend=TRITON,
                module="_src/ops/normalization/triton.py",
                cls="TritonNormalizationVjp",
                vjp=("triton",),
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
                cls="TriangleMultiplication",
            ),
            Impl(
                name="mosaic_gpu",
                backend=MOSAIC_GPU,
                module="_src/ops/triangle_multiplication/pallas_mosaic_gpu.py",
                cls="PallasMosaicGpuTriangleMultiplication",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/triangle_multiplication/pallas_mosaic_tpu.py",
                cls="PallasMosaicTpuTriangleMultiplication",
            ),
            Impl(
                name="triton",
                backend=TRITON,
                module="_src/ops/triangle_multiplication/triton.py",
                cls="TritonTriangleMultiplication",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/triangle_multiplication/base.py",
                cls="TriangleMultiplicationVjp",
                vjp=("xla", "mosaic_gpu", "mosaic_tpu", "triton"),
            ),
            Impl(
                name="mosaic_gpu_vjp",
                backend=MOSAIC_GPU,
                module=(
                    "_src/ops/triangle_multiplication/pallas_mosaic_gpu_vjp.py"
                ),
                cls="PallasMosaicGpuTriangleMultiplicationVjp",
                vjp=("mosaic_gpu",),
            ),
            Impl(
                name="mosaic_tpu_vjp",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/triangle_multiplication/pallas_mosaic_tpu_vjp.py"
                ),
                cls="PallasMosaicTpuTriangleMultiplicationVjp",
                vjp=("mosaic_tpu",),
            ),
            Impl(
                name="triton_vjp",
                backend=TRITON,
                module="_src/ops/triangle_multiplication/triton_vjp.py",
                cls="TritonTriangleMultiplicationVjp",
                vjp=("triton",),
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
                cls="FusedFp8Matmul",
            ),
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/experimental/fused_fp8_matmul/"
                    "pallas_mosaic_tpu.py"
                ),
                cls="PallasMosaicTpuFusedFp8Matmul",
            ),
            Impl(
                name="xla_vjp",
                backend=XLA,
                module="_src/ops/experimental/fused_fp8_matmul/base.py",
                cls="FusedFp8MatmulVjp",
                vjp=("xla", "mosaic_tpu"),
            ),
            Impl(
                name="mosaic_tpu_vjp",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/experimental/fused_fp8_matmul/"
                    "pallas_mosaic_tpu_vjp.py"
                ),
                cls="PallasMosaicTpuFusedFp8MatmulVjp",
                vjp=("mosaic_tpu",),
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
    Op(
        name="ssd_scan",
        apis=(),
        impls=(
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module="_src/ops/experimental/ssd_scan/ssd_scan.py",
            ),
        ),
    ),
    Op(
        name="ragged_causal_conv1d",
        apis=(),
        impls=(
            Impl(
                name="mosaic_tpu",
                backend=MOSAIC_TPU,
                module=(
                    "_src/ops/experimental/ragged_causal_conv1d/"
                    "causal_conv1d.py"
                ),
            ),
        ),
    ),
)
