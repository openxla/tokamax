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
"""Pallas Mosaic TPU ops and VJP ops for mHC-lite."""

import dataclasses
import math
from typing import Annotated, Any, ClassVar, override
import jax
import jax.numpy as jnp
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.mhc_lite import base
from tokamax._src.ops.experimental.mhc_lite import common
from tokamax._src.ops.experimental.mhc_lite import mhc_kernels_bwd
from tokamax._src.ops.experimental.mhc_lite import mhc_kernels_fwd


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PreConfig:
  """Autotuning config for `PallasTpuMhcLitePre`."""

  block_size: Annotated[int, pydantic.Field(gt=0)]
  fwd_pipeline_blocks: Annotated[int, pydantic.Field(ge=1)] = 2
  fwd_strip_tokens: Annotated[int, pydantic.Field(gt=0)]
  fwd_band_width: Annotated[int, pydantic.Field(gt=0)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PreVjpConfig:
  """Autotuning config for `PallasTpuMhcLitePreVjp`."""

  bwd_block_size: Annotated[int, pydantic.Field(gt=0)]
  bwd_num_chunks: Annotated[int, pydantic.Field(ge=1)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PostConfig:
  """Autotuning config for `PallasTpuMhcLitePost`."""

  block_size: Annotated[int, pydantic.Field(gt=0)]
  fwd_pipeline_blocks: Annotated[int, pydantic.Field(ge=1)] = 2
  fwd_sub_tokens: Annotated[int, pydantic.Field(gt=0)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PostVjpConfig:
  """Autotuning config for `PallasTpuMhcLitePostVjp`."""

  bwd_block_size: Annotated[int, pydantic.Field(gt=0)]
  bwd_sub_tokens: Annotated[int, pydantic.Field(gt=0)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PostAndPreConfig:
  """Autotuning config for `PallasTpuMhcLitePostAndPre`."""

  block_size: Annotated[int, pydantic.Field(gt=0)]
  fwd_pipeline_blocks: Annotated[int, pydantic.Field(ge=1)] = 2
  fwd_sub_tokens: Annotated[int, pydantic.Field(gt=0)]
  fwd_strip_tokens: Annotated[int, pydantic.Field(gt=0)]
  fwd_band_width: Annotated[int, pydantic.Field(gt=0)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PostAndPreVjpConfig:
  """Autotuning config for `PallasTpuMhcLitePostAndPreVjp`."""

  bwd_block_size: Annotated[int, pydantic.Field(gt=0)]
  bwd_num_chunks: Annotated[int, pydantic.Field(ge=1)]
  bwd_sub_tokens: Annotated[int, pydantic.Field(gt=0)]


@dataclasses.dataclass(frozen=True, slots=True)
class _Heuristics:
  block_size: int
  fwd_pipeline_blocks: int
  fwd_sub_tokens: int
  fwd_strip_tokens: int
  fwd_band_width: int
  bwd_block_size: int
  bwd_num_chunks: int
  bwd_sub_tokens: int


def _to_op_config(cfg_cls: type[Any], resolved: Any) -> Any:
  return cfg_cls(
      **{f.name: getattr(resolved, f.name) for f in dataclasses.fields(cfg_cls)}
  )


def _heuristics_from_ba(
    ba: op.BoundArguments, interpret: bool
) -> _Heuristics:
  """Resolves safe heuristic defaults for `ba`."""
  tokens, streams, embedding, dtype = base.dims_from_ba(ba)
  resolved = common.resolve_config(
      common.MhcKernelConfig(interpret=interpret),
      tokens=tokens,
      streams=streams,
      embedding=embedding,
      dtype=dtype,
  )
  assert (
      resolved.block_size is not None
      and resolved.fwd_pipeline_blocks is not None
      and resolved.fwd_sub_tokens is not None
      and resolved.fwd_strip_tokens is not None
      and resolved.fwd_band_width is not None
      and resolved.bwd_block_size is not None
      and resolved.bwd_num_chunks is not None
      and resolved.bwd_sub_tokens is not None
  )
  return _Heuristics(
      block_size=resolved.block_size,
      fwd_pipeline_blocks=resolved.fwd_pipeline_blocks,
      fwd_sub_tokens=resolved.fwd_sub_tokens,
      fwd_strip_tokens=resolved.fwd_strip_tokens,
      fwd_band_width=resolved.fwd_band_width,
      bwd_block_size=resolved.bwd_block_size,
      bwd_num_chunks=resolved.bwd_num_chunks,
      bwd_sub_tokens=resolved.bwd_sub_tokens,
  )


def _fwd_candidates(ba: op.BoundArguments, interpret: bool):
  """Returns forward tiling candidate ranges for `ba`."""
  tokens, _, embedding, _ = base.dims_from_ba(ba)
  heur = _heuristics_from_ba(ba, interpret)
  tpu_info = common.get_tpu_info()
  num_lanes, num_sublanes = tpu_info.num_lanes, tpu_info.num_sublanes
  fwd_blocks = common.token_block_candidates(
      tokens, num_lanes, num_sublanes, (1, 2, 3, 4, 6)
  )
  fwd_pairs = {(heur.block_size, heur.fwd_pipeline_blocks)}
  for blk in fwd_blocks:
    fwd_pairs.add((blk, 1))
    if blk <= 6 * num_lanes and tokens % (2 * blk) == 0:
      fwd_pairs.add((blk, 2))
  fwd_bands = sorted(
      {
          math.gcd(embedding, b)
          for b in (num_lanes, 2 * num_lanes)
          if embedding >= num_lanes and math.gcd(embedding, b) >= num_lanes
      }
      or {heur.fwd_band_width}
  )
  return tokens, heur, num_lanes, num_sublanes, sorted(fwd_pairs), fwd_bands


def _bwd_candidates(ba: op.BoundArguments, interpret: bool):
  """Returns backward block size candidates for `ba`."""
  tokens, _, _, _ = base.dims_from_ba(ba)
  heur = _heuristics_from_ba(ba, interpret)
  tpu_info = common.get_tpu_info()
  bwd_blocks = common.token_block_candidates(
      tokens, tpu_info.num_lanes, tpu_info.num_sublanes, (1, 2, 3, 4, 6, 8)
  )
  if heur.bwd_block_size not in bwd_blocks:
    bwd_blocks.append(heur.bwd_block_size)
  return tokens, heur, tpu_info.num_sublanes, sorted(set(bwd_blocks))


def configure_op(
    op_instance: Any,
    config: common.MhcKernelConfig,
    *,
    tokens: int,
    embedding: int,
) -> Any:
  """Applies explicit `MhcKernelConfig` overrides while preserving caching."""
  tpu_info = common.get_tpu_info()
  common.validate_config_fields(
      config,
      tpu_info.num_lanes,
      tpu_info.num_sublanes,
      check_fwd=True,
      check_bwd=False,
  )
  fwd_cfg = None
  if config.has_fwd_tiling_overrides():
    resolved = common.resolve_config(
        config, tokens=tokens, embedding=embedding, check_bwd=False
    )
    fwd_cfg = _to_op_config(op_instance.config_cls, resolved)
  bwd_cfg = None
  if config.has_bwd_tiling_overrides():
    resolved = common.resolve_config(
        config,
        tokens=tokens,
        embedding=embedding,
        check_fwd=False,
        check_bwd=False,
    )
    bwd_cfg = _to_op_config(op_instance.vjp.config_cls, resolved)
  vjp = op_instance.vjp.replace(
      config=bwd_cfg,
      interpret=config.interpret,
      vmem_limit_bytes=config.vmem_limit_bytes,
  )
  return op_instance.replace(
      config=fwd_cfg,
      interpret=config.interpret,
      vmem_limit_bytes=config.vmem_limit_bytes,
      vjp=vjp,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PallasMhcMixin:
  """Shared execution flags and device support check for mHC Pallas ops."""

  interpret: bool = False
  vmem_limit_bytes: int | None = None

  def supported_on(self, device: jax.Device) -> bool:
    return (
        self.interpret
        or self.bypass_device_check  # type: ignore[attr-defined]
        or device.platform == "tpu"
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuMhcLitePreVjp(_PallasMhcMixin, base.MhcLitePreVjp[PreVjpConfig]):
  """Pallas TPU VJP op for `PallasTpuMhcLitePre`."""

  config_cls: ClassVar[type[PreVjpConfig]] = PreVjpConfig

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> PreVjpConfig:
    return _to_op_config(PreVjpConfig, _heuristics_from_ba(ba, self.interpret))

  @override
  def _get_autotuning_configs(
      self, ba: op.BoundArguments
  ) -> set[PreVjpConfig]:
    tokens, heur, _, bwd_blocks = _bwd_candidates(ba, self.interpret)
    configs = {_to_op_config(PreVjpConfig, heur)}
    for bwd_blk in bwd_blocks:
      num_bwd_blocks = max(1, tokens // bwd_blk)
      for chunks in {math.gcd(num_bwd_blocks, c) for c in (1, 2, 4)}:
        configs.add(
            PreVjpConfig(bwd_block_size=bwd_blk, bwd_num_chunks=chunks)
        )
    return configs

  @override
  def _fwd(
      self,
      residuals: base.PreResiduals,
      out: base.PreOutputs,
      dout: base.PreOutputs,
      x: jax.Array,
      weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool,
      config: PreVjpConfig,
  ) -> tuple[tuple[jax.Array, common.MhcWeights, jax.Array], None]:
    del out, return_residuals
    d_layer_input, d_x_out, d_h_post, d_residual = dout
    d_ctx = common.MHCContext(
        x=d_x_out,
        h_post=d_h_post,
        residual=d_residual,
        implementation="mosaic_tpu",
    )
    dx, dweights, _ = mhc_kernels_bwd.pre_bwd(
        config,
        (residuals, (x, weights, permutations)),
        (d_layer_input, d_ctx),
        rms_epsilon=rms_epsilon,
        vmem_limit_bytes=self.vmem_limit_bytes,
        interpret=self.interpret,
    )
    return (dx, dweights, jnp.zeros_like(permutations)), None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuMhcLitePre(_PallasMhcMixin, base.MhcLitePre[PreConfig]):
  """Pallas TPU forward op for mHC-lite pre-branch stage."""

  config_cls: ClassVar[type[PreConfig]] = PreConfig

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self,
          "vjp",
          PallasTpuMhcLitePreVjp(
              interpret=self.interpret,
              bypass_device_check=self.bypass_device_check,
              vmem_limit_bytes=self.vmem_limit_bytes,
          ),
      )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> PreConfig:
    return _to_op_config(PreConfig, _heuristics_from_ba(ba, self.interpret))

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[PreConfig]:
    tokens, heur, num_lanes, _, fwd_pairs, fwd_bands = _fwd_candidates(
        ba, self.interpret
    )
    configs = {_to_op_config(PreConfig, heur)}
    for blk, pipe in fwd_pairs:
      chunk = common.chunk_tokens(tokens, blk, pipe)
      for strip in {chunk, math.gcd(chunk, num_lanes)}:
        for band in fwd_bands:
          configs.add(
              PreConfig(
                  block_size=blk,
                  fwd_pipeline_blocks=pipe,
                  fwd_strip_tokens=strip,
                  fwd_band_width=band,
              )
          )
    return configs

  @override
  def _fwd(
      self,
      x: jax.Array,
      weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool,
      config: PreConfig,
  ) -> tuple[base.PreOutputs, base.PreResiduals | None]:
    (layer_input, ctx), (residuals, _) = mhc_kernels_fwd.pre_fwd(
        x,
        weights,
        permutations,
        config,
        rms_epsilon=rms_epsilon,
        pre_mapping_epsilon=pre_mapping_epsilon,
        vmem_limit_bytes=self.vmem_limit_bytes,
        interpret=self.interpret,
    )
    out = (layer_input, ctx.x, ctx.h_post, ctx.residual)
    return out, (residuals if return_residuals else None)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuMhcLitePostVjp(
    _PallasMhcMixin, base.MhcLitePostVjp[PostVjpConfig]
):
  """Pallas TPU VJP op for `PallasTpuMhcLitePost`."""

  config_cls: ClassVar[type[PostVjpConfig]] = PostVjpConfig

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> PostVjpConfig:
    return _to_op_config(
        PostVjpConfig, _heuristics_from_ba(ba, self.interpret)
    )

  @override
  def _get_autotuning_configs(
      self, ba: op.BoundArguments
  ) -> set[PostVjpConfig]:
    _, heur, num_sublanes, bwd_blocks = _bwd_candidates(ba, self.interpret)
    configs = {_to_op_config(PostVjpConfig, heur)}
    for bwd_blk in bwd_blocks:
      for bwd_sub in {
          math.gcd(bwd_blk, s)
          for s in (num_sublanes, 2 * num_sublanes, 4 * num_sublanes)
      }:
        configs.add(
            PostVjpConfig(bwd_block_size=bwd_blk, bwd_sub_tokens=bwd_sub)
        )
    return configs

  @override
  def _fwd(
      self,
      residuals: None,
      out: jax.Array,
      dout: jax.Array,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      *,
      return_residuals: bool,
      config: PostVjpConfig,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
    del residuals, out, return_residuals
    ctx = common.MHCContext(
        x=x, h_post=h_post, residual=residual, implementation="mosaic_tpu"
    )
    d_layer_output, d_ctx = mhc_kernels_bwd.post_bwd(
        config,
        (layer_output, ctx),
        dout,
        vmem_limit_bytes=self.vmem_limit_bytes,
        interpret=self.interpret,
    )
    return (d_layer_output, d_ctx.x, d_ctx.h_post, d_ctx.residual), None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuMhcLitePost(_PallasMhcMixin, base.MhcLitePost[PostConfig]):
  """Pallas TPU forward op for mHC-lite post-branch stream mixing."""

  config_cls: ClassVar[type[PostConfig]] = PostConfig

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self,
          "vjp",
          PallasTpuMhcLitePostVjp(
              interpret=self.interpret,
              bypass_device_check=self.bypass_device_check,
              vmem_limit_bytes=self.vmem_limit_bytes,
          ),
      )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> PostConfig:
    return _to_op_config(PostConfig, _heuristics_from_ba(ba, self.interpret))

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[PostConfig]:
    tokens, heur, _, num_sublanes, fwd_pairs, _ = _fwd_candidates(
        ba, self.interpret
    )
    configs = {_to_op_config(PostConfig, heur)}
    for blk, pipe in fwd_pairs:
      chunk = common.chunk_tokens(tokens, blk, pipe)
      for sub in {
          math.gcd(chunk, s)
          for s in (num_sublanes, 2 * num_sublanes, 4 * num_sublanes)
      }:
        configs.add(
            PostConfig(
                block_size=blk, fwd_pipeline_blocks=pipe, fwd_sub_tokens=sub
            )
        )
    return configs

  @override
  def _fwd(
      self,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      *,
      return_residuals: bool,
      config: PostConfig,
  ) -> tuple[jax.Array, None]:
    del return_residuals
    ctx = common.MHCContext(
        x=x, h_post=h_post, residual=residual, implementation="mosaic_tpu"
    )
    output, _ = mhc_kernels_fwd.post_fwd(
        layer_output,
        ctx,
        config,
        vmem_limit_bytes=self.vmem_limit_bytes,
        interpret=self.interpret,
    )
    return output, None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuMhcLitePostAndPreVjp(
    _PallasMhcMixin, base.MhcLitePostAndPreVjp[PostAndPreVjpConfig]
):
  """Pallas TPU VJP op for `PallasTpuMhcLitePostAndPre`."""

  config_cls: ClassVar[type[PostAndPreVjpConfig]] = PostAndPreVjpConfig

  @override
  def _get_heuristics_config(
      self, ba: op.BoundArguments
  ) -> PostAndPreVjpConfig:
    return _to_op_config(
        PostAndPreVjpConfig, _heuristics_from_ba(ba, self.interpret)
    )

  @override
  def _get_autotuning_configs(
      self, ba: op.BoundArguments
  ) -> set[PostAndPreVjpConfig]:
    tokens, heur, num_sublanes, bwd_blocks = _bwd_candidates(ba, self.interpret)
    configs = {_to_op_config(PostAndPreVjpConfig, heur)}
    for bwd_blk in bwd_blocks:
      num_bwd_blocks = max(1, tokens // bwd_blk)
      for chunks in {math.gcd(num_bwd_blocks, c) for c in (1, 2, 4)}:
        for bwd_sub in {
            math.gcd(bwd_blk, s) for s in (num_sublanes, 2 * num_sublanes)
        }:
          configs.add(
              PostAndPreVjpConfig(
                  bwd_block_size=bwd_blk,
                  bwd_num_chunks=chunks,
                  bwd_sub_tokens=bwd_sub,
              )
          )
    return configs

  @override
  def _fwd(
      self,
      residuals: base.PreResiduals,
      out: base.PreOutputs,
      dout: base.PreOutputs,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      next_weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool,
      config: PostAndPreVjpConfig,
  ) -> tuple[
      tuple[
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          common.MhcWeights,
          jax.Array,
      ],
      None,
  ]:
    del return_residuals
    ctx = common.MHCContext(
        x=x, h_post=h_post, residual=residual, implementation="mosaic_tpu"
    )
    d_next_li, d_x_next, d_next_hp, d_next_res = dout
    d_next_ctx = common.MHCContext(
        x=d_x_next,
        h_post=d_next_hp,
        residual=d_next_res,
        implementation="mosaic_tpu",
    )
    d_layer_output, d_ctx, d_next_weights, _ = mhc_kernels_bwd.post_and_pre_bwd(
        config,
        (residuals, (layer_output, ctx), (out[1], next_weights, permutations)),
        (d_next_li, d_next_ctx),
        rms_epsilon=rms_epsilon,
        vmem_limit_bytes=self.vmem_limit_bytes,
        interpret=self.interpret,
    )
    return (
        d_layer_output,
        d_ctx.x,
        d_ctx.h_post,
        d_ctx.residual,
        d_next_weights,
        jnp.zeros_like(permutations),
    ), None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuMhcLitePostAndPre(
    _PallasMhcMixin, base.MhcLitePostAndPre[PostAndPreConfig]
):
  """Pallas TPU forward op for fused Stage-1 post and Stage-2 pre."""

  config_cls: ClassVar[type[PostAndPreConfig]] = PostAndPreConfig

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self,
          "vjp",
          PallasTpuMhcLitePostAndPreVjp(
              interpret=self.interpret,
              bypass_device_check=self.bypass_device_check,
              vmem_limit_bytes=self.vmem_limit_bytes,
          ),
      )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> PostAndPreConfig:
    return _to_op_config(
        PostAndPreConfig, _heuristics_from_ba(ba, self.interpret)
    )

  @override
  def _get_autotuning_configs(
      self, ba: op.BoundArguments
  ) -> set[PostAndPreConfig]:
    tokens, heur, num_lanes, num_sublanes, fwd_pairs, fwd_bands = (
        _fwd_candidates(ba, self.interpret)
    )
    configs = {_to_op_config(PostAndPreConfig, heur)}
    for blk, pipe in fwd_pairs:
      chunk = common.chunk_tokens(tokens, blk, pipe)
      for sub in {math.gcd(chunk, s) for s in (num_sublanes, 2 * num_sublanes)}:
        for strip in {chunk, math.gcd(chunk, num_lanes)}:
          for band in fwd_bands:
            configs.add(
                PostAndPreConfig(
                    block_size=blk,
                    fwd_pipeline_blocks=pipe,
                    fwd_sub_tokens=sub,
                    fwd_strip_tokens=strip,
                    fwd_band_width=band,
                )
            )
    return configs

  @override
  def _fwd(
      self,
      layer_output: jax.Array,
      x: jax.Array,
      h_post: jax.Array,
      residual: jax.Array,
      next_weights: common.MhcWeights,
      permutations: jax.Array,
      *,
      rms_epsilon: float = 1e-5,
      pre_mapping_epsilon: float = 1e-6,
      return_residuals: bool,
      config: PostAndPreConfig,
  ) -> tuple[base.PreOutputs, base.PreResiduals | None]:
    ctx = common.MHCContext(
        x=x, h_post=h_post, residual=residual, implementation="mosaic_tpu"
    )
    (next_li, next_ctx), (residuals, _, _) = mhc_kernels_fwd.post_and_pre_fwd(
        layer_output,
        ctx,
        next_weights,
        permutations,
        config,
        rms_epsilon=rms_epsilon,
        pre_mapping_epsilon=pre_mapping_epsilon,
        vmem_limit_bytes=self.vmem_limit_bytes,
        interpret=self.interpret,
    )
    out = (next_li, next_ctx.x, next_ctx.h_post, next_ctx.residual)
    return out, (residuals if return_residuals else None)
