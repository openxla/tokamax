# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Base class for ops."""

import abc
from collections.abc import Callable, Mapping
import copy
import dataclasses
import functools
import inspect
import threading
from typing import Any, ClassVar, Concatenate, Final, Generic, Literal, ParamSpec, Self, TypeVar, cast, overload

from absl import logging
import immutabledict
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
import pydantic
from pydantic_core import core_schema as cs
from tokamax._src import batching
from tokamax._src import benchmarking
from tokamax._src import config as config_lib
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import shape
from tokamax._src import utils
from tokamax._src.autotuning import autotuner as autotuner_lib
from tokamax._src.autotuning import cache as autotuning_cache


_P = ParamSpec("_P")
_T = TypeVar("_T")
_Residuals = TypeVar("_Residuals")
_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
AutotuningData = autotuner_lib.AutotuningData
DeviceKind = autotuning_cache.DeviceKind


@dataclasses.dataclass(frozen=True)  # Make it serializable with pydantic.
class NullConfig:
  ...


_NULL_CONFIG: Final[NullConfig] = NullConfig()


@dataclasses.dataclass(frozen=True)
class Op(abc.ABC, Generic[_P, _T, _Residuals, _Config, _Key]):
  """Base class for operations.

  `Op`s are callable. The `__call__` method will perform the following steps:
    - `bind` the arguments, in order to validate and canonicalize them.
    - If the `Op` is configurable (i.e. `_Config` != `NullConfig`), capture the
      `vmap` environment for the call (as this may affect the choice of config).
      The arguments, including `vmap` axes, are also recorded in the HLO,
      allowing for offline autotuning.
    - Retrieve a config for the `Op`.
      - The config is retrieved using `op.bind(...).default_config`. This can be
        configured using the `tokamax_autotuning_cache_miss_fallback` config
        option (see `BoundArguments.default_config`).
    - Call `_fwd` with the canonicalized arguments and config. If `op.vjp` is
      not `None`, it will be used to compute the gradients.

  The `bind` method binds a set of arguments to the `Op`. The `BoundArguments`
  object returned by `bind` can then be used to retrieve a config for the `Op`.
  For example, to retrieve a config from the autotuning cache:
  ```python
  config = op.bind(*args, **kwargs).cached_autotuning_data.fastest_config
  op = op.with_config(config)
  ```

  Implementors of new ops should do the following:
    - Create a base class that inherits from `Op`, e.g. `RaggedDot`.
    - Implement the `bind` method, performing any necessary validation and
      canonicalization of the arguments.
    - Implement the `_fwd` method, providing a default XLA implementation.

  Implementors of new op backends should do the following:
    - Create a subclass of the base class, e.g. `PallasMosaicGpuRaggedDot`.
    - Optionally, define a config class for the backend.
    - Override the `_fwd` method, providing a backend-specific implementation.
    - Optionally, override `_get_heuristics_config` and
      `_get_autotuning_configs`.
    - Optionally, set a default `vjp` function.
  """
  config_cls: ClassVar[type[_Config]] = NullConfig
  # Whether an op allows abstract inputs with `jax.export.symbolic_shape`
  # instances in array shapes.
  supports_symbolic_shapes: ClassVar[bool] = True

  config: _Config | None = None
  _: dataclasses.KW_ONLY
  # VJP function for the op. `vjp` will be passed the residuals and the output
  # of the op, the output gradients, then all arguments passed to the op.
  # The VJP function must return a tuple of gradients for each positional
  # argument, or a dict `{"argname": gradient, ...}`. If a dict is returned, any
  # input array arguments not in the dict will have gradients set to zeros.
  vjp: Callable[Concatenate[_Residuals, _T, _T, _P], Any] | None = None

  @classmethod
  def __init_subclass__(cls, *args, **kwargs):
    super().__init_subclass__(*args, **kwargs)
    # Pydantic fails to infer the type of `config`, so give it some help.
    cls.__annotations__["config"] = cls.config_cls | None

  @overload
  def __call__(
      self,
      *args: _P.args,
      return_residuals: Literal[False] = ...,
      **kwargs: _P.kwargs,
  ) -> _T:
    ...

  @overload
  def __call__(
      self,
      *args: _P.args,
      return_residuals: Literal[True] = ...,
      **kwargs: _P.kwargs,
  ) -> tuple[_T, _Residuals]:
    ...

  def __call__(
      self, *args: _P.args, return_residuals: bool = False, **kwargs: _P.kwargs
  ) -> _T | tuple[_T, _Residuals]:
    """Applies the operation with the given arguments."""

    if not self.supports_symbolic_shapes and shape.contains_symbolic_shape(
        (args, kwargs)
    ):
      raise NotImplementedError("This op does not support symbolic shapes.")

    bind = cast(Callable[_P, Any], self.bind)  # Work around pytype bug.
    ba = bind(*args, **kwargs)
    args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs))
    is_array = lambda x: isinstance(x, jax.Array)
    arrays, other, merge = utils.split_merge(is_array, args_flat)

    @self._capture_batched_args
    def fwd(*arrays, batched_args, return_res=True):
      args, kwargs = args_tree.unflatten(merge(arrays, other))
      kwargs["return_residuals"] = return_res
      ba = self.bind(*args, **kwargs)
      if batched_args is not None:
        bargs, bkwargs = args_tree.unflatten(merge(batched_args.args, other))
        bkwargs["return_residuals"] = return_res
        arguments = self._fwd_signature.bind(*bargs, **bkwargs).arguments
        ba = BoundArguments(self, arguments)

      config = ba.default_config

      # Serialize args into the HLO to allow for, e.g., offline autotuning.
      json_op = copy.copy(self)
      object.__setattr__(json_op, "config", config)
      object.__setattr__(json_op, "vjp", None)
      json_ba = BoundArguments(json_op, _abstractify(dict(ba.arguments)))
      json_data = str(BOUND_ARGS_ADAPTER.dump_json(json_ba), "utf-8")

      with jax.named_scope(f"tokamax:{json_data}"):
        out, residuals = self._fwd(*args, config=config, **kwargs)
      ret = (out, residuals) if return_residuals else out
      return ret, (arrays, out, residuals)

    def f(*arrays):
      return fwd(*arrays, return_res=return_residuals)[0]

    if self.vjp is not None:

      def bwd(residuals, dout):
        arrays, out, residuals = residuals
        dout = dout[0] if return_residuals else dout
        args, kwargs = args_tree.unflatten(merge(arrays, other))
        kwargs.pop("return_residuals")
        grads = self.vjp(residuals, out, dout, *args, **kwargs)  # pytype: disable=wrong-arg-count

        if isinstance(grads, dict):
          grads_ba = ba.signature.bind_partial(**grads)
        else:
          grads_ba = ba.signature.bind_partial(*grads)

        # Check that grads is tree with same structure as args.
        dargs = jax.tree.map(lambda _, g: g, args, grads_ba.args)

        for k, v in kwargs.items():
          if (dv := grads_ba.kwargs.get(k)) is None:
            # Set any missing grads to zeros.
            zeros_like = lambda x: jnp.zeros_like(x) if is_array(x) else x
            grads_ba.arguments[k] = jax.tree.map(zeros_like, v)
          else:
            # Check that grads is tree with same structure as kwarg value.
            grads_ba.arguments[k] = jax.tree.map(lambda _, g: g, v, dv)

        return list(filter(is_array, jax.tree.leaves((dargs, grads_ba.kwargs))))

      f = jax.custom_vjp(f)
      f.defvjp(fwd, bwd, optimize_remat=True)

    return f(*arrays)

  def bind(
      self, *args: _P.args, return_residuals: bool = False, **kwargs: _P.kwargs
  ) -> "BoundArguments":
    """Binds the op to the given arguments."""
    sig = self._fwd_signature
    ba = sig.bind(*args, return_residuals=return_residuals, **kwargs)
    ba.apply_defaults()
    return BoundArguments(self, ba.arguments)

  def with_config(self, config: _Config) -> Self:
    return dataclasses.replace(self, config=config)

  def get_autotuning_cache(
      self, device_kind: DeviceKind | None = None
  ) -> dict[_Key, AutotuningData[_Config]]:
    self_no_vjp = copy.copy(self)
    object.__setattr__(self_no_vjp, "vjp", None)
    if (cache := _AUTOTUNING_CACHE.get(self_no_vjp)) is None:
      cache = autotuning_cache.AutotuningCache(self_no_vjp)
      _AUTOTUNING_CACHE[self_no_vjp] = cache
    if device_kind is None:
      device_kind = backend.get_default_device().device_kind
    return cache[device_kind]

  @abc.abstractmethod
  def _fwd(self, *args, **kwargs) -> tuple[_T, _Residuals | None]:
    ...

  def _get_heuristics_config(self, ba: "BoundArguments") -> _Config:
    """Returns a config based on heuristics."""
    del ba  # Unused.
    return _NULL_CONFIG

  def _get_autotuning_cache_key(self, ba: "BoundArguments") -> _Key:
    """Returns a key for autotuning cache lookup."""
    ba = batched if (batched := ba.batched).vmap_axis_sizes else ba
    pos_arg_names = tuple(
        name
        for name, param in ba.signature.parameters.items()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    )
    return immutabledict.immutabledict((
        *zip(pos_arg_names, _abstractify(ba.args), strict=True),
        *_abstractify(ba.kwargs).items(),
    ))

  def _get_autotuning_configs(self, ba: "BoundArguments") -> set[_Config]:
    """Returns configs to autotune."""
    del ba  # Unused.
    return set()

  def _capture_batched_args(self, fn: Callable[..., _T]) -> Callable[..., _T]:
    try:
      config = self._get_heuristics_config(None)  # pytype: disable=wrong-arg-types
      if config is _NULL_CONFIG:
        # `custom_vmap` doesn't currently support JVP / VJP. In order to reduce
        # the chance of hitting issues, we avoid its use for ops without a
        # config, as they do not use the `batched_args` anyway.
        return lambda *args, **kwargs: fn(*args, batched_args=None, **kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    return batching.capture_batched_args(fn)

  @property
  def signature(self) -> inspect.Signature:
    """Infers signature of the op."""
    # Use `bind` if available (to get default values), otherwise use `_fwd` and
    # infer the signature.
    if self.bind.__func__ is not Op.bind:
      return inspect.signature(self.bind)
    return self._fwd_signature

  @property
  def _fwd_signature(self) -> inspect.Signature:
    sig = inspect.signature(self._fwd)
    params = sig.parameters.copy()
    del params["config"]
    return sig.replace(parameters=tuple(params.values()))

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)


_AUTOTUNING_CACHE: dict[
    Op, dict[DeviceKind, dict[Any, AutotuningData[Any]]]
] = {}
_AUTOTUNING_CACHE_OVERLAY = threading.local()


def get_autotuning_cache_overlay_state() -> Any:
  if not hasattr(_AUTOTUNING_CACHE_OVERLAY, "stack"):
    _AUTOTUNING_CACHE_OVERLAY.stack = []
    if jax.version.__version_info__ >= (0, 7, 2):
      _AUTOTUNING_CACHE_OVERLAY.context = jax.make_user_context(())
  return _AUTOTUNING_CACHE_OVERLAY


class AUTO:
  ...


@dataclasses.dataclass(frozen=True, init=False)
class BoundArguments(Generic[_Config, _Key]):
  """Bound arguments for an op's `__call__` method."""

  op: Op[..., Any, Any, _Config, _Key]  # pytype: disable=invalid-annotation
  arguments: immutabledict.immutabledict[str, Any]

  def __init__(self, op: Op[..., Any, Any, _Config, _Key], arguments: Mapping[str, Any]):  # pytype: disable=invalid-annotation
    object.__setattr__(self, "op", op)
    immutable_args = immutabledict.immutabledict(arguments)
    object.__setattr__(self, "arguments", immutable_args)

    args_flat = jax.tree.leaves(arguments)
    has_batched = any(map(_is_batched, args_flat))
    if has_batched and any(map(_is_non_batched_shaped, args_flat)):
      raise ValueError("Cannot bind both batched and non-batched shapes")

  @property
  def signature(self) -> inspect.Signature:
    return self.op._fwd_signature  # pylint: disable=protected-access

  @property
  def args(self) -> tuple[Any, ...]:
    return self._bound_args.args

  @property
  def kwargs(self) -> dict[str, Any]:
    return self._bound_args.kwargs

  @property
  def _bound_args(self) -> inspect.BoundArguments:
    arguments = jax.tree.map(_as_unbatched, dict(self.arguments))
    return inspect.BoundArguments(self.signature, arguments)

  @property
  def batched(self) -> batching.Batched[inspect.BoundArguments]:
    arguments = jax.tree.map(_as_batched, dict(self.arguments))
    ba = inspect.BoundArguments(self.signature, arguments)
    return batching.Batched(ba)

  @property
  def default_config(self) -> _Config:
    """Returns the default config for the op.

    The default config is determined as follows:
      1. If `op.config` is not `None`, return it.
      2. If `cached_autotuning_data` is not `None`, return the fastest config.
      3. If the `tokamax_autotuning_cache_miss_fallback` config option is set to
        'autotune', call `autotune` and return the fastest config.
      4. If the `tokamax_autotuning_cache_miss_fallback` config option is set to
        'heuristics', return `heuristics_config`.
      5. Otherwise, i.e. the `tokamax_autotuning_cache_miss_fallback` config
        option is set to 'error', an error will be raised.

    Returns:
      The default config for the op.
    """
    cache_miss_fallback = config_lib.autotuning_cache_miss_fallback.value
    return self.get_config(
        autotune_configs=AUTO if cache_miss_fallback == "autotune" else None,
        allow_heuristics=cache_miss_fallback != "error",
    )

  def get_config(
      self,
      check_autotuning_cache: bool = True,
      autotune_configs: set[_Config] | type[AUTO] | None = None,
      cache_autotuning_results: bool = True,
      allow_heuristics: bool = True,
  ) -> _Config:
    """Returns a config.

    Args:
      check_autotuning_cache: Whether to check the autotuning cache.
      autotune_configs: The configs to autotune. If `AUTO`, the configs from
        `autotuning_configs` will be used. If `None`, no autotuning will be
        performed.
      cache_autotuning_results: Whether to cache the results of autotuning.
      allow_heuristics: Whether to allow heuristics to be used.

    Returns:
      A config for the op.
    """
    # TODO: Add logging.
    if (config := self.op.config) is not None:
      return config

    if (heuristics_config := self.heuristics_config) is _NULL_CONFIG:
      return heuristics_config

    if check_autotuning_cache:
      if (data := self.cached_autotuning_data) is not None and data.items():
        return data.fastest_config  # pytype: disable=unbound-type-param

    if autotune_configs is not None:
      return self.autotune(
          autotune_configs, cache_results=cache_autotuning_results
      ).fastest_config  # pytype: disable=unbound-type-param

    if allow_heuristics:
      return heuristics_config

    raise ValueError(f"No config found for {self}.")

  @property
  def heuristics_config(self) -> _Config:
    """Returns a config based on heuristics."""
    return self.op._get_heuristics_config(self)  # pylint: disable=protected-access

  @property
  def autotuning_cache_key(self) -> _Key:
    """Returns a key for autotuning cache lookup."""
    return self.op._get_autotuning_cache_key(self)  # pylint: disable=protected-access

  @property
  def cached_autotuning_data(self) -> AutotuningData[_Config] | None:
    """Returns autotuning data from the cache, if available."""
    device_kind = infer_device_kind(self)
    if device_kind is None:
      device_kind = backend.get_default_device().device_kind
    key = self.autotuning_cache_key

    for overlay in reversed(get_autotuning_cache_overlay_state().stack):
      data = overlay.get(self.op, {}).get(device_kind, {}).get(key)
      if data is not None:
        return data

    try:
      return self.op.get_autotuning_cache()[key]
    except KeyError:
      return None

  @property
  def autotuning_configs(self) -> set[_Config]:
    """Returns the configs used for autotuning when `AUTO` is specified."""
    return self.op._get_autotuning_configs(self)  # pylint: disable=protected-access

  def benchmark(self) -> benchmarking.BenchmarkData:
    """Benchmarks the op with the bound arguments."""
    ba = batched if (batched := self.batched).vmap_axis_sizes else self
    kwargs = ba.kwargs
    f, x = benchmarking.standardize_function(self.op, *ba.args, kwargs=kwargs)
    return benchmarking.compile_benchmark(f, x)(x)

  def autotune(
      self,
      configs: set[_Config] | type[AUTO] = AUTO,
      autotuner: autotuner_lib.Autotuner = autotuner_lib.Autotuner(),
      cache_results: bool = True,
  ) -> AutotuningData[_Config]:
    """Autotunes the op with the bound arguments."""
    if configs is AUTO:
      configs = self.autotuning_configs

    ba = batched if (batched := self.batched).vmap_axis_sizes else self
    args, kwargs = ba.args, ba.kwargs
    logging.debug("Autotuning %s(%s)", self.op, self.arguments)
    data = autotuner.autotune(self.op.with_config, configs, *args, **kwargs)
    if cache_results:
      d = self.op.get_autotuning_cache()
      d[self.autotuning_cache_key] = data
    return data

  @property
  def vjp_arg_spec(self) -> dict[str, Any]:
    """Returns VJP arg specification for this op and arguments."""
    ba = batched if (batched := self.batched).vmap_axis_sizes else self
    kwargs = ba.kwargs | dict(return_residuals=True)
    f, x = benchmarking.standardize_function(self.op, *ba.args, kwargs=kwargs)
    out, residuals = jax.eval_shape(f, x)
    vjp_arg_spec = dict(ba.arguments)
    vjp_arg_spec["residuals"] = residuals
    vjp_arg_spec["out"] = vjp_arg_spec["dout"] = out
    vjp_arg_spec["return_residuals"] = False
    return vjp_arg_spec

  @classmethod
  def __get_pydantic_core_schema__(cls, source, handler):
    assert source is cls
    op_schema = handler.generate_schema(pydantic_lib.AnyInstanceOf[Op])
    dict_fields = dict(
        op=cs.typed_dict_field(op_schema),
        arguments=cs.typed_dict_field(cs.any_schema()),
    )
    dict_schema = cs.typed_dict_schema(dict_fields)

    def serialize(value, handler, info) -> dict[str, Any]:
      arg_spec_adapter = _get_arg_spec_adapter(value.op)
      arguments = arg_spec_adapter.dump_python(dict(value.arguments), info)
      return dict(op=handler(value.op), arguments=arguments)

    def validate(value: dict[str, Any]) -> Any:
      op = value["op"]
      arguments = _get_arg_spec_adapter(op).validate_python(value["arguments"])
      return BoundArguments(op, arguments)

    to_cls_schema = cs.no_info_plain_validator_function(validate)
    from_dict_schema = cs.chain_schema([dict_schema, to_cls_schema])
    return cs.union_schema(
        [cs.is_instance_schema(cls), from_dict_schema],
        serialization=cs.wrap_serializer_function_ser_schema(
            serialize, info_arg=True, schema=op_schema
        ),
    )


@functools.lru_cache
def _get_arg_spec_adapter(op: Op) -> pydantic_lib.TypeAdapter[dict[str, Any]]:
  spec = pydantic_lib.get_arg_spec_model(f"{type(op).__name__}Spec", op._fwd_signature)  # pylint: disable=protected-access
  return pydantic_lib.get_adapter(spec)


BOUND_ARGS_ADAPTER = pydantic.TypeAdapter(BoundArguments)


def infer_device_kind(ba: BoundArguments) -> DeviceKind | None:
  """Infers the device kind from bound array arguments."""
  device_kinds = set()
  for x in jax.tree.leaves(dict(ba.arguments)):
    if isinstance(x, jax.Array):
      device_kinds |= {d.device_kind for d in x.devices()}
  if not device_kinds:
    return None
  if len(device_kinds) == 1:
    return device_kinds.pop()
  raise ValueError(f"Multiple device kinds found: {device_kinds}")


def _abstractify(pytree):
  def abstractify_leaf(x):
    if isinstance(x, (jax.Array, np.ndarray)):
      return jax.ShapeDtypeStruct(x.shape, x.dtype)
    return x

  return jax.tree.map(abstractify_leaf, pytree)


def _is_shaped(x):
  return isinstance(x, (jax.Array, jax.ShapeDtypeStruct, jax.core.ShapedArray))


def _is_batched(x):
  return isinstance(x, batching.BatchedShapeDtype)


def _is_non_batched_shaped(x):
  return _is_shaped(x) and not _is_batched(x)


def _as_batched(x):
  if _is_non_batched_shaped(x):
    return batching.BatchedShapeDtype(x.shape, x.dtype, ())
  return x


def _as_unbatched(x):
  if _is_batched(x):
    return jax.ShapeDtypeStruct(x.inner_shape, x.dtype)
  return x
