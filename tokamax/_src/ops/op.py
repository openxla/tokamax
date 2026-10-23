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
import collections
from collections.abc import Callable, Hashable, Mapping, Sequence
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import re
import types
from typing import Any, ClassVar, Concatenate, Final, Literal, Self, cast, final, overload, override

from absl import logging
import immutabledict
import jax
from jax.experimental import hijax
from jax.experimental import xla_metadata
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
from tokamax._src.autotuning import autotuner as autotuner_lib
from tokamax._src.autotuning import cache as autotuning_cache

AutotuningData = autotuner_lib.AutotuningData
DeviceKind = autotuning_cache.DeviceKind


@dataclasses.dataclass(frozen=True)  # Make it serializable with pydantic.
class NullConfig:
  ...


_NULL_CONFIG: Final[NullConfig] = NullConfig()
_CAMEL_TO_SNAKE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?!^)([A-Z])")

_ACTIVE_TOKAMAX_PAYLOAD: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tokamax_payload", default=""
)


@contextlib.contextmanager
def _tokamax_metadata(json_data: str):
  """Sets the Tokamax payload, stacking onto an enclosing one outermost-first."""
  prev = _ACTIVE_TOKAMAX_PAYLOAD.get()
  payload = f"{prev}/tokamax:{json_data}" if prev else f"tokamax:{json_data}"
  reset = _ACTIVE_TOKAMAX_PAYLOAD.set(payload)
  try:
    with xla_metadata.set_xla_metadata(xla_metadata_payload=payload):
      yield
  finally:
    _ACTIVE_TOKAMAX_PAYLOAD.reset(reset)


@dataclasses.dataclass(frozen=True)
class Op[**P, T, R, C, K: Hashable](abc.ABC):
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
  op = op.replace(config=config)
  ```

  Implementors of new ops should do the following:
    - Create a base class that inherits from `Op`, e.g. `RaggedDot`.
    - Implement the `bind` method, performing any necessary validation and
      canonicalization of the arguments.
    - Implement the `_fwd` method, providing a default XLA implementation.
    - Optionally, override the `_abstract_eval` method, to evaluate the output
      shapes without evaluating `_fwd`.

  Implementors of new op backends should do the following:
    - Create a subclass of the base class, e.g. `PallasMosaicGpuRaggedDot`.
    - Optionally, define a config class for the backend.
    - Override the `_fwd` method, providing a backend-specific implementation.
    - Optionally, override the `_abstract_eval` method, to check for inputs that
      are valid in the base class, but not in the backend.
    - Optionally, override `_get_heuristics_config` and
      `_get_autotuning_configs`.
    - Optionally, override `supported_on` to specify which devices the backend
      supports.
    - Optionally, set a default `vjp` function.
  """

  config_cls: ClassVar[type[Any]] = NullConfig  # `type[_Config]` not supported.
  # Whether an op allows abstract inputs with `jax.export.symbolic_shape`
  # instances in array shapes.
  supports_symbolic_shapes: ClassVar[bool] = True

  config: C | None = None
  _: dataclasses.KW_ONLY
  bypass_device_check: bool = False
  # VJP function for the op. `vjp` will be passed the residuals and the output
  # of the op, the output gradients, then all arguments passed to the op.
  # The VJP function must return a tuple of gradients for each positional
  # argument, or a dict `{"argname": gradient, ...}`. If a dict is returned, any
  # input array arguments not in the dict will have gradients set to zeros.
  vjp: Callable[Concatenate[R, T, T, P], Any] | None = None

  def __init_subclass__(cls, *args, **kwargs):
    super().__init_subclass__(*args, **kwargs)
    # Pydantic fails to infer the type of `config`, so give it some help.
    cls.__annotations__["config"] = cls.config_cls | None

  @overload
  def __call__(
      self,
      *args: P.args,
      return_residuals: Literal[False] = ...,  # pyrefly: ignore[bad-function-definition]
      **kwargs: P.kwargs,
  ) -> T:
    ...

  @overload
  def __call__(
      self,
      *args: P.args,
      return_residuals: Literal[True],  # pyrefly: ignore[bad-function-definition]
      **kwargs: P.kwargs,
  ) -> tuple[T, R]:
    ...

  def __call__(
      self,
      *args: P.args,
      return_residuals: bool = False,  # pyrefly: ignore[bad-function-definition]
      **kwargs: P.kwargs,
  ) -> T | tuple[T, R]:
    """Applies the operation with the given arguments."""

    if not self.supports_symbolic_shapes:
      if shape.contains_symbolic_shape((args, kwargs)):
        raise NotImplementedError("This op does not support symbolic shapes.")

    ba = self.bind(*args, return_residuals=return_residuals, **kwargs)  # pyrefly: ignore[unexpected-keyword]

    if not (self.bypass_device_check or config_lib.cross_compile.value):
      for device in infer_devices(ba) or {backend.get_default_device()}:
        if not self.supported_on(device):
          raise NotImplementedError(f"Not supported on {device.device_kind}.")

    args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs))
    mask = tuple(map(_is_array, args_flat))
    arrays = tuple(x for x, m in zip(args_flat, mask) if m)
    other = tuple(x for x, m in zip(args_flat, mask) if not m)
    restorer = _BoundArgumentsRestorer(args_tree, other, mask)

    in_avals = tuple(map(jax.typeof, arrays))
    out_aval = self._abstract_eval(restorer(self, in_avals))
    out, residuals = _OpPrimitive(self, in_avals, out_aval, restorer)(*arrays)
    return (out, residuals) if return_residuals else out

  def bind(
      self, *args: P.args, return_residuals: bool = False, **kwargs: P.kwargs  # pyrefly: ignore[bad-function-definition]
  ) -> "BoundArguments":
    """Binds the op to the given arguments."""
    sig = self._fwd_signature
    ba = sig.bind(*args, return_residuals=return_residuals, **kwargs)  # pyrefly: ignore[unexpected-keyword]
    ba.apply_defaults()
    return BoundArguments(self, ba.arguments)

  def replace(self, **kwargs) -> Self:
    new_op = dataclasses.replace(self, **kwargs)
    # NOTE: We cannot just use `dataclasses.replace(self, vjp=vjp)` when `vjp`
    # is `None`, as many ops will set a default VJP in `__post_init__`.
    if "vjp" in kwargs:
      object.__setattr__(new_op, "vjp", kwargs["vjp"])
    return new_op

  def get_autotuning_cache(
      self, device_kind: DeviceKind | None = None
  ) -> dict[K, AutotuningData[C]]:
    self_no_vjp = self.replace(vjp=None)
    final_cache = {}
    if device_kind is None:
      device_kind = backend.get_default_device().device_kind
    if not config_lib.ignore_autotuning_cache.value:
      if (cache := _AUTOTUNING_CACHE.get(self_no_vjp)) is None:
        cache = autotuning_cache.AutotuningCache(
            self_no_vjp, paths=[autotuning_cache.CACHE_PATH]
        )
        _AUTOTUNING_CACHE[self_no_vjp] = cache

      final_cache = cache[device_kind]
      logging.info("Loaded autotuning cache for %s", self_no_vjp)
    else:
      logging.warning("Ignoring autotuning cache for %s", self_no_vjp)

    return final_cache

  @abc.abstractmethod
  def _fwd(self, *args, **kwargs) -> tuple[T, R | None]:
    ...

  def _get_heuristics_config(self, ba: "BoundArguments") -> C:
    """Returns a config based on heuristics."""
    del ba  # Unused.
    if type(self).config_cls is NullConfig:
      return cast(C, _NULL_CONFIG)
    raise NotImplementedError("`_get_heuristics_config` not implemented.")

  def _get_autotuning_cache_key(self, ba: "BoundArguments") -> K:
    """Returns a key for autotuning cache lookup."""
    pos_arg_names = tuple(
        name
        for name, param in ba.signature.parameters.items()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    )
    key = immutabledict.immutabledict((
        *zip(pos_arg_names, _abstractify(ba.args), strict=True),
        *_abstractify(ba.kwargs).items(),
    ))
    return cast(K, key)

  def _get_autotuning_configs(self, ba: "BoundArguments") -> set[C]:
    """Returns configs to autotune."""
    del ba  # Unused.
    return set()

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

  def _abstract_eval(self, ba: "BoundArguments") -> Any:
    """Returns output shapes of the op."""
    args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs))
    mask = tuple(isinstance(x, jax.core.ShapedArray) for x in args_flat)
    arrays = tuple(
        jax.ShapeDtypeStruct(x.shape, x.dtype)
        for x, m in zip(args_flat, mask) if m
    )
    other = tuple(x for x, m in zip(args_flat, mask) if not m)
    restorer = _BoundArgumentsRestorer(args_tree, other, mask)

    def fwd_flat(arrays):
      ba = restorer(self, arrays)
      return self._fwd(*ba.args, config=ba.default_config, **ba.kwargs)

    return jax.tree.map(jax.typeof, jax.eval_shape(fwd_flat, arrays))

  def supported_on(self, device: jax.Device) -> bool:
    """Returns whether this op runs on the given device."""
    del device  # Unused.
    return True

  __pydantic_config__ = pydantic.ConfigDict(arbitrary_types_allowed=True)


def snake_case_name(op: type[Op] | Op) -> str:  # pylint: disable=g-bare-generic
  """Returns the snake_case name of the op implementation."""
  cls = op if isinstance(op, type) else type(op)
  return _CAMEL_TO_SNAKE_PATTERN.sub(r"_\1", cls.__name__).lower()


type AutotuningCache = dict[DeviceKind, dict[Any, AutotuningData[Any]]]

_AUTOTUNING_CACHE: dict[Op, AutotuningCache] = {}  # pylint: disable=g-bare-generic

AUTOTUNING_CACHE_OVERLAY_STACK: Final[
    contextvars.ContextVar[tuple[dict[Op, AutotuningCache], ...]]  # pylint: disable=g-bare-generic
] = contextvars.ContextVar("AUTOTUNING_CACHE_OVERLAY_STACK", default=())
AUTOTUNING_CACHE_OVERLAY_JAX_CONFIG: Final[Any] = jax.make_user_context(())


@final
class AUTO:
  ...


@dataclasses.dataclass(frozen=True, slots=True)
class BoundArguments[C, K: Hashable]:
  """Bound arguments for an op's `__call__` method."""

  op: Op[..., Any, Any, C, K]
  arguments: Mapping[str, Any]

  def __post_init__(self):
    immutable_args = immutabledict.immutabledict(self.arguments)
    object.__setattr__(self, "arguments", immutable_args)

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
    arguments = collections.OrderedDict(self.arguments)
    return inspect.BoundArguments(self.signature, arguments)

  @property
  def vmap_axis_sizes(self) -> tuple[int, ...]:
    return batching.get_vmap_axis_sizes(dict(self.arguments))

  @property
  def default_config(self) -> C:
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
      autotune_configs: set[C] | type[AUTO] | None = None,
      cache_autotuning_results: bool = True,
      allow_heuristics: bool = True,
  ) -> C:
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
        return data.fastest_config

    if autotune_configs is not None:
      return self.autotune(
          autotune_configs, cache_results=cache_autotuning_results
      ).fastest_config

    if allow_heuristics:
      return heuristics_config

    raise ValueError(f"No config found for {self}.")

  @property
  def heuristics_config(self) -> C:
    """Returns a config based on heuristics."""
    return self.op._get_heuristics_config(self)  # pylint: disable=protected-access

  @property
  def autotuning_cache_key(self) -> K:
    """Returns a key for autotuning cache lookup."""
    return self.op._get_autotuning_cache_key(self)  # pylint: disable=protected-access

  @property
  def cached_autotuning_data(self) -> AutotuningData[C] | None:
    """Returns autotuning data from the cache, if available."""
    if config_lib.ignore_autotuning_cache.value:
      return None

    device_kind = infer_device_kind(self)
    if device_kind is None:
      device_kind = backend.get_default_device().device_kind
    key = self.autotuning_cache_key

    for overlay in reversed(AUTOTUNING_CACHE_OVERLAY_STACK.get()):
      data = overlay.get(self.op, {}).get(device_kind, {}).get(key)
      if data is not None:
        return data

    try:
      return self.op.get_autotuning_cache(device_kind)[key]
    except KeyError:
      key = cast(Mapping[str, Any], key)
      json_key_bytes = _get_arg_spec_adapter(self.op).dump_json(dict(key))

      logging.warning(
          "Autotuning cache miss for %s on %s with key %s",
          self.op,
          device_kind,
          str(json_key_bytes, "utf-8"),
      )
      return None

  @property
  def autotuning_configs(self) -> set[C]:
    """Returns the configs used for autotuning when `AUTO` is specified."""
    return {self.heuristics_config} | self.op._get_autotuning_configs(self)  # pylint: disable=protected-access

  def benchmark(
      self, mode: benchmarking.BenchmarkMode = "forward"
  ) -> benchmarking.BenchmarkData:
    """Benchmarks the op with the bound arguments.

    Args:
      mode: The benchmarking mode to use. Defaults to `forward`.

    Returns:
      A `BenchmarkData` object containing the benchmark results.
    """
    f, x = benchmarking.standardize_function(
        self.op, *self.args, kwargs=self.kwargs, mode=mode
    )
    return benchmarking.compile_benchmark(f, x)(x)

  def autotune(
      self,
      configs: set[C] | type[AUTO] = AUTO,
      autotuner: autotuner_lib.Autotuner = autotuner_lib.Autotuner(),
      cache_results: bool = True,
      timeout: float | None = None,
  ) -> AutotuningData[C]:
    """Autotunes the op with the bound arguments."""
    if configs is AUTO:
      configs = self.autotuning_configs
    configs = cast(set[C], configs)

    logging.debug("Autotuning %s(%s)", self.op, self.arguments)
    op_fn = lambda config: self.op.replace(config=config)
    data = autotuner.autotune(
        op_fn,
        configs,
        *self.args,
        timeout=timeout,
        **self.kwargs,
    )
    if cache_results:
      d = self.op.get_autotuning_cache()
      d[self.autotuning_cache_key] = data
    return data

  @property
  def vjp_arg_spec(self) -> dict[str, Any]:
    """Returns VJP arg specification for this op and arguments."""
    kwargs = self.kwargs | dict(return_residuals=True)
    f, x = benchmarking.standardize_function(
        self.op, *self.args, kwargs=kwargs, seed=None
    )
    out, residuals = jax.eval_shape(f, x)
    vjp_arg_spec = dict(self.arguments)
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

  replace = dataclasses.replace


@dataclasses.dataclass(frozen=True, slots=True)
class _BoundArgumentsRestorer:
  """Restores arguments from a flattened list."""

  args_tree: jax.tree_util.PyTreeDef
  other_args: tuple[Any, ...]
  is_array_mask: tuple[bool, ...]

  def __call__[C, K](
      self, op: Op[..., Any, Any, C, K], arrays: Sequence[Any]
  ) -> BoundArguments[C, K]:
    arrays_iter = iter(arrays)
    other_iter = iter(self.other_args)
    flat = [
        next(arrays_iter) if is_arr else next(other_iter)
        for is_arr in self.is_array_mask
    ]
    args, kwargs = self.args_tree.unflatten(flat)
    ba = op._fwd_signature.bind(*args, **kwargs)
    return BoundArguments(op, ba.arguments)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class _ConvertedClosure[**P, T]:
  """A closure converted to a pure function and closed arguments."""

  f: Callable[..., T] = dataclasses.field(metadata=dict(static=True))
  closure_args: tuple[Any, ...] | None
  default_args: tuple[Any, ...] | None
  default_kwargs: dict[str, Any] | None

  @property
  def __wrapped__(self) -> Callable[..., T]:
    return self.f

  def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
    assert inspect.isfunction(self.f)

    if (closure := self.closure_args) is not None:
      closure = tuple(types.CellType(v) for v in self.closure_args)

    f_closure = types.FunctionType(
        self.f.__code__,
        self.f.__globals__,
        self.f.__name__,
        self.default_args,
        closure=closure,
    )
    f_closure.__kwdefaults__ = self.default_kwargs
    return f_closure(*args, **kwargs)


def convert_closure[**P, T](f: Callable[P, T]) -> Callable[P, T]:
  """Converts a closure to a `_ConvertedClosure` with explicit arguments."""
  if not inspect.isfunction(f):
    raise ValueError(f"Expected a function, got {type(f)}")

  if f.__closure__ is None:
    if f.__defaults__ is None and f.__kwdefaults__ is None:
      return f
    closure_args = ()
  else:
    closure_args = tuple(cell.cell_contents for cell in f.__closure__)

  return _ConvertedClosure(f, closure_args, f.__defaults__, f.__kwdefaults__)


def _strip_vjps(x: Any) -> Any:
  if isinstance(x, Op):
    kwargs = {}
    if hasattr(x, "vjp") and x.vjp is not None:
      kwargs["vjp"] = None
    if hasattr(x, "impl") and x.impl is not None:
      kwargs["impl"] = _strip_vjps(x.impl)
    return x.replace(**kwargs) if kwargs else x
  return x


class _OpPrimitive[T, R](batching.ExpandWithVmapData):
  """`Op` primitive to capture the batch environment and handle VJPs."""

  op: Op[..., T, R, Any, Any]
  restorer: _BoundArgumentsRestorer

  def __init__(
      self,
      op: Op[..., T, R, Any, Any],
      in_avals: tuple[Any, ...],
      out_aval: Any,
      restorer: _BoundArgumentsRestorer,
  ):
    self.in_avals = in_avals
    self.out_aval = out_aval
    self.params = dict(op=op, restorer=restorer)
    super().__init__()

  @override
  def expand_with_vmap_data(self, vmap_data, *args) -> tuple[T, R | None]:
    batched_args = tuple(
        batching.BatchedShapeDtype(x.shape, x.dtype, ()) for x in args
    )

    for axis_data, dims in vmap_data:
      size = axis_data.size
      batched_args = tuple(
          batching.BatchedShapeDtype(
              x.shape, x.dtype, (*x.vmap_axes, None if d is None else (d, size))
          )
          for x, d in zip(batched_args, dims, strict=True)
      )

    # We first bind the batched args to get the config (as we need vmap axes).
    ba = self.restorer(self.op, batched_args)
    config = ba.default_config

    # Serialize args into the HLO to allow for, e.g., offline autotuning.
    json_ba = ba.replace(op=_strip_vjps(self.op.replace(config=config)))
    json_data = str(BOUND_ARGS_ADAPTER.dump_json(json_ba), "utf-8")

    ba = self.restorer(self.op, args)
    with _tokamax_metadata(json_data):
      return self.op._fwd(*ba.args, config=config, **ba.kwargs)  # pylint: disable=protected-access

  @override
  def vjp_fwd(self, nzs_in, *args) -> tuple[tuple[T, R], Any]:
    if self.op.vjp is None:
      return hijax.vjp_from_jvp[0](self, nzs_in, *args)

    out, residuals = self._with_kwargs(return_residuals=True)(*args)
    ba = self.restorer(self.op, args)
    if ba.kwargs["return_residuals"]:
      ret = (out, residuals)
    else:
      ret = (out, jax.tree.map(jnp.zeros_like, self.out_aval[1]))
    return ret, (args, out, residuals)

  @override
  def vjp_bwd_retval(self, res, outgrad) -> tuple[jax.Array, ...]:
    if self.op.vjp is None:
      return hijax.vjp_from_jvp[1](self, res, outgrad)
    return super().vjp_bwd_retval(res, outgrad)

  @override
  def vjp_bwd_retval_with_vmap_data(
      self, vmap_data, res, outgrad
  ) -> tuple[jax.Array, ...]:
    del vmap_data
    assert self.op.vjp is not None
    args, out, residuals = res
    is_zero = lambda x: isinstance(x, hijax.Zero)
    to_array = (
        lambda x: hijax.instantiate_zeros(x) if is_zero(x) else jnp.asarray(x)
    )
    dout = jax.tree.map(to_array, outgrad[0], is_leaf=is_zero)
    ba = self.restorer(self.op, args)
    kwargs = ba.kwargs
    kwargs.pop("return_residuals")
    grads = self.op.vjp(residuals, out, dout, *ba.args, **kwargs)

    # We allow the VJP function to return either a tuple of gradients for
    # the positional args, or a dict mapping arg names to gradients.
    if isinstance(grads, dict):
      grads_ba = self.op._fwd_signature.bind_partial(**grads)  # pylint: disable=protected-access
    else:
      grads_ba = self.op._fwd_signature.bind_partial(*grads)  # pylint: disable=protected-access

    # Check that grads is tree with same structure as args.
    dargs = jax.tree.map(lambda _, g: g, ba.args, grads_ba.args)

    for k, v in kwargs.items():
      if (dv := grads_ba.kwargs.get(k)) is None:
        # Set any missing grads to zeros.
        zeros_like = lambda x: jnp.zeros_like(x) if _is_array(x) else x
        grads_ba.arguments[k] = jax.tree.map(zeros_like, v)
      else:
        # Check that grads is tree with same structure as kwarg value.
        grads_ba.arguments[k] = jax.tree.map(lambda _, g: g, v, dv)

    return tuple(filter(_is_array, jax.tree.leaves((dargs, grads_ba.kwargs))))

  @override
  def batch_dim_rule(self, axis_data, in_dims):
    out_dim = None if all(d is None for d in jax.tree.leaves(in_dims)) else 0
    return jax.tree.map(lambda _: out_dim, self.out_aval)

  @override
  def res_batch_dim_rule(self, axis_data, in_dims):
    out_dim = None if all(d is None for d in jax.tree.leaves(in_dims)) else 0
    return (in_dims, out_dim, out_dim)

  @override
  def jvp(self, primals, tangents) -> tuple[tuple[T, R], tuple[T, R]]:
    primals = tuple(map(jnp.asarray, primals))
    is_zero = lambda x: isinstance(x, hijax.Zero)
    to_array = (
        lambda x: hijax.instantiate_zeros(x) if is_zero(x) else jnp.asarray(x)
    )
    tangents_instantiated = jax.tree.map(to_array, tangents, is_leaf=is_zero)
    try:
      return jax.jvp(self.expand, primals, tangents_instantiated)
    except NotImplementedError:
      if self.op.vjp is None:
        raise

    nzs_in = tuple(
        jax.tree.map(lambda t: not is_zero(t), t, is_leaf=is_zero)
        for t in tangents
    )
    primals_out, residuals = self.vjp_fwd(nzs_in, *primals)
    nzs_in_flat = [True] * len(self.in_avals_flat)
    nzs_out_flat = [True] * len(self.out_avals_flat)
    tangents_flat = jax.tree.leaves(tangents_instantiated)
    tangents_out_flat = jax._src.hijax.fake_linear_op(  # pylint: disable=protected-access
        self, nzs_in_flat, nzs_out_flat, residuals, None, *tangents_flat
    )
    tangents_out = jax.tree.unflatten(self.out_tree, tangents_out_flat)
    return primals_out, tangents_out

  @override
  def dce(self, used_outs, live_ins):
    del live_ins
    used_primals, used_res = used_outs
    if any(jax.tree.leaves(used_res)):
      return True, (True, True), self
    elif any(jax.tree.leaves(used_primals)):
      return True, (True, False), self._with_kwargs(return_residuals=False)
    else:
      return False, (False, False), None

  def _with_kwargs(self, **kwargs) -> "_OpPrimitive":
    in_avals = self.in_avals
    for axis_data, dims in self.vmap_data:
      map_aval = functools.partial(jax._src.core.mapped_aval, axis_data.size)  # pylint: disable=protected-access
      maybe_map_aval = lambda a, d: (a if d is None else map_aval(d, a))  # pylint: disable=cell-var-from-loop
      in_avals = jax.tree.map(maybe_map_aval, in_avals, dims)
    ba = self.restorer(self.op, in_avals)
    ba = dataclasses.replace(ba, arguments=dict(ba.arguments) | kwargs)
    out_aval = self.op._abstract_eval(ba)  # pylint: disable=protected-access
    args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs))
    mask = self.restorer.is_array_mask
    other = tuple(x for x, m in zip(args_flat, mask) if not m)
    restorer = _BoundArgumentsRestorer(args_tree, other, mask)
    prim = _OpPrimitive(self.op, in_avals, out_aval, restorer)
    for axis_data, dims in reversed(self.vmap_data):
      prim, _ = prim.batch_prim(axis_data, dims)
    return prim


@functools.lru_cache
def _get_arg_spec_adapter(op: Op) -> pydantic_lib.TypeAdapter[dict[str, Any]]:
  spec = pydantic_lib.get_arg_spec_model(f"{type(op).__name__}Spec", op._fwd_signature)  # pylint: disable=protected-access
  return pydantic_lib.get_adapter(spec)  # pyrefly: ignore[bad-return]


BOUND_ARGS_ADAPTER = pydantic.TypeAdapter(BoundArguments)


def infer_devices(ba: BoundArguments) -> set[jax.Device]:
  """Infers the devices from bound array arguments."""
  devices = set()
  for x in jax.tree.leaves(dict(ba.arguments)):
    if isinstance(x, jax.Array):
      try:
        devices |= x.devices()
      except jax.errors.ConcretizationTypeError:
        pass
  return devices


def infer_device_kind(ba: BoundArguments) -> DeviceKind | None:
  """Infers the device kind from bound array arguments."""
  device_kinds = {d.device_kind for d in infer_devices(ba)}
  if not device_kinds:
    # Fall back to active abstract mesh during export or cross-compilation.
    abstract_mesh = jax.sharding.get_abstract_mesh()
    if abstract_mesh.abstract_device is not None:
      return abstract_mesh.abstract_device.device_kind
    return None
  if len(device_kinds) == 1:
    return device_kinds.pop()
  raise ValueError(f"Multiple device kinds found: {device_kinds}")


_is_array = lambda x: isinstance(x, (jax.Array, np.ndarray))  # pylint: disable=invalid-name


def _abstractify(pytree):
  def abstractify_leaf(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype) if _is_array(x) else x

  return jax.tree.map(abstractify_leaf, pytree)
