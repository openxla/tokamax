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
"""Utilities for benchmarking."""

from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import datetime
import inspect
import time
from typing import Any, Literal, TypeAlias, TypeVar

import google_benchmark
import jax
from jax.experimental.mosaic.gpu import profiler
import numpy as np
from tokamax._src import batching
from tokamax._src import numerics
from tokamax._src import utils

# TODO: Support xprof once it's programmatically available in jaxlib.
xprof_session = None
profile_data = None


PyTree = Any

# Timer functions return the time delta in ms and a dictionary of metadata.
Timer: TypeAlias = Callable[[bool], tuple[float, dict[str, Any]]]

T = TypeVar('T')
RetT: TypeAlias = T | list[jax.Array] | tuple[T, list[jax.Array]]


@jax.custom_vjp
def _optimization_barrier(x: T) -> T:
  return jax.lax.optimization_barrier(x)


_optimization_barrier.defvjp(
    fwd=lambda x: (_optimization_barrier(x), None),
    bwd=lambda _, dout: (dout,),
)


@dataclasses.dataclass(frozen=True)
class BenchmarkData:
  """Time and memory benchmarking data."""

  compile_time_ms: float
  lower_time_ms: float
  evaluation_times_ms: tuple[float, ...]
  metadata: dict[str, Any]

  @property
  def median_evaluation_time_ms(self) -> float:
    return float(np.median(self.evaluation_times_ms))

  def asdict(self) -> dict[str, float | Sequence[float]]:
    """Represent the BenchmarkData object as a dictionary."""
    ret = dataclasses.asdict(self)
    ret['median_evaluation_time_ms'] = self.median_evaluation_time_ms
    return ret


class XprofProfileSession(contextlib.AbstractContextManager):
  """XProf context manager for profiling XLA Ops.

  This is useful for profiling JAX functions in a way that ignores Python
  overhead, useful for benchmarking small kernels.
  The `total_op_time` property should give a similar time to the 'XLA Ops' line
  in the XProf Trace Viewer.

  On GPU, XProf calls CUPTI, which adds dynamic instrumentation that can
  increase the apparent runtime. On TPU, instrumentation is added at compile
  time and relies on HW support for near zero overhead.

  Note: on GPU, any use of this requires building with `--config=cuda`.
  Note: In case of multiple XLA Ops, the one with the most events is used.
  """

  def __init__(self, hermetic: bool = True):
    """Initializer.

    Arguments:
      hermetic: If False, creates XProf server session, with the URL accessible
        via self.xprof_url. If True (default), `self.xprof_url=None` and this
        context manager is hermetic.
    """

    if jax.default_backend() == 'cpu':
      raise ValueError('Profiling XLA:CPU is not currently supported.')

    self._profile = None
    self._xprof_session = None
    self._hermetic = hermetic
    self.xprof_url: str | None = None

  @property
  def total_op_time(self) -> datetime.timedelta:
    """Returns the total device time of XLA operators."""
    profile = self._profile
    if profile is None:
      raise ValueError('XProfProfileSession has not been started.')

    xla_xlines = []
    for xplane in profile.planes:
      if xplane.name.startswith('/device:'):
        for xline in xplane.lines:
          if 'XLA Ops' in xline.name:
            xla_xlines.append(xline)

    if not xla_xlines:
      msg = (
          'No XLA device code executed in the context manager. Check that JAX'
          ' functions inside the context are blocked using'
          ' `jax.block_until_ready`.'
      )
      if jax.default_backend() == 'gpu':
        msg += ' Check also that build flag `--config=cuda` is used.'
      raise ValueError(msg)

    xla_xline = max(xla_xlines, key=lambda x: len(list(x.events)))
    # WARNING: If there are nested ops in the trace, duration_ns will
    # count time both in the parent and children ops.
    duration_ns = sum(e.duration_ns for e in xla_xline.events)

    # timedelta will round to the nearest microsecond, which is the smallest
    # time resolution supported by this object.
    return datetime.timedelta(microseconds=duration_ns / 1000.0)

  def __enter__(self):
    if profile_data is None or xprof_session is None:
      raise ValueError('Xprof modules are missing, cannot use xprof profile.')
    assert profile_data is not None
    assert xprof_session is not None

    self._xprof_session = xprof_session.XprofSession()
    try:
      self._xprof_session.start_session(
          enable_python_tracer=False,
          host_trace_level=2,
      )
    except Exception as e:
      raise RuntimeError('Unable to start xprof session.') from e
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    assert profile_data is not None and self._xprof_session is not None

    del exc_type, exc_tb
    if self._xprof_session is None:
      raise AssertionError('__exit__ called without a prior call to __enter__')
    if self._hermetic:
      xspace = self._xprof_session.end_session_and_get_xspace()
    else:
      xspace, url = self._xprof_session.end_session_and_get_xspace_and_url()
      self.xprof_url = url

    self._profile = profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    )


def standardize_function(
    f: Callable[..., T],
    *args: PyTree,
    kwargs: Mapping[str, PyTree] | None = None,
    mode: Literal[
        'forward', 'forward_res', 'vjp', 'forward_and_vjp'
    ] = 'forward',
    seed: int | None = 0,
) -> tuple[Callable[[list[jax.Array]], RetT], list[jax.Array]]:
  """Creates a standardized function for testing and benchmarking.

  Any jax.ShapeDtypeStruct in kwargs is initialized randomly. In addition,
  output gradients are randomly initialized.

  Arguments:
    f: a JAX function.
    *args: Positional arguments to `f`. Any `jax.ShapeDtypeStruct` objects will
      be replaced with randomly initialized arrays.
    kwargs: Keyword arguments to `f`. Any `jax.ShapeDtypeStruct` objects will be
      replaced with randomly initialized arrays.
    mode: One of 'forward' (default), 'forward_res', 'vjp' or 'forward_and_vjp'.
      'forward' is the standard function evaluation. 'forward_res' is a forward
      pass that computes residuals. 'vjp' computes the VJP-function.
      'forward_and_vjp' computes a full forward and VJP pass. Note that 'vjp'
      will bake in all intermediates into the HLO, which can cause OOM errors.
    seed: The seed used for initializing arrays. If `None`, the arguments are
      not initialized.

  Returns:
    A tuple `(new_function, array_args)`, where `array_args` is a list of all
    arrays in `args`. `new_function(array_args)` will evaluate.
  """
  ba = inspect.signature(f).bind(*args, **({} if kwargs is None else kwargs))
  ba.apply_defaults()

  is_leaf = lambda x: isinstance(x, numerics.InitializableArray)
  args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs), is_leaf=is_leaf)
  is_array = lambda x: isinstance(
      x, (jax.Array, numerics.InitializableArray, jax.ShapeDtypeStruct)
  )
  arrays, other, merge = utils.split_merge(is_array, args_flat)

  def forward(arrays: list[jax.Array]) -> T:
    args, kwargs = args_tree.unflatten(merge(arrays, other))
    return f(*args, **kwargs)

  is_batched = lambda x: isinstance(x, batching.BatchedShapeDtype)
  if any(map(is_batched, arrays)):
    if not all(map(is_batched, arrays)):
      raise ValueError('Cannot mix batched and non-batched arguments.')

    batched = batching.Batched(arrays)
    if batched.vmap_axis_sizes:
      array_vmap_axes = (x.vmap_axes for x in arrays)
      for in_axes in reversed(list(zip(*array_vmap_axes, strict=True))):
        forward = jax.vmap(forward, in_axes=(list(in_axes),))

  if seed is not None:
    arrays = numerics.random_initialize(arrays, seed=seed)
  else:
    # Initializable arrays are recognized as traceable objects by jax
    # facilities (eg. kernel.lower(...)). For that reason we need to
    # initialize them.
    is_init_array = lambda x: isinstance(x, numerics.InitializableArray)
    arrays = jax.tree.map(
        lambda x: numerics.random_initialize(x) if is_init_array(x) else x,
        arrays,
        is_leaf=is_init_array,
    )

  if mode == 'forward':
    func = forward
  elif mode == 'forward_res':
    func = lambda arrays: jax.vjp(forward, arrays)[0]
  elif mode == 'forward_and_vjp':

    def vjp_full(arrays: list[jax.Array]) -> tuple[T, list[jax.Array]]:
      fwd_opt_barrier = lambda x: _optimization_barrier(forward(x))
      out, f_vjp = jax.vjp(fwd_opt_barrier, arrays)
      return out, f_vjp(out)

    func = vjp_full
  elif mode == 'vjp':
    out, f_vjp = jax.vjp(forward, arrays)
    arrays, dout_tree = jax.tree.flatten(out)
    func = lambda arrays: f_vjp(dout_tree.unflatten(arrays))
  else:
    raise ValueError(f'Unsupported mode: {mode}')

  return func, arrays


def wallclock_timer(f: Callable[[T], Any], args: T) -> Timer:
  def timer(_):
    jax.block_until_ready(f(args))  # Warmup.
    start_time = time.perf_counter()
    jax.block_until_ready(f(args))
    return (time.perf_counter() - start_time) * 10**3, {}

  return timer


def cuda_events_timer(f: Callable[[T], Any], args: T) -> Timer:
  timer = profiler.measure(f)
  return lambda _: (timer(args)[1], {})


def cupti_timer(f: Callable[[T], Any], args: T) -> Timer:
  timer = profiler.Cupti(finalize=False).measure(f)
  return lambda _: (timer(args)[1], {})


def xprof_timer(f: Callable[[T], Any], args: T) -> Timer:
  def timer(return_metadata):
    jax.block_until_ready(f(args))  # Warmup.
    with XprofProfileSession(hermetic=not return_metadata) as profile:
      jax.block_until_ready(f(args))

    metadata = dict(xprof_url=profile.xprof_url) if return_metadata else {}
    return profile.total_op_time / datetime.timedelta(milliseconds=1), metadata

  return timer


def hermetic_xprof_timer(f: Callable[[T], Any], args: T) -> Timer:
  timer = xprof_timer(f, args)
  return lambda _: timer(False)


_TIMERS: dict[str, Callable[[Callable[[T], Any], T], Timer]] = {
    'wallclock': wallclock_timer,
    'cuda_events': cuda_events_timer,
    'cupti': cupti_timer,
    'xprof': xprof_timer,
    'hermetic_xprof': hermetic_xprof_timer,
}

# TODO: TPU default should be 'xprof', fix once it's supported outside.
_DEFAULT_TIMING_METHOD = {'gpu': 'cupti', 'tpu': 'wallclock'}
_FALLBACK_TIMING_METHOD = 'wallclock'


def _get_metadata(lowered: jax.stages.Lowered) -> dict[str, Any]:
  del lowered  # Unused.
  return {}  # Overridden internally.


def compile_benchmark(
    f: Callable[[T], Any], x: T
) -> Callable[..., BenchmarkData]:
  """Compiles a function and returns a function to benchmark it.

  Args:
    f: A JITable function.
    x: Input to `f`

  Returns:
    A function to run the benchmark and return a `BenchmarkData` object.
  """
  f = jax.jit(f)
  start_time = time.perf_counter()
  lowered = f.lower(x)
  lowering_time = time.perf_counter() - start_time
  start_time = time.perf_counter()
  f_compiled = lowered.compile()  # TODO: Add test.
  compile_time = time.perf_counter() - start_time

  def runner(x: T, *, iterations: int = 5, method: str | None = None):
    """Runs the compiled benchmark.

    Args:
      x: Input to the compiled function.
      iterations: The number of iterations to evaluate the function for after
        the first iteration.
      method: The timing method. 'wallclock' uses Python `time.perf_counter()`
        to measure blocked JAX function execution time. This works for any XLA
        backend, and does not add any device overhead, but does measure Python
        overhead. 'cuda_events' uses CUDA synchronization events to measure the
        device execution time. If `None`, will pick a sensible default for the
        backend.

    Returns:
      A `BenchmarkData` object.
    """
    if method is None:
      method = _DEFAULT_TIMING_METHOD.get(
          jax.default_backend(), _FALLBACK_TIMING_METHOD
      )

    # TODO: Check if default_backend() is the best way to get the device.
    if method == 'cuda_events':
      if jax.default_backend() != 'gpu':
        raise ValueError('CUDA events are only supported on GPU.')
      f_ = f  # CUDA events needs to `jit` the function.
    elif method == 'cupti':
      if jax.default_backend() != 'gpu':
        raise ValueError('CUPTI profiler is only supported on GPU.')
      f_ = f_compiled
    elif method in ('hermmetic_xprof', 'xprof'):
      if jax.default_backend() not in ('gpu', 'tpu'):
        raise ValueError('XProf profiling is only supported on GPU or TPU.')
      f_ = f_compiled
    else:
      f_ = f_compiled

    # start of timing code
    timer = _TIMERS[method](f_, x)
    times = [timer(False)[0] for _ in range(iterations - 1)]
    # end of timing code

    dt, metadata = timer(True)  # Capture metadata on last iteration.
    return BenchmarkData(
        lower_time_ms=lowering_time * 10**3,
        compile_time_ms=compile_time * 10**3,
        evaluation_times_ms=(*times, dt),
        metadata=_get_metadata(lowered) | metadata,
    )

  return runner


# TODO: Add support for autotuning VJP.
def get_impl_and_metadata(
    impls: dict[str, Callable[..., Any]], impl_name: str, *args, **kwargs
) -> tuple[Callable[..., Any], dict[str, Any]]:
  """Returns the implementation for the given name and arguments.

  If the implementation in an `Op`, the name can be given a suffix indicating
  the config mode, e.g. ':heuristics' or ':autotuned'.

  Args:
    impls: A mapping from implementation name to implementation.
    impl_name: The name of the implementation.
    *args: Positional arguments to bind.
    **kwargs: Keyword arguments to bind.
  """
  impl_name, _, config_mode = impl_name.partition(':')
  impl = impls[impl_name]
  if not hasattr(impl, 'bind'):
    if config_mode:
      raise ValueError('Config modes are only supported for `Op`s.')
    return impl, {}

  ba = impl.bind(*args, **kwargs)
  match config_mode:
    case '':
      config = ba.get_config()
    case 'heuristics':
      config = ba.heuristics_config
    case 'autotuned':
      config = (ba.cached_autotuning_data or ba.autotune()).fastest_config
    case 'autotuned_ignore_cache':
      config = ba.autotune().fastest_config
    case _:
      raise ValueError(f'Unsupported config mode: {config_mode}')
  return impl.with_config(config), dict(config=config)  # pytype: disable=attribute-error


def register_benchmark(
    name: str,
    impl_name: str,
    impl: Callable[..., Any],
    kwargs: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    *,
    mode: Literal[
        'forward', 'forward_res', 'vjp', 'forward_and_vjp'
    ] = 'forward',
    items_processed_fn: Callable[..., int] | None = None,
    raise_on_error: bool = True,
    metadata: dict[str, Any] | None = None,
    **bmark_kwargs: Any,
):
  """Creates and registers a Google benchmark."""

  bmark_name = f'{name}_{mode}_{impl_name}'
  if metadata is None:
    metadata = {}

  @google_benchmark.option.unit(google_benchmark.kMicrosecond)
  @google_benchmark.option.use_manual_time()
  @google_benchmark.option.iterations(1)
  def bmark(state, metadata=metadata):
    kwargs_ = kwargs() if callable(kwargs) else kwargs
    f, x = standardize_function(impl, kwargs=kwargs_, mode=mode)
    skip_fn = lambda e: state.skip_with_error(str(e).lstrip().splitlines()[0])

    try:
      benchmark_data = compile_benchmark(f, x)(x, **bmark_kwargs)
    except NotImplementedError as e:
      skip_fn(e)
      return
    except Exception as e:  # pylint: disable=broad-except
      if raise_on_error:
        raise RuntimeError(f'Benchmark failed: {bmark_name}') from e
      skip_fn(e)
      return

    median = benchmark_data.median_evaluation_time_ms
    min_ = min(benchmark_data.evaluation_times_ms)
    max_ = max(benchmark_data.evaluation_times_ms)
    stddev = np.std(benchmark_data.evaluation_times_ms)
    label = f'min={min_:.3f}, max={max_:.3f}, σ/median={stddev / median:.3f}'
    metadata |= benchmark_data.metadata
    if metadata:
      label += f', {metadata}'

    state.set_iteration_time(median / 1e3)
    state.set_label(label)
    if items_processed_fn is not None:
      state.items_processed = items_processed_fn(**kwargs_)

  google_benchmark.register(bmark, name=bmark_name)


def get_benchmark_registrar(
    impls: dict[str, Callable[..., Any]],
) -> Callable[..., None]:
  """Returns a function that registers benchmarks by implementation name."""

  def registrar(name, impl_name, kwargs, **bmark_kwargs):
    impl = impls[impl_name]
    if hasattr(impl, 'bind') and hasattr(impl, 'with_config'):
      kwargs_ = kwargs() if callable(kwargs) else kwargs
      config = impl.bind(**kwargs_).default_config
      impl = impl.with_config(config)
      is_null_config = type(config).__name__ == 'NullConfig'
      metadata = None if is_null_config else dict(config=config)
    else:
      metadata = None

    register_benchmark(
        name, impl_name, impl, kwargs, metadata=metadata, **bmark_kwargs
    )

  return registrar
