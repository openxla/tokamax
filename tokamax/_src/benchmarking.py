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
import logging
import os
import pathlib
import shutil
import tempfile
import time
from typing import Any, Final, Literal, cast, overload

import jax
from jax.experimental.mosaic.gpu import profiler
import numpy as np
from tokamax._src import batching
from tokamax._src import numerics
from tokamax._src import utils

xprof_session, profile_data = None, None  # stubs for internal benchmarking

from xprof.cli.tools import get_kernel_stats_tool

type BenchmarkMode = Literal['forward', 'forward_res', 'vjp', 'forward_and_vjp']
# Timer functions return the time delta in ms and a dictionary of metadata.
type Timer = Callable[[bool], tuple[float, dict[str, Any]]]
type TimingMethod = Literal['wallclock', 'cupti', 'xprof', 'hermetic_xprof']

type PyTree = Any
type Ret[T] = T | tuple[jax.Array, ...] | tuple[T, tuple[jax.Array, ...]]

logger = logging.getLogger(__name__)

# for CI
WORKLOAD_ARTIFACTS_DIR_VARNAME: Final[str] = 'WORKLOAD_ARTIFACTS_DIR'
RETAIN_ARTIFACTS_VARNAME: Final[str] = 'TOKAMAX_DUMP_XPROF'


def get_tempdir(
    prefix: str, dir: str | pathlib.Path | None = None
) -> pathlib.Path:
  if dir is not None:
    dir = pathlib.Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
  return pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=dir))


@jax.custom_vjp
def _optimization_barrier[T](x: T) -> T:
  return jax.lax.optimization_barrier(x)


_optimization_barrier.defvjp(
    fwd=lambda x: (_optimization_barrier(x), None),
    bwd=lambda _, dout: (dout,),
)


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkData:
  """Time and memory benchmarking data."""

  compile_time_ms: float
  lower_time_ms: float
  evaluation_times_ms: tuple[float, ...]
  metadata: dict[str, Any]
  # TODO: Remove default value once all users have been migrated.
  peak_memory_mb: float | None = None

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

  def __init__(
      self,
      hermetic: bool = True,
      use_jax_profiler: bool = False,
      event_filter_regex: str | None = None,
      **xprof_session_kwargs,
  ):
    """Initializer.

    Arguments:
      hermetic: If False, creates XProf server session, with the URL accessible
        via self.xprof_url. If True (default), `self.xprof_url=None` and this
        context manager is hermetic.
      use_jax_profiler: Profile with the jax.profiler API writing a temporary
        profile file instead of invoking xprof directly. If False (default),
        profile with xprof directly.
      event_filter_regex: A regex pattern to include only matching event names.
      **xprof_session_kwargs: Additional keyword arguments to pass to
        `xprof_session.start_session`.
    """

    if jax.default_backend() == 'cpu':
      raise ValueError('Profiling XLA:CPU is not currently supported.')

    self._profile = None
    self._xprof_session = None
    self._hermetic = hermetic
    self.xprof_url: str | None = None
    self._jax_profiler_mode = use_jax_profiler
    if xprof_session is None or profile_data is None:
      self._jax_profiler_mode = True
    self._profiler_wallclock_start_time: float | None = None
    self._profiler_wallclock_time: float | None = None
    self._profile_tempdir: pathlib.Path | None = None
    self._event_filter_regex = event_filter_regex
    self._xprof_session_kwargs = xprof_session_kwargs
    self._retain_artifacts = False
    self._timing_summary: dict[str, Any] | None = None

    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    options.enable_hlo_proto = False
    if jax.default_backend() == 'tpu':
      options.advanced_configuration = {
          'tpu_trace_mode': 'TRACE_ONLY_XLA',
          'tpu_perf_counters': True,
      }
    self._profiler_options = options

  @property
  def total_op_time(self) -> datetime.timedelta:
    """Returns the total active device time of XLA operators.

    This corresponds to the ground-truth device compute duration computed via
    Disjoint Interval Union (a sweep-line algorithm that merges overlapping
    hardware event intervals to measure exact active compute time without
    double-counting; see the xprof CLI docs:
    https://github.com/openxla/xprof/blob/master/plugin/xprof/cli/README.md#kernel-statistics--disjoint-interval-union)
    over hardware event traces by
    ``compute_kernel_stats(include_summary=True)``.

    The timing is pre-computed during ``__exit__`` and cached in
    ``_timing_summary`` rather than evaluated on-demand, because the temporary
    profile trace files are deleted upon context exit (``shutil.rmtree``),
    making direct re-evaluation impossible after the session closes.

    Raises:
      ValueError: If timing summary failed to compute or no XLA code executed.
      RuntimeError: If profiler wallclock time is smaller than parsed profile
        time.
    """
    if self._timing_summary is None:
      raise ValueError(
          'XprofProfileSession has not been started or failed to compute'
          ' timing summary.'
      )
    total_us = self._timing_summary.get('total_device_duration_us', 0.0)
    if total_us == 0:
      raise ValueError(
          'No XLA device code executed in the context manager. Check that JAX'
          ' functions inside the context are blocked using'
          ' `jax.block_until_ready`.'
      )
    if (
        self._profiler_wallclock_time is not None
        and self._profiler_wallclock_time < total_us / 1e6
    ):
      raise RuntimeError(
          f'Profiler wallclock time {self._profiler_wallclock_time:.4e} s is'
          f' smaller than parsed profile time {total_us / 1e6:.4e} s.'
      )
    return datetime.timedelta(microseconds=total_us)

  def __enter__(self):
    self._retain_artifacts = os.environ.get(
        RETAIN_ARTIFACTS_VARNAME, 'false'
    ).lower() in ['true', '1', 't', 'y', 'yes']
    if self._jax_profiler_mode:
      try:
        root_dir = os.environ.get(WORKLOAD_ARTIFACTS_DIR_VARNAME, None)
        self._profile_tempdir = get_tempdir(
            prefix='tokamax_xprof_profile_', dir=root_dir
        )
        # get profiling wallclock time right before the profiling starts
        self._profiler_wallclock_start_time = time.perf_counter()
        self._profiler_wallclock_time = None

        jax.profiler.start_trace(
            self._profile_tempdir, profiler_options=self._profiler_options
        )
        logger.info('Writing JAX profiler trace to: %s', self._profile_tempdir)
      except Exception as e:
        raise RuntimeError('Unable to start jax profiling session.') from e
    else:
      if profile_data is None or xprof_session is None:
        raise ValueError('Xprof modules are missing, cannot use xprof profile.')
      self._xprof_session = xprof_session.XprofSession()
      try:
        # get profiling wallclock time right before the profiling starts
        self._profiler_wallclock_start_time = time.perf_counter()
        self._profiler_wallclock_time = None
        self._xprof_session.start_session(
            enable_python_tracer=False,
            host_trace_level=0,
            perf_counters=False,
            **self._xprof_session_kwargs,
        )
      except Exception as e:
        raise RuntimeError('Unable to start xprof session.') from e
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    del exc_type, exc_tb

    if self._jax_profiler_mode:
      jax.profiler.stop_trace()
      # get profiling wallclock time right after the profiling ends
      end_time = time.perf_counter()
      assert (start_time := self._profiler_wallclock_start_time) is not None
      self._profiler_wallclock_start_time = None
      self._profiler_wallclock_time = end_time - start_time
      assert self._profile_tempdir is not None, 'Profile tempdir should be set.'
      profile_paths = list(
          pathlib.Path(self._profile_tempdir).glob('**/*.xplane.pb')
      )
      if len(profile_paths) != 1:
        raise RuntimeError(
            f'Expected exactly one profile file, but found {len(profile_paths)}'
        )
      profile_path = profile_paths[0]
      self._profile = jax.profiler.ProfileData.from_serialized_xspace(
          profile_path.read_bytes()
      )
      if (
          not self._retain_artifacts
          or WORKLOAD_ARTIFACTS_DIR_VARNAME not in os.environ
      ):
        if self._profile_tempdir is not None and self._profile_tempdir.exists():
          shutil.rmtree(self._profile_tempdir)
      logger.info('JAX profiler trace file written to: %s', profile_path)
      self._profile_tempdir = None
    else:
      assert self._xprof_session is not None
      if self._hermetic:
        xspace = self._xprof_session.end_session_and_get_xspace()
      else:
        xspace, url = self._xprof_session.end_session_and_get_xspace_and_url()
        self.xprof_url = url
      # get profiling wallclock time right after the profiling ends
      end_time = time.perf_counter()
      assert (start_time := self._profiler_wallclock_start_time) is not None
      self._profiler_wallclock_start_time = None
      self._profiler_wallclock_time = end_time - start_time
      assert profile_data is not None
      self._profile = profile_data.ProfileData.from_serialized_xspace(
          xspace.SerializeToString()
      )

    self._compute_timing_summary()

  def _compute_timing_summary(self) -> None:
    """Pre-computes Disjoint Interval Union timings for this session."""
    if self._profile is None:
      return

    trace_matchers = None if (re := self._event_filter_regex) is None else (re,)

    try:
      self._timing_summary = get_kernel_stats_tool.compute_kernel_stats(
          self._profile,
          output_format='dict',
          include_summary=True,
          trace_matchers=trace_matchers,
      )
    except Exception as e:
      raise RuntimeError(
          f'Could not compute timing summary via xprof_cli for {self._profile}.'
      ) from e


_ARRAY_TYPES = (
    jax.Array,
    numerics.ArrayInitializer,
    jax.ShapeDtypeStruct,
    np.ndarray,
)


@overload
def standardize_function[T](
    f: Callable[..., T],
    *args: PyTree,
    kwargs: Mapping[str, PyTree] | None = ...,
    mode: BenchmarkMode = ...,
    seed: int = ...,
) -> tuple[Callable[[Sequence[jax.Array]], Ret[T]], tuple[jax.Array, ...]]:
  ...


@overload
def standardize_function[T](
    f: Callable[..., T],
    *args: PyTree,
    kwargs: Mapping[str, PyTree] | None = ...,
    mode: BenchmarkMode = ...,
    seed: None,
) -> tuple[
    Callable[[Sequence[jax.Array]], Ret[T]],
    tuple[jax.Array | jax.ShapeDtypeStruct, ...],
]:
  ...


def standardize_function[T](
    f: Callable[..., T],
    *args: PyTree,
    kwargs: Mapping[str, PyTree] | None = None,
    mode: BenchmarkMode = 'forward',
    seed: int | None = 0,
) -> tuple[
    Callable[[Sequence[jax.Array]], Ret[T]],
    tuple[jax.Array | jax.ShapeDtypeStruct, ...],
]:
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
      not initialized, but any `batching.BatchedShapeDtype` objects are
      conformed to `jax.ShapeDtypeStruct`s.

  Returns:
    A tuple `(new_function, array_args)`, where `array_args` is a list of all
    arrays in `args`. `new_function(array_args)` will evaluate.
  """
  ba = inspect.signature(f).bind(*args, **({} if kwargs is None else kwargs))
  ba.apply_defaults()

  is_leaf = lambda x: isinstance(x, numerics.ArrayInitializer)
  args_flat, args_tree = jax.tree.flatten((ba.args, ba.kwargs), is_leaf=is_leaf)
  is_array = lambda x: isinstance(x, _ARRAY_TYPES)
  arrays, other, merge = utils.split_merge(is_array, args_flat)

  def forward(arrays: Sequence[jax.Array]) -> T:
    args, kwargs = args_tree.unflatten(merge(arrays, other))
    return f(*args, **kwargs)

  is_batched = lambda x: isinstance(x, batching.BatchedShapeDtype)
  if any(map(is_batched, arrays)):
    if not all(map(is_batched, arrays)):
      raise ValueError('Cannot mix batched and non-batched arguments.')

    if batching.get_vmap_axis_sizes(arrays):
      array_vmap_axes = (
          (None if axis is None else axis[0] for axis in x.vmap_axes)
          for x in arrays
      )
      for in_axes in reversed(list(zip(*array_vmap_axes, strict=True))):
        forward = jax.vmap(forward, in_axes=(in_axes,))

  if seed is None:
    # `standardize_function` should be idempotent, but the vmaps added to
    # functions with `BatchedShapeDtype` arguments would be added twice unless
    # `BatchedShapeDtype` is converted to `jax.ShapeDtypeStruct`s.
    def convert_batched(x):
      if isinstance(x, batching.BatchedShapeDtype):
        return jax.ShapeDtypeStruct(x.vmap_shape, x.dtype)
      return x

    arrays = tuple(convert_batched(x) for x in arrays)
  else:
    arrays = tuple(numerics.random_initialize(arrays, seed=seed))

  if mode == 'forward':
    func = forward
  elif mode == 'forward_res':
    func = lambda arrays: jax.vjp(forward, arrays)[0]
  elif mode == 'forward_and_vjp':

    def vjp_full(arrays):
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

  return cast(Callable[[Sequence[jax.Array]], Ret[T]], func), tuple(arrays)


def wallclock_timer[T](f: Callable[[T], Any], args: T) -> Timer:
  def timer(_):
    jax.block_until_ready(f(args))  # Warmup.
    start_time = time.perf_counter()
    jax.block_until_ready(f(args))
    return (time.perf_counter() - start_time) * 10**3, {}

  return timer


def cupti_timer[T](f: Callable[[T], Any], args: T) -> Timer:
  timer = profiler.Cupti(finalize=False).measure(f)
  return lambda _: (timer(args)[1], {})


def xprof_timer[T](
    f: Callable[[T], Any],
    args: T,
    event_filter_regex: str | None = None,
    use_jax_profiler: bool = False,
) -> Timer:
  def timer(return_metadata):
    jax.block_until_ready(f(args))  # Warmup.
    with XprofProfileSession(
        hermetic=not return_metadata,
        event_filter_regex=event_filter_regex,
        use_jax_profiler=use_jax_profiler,
    ) as profile:
      jax.block_until_ready(f(args))

    metadata = dict(xprof_url=profile.xprof_url) if return_metadata else {}
    return profile.total_op_time / datetime.timedelta(milliseconds=1), metadata

  return timer


def hermetic_xprof_timer[T](
    f: Callable[[T], Any], args: T, event_filter_regex: str | None = None
) -> Timer:
  timer = xprof_timer(
      f,
      args,
      event_filter_regex=event_filter_regex,
      use_jax_profiler=True,
  )
  return lambda _: timer(False)


_TIMERS: dict[str, Callable[[Callable[[Any], Any], Any], Timer]] = {
    'wallclock': wallclock_timer,
    'cupti': cupti_timer,
    'xprof': xprof_timer,
    'hermetic_xprof': hermetic_xprof_timer,
}

_DEFAULT_TIMING_METHOD: Final[dict[str, TimingMethod]] = {
    'gpu': 'cupti',
    'tpu': 'hermetic_xprof',
}
_FALLBACK_TIMING_METHOD = 'wallclock'


def _is_concrete(x):
  if isinstance(x, jax.core.Tracer):
    return x.to_concrete_value() is not None
  return x is not None


def compile_benchmark[T](
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
  f_compiled = lowered.compile()
  compile_time = time.perf_counter() - start_time

  assert (memory_analysis := f_compiled.memory_analysis()) is not None
  peak_mem_mb = memory_analysis.peak_memory_in_bytes / 10**6

  def runner(
      x: T,
      *,
      iterations: int = 5,
      method: TimingMethod | None = None,
      event_filter_regex: str | None = None,
  ) -> BenchmarkData:
    """Runs the compiled benchmark.

    Args:
      x: Input to the compiled function.
      iterations: The number of iterations to evaluate the function for after
        the first iteration.
      method: The timing method. 'wallclock' uses Python `time.perf_counter()`
        to measure blocked JAX function execution time. This works for any XLA
        backend, and does not add any device overhead, but does measure Python
        overhead. 'cupti' uses the CUPTI profiling API to measure the device
        execution time. If `None`, will pick a sensible default for the backend.
      event_filter_regex: By default, the reported timing result sums the
        execution time of all XLA Ops present in `x`. This regex enables
        filtering to consider only a subset of Ops whose event names match the
        pattern.

    Returns:
      A `BenchmarkData` object.
    """
    concrete_inputs = [
        z
        for z in jax.tree.leaves(x)
        if isinstance(z, jax.Array) and _is_concrete(z)
    ]
    if concrete_inputs:
      platform = list(concrete_inputs[0].devices())[0].platform
    else:
      platform = jax.default_backend()
    if method is None:
      method = _DEFAULT_TIMING_METHOD.get(platform, _FALLBACK_TIMING_METHOD)

    if method == 'cupti':
      if platform != 'gpu':
        raise ValueError('CUPTI profiling is only supported on GPU.')
    elif method in ('hermetic_xprof', 'xprof'):
      if platform not in ('gpu', 'tpu'):
        raise ValueError('XProf profiling is only supported on GPU or TPU.')

    if method == 'xprof':
      timer = xprof_timer(f_compiled, x, event_filter_regex=event_filter_regex)
    elif method == 'hermetic_xprof':
      timer = hermetic_xprof_timer(
          f_compiled, x, event_filter_regex=event_filter_regex
      )
    else:
      timer = _TIMERS[method](f_compiled, x)
    times = [timer(False)[0] for _ in range(iterations - 1)]
    dt, metadata = timer(True)  # Capture metadata on last iteration.
    return BenchmarkData(
        lower_time_ms=lowering_time * 10**3,
        compile_time_ms=compile_time * 10**3,
        evaluation_times_ms=(*times, dt),
        peak_memory_mb=peak_mem_mb,
        metadata=metadata,
    )

  return runner


def benchmark[T](
    f: Callable[[T], Any],
    x: T,
    *,
    iterations: int = 5,
    method: TimingMethod | None = None,
    event_filter_regex: str | None = None,
) -> BenchmarkData:
  """Benchmarks a function on a specific input.

  Typically, `f` and `x` are generated by
  `tokamax.benchmarking.standardize_function`.

  Args:
    f: A JITable function with a single argument.
    x: Input to `f`.
    iterations: The number of iterations to evaluate the function for after the
      first iteration.
    method: The timing method. `'wallclock'` uses Python `time.perf_counter()`
      to measure blocked JAX function execution time. This works for any XLA
      backend, and does not add any device overhead, but does measure Python
      overhead. `'cupti'` is only supported on GPU, and uses the CUPTI profiling
      API to measure the device execution time, which adds some small device
      overhead. 'hermetic_xprof' uses XProf as the profiler, and is the
      recommended timing method for TPU. If `None` (default), a sensible default
      is chosen for the backend.
    event_filter_regex: Reported timing sums all XLA operations in `f` by
      default. This regex enables filtering by specific event names to report
      timing for just a subset of events that match the pattern.

  Returns:
    A `BenchmarkData` object.
  """
  res = compile_benchmark(f, x)(
      x,
      iterations=iterations,
      method=method,
      event_filter_regex=event_filter_regex,
  )
  return res


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

  # Import lazily, allowing this module to be used without `google_benchmark`.
  import google_benchmark  # pylint: disable=g-import-not-at-top

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

    while state:
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
    if hasattr(impl, 'bind') and hasattr(impl, 'replace'):
      kwargs_ = kwargs() if callable(kwargs) else kwargs
      try:
        config = impl.bind(**kwargs_).default_config
      except NotImplementedError:
        metadata = None
      else:
        impl = impl.replace(config=config)
        is_null_config = type(config).__name__ == 'NullConfig'
        metadata = None if is_null_config else dict(config=config)
    else:
      metadata = None

    register_benchmark(
        name, impl_name, impl, kwargs, metadata=metadata, **bmark_kwargs
    )

  return registrar
