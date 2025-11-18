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
"""Tokamax autotuner."""
from __future__ import annotations

from collections.abc import Callable
from concurrent import futures
from concurrent.futures import process
import dataclasses
import os
import typing
from typing import Any, ParamSpec, Self, TypeVar, cast

from absl import logging
import immutabledict
from pydantic_core import core_schema as cs
from tokamax._src import benchmarking
from tokamax._src import numerics


_Config = TypeVar("_Config")
_P = ParamSpec("_P")
BenchmarkData = benchmarking.BenchmarkData


class AutotuningData(immutabledict.immutabledict[_Config, BenchmarkData]):
  """Results from autotuning."""

  # This is needed because pytype doesn't know that `__new__` returns a
  # `AutotuningData`.
  def __new__(cls, *args: Any, **kwargs: Any) -> AutotuningData:
    return cast(AutotuningData, super().__new__(cls, *args, **kwargs))

  @property
  def fastest_config(self) -> _Config:
    key_fn = lambda x: x[1].median_evaluation_time_ms
    return min(self.items(), key=key_fn)[0]

  def prune(self) -> Self:
    if not self:
      return self
    config = self.fastest_config
    return AutotuningData({config: self[config]})

  @classmethod
  def __get_pydantic_core_schema__(cls, source, handler):
    assert typing.get_origin(source) is cls
    key_schema = handler.generate_schema(typing.get_args(source)[0])
    value_schema = handler.generate_schema(BenchmarkData)
    dict_schema = cs.dict_schema(cs.json_schema(key_schema), value_schema)
    to_cls_schema = cs.no_info_plain_validator_function(cls)
    from_dict_schema = cs.chain_schema([dict_schema, to_cls_schema])
    return cs.union_schema(
        [cs.is_instance_schema(cls), from_dict_schema],
        serialization=cs.wrap_serializer_function_ser_schema(
            lambda d, handler: handler(dict(d)), schema=dict_schema
        ),
    )


def _compile(fn_factory, config, args, kwargs, *, seed=None):
  fn = fn_factory(config)
  fn, x = benchmarking.standardize_function(fn, *args, kwargs=kwargs, seed=seed)
  return benchmarking.compile_benchmark(fn, x), x


def _benchmark(fn_factory, config, args, kwargs):
  runner, x = _compile(fn_factory, config, args, kwargs, seed=0)
  return runner(x)


class _SyncExecutor(futures.Executor):
  """A "no-op" `Executor` that runs submitted functions synchronously."""

  def submit(self, fn, /, *args, **kwargs):
    future = futures.Future()
    try:
      future.set_result(fn(*args, **kwargs))
    except Exception as e:  # pylint: disable=broad-exception-caught
      future.set_exception(e)
    return future


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Autotuner:
  """Autotuner for configurable JAX functions."""

  compile_executor_fn: Callable[[], futures.Executor] | None = (
      futures.ThreadPoolExecutor
  )
  executor_fn: Callable[[], futures.Executor] = _SyncExecutor
  timeout_seconds: float = 600.0

  def autotune(
      self,
      fn_factory: Callable[[_Config], Callable[_P, Any]],
      configs: set[_Config],
      *args: _P.args,
      **kwargs: _P.kwargs,
  ) -> AutotuningData[_Config]:
    """Autotunes over configs for the given arguments."""
    executor = self.executor_fn()
    executor_args = {}
    timeout = self.timeout_seconds
    if self.compile_executor_fn is not None:
      if isinstance(executor, process.ProcessPoolExecutor):
        raise ValueError(
            "Cannot specify a `compile_executor_fn` when using a"
            " `ProcessPoolExecutor` executor."
        )
      # pytype: disable=wrong-keyword-args
      with self.compile_executor_fn(max_workers=os.cpu_count()) as compile_exec:
        # pytype: enable=wrong-keyword-args
        compiled = {
            compile_exec.submit(_compile, fn_factory, cfg, args, kwargs): cfg
            for cfg in configs
        }
        initialized_args = None  # All configs share the same arguments.
        try:
          for future in futures.as_completed(compiled, timeout=timeout):
            config = compiled[future]
            try:
              compiled_fn, args = future.result()
              if initialized_args is None:
                initialized_args = numerics.random_initialize(args)
              executor_args[config] = (compiled_fn, initialized_args)
            except Exception:  # pylint: disable=broad-exception-caught
              logging.vlog(2, "Config failed to compile: %s", config)
        except TimeoutError:
          slow_configs = [c for c in configs if c not in executor_args]
          logging.vlog(
              2, "Configs timed out during compilation: %s", slow_configs
          )
    else:
      for config in configs:
        executor_args[config] = (_benchmark, fn_factory, config, args, kwargs)

    with executor:
      future_to_config = {
          executor.submit(*args): cfg for cfg, args in executor_args.items()
      }
      results = {}
      try:
        for future in futures.as_completed(future_to_config, timeout=timeout):
          config = future_to_config[future]
          try:
            data = future.result()
          except process.BrokenProcessPool:
            logging.vlog(2, "Config broken: %s", config)
          except Exception:  # pylint: disable=broad-exception-caught
            logging.vlog(2, "Config failed: %s", config)
          else:
            results[config] = data
            logging.vlog(
                1,
                "%s: lowering time (ms): %f, compile time (ms): %f, "
                "execution times (ms): %s, median: %f",
                config,
                data.lower_time_ms,
                data.compile_time_ms,
                data.evaluation_times_ms,
                data.median_evaluation_time_ms,
            )
      except TimeoutError:
        slow_configs = [c for c in configs if c not in results]
        logging.exception("Configs timed out: %s", slow_configs)

    results = AutotuningData(results)
    if results:
      config = results.fastest_config
      logging.vlog(
          1,
          "best config is %s (median execution time: %f ms)",
          config,
          results[config].median_evaluation_time_ms,
      )
    else:
      logging.error("all configs failed for %s", fn_factory)
    return results
