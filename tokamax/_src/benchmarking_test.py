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
import time

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import batching
from tokamax._src import benchmarking
from tokamax._src import numerics


class BenchmarkingTest(parameterized.TestCase):

  def test_standardize_function(self):
    seed = 123

    def f_orig(x, *, y, z, square=True):
      if square:
        return {'out': x**2 + y[0] * y[1] + z}
      else:
        return {'out': x + jnp.sin(y[0]) * y[1] ** 2 + z}

    rng_keys = jax.random.split(jax.random.PRNGKey(0), 2)
    shape, dtype = ((2, 2), jnp.float32)
    square = False
    y = (
        jax.random.normal(rng_keys[0], shape, dtype=dtype),
        jax.random.normal(rng_keys[1], shape, dtype=dtype),
    )
    x = jax.ShapeDtypeStruct(shape, dtype)

    initable = numerics.RangedArrayInitializer(shape, jnp.int32, -10, 10)
    kwargs = {'square': square, 'y': y, 'x': x, 'z': initable}
    kwargs_init = numerics.random_initialize(kwargs, seed=seed)
    out_orig = f_orig(**kwargs_init)

    with self.subTest('Forward mode correctness.'):
      f_eval, x_eval = benchmarking.standardize_function(
          f_orig, kwargs=kwargs, mode='forward', seed=seed
      )
      out_eval = f_eval(x_eval)
      f_res, x_res = benchmarking.standardize_function(
          f_orig, kwargs=kwargs, mode='forward_res', seed=seed
      )
      out_res = f_res(x_res)

      chex.assert_trees_all_equal(x_eval, x_res)
      chex.assert_trees_all_close(out_eval, out_orig)
      chex.assert_trees_all_close(out_res, out_orig)

    with self.subTest('Equivalent positional and keyword args.'):
      f_std, x_std = benchmarking.standardize_function(f_orig, kwargs=kwargs)
      out_kwargs = f_std(x_std)
      f_std, x_std = benchmarking.standardize_function(
          f_orig, x, kwargs=dict(y=y, square=square, z=initable)
      )
      chex.assert_trees_all_equal(f_std(x_std), out_kwargs)

    with self.subTest('Idempotency.'):
      f_std_1, x_std_1 = benchmarking.standardize_function(
          f_orig, kwargs=kwargs
      )
      out_1 = f_std_1(x_std_1)
      f_std_2, x_std_2 = benchmarking.standardize_function(f_std_1, x_std_1)
      chex.assert_trees_all_close(out_1, f_std_2(x_std_2))
      chex.assert_trees_all_equal(x_std_1, x_std_2)

    with self.subTest('VJP correctness.'):
      f_fwd_vjp, args_fwd_vjp = benchmarking.standardize_function(
          f_orig, kwargs=kwargs, mode='forward_and_vjp', seed=seed
      )
      out_fwd_vjp, (dargs_fwd_vjp,) = f_fwd_vjp(args_fwd_vjp)
      golden_out = {
          'out': jnp.array(
              [[-8.127494, -6.5432076], [6.6674175, -0.9291021]],
              dtype=jnp.float32,
          )
      }
      golden_vjp = [
          jnp.array(
              [[-8.127494, -6.5432076], [6.6674175, -0.9291021]],
              dtype=jnp.float32,
          ),
          jnp.array(
              [[-10.421817, -0.026716264], [2.0115848, -0.58659554]],
              dtype=jnp.float32,
          ),
          jnp.array(
              [[-2.561449, 1.8380386], [3.9955664, -0.030029658]],
              dtype=jnp.float32,
          ),
      ]
      # Tests against goldens ensures that initialization is constant over
      # time. Tests both forward and backward consistency.
      chex.assert_trees_all_close(out_fwd_vjp, golden_out, atol=1e-5)
      chex.assert_trees_all_close(dargs_fwd_vjp[:3], golden_vjp, atol=5e-5)
      chex.assert_trees_all_close(out_fwd_vjp, out_orig)

      f_vjp, args_vjp = benchmarking.standardize_function(
          f_orig, kwargs=kwargs, mode='vjp', seed=seed
      )
      chex.assert_trees_all_close(dargs_fwd_vjp[:3], f_vjp(args_vjp)[0][:3])

  def test_standardize_function_with_batching(self):
    seed = 123

    def f_orig(x, y):
      return x + jnp.sin(y[0]) * y[1] ** 2

    x = batching.BatchedShapeDtype((2, 3, 4), jnp.float32, vmap_axes=(0, 0))
    y = (
        batching.BatchedShapeDtype((4, 2), jnp.float32, vmap_axes=(1, None)),
        batching.BatchedShapeDtype((3, 4), jnp.float32, vmap_axes=(None, 0)),
    )
    f, args = benchmarking.standardize_function(f_orig, x, y, seed=seed)

    f_vmap = jax.vmap(f_orig, in_axes=(0, (None, 0)))
    f_vmap = jax.vmap(f_vmap, in_axes=(0, (1, None)))
    x, y = numerics.random_initialize((x, y), seed=seed)
    chex.assert_trees_all_equal(f_vmap(x, y), f(args))

  @parameterized.parameters(benchmarking._TIMERS.keys())
  def test_compile_benchmark(self, method):
    if jax.default_backend() != 'gpu' and method in ('cuda_events', 'cupti'):
      self.skipTest('CUDA timers are only supported on GPU.')

    @jax.jit
    def f(x):
      return jnp.sin(x) ** 2 + 10.0

    num_iters = 11
    kwargs = {'x': jnp.ones((512, 512))}
    f, x = benchmarking.standardize_function(f, kwargs=kwargs)
    run = benchmarking.compile_benchmark(f, x)
    bench = run(x)
    self.assertGreater(bench.median_evaluation_time_ms, 0.0)
    bench = run(x)
    self.assertGreater(bench.median_evaluation_time_ms, 0.0)
    bench = run(x, iterations=num_iters, method=method)
    self.assertGreater(bench.median_evaluation_time_ms, 0.0)
    self.assertGreater(bench.compile_time_ms, 0.0)
    self.assertGreater(bench.lower_time_ms, 0.0)
    self.assertLen(bench.evaluation_times_ms, num_iters)

  def test_benchmark_function_known_time(self):
    time_sleep_s = 0.5

    def f_python(x):
      time.sleep(time_sleep_s)
      return x

    def f(x):
      x = jnp.sin(x) ** 2
      return jax.pure_callback(
          f_python, jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype), x
      )

    kwargs = {'x': jnp.ones((512, 512))}
    f, x = benchmarking.standardize_function(f, kwargs=kwargs)
    run = benchmarking.compile_benchmark(f, x)
    bench = run(x, method='wallclock', iterations=1)
    time_sleep_ms = time_sleep_s * 1000
    self.assertGreaterEqual(bench.median_evaluation_time_ms, time_sleep_ms)

  def test_xprof_profile_session(self):
    def f(x):
      return jnp.sum(x @ x)

    x = jnp.ones((512, 512))
    f = jax.jit(f)

    with self.subTest('Non-zero time.'):
      with benchmarking.XprofProfileSession() as profile:
        jax.block_until_ready(f(x))

      self.assertGreater(profile.total_op_time.total_seconds(), 0)

    with self.subTest('Hermetic mode.'):
      with benchmarking.XprofProfileSession(hermetic=True) as profile:
        jax.block_until_ready(f(x))
      assert profile.total_op_time.total_seconds() > 0  # check is nonzero
      self.assertIsNone(profile.xprof_url)

    with self.subTest('Non-hermetic mode.'):
      with benchmarking.XprofProfileSession(hermetic=False) as profile:
        jax.block_until_ready(f(x))
      assert profile.total_op_time.total_seconds() > 0  # check is nonzero

    with self.subTest('JAX profiler mode.'):
      with benchmarking.XprofProfileSession(use_jax_profiler=True) as profile:
        jax.block_until_ready(f(x))
      assert profile.total_op_time.total_seconds() > 0  # check is nonzero
      self.assertIsNone(profile.xprof_url)

  def test_xprof_profile_session_exception(self):
    with benchmarking.XprofProfileSession(hermetic=True) as profile:
      _ = 1 + 1
    with self.assertRaises(ValueError):
      _ = profile.total_op_time


if __name__ == '__main__':
  jax.config.update('jax_threefry_partitionable', False)
  absltest.main()
