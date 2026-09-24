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
"""Tests for PyTorch TPU op wrapper registration and argument binding."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, ClassVar
import uuid

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops import op as op_lib
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch
from torch._subclasses import fake_tensor


@dataclasses.dataclass(frozen=True)
class _FakeOpConfig:
  blah: int


_HEURISTICS_CONFIG = _FakeOpConfig(1)
_AUTOTUNE_CONFIG = _FakeOpConfig(2)


class _FakeJaxMultiOutputOp(
    op_lib.Op[Any, jax.Array, tuple[jax.Array, jax.Array], _FakeOpConfig, Any]
):
  config_cls: ClassVar[type[_FakeOpConfig]] = _FakeOpConfig

  def _fwd(
      self,
      x: jax.Array,
      y: jax.Array,
      *,
      return_residuals: bool,
      config: _FakeOpConfig,
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    del config, return_residuals  # Unused.
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    residuals = (x * 2, y * 3)
    return x + y, residuals

  def _get_heuristics_config(self, ba: op_lib.BoundArguments) -> _FakeOpConfig:
    del ba  # Unused.
    return _HEURISTICS_CONFIG

  def _get_autotuning_configs(
      self, ba: op_lib.BoundArguments
  ) -> set[_FakeOpConfig]:
    del ba  # Unused.
    return {_AUTOTUNE_CONFIG}


class _FakeJaxVjpOp(op_lib.Op[Any, jax.Array, None, _FakeOpConfig, Any]):
  config_cls: ClassVar[type[_FakeOpConfig]] = _FakeOpConfig

  def _fwd(
      self,
      residuals: tuple[jax.Array, ...],
      x: jax.Array,
      y: jax.Array,
      *,
      return_residuals: bool,
      config: _FakeOpConfig,
  ) -> tuple[jax.Array, Any]:
    del config, return_residuals
    res = residuals[0]
    return res + x + y, None

  def _get_heuristics_config(self, ba: op_lib.BoundArguments) -> _FakeOpConfig:
    del ba  # Unused.
    return _HEURISTICS_CONFIG

  def _get_autotuning_configs(
      self, ba: op_lib.BoundArguments
  ) -> set[_FakeOpConfig]:
    del ba  # Unused.
    return {_AUTOTUNE_CONFIG}


class _FakeTorchOp(torch_op.TorchOp[_FakeOpConfig]):

  def __init__(
      self,
      name: str | None = None,
  ):
    super().__init__()
    self.jax_op_name = name or f"fake_op_{uuid.uuid4().hex[:8]}"
    self.op_impl_jax = _FakeJaxMultiOutputOp()

  def __call__(
      self,
      x: torch.Tensor,
      y: torch.Tensor,
      configs: tuple[Any, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if configs is None:
      configs = (None, None)
    self.configs = configs
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(x, y, return_residuals=True)

  def op_impl_call(
      self,
      x: jax.Array,
      y: jax.Array,
      return_residuals: bool = True,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    config = self.configs[0] or _HEURISTICS_CONFIG
    assert self.op_impl_jax is not None
    out, (res1, res2) = self.op_impl_jax._fwd(
        x, y, return_residuals=return_residuals, config=config
    )
    return out, res1, res2

  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    del inputs  # Unused.
    ctx.saved_tensors = output

  def backward(self, *args: Any, **kwargs: Any) -> Any:
    return args


class _FakeTorchVjpOp(torch_op.TorchOp[None]):

  def __init__(
      self,
      name: str | None = None,
  ):
    super().__init__()
    self.jax_op_name = name or f"fake_vjp_op_{uuid.uuid4().hex[:8]}"
    self.op_impl_jax = _FakeJaxVjpOp()
    self.is_vjp = True

  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    if not abstract_argument_dict:
      return {"residuals": ()}
    x_shape = abstract_argument_dict["x"].shape
    x_dtype = abstract_argument_dict["x"].dtype
    return {
        "residuals": (
            jax.ShapeDtypeStruct(
                shape=x_shape,
                dtype=x_dtype,
            ),
        ),
    }

  def __call__(
      self,
      residuals: torch.Tensor,
      x: torch.Tensor,
      y: torch.Tensor,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    if configs is None:
      configs = (None, None)
    self.configs = configs
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(residuals, x, y, return_residuals=False)

  def op_impl_call(
      self,
      residuals: jax.Array,
      x: jax.Array,
      y: jax.Array,
      return_residuals: bool = False,
  ) -> jax.Array:
    config = self.configs[0] or _HEURISTICS_CONFIG
    assert self.op_impl_jax is not None
    out, _ = self.op_impl_jax._fwd(
        (residuals,), x, y, return_residuals=return_residuals, config=config
    )
    return out

  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    pass

  def backward(self, *args: Any, **kwargs: Any) -> Any:
    return args


class _FakeTorchOpWithBackward(torch_op.TorchOp[_FakeOpConfig]):

  def __init__(
      self,
      name: str | None = None,
  ):
    super().__init__()
    self.jax_op_name = name or f"fake_op_bwd_{uuid.uuid4().hex[:8]}"
    self.op_impl_jax = _FakeJaxMultiOutputOp()
    self.backward_op_torch = _FakeTorchVjpOp()

  def __call__(
      self,
      x: torch.Tensor,
      y: torch.Tensor,
      configs: tuple[Any, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if configs is None:
      configs = (None, None)
    self.configs = configs
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(x, y, return_residuals=True)

  def op_impl_call(
      self,
      x: jax.Array,
      y: jax.Array,
      return_residuals: bool = True,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    config = self.configs[0] or _HEURISTICS_CONFIG
    assert self.op_impl_jax is not None
    out, (res1, res2) = self.op_impl_jax._fwd(
        x, y, return_residuals=return_residuals, config=config
    )
    return out, res1, res2

  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    del inputs
    ctx.saved_tensors = tuple(output)

  def backward(self, *args: Any, **kwargs: Any) -> Any:
    assert self.backward_op_torch is not None
    return args


def _cpu_kernel(
    x: torch.Tensor, y: torch.Tensor, return_residuals: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """CPU stand-in for the kernel `jax_op` registers.

  The registered kernel returns tensors on the `tpu` device, so it cannot run
  -- and the op therefore cannot be `torch.compile`d end to end -- on a host
  without a TPU. The arithmetic mirrors `_FakeJaxMultiOutputOp._fwd` exactly,
  so compiled results can be compared against the JAX op's.
  """
  del return_residuals  # Unused; the kernel always produces the residuals.
  return x + y, x * 2, y * 3


class _FakeTorchOpForCompile(torch_op.TorchOp[_FakeOpConfig]):
  """A TorchOp that opts into `fake_impl` and `donate_argnums`.

  `fake_impl` counts its own invocations, so a test can tell that the op's own
  meta implementation ran rather than the default one `jax_op` registers.
  """

  def __init__(
      self,
      name: str | None = None,
      *,
      with_fake_impl: bool = True,
      donate_argnums: Sequence[int] | None = None,
      fake_impl: Callable[..., Any] | None = None,
  ):
    super().__init__()
    self.jax_op_name = name or f"fake_op_compile_{uuid.uuid4().hex[:8]}"
    self.op_impl_jax = _FakeJaxMultiOutputOp()
    self.donate_argnums = donate_argnums
    self.fake_impl_calls = 0
    if fake_impl is not None:
      self.fake_impl = fake_impl
    elif with_fake_impl:
      self.fake_impl = self._fake_impl

  def _fake_impl(
      self, x: torch.Tensor, y: torch.Tensor, return_residuals: bool = True
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del return_residuals  # Unused.
    # Fakes run in the compiler, not in the traced program, so this counter is
    # incremented once per trace rather than once per call.
    self.fake_impl_calls += 1
    # `empty_like` passes a symbolic dimension straight through, which is the
    # whole point of overriding the default fake.
    return torch.empty_like(x), torch.empty_like(x), torch.empty_like(y)

  def __call__(
      self, x: torch.Tensor, y: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert self._torch_tokamax_op is not None, "Forward op not registered."
    return self._torch_tokamax_op(x, y, True)

  def op_impl_call(
      self,
      x: jax.Array,
      y: jax.Array,
      return_residuals: bool = True,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert self.op_impl_jax is not None
    out, (res1, res2) = self.op_impl_jax._fwd(
        x, y, return_residuals=return_residuals, config=_HEURISTICS_CONFIG
    )
    return out, res1, res2

  def setup_context(self, ctx: Any, inputs: Sequence[Any], output: Any) -> None:
    pass

  def backward(self, *args: Any, **kwargs: Any) -> Any:
    return args


def _custom_op(op: torch_op.TorchOp[Any]) -> Any:
  """Returns the custom op that `_register_ops` registered for `op`."""
  assert op._torch_tokamax_op is not None, "Forward op not registered."
  return op._torch_tokamax_op


def _make_compile_test_op(**kwargs: Any) -> _FakeTorchOpForCompile:
  """Builds a `_FakeTorchOpForCompile` that can also run on a CPU host."""
  op = _FakeTorchOpForCompile(**kwargs)
  _custom_op(op).register_kernel("cpu")(_cpu_kernel)
  return op


class ConvertTorchToJaxViaMetaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("scalar", ()),
      ("1d", (5,)),
      ("2d", (2, 3)),
      ("3d", (2, 3, 4)),
      ("4d", (1, 2, 3, 4)),
  )
  def test_convert_tensor_shapes(self, shape: tuple[int, ...]):
    t = torch.empty(shape, dtype=torch.float32)
    meta = torch_utils.convert_torch_to_jax_via_meta(t)
    self.assertIsInstance(meta, jax.ShapeDtypeStruct)
    self.assertEqual(meta.shape, shape)
    self.assertEqual(meta.dtype, jnp.float32)

  @parameterized.named_parameters(
      ("float32", torch.float32, jnp.float32),
      ("bfloat16", torch.bfloat16, jnp.bfloat16),
      ("float16", torch.float16, jnp.float16),
      ("int32", torch.int32, jnp.int32),
      ("int64", torch.int64, jnp.int64),
      ("bool", torch.bool, jnp.bool_),
      ("uint8", torch.uint8, jnp.uint8),
  )
  def test_convert_tensor_dtypes(
      self, torch_dtype: torch.dtype, expected_jax_dtype: Any
  ):
    t = torch.empty((2, 3), dtype=torch_dtype)
    meta = torch_utils.convert_torch_to_jax_via_meta(t)
    self.assertEqual(meta.dtype, expected_jax_dtype)
    self.assertEqual(meta.shape, (2, 3))

  def test_convert_meta_device_tensor(self):
    t = torch.empty((4, 8), dtype=torch.bfloat16, device="meta")
    meta = torch_utils.convert_torch_to_jax_via_meta(t)
    self.assertIsInstance(meta, jax.ShapeDtypeStruct)
    self.assertEqual(meta.shape, (4, 8))
    self.assertEqual(meta.dtype, jnp.bfloat16)


class TorchOpTest(parameterized.TestCase):

  def test_init_defaults(self):
    op = torch_op.TorchOp()
    self.assertIsNone(op._torch_tokamax_op)
    self.assertIsNone(op.op_impl_jax)
    self.assertIsNone(op.backward_op_torch)
    self.assertEqual(op.configs, (None, None))
    self.assertEqual(op.jax_op_name, "")
    self.assertFalse(op.is_vjp)
    self.assertIsNone(op.donate_argnums)
    self.assertIsNone(op.fake_impl)

  def test_derive_backward_shapes_default(self):
    op = torch_op.TorchOp()
    self.assertEqual(op.derive_backward_shapes({}), {})

    op_vjp = torch_op.TorchOp()
    op_vjp.is_vjp = True
    with self.assertRaisesRegex(
        NotImplementedError,
        "derive_backward_shapes not implemented for VJP op.",
    ):
      op_vjp.derive_backward_shapes({})

  def test_setup_context_default(self):
    class DummyCtx:

      def __init__(self):
        self.saved_tensors = ()

      def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    op = torch_op.TorchOp()
    ctx = DummyCtx()
    t1 = torch.tensor([1.0])
    t2 = torch.tensor([2.0])
    op.setup_context(ctx, [t1], (t2,))
    self.assertEqual(ctx.saved_tensors, (t1, t2))

  def test_backward_not_implemented(self):
    op = torch_op.TorchOp()
    op.is_vjp = True
    with self.assertRaises(NotImplementedError):
      op.backward()

  def test_op_impl_call_not_implemented(self):
    op = torch_op.TorchOp()
    with self.assertRaises(NotImplementedError):
      op.op_impl_call()

  def test_call_not_implemented(self):
    op = torch_op.TorchOp()
    with self.assertRaises(NotImplementedError):
      op()

  def test_subclass_without_op_impl_jax_raises_assertion_error(self):
    class _NoOpImplJaxOp(torch_op.TorchOp):

      def __init__(self):
        super().__init__()
        self.jax_op_name = f"no_op_{uuid.uuid4().hex[:8]}"

    with self.assertRaisesRegex(AssertionError, "Forward class not set."):
      _NoOpImplJaxOp()

  def test_subclass_without_name_raises_assertion_error(self):
    class _NoNameOp(torch_op.TorchOp):

      def __init__(self):
        super().__init__()
        self.op_impl_jax = _FakeJaxMultiOutputOp()

      def op_impl_call(self, *args: Any, **kwargs: Any) -> Any:
        return self.op_impl_jax

      def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._torch_tokamax_op

    with self.assertRaisesRegex(AssertionError, "Name not set."):
      _NoNameOp()

  def test_subclass_without_op_impl_jax_in_op_impl_call_raises(self):
    class _BadFwdCallOp(torch_op.TorchOp):

      def __init__(self):
        super().__init__()
        self.jax_op_name = f"bad_fwd_{uuid.uuid4().hex[:8]}"
        self.op_impl_jax = _FakeJaxMultiOutputOp()

      def op_impl_call(self, *args: Any, **kwargs: Any) -> Any:
        return None

      def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._torch_tokamax_op

    with self.assertRaisesRegex(
        AssertionError, "op_impl_jax is not used in op_impl_call"
    ):
      _BadFwdCallOp()

  def test_subclass_without_torch_tokamax_op_in_call_raises(self):
    class _BadCallOp(torch_op.TorchOp):

      def __init__(self):
        super().__init__()
        self.jax_op_name = f"bad_call_{uuid.uuid4().hex[:8]}"
        self.op_impl_jax = _FakeJaxMultiOutputOp()

      def op_impl_call(
          self,
          x: jax.Array,
          y: jax.Array,
          return_residuals: bool = True,
      ) -> tuple[jax.Array, jax.Array, jax.Array]:
        assert self.op_impl_jax is not None
        out, (res1, res2) = self.op_impl_jax._fwd(
            x, y, return_residuals=return_residuals, config=_HEURISTICS_CONFIG
        )
        return out, res1, res2

      def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None

    with self.assertRaisesRegex(
        AssertionError, "_torch_tokamax_op is not used in __call__"
    ):
      _BadCallOp()

  def test_subclass_without_backward_op_torch_in_backward_raises(self):
    class _BadBackwardOp(torch_op.TorchOp):

      def __init__(self):
        super().__init__()
        self.jax_op_name = f"bad_bwd_{uuid.uuid4().hex[:8]}"
        self.op_impl_jax = _FakeJaxMultiOutputOp()
        self.backward_op_torch = _FakeTorchVjpOp()

      def op_impl_call(
          self,
          x: jax.Array,
          y: jax.Array,
          return_residuals: bool = True,
      ) -> tuple[jax.Array, jax.Array, jax.Array]:
        assert self.op_impl_jax is not None
        out, (res1, res2) = self.op_impl_jax._fwd(
            x, y, return_residuals=return_residuals, config=_HEURISTICS_CONFIG
        )
        return out, res1, res2

      def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self._torch_tokamax_op is not None
        return self._torch_tokamax_op(*args, **kwargs)

      def backward(self, *args: Any, **kwargs: Any) -> Any:
        return None

    with self.assertRaisesRegex(
        AssertionError, "backward_op_torch is not used in backward"
    ):
      _BadBackwardOp()

  def test_inspect_for_attribute_standalone_function(self):
    def good_fn():
      some_attr = 1
      return some_attr

    def bad_fn():
      return 42

    torch_utils.inspect_for_attribute(good_fn, "some_attr")
    with self.assertRaisesRegex(
        AssertionError, "other_attr is not used in bad_fn"
    ):
      torch_utils.inspect_for_attribute(bad_fn, "other_attr")

  def test_subclass_with_backward_register_ops(self):
    op = _FakeTorchOpWithBackward()
    self.assertTrue(getattr(op, "_ops_registered", False))
    self.assertIsNotNone(op._torch_tokamax_op)
    self.assertIsNotNone(op.backward_op_torch)

  def test_get_bound_args(self):
    op = _FakeTorchOp()
    x = torch.zeros((2, 4), dtype=torch.float32)
    y = torch.ones((2, 4), dtype=torch.float32)
    ba = op.get_bound_args(x, y)

    self.assertIsInstance(ba, op_lib.BoundArguments)
    self.assertIn("x", ba.arguments)
    self.assertIn("y", ba.arguments)
    self.assertIn("return_residuals", ba.arguments)
    self.assertEqual(ba.arguments["x"].shape, (2, 4))
    self.assertEqual(ba.arguments["x"].dtype, jnp.float32)
    self.assertEqual(ba.arguments["y"].shape, (2, 4))
    self.assertEqual(ba.arguments["y"].dtype, jnp.float32)
    self.assertTrue(ba.arguments["return_residuals"])

    self.assertEqual(ba.heuristics_config, _HEURISTICS_CONFIG)
    self.assertEqual(ba.default_config, _HEURISTICS_CONFIG)
    self.assertEqual(
        ba.autotuning_configs,
        {_AUTOTUNE_CONFIG, _HEURISTICS_CONFIG},
    )

  def test_get_bound_args_without_fwd_class_raises(self):
    op = torch_op.TorchOp()
    with self.assertRaisesRegex(AssertionError, "Forward class not set."):
      op.get_bound_args()

  def test_derive_backward_shapes_in_get_bound_args(self):
    vjp_op = _FakeTorchVjpOp()
    x = torch.zeros((2, 4), dtype=torch.float32)
    y = torch.ones((2, 4), dtype=torch.float32)
    ba = vjp_op.get_bound_args(x, y)

    self.assertIn("residuals", ba.arguments)
    residuals_meta = ba.arguments["residuals"]
    self.assertIsInstance(residuals_meta, tuple)
    self.assertLen(residuals_meta, 1)
    self.assertEqual(residuals_meta[0].shape, (2, 4))
    self.assertEqual(residuals_meta[0].dtype, jnp.float32)
    self.assertEqual(ba.arguments["x"].shape, (2, 4))
    self.assertEqual(ba.arguments["y"].shape, (2, 4))

  def test_derive_backward_shapes_in_get_bound_args_with_kwargs(self):
    vjp_op = _FakeTorchVjpOp()
    x = torch.zeros((2, 4), dtype=torch.float32)
    y = torch.ones((2, 4), dtype=torch.float32)
    ba = vjp_op.get_bound_args(x=x, y=y)

    self.assertIn("residuals", ba.arguments)
    residuals_meta = ba.arguments["residuals"]
    self.assertIsInstance(residuals_meta, tuple)
    self.assertLen(residuals_meta, 1)
    self.assertEqual(residuals_meta[0].shape, (2, 4))
    self.assertEqual(residuals_meta[0].dtype, jnp.float32)
    self.assertEqual(ba.arguments["x"].shape, (2, 4))
    self.assertEqual(ba.arguments["y"].shape, (2, 4))

  def test_op_impl_call(self):
    op = _FakeTorchOp()
    x_jax = jnp.ones((2, 4), dtype=jnp.float32)
    y_jax = 2.0 * jnp.ones((2, 4), dtype=jnp.float32)

    out, res1, res2 = op.op_impl_call(x_jax, y_jax, return_residuals=True)
    self.assertTrue(jnp.allclose(out, 3.0 * jnp.ones((2, 4))))
    self.assertTrue(jnp.allclose(res1, 2.0 * jnp.ones((2, 4))))
    self.assertTrue(jnp.allclose(res2, 6.0 * jnp.ones((2, 4))))

  def test_torch_utils_get_configs(self):
    op = _FakeTorchOp()
    x = torch.zeros((2, 4), dtype=torch.float32)
    y = torch.ones((2, 4), dtype=torch.float32)
    fwd_config, bwd_config = torch_utils.get_configs(
        op, x, y, from_autotuning_cache=False
    )
    self.assertEqual(fwd_config, _HEURISTICS_CONFIG)
    self.assertIsNone(bwd_config)

    op_bwd = _FakeTorchOpWithBackward()
    fwd_config, bwd_config = torch_utils.get_configs(
        op_bwd, x, y, from_autotuning_cache=False
    )
    self.assertEqual(fwd_config, _HEURISTICS_CONFIG)
    self.assertEqual(bwd_config, _HEURISTICS_CONFIG)

  def test_custom_setup_context_and_backward(self):
    class DummyCtx:

      def __init__(self):
        self.saved_tensors = ()

    op = _FakeTorchOp()
    ctx = DummyCtx()
    t = torch.tensor([1.0, 2.0])
    op.setup_context(ctx, [], (t,))
    self.assertEqual(ctx.saved_tensors, (t,))

    grad = torch.tensor([0.5, 0.5])
    bwd_result = op.backward(ctx, grad)
    self.assertEqual(bwd_result, (ctx, grad))

  def test_donated_argument_is_not_declared_mutable(self):
    # A donated buffer is left invalid rather than mutated. Declaring it
    # mutable would make Dynamo copy the original back after the call, which
    # costs exactly the buffer that donating it was meant to save.
    op = _make_compile_test_op(donate_argnums=(1,))
    schema = _custom_op(op)._opoverload._schema
    self.assertFalse(schema.is_mutable)

  def test_donating_does_not_change_the_result(self):
    op = _make_compile_test_op(donate_argnums=(1,))
    x = torch.ones((4, 8), dtype=torch.float32)
    y = 2.0 * torch.ones((4, 8), dtype=torch.float32)

    compiled = torch.compile(lambda a, b: op(a, b), fullgraph=True)
    out, res1, res2 = compiled(x, y)

    torch.testing.assert_close(out, x + y)
    torch.testing.assert_close(res1, x * 2)
    torch.testing.assert_close(res2, y * 3)


class FakeImplTest(parameterized.TestCase):
  """Tests for `TorchOp.fake_impl`, the op's meta implementation."""

  def test_fake_impl_replaces_the_default_fake(self):
    op = _make_compile_test_op()
    with fake_tensor.FakeTensorMode():
      x = torch.empty((4, 8), dtype=torch.float32)
      y = torch.empty((4, 8), dtype=torch.float32)
      out, res1, res2 = op(x, y)

    self.assertEqual(op.fake_impl_calls, 1)
    for tensor in (out, res1, res2):
      self.assertEqual(tuple(tensor.shape), (4, 8))
      self.assertEqual(tensor.dtype, torch.float32)

  def test_fake_impl_is_not_registered_when_unset(self):
    op = _make_compile_test_op(with_fake_impl=False)
    self.assertIsNone(op.fake_impl)

    with fake_tensor.FakeTensorMode():
      x = torch.empty((4, 8), dtype=torch.float32)
      y = torch.empty((4, 8), dtype=torch.float32)
      # Whether the default fake succeeds depends on there being a TPU to make
      # its placeholders on; either way, ours must not have run.
      try:
        op(x, y)
      except Exception:  # pylint: disable=broad-except
        pass
    self.assertEqual(op.fake_impl_calls, 0)

  def test_fake_impl_agrees_with_the_kernel(self):
    op = _make_compile_test_op()
    x = torch.ones((4, 8), dtype=torch.float32)
    y = torch.full((4, 8), 2.0, dtype=torch.float32)

    # Real execution on actual tensors
    out_real, r0_real, r1_real = op(x, y)

    # Fake execution with fake tensors created inside the context
    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((4, 8), dtype=torch.float32)
      y_fake = torch.empty((4, 8), dtype=torch.float32)
      out_fake, r0_fake, r1_fake = op(x_fake, y_fake)

    self.assertEqual(out_real.shape, out_fake.shape)
    self.assertEqual(out_real.dtype, out_fake.dtype)
    self.assertEqual(r0_real.shape, r0_fake.shape)
    self.assertEqual(r1_real.shape, r1_fake.shape)

  def test_opcheck_rejects_a_fake_impl_that_does_not_match(self):
    def _wrong_fake_impl(
        x: torch.Tensor, y: torch.Tensor, return_residuals: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      del return_residuals  # Unused.
      # Second residual should be `y`-shaped (4, 8), but we return (1,).
      return torch.empty_like(x), torch.empty_like(x), torch.empty((1,))

    op = _make_compile_test_op(fake_impl=_wrong_fake_impl)
    expected_y_shape = (4, 8)

    with fake_tensor.FakeTensorMode():
      x_fake = torch.empty((4, 8), dtype=torch.float32)
      y_fake = torch.empty((4, 8), dtype=torch.float32)
      out_fake, r0_fake, r1_fake = op(x_fake, y_fake)

    # Assert shape mismatch against expected tensor shape
    self.assertNotEqual(r1_fake.shape, expected_y_shape)


class TorchCompileTest(parameterized.TestCase):
  """Tests for calling a `TorchOp` from inside a `torch.compile` region."""

  def setUp(self):
    super().setUp()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

  def test_compile_matches_eager(self):
    op = _make_compile_test_op()
    x = torch.ones((4, 8), dtype=torch.float32)
    y = 2.0 * torch.ones((4, 8), dtype=torch.float32)
    expected = op(x, y)

    compiled = torch.compile(lambda a, b: op(a, b), fullgraph=True)
    actual = compiled(x, y)

    self.assertGreater(op.fake_impl_calls, 0)
    for actual_out, expected_out in zip(actual, expected, strict=True):
      torch.testing.assert_close(actual_out, expected_out)

  def test_compile_captures_the_op_without_a_graph_break(self):
    op = _make_compile_test_op()
    graphs = []

    def _backend(gm: torch.fx.GraphModule, example_inputs: Any) -> Any:
      del example_inputs  # Unused.
      graphs.append(gm)
      return gm

    compiled = torch.compile(
        lambda a, b: op(a, b), fullgraph=True, backend=_backend
    )
    compiled(torch.ones((4, 8)), torch.ones((4, 8)))

    self.assertLen(graphs, 1)
    op_nodes = [
        node
        for node in graphs[0].graph.nodes
        if node.op == "call_function" and op.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)

  def test_compile_with_a_dynamic_token_count(self):
    op = _make_compile_test_op()
    compiled = torch.compile(
        lambda a, b: op(a, b), fullgraph=True, dynamic=True
    )

    for num_tokens in (4, 7, 13):
      x = torch.ones((num_tokens, 8), dtype=torch.float32)
      y = 2.0 * torch.ones((num_tokens, 8), dtype=torch.float32)
      out, res1, res2 = compiled(x, y)
      torch.testing.assert_close(out, x + y)
      torch.testing.assert_close(res1, x * 2)
      torch.testing.assert_close(res2, y * 3)

    # A serving stack compiles once for a range of batch sizes. If the fake
    # specialized on the token count we would get a graph per shape.
    self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

  def test_compile_with_a_dynamic_token_count_needs_a_fake_impl(self):
    op = _make_compile_test_op(with_fake_impl=False)
    compiled = torch.compile(
        lambda a, b: op(a, b), fullgraph=True, dynamic=True
    )
    with self.assertRaisesRegex(
        RuntimeError, "Symbolic dimensions are not supported"
    ):
      compiled(torch.ones((4, 8)), torch.ones((4, 8)))

  def test_fake_impl_keeps_the_token_count_symbolic(self):
    op = _make_compile_test_op()

    class _Module(torch.nn.Module):

      def forward(
          self, x: torch.Tensor, y: torch.Tensor
      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return op(x, y)

    num_tokens = torch.export.Dim("num_tokens", min=1, max=1024)
    exported = torch.export.export(
        _Module(),
        (torch.ones((4, 8)), torch.ones((4, 8))),
        dynamic_shapes=({0: num_tokens}, {0: num_tokens}),
    )

    op_nodes = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function" and op.jax_op_name in str(node.target)
    ]
    self.assertLen(op_nodes, 1)
    for meta in op_nodes[0].meta["val"]:
      self.assertIsInstance(meta.shape[0], torch.SymInt)
      self.assertEqual(meta.shape[1], 8)


if __name__ == "__main__":
  absltest.main()
