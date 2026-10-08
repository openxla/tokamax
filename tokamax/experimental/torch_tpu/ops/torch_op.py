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

"""Top level wrapper to create PyTorch interfaces to Tokamax JAX Ops."""

from abc import abstractmethod
from collections.abc import Sequence
import dataclasses
from functools import wraps
import inspect
import logging
import types
from typing import Any, Callable, Generic, Optional, TypeVar, overload
import jax
from tokamax._src.ops import op as jax_tokamax_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch

_Config = TypeVar("_Config")

"""
  How to use this class to set up a Jax Tokamax Op with a PyTorch Interface:
  
  Given a Jax Tokamax Op:
  
  - Create a class inheriting from TorchOp for the forward pass.
  - Create a class inheriting from TorchOp for the backward pass if it exists.
  
  In both classes, fill out these attributes in the constructor:
  - jax_op_name: the unique name of the Torch Op passed to
    torch_tpu.jax_op.
  - op_impl_jax: the JAX Tokamax op class to wrap using torch_tpu.jax_op.
  - is_vjp: whether the Torch Op is a VJP op. This is false for the forward pass
    and true for the backward pass.

  Optionally, also in the constructor:
  - donate_argnums: positional indices of op_impl_call's inputs whose buffers
    the kernel may reuse for its outputs. These are passed through to jax.jit
    which uses them to reduce the number of buffers that need to be allocated.
  - fake_impl: a meta implementation, needed when the op is called from inside
    a torch.compile region with a dynamic dimension. If not supplied, the
    default implementation will re-export the JAX function, which rejects
    symbolic shapes.

  In both classes, implement the following methods:
  - op_impl_call: The kernel invocation method for the JAX Tokamax op. This must
    invoke op_impl_jax with the full list of inputs and config to run the
    kernel. TorchOp's __init_subclass__ automatically decorates op_impl_call
    with ensure_op_impl_call_config_setup so that op_impl_call_config_setup is
    always invoked before op_impl_call runs, passing the resolved config tuple
    into op_impl_call. This is done to ensure that there is a valid config for
    the kernel before it is invoked. TorchOp's register_ops will wrap this
    method in a custom PyTorch op called _torch_tokamax_op.
  - __call__: The main function that will be called by the user. This must
    call _torch_tokamax_op with the correct arguments. This function will take
    in the configs for the forward and backward pass additionally, deconstructing
    them via deconstruct_config before passing them to _torch_tokamax_op.

  Optionally, override the following config serialization methods:
  - deconstruct_config and reconstruct_config: PyTorch custom ops
    (torch_tpu.jax_op) only support primitive types and homogeneously typed
    sequences (e.g. tuple[int, ...], str, int, float, bool) in their schema and
    cannot accept arbitrary Config objects directly. deconstruct_config converts
    a Config object in __call__ into jax_op-compatible primitive/tuple types
    before calling _torch_tokamax_op, and reconstruct_config rebuilds the Config
    object inside op_impl_call before invoking op_impl_jax. By default,
    deconstruct_config converts the Config dataclass into a single tuple via
    dataclasses.astuple, and reconstruct_config unpacks that tuple into
    op_impl_jax.config_cls(*config). Override these methods when the Config
    contains mixed field types (e.g. both int fields and a str field) or nested
    structures that must be split into multiple separate jax_op-compatible
    arguments and reassembled in op_impl_call.
    
  If there is a backward pass:
  - In the forward Op constructor, set backward_op_torch to an instance of the
    created backward TorchOp class.
  - There are additional methods to implement for the forward TorchOp class.
    These methods are required to properly hook up the backward TorchOp class to
    the forward TorchOp class:
    - setup_context: Sets up the context for the backward pass. This should save
      the inputs and outputs of the forward pass that are needed for the
      backward pass into ctx based on PyTorch's register_autograd API.
    - backward: Calls the Torch Tokamax op backward class to compute the
      gradients. This must invoke backward_op_torch with the correct arguments
      and config and use the context saved in setup_context.
    
  Additionally, for the backward TorchOp class, implement:
  - derive_backward_shapes: Derives the backward shapes for the backward pass
    residuals and outputs. Only needed if the Torch Op is a backward op; this
    will be used to look up the autotuning configs for the backward pass op.
    The shapes derived here should be a dictionary of the backward pass
    parameter names that are not in the forward pass. The derivation for these
    shapes should be based on the shapes given in the forward pass inputs.
    
  
  Finally, in the module, make the forward TorchOp class a singleton instance.
"""


class _OpImplCallDescriptor:
  """Descriptor ensuring op_impl_call_config_setup runs before op_impl_call."""

  def __init__(self, fn: Callable[..., Any]) -> None:
    # Unwrap `fn` in case the decorator is applied more than once (e.g. both
    # explicitly with `@ensure_op_impl_call_config_setup` and automatically via
    # `TorchOp.__init_subclass__`), and copy its function metadata onto `self`.
    self._fn = inspect.unwrap(fn)
    wraps(self._fn)(self)

  def __call__(self, instance: Any, *args: Any, **kwargs: Any) -> Any:
    # If `op_impl_call` accepts a `config` parameter, bind the call arguments
    # to extract `config` (whether passed positionally, by keyword, or defaulted
    # to `None`), resolve it via `op_impl_call_config_setup` (which computes a
    # default config tuple when `config` is `None`), and pass the resolved
    # `config` into `op_impl_call`. Otherwise, still run
    # `op_impl_call_config_setup` before invoking `op_impl_call` without a
    # `config` argument.
    sig = inspect.signature(self._fn)
    if "config" in sig.parameters:
      bound = sig.bind_partial(instance, *args, **kwargs)
      bound.apply_defaults()
      config = bound.arguments.get("config")
      kwargs_without_config = {k: v for k, v in kwargs.items() if k != "config"}
      config = instance.op_impl_call_config_setup(
          *args, config=config, **kwargs_without_config
      )
      bound.arguments["config"] = config
      return self._fn(*bound.args, **bound.kwargs)

    config = kwargs.pop("config", None)
    instance.op_impl_call_config_setup(*args, config=config, **kwargs)
    return self._fn(instance, *args, **kwargs)

  def __get__(self, instance: Any, owner: Any = None) -> Callable[..., Any]:
    del owner
    if instance is None:
      return self
    # Bind `self._fn` to `instance` so `inspect.unwrap(op.op_impl_call)` returns
    # a bound method with `self` already bound, matching a standard method for
    # `torch_tpu.jax_op` and `torch_utils.inspect_for_attribute`.
    bound_fn = types.MethodType(self._fn, instance)

    @wraps(bound_fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
      return self(instance, *args, **kwargs)

    return wrapped


def ensure_op_impl_call_config_setup(
    fn: Callable[..., Any],
) -> Callable[..., Any]:
  """Decorator that ensures op_impl_call_config_setup runs before op_impl_call."""
  if isinstance(fn, _OpImplCallDescriptor):
    return fn
  return _OpImplCallDescriptor(fn)


class TorchOp(Generic[_Config]):
  """Top level wrapper to create PyTorch interfaces to Tokamax JAX Ops.

  Setting up the TorchOp for the inheriting class:

  For inheriting classes, the constructor needs to follow this sequence:
    - Call the parent class (TorchOp) constructor.
    - In the constructor, fill out jax_op_name, op_impl_jax,
      backward_op_torch (if applicable), and is_vjp. See the top level docstring
      for more details on each of these attributes.
    - register_ops() will automatically be called at the end of the
      constructor. This will register the JAX Tokamax op with torch_tpu and
      also check that the required attributes are set correctly.

  There are five abstract methods that must be implemented by the inheriting
  class:

  For any TorchOp:
    - op_impl_call
    - __call__
  For TorchOps with a backward pass:
  - setup_context
  - backward
  For TorchOps that are VJP ops:
  - derive_backward_shapes

  Please see the top level docstring for more details on each of these methods.

  register_ops will check to ensure that op_impl_jax is used in
  op_impl_call and backward_op_torch is used in backward (if applicable).
  __init_subclass__ also wraps op_impl_call with
  ensure_op_impl_call_config_setup so that op_impl_call_config_setup is always
  invoked prior to executing op_impl_call.

  Note:
  All Torch Tokamax ops are singleton instances. You may not have
  multiple instances of the same Torch Op class.
  """

  def __init__(
      self,
  ) -> None:
    """Initializes the TorchOp.

    This should be called by the inheriting class at the beginning of the
    constructor.
    """

    # The op must be set by the inheriting class in the constructor.
    # This should be the JAX Tokamax op class that this Torch Op wraps.
    self.op_impl_jax = None
    # Generated by register_ops. Do not set this attribute in the constructor.
    # This is op_impl_jax wrapped in a custom op that is registered with
    # torch_tpu.jax_op.
    self._torch_tokamax_op: torch._library.custom_ops.CustomOpDef | None = None

    # backward_op_torch must be an instance of a TorchOp subclass. The
    # inheriting class must set this attribute in the constructor. If there is
    # no backward pass, this should be None.
    self.backward_op_torch: Any | None = None

    # jax_op_name should be unique to the Torch Op. This is the name that will
    # be used in the custom op name in torch_tpu.jax_op.
    self.jax_op_name = ""

    # Whether the Torch Op is a VJP op.
    self.is_vjp = False

    # Positional indices of op_impl_call's arguments whose buffers the kernel
    # is allowed to reuse for its outputs. See the Jax Buffer Donation article
    # for more information. The inheriting class sets this in its constructor;
    # None donates nothing.
    self.donate_argnums: Sequence[int] | None = None

    # Meta/fake implementation for the registered custom op. jax_op registers
    # a default that re-exports the JAX function, which rejects symbolic
    # shapes, so an op that is called from inside a torch.compile region with
    # a dynamic dimension has to supply its own. The inheriting class sets
    # this in its constructor; None keeps jax_op's default.
    self.fake_impl: Callable[..., Any] | None = None

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Initializes the TorchOp subclass and automatically registers ops."""
    super().__init_subclass__(**kwargs)

    if "op_impl_call" in cls.__dict__:
      cls.op_impl_call = ensure_op_impl_call_config_setup(  # type: ignore[assignment]
          cls.__dict__["op_impl_call"]
      )

    # Capture the child class's __init__
    original_init: Callable[..., None] = cls.__init__

    @wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
      # 1. Run the child's initialization logic
      original_init(self, *args, **kwargs)

      # 2. Automatically invoke register_ops once when the most-derived
      # subclass __init__ finishes.
      if self.__class__ is cls and not getattr(self, "_ops_registered", False):
        setattr(self, "_ops_registered", True)
        self._register_ops()

    cls.__init__ = wrapped_init

  def get_bound_args(
      self,
      *args: Any,
      **kwargs: Any,
  ) -> jax_tokamax_op.BoundArguments:
    """Returns the BoundArguments for the JAX Tokamax op.

    This handles both standard forward ops and VJP ops (when `self.is_vjp` is
    True).
    
    This function figures out the correct positional arguments for the JAX op by
    comparing the positional arguments of `op_impl_call` and the forward JAX
    op's signature then binds all arguments by name to create a
    `BoundArguments` object. It does this in 4 steps:
    
    1. Inspect `self.op_impl_jax._fwd`'s parameters (excluding `config`) and
       identify any backward-only parameter names via
       `self.derive_backward_shapes({})`. This builds the forward and backward
       parameter names.
       
    2. Map positional `args` into `bound_kwargs` by parameter name.
    
    3. Populate `abstract_argument_dict` in `_fwd` parameter order from
       `bound_kwargs`, `self` attributes, and `_fwd` parameter defaults.

    4. If this is a VJP op, populate backward-only shapes via
       `self.derive_backward_shapes(abstract_argument_dict)` and return
       `BoundArguments`.

    Args:
      *args: Positional arguments to the JAX Tokamax op.
      **kwargs: Keyword arguments to the JAX Tokamax op.

    Returns:
      A BoundArguments object for the JAX Tokamax op.
    """
    assert self.op_impl_jax is not None, "Forward class not set."

    def _to_abstract(val: Any) -> Any:
      if isinstance(val, torch.Tensor):
        return torch_utils.convert_torch_to_jax_via_meta(val)
      if isinstance(val, jax.Array):
        return jax.ShapeDtypeStruct(shape=val.shape, dtype=val.dtype)
      return val

    # 1. Get `_fwd`'s parameters excluding `config`.
    sig = inspect.signature(
        self.op_impl_jax._fwd  # pylint: disable=protected-access
    )
    sig_params = {k: p for k, p in sig.parameters.items() if k != "config"}
    backward_param_names = set(self.derive_backward_shapes({}).keys())

    # 2. Map positional `args` to parameter names:
    #    - When a VJP op is called with only forward inputs (e.g. from
    #      `torch_utils.get_configs`), map `args` to `_fwd`'s forward positional
    #      parameters (skipping backward-only parameters like `residuals`).
    #    - Otherwise (when called from `op_impl_call_config_setup`), map `args`
    #      using `op_impl_call`'s positional parameter names so arguments that
    #      are positional in `op_impl_call` but keyword-only in `_fwd` (such as
    #      `cp_rank` or `reduction`) are bound by name.
    fwd_pos_names = [
        name
        for name, p in sig_params.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    forward_only_pos_names = [
        name for name in fwd_pos_names if name not in backward_param_names
    ]
    op_call_pos_names = [
        p.name
        for p in inspect.signature(
            inspect.unwrap(self.op_impl_call)
        ).parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    if self.is_vjp and len(args) <= len(forward_only_pos_names):
      pos_names = forward_only_pos_names
    else:
      pos_names = op_call_pos_names or fwd_pos_names

    bound_kwargs = dict(zip(pos_names, args), **kwargs)

    # 3. Populate `abstract_argument_dict` for each `_fwd` parameter from:
    #    (a) explicitly passed arguments (converted to `jax.ShapeDtypeStruct`),
    #    (b) `_fwd` parameter defaults (or `return_residuals = not self.is_vjp`
    #        when omitted).
    abstract_argument_dict: dict[str, Any] = {}
    for name, param in sig_params.items():
      if name in bound_kwargs:
        abstract_argument_dict[name] = jax.tree.map(
            _to_abstract, bound_kwargs[name]
        )
      elif name in backward_param_names:
        continue
      elif param.default is not inspect.Parameter.empty:
        abstract_argument_dict[name] = param.default
      elif name == "return_residuals":
        abstract_argument_dict[name] = not self.is_vjp

    # 4. For VJP ops, derive backward-only shapes (e.g. `residuals`, `out`,
    #    `dout`) from the forward input shapes.
    if self.is_vjp:
      abstract_argument_dict.update(
          self.derive_backward_shapes(abstract_argument_dict)
      )

    return jax_tokamax_op.BoundArguments(
        self.op_impl_jax, abstract_argument_dict
    )

  def _register_ops(
      self,
  ) -> None:
    """Registers the JAX Tokamax op with torch_tpu.

    Binds torch_tokamax_fwd to a PyTorch custom_op that contains the JAX op.
    If the backward op is set, PyTorch's register_autograd API is also called to
    bind the
    backward pass to the custom op.

    Register_ops will check to ensure that the inheriting class has set the
    jax_op_name and op_impl_jax attributes, and verify that op_impl_jax is
    used in op_impl_call and backward_op_torch is used in backward
    (if applicable).

    This function will be called automatically when the child class is
    initialized at the end of the constructor.
    """
    assert self.jax_op_name, "Name not set."
    assert self.op_impl_jax is not None, "Forward class not set."
    assert self._torch_tokamax_op is None, "Forward op already registered."
    torch_utils.inspect_for_attribute(self.op_impl_call, "op_impl_jax")

    self._torch_tokamax_op = torch.tpu.pallas.jax_op(
        f"tokamax::{self.jax_op_name}",
        self.op_impl_call,
        donate_argnums=self.donate_argnums,
    )

    if self.fake_impl is not None:
      # Replaces the default fake that jax_op registers.
      self._torch_tokamax_op.register_fake(self.fake_impl)

    torch_utils.inspect_for_attribute(self.__call__, "_torch_tokamax_op")

    # If the backward op is set, register it with torch_tpu. Torch Ops with a
    # backward pass will have setup_context and backward methods.
    if self.backward_op_torch is not None:
      torch_utils.inspect_for_attribute(self.backward, "backward_op_torch")
      self._torch_tokamax_op.register_autograd(
          self.backward, setup_context=self.setup_context
      )

  @abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Calls the JAX Tokamax op wrapped by the TorchOp.

    This is the main function that will be called by the user. It will call the
    JAX Tokamax op's forward pass and return the output.

    The inheriting class must implement this method and invoke
    _torch_tokamax_op with the correct arguments. This method is responsible
    for setting the configs for the forward and backward pass and also
    re-arranging the arguments to match the JAX Tokamax op's signature.

    Args:
      *args: Positional arguments to the JAX Tokamax op.
      **kwargs: Keyword arguments to the JAX Tokamax op.

    Returns:
      The output of the JAX Tokamax op.
    """
    raise NotImplementedError(
        "__call__ not implemented. The inheriting class must implement this"
        " method."
    )

  def deconstruct_config(
      self, config: _Config | tuple[Any, ...] | list[Any] | None
  ) -> Any:
    """Deconstructs a Config object into jax_op-compatible primitive/tuple types."""
    if config is None:
      return None
    if isinstance(config, (tuple, list)):
      return tuple(config)
    return dataclasses.astuple(config)  # type: ignore[arg-type]

  def reconstruct_config(self, *config_parts: Any) -> _Config:
    """Reconstructs a Config object from its deconstructed representation."""
    assert self.op_impl_jax is not None, "Forward class not set."
    config = config_parts[0]
    assert config is not None, "Config not set."
    return self.op_impl_jax.config_cls(*config)

  def op_impl_call_config_setup(
      self, *args: Any, config: Any = None, **kwargs: Any
  ) -> Any:
    """Sets up the config for the JAX Tokamax op."""
    if config is None:
      config = dataclasses.astuple(
          self.get_bound_args(*args, **kwargs).get_config(
              check_autotuning_cache=False,
          )
      )
    return config

  @abstractmethod
  def op_impl_call(self, *args: Any, **kwargs: Any) -> Any:
    """Forward pass method for the JAX Tokamax op.

    This method is called by the PyTorch framework when the custom_op is
    invoked,
    via the custom op wrapper
    around the JAX Tokamax op. The inheriting class must implement this method.
    This function must invoke op_impl_jax with the full list of inputs and
    config to run the kernel.

    Args:
      *args: Positional arguments to the JAX Tokamax op.
      **kwargs: Keyword arguments to the JAX Tokamax op.

    Returns:
      The output of the JAX Tokamax op.
    """
    raise NotImplementedError(
        "op_impl_call not implemented. The inheriting class must implement this"
        " method."
    )

  @abstractmethod
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    """Sets up the context for the backward pass.

    TorchOp provides a default implementation that saves the input and output of
    the forward pass for the backward pass. If this default implementation is
    not sufficient, the inheriting class can override this method.

    Args:
      ctx: The context object to store information for the backward pass.
      inputs: The inputs to the forward pass.
      output: The output of the forward pass.
    """
    ctx.save_for_backward(*inputs, *output)

  @abstractmethod
  def backward(self, *args: Any, **kwargs: Any) -> Any:
    """Calls the JAX Tokamax op backward pass.

    This is the main function that will be called by the Torch Op for the
    backward pass. The inheriting class must implement this method. This must
    invoke backward_op_torch with the correct arguments and config.

    The inheriting class must also implement the setup_context function to
    initialize the context for the backward pass.
    """
    if self.is_vjp:
      raise NotImplementedError("backward not implemented.")

  @abstractmethod
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    """Derives the backward shapes for the JAX Tokamax op.

    This is needed for the VJP op so that we can create a TorchOp for the
    backward pass with known shapes for BoundArguments. Right now they are
    derived manually but there is probably a way to do this automatically.

    All inheriting VJP classes must implement this method. If
    abstract_argument_dict
    is empty, we return the keys and () for the shapes.

    If not implemented, we will return an empty dict.

    Args:
      abstract_argument_dict: The abstract argument dictionary for the JAX
        Tokamax op.

    Returns:
      A dictionary of the backward shapes for the JAX Tokamax op.
    """
    if self.is_vjp:
      raise NotImplementedError(
          "derive_backward_shapes not implemented for VJP op."
      )
    return {}
