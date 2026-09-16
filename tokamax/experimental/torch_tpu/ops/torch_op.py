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
from functools import wraps
import inspect
import logging
import types
from typing import Any, Callable, Generic, Optional, TypeVar, overload
import jax
from tokamax._src.ops import op as jax_tokamax_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch
import torch_tpu._internal.pallas.pallas

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
  
  In both classes, implement the following methods:
  - op_impl_call: The kernel invocation method for the JAX Tokamax op. This must
    invoke op_impl_jax with the full list of inputs and config to run the
    kernel. TorchOp's register_ops will wrap this method in a custom PyTorch op
    called _torch_tokamax_op.
  - __call__: The main function that will be called by the user. This must
    call _torch_tokamax_op with the correct arguments. This function will take
    in the configs for the forward and backward pass additionally.
    
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
    
  
  Finally, in the module, make the forward TorchOp class a singleton instance.
"""


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

    # Configs for the forward and backward pass.
    # This should be passed in from the __call__ function.
    self.configs: tuple[Any, Any] = (None, None)

    # jax_op_name should be unique to the Torch Op. This is the name that will
    # be used in the custom op name in torch_tpu.jax_op.
    self.jax_op_name = ""

    # Whether the Torch Op is a VJP op.
    self.is_vjp = False

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Initializes the TorchOp subclass and automatically registers ops."""
    super().__init_subclass__(**kwargs)

    # Capture the child class's __init__
    original_init: Callable[..., None] = cls.__init__

    @wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
      # 1. Run the child's initialization logic
      original_init(self, *args, **kwargs)

      # 2. Automatically invoke register_ops once
      if not getattr(self, "_ops_registered", False):
        self._ops_registered = True
        self._register_ops()

    cls.__init__ = wrapped_init

  def get_bound_args(
      self,
      *args: Any,
      **kwargs: Any,
  ) -> jax_tokamax_op.BoundArguments:
    """Returns the BoundArguments for the JAX Tokamax op.

    This code will handle both standard forward ops and VJP ops. It detects
    whether the op is a VJP op by checking if  is_vjp is True.

    Args:
      *args: Positional arguments to the JAX Tokamax op.
      **kwargs: Keyword arguments to the JAX Tokamax op.

    Returns:
      A BoundArguments object for the JAX Tokamax op.
    """
    assert self.op_impl_jax is not None, "Forward class not set."

    def _fwd_signature(fwd: Any) -> inspect.Signature:
      sig = inspect.signature(fwd)
      params = sig.parameters.copy()
      del params["config"]
      return sig.replace(parameters=tuple(params.values()))

    sig = _fwd_signature(self.op_impl_jax._fwd)

    # For backward ops, we need to bind using the backward shapes.
    # We use derive_backward_shapes to get the backward shapes and help identify
    # the forward pass inputs.
    backward_param_names = set(self.derive_backward_shapes({}).keys())
    abstract_argument_dict: dict[str, Any] = {}

    if self.is_vjp:
      # --- Handling VJP / Backward Op ---
      # Identify parameters that represent the original forward inputs
      forward_positional_params = [
          param
          for name, param in sig.parameters.items()
          if name not in backward_param_names
          and param.kind
          in (
              inspect.Parameter.POSITIONAL_ONLY,
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
          )
      ]

      # Map any number of positional args to forward positional parameters
      for param, arg in zip(forward_positional_params, args):
        abstract_argument_dict[param.name] = (
            torch_utils.convert_torch_to_jax_via_meta(arg)
            if isinstance(arg, torch.Tensor)
            else arg
        )

      # Map keyword arguments
      for k, v in kwargs.items():
        abstract_argument_dict[k] = (
            torch_utils.convert_torch_to_jax_via_meta(v)
            if isinstance(v, torch.Tensor)
            else v
        )

      # Apply defaults from signature for any omitted parameters
      for name, param in sig.parameters.items():
        if (
            name not in abstract_argument_dict
            and name not in backward_param_names
        ):
          if param.default is not inspect.Parameter.empty:
            abstract_argument_dict[name] = param.default
          elif name == "return_residuals":
            abstract_argument_dict[name] = False

      # Populate backward shapes dynamically using the mapped forward abstract
      # shapes
      abstract_argument_dict.update(
          self.derive_backward_shapes(abstract_argument_dict)
      )

    else:
      # --- Handling Standard Forward Op ---
      ba = sig.bind(*args, return_residuals=True, **kwargs)
      ba.apply_defaults()
      abstract_argument_dict = {
          name: (
              torch_utils.convert_torch_to_jax_via_meta(arg)
              if isinstance(arg, torch.Tensor)
              else arg
          )
          for name, arg in ba.arguments.items()
      }
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

    self._torch_tokamax_op = torch_tpu._internal.pallas.pallas.jax_op(
        f"tokamax::{self.jax_op_name}",
        self.op_impl_call,
    )

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
