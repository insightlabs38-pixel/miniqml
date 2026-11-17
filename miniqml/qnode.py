"""PennyLane-style QNode construction."""

from functools import update_wrapper
from typing import Any, Callable

from .device import Device
from .ir import QuantumTape
from .tape import recording


class QNode:
    def __init__(
        self,
        func: Callable[..., Any],
        device: Device,
        *,
        interface: str = "torch",
        diff_method: str = "parameter-shift",
    ) -> None:
        if not callable(func):
            raise TypeError("A QNode requires a callable quantum function.")
        if not isinstance(device, Device):
            raise TypeError("A QNode requires a miniqml Device.")
        if interface != "torch":
            raise ValueError("Only interface='torch' is planned for this subset.")
        if diff_method != "parameter-shift":
            raise ValueError("Only diff_method='parameter-shift' is supported.")
        self.func = func
        self.device = device
        self.interface = interface
        self.diff_method = diff_method
        update_wrapper(self, func)

    def construct(self, *args: Any, **kwargs: Any) -> QuantumTape:
        with recording(self.device.wires) as recorder:
            returned = self.func(*args, **kwargs)
            return recorder.finish(returned)

    def batch(self, features: Any, *args: Any, **kwargs: Any) -> Any:
        """Run one circuit per feature row with shared remaining arguments."""
        if getattr(features, "ndim", 2) != 2:
            raise ValueError("Batch features must have shape (batch, features).")
        if not hasattr(features, "__len__") or not len(features):
            raise ValueError("Batch features must contain at least one sample.")
        tapes = [self.construct(sample, *args, **kwargs) for sample in features]
        from .interfaces.torch import execute_torch_batch

        return execute_torch_batch(self.device, tapes)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tape = self.construct(*args, **kwargs)
        from .interfaces.torch import execute_torch

        return execute_torch(self.device, tape)


def qnode(
    dev: Device, *, interface: str = "torch", diff_method: str = "parameter-shift"
) -> Callable[[Callable[..., Any]], QNode]:
    def decorate(func: Callable[..., Any]) -> QNode:
        return QNode(func, dev, interface=interface, diff_method=diff_method)

    return decorate
