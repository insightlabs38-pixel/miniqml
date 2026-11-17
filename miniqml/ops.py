"""Supported gates and their validation rules."""

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any

from .ir import Operation
from .tape import current_recorder
from .wires import operation_wires


@dataclass(frozen=True)
class GateSpec:
    wires: int
    parameters: int


GATE_SPECS = MappingProxyType({
    "RX": GateSpec(1, 1),
    "RY": GateSpec(1, 1),
    "RZ": GateSpec(1, 1),
    "Hadamard": GateSpec(1, 0),
    "X": GateSpec(1, 0),
    "CNOT": GateSpec(2, 0),
    "CZ": GateSpec(2, 0),
})


def _validate_angle(angle: Any) -> None:
    if isinstance(angle, bool):
        raise TypeError("Rotation angles must be real scalars, not bool.")
    if isinstance(angle, Real):
        if not math.isfinite(angle):
            raise ValueError("Rotation angles must be finite.")
        return
    if getattr(angle, "ndim", None) != 0:
        raise TypeError("Rotation angles must be real scalar values or scalar tensors.")
    dtype = getattr(angle, "dtype", None)
    kind = getattr(dtype, "kind", None)
    dtype_name = str(dtype)
    if kind != "f" and dtype_name not in {
        "torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"
    }:
        raise TypeError("Rotation angles must have a real floating-point dtype.")


def _gate(name: str, wires: object, *parameters: Any) -> Operation:
    spec = GATE_SPECS[name]
    if len(parameters) != spec.parameters:
        raise ValueError(f"{name} expects {spec.parameters} parameter(s).")
    for parameter in parameters:
        _validate_angle(parameter)
    operation = Operation(
        name=name,
        wires=operation_wires(wires, spec.wires),
        parameters=tuple(parameters),
    )
    current_recorder().add_operation(operation)
    return operation


def RX(theta: Any, wires: object) -> Operation:
    return _gate("RX", wires, theta)


def RY(theta: Any, wires: object) -> Operation:
    return _gate("RY", wires, theta)


def RZ(theta: Any, wires: object) -> Operation:
    return _gate("RZ", wires, theta)


def Hadamard(wires: object) -> Operation:
    return _gate("Hadamard", wires)


def H(wires: object) -> Operation:
    return Hadamard(wires)


def X(wires: object) -> Operation:
    return _gate("X", wires)


def CNOT(wires: object) -> Operation:
    return _gate("CNOT", wires)


def CZ(wires: object) -> Operation:
    return _gate("CZ", wires)
