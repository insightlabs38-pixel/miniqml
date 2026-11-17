"""Feature encoding and shallow variational circuit templates."""

from typing import Any

from ..ops import RX, RY, RZ, CNOT
from ..wires import device_wires


_ROTATIONS = {"X": RX, "Y": RY, "Z": RZ}


def _rotation(name: str):
    try:
        return _ROTATIONS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError("rotation must be 'X', 'Y', or 'Z'.") from exc


def _wires(wires: object) -> tuple[int, ...]:
    return device_wires(wires)


def AngleEmbedding(features: Any, wires: object, rotation: str = "X") -> None:
    """Encode up to one feature per wire with single-qubit rotations."""
    rotate = _rotation(rotation)
    selected_wires = _wires(wires)
    if getattr(features, "ndim", 1) != 1:
        raise ValueError("features must be one-dimensional.")
    if isinstance(features, (str, bytes)):
        raise TypeError("features must be a sequence of real scalars.")
    try:
        length = len(features)
    except TypeError as exc:
        raise TypeError("features must be a one-dimensional sequence.") from exc
    if length > len(selected_wires):
        raise ValueError("More features were provided than available wires.")
    for angle, wire in zip(features, selected_wires):
        rotate(angle, wires=wire)


def BasicEntanglerLayers(weights: Any, wires: object, rotation: str = "X") -> None:
    """Apply rotations and a nearest-neighbor CNOT ring at each layer."""
    rotate = _rotation(rotation)
    selected_wires = _wires(wires)
    if getattr(weights, "ndim", 2) != 2:
        raise ValueError("weights must have shape (layers, wires).")
    if isinstance(weights, (str, bytes)):
        raise TypeError("weights must be a two-dimensional collection.")
    try:
        num_layers = len(weights)
        if num_layers < 1 or any(len(layer) != len(selected_wires) for layer in weights):
            raise ValueError("weights must have shape (layers, wires), with at least one layer.")
    except TypeError as exc:
        raise TypeError("weights must have shape (layers, wires).") from exc

    for layer in weights:
        for angle, wire in zip(layer, selected_wires):
            rotate(angle, wires=wire)
        if len(selected_wires) == 2:
            CNOT(wires=(selected_wires[0], selected_wires[1]))
        elif len(selected_wires) > 2:
            for index, wire in enumerate(selected_wires):
                CNOT(wires=(wire, selected_wires[(index + 1) % len(selected_wires)]))
