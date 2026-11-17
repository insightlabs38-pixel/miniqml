"""Single-qubit Pauli observables and expectation measurements."""

from .ir import Measurement, Observable
from .tape import current_recorder
from .wires import operation_wires


def PauliX(wires: object) -> Observable:
    return Observable("PauliX", operation_wires(wires, 1))


def PauliY(wires: object) -> Observable:
    return Observable("PauliY", operation_wires(wires, 1))


def PauliZ(wires: object) -> Observable:
    return Observable("PauliZ", operation_wires(wires, 1))


def expval(observable: Observable) -> Measurement:
    if not isinstance(observable, Observable) or observable.name not in {
        "PauliX", "PauliY", "PauliZ"
    }:
        raise TypeError("expval requires a supported Pauli observable.")
    measurement = Measurement("expval", observable)
    current_recorder().add_measurement(measurement)
    return measurement
