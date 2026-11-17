"""Device interface and backend selection."""

from abc import ABC, abstractmethod
from dataclasses import replace
from collections.abc import Sequence
from numbers import Integral
from importlib.util import find_spec
from typing import Any

from .ir import QuantumTape
from .wires import device_wires


class Device(ABC):
    def __init__(self, name: str, wires: object, shots: int | None = None) -> None:
        if shots is not None and (
            isinstance(shots, bool) or not isinstance(shots, Integral) or shots <= 0
        ):
            raise ValueError("shots must be None or a positive integer.")
        self.name = name
        self.wires = device_wires(wires)
        self.shots = int(shots) if shots is not None else None

    @abstractmethod
    def execute(self, tape: QuantumTape) -> Any:
        """Execute a recorded circuit and return its measurement result."""

    def execute_many(
        self, tape: QuantumTape, parameter_sets: Sequence[Sequence[float]]
    ) -> list[float]:
        """Execute multiple rotation bindings, with a generic fallback."""
        positions = [
            i for i, operation in enumerate(tape.operations)
            if operation.name in {"RX", "RY", "RZ"}
        ]
        results = []
        for values in parameter_sets:
            if len(values) != len(positions):
                raise ValueError("Parameter count does not match the circuit.")
            operations = list(tape.operations)
            for index, value in zip(positions, values):
                operations[index] = replace(operations[index], parameters=(float(value),))
            results.append(float(self.execute(replace(tape, operations=tuple(operations)))))
        return results


def device(name: str = "qiskit.auto", wires: object = None, shots: int | None = None) -> Device:
    """Create a Qiskit-backed simulator, choosing a backend by default.

    The automatic choice uses Aer if installed and falls back to Qiskit's
    built-in statevector estimator. All supported backends use exact
    expectation values.
    """
    if name not in {"qiskit.auto", "qiskit.statevector", "qiskit.aer"}:
        raise ValueError(f"Unsupported device {name!r}.")
    if shots is not None:
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots <= 0:
            raise ValueError("shots must be None or a positive integer.")
        raise NotImplementedError("Finite-shot execution is not supported.")

    if name == "qiskit.auto":
        name = "qiskit.aer" if find_spec("qiskit_aer") is not None else "qiskit.statevector"

    from .devices.qiskit import QiskitAerDevice, QiskitStatevectorDevice

    backend_type = (
        QiskitStatevectorDevice if name == "qiskit.statevector" else QiskitAerDevice
    )
    return backend_type(wires=wires)
