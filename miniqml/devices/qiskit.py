"""Qiskit circuit translation and exact expectation-value backends."""

from dataclasses import dataclass, replace
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..device import Device
from ..ir import QuantumTape


@dataclass(frozen=True)
class QiskitProgram:
    """A parameterized circuit with values for each rotation occurrence."""

    circuit: Any
    observable: Any
    parameters: tuple[Any, ...]
    values: tuple[float, ...]

    def bind(self) -> Any:
        mapping = dict(zip(self.parameters, self.values))
        return self.circuit.assign_parameters(mapping) if mapping else self.circuit


def _scalar_float(value: Any) -> float:
    # Tensor values are converted only at the simulator boundary.
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "item", None)):
        value = value.item()
    if isinstance(value, (complex, bool)):
        raise TypeError("Rotation angles must be real scalars.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Rotation angles must be real scalars.") from exc
    if not math.isfinite(converted):
        raise ValueError("Rotation angles must be finite.")
    return converted


def compile_tape(tape: QuantumTape) -> QiskitProgram:
    """Translate a tape to a parameterized Qiskit circuit and Pauli observable."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp

    if len(tape.measurements) != 1 or tape.measurement.kind != "expval":
        raise ValueError("Exactly one expectation measurement is required.")

    positions = {wire: position for position, wire in enumerate(tape.wires)}
    circuit = QuantumCircuit(len(positions))
    parameters = []
    values = []
    gate_methods = {
        "RX": circuit.rx,
        "RY": circuit.ry,
        "RZ": circuit.rz,
        "Hadamard": circuit.h,
        "X": circuit.x,
        "CNOT": circuit.cx,
        "CZ": circuit.cz,
    }
    for index, operation in enumerate(tape.operations):
        try:
            wires = [positions[wire] for wire in operation.wires]
        except KeyError as exc:
            raise ValueError(f"Unknown operation wire: {exc.args[0]}.") from exc
        try:
            method = gate_methods[operation.name]
        except KeyError as exc:
            raise ValueError(f"Unsupported gate: {operation.name}.") from exc

        if operation.name in {"RX", "RY", "RZ"}:
            if len(operation.parameters) != 1 or len(wires) != 1:
                raise ValueError(f"Malformed {operation.name} operation.")
            parameter = Parameter(f"angle_{index}")
            method(parameter, wires[0])
            parameters.append(parameter)
            values.append(_scalar_float(operation.parameters[0]))
        else:
            if operation.parameters:
                raise ValueError(f"Unexpected parameters for {operation.name}.")
            method(*wires)

    observable = tape.measurement.observable
    labels = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}
    try:
        label = labels[observable.name]
        observable_wire = positions[observable.wires[0]]
    except KeyError as exc:
        raise ValueError(f"Unsupported observable or wire: {exc.args[0]}.") from exc
    if len(observable.wires) != 1:
        raise ValueError("Only single-qubit Pauli observables are supported.")

    pauli = SparsePauliOp.from_sparse_list(
        [(label, [observable_wire], 1.0)], num_qubits=len(positions)
    )
    return QiskitProgram(circuit, pauli, tuple(parameters), tuple(values))


class _QiskitEstimatorDevice(Device):
    def __init__(self, name: str, wires: object, shots: int | None = None) -> None:
        super().__init__(name, wires, shots)
        if shots is not None:
            raise NotImplementedError("Finite-shot execution is not supported.")
        self._estimator: Any = None
        self._cached_key: Any = None
        self._cached_program: QiskitProgram | None = None

    def _new_estimator(self) -> Any:
        raise NotImplementedError

    def _program(self, tape: QuantumTape) -> QiskitProgram:
        # Reuse the parameterized Qiskit circuit when only angle values change.
        key = (
            tape.wires,
            tuple((op.name, op.wires) for op in tape.operations),
            tape.measurements,
        )
        if key != self._cached_key:
            self._cached_program = compile_tape(tape)
            self._cached_key = key
            return self._cached_program
        values = tuple(
            _scalar_float(op.parameters[0])
            for op in tape.operations if op.name in {"RX", "RY", "RZ"}
        )
        return replace(self._cached_program, values=values)

    def execute(self, tape: QuantumTape) -> float:
        if tape.wires != self.wires:
            raise ValueError("The circuit wires do not match this device.")
        program = self._program(tape)
        if self._estimator is None:
            self._estimator = self._new_estimator()
        result = self._estimator.run([(program.bind(), program.observable)]).result()
        # A single observable without a parameter sweep produces one scalar EV.
        return float(result[0].data.evs.item())

    def execute_many(
        self, tape: QuantumTape, parameter_sets: Sequence[Sequence[float]]
    ) -> list[float]:
        if tape.wires != self.wires:
            raise ValueError("The circuit wires do not match this device.")
        if not parameter_sets:
            return []
        program = self._program(tape)
        rows = np.asarray(parameter_sets, dtype=float)
        if rows.shape != (len(parameter_sets), len(program.parameters)):
            raise ValueError("Parameter count does not match the circuit.")
        if not np.isfinite(rows).all():
            raise ValueError("Rotation angles must be finite.")
        if not program.parameters:
            value = self.execute(tape)
            return [value] * len(parameter_sets)

        from qiskit.primitives import BindingsArray

        bindings = BindingsArray(
            {program.parameters: rows}, shape=(len(parameter_sets),)
        )
        if self._estimator is None:
            self._estimator = self._new_estimator()
        result = self._estimator.run(
            [(program.circuit, program.observable, bindings)]
        ).result()
        values = np.asarray(result[0].data.evs, dtype=float).reshape(-1)
        if len(values) != len(parameter_sets):
            raise RuntimeError("Estimator returned an unexpected number of values.")
        return values.tolist()


class QiskitStatevectorDevice(_QiskitEstimatorDevice):
    def __init__(self, wires: object, shots: int | None = None) -> None:
        super().__init__("qiskit.statevector", wires, shots)

    def _new_estimator(self) -> Any:
        from qiskit.primitives import StatevectorEstimator

        return StatevectorEstimator()


class QiskitAerDevice(_QiskitEstimatorDevice):
    def __init__(self, wires: object, shots: int | None = None) -> None:
        super().__init__("qiskit.aer", wires, shots)

    def _new_estimator(self) -> Any:
        from qiskit_aer.primitives import EstimatorV2

        return EstimatorV2()
