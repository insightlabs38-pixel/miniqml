"""Qiskit API contract tests using small test doubles when Qiskit is unavailable."""

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
import sys

import pytest

import miniqml as qml
from miniqml.devices.qiskit import compile_tape


@dataclass(frozen=True)
class FakeParameter:
    name: str


class FakeQuantumCircuit:
    def __init__(self, n):
        self.num_qubits = n
        self.ops = []
        self.assigned = None

    def _record(self, name, *args):
        self.ops.append((name, *args))

    def rx(self, *args): self._record("RX", *args)
    def ry(self, *args): self._record("RY", *args)
    def rz(self, *args): self._record("RZ", *args)
    def h(self, *args): self._record("Hadamard", *args)
    def x(self, *args): self._record("X", *args)
    def cx(self, *args): self._record("CNOT", *args)
    def cz(self, *args): self._record("CZ", *args)

    def assign_parameters(self, values):
        bound = FakeQuantumCircuit(self.num_qubits)
        bound.ops = self.ops[:]
        bound.assigned = values
        return bound


class FakeSparsePauliOp:
    @staticmethod
    def from_sparse_list(terms, num_qubits):
        return SimpleNamespace(terms=terms, num_qubits=num_qubits)


class FakeBindingsArray:
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape


class FakeJob:
    def __init__(self, pubs):
        self.pubs = pubs

    def result(self):
        if len(self.pubs[0]) == 3:
            bindings = self.pubs[0][2]
            rows = next(iter(bindings.data.values()))
            return [SimpleNamespace(data=SimpleNamespace(evs=np.sum(rows, axis=1)))]
        return [SimpleNamespace(data=SimpleNamespace(evs=SimpleNamespace(item=lambda: 0.625)))]


class FakeEstimator:
    def __init__(self):
        self.pubs = None

    def run(self, pubs):
        self.pubs = pubs
        return FakeJob(pubs)


@pytest.fixture
def fake_qiskit(monkeypatch):
    qiskit = ModuleType("qiskit")
    qiskit.QuantumCircuit = FakeQuantumCircuit
    circuit_module = ModuleType("qiskit.circuit")
    circuit_module.Parameter = FakeParameter
    info_module = ModuleType("qiskit.quantum_info")
    info_module.SparsePauliOp = FakeSparsePauliOp
    primitives = ModuleType("qiskit.primitives")
    primitives.StatevectorEstimator = FakeEstimator
    primitives.BindingsArray = FakeBindingsArray
    aer = ModuleType("qiskit_aer")
    aer_primitives = ModuleType("qiskit_aer.primitives")
    aer_primitives.EstimatorV2 = FakeEstimator
    for name, module in [
        ("qiskit", qiskit), ("qiskit.circuit", circuit_module),
        ("qiskit.quantum_info", info_module), ("qiskit.primitives", primitives),
        ("qiskit_aer", aer), ("qiskit_aer.primitives", aer_primitives),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(qiskit=qiskit, aer=aer)


def test_compiles_gate_order_wires_observable_and_values(fake_qiskit):
    dev = qml.device("qiskit.statevector", wires=[7, 3])

    @qml.qnode(dev)
    def circuit(a, b):
        qml.RX(a, wires=7)
        qml.RY(b, wires=3)
        qml.RZ(-0.3, wires=7)
        qml.H(7)
        qml.X(3)
        qml.CNOT([3, 7])
        qml.CZ([7, 3])
        return qml.expval(qml.PauliY(3))

    compiled = compile_tape(circuit.construct(0.2, 0.4))
    assert compiled.circuit.num_qubits == 2
    assert [op[0] for op in compiled.circuit.ops] == [
        "RX", "RY", "RZ", "Hadamard", "X", "CNOT", "CZ"
    ]
    assert compiled.circuit.ops[5] == ("CNOT", 1, 0)
    assert compiled.observable.terms == [("Y", [1], 1.0)]
    assert compiled.values == (0.2, 0.4, -0.3)
    bound = compiled.bind()
    assert bound.assigned == {
        FakeParameter("angle_0"): 0.2,
        FakeParameter("angle_1"): 0.4,
        FakeParameter("angle_2"): -0.3,
    }


@pytest.mark.parametrize("backend", ["qiskit.statevector", "qiskit.aer"])
def test_estimator_dispatch_and_result(backend, fake_qiskit):
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit(x):
        qml.RY(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert circuit(0.25) == 0.625
    assert dev._estimator.pubs[0][0].assigned == {FakeParameter("angle_0"): 0.25}
    assert dev._estimator.pubs[0][1].terms == [("Z", [0], 1.0)]
    instance = dev._estimator
    circuit(0.7)
    assert dev._estimator is instance


def test_no_parameter_binding(fake_qiskit):
    dev = qml.device("qiskit.statevector", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.H(0)
        return qml.expval(qml.PauliX(0))

    program = compile_tape(circuit.construct())
    assert program.parameters == ()
    assert program.bind() is program.circuit


def test_bad_manual_tape_rejected(fake_qiskit):
    from miniqml.ir import Operation, QuantumTape

    dev = qml.device("qiskit.aer", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.PauliZ(0))

    tape = circuit.construct()
    bad_tape = QuantumTape(
        wires=tape.wires, operations=(Operation("Bogus", (0,)),),
        measurements=tape.measurements,
    )
    with pytest.raises(ValueError, match="Unsupported gate"):
        dev.execute(bad_tape)


def test_wrong_device_wires_rejected(fake_qiskit):
    device_a = qml.device("qiskit.aer", wires=[0, 1])
    device_b = qml.device("qiskit.aer", wires=[1, 0])

    @qml.qnode(device_a)
    def circuit():
        return qml.expval(qml.PauliZ(1))

    with pytest.raises(ValueError, match="do not match"):
        device_b.execute(circuit.construct())


def test_parameter_sweep_keeps_gate_order(fake_qiskit):
    dev = qml.device("qiskit.statevector", wires=1)

    @qml.qnode(dev)
    def circuit(a):
        for i in range(12):
            qml.RY(a + i * 0.01, wires=0)
        return qml.expval(qml.PauliZ(0))

    tape = circuit.construct(0.2)
    first = [float(i) for i in range(12)]
    second = [float(i + 1) for i in range(12)]
    assert dev.execute_many(tape, [first, second]) == [sum(first), sum(second)]
    bindings = dev._estimator.pubs[0][2]
    assert bindings.shape == (2,)
    assert [param.name for param in next(iter(bindings.data))] == [
        f"angle_{i}" for i in range(12)
    ]
