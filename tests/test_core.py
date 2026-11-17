from dataclasses import FrozenInstanceError

import pytest

import miniqml as qml


@pytest.fixture
def dev():
    return qml.device("qiskit.aer", wires=2)


def test_records_circuit_and_its_measurement(dev):
    @qml.qnode(dev)
    def circuit(theta):
        qml.H(wires=0)
        qml.RX(theta, wires=0)
        qml.RY(-0.2, wires=1)
        qml.RZ(0.3, wires=1)
        qml.X(wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CZ(wires=(1, 0))
        return qml.expval(qml.PauliY(wires=1))

    tape = circuit.construct(0.5)
    assert tape.wires == (0, 1)
    assert [op.name for op in tape.operations] == [
        "Hadamard", "RX", "RY", "RZ", "X", "CNOT", "CZ"
    ]
    assert tape.operations[1].parameters == (0.5,)
    assert tape.operations[-2].wires == (0, 1)
    assert tape.measurement.observable == qml.PauliY(1)
    assert circuit.__name__ == "circuit"
    with pytest.raises(FrozenInstanceError):
        tape.wires = (1,)


def test_construct_without_gates(dev):
    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.PauliZ(0))

    assert circuit.construct().operations == ()


def test_device_dispatch_uses_constructed_tape():
    class EchoDevice(qml.Device):
        def execute(self, tape):
            return tape.measurement.observable.name

    @qml.qnode(EchoDevice("echo", wires=[5, 7]))
    def circuit():
        qml.CNOT(wires=[5, 7])
        return qml.expval(qml.PauliZ(7))

    assert circuit() == "PauliZ"


@pytest.mark.parametrize("wire_args", [0, -1, True, [], [0, 0], [0, -1], [False]])
def test_invalid_device_wires(wire_args):
    with pytest.raises(ValueError):
        qml.device("qiskit.aer", wires=wire_args)


def test_unrecognized_device_or_shots():
    with pytest.raises(ValueError, match="Unsupported device"):
        qml.device("unknown", wires=1)
    with pytest.raises(NotImplementedError, match="Finite-shot"):
        qml.device("qiskit.aer", wires=1, shots=100)


@pytest.mark.parametrize("wires", [[0, 0], [0], [1, 2], [-1, 0], [True, 0]])
def test_bad_cnot_wires(dev, wires):
    @qml.qnode(dev)
    def circuit():
        qml.CNOT(wires=wires)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError):
        circuit.construct()


@pytest.mark.parametrize("angle", [None, True, float("nan"), float("inf"), 1j, [0.1], "0.1"])
def test_invalid_angles(dev, angle):
    @qml.qnode(dev)
    def circuit():
        qml.RX(angle, wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises((TypeError, ValueError)):
        circuit.construct()


def test_bad_gate_after_measurement(dev):
    @qml.qnode(dev)
    def circuit():
        result = qml.expval(qml.PauliZ(0))
        qml.RY(1.2, wires=0)
        return result

    with pytest.raises(ValueError, match="Gates cannot follow"):
        circuit.construct()


def test_requires_returned_measurement(dev):
    @qml.qnode(dev)
    def circuit():
        qml.RY(1.2, wires=0)
        return None

    with pytest.raises(ValueError, match="exactly one"):
        circuit.construct()

    @qml.qnode(dev)
    def circuit2():
        qml.expval(qml.PauliZ(0))
        return None

    with pytest.raises(ValueError, match="return"):
        circuit2.construct()


def test_rejects_multiple_measurements(dev):
    @qml.qnode(dev)
    def circuit():
        qml.expval(qml.PauliX(0))
        return qml.expval(qml.PauliZ(1))

    with pytest.raises(ValueError, match="one expectation"):
        circuit.construct()


def test_rejects_invalid_qnode_arguments(dev):
    with pytest.raises(ValueError, match="interface"):
        qml.qnode(dev, interface="jax")(lambda: None)
    with pytest.raises(ValueError, match="diff_method"):
        qml.qnode(dev, diff_method="adjoint")(lambda: None)
    with pytest.raises(TypeError, match="Device"):
        qml.qnode("qiskit.aer")(lambda: None)


def test_recording_context_is_cleaned_after_error(dev):
    @qml.qnode(dev)
    def broken():
        qml.RY(0.5, wires=99)
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev)
    def valid():
        qml.RX(0.5, wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError, match="not on this device"):
        broken.construct()
    assert valid.construct().operations[0].name == "RX"


def test_nested_recordings_rejected(dev):
    @qml.qnode(dev)
    def inner():
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev)
    def outer():
        inner.construct()
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(RuntimeError, match="Nested"):
        outer.construct()


def test_requires_active_recording():
    with pytest.raises(RuntimeError, match="active QNode"):
        qml.H(wires=0)
    with pytest.raises(RuntimeError, match="active QNode"):
        qml.expval(qml.PauliZ(0))


def test_scalar_torch_parameter_is_preserved(dev):
    torch = pytest.importorskip("torch")
    theta = torch.tensor(0.2, requires_grad=True)

    @qml.qnode(dev)
    def circuit(value):
        qml.RY(value, wires=1)
        return qml.expval(qml.PauliX(1))

    assert circuit.construct(theta).operations[0].parameters[0] is theta
    with pytest.raises(TypeError):
        circuit.construct(torch.tensor([0.2, 0.4]))


def test_gate_registry_has_expected_arities():
    assert qml.GATE_SPECS["RY"].parameters == 1
    assert qml.GATE_SPECS["CNOT"].wires == 2
    with pytest.raises(TypeError):
        qml.GATE_SPECS["XX"] = object()


def test_observable_must_use_device_wire(dev):
    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.PauliZ(5))

    with pytest.raises(ValueError, match="not on this device"):
        circuit.construct()


def test_expval_rejects_non_observables(dev):
    @qml.qnode(dev)
    def circuit():
        return qml.expval("Z")

    with pytest.raises(TypeError, match="Pauli observable"):
        circuit.construct()


def test_reconstruction_creates_independent_tapes(dev):
    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    first = circuit.construct(0.1)
    second = circuit.construct(0.2)
    assert first.operations[0].parameters == (0.1,)
    assert second.operations[0].parameters == (0.2,)
