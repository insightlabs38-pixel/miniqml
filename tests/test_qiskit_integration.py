"""End-to-end tests, run when Qiskit is installed."""

import math

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

import miniqml as qml
from miniqml.devices.qiskit import compile_tape


@pytest.fixture(params=["qiskit.statevector", "qiskit.aer"])
def backend(request):
    return request.param


def test_ry_z_expectation(backend):
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    for theta in (0.0, 0.3, math.pi / 2, 1.7, math.pi):
        assert circuit(theta) == pytest.approx(math.cos(theta), abs=1e-10)


def test_other_pauli_and_rotations(backend):
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def xy_circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliY(0))

    @qml.qnode(dev)
    def x_circuit(theta):
        qml.H(0)
        qml.RZ(theta, wires=0)
        return qml.expval(qml.PauliX(0))

    assert xy_circuit(0.6) == pytest.approx(-math.sin(0.6), abs=1e-10)
    assert x_circuit(0.6) == pytest.approx(math.cos(0.6), abs=1e-10)


def test_ry_x_expectation(backend):
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliX(0))

    assert circuit(0.7) == pytest.approx(math.sin(0.7), abs=1e-10)


def test_wire_mapping_and_cnot_order(backend):
    dev = qml.device(backend, wires=[5, 2])

    @qml.qnode(dev)
    def circuit():
        qml.X(wires=5)
        qml.CNOT(wires=[5, 2])
        return qml.expval(qml.PauliZ(2))

    assert circuit() == pytest.approx(-1.0, abs=1e-10)

    tape = circuit.construct()
    compiled = compile_tape(tape)
    assert str(compiled.observable.paulis[0]) == "ZI"
    assert [gate.operation.name for gate in compiled.circuit.data] == ["x", "cx"]


def test_h_and_cz_and_h(backend):
    dev = qml.device(backend, wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.H(wires=0)
        qml.X(wires=1)
        qml.CZ(wires=[0, 1])
        qml.H(wires=0)
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(-1.0, abs=1e-10)


def test_bell_state_single_qubit_measurements(backend):
    dev = qml.device(backend, wires=2)

    @qml.qnode(dev)
    def bell():
        qml.H(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    assert bell() == pytest.approx(0.0, abs=1e-10)


def test_many_rotations_bind_in_gate_order(backend):
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit():
        for k in range(13):
            qml.RY(k / 100, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(math.cos(sum(k / 100 for k in range(13))), abs=1e-10)


def test_tensor_parameter_is_accepted(backend):
    torch = pytest.importorskip("torch")
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = torch.tensor(0.25, requires_grad=True)
    expectation = circuit(theta)
    assert isinstance(expectation, torch.Tensor)
    assert expectation.item() == pytest.approx(math.cos(0.25), abs=1e-7)
    expectation.backward()
    assert theta.grad.item() == pytest.approx(-math.sin(0.25), abs=1e-7)


def test_no_operations(backend):
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.PauliZ(0))

    assert circuit() == pytest.approx(1.0, abs=1e-10)


def test_invalid_tensor_parameter_rejected(backend):
    torch = pytest.importorskip("torch")
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError, match="finite"):
        circuit(torch.tensor(float("nan")))


def test_reused_weights_qiskit_gradient(backend):
    torch = pytest.importorskip("torch")
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit(weights):
        qml.RY(weights[0], wires=0)
        qml.RY(2 * weights[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    weights = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    circuit(weights).backward()
    assert weights.grad[0].item() == pytest.approx(-3 * math.sin(0.6), abs=1e-9)


def test_two_qubit_torch_gradients(backend):
    torch = pytest.importorskip("torch")
    dev = qml.device(backend, wires=2)

    @qml.qnode(dev)
    def circuit(x, weights):
        qml.RY(x, wires=0)
        qml.RY(weights[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    x = torch.tensor(0.3, dtype=torch.float64)
    weights = torch.tensor([0.4, 0.2], dtype=torch.float64, requires_grad=True)
    output = circuit(x, weights)
    output.backward()
    assert output.item() == pytest.approx(math.cos(0.7) * math.cos(0.2), abs=1e-9)
    assert weights.grad.tolist() == pytest.approx([
        -math.sin(0.7) * math.cos(0.2),
        -math.cos(0.7) * math.sin(0.2),
    ], abs=1e-9)


def test_batched_qnode_qiskit_gradients(backend):
    torch = pytest.importorskip("torch")
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=[0], rotation="Y")
        qml.RY(weights[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    x = torch.tensor([[-0.3], [0.2], [0.9]], dtype=torch.float64)
    weights = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
    prediction = circuit.batch(x, weights)
    expected = torch.cos(x[:, 0] + weights[0])
    assert torch.allclose(prediction, expected, atol=1e-10)
    prediction.sum().backward()
    assert weights.grad.item() == pytest.approx(
        -torch.sin(x[:, 0] + 0.4).sum().item(), abs=1e-9
    )


def test_default_device_executes_qiskit():
    dev = qml.device(wires=1)
    assert dev.name == "qiskit.aer"

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert circuit(0.4) == pytest.approx(math.cos(0.4), abs=1e-10)
