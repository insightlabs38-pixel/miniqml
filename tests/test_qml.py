"""Focused tests for circuit templates, batching, and the PyTorch layer."""

import math

import pytest
import torch

import miniqml as qml
from miniqml.device import Device


class TrigDevice(Device):
    """A one-qubit RY expectation used to test autograd without Qiskit."""

    def __init__(self):
        super().__init__("test.cos", wires=1)
        self.calls = 0

    def execute(self, tape):
        return math.cos(sum(float(op.parameters[0]) for op in tape.operations))

    def execute_many(self, tape, parameter_sets):
        self.calls += 1
        return [math.cos(sum(row)) for row in parameter_sets]


def test_templates_record_expected_gates_and_validate_shapes():
    @qml.qnode(qml.device("qiskit.statevector", wires=[3, 5, 7]))
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=[3, 5, 7], rotation="X")
        qml.BasicEntanglerLayers(weights, wires=[3, 5, 7], rotation="Z")
        return qml.expval(qml.PauliZ(7))

    tape = circuit.construct(
        torch.tensor([0.2, 0.3], dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.float64),
    )
    assert [op.name for op in tape.operations] == ["RX", "RX"] + ["RZ"] * 3 + ["CNOT"] * 3 + ["RZ"] * 3 + ["CNOT"] * 3
    assert [op.wires for op in tape.operations if op.name == "CNOT"] == [
        (3, 5), (5, 7), (7, 3), (3, 5), (5, 7), (7, 3)
    ]
    with pytest.raises(ValueError, match="More features"):
        circuit.construct([0.1] * 4, torch.ones((2, 3)))
    with pytest.raises(ValueError, match="weights must have shape"):
        circuit.construct([0.1], torch.ones((2, 2)))


def test_batch_sweep_and_gradients_for_reused_weights_and_features():
    dev = TrigDevice()

    @qml.qnode(dev)
    def circuit(x, weights):
        qml.RY(x[0], wires=0)
        qml.RY(weights[0], wires=0)
        qml.RY(2 * weights[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    x = torch.tensor([[-0.5], [0.1], [0.4]], dtype=torch.float64, requires_grad=True)
    weights = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
    predictions = circuit.batch(x, weights)
    expected = torch.cos(x[:, 0] + 3 * weights[0])
    assert predictions.shape == (3,)
    assert torch.allclose(predictions, expected)
    predictions.sum().backward()
    assert torch.allclose(x.grad[:, 0], -torch.sin(x.detach()[:, 0] + 0.6))
    assert weights.grad.item() == pytest.approx((-3 * torch.sin(x.detach()[:, 0] + 0.6)).sum().item())
    assert dev.calls == 2


def test_quantum_layer_trains_and_registers_weights():
    dev = TrigDevice()

    @qml.qnode(dev)
    def circuit(features, weights):
        qml.AngleEmbedding(features, wires=[0], rotation="Y")
        qml.RY(weights[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    model = qml.QuantumLayer(circuit, weight_shape=(1,), initial_weights=torch.tensor([-0.3]))
    assert next(model.parameters()) is model.weights
    x = torch.linspace(-0.7, 0.9, 8, dtype=torch.float64).unsqueeze(-1)
    y = torch.cos(x[:, 0] + 0.55)
    assert model(x[0]).ndim == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.12)

    initial_loss = torch.mean((model(x).detach() - y) ** 2).item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
    assert loss.item() < initial_loss * 0.5
    assert torch.isfinite(model.weights.grad).all()


def test_batch_rejects_inconsistent_circuit_topology():
    dev = TrigDevice()

    @qml.qnode(dev)
    def circuit(x, w):
        if x[0] > 0:
            qml.X(0)
        qml.RY(w[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError, match="same circuit layout"):
        circuit.batch(torch.tensor([[-1.0], [1.0]]), torch.tensor([0.3]))
    with pytest.raises(ValueError, match="at least one sample"):
        circuit.batch(torch.empty((0, 1)), torch.tensor([0.3]))
