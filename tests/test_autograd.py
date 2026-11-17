"""Focused autograd tests without a Qiskit installation."""

import math

import pytest
import torch

import miniqml as qml


class CosineDevice(qml.Device):
    def __init__(self):
        super().__init__("cosine", wires=1)

    def execute(self, tape):
        return math.cos(sum(
            float(op.parameters[0]) for op in tape.operations if op.name == "RY"
        ))


def test_parameter_shift_matches_analytic_gradient():
    @qml.qnode(CosineDevice())
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = torch.tensor(0.42, dtype=torch.float64, requires_grad=True)
    expectation = circuit(theta)
    assert isinstance(expectation, torch.Tensor)
    expectation.backward()
    assert expectation.item() == pytest.approx(math.cos(0.42))
    assert theta.grad.item() == pytest.approx(-math.sin(0.42), abs=1e-12)


def test_reused_weights_and_classical_chain_rule():
    @qml.qnode(CosineDevice())
    def circuit(x, weights):
        qml.RY(x / 2, wires=0)
        qml.RY(weights[0], wires=0)
        qml.RY(2 * weights[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    x = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    circuit(x, weight).backward()
    sine = math.sin(0.6 / 2 + 3 * 0.25)
    assert x.grad.item() == pytest.approx(-0.5 * sine, abs=1e-12)
    assert weight.grad[0].item() == pytest.approx(-3 * sine, abs=1e-12)


def test_optimizer_step_reduces_loss():
    @qml.qnode(CosineDevice())
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    optimizer = torch.optim.SGD([theta], lr=0.1)

    def loss_fn():
        return (circuit(theta) - 0.2).square()

    initial_loss = loss_fn().item()
    optimizer.zero_grad()
    loss_fn().backward()
    optimizer.step()
    assert loss_fn().item() < initial_loss
