"""Small checks for the default simulator selection and real-data example."""

import importlib

import numpy as np

import miniqml as qml
from examples.iris_classifier import iris_data


def test_auto_backend_selection(monkeypatch):
    device_module = importlib.import_module("miniqml.device")
    monkeypatch.setattr(device_module, "find_spec", lambda name: object())
    assert qml.device(wires=2).name == "qiskit.aer"
    monkeypatch.setattr(device_module, "find_spec", lambda name: None)
    assert qml.device(wires=2).name == "qiskit.statevector"
    assert qml.device("qiskit.auto", wires=2).name == "qiskit.statevector"


def test_iris_sample_split_and_preprocessing():
    x_train, x_test, y_train, y_test, metadata = iris_data()
    assert x_train.shape == (40, 2)
    assert x_test.shape == (20, 2)
    assert y_train.sum().item() == 20
    assert y_test.sum().item() == 10
    assert metadata["classes"] == ["setosa", "versicolor"]
    assert np.isfinite(x_train.numpy()).all()
    assert np.abs(x_test.numpy()).max() <= 1.5
    assert np.array_equal(iris_data()[0].numpy(), x_train.numpy())
