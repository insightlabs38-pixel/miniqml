# miniqml

A small PennyLane-style quantum machine learning library using Qiskit simulators and PyTorch. Requires Python 3.11+.

## Install

```bash
pip install -e '.[demo]'
```

## Example

```python
import torch
import miniqml as qml

dev = qml.device(wires=2)

@qml.qnode(dev)
def circuit(x, weights):
    qml.AngleEmbedding(x, wires=[0, 1], rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=[0, 1], rotation="Y")
    return qml.expval(qml.PauliZ(1))

model = qml.QuantumLayer(circuit, weight_shape=(2, 2))
features = torch.rand(8, 2, dtype=torch.float64)
labels = torch.zeros(8, dtype=torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

optimizer.zero_grad()
loss = ((model(features) - labels) ** 2).mean()
loss.backward()
optimizer.step()
```

`qml.device(wires=2)` automatically selects Qiskit Aer if available, otherwise Qiskit's built-in statevector estimator. To choose explicitly, use `qml.device("qiskit.aer", wires=2)` or `qml.device("qiskit.statevector", wires=2)`.

## Iris classifier

Train a two-qubit variational classifier on 60 real Iris flower samples (40 train, 20 test). The example uses petal length and width to distinguish setosa from versicolor, and reports measured accuracy, loss, and quantum gradients.

```bash
python examples/iris_classifier.py
```

Training requires no GPU.

## Features and limits

- RX, RY, RZ, H, X, CNOT, CZ; Pauli X/Y/Z expectation values.
- QNodes, `AngleEmbedding`, `BasicEntanglerLayers`, and `QuantumLayer`.
- PyTorch first-order parameter-shift gradients and batched Qiskit parameter sweeps.
- Exact noiseless simulation. Finite-shot sampling, multiple measurements per circuit, and higher-order gradients are not supported.

## Tests

```bash
pip install -e '.[test,demo]'
pytest -q
```
