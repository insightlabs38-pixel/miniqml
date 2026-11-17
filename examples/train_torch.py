"""Train a small variational circuit using PyTorch and Qiskit."""

import torch

import miniqml as qml


torch.manual_seed(3)
dev = qml.device("qiskit.statevector", wires=2)


@qml.qnode(dev)
def model(x, weights):
    qml.RY(x, wires=0)
    qml.RY(weights[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[1], wires=1)
    return qml.expval(qml.PauliZ(1))


x_train = torch.linspace(-1, 1, 6, dtype=torch.float64)
y_train = torch.cos(x_train + 0.7) * torch.cos(torch.tensor(0.4))
weights = torch.nn.Parameter(torch.tensor([0.1, 0.2], dtype=torch.float64))
optimizer = torch.optim.Adam([weights], lr=0.12)

for step in range(20):
    optimizer.zero_grad()
    predictions = torch.stack([model(x, weights) for x in x_train])
    loss = torch.mean((predictions - y_train) ** 2)
    loss.backward()
    optimizer.step()
    if step % 5 == 0 or step == 19:
        print(f"step={step:02d} loss={loss.item():.6f} weights={weights.tolist()}")
