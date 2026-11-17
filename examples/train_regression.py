"""Train a two-qubit quantum regressor with a batched QNode."""

import torch

import miniqml as qml


def main():
    torch.manual_seed(7)
    dev = qml.device("qiskit.statevector", wires=2)

    @qml.qnode(dev)
    def circuit(features, weights):
        qml.AngleEmbedding(features, wires=[0, 1], rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=[0, 1], rotation="Y")
        return qml.expval(qml.PauliZ(1))

    grid = torch.linspace(-1.2, 1.2, 12, dtype=torch.float64)
    x_train = torch.stack((grid, torch.sin(2 * grid)), dim=1)
    teacher_weights = torch.tensor(
        [[1.2, -0.9], [0.5, 1.0]], dtype=torch.float64
    )
    with torch.no_grad():
        y_train = circuit.batch(x_train, teacher_weights)

    model = qml.QuantumLayer(
        circuit,
        weight_shape=(2, 2),
        initial_weights=torch.tensor(
            [[-0.15, 0.1], [0.1, -0.05]], dtype=torch.float64
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.12)

    for step in range(25):
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = torch.mean((prediction - y_train) ** 2)
        loss.backward()
        optimizer.step()
        if step % 5 == 0 or step == 24:
            print(f"step={step:02d} mse={loss.item():.6f}")

    torch.save(model.state_dict(), "quantum_regressor.pt")
    print("Saved model weights to quantum_regressor.pt")


if __name__ == "__main__":
    main()
