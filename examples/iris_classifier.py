"""Train a two-qubit classifier on a small subset of the real Iris dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import miniqml as qml


def iris_data(seed: int = 13):
    """Return a balanced 40/20 train/test split of Iris classes 0 and 1."""
    iris = load_iris()
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(iris.target == label), size=30, replace=False)
        for label in (0, 1)
    ])
    features = iris.data[selected][:, [2, 3]]
    targets = (iris.target[selected] == 1).astype(np.float64)
    train_idx, test_idx = train_test_split(
        np.arange(len(selected)), test_size=1 / 3, stratify=targets,
        random_state=seed,
    )

    scaler = StandardScaler().fit(features[train_idx])
    # Gate arguments are angles in radians. Fit the scaler on training data only.
    train_angles = np.clip(scaler.transform(features[train_idx]) * 0.8, -1.5, 1.5)
    test_angles = np.clip(scaler.transform(features[test_idx]) * 0.8, -1.5, 1.5)

    def tensor(array):
        return torch.tensor(array, dtype=torch.float64)

    metadata = {
        "dataset": "Iris",
        "classes": ["setosa", "versicolor"],
        "features": ["petal length (cm)", "petal width (cm)"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "angle_multiplier": 0.8,
        "angle_clip": [-1.5, 1.5],
        "seed": seed,
    }
    return tensor(train_angles), tensor(test_angles), tensor(targets[train_idx]), tensor(targets[test_idx]), metadata


class IrisQuantumClassifier(torch.nn.Module):
    def __init__(self, dev: qml.Device):
        super().__init__()

        @qml.qnode(dev)
        def circuit(features, weights):
            qml.AngleEmbedding(features, wires=dev.wires, rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=dev.wires, rotation="Y")
            return qml.expval(qml.PauliZ(dev.wires[-1]))

        self.quantum = qml.QuantumLayer(circuit, weight_shape=(2, 2))
        self.readout = torch.nn.Linear(1, 1, dtype=torch.float64)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.quantum(features)
        return self.readout(values.reshape(-1, 1)).reshape(-1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return ((logits >= 0) == (targets >= 0.5)).to(torch.float64).mean().item()


def run_experiment(epochs: int = 35, *, lr: float = 0.09, seed: int = 13,
                   backend: str = "qiskit.auto", output_dir: str | Path = "artifacts"):
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, x_test, y_train, y_test, metadata = iris_data(seed)
    dev = qml.device(backend, wires=2)
    model = IrisQuantumClassifier(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    history = {"epoch": [], "train_loss": [], "test_loss": [],
               "train_accuracy": [], "test_accuracy": [], "quantum_grad_norm": []}

    print(f"Backend: {dev.name} | Iris: {len(x_train)} training, {len(x_test)} test")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(x_train)
        loss = criterion(train_logits, y_train)
        loss.backward()
        grad = model.quantum.weights.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError("Quantum parameters did not receive finite gradients.")
        grad_norm = grad.detach().norm().item()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # Measurements are still computed with the configured Qiskit estimator.
            evaluated_train = model(x_train)
            evaluated_test = model(x_test)
            train_loss = criterion(evaluated_train, y_train).item()
            test_loss = criterion(evaluated_test, y_test).item()
            train_accuracy = accuracy(evaluated_train, y_train)
            test_accuracy = accuracy(evaluated_test, y_test)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)
        history["quantum_grad_norm"].append(grad_norm)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch={epoch:02d} loss={train_loss:.4f} "
                  f"train_acc={train_accuracy:.1%} test_acc={test_accuracy:.1%} "
                  f"quantum_grad={grad_norm:.5f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "iris_quantum.pt"
    history_path = output_dir / "iris_history.json"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata,
                "backend": dev.name, "epochs": epochs}, checkpoint)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {checkpoint} and {history_path}")
    return model, history, (x_test, y_test), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--backend", default="qiskit.auto",
                        choices=["qiskit.auto", "qiskit.statevector", "qiskit.aer"])
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    run_experiment(args.epochs, lr=args.lr, seed=args.seed,
                   backend=args.backend, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
