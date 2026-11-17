"""PyTorch module wrapper for a trainable QNode."""

import torch
from torch import nn

from .qnode import QNode


class QuantumLayer(nn.Module):
    """Call a QNode with (features, trainable_weights) on single or batched input."""

    def __init__(
        self,
        circuit: QNode,
        weight_shape: int | tuple[int, ...],
        *,
        initial_weights: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if not isinstance(circuit, QNode):
            raise TypeError("circuit must be a miniqml QNode.")
        if isinstance(weight_shape, int) and not isinstance(weight_shape, bool):
            weight_shape = (weight_shape,)
        if (
            not isinstance(weight_shape, (tuple, list, torch.Size))
            or not weight_shape
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                   for dim in weight_shape)
        ):
            raise ValueError("weight_shape must contain positive integers.")
        shape = tuple(weight_shape)
        self.circuit = circuit
        if initial_weights is None:
            initial = torch.empty(shape, dtype=dtype).uniform_(-0.1, 0.1)
        else:
            initial = torch.as_tensor(initial_weights, dtype=dtype).detach().clone()
            if tuple(initial.shape) != shape:
                raise ValueError("initial_weights do not match weight_shape.")
        self.weights = nn.Parameter(initial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("QuantumLayer input must be a torch.Tensor.")
        if x.ndim == 1:
            return self.circuit(x, self.weights)
        if x.ndim == 2:
            return self.circuit.batch(x, self.weights)
        raise ValueError("QuantumLayer expects shape (features,) or (batch, features).")
