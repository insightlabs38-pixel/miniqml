"""A minimal PennyLane-style circuit API."""

from .device import Device, device
from .gradients import parameter_shift
from .layers import QuantumLayer
from .templates import AngleEmbedding, BasicEntanglerLayers
from .ir import Measurement, Observable, Operation, QuantumTape
from .measurements import PauliX, PauliY, PauliZ, expval
from .ops import CNOT, CZ, H, RX, RY, RZ, X, GATE_SPECS, Hadamard
from .qnode import QNode, qnode

__version__ = "0.5.0"

__all__ = [
    "AngleEmbedding", "BasicEntanglerLayers", "CNOT", "CZ", "Device", "GATE_SPECS", "H", "Hadamard", "Measurement",
    "Observable", "Operation", "QuantumLayer", "PauliX", "PauliY", "PauliZ", "QNode",
    "QuantumTape", "RX", "RY", "RZ", "X", "device", "expval", "parameter_shift", "qnode",
]
