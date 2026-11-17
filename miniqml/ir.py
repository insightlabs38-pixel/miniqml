"""Immutable circuit representation shared by recorders and devices."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Operation:
    name: str
    wires: tuple[int, ...]
    parameters: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Observable:
    name: str
    wires: tuple[int, ...]


@dataclass(frozen=True)
class Measurement:
    kind: str
    observable: Observable


@dataclass(frozen=True)
class QuantumTape:
    wires: tuple[int, ...]
    operations: tuple[Operation, ...]
    measurements: tuple[Measurement, ...]

    @property
    def measurement(self) -> Measurement:
        if len(self.measurements) != 1:
            raise ValueError("A single measurement is required.")
        return self.measurements[0]
