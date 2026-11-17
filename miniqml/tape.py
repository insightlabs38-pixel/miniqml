"""Context-local quantum circuit recording."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .ir import Measurement, Operation, QuantumTape


_ACTIVE_RECORDER: ContextVar["TapeRecorder | None"] = ContextVar(
    "miniqml_active_recorder", default=None
)


class TapeRecorder:
    def __init__(self, wires: tuple[int, ...]) -> None:
        self.wires = wires
        self._operations: list[Operation] = []
        self._measurements: list[Measurement] = []

    def add_operation(self, operation: Operation) -> None:
        self._validate_wires(operation.wires)
        if self._measurements:
            raise ValueError("Gates cannot follow a measurement.")
        self._operations.append(operation)

    def add_measurement(self, measurement: Measurement) -> None:
        self._validate_wires(measurement.observable.wires)
        if self._measurements:
            raise ValueError("Only one expectation measurement per QNode is supported.")
        self._measurements.append(measurement)

    def finish(self, returned: object) -> QuantumTape:
        if len(self._measurements) != 1:
            raise ValueError("A QNode must contain exactly one expval measurement.")
        if returned is not self._measurements[0]:
            raise ValueError("A QNode must return the measurement it recorded.")
        return QuantumTape(
            wires=self.wires,
            operations=tuple(self._operations),
            measurements=tuple(self._measurements),
        )

    def _validate_wires(self, wires: tuple[int, ...]) -> None:
        for wire in wires:
            if wire not in self.wires:
                raise ValueError(f"Wire {wire} is not on this device: {self.wires}.")


@contextmanager
def recording(wires: tuple[int, ...]) -> Iterator[TapeRecorder]:
    if _ACTIVE_RECORDER.get() is not None:
        raise RuntimeError("Nested quantum recordings are not supported.")
    recorder = TapeRecorder(wires)
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


def current_recorder() -> TapeRecorder:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        raise RuntimeError("Quantum operations and measurements require an active QNode.")
    return recorder
