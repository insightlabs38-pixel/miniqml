"""Parameter-shift derivatives for RX, RY, and RZ gate angles."""

import math
from collections.abc import Sequence

from ..device import Device
from ..ir import QuantumTape


def parameter_shift(
    device: Device,
    tape: QuantumTape,
    values: Sequence[float],
    indices: Sequence[int] | None = None,
) -> tuple[float, ...]:
    """Differentiate an expectation value with respect to rotation occurrences.

    Indices refer to rotation gate occurrences, not unique user-defined weights.
    Classical chain rules are handled by the calling tensor framework.
    """
    count = sum(op.name in {"RX", "RY", "RZ"} for op in tape.operations)
    if len(values) != count:
        raise ValueError(f"Expected {count} rotation angle values, got {len(values)}.")
    if indices is None:
        indices = tuple(range(count))
    if any(not isinstance(index, int) or index < 0 or index >= count for index in indices):
        raise ValueError("Invalid rotation angle index.")
    if not indices:
        return ()

    shifted = []
    for index in indices:
        plus = list(values)
        minus = list(values)
        plus[index] += math.pi / 2
        minus[index] -= math.pi / 2
        shifted.extend((plus, minus))

    results = device.execute_many(tape, shifted)
    if len(results) != len(shifted):
        raise RuntimeError("The device returned the wrong number of expectation values.")
    return tuple(
        (float(results[2 * index]) - float(results[2 * index + 1])) / 2
        for index in range(len(indices))
    )


def batched_parameter_shift(
    device: Device,
    tape: QuantumTape,
    parameter_sets: Sequence[Sequence[float]],
    indices: Sequence[int],
) -> tuple[float, ...]:
    """Shift selected flattened (sample, rotation occurrence) positions in one sweep."""
    if not parameter_sets:
        raise ValueError("At least one parameter set is required.")
    width = sum(op.name in {"RX", "RY", "RZ"} for op in tape.operations)
    if any(len(row) != width for row in parameter_sets):
        raise ValueError("Parameter count does not match the circuit.")
    if any(not isinstance(index, int) or index < 0 or index >= len(parameter_sets) * width
           for index in indices):
        raise ValueError("Invalid flattened rotation angle index.")
    if not indices:
        return ()

    shifted = []
    for index in indices:
        sample_index, angle_index = divmod(index, width)
        plus = list(parameter_sets[sample_index])
        minus = plus.copy()
        plus[angle_index] += math.pi / 2
        minus[angle_index] -= math.pi / 2
        shifted.extend((plus, minus))
    results = device.execute_many(tape, shifted)
    if len(results) != len(shifted):
        raise RuntimeError("The device returned the wrong number of expectation values.")
    return tuple(
        (float(results[2 * i]) - float(results[2 * i + 1])) / 2
        for i in range(len(indices))
    )
