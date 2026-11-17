"""Integer wire labels for the first version of the API."""

from numbers import Integral


def _as_wire(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Wire labels must be non-negative integers (not bool).")
    return int(value)


def device_wires(wires: object) -> tuple[int, ...]:
    if isinstance(wires, Integral) and not isinstance(wires, bool):
        if wires <= 0:
            raise ValueError("The number of device wires must be positive.")
        return tuple(range(wires))
    if not isinstance(wires, (tuple, list)) or not wires:
        raise ValueError("Device wires must be a positive integer or nonempty list/tuple.")
    result = tuple(_as_wire(wire) for wire in wires)
    if len(result) != len(set(result)):
        raise ValueError("Device wires cannot contain duplicates.")
    return result


def operation_wires(wires: object, count: int) -> tuple[int, ...]:
    if isinstance(wires, Integral) and not isinstance(wires, bool):
        result = (_as_wire(wires),)
    elif isinstance(wires, (list, tuple)):
        result = tuple(_as_wire(wire) for wire in wires)
    else:
        raise ValueError("Operation wires must be an integer or a list/tuple of integers.")
    if len(result) != count:
        raise ValueError(f"Expected {count} wire(s), got {len(result)}.")
    if len(set(result)) != len(result):
        raise ValueError("An operation cannot use the same wire twice.")
    return result
