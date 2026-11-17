"""PyTorch autograd bridge for expectation-value circuits."""

import torch
from torch.autograd.function import once_differentiable

from ..device import Device
from ..gradients import parameter_shift
from ..ir import QuantumTape


class _QuantumExpectation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, device: Device, tape: QuantumTape, *angles):
        ctx.device = device
        ctx.tape = tape
        ctx.values = tuple(
            float(angle.detach().cpu().item())
            if isinstance(angle, torch.Tensor) else float(angle)
            for angle in angles
        )
        tensors = [angle for angle in angles if isinstance(angle, torch.Tensor)]
        dtype = tensors[0].dtype
        for angle in tensors[1:]:
            dtype = torch.promote_types(dtype, angle.dtype)
        return torch.tensor(float(device.execute(tape)), dtype=dtype, device=tensors[0].device)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        indices = [
            i for i in range(len(ctx.values)) if ctx.needs_input_grad[i + 2]
        ]
        gradients = parameter_shift(ctx.device, ctx.tape, ctx.values, indices)
        result = [None] * len(ctx.values)
        for index, derivative in zip(indices, gradients):
            result[index] = grad_output * derivative
        return (None, None, *result)


def execute_torch(device: Device, tape: QuantumTape):
    """Return a differentiable tensor when circuit angles contain tensors."""
    angles = tuple(
        op.parameters[0]
        for op in tape.operations
        if op.name in {"RX", "RY", "RZ"}
    )
    tensor_angles = [angle for angle in angles if isinstance(angle, torch.Tensor)]
    if not tensor_angles:
        return device.execute(tape)

    output_device = tensor_angles[0].device
    if any(angle.device != output_device for angle in tensor_angles):
        raise ValueError("All tensor rotation angles must be on the same device.")
    return _QuantumExpectation.apply(device, tape, *angles)


def _angles(tape: QuantumTape):
    return tuple(
        operation.parameters[0]
        for operation in tape.operations
        if operation.name in {"RX", "RY", "RZ"}
    )


class _BatchedQuantumExpectation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, device: Device, tape: QuantumTape, batch_size: int,
                angle_count: int, *angles):
        ctx.device = device
        ctx.tape = tape
        ctx.angle_count = angle_count
        ctx.batch_size = batch_size
        ctx.values = tuple(
            float(angle.detach().cpu().item())
            if isinstance(angle, torch.Tensor) else float(angle)
            for angle in angles
        )
        rows = [
            ctx.values[i * angle_count:(i + 1) * angle_count]
            for i in range(batch_size)
        ]
        results = device.execute_many(tape, rows)
        tensors = [angle for angle in angles if isinstance(angle, torch.Tensor)]
        dtype = tensors[0].dtype
        for angle in tensors[1:]:
            dtype = torch.promote_types(dtype, angle.dtype)
        return torch.tensor(results, dtype=dtype, device=tensors[0].device)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        from ..gradients.parameter_shift import batched_parameter_shift

        indices = [
            i for i in range(len(ctx.values)) if ctx.needs_input_grad[i + 4]
        ]
        width = ctx.angle_count
        rows = [
            ctx.values[i * width:(i + 1) * width]
            for i in range(ctx.batch_size)
        ]
        derivatives = batched_parameter_shift(ctx.device, ctx.tape, rows, indices)
        gradients = [None] * len(ctx.values)
        for index, derivative in zip(indices, derivatives):
            gradients[index] = grad_output[index // width] * derivative
        return (None, None, None, None, *gradients)


def execute_torch_batch(device: Device, tapes: list[QuantumTape]) -> torch.Tensor:
    """Evaluate identical circuit layouts using one estimator parameter sweep."""
    if not tapes:
        raise ValueError("Cannot execute an empty batch.")
    first = tapes[0]
    signature = tuple((op.name, op.wires) for op in first.operations)
    for tape in tapes[1:]:
        if (
            tape.wires != first.wires
            or tape.measurements != first.measurements
            or tuple((op.name, op.wires) for op in tape.operations) != signature
        ):
            raise ValueError("All samples in a batch must construct the same circuit layout.")

    parameter_sets = [_angles(tape) for tape in tapes]
    flat_angles = tuple(angle for row in parameter_sets for angle in row)
    tensor_angles = [angle for angle in flat_angles if isinstance(angle, torch.Tensor)]
    if not tensor_angles:
        results = device.execute_many(first, parameter_sets)
        return torch.tensor(results, dtype=torch.get_default_dtype())

    output_device = tensor_angles[0].device
    if any(angle.device != output_device for angle in tensor_angles):
        raise ValueError("All tensor rotation angles must be on the same device.")
    return _BatchedQuantumExpectation.apply(
        device, first, len(tapes), len(parameter_sets[0]), *flat_angles
    )
