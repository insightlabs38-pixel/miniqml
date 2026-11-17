import miniqml as qml


dev = qml.device("qiskit.aer", wires=2)


@qml.qnode(dev)
def circuit(x, weights):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.RX(weights[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))


tape = circuit.construct([0.2, 0.4], [0.8])
print("Wires:", tape.wires)
for operation in tape.operations:
    print(operation)
print("Measurement:", tape.measurement)
