import math

import miniqml as qml


def main():
    for name in ("qiskit.statevector", "qiskit.aer"):
        dev = qml.device(name, wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        theta = 0.4
        print(name, circuit(theta), "expected", math.cos(theta))


if __name__ == "__main__":
    main()
