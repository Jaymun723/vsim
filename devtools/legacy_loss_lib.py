"""Legacy reference implementation (pre–loss_lib_fast). Used only by regenerate_loss_snapshot."""

from __future__ import annotations

import bisect
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import stim


class LossInstruction:
    def __init__(self, line: str):
        line = line.strip()
        self.line = line
        self.p = float(line.split("(")[1].split(")")[0])
        self.targets = [stim.GateTarget(int(target)) for target in line.split(" ")[1:]]

    def __str__(self):
        return self.line

    def __repr__(self):
        return self.line

    @staticmethod
    def new(p: float, targets: list[stim.GateTarget | int]):
        line = f"LOSS({p}) "
        for target in targets:
            if isinstance(target, stim.GateTarget):
                line += f"{target.value} "
            else:
                line += f"{target} "
        return LossInstruction(line.strip())


class LossSyndrome:
    compare_rot = False

    def __init__(self, rot: int | None = None):
        self.rot = rot
        self.data: list[tuple[int, int]] = []

    def add(self, loss_index: int, qubit_index: int):
        bisect.insort(self.data, (loss_index, qubit_index))

    def __repr__(self) -> str:
        return f"LossSyndrome(data={self.data}, rot={self.rot})"

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other) -> bool:
        if not isinstance(other, LossSyndrome):
            return NotImplemented

        rot = self.rot == other.rot if self.compare_rot else True

        return str(self) == str(other) and rot

    def __hash__(self) -> int:
        return hash(tuple(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        res = ""
        for loss_index, qubit_index in self.data:
            res += f"#{loss_index},{qubit_index}"
        return res


class LossyCircuit:
    def __init__(self, circuit_path: Path):
        self.circuit_path = circuit_path
        with open(circuit_path, encoding="utf-8") as f:
            self.circuit = f.read()
        self._get_infos()
        self.qubits_missing = np.zeros(self.nominal_circuit.num_qubits, dtype=np.uint8)

    def __str__(self):
        return self.circuit

    def __repr__(self):
        return self.circuit

    def _get_infos(self) -> stim.Circuit:
        final_circuit = ""
        self.loss_instructions = []
        self.qubits_coords = {}
        self.coords_qubits = {}

        coord_pattern = re.compile(r"QUBIT_COORDS\((\d+),\s*(\d+)\)\s+(\d+)")

        for line in self.circuit.split("\n"):
            line = line.strip()
            if line.startswith("LOSS"):
                self.loss_instructions.append(LossInstruction(line))
            else:
                final_circuit += line + "\n"

                coord_match = coord_pattern.match(line)
                if coord_match:
                    x, y, qubit_index = (
                        int(coord_match.group(1)),
                        int(coord_match.group(2)),
                        int(coord_match.group(3)),
                    )
                    self.qubits_coords[qubit_index] = (x, y)
                    self.coords_qubits[(x, y)] = qubit_index

        self.nominal_circuit = stim.Circuit(final_circuit)

    def syndrome(self, rng: np.random.Generator) -> tuple[LossSyndrome, int]:
        loss_index = 0
        loss_syndrome = LossSyndrome(rot=None)
        dices = [rng.uniform(size=len(inst.targets)) for inst in self.loss_instructions]
        missed_loss = 0
        self.qubits_missing.fill(0)

        for line in self.circuit.split("\n"):
            if line.startswith("LOSS"):
                instruction = self.loss_instructions[loss_index]
                for i, target in enumerate(instruction.targets):
                    if dices[loss_index][i] < instruction.p:
                        if not self.qubits_missing[target.value]:
                            loss_syndrome.add(loss_index, target.value)
                            self.qubits_missing[target.value] = True
                            missed_loss += 1
                loss_index += 1
            else:
                if line.startswith("R") or line.startswith("MR"):
                    for part in line.split(" "):
                        if part.isdigit():
                            qubit = int(part)
                            self.qubits_missing[qubit] = False

        return loss_syndrome, missed_loss


def add_noise(circuit: stim.Circuit, p_2q: float, p_reset: float):
    circuit = circuit.flattened()
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "circuit.stim"
        with open(path, "w", encoding="utf-8") as f:
            for instruction in circuit:
                f.write(str(instruction))
                f.write("\n")

                name = instruction.name
                targets = instruction.targets_copy()
                if name == "CX":
                    loss_instruction = LossInstruction.new(p_2q, targets)
                    f.write(str(loss_instruction))
                    f.write("\n")
                elif name == "R":
                    loss_instruction = LossInstruction.new(p_reset, targets)
                    f.write(str(loss_instruction))
                    f.write("\n")

        return LossyCircuit(path)


class SymmetricalLossyCircuit(LossyCircuit):
    def __init__(self, circuit_path: Path):
        super().__init__(circuit_path)
        self.rot = None

    def get_quadrant(self, qubit_index: int) -> int:
        mid_width = max(x for x, y in self.qubits_coords.values()) // 2
        x, y = self.qubits_coords[qubit_index]

        if x == mid_width and y == mid_width:
            return 0

        if x <= mid_width and y < mid_width:
            return 0
        elif x > mid_width and y <= mid_width:
            return 1
        elif x >= mid_width and y > mid_width:
            return 2
        elif x < mid_width and y >= mid_width:
            return 3
        else:
            raise ValueError(f"Invalid coordinates for qubit {qubit_index}: ({x}, {y})")

    def rotate_qubit(self, qubit_index: int, rot: int | None) -> int:
        if rot is None:
            raise ValueError("Rotation is not defined yet")

        x, y = self.qubits_coords[qubit_index]
        mid_width = max(x for x, y in self.qubits_coords.values()) // 2

        for _ in range(rot % 4):
            x, y = y, mid_width - (x - mid_width)

        return self.coords_qubits[(x, y)]

    def syndrome(self, rng: np.random.Generator) -> tuple[LossSyndrome, LossSyndrome, int]:
        loss_index = 0
        loss_syndrome_symmetrical = LossSyndrome(rot=None)
        loss_syndrome = LossSyndrome(rot=None)
        dices = [rng.uniform(size=len(inst.targets)) for inst in self.loss_instructions]
        missed_loss = 0
        self.qubits_missing.fill(0)

        for line in self.circuit.split("\n"):
            if line.startswith("LOSS"):
                instruction = self.loss_instructions[loss_index]
                draw = dices[loss_index] < instruction.p
                if draw.any():
                    if loss_syndrome_symmetrical.rot is None:
                        i = np.argmax(draw)
                        qubit_index = instruction.targets[i].value
                        loss_syndrome_symmetrical.rot = self.get_quadrant(qubit_index)

                    for i, target in enumerate(instruction.targets):
                        qubit_index_symmetrical = self.rotate_qubit(
                            target.value, loss_syndrome_symmetrical.rot
                        )
                        qubit_index = target.value

                        if draw[i]:
                            if not self.qubits_missing[qubit_index]:
                                self.qubits_missing[qubit_index] = True

                                loss_syndrome.add(loss_index, qubit_index)
                                loss_syndrome_symmetrical.add(
                                    loss_index, qubit_index_symmetrical
                                )
                            else:
                                missed_loss += 1

                loss_index += 1
            else:
                if line.startswith("R") or line.startswith("MR"):
                    for part in line.split(" "):
                        if part.isdigit():
                            qubit = int(part)
                            self.qubits_missing[qubit] = False

        return loss_syndrome, loss_syndrome_symmetrical, missed_loss
