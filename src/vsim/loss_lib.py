import re
from pathlib import Path

import numpy as np
import stim

_COORD_PATTERN = re.compile(r"QUBIT_COORDS\((\d+),\s*(\d+)\)\s+(\d+)")


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
        self.data.append((loss_index, qubit_index))

    def _finalize(self) -> None:
        self.data.sort()

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


def _parse_nominal_and_events(circuit_text: str) -> tuple[str, list[LossInstruction], list[tuple]]:
    """Build nominal stim text (no LOSS lines), loss instructions, and ordered syndrome events."""
    loss_instructions: list[LossInstruction] = []
    nominal_lines: list[str] = []
    syndrome_events: list[tuple] = []

    for raw in circuit_text.split("\n"):
        line = raw.strip()
        if line.startswith("LOSS"):
            idx = len(loss_instructions)
            loss_instructions.append(LossInstruction(line))
            syndrome_events.append(("loss", idx))
        else:
            nominal_lines.append(line)
            if line.startswith("R") or line.startswith("MR"):
                qubits = tuple(int(p) for p in line.split() if p.isdigit())
                syndrome_events.append(("clear", qubits))

    final_circuit = "\n".join(nominal_lines)
    return final_circuit, loss_instructions, syndrome_events


class LossyCircuit:
    def __init__(self, circuit_path: Path | str):
        self.circuit_path = Path(circuit_path)
        self.circuit = self.circuit_path.read_text(encoding="utf-8")
        self._build_structure()

    @classmethod
    def from_text(cls, text: str) -> "LossyCircuit":
        inst = object.__new__(cls)
        inst.circuit_path = None
        inst.circuit = text
        inst._build_structure()
        return inst

    def _build_structure(self) -> None:
        final_circuit, self.loss_instructions, self.syndrome_events = _parse_nominal_and_events(
            self.circuit
        )
        self.qubits_coords: dict[int, tuple[int, int]] = {}
        self.coords_qubits: dict[tuple[int, int], int] = {}

        for line in final_circuit.split("\n"):
            line = line.strip()
            coord_match = _COORD_PATTERN.match(line)
            if coord_match:
                x, y, qubit_index = (
                    int(coord_match.group(1)),
                    int(coord_match.group(2)),
                    int(coord_match.group(3)),
                )
                self.qubits_coords[qubit_index] = (x, y)
                self.coords_qubits[(x, y)] = qubit_index

        self.nominal_circuit = stim.Circuit(final_circuit)

        n_targets = sum(len(inst.targets) for inst in self.loss_instructions)
        self._target_offsets = np.zeros(len(self.loss_instructions) + 1, dtype=np.int64)
        for i, inst in enumerate(self.loss_instructions):
            self._target_offsets[i + 1] = self._target_offsets[i] + len(inst.targets)

        self._dice_buf = np.empty(max(n_targets, 1), dtype=np.float64)

        self.qubits_missing = np.zeros(self.nominal_circuit.num_qubits, dtype=np.uint8)

    def __str__(self):
        return self.circuit

    def __repr__(self):
        return self.circuit

    def syndrome(self, rng: np.random.Generator) -> tuple[LossSyndrome, int]:
        loss_syndrome = LossSyndrome(rot=None)
        missed_loss = 0
        self.qubits_missing.fill(0)

        n_targets = int(self._target_offsets[-1])
        if n_targets:
            self._dice_buf[:n_targets] = rng.random(n_targets, dtype=np.float64)

        for ev in self.syndrome_events:
            if ev[0] == "loss":
                loss_index = ev[1]
                instruction = self.loss_instructions[loss_index]
                lo = int(self._target_offsets[loss_index])
                hi = int(self._target_offsets[loss_index + 1])
                dice = self._dice_buf[lo:hi]
                p = instruction.p
                for i, target in enumerate(instruction.targets):
                    if dice[i] < p:
                        q = target.value
                        if not self.qubits_missing[q]:
                            loss_syndrome.add(loss_index, q)
                            self.qubits_missing[q] = True
                            missed_loss += 1
            else:
                _, qubits = ev
                for qubit in qubits:
                    self.qubits_missing[qubit] = False

        loss_syndrome._finalize()
        return loss_syndrome, missed_loss


def add_noise(circuit: stim.Circuit, p_2q: float, p_reset: float) -> LossyCircuit:
    circuit = circuit.flattened()
    parts: list[str] = []
    for instruction in circuit:
        parts.append(str(instruction))
        name = instruction.name
        targets = instruction.targets_copy()
        if name == "CX":
            parts.append(str(LossInstruction.new(p_2q, targets)))
        elif name == "R":
            parts.append(str(LossInstruction.new(p_reset, targets)))
    return LossyCircuit.from_text("\n".join(parts))


class SymmetricalLossyCircuit(LossyCircuit):
    def __init__(self, circuit_path: Path | str):
        super().__init__(circuit_path)
        self._attach_symmetry()

    def _attach_symmetry(self) -> None:
        self.rot = None
        self._mid_width = max(x for x, y in self.qubits_coords.values()) // 2
        n = self.nominal_circuit.num_qubits
        self._rot_map = np.zeros((n, 4), dtype=np.int32)
        for q, (x, y) in self.qubits_coords.items():
            xr, yr = x, y
            for rot in range(4):
                xa, ya = xr, yr
                for _ in range(rot):
                    xa, ya = ya, self._mid_width - (xa - self._mid_width)
                self._rot_map[q, rot] = self.coords_qubits[(xa, ya)]

    @classmethod
    def from_text(cls, text: str) -> "SymmetricalLossyCircuit":
        inst = object.__new__(cls)
        inst.circuit_path = None
        inst.circuit = text
        inst._build_structure()
        inst._attach_symmetry()
        return inst

    def get_quadrant(self, qubit_index: int) -> int:
        x, y = self.qubits_coords[qubit_index]
        mid_width = self._mid_width

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
            raise ValueError(
                f"Invalid coordinates for qubit {qubit_index}: ({x}, {y})"
            )  # pragma: no cover

    def rotate_qubit(self, qubit_index: int, rot: int | None) -> int:
        if rot is None:
            raise ValueError("Rotation is not defined yet")
        return int(self._rot_map[qubit_index, rot % 4])

    def syndrome(self, rng: np.random.Generator) -> tuple[LossSyndrome, LossSyndrome, int]:
        loss_syndrome_symmetrical = LossSyndrome(rot=None)
        loss_syndrome = LossSyndrome(rot=None)
        missed_loss = 0
        self.qubits_missing.fill(0)

        n_targets = int(self._target_offsets[-1])
        if n_targets:
            self._dice_buf[:n_targets] = rng.random(n_targets, dtype=np.float64)

        for ev in self.syndrome_events:
            if ev[0] == "loss":
                loss_index = ev[1]
                instruction = self.loss_instructions[loss_index]
                lo = int(self._target_offsets[loss_index])
                hi = int(self._target_offsets[loss_index + 1])
                dice = self._dice_buf[lo:hi]
                p = instruction.p
                draw = dice < p
                if np.any(draw):
                    if loss_syndrome_symmetrical.rot is None:
                        i = int(np.argmax(draw))
                        qubit_index = instruction.targets[i].value
                        loss_syndrome_symmetrical.rot = self.get_quadrant(qubit_index)

                    r = loss_syndrome_symmetrical.rot % 4

                    for i, target in enumerate(instruction.targets):
                        if not draw[i]:
                            continue
                        qubit_index = target.value
                        qubit_index_symmetrical = int(self._rot_map[qubit_index, r])

                        if not self.qubits_missing[qubit_index]:
                            self.qubits_missing[qubit_index] = True
                            loss_syndrome.add(loss_index, qubit_index)
                            loss_syndrome_symmetrical.add(loss_index, qubit_index_symmetrical)
                        else:
                            missed_loss += 1
            else:
                _, qubits = ev
                for qubit in qubits:
                    self.qubits_missing[qubit] = False

        loss_syndrome._finalize()
        loss_syndrome_symmetrical._finalize()
        return loss_syndrome, loss_syndrome_symmetrical, missed_loss
