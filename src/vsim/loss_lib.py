from collections.abc import Sequence
from pathlib import Path

import numpy as np
import stim

from .stim_parse import parse_stim_line

_1Q_GATES = {
    "I",
    "X",
    "Y",
    "Z",
    "C_XYZ",
    "C_ZYX",
    "H",
    "H_XY",
    "H_XZ",
    "H_YZ",
    "S",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "SQRT_Z",
    "SQRT_Z_DAG",
    "S_DAG",
}

_1Q_ERROR_GATES = {
    "HERALDED_ERASE",
    "HERALDED_PAULI_CHANNEL_1",
    "PAULI_CHANNEL_1",
    "DEPOLARIZE1",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
}

_2Q_GATES = {
    "CNOT",
    "CX",
    "CXSWAP",
    "CY",
    "CZ",
    "CZSWAP",
    "ISWAP",
    "ISWAP_DAG",
    "SQRT_XX",
    "SQRT_XX_DAG",
    "SQRT_YY",
    "SQRT_YY_DAG",
    "SQRT_ZZ",
    "SQRT_ZZ_DAG",
    "SWAP",
    "SWAPCX",
    "SWAPCZ",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "ZCX",
    "ZCY",
    "ZCZ",
}
_2Q_ERROR_GATES = {
    "DEPOLARIZE2",
    "PAULI_CHANNEL_2",
}

_RESET_GATES = {
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "R",
    "RX",
    "RY",
    "RZ",
}
_MEASURE_GATES = {
    "M",
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "MX",
    "MY",
    "MZ",
}


def apply_loss_to_measurement_record(
    measurements: np.ndarray,
    loss_measurement_record: Sequence[int],
) -> np.ndarray:
    """Set measurement slots for heralded loss to ``2`` (same convention as :meth:`LossyCircuit.run`)."""
    out = np.asarray(measurements, dtype=np.uint8).copy()
    idx = list(loss_measurement_record)
    if not idx:
        return out
    if out.ndim == 1:
        out[idx] = 2
    elif out.ndim == 2:
        out[:, idx] = 2
    else:
        raise ValueError("measurements must be 1d or 2d")
    return out


class LossInstruction:
    def __init__(self, line: str):
        line = line.strip()
        self.line = line
        self.p = float(line.split("(")[1].split(")")[0])
        self.targets = [stim.GateTarget(int(target)) for target in line.split(" ")[1:]]
        self.name = "LOSS"

    def __str__(self):
        return self.line

    def __repr__(self):
        return self.line

    def targets_copy(self) -> list[stim.GateTarget]:
        return [stim.GateTarget(t.value) for t in self.targets]

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

    def has_loss_index(self, loss_index: int) -> bool:
        return any(li == loss_index for li, _ in self.data)

    def get_qubits_for_loss_index(self, loss_index: int) -> list[int]:
        return [qi for li, qi in self.data if li == loss_index]

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        res = ""
        for loss_index, qubit_index in self.data:
            res += f"#{loss_index},{qubit_index}"
        return res


def _parse_events(
    circuit: stim.Circuit,
) -> tuple[list[LossInstruction], list[tuple]]:
    """Build loss instructions, and ordered syndrome events."""
    loss_instructions: list[LossInstruction] = []
    syndrome_events: list[tuple] = []

    for instruction in circuit:
        if instruction.name == "LOSS":
            idx = len(loss_instructions)
            loss_instructions.append(LossInstruction(str(instruction)))
            syndrome_events.append(("loss", idx))
        else:
            if instruction.name in _RESET_GATES:
                qubits = tuple(
                    int(target.value) for target in instruction.targets_copy()
                )
                syndrome_events.append(("clear", qubits))

    return loss_instructions, syndrome_events


class LossyCircuit:
    def __init__(self, circuit_path: Path | str):
        path = Path(circuit_path)
        self.circuit_path = path
        self.circuit = LossyCircuit.parse_text(path.read_text(encoding="utf-8"))
        self._build_structure()

    @staticmethod
    def parse_text(text: str) -> list[stim.CircuitInstruction | LossInstruction]:
        instructions: list[stim.CircuitInstruction | LossInstruction] = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("LOSS"):
                instructions.append(LossInstruction(line))
            else:
                inst = parse_stim_line(line)
                if inst is not None:
                    instructions.append(inst)
        return instructions

    @classmethod
    def from_text(cls, text: str) -> "LossyCircuit":
        inst = object.__new__(cls)
        inst.circuit_path = None
        inst.circuit = LossyCircuit.parse_text(text)
        inst._build_structure()
        return inst

    def _build_structure(self) -> None:
        self.loss_instructions, self.syndrome_events = _parse_events(self.circuit)
        self.qubits_coords: dict[int, tuple[int, int]] = {}
        self.coords_qubits: dict[tuple[int, int], int] = {}
        self.nominal_circuit = stim.Circuit()

        for instruction in self.circuit:
            if instruction.name == "QUBIT_COORDS":
                qubit_index = int(instruction.targets_copy()[0].value)
                args = instruction.gate_args_copy()
                x = int(args[0])
                y = int(args[1])
                self.qubits_coords[qubit_index] = (x, y)
                self.coords_qubits[(x, y)] = qubit_index
            if not isinstance(instruction, LossInstruction):
                self.nominal_circuit.append(instruction)

        n_targets = sum(len(inst.targets) for inst in self.loss_instructions)
        self._target_offsets = np.zeros(len(self.loss_instructions) + 1, dtype=np.int64)
        for i, inst in enumerate(self.loss_instructions):
            self._target_offsets[i + 1] = self._target_offsets[i] + len(inst.targets)

        self._dice_buf = np.empty(max(n_targets, 1), dtype=np.float64)

        self.qubits_missing = np.zeros(self.nominal_circuit.num_qubits, dtype=np.uint8)

    def pretty_print(self, spaces: int = 0) -> str:
        return "\n".join(" " * spaces + str(inst) for inst in self.circuit)

    def __str__(self):
        return self.pretty_print()

    def __repr__(self):
        return f"LossyCircuit('''\n{self.pretty_print(2)}\n''')"

    def syndrome(self, rng: np.random.Generator) -> tuple[LossSyndrome, int]:
        loss_syndrome = LossSyndrome(rot=None)
        # missed_loss = 0
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
                            # missed_loss += 1
            else:
                _, qubits = ev
                for qubit in qubits:
                    self.qubits_missing[qubit] = False

        loss_syndrome._finalize()
        return loss_syndrome

    def rewrite_circuit_from_syndrome(
        self, loss_syndrome: LossSyndrome
    ) -> tuple[stim.Circuit, list[int]]:
        loss_index = 0
        self.qubits_missing.fill(0)
        final_circuit = stim.Circuit()

        measurement_index = 0
        loss_measurement_record = []

        for instruction in self.circuit:
            if isinstance(instruction, LossInstruction):
                if loss_syndrome.has_loss_index(loss_index):
                    qubits_to_remove = loss_syndrome.get_qubits_for_loss_index(
                        loss_index
                    )
                    for q in qubits_to_remove:
                        self.qubits_missing[q] = True
                loss_index += 1
            else:
                if instruction.name in _1Q_GATES.union(_1Q_ERROR_GATES):
                    targets = [
                        t
                        for t in instruction.targets_copy()
                        if not self.qubits_missing[t.value]
                    ]
                    if targets:
                        final_circuit.append(
                            stim.CircuitInstruction(
                                instruction.name, targets, instruction.gate_args_copy()
                            )
                        )
                elif instruction.name in _2Q_GATES.union(_2Q_ERROR_GATES):
                    instuction_targets = instruction.targets_copy()
                    targets = []

                    for i in range(0, len(instuction_targets) - 1, 2):
                        t1 = instuction_targets[i]
                        t2 = instuction_targets[i + 1]
                        if (
                            not self.qubits_missing[t1.value]
                            and not self.qubits_missing[t2.value]
                        ):
                            targets.extend([t1, t2])

                    if targets:
                        final_circuit.append(
                            stim.CircuitInstruction(
                                instruction.name,
                                targets,
                                instruction.gate_args_copy(),
                            )
                        )
                else:
                    if instruction.name in _MEASURE_GATES:
                        missing_targets = []

                        for t in instruction.targets_copy():
                            if self.qubits_missing[t.value]:
                                loss_measurement_record.append(measurement_index)
                                missing_targets.append(t)
                            measurement_index += 1

                        if missing_targets:
                            final_circuit.append(
                                stim.CircuitInstruction("R", missing_targets)
                            )
                            final_circuit.append(
                                stim.CircuitInstruction("x", missing_targets)
                            )

                        final_circuit.append(instruction)

                    if instruction.name in _RESET_GATES:
                        for t in instruction.targets_copy():
                            self.qubits_missing[t.value] = False
                        if instruction.name not in _MEASURE_GATES:
                            final_circuit.append(instruction)

                    if instruction.name not in _MEASURE_GATES.union(_RESET_GATES):
                        final_circuit.append(instruction)

        return final_circuit, loss_measurement_record

    def run(self, seed: int | None = None) -> tuple[LossSyndrome, np.ndarray]:
        tableau = stim.TableauSimulator(seed=seed)
        rng = np.random.default_rng(seed)
        loss_syndrome = LossSyndrome(rot=None)
        self.qubits_missing.fill(0)

        loss_index = 0

        n_targets = int(self._target_offsets[-1])
        if n_targets:
            self._dice_buf[:n_targets] = rng.random(n_targets, dtype=np.float64)

        measurement_index = 0
        loss_measurement_record = []

        for instruction in self.circuit:
            if isinstance(instruction, LossInstruction):
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
                loss_index += 1
            else:
                if instruction.name in _1Q_GATES.union(_1Q_ERROR_GATES):
                    targets = [
                        t
                        for t in instruction.targets_copy()
                        if not self.qubits_missing[t.value]
                    ]
                    if targets:
                        tableau.do(
                            stim.CircuitInstruction(
                                instruction.name, targets, instruction.gate_args_copy()
                            )
                        )
                elif instruction.name in _2Q_GATES.union(_2Q_ERROR_GATES):
                    instuction_targets = instruction.targets_copy()
                    targets = []

                    for i in range(0, len(instuction_targets) - 1, 2):
                        t1 = instuction_targets[i]
                        t2 = instuction_targets[i + 1]
                        if (
                            not self.qubits_missing[t1.value]
                            and not self.qubits_missing[t2.value]
                        ):
                            targets.extend([t1, t2])

                    if targets:
                        tableau.do(
                            stim.CircuitInstruction(
                                instruction.name,
                                targets,
                                instruction.gate_args_copy(),
                            )
                        )
                else:
                    if instruction.name in _MEASURE_GATES:
                        targets = instruction.targets_copy()

                        for t in targets:
                            if self.qubits_missing[t.value]:
                                tableau.reset(t.value)
                                loss_measurement_record.append(measurement_index)
                            measurement_index += 1

                        tableau.do(instruction)

                    if instruction.name in _RESET_GATES:
                        for t in instruction.targets_copy():
                            self.qubits_missing[t.value] = False
                        if instruction.name not in _MEASURE_GATES:
                            tableau.do(instruction)

                    if instruction.name not in _MEASURE_GATES.union(_RESET_GATES):
                        tableau.do(instruction)

        result = np.array(tableau.current_measurement_record(), dtype=np.uint8)

        return loss_syndrome, apply_loss_to_measurement_record(
            result, loss_measurement_record
        )

    def simulate_rewrite_measurement_record(
        self,
        loss_syndrome: LossSyndrome,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """Tableau simulation of the rewritten circuit plus loss masking.

        Matches the measurement vector returned by :meth:`run` for the same
        ``loss_syndrome``, tableau ``seed``, and implicitly the same loss dice (caller
        must obtain ``loss_syndrome`` from the desired scenario — e.g. ``run(seed)[0]``).
        """
        rew, loss_idxs = self.rewrite_circuit_from_syndrome(loss_syndrome)
        tableau = stim.TableauSimulator(seed=seed)
        tableau.do(rew.flattened())
        raw = np.array(tableau.current_measurement_record(), dtype=np.uint8)
        return apply_loss_to_measurement_record(raw, loss_idxs)


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
        inst.circuit = LossyCircuit.parse_text(text)
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

    def syndrome(
        self, rng: np.random.Generator
    ) -> tuple[LossSyndrome, LossSyndrome, int]:
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
                            loss_syndrome_symmetrical.add(
                                loss_index, qubit_index_symmetrical
                            )
                        else:
                            missed_loss += 1
            else:
                _, qubits = ev
                for qubit in qubits:
                    self.qubits_missing[qubit] = False

        loss_syndrome._finalize()
        loss_syndrome_symmetrical._finalize()
        return loss_syndrome, loss_syndrome_symmetrical, missed_loss
