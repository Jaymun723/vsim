"""Unit tests for vsim.loss_lib (100% line coverage)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import stim

import vsim
from vsim import loss_lib as ll
from tests.stim_fixtures import SURFACE_LOSS_STIM
from vsim.loss_lib import (
    LossInstruction,
    LossSyndrome,
    LossyCircuit,
    SymmetricalLossyCircuit,
    add_noise,
)


def test_version_exported():
    assert vsim.__version__ == "0.1.0"


def test_loss_instruction_new_gate_target_vs_int():
    t = [stim.GateTarget(3), 5]
    inst = LossInstruction.new(0.1, t)
    assert "3" in str(inst) and "5" in str(inst)


def test_repr_loss_instruction_and_syndrome():
    li = LossInstruction("LOSS(0.2) 3")
    assert repr(li) == str(li)
    s = LossSyndrome(rot=2)
    assert "rot=2" in repr(s)


def test_loss_syndrome_eq_hash_iter_getitem():
    a = LossSyndrome()
    a.add(1, 2)
    a.add(0, 1)
    a._finalize()
    b = LossSyndrome()
    b.data = [(0, 1), (1, 2)]
    assert a == b
    assert hash(a) == hash(b)
    assert len(a) == 2
    assert list(iter(a)) == a.data
    assert a[0] == (0, 1)


def test_loss_syndrome_eq_not_implemented():
    assert LossSyndrome().__eq__(object()) is NotImplemented


def test_loss_syndrome_compare_rot():
    x = LossSyndrome(rot=1)
    x.data = [(0, 0)]
    y = LossSyndrome(rot=2)
    y.data = [(0, 0)]
    LossSyndrome.compare_rot = True
    try:
        assert x != y
        z = LossSyndrome(rot=1)
        z.data = [(0, 0)]
        assert x == z
    finally:
        LossSyndrome.compare_rot = False


def test_symmetrical_rotate_qubit_none_raises():
    c = SymmetricalLossyCircuit.from_text(SURFACE_LOSS_STIM)
    with pytest.raises(ValueError, match="Rotation"):
        c.rotate_qubit(0, None)


def test_rotate_qubit_returns_int():
    c = SymmetricalLossyCircuit.from_text(SURFACE_LOSS_STIM)
    q = c.rotate_qubit(0, 1)
    assert isinstance(q, int)


def test_lossy_circuit_load_from_path(tmp_path):
    p = tmp_path / "c.stim"
    p.write_text(SURFACE_LOSS_STIM, encoding="utf-8")
    c = LossyCircuit(p)
    assert c.circuit_path == p


def test_symmetrical_load_from_path(tmp_path):
    p = tmp_path / "c.stim"
    p.write_text(SURFACE_LOSS_STIM, encoding="utf-8")
    c = SymmetricalLossyCircuit(p)
    assert c.circuit_path == p


def test_lossy_circuit_repr_matches_str():
    c = LossyCircuit.from_text(SURFACE_LOSS_STIM)
    assert repr(c) == str(c)


def test_lossy_circuit_from_text_surface_fixture():
    c = LossyCircuit.from_text(SURFACE_LOSS_STIM)
    assert c.circuit_path is None
    rng = np.random.default_rng(0)
    s, m = c.syndrome(rng)
    assert isinstance(s, LossSyndrome)
    assert m >= 0


def test_symmetrical_from_text():
    c = SymmetricalLossyCircuit.from_text(SURFACE_LOSS_STIM)
    rng = np.random.default_rng(1)
    a, b, m = c.syndrome(rng)
    assert str(a)
    assert str(b)
    assert m >= 0


def test_add_noise_cx_and_r():
    c = stim.Circuit()
    c.append("R", [0])
    c.append("CX", [0, 1])
    lossy = add_noise(c, 0.02, 0.03)
    assert "LOSS(0.02)" in lossy.circuit
    assert "LOSS(0.03)" in lossy.circuit


def test_parse_nominal_and_events_helpers():
    stim_txt = (
        "QUBIT_COORDS(0, 0) 0\n"
        "R 0\n"
        "LOSS(0.1) 0\n"
        "MR 0\n"
    )
    nominal, losses, events = ll._parse_nominal_and_events(stim_txt)
    assert "LOSS" not in nominal
    assert len(losses) == 1
    assert events == [("clear", (0,)), ("loss", 0), ("clear", (0,))]


def test_syndrome_phys_no_loss_instructions(tmp_path):
    p = tmp_path / "t.stim"
    p.write_text("QUBIT_COORDS(0, 0) 0\nR 0\n", encoding="utf-8")
    c = LossyCircuit(p)
    s, m = c.syndrome(np.random.default_rng(0))
    assert str(s) == ""
    assert m == 0


def test_phys_syndrome_second_hit_same_qubit_skipped(tmp_path):
    p = tmp_path / "t.stim"
    p.write_text(
        "QUBIT_COORDS(0, 0) 0\nLOSS(1.0) 0\nLOSS(1.0) 0\n",
        encoding="utf-8",
    )
    c = LossyCircuit(p)
    rng = np.random.default_rng(0)
    s, m = c.syndrome(rng)
    assert m == 1


def test_symmetrical_syndrome_no_draw_branch():
    c = SymmetricalLossyCircuit.from_text(SURFACE_LOSS_STIM)
    rng = MagicMock()

    def ones_random(size=None, dtype=np.float64):
        return np.ones(size, dtype=dtype)

    rng.random = ones_random
    a, b, m = c.syndrome(rng)
    assert m == 0
    assert str(a) == ""
    assert str(b) == ""
    assert b.rot is None


def test_symmetrical_no_loss_instructions_skips_random_fill():
    stripped = "\n".join(L for L in SURFACE_LOSS_STIM.splitlines() if not L.strip().startswith("LOSS"))
    c = SymmetricalLossyCircuit.from_text(stripped)
    rng = np.random.default_rng(0)
    _, _, m = c.syndrome(rng)
    assert m == 0
