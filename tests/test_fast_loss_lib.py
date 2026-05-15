"""Tests for the native FastLossyCircuit C++ extension."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import stim

import vsim
from tests.stim_fixtures import SURFACE_LOSS_STIM
from vsim import FastLossyCircuit, LossyCircuit
from vsim.loss_lib import add_noise


pytestmark = pytest.mark.skipif(
    FastLossyCircuit is None, reason="FastLossyCircuit extension not built"
)


def _strip_loss(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("LOSS")
    )


def test_extension_is_loaded():
    # Imported from the C++ module, not the Python loss_lib.
    assert FastLossyCircuit.__module__.endswith("_fast_loss_lib")


def test_basic_shape(tmp_path: pathlib.Path):
    p = tmp_path / "c.stim"
    p.write_text(SURFACE_LOSS_STIM)
    fc = FastLossyCircuit(str(p))
    # 5x5 patch → 25 qubits.
    assert fc.num_qubits == 25
    result = fc.run(0)
    assert result.dtype == np.uint8
    # All measurement results must be 0, 1, or 2 (2 = heralded loss).
    assert set(int(x) for x in result.tolist()) <= {0, 1, 2}


def test_run_lengths_match_python(tmp_path: pathlib.Path):
    """Measurement record length must equal LossyCircuit's for the same circuit."""
    p = tmp_path / "c.stim"
    p.write_text(SURFACE_LOSS_STIM)
    fc = FastLossyCircuit(str(p))
    lc = LossyCircuit(p)
    for seed in range(8):
        _, meas_py = lc.run(seed)
        meas_fast = fc.run(seed)
        assert len(meas_fast) == len(meas_py)


def test_run_no_loss_matches_stim_directly():
    """With LOSS lines stripped, FastLossyCircuit must agree bitwise with stim."""
    text = _strip_loss(SURFACE_LOSS_STIM)
    fc = FastLossyCircuit.from_text(text)
    circuit = stim.Circuit(text)
    for seed in range(8):
        sim = stim.TableauSimulator(seed=seed)
        sim.do_circuit(circuit)
        expected = np.array(sim.current_measurement_record(), dtype=np.uint8)
        got = fc.run(seed)
        np.testing.assert_array_equal(got, expected)


def test_from_text_roundtrip_via_lossy_pretty_print():
    """Building from LossyCircuit.pretty_print() must succeed and produce the same length."""
    surface = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    lc = add_noise(surface, 0.01, 0.01)
    fc = FastLossyCircuit.from_text(lc.pretty_print())
    _, meas_py = lc.run(0)
    meas_fast = fc.run(0)
    assert len(meas_fast) == len(meas_py)


def test_seed_none_does_not_raise():
    """Calling run(seed=None) should pick a fresh RNG without throwing."""
    fc = FastLossyCircuit.from_text(SURFACE_LOSS_STIM)
    a = fc.run(None)
    b = fc.run(None)
    # Two independent draws shouldn't be required to differ, but at minimum
    # they must each return a valid measurement record.
    assert a.shape == b.shape


def test_seeds_are_deterministic():
    fc = FastLossyCircuit.from_text(SURFACE_LOSS_STIM)
    a = fc.run(123)
    b = fc.run(123)
    np.testing.assert_array_equal(a, b)


def test_loss_rate_within_expected_band():
    """Statistical sanity: at p_loss=0.01 on every CX/R, the heralded-loss
    rate per measurement should be well below 50%."""
    fc = FastLossyCircuit.from_text(SURFACE_LOSS_STIM)
    shots = 200
    counts = np.zeros(3, dtype=np.int64)
    for s in range(shots):
        r = fc.run(s)
        for v in (0, 1, 2):
            counts[v] += int((r == v).sum())
    total = counts.sum()
    # Heralded loss column should be a small minority for these p_loss values.
    assert counts[2] / total < 0.5


def test_version_still_exported():
    # Loading the C++ module must not interfere with the package metadata.
    assert isinstance(vsim.__version__, str)
    assert vsim.__version__ != "unknown"


def test_same_output_as_lossy_circuit():
    """For a fixed seed, FastLossyCircuit and LossyCircuit must produce the same measurement record."""
    fc = FastLossyCircuit.from_text(SURFACE_LOSS_STIM)
    lc = LossyCircuit.from_text(SURFACE_LOSS_STIM)
    for seed in range(8):
        _, meas_py = lc.run(seed)
        meas_fast = fc.run(seed)
        np.testing.assert_array_equal(meas_fast, meas_py)
