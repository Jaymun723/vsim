"""Tests for vsim.FastLossyCircuit (C++ extension)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import stim

import vsim
from vsim import FastLossyCircuit
from tests.stim_fixtures import SURFACE_LOSS_STIM
from vsim.loss_lib import add_noise


@pytest.fixture
def surface_stim_path(tmp_path: Path) -> Path:
    p = tmp_path / "surface.stim"
    p.write_text(SURFACE_LOSS_STIM, encoding="utf-8")
    return p


# ── Basic interface tests ─────────────────────────────────────────────────────


def test_fast_lossy_circuit_is_exported():
    assert hasattr(vsim, "FastLossyCircuit")
    assert FastLossyCircuit is vsim.FastLossyCircuit


def test_run_returns_numpy_uint8_array(surface_stim_path: Path):
    flc = FastLossyCircuit(str(surface_stim_path))
    result = flc.run(seed=0)
    arr = np.asarray(result)
    assert arr.ndim == 1
    assert arr.dtype == np.uint8


def test_run_length_is_constant(surface_stim_path: Path):
    """Measurement count must not change across seeds."""
    flc = FastLossyCircuit(str(surface_stim_path))
    lengths = {len(np.asarray(flc.run(seed=s))) for s in range(50)}
    assert len(lengths) == 1, f"Variable measurement counts: {lengths}"


def test_run_values_in_valid_set(surface_stim_path: Path):
    flc = FastLossyCircuit(str(surface_stim_path))
    for seed in range(20):
        arr = np.asarray(flc.run(seed=seed))
        assert set(arr.tolist()).issubset({0, 1, 2}), f"Unexpected values: {set(arr.tolist())}"


def test_run_accepts_none_seed(surface_stim_path: Path):
    flc = FastLossyCircuit(str(surface_stim_path))
    result = flc.run()  # seed=None
    arr = np.asarray(result)
    assert arr.ndim == 1
    assert arr.dtype == np.uint8


def test_run_accepts_integer_seed(surface_stim_path: Path):
    flc = FastLossyCircuit(str(surface_stim_path))
    result = flc.run(seed=42)
    arr = np.asarray(result)
    assert arr.ndim == 1
    assert arr.dtype == np.uint8


def test_run_deterministic_with_same_seed(surface_stim_path: Path):
    flc = FastLossyCircuit(str(surface_stim_path))
    r1 = np.asarray(flc.run(seed=7))
    r2 = np.asarray(flc.run(seed=7))
    np.testing.assert_array_equal(r1, r2)


def test_run_different_seeds_differ(surface_stim_path: Path):
    flc = FastLossyCircuit(str(surface_stim_path))
    results = [np.asarray(flc.run(seed=s)) for s in range(10)]
    # With low-loss probability and many seeds, results should not all be identical
    assert not all(np.array_equal(results[0], r) for r in results[1:])


# ── Loss-probability statistical tests ───────────────────────────────────────


def test_no_loss_no_twos(tmp_path: Path):
    """With LOSS(0.0), no measurement should be marked 2."""
    circuit = "QUBIT_COORDS(0, 0) 0\nR 0\nLOSS(0.0) 0\nM 0\n"
    p = tmp_path / "no_loss.stim"
    p.write_text(circuit, encoding="utf-8")
    flc = FastLossyCircuit(str(p))
    for seed in range(50):
        arr = np.asarray(flc.run(seed=seed))
        assert 2 not in arr.tolist(), f"Unexpected 2 with LOSS(0.0), seed={seed}"


def test_certain_loss_all_twos(tmp_path: Path):
    """With LOSS(1.0) on the measured qubit, that measurement must be 2."""
    circuit = "QUBIT_COORDS(0, 0) 0\nR 0\nLOSS(1.0) 0\nM 0\n"
    p = tmp_path / "full_loss.stim"
    p.write_text(circuit, encoding="utf-8")
    flc = FastLossyCircuit(str(p))
    for seed in range(20):
        arr = np.asarray(flc.run(seed=seed))
        assert arr[0] == 2, f"Expected 2 with LOSS(1.0), got {arr[0]} at seed={seed}"


def test_loss_rate_statistical(tmp_path: Path):
    """Fraction of loss-marked measurements should match the LOSS probability."""
    p_loss = 0.2
    circuit = f"R 0\nLOSS({p_loss}) 0\nM 0\n"
    path = tmp_path / "stat_loss.stim"
    path.write_text(circuit, encoding="utf-8")
    flc = FastLossyCircuit(str(path))

    n_shots = 2000
    n_twos = sum(
        int(np.asarray(flc.run(seed=s))[0] == 2) for s in range(n_shots)
    )
    observed = n_twos / n_shots
    # Allow ±5 percentage points (very loose — avoids flakiness)
    assert abs(observed - p_loss) < 0.05, f"observed={observed:.3f} expected≈{p_loss}"


# ── Multi-qubit / surface-code tests ─────────────────────────────────────────


def test_surface_circuit_loss_count_is_plausible(surface_stim_path: Path):
    """Over many shots, average number of 2s should be positive but small."""
    flc = FastLossyCircuit(str(surface_stim_path))
    n_shots = 200
    total_twos = sum(
        int((np.asarray(flc.run(seed=s)) == 2).sum()) for s in range(n_shots)
    )
    avg = total_twos / n_shots
    # SURFACE_LOSS_STIM has many LOSS(0.01) instructions so avg ~1-2 expected
    assert 0 < avg < 10, f"Unexpected average 2s per shot: {avg:.3f}"


def test_measure_reset_clears_loss_flag(tmp_path: Path):
    """After MR, the qubit should no longer be treated as lost in subsequent measurements."""
    # LOSS=1.0 on qubit 0, then MR (should mark first meas as 2, clear missing),
    # then M again (should NOT be 2 since qubit was reset by MR).
    circuit = (
        "QUBIT_COORDS(0, 0) 0\n"
        "R 0\n"
        "LOSS(1.0) 0\n"
        "MR 0\n"  # measurement[0] = 2 (lost), then reset
        "M 0\n"   # measurement[1] = 0 or 1 (fresh qubit, NOT 2)
    )
    p = tmp_path / "mr_clears.stim"
    p.write_text(circuit, encoding="utf-8")
    flc = FastLossyCircuit(str(p))
    for seed in range(20):
        arr = np.asarray(flc.run(seed=seed))
        assert len(arr) == 2, f"Expected 2 measurements, got {len(arr)}"
        assert arr[0] == 2, f"Expected first measurement = 2 (lost)"
        assert arr[1] != 2, f"Expected second measurement ≠ 2 after MR reset"


@pytest.mark.parametrize("distance", [3, 5, 7, 9])
def test_fast_lossy_runs_across_code_distances(tmp_path: Path, distance: int):
    """Fast path should execute without shape/value errors for common distances."""
    rounds = distance
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    lossy = add_noise(circuit, p_loss_2q=0.01, p_loss_reset=0.01)
    p = tmp_path / f"d{distance}.stim"
    p.write_text(str(lossy), encoding="utf-8")

    flc = FastLossyCircuit(str(p))
    lengths = set()
    for seed in range(10):
        arr = np.asarray(flc.run(seed=seed))
        lengths.add(arr.size)
        assert arr.dtype == np.uint8
        assert set(arr.tolist()).issubset({0, 1, 2})
    assert len(lengths) == 1


def test_error_on_missing_file():
    with pytest.raises(RuntimeError, match="cannot open"):
        FastLossyCircuit("/nonexistent/path/circuit.stim")
