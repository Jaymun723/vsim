"""Equivalence of ``run`` vs rewritten tableau + loss measurement mask."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from vsim.loss_lib import add_noise, apply_loss_to_measurement_record


@pytest.fixture
def surface_lossy_d3():
    return add_noise(
        stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
            before_measure_flip_probability=0.01,
            after_reset_flip_probability=0.01,
        ),
        0.01,
        0.01,
    )


def test_simulate_rewrite_measurement_matches_run(surface_lossy_d3):
    lc = surface_lossy_d3
    for seed in range(128):
        synd, meas_run = lc.run(seed)
        meas_rw = lc.simulate_rewrite_measurement_record(synd, seed=seed)
        assert np.array_equal(meas_run, meas_rw)


def test_apply_loss_mask_batch():
    raw = np.zeros((4, 5), dtype=np.uint8)
    marked = apply_loss_to_measurement_record(raw, [1, 4])
    assert marked[0, 1] == 2 and marked[0, 4] == 2
    assert marked[2, 1] == 2
