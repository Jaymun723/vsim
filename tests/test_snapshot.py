"""Golden histogram vs legacy snapshot (devtools/regenerate_loss_snapshot.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import stim

from vsim.loss_lib import SymmetricalLossyCircuit, add_noise


def _sym_key(s):
    import json as _json

    return _json.dumps({"s": str(s), "rot": s.rot}, sort_keys=True)


def _replay_from_meta(meta: dict):
    stim_surface = stim.Circuit.generated(
        meta["generator"],
        distance=int(meta["distance"]),
        rounds=int(meta["rounds"]),
        after_clifford_depolarization=float(meta["p_depol"]),
        before_measure_flip_probability=float(meta["p_spam"]),
        after_reset_flip_probability=float(meta["p_spam"]),
    )
    lossy = add_noise(
        stim_surface,
        float(meta["p_loss_2q"]),
        float(meta["p_loss_reset"]),
    )
    sym = SymmetricalLossyCircuit.from_text(str(lossy))
    rng = np.random.default_rng(seed=int(meta["seed"]))
    shots = int(meta["shots"])
    phys: dict[str, int] = {}
    sym_h: dict[str, int] = {}
    missed: dict[int, int] = {}
    for _ in range(shots):
        phy, sym_s, m = sym.syndrome(rng)
        sk = str(phy)
        phys[sk] = phys.get(sk, 0) + 1
        kk = _sym_key(sym_s)
        sym_h[kk] = sym_h.get(kk, 0) + 1
        missed[m] = missed.get(m, 0) + 1
    missed_out = {str(k): v for k, v in missed.items()}
    return phys, sym_h, missed_out


def test_snapshot_matches_legacy_histogram():
    snap_path = Path(__file__).resolve().parent / "snapshots" / "loss_histogram_d3_seed7.json"
    raw = json.loads(snap_path.read_text(encoding="utf-8"))
    meta = raw["meta"]
    phys, sym_h, missed = _replay_from_meta(meta)
    assert phys == raw["phys_histogram"]
    assert sym_h == raw["sym_histogram"]
    assert missed == raw["missed_loss_histogram"]
