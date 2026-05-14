"""
Regenerate tests/snapshots/loss_histogram_d3_seed7.json using legacy_loss_lib.

Matches test.ipynb get_dict_loss_syndrome parameters (distance=3 default run).
Run from repo root:

  uv run --group dev python devtools/regenerate_loss_snapshot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import stim

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from devtools import legacy_loss_lib as legacy


def sym_key(s: legacy.LossSyndrome) -> str:
    return json.dumps({"s": str(s), "rot": s.rot}, sort_keys=True)


def build_histograms(
    distance: int,
    rounds: int,
    shots: int,
    seed: int,
    *,
    p_loss_2q: float = 0.01,
    p_loss_reset: float = 0.01,
    p_depol: float = 0.01,
    p_spam: float = 0.01,
) -> dict:
    stim_surface = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_depol,
        before_measure_flip_probability=p_spam,
        after_reset_flip_probability=p_spam,
    )
    lossy = legacy.add_noise(stim_surface, p_loss_2q, p_loss_reset)
    tmp_path = _REPO_ROOT / ".snapshot_tmp_lossy.stim"
    tmp_path.write_text(str(lossy), encoding="utf-8")
    try:
        sym_circuit = legacy.SymmetricalLossyCircuit(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    dict_syndrome: dict[str, int] = {}
    dict_syndrome_symmetrical: dict[str, int] = {}
    dict_missed_loss: dict[int, int] = {}
    rng = np.random.default_rng(seed=seed)
    for _ in range(shots):
        phy, sym_s, missed = sym_circuit.syndrome(rng)
        sk = str(phy)
        dict_syndrome[sk] = dict_syndrome.get(sk, 0) + 1
        sym_sk = sym_key(sym_s)
        dict_syndrome_symmetrical[sym_sk] = dict_syndrome_symmetrical.get(sym_sk, 0) + 1
        dict_missed_loss[missed] = dict_missed_loss.get(missed, 0) + 1

    return {
        "meta": {
            "generator": "surface_code:rotated_memory_z",
            "distance": distance,
            "rounds": rounds,
            "shots": shots,
            "seed": seed,
            "p_loss_2q": p_loss_2q,
            "p_loss_reset": p_loss_reset,
            "p_depol": p_depol,
            "p_spam": p_spam,
        },
        "phys_histogram": dict_syndrome,
        "sym_histogram": dict_syndrome_symmetrical,
        "missed_loss_histogram": {str(k): v for k, v in dict_missed_loss.items()},
    }


def main() -> None:
    # Matches notebook get_dict_loss_syndrome(d=3); shots reduced vs 100k for repo size/CI.
    data = build_histograms(distance=3, rounds=3, shots=2500, seed=7)
    out = _REPO_ROOT / "tests" / "snapshots" / "loss_histogram_d3_seed7.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
