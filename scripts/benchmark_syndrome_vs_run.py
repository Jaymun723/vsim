"""
Benchmark two ways to spend the same total number of Monte Carlo shots:

  Path A (histogram + rewritten sampler)
    1) Draw `shots` loss syndromes.
    2) Count shots required per distinct syndrome pattern.
    3) For each pattern: rewrite once, then sample the rewritten circuit the required number of times.

  Path B (integrated tableau path)
    Call ``run(seed)`` exactly ``shots`` times with sequential seeds.

Distances default to 3, 5, 7 (surface_code:rotated_memory_z + add_noise).

Example:

  uv run python scripts/benchmark_syndrome_vs_run.py --shots 500 --repeats 3
"""

from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np
import stim

from vsim.loss_lib import (
    LossSyndrome,
    LossyCircuit,
    add_noise,
    apply_loss_to_measurement_record,
)


def _canonical_syndrome_tuple(s: LossSyndrome) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(s.data))


def _clone_syndrome(s: LossSyndrome) -> LossSyndrome:
    o = LossSyndrome(s.rot)
    o.data = list(s.data)
    return o


def build_lossy_circuit(
    distance: int,
    rounds: int | None,
    p_loss_2q: float,
    p_loss_reset: float,
) -> LossyCircuit:
    rounds = rounds if rounds is not None else distance
    stim_surface = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    return add_noise(stim_surface, p_loss_2q, p_loss_reset)


def check_syndrome_agreement(lc: LossyCircuit, n_seeds: int) -> tuple[int, list[int]]:
    mismatches: list[int] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        s_a = lc.syndrome(rng)
        s_b, _rec = lc.run(seed)
        if _canonical_syndrome_tuple(s_a) != _canonical_syndrome_tuple(s_b):
            mismatches.append(seed)
    return n_seeds - len(mismatches), mismatches


def bench_histogram_and_sampler(
    lc: LossyCircuit,
    shots: int,
    seed: int,
) -> tuple[float, float, int]:
    """Returns (histogram_seconds, rewrite_tableau_seconds, unique_patterns)."""
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    counts: dict[tuple[tuple[int, int], ...], int] = {}
    templates: dict[tuple[tuple[int, int], ...], LossSyndrome] = {}
    for _ in range(shots):
        s = lc.syndrome(rng)
        key = _canonical_syndrome_tuple(s)
        counts[key] = counts.get(key, 0) + 1
        templates.setdefault(key, _clone_syndrome(s))
    t_hist = time.perf_counter() - t0

    t1 = time.perf_counter()
    for bi, key in enumerate(sorted(counts.keys())):
        cnt = counts[key]
        rew, loss_measurement_idxs = lc.rewrite_circuit_from_syndrome(templates[key])
        result = np.array(
            rew.compile_sampler(seed=int(seed + bi)).sample(cnt),
            dtype=np.uint8,
        )
        apply_loss_to_measurement_record(result, loss_measurement_idxs)

    t_samplers = time.perf_counter() - t1

    return t_hist, t_samplers, len(counts)


def bench_run_path(lc: LossyCircuit, shots: int, seed_start: int) -> float:
    t0 = time.perf_counter()
    for i in range(shots):
        lc.run(seed_start + i)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark histogram+rewrite+sampler vs integrated run()",
    )
    p.add_argument(
        "--distances", type=int, nargs="+", default=[3, 5, 7], help="default=3 5 7"
    )
    p.add_argument("--rounds", type=int, default=None, help="default: same as distance")
    p.add_argument(
        "--shots", type=int, default=200, help="total Monte Carlo shots, default=200"
    )
    p.add_argument(
        "--repeats", type=int, default=5, help="repeat timings; report mean, default=5"
    )
    p.add_argument(
        "--p-loss-2q",
        type=float,
        default=0.01,
        help="probability of loss on 2-qubit measurements, default=0.01",
    )
    p.add_argument(
        "--p-loss-reset",
        type=float,
        default=0.01,
        help="probability of loss on reset operations, default=0.01",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="RNG for syndrome histogram phase, default=7",
    )
    args = p.parse_args()

    for d in args.distances:
        lc = build_lossy_circuit(d, args.rounds, args.p_loss_2q, args.p_loss_reset)
        print(f"\ndistance={d}")

        hist_times: list[float] = []
        samp_times: list[float] = []
        uniqs: list[int] = []
        run_times: list[float] = []

        for repeat_index in range(args.repeats):
            th, ts, u = bench_histogram_and_sampler(lc, args.shots, args.seed)
            hist_times.append(th)
            samp_times.append(ts)
            uniqs.append(u)
            run_times.append(bench_run_path(lc, args.shots, args.seed + repeat_index))

        mt_hist = mean(hist_times)
        mt_samp = mean(samp_times)
        mt_rewrite_total = mt_hist + mt_samp
        mt_run = mean(run_times)
        ratio = mt_run / mt_rewrite_total if mt_rewrite_total > 0 else float("inf")

        print(
            f"  shots={args.shots}  unique_syndromes~{mean(uniqs):.1f} (mean over repeats)"
        )
        print(f"  histogram phase (syndrome draws): {mt_hist:.4f}s")
        print(f"  rewrite+sampler phase (flattened circuit per bucket): {mt_samp:.4f}s")
        print(f"  path A total: {mt_rewrite_total:.4f}s")
        print(f"  path B run(): {mt_run:.4f}s")
        print(f"  ratio(run/A_total)={ratio:.2f}x")


if __name__ == "__main__":
    main()
