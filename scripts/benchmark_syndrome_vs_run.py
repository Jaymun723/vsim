"""
Benchmark different simulation paths:

  Path A (histogram + rewritten sampler)
    1) Draw `shots` loss syndromes.
    2) Count shots required per distinct syndrome pattern.
    3) For each pattern: rewrite once, then sample the rewritten circuit the required number of times.

  Path B (histogram + rewritten tableau loop)
    1) Draw `shots` loss syndromes.
    2) Count shots required per distinct syndrome pattern.
    3) For each pattern: rewrite once, then run TableauSimulator.do_circuit for each shot.

  Path C (integrated tableau path)
    Call ``run(seed)`` exactly ``shots`` times with sequential seeds.

  Path D (C++ FastLossyCircuit.run() loop)
    Same shot loop as Path C, but using the native FastLossyCircuit that
    drives ``stim::TableauSimulator`` from C++ with parsing amortised in
    ``__init__``.

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

from vsim import FastLossyCircuit
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


def bench_histogram_and_tableau(
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

        results = np.zeros((cnt, rew.num_measurements), dtype=np.uint8)

        for i in range(cnt):
            tableau = stim.TableauSimulator(seed=int(seed + i))
            tableau.do_circuit(rew)
            results[i, :] = np.array(
                tableau.current_measurement_record(), dtype=np.uint8
            )

        apply_loss_to_measurement_record(results, loss_measurement_idxs)

    t_tableaus = time.perf_counter() - t1

    return t_hist, t_tableaus, len(counts)


def bench_run_path(lc: LossyCircuit, shots: int, seed_start: int) -> float:
    t0 = time.perf_counter()
    for i in range(shots):
        lc.run(seed_start + i)
    return time.perf_counter() - t0


def bench_fast_run_path(
    fc: FastLossyCircuit, shots: int, seed_start: int
) -> float:
    t0 = time.perf_counter()
    for i in range(shots):
        fc.run(seed_start + i)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark simulation paths A, B, and C",
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
    p.add_argument("--skip-a", action="store_true", help="Skip path A")
    p.add_argument("--skip-b", action="store_true", help="Skip path B")
    p.add_argument("--skip-c", action="store_true", help="Skip path C")
    p.add_argument("--skip-d", action="store_true", help="Skip path D")
    args = p.parse_args()

    for d in args.distances:
        lc = build_lossy_circuit(d, args.rounds, args.p_loss_2q, args.p_loss_reset)
        print(f"\ndistance={d}")

        # FastLossyCircuit reuses the same circuit text — build once per distance.
        if FastLossyCircuit is not None and not args.skip_d:
            fc = FastLossyCircuit.from_text(lc.pretty_print())
        else:
            fc = None

        res_a: list[tuple[float, float, int]] = []
        res_b: list[tuple[float, float, int]] = []
        res_c: list[float] = []
        res_d: list[float] = []

        for repeat_index in range(args.repeats):
            if not args.skip_a:
                res_a.append(bench_histogram_and_sampler(lc, args.shots, args.seed))
            if not args.skip_b:
                res_b.append(bench_histogram_and_tableau(lc, args.shots, args.seed))
            if not args.skip_c:
                res_c.append(bench_run_path(lc, args.shots, args.seed + repeat_index))
            if not args.skip_d and fc is not None:
                res_d.append(bench_fast_run_path(fc, args.shots, args.seed + repeat_index))

        if not args.skip_a:
            m_hist = mean(r[0] for r in res_a)
            m_samp = mean(r[1] for r in res_a)
            m_uniqs = mean(r[2] for r in res_a)
            print(f"  Path A (hist+sampler) total: {m_hist + m_samp:.4f}s (hist={m_hist:.4f}s, samp={m_samp:.4f}s, uniq={m_uniqs:.1f})")

        if not args.skip_b:
            m_hist = mean(r[0] for r in res_b)
            m_tab = mean(r[1] for r in res_b)
            m_uniqs = mean(r[2] for r in res_b)
            print(f"  Path B (hist+tableau) total: {m_hist + m_tab:.4f}s (hist={m_hist:.4f}s, tab={m_tab:.4f}s, uniq={m_uniqs:.1f})")

        if not args.skip_c:
            m_run = mean(res_c)
            print(f"  Path C (run() loop)   total: {m_run:.4f}s")

        if not args.skip_d and res_d:
            m_fast = mean(res_d)
            speedup_c = (mean(res_c) / m_fast) if res_c else float("nan")
            print(
                f"  Path D (Fast run loop) total: {m_fast:.4f}s"
                + (f" (speedup vs C: {speedup_c:.1f}x)" if res_c else "")
            )


if __name__ == "__main__":
    main()
