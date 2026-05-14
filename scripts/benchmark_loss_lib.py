"""
Benchmark LossyCircuit / SymmetricalLossyCircuit syndrome throughput.

Requires an editable install: uv sync --group dev

Example:

```bash
uv run python scripts/benchmark_loss_lib.py --number 800
```
"""

from __future__ import annotations

import argparse
import sys
import timeit
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.stim_fixtures import SURFACE_LOSS_STIM

from vsim.loss_lib import LossyCircuit, SymmetricalLossyCircuit


def bench_loop(
    circ: LossyCircuit | SymmetricalLossyCircuit,
    number: int,
    repeat: int,
) -> float:
    def call():
        rng = np.random.default_rng()
        circ.syndrome(rng)

    return min(timeit.repeat(call, repeat=repeat, number=number))


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark vsim.loss_lib syndrome sampling")
    p.add_argument(
        "--circuit",
        type=Path,
        default=None,
        help="Path to .stim file with LOSS lines (default: embedded surface sample)",
    )
    p.add_argument("--symmetrical", action="store_true")
    p.add_argument("--number", type=int, default=800)
    p.add_argument("--repeat", type=int, default=7)
    args = p.parse_args()

    if args.circuit is None:
        circ: LossyCircuit | SymmetricalLossyCircuit = (
            SymmetricalLossyCircuit.from_text(SURFACE_LOSS_STIM)
            if args.symmetrical
            else LossyCircuit.from_text(SURFACE_LOSS_STIM)
        )
        label_src = "embedded surface sample (tests/stim_fixtures.py)"
    else:
        circuit = args.circuit.resolve()
        if not circuit.is_file():
            raise SystemExit(f"Missing circuit file: {circuit}")
        circ = (
            SymmetricalLossyCircuit(circuit)
            if args.symmetrical
            else LossyCircuit(circuit)
        )
        label_src = str(circuit)

    t = bench_loop(circ, args.number, args.repeat)
    label = "SymmetricalLossyCircuit.syndrome" if args.symmetrical else "LossyCircuit.syndrome"
    print(f"circuit: {label_src}")
    print(f"bench:   {label}, number={args.number}, repeat={args.repeat} (min wall time)")
    print(f"time:    {t:.4f} s")


if __name__ == "__main__":
    main()
