# vsim

Loss-aware sampling for Stim circuits (surface-code workflows).

## Install

```bash
uv add vsim
# Or using pip
pip install vsim
```

Building the native C++ extension requires a C++20 compiler (GCC 10+ / Clang 12+) and CMake ≥ 3.20.

## Usage

```python
from stim import Circuit
from vsim import add_noise, FastLossyCircuit

# Generate a surface code circuit with stim
circuit = Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.01,
    before_measure_flip_probability=0.01,
    after_reset_flip_probability=0.01,
)

# Add loss probabilities after 2 qubit gates, and reset gates
lossy_circuit = add_noise(circuit, p_2q = 0.01, p_reset = 0.01)

results = []
shots = 10_000

# this should run in less than 1s
for _ in range(shots):
    results.append(
        lossy_circuit.run()
    )
```

## FastLossyCircuit

`FastLossyCircuit` is a drop-in C++ replacement for the per-shot path of `LossyCircuit.run()`. The expensive work — parsing and categorising instructions — happens once in `__init__`; each `run(seed)` then samples loss dice and drives `stim::TableauSimulator` in C++.

```python
from vsim import FastLossyCircuit

fc = FastLossyCircuit("path/to/circuit.stim")
# or: fc = FastLossyCircuit.from_text(circuit_text)

measurements = fc.run(seed=0)
# np.ndarray[uint8] of length num_measurements
#   0, 1 → measurement outcome
#   2    → heralded-loss slot
```

Typical speedups vs. Python-based sampling on rotated-memory-z surface codes: ~50× at d=3, ~60× at d=5, ~75× at d=7.

## Development

Clone the repository with submodules:
```bash
git clone --recurse-submodules <repo-url>
```

Setup the development environment:
```bash
uv sync --group dev
```

*Note: On Python 3.14 with GCC 15+, you may need `CXXFLAGS="-include cstdint" uv sync --group dev` to build `stim` correctly.*

Run tests:
```bash
uv run --group dev pytest
```

### Benchmarks

Start with this command to see the help message:
```bash
uv run python scripts/benchmark_syndrome_vs_run.py --help
```

Full benchmark run:
```bash
uv run scripts/benchmark_syndrome_vs_run.py -A -B -C -D --shots 1000
```
