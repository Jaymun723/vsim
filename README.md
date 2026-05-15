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
lossy_circuit = FastLossyCircuit.from_text(
    str(add_noise(circuit, p_2q = 0.01, p_reset = 0.01))
)

results = []
shots = 10_000

# this should run in less than 1s
for _ in range(shots):
    results.append(
        lossy_circuit.run()
    )
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
