# vsim

Loss-aware sampling for Stim circuits (surface-code workflows).


## Install

```bash
uv add vsim
# Or using pip
pip install vsim
```

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
lossy_circuit = add_noise(stim_surface, p_2q = 0.01, p_reset = 0.01)

results = []
shots = 10_000

# this should run in less than 1s
for _ in range(shots):
    results.append(
        lossy_circuit.run()
    )
```

## Development

Clone the repository then:
```bash
uv sync --group dev        # editable dev env (pytest, notebooks, …)
```

Run tests:

```bash
uv run --group dev pytest
```

### Benchmarks:

Start with this command to have the help message printed:

```bash
uv run python scripts/benchmark_syndrome_vs_run.py --help
```

Full benchmark run:

```bash
uv run scripts/benchmark_syndrome_vs_run.py -A -B -C -D --shots 1000
```