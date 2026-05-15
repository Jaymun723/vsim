# vsim

Loss-aware sampling for Stim circuits (surface-code workflows).


## Install

vsim ships a native C++ extension (`FastLossyCircuit`) that drives
`stim::TableauSimulator` directly. Building it requires:

- a C++20 compiler (GCC 10+ / Clang 12+)
- CMake ≥ 3.20
- Python ≥ 3.14
- the upstream stim source tree, vendored as a git submodule

Clone with submodules, or initialise them after the fact:

```bash
git clone --recurse-submodules <repo-url>
# or, in an existing checkout:
git submodule update --init --recursive
```

Then install:

```bash
uv sync --group dev        # editable dev env (pytest, notebooks, …)
# or
pip install vsim
```

The build uses `scikit-build-core` + CMake to compile the extension
together with the vendored stim sources. The CMake step filters out
stim's tests, perf benchmarks, pybind layer, and subsystems unused by
`TableauSimulator`; the rest is compiled into the extension module.

**GCC ≥ 15 note.** Several stim headers omit `<cstdint>` and rely on it
being pulled in transitively. The CMake build force-includes it for our
extension, but the `stim` PyPI wheel (a build-time dependency on Python
3.14) needs the same workaround at install time:

```bash
CXXFLAGS="-include cstdint" uv sync --group dev
```

Run tests:

```bash
uv run --group dev pytest
```

Regenerate the golden loss histogram snapshot (uses `devtools/legacy_loss_lib.py`):

```bash
uv run --group dev python devtools/regenerate_loss_snapshot.py
```


## FastLossyCircuit

`FastLossyCircuit` is a drop-in C++ replacement for the per-shot path of
`LossyCircuit.run()`. The expensive work — parsing, categorising
instructions — happens once in `__init__`; each `run(seed)` then samples
loss dice and drives `stim::TableauSimulator` in C++.

```python
from vsim import FastLossyCircuit

fc = FastLossyCircuit("path/to/circuit.stim")
# or: fc = FastLossyCircuit.from_text(circuit_text)

measurements = fc.run(seed=0)
# np.ndarray[uint8] of length num_measurements
#   0, 1 → measurement outcome
#   2    → heralded-loss slot
```

`run(seed=None)` picks a fresh OS-seeded RNG. Integer seeds are XORed
with stim's `INTENTIONAL_VERSION_SEED_INCOMPATIBILITY` constant so the
gate-side RNG stream matches `stim.TableauSimulator(seed=...)`.

A Path D arm in `scripts/benchmark_syndrome_vs_run.py` exercises this
class against the existing Python paths. Typical speedups vs.
`LossyCircuit.run()` on rotated-memory-z surface codes: ~50× at d=3,
~60× at d=5, ~75× at d=7.
