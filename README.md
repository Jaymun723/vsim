# vsim

Loss-aware sampling for Stim circuits (surface-code workflows).


## Install

vsim ships a native C++ extension (`FastLossyCircuit`) that drives
`stim::TableauSimulator` directly. Building it requires:

- a C++20 compiler (GCC 10+ / Clang 12+)
- CMake ≥ 3.20
- Python ≥ 3.14
- the upstream stim source tree, included as a git submodule

### 1. Clone submodules

```bash
git clone --recurse-submodules <repo-url>
# or, in an existing checkout:
git submodule update --init --recursive
```

### 2. Install with the GCC workaround if needed

On Python 3.14 the `stim` PyPI wheel is built from source — and several
upstream stim headers omit `<cstdint>`, which trips GCC ≥ 15. **If your
compiler is GCC 15+ (or you're unsure), use this command:**

```bash
CXXFLAGS="-include cstdint" uv sync --group dev
```

Check your compiler with `g++ --version | head -1`. Plain `uv sync
--group dev` works for GCC 14 and earlier.

If a previous attempt left the venv half-built, blow it away first:

```bash
rm -rf .venv
CXXFLAGS="-include cstdint" uv sync --group dev
```

### 3. Verify

```bash
uv run python -c "import stim, vsim; from vsim import FastLossyCircuit; print('ok', vsim.__version__)"
```

This should print `ok 0.1.0`. If it errors with `ModuleNotFoundError:
No module named 'stim'`, the stim wheel failed to build — re-run step 2
with the `CXXFLAGS` prefix.

### 4. Run tests

```bash
uv run --group dev pytest
```

The build uses `scikit-build-core` + CMake to compile the extension
together with the vendored stim sources. The CMake step filters out
stim's tests, perf benchmarks, pybind layer, and subsystems unused by
`TableauSimulator`; the rest is compiled into the extension module.

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
