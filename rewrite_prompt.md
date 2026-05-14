# Goal: High-Performance C++ Rewrite of Lossy Simulation

You are an expert in writing high-performance C++ and Python extensions. Your goal is to rewrite the `.run()` method of `LossyCircuit` (from `src/vsim/loss_lib.py`) in C++ to achieve significant speedups for single-shot simulations.

## Background & Motivation
Benchmarks show that for large distance $d$, the `run()` loop is the bottleneck. Most use cases only require one shot, making the overhead of Python loops and `stim` Python API calls significant.

**Benchmark Data (scripts/benchmark_syndrome_vs_run.py):**
- distance=5: Path C (run() loop) = 6.1s vs Path A = 8.8s
- distance=7: Path C (run() loop) = 21.5s vs Path A = 22.7s

## Technical Requirements
1. **Target Interface:**
   Implement a Python class `FastLossyCircuit` (using `nanobind` or `pybind11`):
   ```python
   class FastLossyCircuit:
       def __init__(self, circuit_path: Path):
           """Pre-parse the circuit into an efficient C++ internal representation."""
           ...
       
       def run(self, seed: int | None = None) -> np.ndarray:
           """
           Execute one shot.
           Returns: 1D np.ndarray (dtype=np.uint8)
           Values: 0 or 1 (measurement results), 2 (heralded loss).
           """
           ...
   ```

2. **Custom "LOSS" Instruction:**
   The circuits contain custom `LOSS(p) target1 target2 ...` instructions.
   - Standard Stim parsers will fail on this. You must implement a custom parser or extend the Stim parser logic in C++.
   - **Logic:** For each target in `LOSS(p)`, with probability `p`, the qubit is marked as "lost".
   - Lost qubits remain lost until a `RESET` or `MR` (Measure-Reset) instruction is applied to them.

3. **Core Simulation Logic:**
   - Use the **Stim C++ API** (specifically `stim::TableauSimulator`).
   - Maintain a "missing qubits" bitset in C++.
   - **Gates:** Skip operations on lost qubits. For 2-qubit gates, skip if *either* qubit is lost.
   - **Measurements:** If a qubit is lost, the result in the measurement record should be `2`.
   - **Reset:** If a qubit is reset (via `R` or `MR`), it is no longer "lost".

4. **Performance Constraints:**
   - Avoid parallelization for now.
   - Focus on minimizing allocation and string processing inside the `run()` method.
   - The `__init__` should do the heavy lifting of parsing.

## Verification
1. Add a **Path D** to `scripts/benchmark_syndrome_vs_run.py` that utilizes your `FastLossyCircuit`.
2. Run the benchmark and ensure `Path D` is significantly faster than `Path C`.
3. Ensure all tests in `tests/test_loss_lib.py` pass when compared against the original `LossyCircuit`.
