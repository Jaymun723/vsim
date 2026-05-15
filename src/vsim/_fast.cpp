/*
 * FastLossyCircuit – high-performance C++ replacement for LossyCircuit.run().
 *
 * Parses a Stim circuit file once in __init__, classifying every instruction
 * into a compact enum so that run() never touches strings or Python objects.
 *
 * Instruction handling mirrors loss_lib.py:
 *   LOSS(p) targets   – probabilistically mark qubits as lost.
 *   1-qubit gates     – skip if the qubit is lost.
 *   2-qubit gates     – skip the pair if either qubit is lost.
 *   Measurements      – if a qubit is lost, reset it to |0⟩ first
 *                       (so the tableau records 0), then overwrite the
 *                       corresponding result slot with 2.
 *   Resets            – clear the "lost" flag and apply normally.
 *   Annotations       – ignored (DETECTOR, OBSERVABLE_INCLUDE, TICK, …).
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

#include "stim/circuit/circuit.h"
#include "stim/circuit/circuit_instruction.h"
#include "stim/circuit/gate_target.h"
#include "stim/gates/gates.h"
#include "stim/simulators/tableau_simulator.h"
#include "stim/mem/simd_word.h"   // stim::MAX_BITWORD_WIDTH

#include <cstdint>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

// ─────────────────────────────────────────────────────────────────────────────
// Gate classification
// ─────────────────────────────────────────────────────────────────────────────

enum class InstrKind : uint8_t {
    LOSS,          // custom LOSS(p) instruction
    GATE_1Q,       // 1-qubit Clifford or error channel
    GATE_2Q,       // 2-qubit Clifford or error channel (targets in pairs)
    MEASURE,       // M, MX, MY, MZ  (measurement without reset)
    RESET,         // R, RX, RY, RZ  (reset without measurement)
    MEASURE_RESET, // MR, MRX, MRY, MRZ
    IGNORE,        // annotations / block markers with no qubit effect
};

// Classify a stim GateType using the gate's flag bits.
//
// Note: in stim, all measurement gates (M, MR, MRX, ...) are flagged as
// GATE_IS_NOISY because they accept an optional flip-probability argument.
// We must therefore NOT use !is_noisy to identify measurement gates.
// Instead we exclude only the explicitly heralded noise gates
// (HERALDED_ERASE, HERALDED_PAULI_CHANNEL_1), which produce results but
// should be skipped for missing qubits like any other error channel.
static InstrKind classify_gate(stim::GateType gt) noexcept {
    const auto &gate  = stim::GATE_DATA[gt];
    const uint16_t f  = gate.flags;

    if (f & stim::GateFlags::GATE_HAS_NO_EFFECT_ON_QUBITS ||
        f & stim::GateFlags::GATE_IS_BLOCK) {
        return InstrKind::IGNORE;
    }

    const bool produces = (f & stim::GateFlags::GATE_PRODUCES_RESULTS) != 0;
    const bool is_reset = (f & stim::GateFlags::GATE_IS_RESET)          != 0;
    const bool pairs    = (f & stim::GateFlags::GATE_TARGETS_PAIRS)     != 0;

    // Heralded noise gates (HERALDED_ERASE, HERALDED_PAULI_CHANNEL_1) produce
    // results but behave like ordinary error channels in the lossy simulation:
    // simply skip the gate for missing qubits rather than tracking their
    // measurements as heralded-loss entries.
    const bool is_heralded = (gt == stim::GateType::HERALDED_ERASE ||
                               gt == stim::GateType::HERALDED_PAULI_CHANNEL_1);

    if (produces && is_reset && !is_heralded) return InstrKind::MEASURE_RESET;
    if (produces && !is_heralded)             return InstrKind::MEASURE;
    if (is_reset)                             return InstrKind::RESET;
    if (pairs)                                return InstrKind::GATE_2Q;
    return InstrKind::GATE_1Q;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-parsed instruction record (fully owns its data)
// ─────────────────────────────────────────────────────────────────────────────

struct ParsedInstr {
    InstrKind                      kind;
    stim::GateType                 gate_type;
    std::vector<stim::GateTarget>  targets;   // owned copy
    std::vector<double>            args;      // owned copy
    double                         prob;      // LOSS only
};

// ─────────────────────────────────────────────────────────────────────────────
// FastLossyCircuit
// ─────────────────────────────────────────────────────────────────────────────

class FastLossyCircuit {
public:
    std::vector<ParsedInstr> instructions;
    uint32_t                 num_qubits          = 0;
    uint32_t                 n_total_loss_targets = 0;
    // prefix-sum over LOSS target counts; size == (number of LOSS instructions + 1)
    std::vector<uint32_t>    loss_target_offsets;

    // ── Constructor: parse circuit file ──────────────────────────────────────
    explicit FastLossyCircuit(const std::string &circuit_path) {
        std::ifstream f(circuit_path);
        if (!f.is_open())
            throw std::runtime_error("FastLossyCircuit: cannot open file: " + circuit_path);

        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

        // ── Build nominal circuit (no LOSS) to count qubits ─────────────────
        std::string nominal_text;
        {
            std::istringstream ss(text);
            std::string line;
            while (std::getline(ss, line)) {
                line = strip(line);
                if (line.empty() || line[0] == '#') continue;
                if (line.rfind("LOSS", 0) == 0)    continue;  // skip LOSS lines
                nominal_text += line;
                nominal_text += '\n';
            }
        }
        {
            stim::Circuit nominal;
            nominal.append_from_text(nominal_text);
            num_qubits = (uint32_t)nominal.count_qubits();
        }

        // ── Parse all instructions ───────────────────────────────────────────
        loss_target_offsets.push_back(0);  // sentinel at index 0

        std::istringstream ss(text);
        std::string line;
        while (std::getline(ss, line)) {
            line = strip(line);
            if (line.empty() || line[0] == '#') continue;

            if (line.rfind("LOSS", 0) == 0) {
                parse_loss_line(line);
            } else {
                parse_stim_line(line);
            }
        }
    }

    // ── run(): execute one shot ───────────────────────────────────────────────
    nb::ndarray<nb::numpy, uint8_t> run(std::optional<uint64_t> seed) {
        // Seed both the loss RNG and the stim RNG from the same value so that
        // behaviour matches: using the same seed always yields the same result.
        const uint64_t s = seed.has_value() ? *seed : std::random_device{}();
        std::mt19937_64 loss_rng(s);
        std::mt19937_64 sim_rng(s);

        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        // Pre-roll all LOSS dice
        std::vector<double> dice(n_total_loss_targets);
        for (auto &d : dice) d = uniform(loss_rng);

        // Create stim tableau simulator
        stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> sim(std::move(sim_rng), num_qubits);

        // Track which qubits are currently lost
        std::vector<bool> missing(num_qubits, false);

        uint32_t loss_instr_idx = 0;  // index into loss_target_offsets
        uint32_t meas_idx       = 0;
        std::vector<uint32_t> loss_meas_record;

        // Reusable temporary buffer for filtered targets
        std::vector<stim::GateTarget> tmp_targets;

        for (const auto &instr : instructions) {
            switch (instr.kind) {

            case InstrKind::LOSS: {
                const uint32_t lo = loss_target_offsets[loss_instr_idx];
                const uint32_t hi = loss_target_offsets[loss_instr_idx + 1];
                const double   p  = instr.prob;
                for (uint32_t i = 0; i < hi - lo; ++i) {
                    if (dice[lo + i] < p) {
                        const uint32_t q = (uint32_t)instr.targets[i].qubit_value();
                        missing[q] = true;   // once lost, stays lost until reset
                    }
                }
                ++loss_instr_idx;
                break;
            }

            case InstrKind::GATE_1Q: {
                tmp_targets.clear();
                for (const auto &t : instr.targets) {
                    if (!missing[t.qubit_value()])
                        tmp_targets.push_back(t);
                }
                if (!tmp_targets.empty())
                    sim.do_gate(make_ci(instr.gate_type, instr.args, tmp_targets));
                break;
            }

            case InstrKind::GATE_2Q: {
                tmp_targets.clear();
                for (size_t i = 0; i + 1 < instr.targets.size(); i += 2) {
                    if (!missing[instr.targets[i].qubit_value()] &&
                        !missing[instr.targets[i + 1].qubit_value()]) {
                        tmp_targets.push_back(instr.targets[i]);
                        tmp_targets.push_back(instr.targets[i + 1]);
                    }
                }
                if (!tmp_targets.empty())
                    sim.do_gate(make_ci(instr.gate_type, instr.args, tmp_targets));
                break;
            }

            case InstrKind::MEASURE:
            case InstrKind::MEASURE_RESET: {
                // Reset any lost qubits to |0⟩ so measurement yields 0,
                // then record their measurement positions for later marking as 2.
                for (const auto &t : instr.targets) {
                    const uint32_t q = (uint32_t)t.qubit_value();
                    if (missing[q]) {
                        stim::GateTarget rt = stim::GateTarget::qubit(q);
                        static const std::vector<double> no_args;
                        sim.do_gate(stim::CircuitInstruction(
                            stim::GateType::R,
                            stim::SpanRef<const double>(no_args),
                            stim::SpanRef<const stim::GateTarget>(&rt, &rt + 1),
                            ""));
                        loss_meas_record.push_back(meas_idx);
                    }
                    ++meas_idx;
                }
                // Apply the measurement to ALL targets (lost ones are now |0⟩)
                sim.do_gate(make_ci(instr.gate_type, instr.args, instr.targets));
                // Measure-reset also clears the missing flag
                if (instr.kind == InstrKind::MEASURE_RESET) {
                    for (const auto &t : instr.targets)
                        missing[t.qubit_value()] = false;
                }
                break;
            }

            case InstrKind::RESET: {
                for (const auto &t : instr.targets)
                    missing[t.qubit_value()] = false;
                sim.do_gate(make_ci(instr.gate_type, instr.args, instr.targets));
                break;
            }

            case InstrKind::IGNORE:
                break;
            }
        }

        // ── Build numpy output array ─────────────────────────────────────────
        const auto  &storage = sim.measurement_record.storage;
        const size_t n       = storage.size();

        uint8_t *data = new uint8_t[n ? n : 1];
        for (size_t i = 0; i < n; ++i)
            data[i] = storage[i] ? 1u : 0u;
        for (const uint32_t idx : loss_meas_record)
            data[idx] = 2u;

        // Transfer ownership to a capsule so numpy keeps the data alive.
        auto cleanup = [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); };
        nb::capsule owner(data, cleanup);
        const size_t shape[1] = {n};
        return nb::ndarray<nb::numpy, uint8_t>(data, 1, shape, owner);
    }

private:
    // ── Helpers ──────────────────────────────────────────────────────────────

    static std::string strip(std::string s) {
        // Remove leading whitespace
        auto i = s.find_first_not_of(" \t\r\n");
        if (i == std::string::npos) return {};
        s = s.substr(i);
        // Remove inline comment
        auto c = s.find('#');
        if (c != std::string::npos) s = s.substr(0, c);
        // Remove trailing whitespace
        auto j = s.find_last_not_of(" \t\r\n");
        if (j == std::string::npos) return {};
        return s.substr(0, j + 1);
    }

    // Construct a CircuitInstruction that borrows from our owned vectors.
    // Safe because do_gate() consumes the instruction synchronously.
    static stim::CircuitInstruction make_ci(
        stim::GateType                        gt,
        const std::vector<double>            &args,
        const std::vector<stim::GateTarget>  &targets)
    {
        return stim::CircuitInstruction(
            gt,
            stim::SpanRef<const double>(args),
            stim::SpanRef<const stim::GateTarget>(targets),
            "");
    }

    void parse_loss_line(const std::string &line) {
        // Syntax: LOSS(<prob>) <q1> <q2> ...
        ParsedInstr instr;
        instr.kind      = InstrKind::LOSS;
        instr.gate_type = stim::GateType::NOT_A_GATE;
        instr.prob      = 0.0;

        const auto open  = line.find('(');
        const auto close = line.find(')');
        if (open == std::string::npos || close == std::string::npos)
            throw std::runtime_error("FastLossyCircuit: malformed LOSS line: " + line);

        instr.prob = std::stod(line.substr(open + 1, close - open - 1));

        std::istringstream ts(line.substr(close + 1));
        std::string tok;
        while (ts >> tok)
            instr.targets.push_back(stim::GateTarget::qubit((uint32_t)std::stoi(tok)));

        n_total_loss_targets += (uint32_t)instr.targets.size();
        loss_target_offsets.push_back(n_total_loss_targets);

        instructions.push_back(std::move(instr));
    }

    void parse_stim_line(const std::string &line) {
        // Delegate to stim's own parser; copy targets/args so they outlive
        // the temporary Circuit object.
        stim::Circuit tmp;
        tmp.append_from_text(line);

        for (const auto &op : tmp.operations) {
            ParsedInstr instr;
            instr.gate_type = op.gate_type;
            instr.kind      = classify_gate(op.gate_type);
            instr.prob      = 0.0;

            for (const auto &t : op.targets) instr.targets.push_back(t);
            for (const double a : op.args)   instr.args.push_back(a);

            instructions.push_back(std::move(instr));
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// nanobind module
// ─────────────────────────────────────────────────────────────────────────────

NB_MODULE(_fast, m) {
    nb::class_<FastLossyCircuit>(m, "FastLossyCircuit")
        .def(nb::init<std::string>(),
             nb::arg("circuit_path"),
             "Pre-parse the circuit file into an efficient C++ internal representation.")
        .def("run",
             &FastLossyCircuit::run,
             nb::arg("seed") = nb::none(),
             "Execute one shot.  Returns a 1-D numpy uint8 array; "
             "values are 0 or 1 (measurement results) or 2 (heralded loss).");
}
