// Fast C++ implementation of LossyCircuit.run() for vsim.
//
// Exposes a `FastLossyCircuit` Python class. Parsing is performed once in
// __init__; run() reuses the parsed representation, samples loss dice in C++,
// and drives stim::TableauSimulator directly.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "stim/circuit/circuit.h"
#include "stim/circuit/circuit_instruction.h"
#include "stim/circuit/gate_target.h"
#include "stim/gates/gates.h"
#include "stim/io/measure_record.h"
#include "stim/mem/simd_word.h"
#include "stim/simulators/tableau_simulator.h"
#include "stim/util_bot/probability_util.h"

// Defined in stim's py/base.pybind.cc (which we don't compile); inline the
// constant here to avoid pulling in stim's pybind layer.

namespace py = pybind11;

namespace {

constexpr size_t W = stim::MAX_BITWORD_WIDTH;

// What kind of handling each parsed step needs at run time.
enum class StepCat : uint8_t {
    LOSS,         // Custom LOSS(p) instruction; sample dice, update missing bitset.
    ONEQ,         // 1-qubit (gate or 1Q error / heralded error): drop missing targets.
    TWOQ,         // 2-qubit gate / 2Q error: drop a pair if either qubit is missing.
    MEASURE,      // M, MX, MY, MZ: reset missing targets then measure all; record loss positions.
    RESET,        // R, RX, RY, RZ: clear missing bits, pass through.
    MEAS_RESET,   // MR, MRX, MRY, MRZ: like MEASURE then clear missing bits.
    PASSTHROUGH,  // Everything else (DETECTOR, TICK, QUBIT_COORDS, OBSERVABLE_INCLUDE, ...).
};

struct LossInst {
    double p;
    std::vector<uint32_t> qubits;
};

struct Step {
    StepCat cat;
    uint32_t op_index;    // Index into nominal_circuit.operations (for non-LOSS).
    uint32_t loss_index;  // Index into loss_instructions (for LOSS).
};

StepCat categorize_gate(stim::GateType g) {
    using stim::GateType;
    switch (g) {
        case GateType::MR:
        case GateType::MRX:
        case GateType::MRY:
            return StepCat::MEAS_RESET;
        case GateType::M:
        case GateType::MX:
        case GateType::MY:
            return StepCat::MEASURE;
        case GateType::R:
        case GateType::RX:
        case GateType::RY:
            return StepCat::RESET;
        // Match the Python set _1Q_ERROR_GATES: heralded gates are treated as
        // 1Q errors (filter targets), not as measurements.
        case GateType::HERALDED_ERASE:
        case GateType::HERALDED_PAULI_CHANNEL_1:
            return StepCat::ONEQ;
        default:
            break;
    }
    auto flags = stim::GATE_DATA[g].flags;
    if (flags & stim::GateFlags::GATE_TARGETS_PAIRS) {
        return StepCat::TWOQ;
    }
    if (flags & stim::GateFlags::GATE_IS_SINGLE_QUBIT_GATE) {
        return StepCat::ONEQ;
    }
    return StepCat::PASSTHROUGH;
}

LossInst parse_loss_line(std::string_view line) {
    // Expects "LOSS(p) q1 q2 ..."
    if (line.substr(0, 4) != "LOSS") {
        throw std::runtime_error("Not a LOSS line");
    }
    line.remove_prefix(4);
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
        line.remove_prefix(1);
    }
    if (line.empty() || line.front() != '(') {
        throw std::runtime_error("LOSS missing '('");
    }
    line.remove_prefix(1);
    auto close = line.find(')');
    if (close == std::string_view::npos) {
        throw std::runtime_error("LOSS missing ')'");
    }
    std::string p_str(line.substr(0, close));
    LossInst out;
    out.p = std::stod(p_str);
    line.remove_prefix(close + 1);

    std::string targets_str(line);
    std::stringstream ss(targets_str);
    uint32_t q;
    while (ss >> q) {
        out.qubits.push_back(q);
    }
    return out;
}

class FastLossyCircuit {
   public:
    stim::Circuit nominal_circuit;
    std::vector<LossInst> loss_instructions;
    std::vector<Step> steps;
    size_t num_qubits = 0;
    size_t total_loss_targets = 0;
    size_t total_measurements_upper_bound = 0;

    explicit FastLossyCircuit(const std::string &path) {
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        std::stringstream buf;
        buf << f.rdbuf();
        std::string text = buf.str();
        parse_text(text);
    }

    static FastLossyCircuit from_text(const std::string &text) {
        FastLossyCircuit out;
        out.parse_text(text);
        return out;
    }

    void parse_text(const std::string &text) {
        size_t pos = 0;
        while (pos <= text.size()) {
            size_t end = text.find('\n', pos);
            if (end == std::string::npos) {
                end = text.size();
            }
            std::string_view line(text.data() + pos, end - pos);
            pos = end + 1;

            // Strip whitespace
            while (!line.empty() &&
                   (line.front() == ' ' || line.front() == '\t' || line.front() == '\r')) {
                line.remove_prefix(1);
            }
            while (!line.empty() &&
                   (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) {
                line.remove_suffix(1);
            }
            if (line.empty()) {
                continue;
            }

            // Strip inline comment
            auto hash = line.find('#');
            if (hash != std::string_view::npos) {
                line.remove_suffix(line.size() - hash);
                while (!line.empty() &&
                       (line.back() == ' ' || line.back() == '\t')) {
                    line.remove_suffix(1);
                }
                if (line.empty()) {
                    continue;
                }
            }

            if (line.size() >= 4 && line.substr(0, 4) == "LOSS") {
                Step s;
                s.cat = StepCat::LOSS;
                s.loss_index = (uint32_t)loss_instructions.size();
                s.op_index = 0;
                loss_instructions.push_back(parse_loss_line(line));
                total_loss_targets += loss_instructions.back().qubits.size();
                steps.push_back(s);
                for (uint32_t q : loss_instructions.back().qubits) {
                    if ((size_t)q + 1 > num_qubits) {
                        num_qubits = q + 1;
                    }
                }
            } else {
                stim::Circuit tmp;
                try {
                    tmp.append_from_text(std::string(line));
                } catch (const std::exception &e) {
                    throw std::runtime_error(std::string("Failed to parse line '") +
                                             std::string(line) + "': " + e.what());
                }
                for (const auto &op : tmp.operations) {
                    Step s;
                    s.op_index = (uint32_t)nominal_circuit.operations.size();
                    s.cat = categorize_gate(op.gate_type);
                    s.loss_index = 0;
                    nominal_circuit.safe_append(op, /*block_fusion=*/true);
                    steps.push_back(s);

                    if (s.cat == StepCat::MEASURE || s.cat == StepCat::MEAS_RESET) {
                        total_measurements_upper_bound += op.targets.size();
                    }
                }
            }
        }

        size_t c = nominal_circuit.count_qubits();
        if (c > num_qubits) {
            num_qubits = c;
        }
    }

    py::array_t<uint8_t> run(py::object seed_obj, bool use_numpy_rng = true) {
        // Construct an RNG matching stim's seeding policy
        // (stim.TableauSimulator(seed=...) XORs in INTENTIONAL_VERSION_SEED_INCOMPATIBILITY).
        std::mt19937_64 rng;
        uint64_t raw_seed = 0;
        bool has_seed = !seed_obj.is_none();
        if (has_seed) {
            try {
                raw_seed = py::cast<uint64_t>(seed_obj);
            } catch (const py::cast_error &) {
                throw std::invalid_argument(
                    "Expected seed to be None or a 64 bit unsigned integer.");
            }
            rng = std::mt19937_64(raw_seed ^ stim::INTENTIONAL_VERSION_SEED_INCOMPATIBILITY);
        } else {
            rng = stim::externally_seeded_rng();
        }

        // Roll all loss dice up-front.
        std::vector<double> dice;
        dice.resize(total_loss_targets);
        if (total_loss_targets > 0) {
            if (use_numpy_rng) {
                // Use Numpy to match Python's PCG64 exactly.
                py::object numpy = py::module_::import("numpy");
                py::object numpy_random = py::module_::import("numpy.random");
                py::object rng_np = numpy_random.attr("default_rng")(seed_obj);
                py::array_t<double> py_dice = rng_np.attr("random")(
                    total_loss_targets, py::arg("dtype") = numpy.attr("float64"));
                auto r = py_dice.unchecked<1>();
                for (size_t i = 0; i < total_loss_targets; ++i) {
                    dice[i] = r(i);
                }
            } else {
                // Use native C++ RNG for performance/independence.
                std::mt19937_64 dice_rng;
                if (has_seed) {
                    dice_rng = std::mt19937_64(raw_seed);
                } else {
                    dice_rng = stim::externally_seeded_rng();
                }
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                for (size_t i = 0; i < total_loss_targets; ++i) {
                    dice[i] = dist(dice_rng);
                }
            }
        }
        size_t dice_offset = 0;

        std::vector<uint8_t> missing(num_qubits, 0);

        std::vector<size_t> loss_measurement_positions;
        loss_measurement_positions.reserve(64);
        size_t measurement_index = 0;

        // Build the "rewritten" circuit in C++ while walking steps. After all
        // steps are processed we hand it to TableauSimulator in a single shot.
        stim::Circuit out;
        out.target_buf = stim::MonotonicBuffer<stim::GateTarget>(
            nominal_circuit.target_buf.total_allocated());
        out.arg_buf = stim::MonotonicBuffer<double>(
            nominal_circuit.arg_buf.total_allocated());
        out.tag_buf = stim::MonotonicBuffer<char>(nominal_circuit.tag_buf.total_allocated());
        out.operations.reserve(nominal_circuit.operations.size() + 16);

        std::vector<stim::GateTarget> tmp_targets;

        for (const auto &step : steps) {
            if (step.cat == StepCat::LOSS) {
                const auto &li = loss_instructions[step.loss_index];
                double p = li.p;
                for (uint32_t q : li.qubits) {
                    if (dice[dice_offset++] < p) {
                        missing[q] = 1;
                    }
                }
                continue;
            }

            const auto &op = nominal_circuit.operations[step.op_index];

            switch (step.cat) {
                case StepCat::ONEQ: {
                    tmp_targets.clear();
                    for (const auto &t : op.targets) {
                        uint32_t q = t.qubit_value();
                        if (!missing[q]) {
                            tmp_targets.push_back(t);
                        }
                    }
                    if (!tmp_targets.empty()) {
                        stim::CircuitInstruction filtered(
                            op.gate_type,
                            op.args,
                            tmp_targets,
                            op.tag);
                        out.safe_append(filtered);
                    }
                    break;
                }
                case StepCat::TWOQ: {
                    tmp_targets.clear();
                    size_t n = op.targets.size();
                    for (size_t i = 0; i + 1 < n; i += 2) {
                        uint32_t q1 = op.targets[i].qubit_value();
                        uint32_t q2 = op.targets[i + 1].qubit_value();
                        if (!missing[q1] && !missing[q2]) {
                            tmp_targets.push_back(op.targets[i]);
                            tmp_targets.push_back(op.targets[i + 1]);
                        }
                    }
                    if (!tmp_targets.empty()) {
                        stim::CircuitInstruction filtered(
                            op.gate_type,
                            op.args,
                            tmp_targets,
                            op.tag);
                        out.safe_append(filtered);
                    }
                    break;
                }
                case StepCat::MEASURE:
                case StepCat::MEAS_RESET: {
                    tmp_targets.clear();
                    for (const auto &t : op.targets) {
                        uint32_t q = t.qubit_value();
                        if (missing[q]) {
                            tmp_targets.push_back(stim::GateTarget::qubit(q));
                            loss_measurement_positions.push_back(measurement_index);
                        }
                        measurement_index++;
                    }
                    if (!tmp_targets.empty()) {
                        // Insert an explicit RZ on the missing qubits before the
                        // measurement — same trick as the Python reference path,
                        // ensuring the recorded value is well-defined (and will be
                        // overwritten with 2 below).
                        stim::CircuitInstruction reset_inst(
                            stim::GateType::R, {}, tmp_targets, std::string_view{});
                        out.safe_append(reset_inst);
                    }
                    out.safe_append(op);
                    if (step.cat == StepCat::MEAS_RESET) {
                        for (const auto &t : op.targets) {
                            missing[t.qubit_value()] = 0;
                        }
                    }
                    break;
                }
                case StepCat::RESET: {
                    for (const auto &t : op.targets) {
                        missing[t.qubit_value()] = 0;
                    }
                    out.safe_append(op);
                    break;
                }
                case StepCat::PASSTHROUGH:
                default:
                    out.safe_append(op);
                    break;
                case StepCat::LOSS:
                    // already handled above
                    break;
            }
        }

        stim::TableauSimulator<W> sim(std::move(rng), num_qubits);
        sim.safe_do_circuit(out);

        const auto &storage = sim.measurement_record.storage;
        size_t n_meas = storage.size();
        py::array_t<uint8_t> arr(static_cast<py::ssize_t>(n_meas));
        uint8_t *out_ptr = arr.mutable_data();
        for (size_t i = 0; i < n_meas; ++i) {
            out_ptr[i] = storage[i] ? 1 : 0;
        }
        for (size_t pos : loss_measurement_positions) {
            if (pos < n_meas) {
                out_ptr[pos] = 2;
            }
        }
        return arr;
    }

   private:
    FastLossyCircuit() = default;
};

}  // namespace

PYBIND11_MODULE(_fast_loss_lib, m) {
    m.doc() =
        "C++ acceleration for LossyCircuit.run(): a `FastLossyCircuit` class "
        "that pre-parses the circuit and drives stim::TableauSimulator "
        "directly per shot.";

    py::class_<FastLossyCircuit>(m, "FastLossyCircuit")
        .def(py::init<const std::string &>(), py::arg("circuit_path"))
        .def_static(
            "from_text", &FastLossyCircuit::from_text, py::arg("text"))
        .def("run", &FastLossyCircuit::run, py::arg("seed") = py::none(), py::arg("use_numpy_rng") = true)
        .def_readonly("num_qubits", &FastLossyCircuit::num_qubits)
        .def_readonly("num_loss_instructions",
                      &FastLossyCircuit::total_loss_targets);
}
