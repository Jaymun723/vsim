"""vsim: quantum circuit simulation helpers."""

from vsim.loss_lib import (
    LossInstruction,
    LossSyndrome,
    LossyCircuit,
    SymmetricalLossyCircuit,
    add_noise,
    apply_loss_to_measurement_record,
)
from vsim._fast import FastLossyCircuit

__version__ = "0.1.0"

__all__ = [
    "FastLossyCircuit",
    "LossInstruction",
    "LossSyndrome",
    "LossyCircuit",
    "SymmetricalLossyCircuit",
    "add_noise",
    "apply_loss_to_measurement_record",
    "__version__",
]
