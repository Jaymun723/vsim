"""vsim: quantum circuit simulation helpers."""

from vsim.loss_lib import (
    LossInstruction,
    LossSyndrome,
    LossyCircuit,
    SymmetricalLossyCircuit,
    add_noise,
    apply_loss_to_measurement_record,
)

try:
    from vsim._fast_loss_lib import FastLossyCircuit  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    FastLossyCircuit = None  # type: ignore[assignment]

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
