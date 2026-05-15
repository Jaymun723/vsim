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

try:
    from importlib.metadata import version as _version
except ImportError:  # pragma: no cover
    from importlib_metadata import version as _version  # type: ignore

try:
    __version__ = _version("vsim")
except Exception:  # pragma: no cover
    __version__ = "unknown"

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
