import numpy as np
import stim
from vsim.loss_lib import LossyCircuit, LossInstruction, LossSyndrome

def test_no_loss_instructions():
    # Covers 400->403: if n_targets:
    circuit_text = "X 0\nM 0"
    lossy = LossyCircuit.from_text(circuit_text)
    syndrome, measurements = lossy.run()
    assert len(syndrome.data) == 0
    assert measurements.shape == (1,)

def test_all_qubits_lost_1q_gate():
    # Covers 333->317 and 426->406: if targets: for 1Q gates
    # We use p=1.0 to guarantee loss
    # We use Y 0 so we can distinguish it from the recovery X 0
    circuit_text = "LOSS(1.0) 0\nY 0\nM 0"
    lossy = LossyCircuit.from_text(circuit_text)
    
    # In run()
    syndrome, measurements = lossy.run(seed=123)
    assert len(syndrome.data) == 1
    
    # In rewrite_circuit_from_syndrome()
    rewritten, _ = lossy.rewrite_circuit_from_syndrome(syndrome)
    
    # Check that Y 0 is NOT in rewritten circuit
    assert "Y 0" not in str(rewritten)
    # Check that recovery R 0 and X 0 are there (for the measurement)
    assert "R 0" in str(rewritten)
    assert "X 0" in str(rewritten)

def test_all_qubits_lost_2q_gate():
    # Covers 352->317 and 445->406: if targets: for 2Q gates
    circuit_text = "LOSS(1.0) 0 1\nCNOT 0 1\nM 0 1"
    lossy = LossyCircuit.from_text(circuit_text)
    
    syndrome, measurements = lossy.run(seed=123)
    rewritten, _ = lossy.rewrite_circuit_from_syndrome(syndrome)
    
    assert "CNOT 0 1" not in str(rewritten)

def test_partial_loss_2q_gate():
    # One qubit lost, one not. 2Q gate should still be skipped.
    circuit_text = "LOSS(1.0) 0\nCNOT 0 1\nM 0 1"
    lossy = LossyCircuit.from_text(circuit_text)
    
    syndrome, measurements = lossy.run(seed=123)
    rewritten, _ = lossy.rewrite_circuit_from_syndrome(syndrome)
    
    assert "CNOT 0 1" not in str(rewritten)

def test_lossy_circuit_from_path(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "test.stim"
    p.write_text("X 0\nM 0")
    lossy = LossyCircuit(p)
    assert len(lossy.circuit) == 2
