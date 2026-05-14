import stim
import pytest
from vsim.stim_parse import parse_stim_target, parse_stim_line

def test_parse_stim_target_special():
    assert parse_stim_target("*") == stim.target_combiner()
    assert parse_stim_target("sweep[5]") == stim.target_sweep_bit(5)
    assert parse_stim_target("!3") == stim.target_inv(3)
    assert parse_stim_target("X10") == stim.target_x(10)
    assert parse_stim_target("Y20") == stim.target_y(20)
    assert parse_stim_target("Z30") == stim.target_z(30)
    assert parse_stim_target("42") == 42

def test_parse_stim_line_invalid():
    with pytest.raises(ValueError, match="Invalid Stim instruction format"):
        parse_stim_line("!!! invalid")

def test_parse_stim_line_basic():
    inst = parse_stim_line("X 0 1")
    assert inst.name == "X"
    assert inst.targets_copy() == [stim.GateTarget(0), stim.GateTarget(1)]

def test_parse_stim_line_with_args():
    inst = parse_stim_line("X_ERROR(0.1) 0")
    assert inst.name == "X_ERROR"
    assert inst.gate_args_copy() == [0.1]
    assert inst.targets_copy() == [stim.GateTarget(0)]

def test_parse_stim_line_empty_or_comment():
    assert parse_stim_line("") is None
    assert parse_stim_line("  ") is None
    assert parse_stim_line("# comment") is None
    assert parse_stim_line("X 0 # comment") is not None

def test_parse_stim_target_rec():
    assert parse_stim_target("rec[-1]") == stim.target_rec(-1)
