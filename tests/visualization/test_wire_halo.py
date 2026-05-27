import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from noisiq.ir import Circuit, gates as ir
from noisiq.noise import get_hardware, fill_idle_with_identities
from noisiq.visualization.gate_info import WireSegment
from noisiq.visualization.charts.heatmap import (
    _compute_wire_segments,
    _build_wire_segment_bboxes,
    _hit_test_wire_segment,
)


def _make_filled_circuit_and_kraus_noise():
    """2-qubit circuit filled with IDLEs, using default Kraus noise."""
    c = Circuit(2)
    c.add_gate(ir.H,    (0,), t=0)
    c.add_gate(ir.CNOT, (0, 1), t=1)
    c.add_gate(ir.X,    (0,), t=3)

    profile = get_hardware("ibm_eagle_r3")
    filled = fill_idle_with_identities(c, profile.gate_times)
    noise = profile.to_noise_model(filled)
    return filled, noise, profile


def test_segment_detection_finds_idle_runs():
    filled, noise, profile = _make_filled_circuit_and_kraus_noise()
    segments = _compute_wire_segments(filled, noise)

    # Must have at least one segment
    assert len(segments) >= 1
    for seg in segments:
        assert isinstance(seg, WireSegment)
        assert seg.t_lo < seg.t_hi
        assert len(seg.idle_op_idxs) >= 1


def test_segment_qubit_assignment():
    filled, noise, profile = _make_filled_circuit_and_kraus_noise()
    segments = _compute_wire_segments(filled, noise)

    for seg in segments:
        # Every idle op in the run must be on the declared qubit
        for idx in seg.idle_op_idxs:
            op = filled.operations[idx]
            assert seg.qubit in op.qubits
            assert op.gate is ir.IDLE


def test_segment_intensity_in_unit_interval():
    filled, noise, profile = _make_filled_circuit_and_kraus_noise()
    segments = _compute_wire_segments(filled, noise)

    for seg in segments:
        assert 0.0 <= seg.gamma <= 1.0
        assert 0.0 <= seg.lam <= 1.0
        assert 0.0 <= seg.intensity <= 1.0


def test_combined_intensity_formula():
    """intensity == 1 - (1 - gamma) * (1 - lam)."""
    filled, noise, profile = _make_filled_circuit_and_kraus_noise()
    segments = _compute_wire_segments(filled, noise)

    for seg in segments:
        expected = 1.0 - (1.0 - seg.gamma) * (1.0 - seg.lam)
        assert abs(seg.intensity - expected) < 1e-12


def test_no_segments_without_noise_config():
    """When noise_config is None, no channels → all intensities zero."""
    filled, _, profile = _make_filled_circuit_and_kraus_noise()
    segments = _compute_wire_segments(filled, None)

    for seg in segments:
        assert seg.gamma == 0.0
        assert seg.lam == 0.0
        assert seg.intensity == 0.0


def test_pauli_twirl_noise_gives_zero_intensity():
    """Pauli-twirl noise dicts have no Kraus channels; wire halos are invisible."""
    c = Circuit(2)
    c.add_gate(ir.H,    (0,), t=0)
    c.add_gate(ir.CNOT, (0, 1), t=2)

    profile = get_hardware("ibm_eagle_r3")
    filled = fill_idle_with_identities(c, profile.gate_times)
    pauli_noise = profile.to_pauli_noise_model(filled)

    segments = _compute_wire_segments(filled, pauli_noise)
    for seg in segments:
        assert seg.intensity == 0.0


def test_wire_segment_bbox_hit_test():
    filled, noise, profile = _make_filled_circuit_and_kraus_noise()
    segments = _compute_wire_segments(filled, noise)
    bboxes = _build_wire_segment_bboxes(segments, filled.n_qubits)

    # For each segment, a cursor at its midpoint should hit it.
    for i, seg in enumerate(segments):
        cx = (seg.t_lo + seg.t_hi) / 2.0
        cy = filled.n_qubits - 1 - seg.qubit
        result = _hit_test_wire_segment(cx, cy, bboxes)
        assert result is not None
        assert result is seg


def test_wire_segment_bbox_miss():
    filled, noise, profile = _make_filled_circuit_and_kraus_noise()
    segments = _compute_wire_segments(filled, noise)
    bboxes = _build_wire_segment_bboxes(segments, filled.n_qubits)

    # Far off the diagram should miss everything.
    result = _hit_test_wire_segment(999.0, 999.0, bboxes)
    assert result is None


def test_segment_duration_ns_sums_idle_ops():
    filled, noise, profile = _make_filled_circuit_and_kraus_noise()
    segments = _compute_wire_segments(filled, noise)

    for seg in segments:
        expected_dur = sum(
            filled.operations[idx].params.get("duration_ns", 0.0)
            for idx in seg.idle_op_idxs
            if filled.operations[idx].params
        )
        assert abs(seg.duration_ns - expected_dur) < 1e-9
