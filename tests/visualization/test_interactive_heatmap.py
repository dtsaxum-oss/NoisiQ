import matplotlib
matplotlib.use("Agg")

import pytest

from noisiq.ir import Circuit, gates as ir
from noisiq.noise import get_hardware, fill_idle_with_identities
from noisiq.backends import ManyShotRunner
from noisiq.visualization import interactive_heatmap
from noisiq.visualization.gate_info import GateInfoExtractor


def _small_circuit_and_result():
    c = Circuit(2)
    c.add_gate(ir.H,    (0,), t=0)
    c.add_gate(ir.CNOT, (0, 1), t=1)

    profile = get_hardware("ibm_eagle_r3")
    noise = profile.to_noise_model(c, representation="pauli_twirl")
    result = ManyShotRunner().run(c, n_shots=100, noise_config=noise, seed=42)
    return c, result, noise


def _kraus_circuit_and_result():
    """Circuit filled with IDLEs + Kraus noise for wire-halo tests."""
    c = Circuit(2)
    c.add_gate(ir.H,    (0,), t=0)
    c.add_gate(ir.CNOT, (0, 1), t=2)

    profile = get_hardware("ibm_eagle_r3")
    filled = fill_idle_with_identities(c, profile.gate_times)
    pauli_noise = profile.to_noise_model(filled, representation="pauli_twirl")
    result = ManyShotRunner().run(filled, n_shots=100, noise_config=pauli_noise, seed=42)
    kraus_noise = profile.to_noise_model(filled)
    return filled, result, pauli_noise, kraus_noise


# ---------------------------------------------------------------------------
# GateInfoExtractor — noise-aware fields
# ---------------------------------------------------------------------------

def test_extractor_has_noise_keys_when_config_given():
    c, result, noise = _small_circuit_and_result()
    info = GateInfoExtractor.for_gate(0, result, circuit=c, noise_config=noise)
    assert "noise" in info
    assert "cumulative" in info
    assert "per_gate_decoherence" in info


def test_extractor_without_circuit_no_noise_fields():
    c, result, noise = _small_circuit_and_result()
    info = GateInfoExtractor.for_gate(0, result)
    assert "noise" not in info
    assert "cumulative" not in info


def test_format_full_runs_without_error():
    c, result, noise = _small_circuit_and_result()
    info = GateInfoExtractor.for_gate(0, result, circuit=c, noise_config=noise)
    info["op_idx"] = 0
    s = GateInfoExtractor.format_full(info)
    assert "Gate" in s
    assert "Qubits" in s


def test_format_compact_runs_without_error():
    c, result, noise = _small_circuit_and_result()
    info = GateInfoExtractor.for_gate(0, result, circuit=c, noise_config=noise)
    info["op_idx"] = 0
    s = GateInfoExtractor.format_compact(info)
    assert len(s) > 0


def test_qubit_endcap_info():
    c, result, noise = _small_circuit_and_result()
    info = GateInfoExtractor.for_qubit_endcap(0, c, result, noise_config=noise)
    assert info["qubit"] == 0
    assert "last_op_idx" in info
    assert "total_error_events" in info


# ---------------------------------------------------------------------------
# interactive_heatmap — annotate mode (works with Agg backend)
# ---------------------------------------------------------------------------

def test_interactive_annotate_smokes():
    c, result, noise = _small_circuit_and_result()
    handle = interactive_heatmap(result, c, noise_config=noise, display_mode="annotate")
    assert handle.figure is not None
    assert hasattr(handle, "_annotation")


def test_interactive_annotate_no_noise_config():
    c, result, _ = _small_circuit_and_result()
    handle = interactive_heatmap(result, c, display_mode="annotate")
    assert handle.figure is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_interactive_annotate_with_wire_halo_metric():
    _, result, _, kraus_noise = _kraus_circuit_and_result()
    filled = result.circuit
    handle = interactive_heatmap(
        result, filled,
        noise_config=kraus_noise,
        wire_halo_metric=lambda s: s.gamma,
        display_mode="annotate",
    )
    assert handle.figure is not None


# ---------------------------------------------------------------------------
# interactive_heatmap — panel mode (requires ipywidgets)
# ---------------------------------------------------------------------------

def test_interactive_panel_smokes():
    pytest.importorskip("ipywidgets")
    pytest.importorskip("ipympl")
    c, result, noise = _small_circuit_and_result()
    handle = interactive_heatmap(result, c, noise_config=noise, display_mode="panel")
    assert handle.figure is not None


# ---------------------------------------------------------------------------
# interactive_heatmap — invalid display_mode
# ---------------------------------------------------------------------------

def test_invalid_display_mode_raises():
    c, result, noise = _small_circuit_and_result()
    with pytest.raises(ValueError, match="Unknown display_mode"):
        interactive_heatmap(result, c, display_mode="bad_mode")


# ---------------------------------------------------------------------------
# plot_error_heatmap — backward compat (no noise_config)
# ---------------------------------------------------------------------------

def test_plot_error_heatmap_no_noise_config():
    from noisiq.visualization import plot_error_heatmap
    c, result, _ = _small_circuit_and_result()
    fig = plot_error_heatmap(result, c)
    assert fig is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_error_heatmap_with_kraus_noise():
    from noisiq.visualization import plot_error_heatmap
    filled, result, _, kraus_noise = _kraus_circuit_and_result()
    fig = plot_error_heatmap(result, filled, noise_config=kraus_noise)
    assert fig is not None
