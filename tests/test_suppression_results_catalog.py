from pathlib import Path

import numpy as np
import pytest

import noisiq as nq
from noisiq.noise import NoiseModel, depolarizing
from noisiq.results import Result
from tests.conftest import tvd


# ----------------------------------------------------------------------------- dynamical decoupling
@pytest.mark.parametrize("seq", nq.suppression.list_sequences())
def test_dd_preserves_ideal_circuit(seq):
    rng = np.random.default_rng(3)
    c = nq.Circuit(3)
    c.h(0).cx(0, 1)
    for _ in range(6):
        c.x(2).x(2)  # keep qubit 2 busy so qubits 0/1 idle
    c.cx(0, 1).h(0).delay(1, 2e-6).measure_all()
    dd = nq.suppression.apply_dd(c, seq)
    assert len(dd) > len(c)
    p1 = nq.simulate(c, shots=1, backend="density_matrix").probabilities()
    p2 = nq.simulate(dd, shots=1, backend="density_matrix").probabilities()
    assert tvd(p1, p2) < 1e-9
    _ = rng


def test_dd_aliases_and_errors():
    c = nq.Circuit(1).h(0).delay(0, 1e-6).h(0)
    assert nq.suppression.apply_dd(c, "hahn").count_ops()["X"] == 2
    assert nq.suppression.apply_dd(c, "xy-4").count_ops()["Y"] == 2
    with pytest.raises(ValueError):
        nq.suppression.apply_dd(c, "nope")


def test_dd_handles_measurements_conditionals_and_registers():
    c = nq.algorithms.repetition_code(3, 2)
    dd = nq.suppression.apply_dd(c, "XY4")
    assert dd.n_clbits == c.n_clbits
    r1 = nq.simulate(c, shots=1, backend="density_matrix").probabilities()
    r2 = nq.simulate(dd, shots=1, backend="density_matrix").probabilities()
    assert tvd(r1, r2) < 1e-9


def test_dd_refocuses_quasistatic_detuning():
    """Spin echo: a long wait on |+> under quasi-static detuning loses coherence; DD restores it."""
    c = nq.Circuit(1).h(0).delay(0, 20e-6).h(0).measure(0)
    nm = NoiseModel().add_quasistatic_detuning(sigma=2 * np.pi * 30e3)
    plain = nq.simulate(c, nm, shots=3000, seed=1, backend="statevector")
    echo = nq.simulate(nq.suppression.apply_dd(c, "X2"), nm, shots=3000, seed=1, backend="statevector")
    p_err_plain = plain.probabilities().get("1", 0)
    p_err_echo = echo.probabilities().get("1", 0)
    assert p_err_plain > 0.3
    assert p_err_echo < 0.01
    # Twirled (Markovian) version of the same noise cannot be echoed away
    tw = nm.twirled()
    p_tw = nq.simulate(nq.suppression.apply_dd(c, "X2"), tw, shots=3000, seed=1).probabilities().get("1", 0)
    assert p_tw > 0.1


# ----------------------------------------------------------------------------- results
def _noisy_result(**kw):
    c = nq.algorithms.ghz(4, measure=True)
    nm = NoiseModel.uniform_depolarizing(0.01, readout=0.01)
    nm.add_idle_error(lambda d: nq.noise.phase_flip(0.005))
    return nq.simulate(c, nm, shots=3000, seed=11, **kw)


@pytest.mark.parametrize("backend", ["frame", "statevector"])
def test_save_and_load_round_trip(tmp_path: Path, backend):
    r = _noisy_result(backend=backend)
    path = r.save(tmp_path / "run")
    assert path.suffix == ".npz"
    back = Result.load(path)
    assert back.counts == r.counts and back.shots == r.shots and back.backend == r.backend
    assert np.allclose(back.instruction_faults, r.instruction_faults)
    assert back.circuit.count_ops() == r.circuit.count_ops()
    if r.final_errors is not None:
        assert np.array_equal(back.final_errors[0], r.final_errors[0])
    if r.fidelity is not None:
        assert back.fidelity.value == pytest.approx(r.fidelity.value)


def test_error_budget_shares_sum_to_one():
    r = _noisy_result()
    for metric in ("rate", "burden"):
        rows = r.error_budget(metric)
        assert sum(x["share"] for x in rows) == pytest.approx(1.0)
        assert rows == sorted(rows, key=lambda x: -x["value"])
    assert any(row["label"].startswith("Idle") for row in r.error_budget())


def test_result_metrics_and_summary():
    r = _noisy_result()
    ideal = nq.simulate(r.circuit, shots=3000, seed=1)
    cmp = r.compare(ideal)
    assert 0.5 < cmp["hellinger_fidelity"] <= 1
    assert 0 < cmp["total_variation_distance"] < 0.4
    assert "fault-free" in r.summary()
    assert r.output_error_rate().value <= 1 - r.fault_free_fraction.value + 0.05
    assert r.marginal_counts([0]) and r.most_frequent() in ("0000", "1111")
    assert r.qubit_error_rates().shape == (4, 4)
    assert repr(r).startswith("Result(")


def test_density_backend_has_no_fault_stats():
    r = nq.simulate(nq.algorithms.bell(), shots=10, backend="density_matrix")
    with pytest.raises(ValueError):
        r.fault_rates()
    with pytest.raises(ValueError):
        r.output_error_rate()
    assert r.probabilities_exact is not None


# ----------------------------------------------------------------------------- catalog
def test_explain_entries_and_synonyms(capsys):
    for key in nq.catalog.CATALOG:
        text = nq.explain(key)
        assert nq.catalog.CATALOG[key].title in text
    assert "Amplitude damping" in nq.explain("T1")
    assert "Thermal relaxation" in nq.explain(nq.noise.ThermalRelaxation(1, 1, 0.1))
    assert "depolarizing" in nq.explain(depolarizing(0.1)).lower()
    nq.explain()
    with pytest.raises(KeyError):
        nq.explain("definitely not a thing")
    capsys.readouterr()


def test_every_channel_class_has_a_catalog_entry():
    for cls in (
        nq.noise.PauliChannel,
        nq.noise.KrausChannel,
        nq.noise.AmplitudeDamping,
        nq.noise.PhaseDamping,
        nq.noise.ThermalRelaxation,
        nq.noise.CoherentRotation,
        nq.noise.ReadoutError,
    ):
        assert cls.catalog_key in nq.catalog.CATALOG


def test_docs_math_page_is_up_to_date():
    path = Path(__file__).resolve().parents[1] / "docs" / "math.md"
    assert path.exists(), "run: python -m noisiq.catalog docs/math.md"
    assert path.read_text(encoding="utf-8") == nq.catalog.to_markdown(), "regenerate with: python -m noisiq.catalog"
