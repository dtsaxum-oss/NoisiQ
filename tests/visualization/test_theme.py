import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

from noisiq.visualization.theme import (
    CLIFFORD_GATE_COLOR,
    T_GATE_COLOR,
    OTHER_GATE_COLOR,
    WIRE_COLOR,
    ERROR_COLOR,
    HALO_ZERO_COLOR,
    HALO_MAX_COLOR,
    CLIFFORD_GATES,
    T_GATES,
    gate_color,
    halo_color,
    get_halo_colormap,
    apply_global_style,
)


# ---------------------------------------------------------------------------
# gate_color
# ---------------------------------------------------------------------------

def test_clifford_gates_return_clifford_color():
    for name in ("H", "X", "Y", "Z", "S", "CNOT", "CX", "CZ", "I"):
        assert gate_color(name) == CLIFFORD_GATE_COLOR, f"Failed for {name}"

def test_t_gates_return_t_color():
    for name in ("T", "T†", "T_DAG", "TDG"):
        assert gate_color(name) == T_GATE_COLOR, f"Failed for {name}"

def test_unknown_gate_returns_other_color():
    for name in ("RZ", "RX", "U3", "CUSTOM", "MYCOOLGATE"):
        assert gate_color(name) == OTHER_GATE_COLOR, f"Failed for {name}"

def test_gate_color_case_insensitive():
    assert gate_color("h") == gate_color("H")
    assert gate_color("cnot") == gate_color("CNOT")
    assert gate_color("t") == gate_color("T")

def test_gate_color_returns_string():
    assert isinstance(gate_color("H"), str)
    assert isinstance(gate_color("T"), str)
    assert isinstance(gate_color("RZ"), str)


# ---------------------------------------------------------------------------
# halo_color
# ---------------------------------------------------------------------------

def test_halo_color_returns_tuple_of_four():
    rgba = halo_color(0.5)
    assert len(rgba) == 4

def test_halo_color_alpha_is_one():
    # Colormap returns fully opaque RGBA
    assert halo_color(0.0)[3] == pytest.approx(1.0)
    assert halo_color(1.0)[3] == pytest.approx(1.0)

def test_halo_color_zero_is_bluish():
    r, g, b, _ = halo_color(0.0)
    # Light blue: blue channel should dominate over red
    assert b > r

def test_halo_color_one_is_reddish():
    r, g, b, _ = halo_color(1.0)
    # Bright red: red channel should dominate
    assert r > b

def test_halo_color_clips_below_zero():
    # Should not crash and should equal halo_color(0.0)
    assert halo_color(-1.0) == halo_color(0.0)

def test_halo_color_clips_above_one():
    assert halo_color(2.0) == halo_color(1.0)

def test_halo_color_midpoint_between_extremes():
    r0, g0, b0, _ = halo_color(0.0)
    r1, g1, b1, _ = halo_color(1.0)
    rm, gm, bm, _ = halo_color(0.5)
    # Midpoint red channel should be between the two extremes
    assert min(r0, r1) <= rm <= max(r0, r1)


# ---------------------------------------------------------------------------
# get_halo_colormap
# ---------------------------------------------------------------------------

def test_get_halo_colormap_returns_colormap():
    import matplotlib.colors as mcolors
    cmap = get_halo_colormap()
    assert isinstance(cmap, mcolors.Colormap)

def test_halo_colormap_name():
    cmap = get_halo_colormap()
    assert "noisiq" in cmap.name.lower()


# ---------------------------------------------------------------------------
# Constants are valid hex colors
# ---------------------------------------------------------------------------

def test_color_constants_are_valid_hex():
    import matplotlib.colors as mcolors
    for name, val in [
        ("WIRE_COLOR", WIRE_COLOR),
        ("ERROR_COLOR", ERROR_COLOR),
        ("CLIFFORD_GATE_COLOR", CLIFFORD_GATE_COLOR),
        ("T_GATE_COLOR", T_GATE_COLOR),
        ("OTHER_GATE_COLOR", OTHER_GATE_COLOR),
        ("HALO_ZERO_COLOR", HALO_ZERO_COLOR),
        ("HALO_MAX_COLOR", HALO_MAX_COLOR),
    ]:
        assert mcolors.is_color_like(val), f"{name}={val!r} is not a valid color"


# ---------------------------------------------------------------------------
# Gate category sets
# ---------------------------------------------------------------------------

def test_clifford_and_t_sets_are_disjoint():
    overlap = CLIFFORD_GATES & T_GATES
    assert len(overlap) == 0, f"Overlap between Clifford and T sets: {overlap}"

def test_clifford_gates_set_is_frozenset():
    assert isinstance(CLIFFORD_GATES, frozenset)

def test_t_gates_set_is_frozenset():
    assert isinstance(T_GATES, frozenset)


# ---------------------------------------------------------------------------
# apply_global_style (smoke)
# ---------------------------------------------------------------------------

def test_apply_global_style_does_not_crash():
    apply_global_style()


def teardown_module(module):
    plt.close("all")
