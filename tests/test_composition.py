"""Tests for composition <-> geometry math."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.composition import (
    counts_from_mole_percent,
    box_length_for_density,
    density_from_box,
)


def test_eutectic_counts_match_reference_structure():
    # 33.33 mol% BeF2 with 720 formula units must reproduce the shipped Flibe box:
    # 480 LiF + 240 BeF2 -> Li=480, Be=240, F=960, 1680 atoms.
    comp = counts_from_mole_percent(33.33, 720)
    assert comp.n_LiF == 480
    assert comp.n_BeF2 == 240
    assert comp.element_counts() == {"F": 960, "Li": 480, "Be": 240}
    assert comp.n_atoms == 1680
    assert comp.mole_percent_BeF2 == pytest.approx(33.3333, abs=1e-3)


def test_fluorine_count_balances():
    # Each LiF has 1 F, each BeF2 has 2 F, for any composition.
    for x in (0, 10, 25, 50, 75, 100):
        comp = counts_from_mole_percent(x, 500)
        assert comp.n_F == comp.n_LiF + 2 * comp.n_BeF2


def test_density_box_roundtrip():
    comp = counts_from_mole_percent(33.33, 720)
    for rho in (1.8, 1.94, 2.1):
        L = box_length_for_density(comp, rho)
        assert density_from_box(comp, L) == pytest.approx(rho, rel=1e-9)


def test_reference_box_density():
    # The shipped 26.846 A box should sit near ~2.0 g/cm^3 (the initial pack).
    comp = counts_from_mole_percent(33.33, 720)
    rho = density_from_box(comp, 26.846)
    assert 1.95 < rho < 2.10


def test_invalid_inputs():
    with pytest.raises(ValueError):
        counts_from_mole_percent(150, 100)
    with pytest.raises(ValueError):
        counts_from_mole_percent(33, 0)
    with pytest.raises(ValueError):
        box_length_for_density(counts_from_mole_percent(33, 100), 0)
