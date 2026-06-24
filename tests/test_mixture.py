"""Tests for the general (multi-salt) composition + structure machinery."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.composition import (
    parse_formula,
    atomic_mass,
    Component,
    Mixture,
    mixture_from_mole_percent,
    box_length_for_density,
    density_from_box,
    ATOMIC_MASS,
)
from saltmd import presets
from saltmd.structure import build_box, write_pdb


def test_parse_formula():
    assert parse_formula("LiF") == {"Li": 1, "F": 1}
    assert parse_formula("BeF2") == {"Be": 1, "F": 2}
    assert parse_formula("MgCl2") == {"Mg": 1, "Cl": 2}
    assert parse_formula("Al2O3") == {"Al": 2, "O": 3}
    with pytest.raises(ValueError):
        parse_formula("")
    with pytest.raises(ValueError):
        parse_formula("2LiF")  # leading number not a valid element token


def test_curated_masses_match_ase():
    """The hand-curated table must agree with ase's standard atomic weights."""
    ase = pytest.importorskip("ase.data")
    for sym, mass in ATOMIC_MASS.items():
        ref = ase.atomic_masses[ase.atomic_numbers[sym]]
        assert mass == pytest.approx(ref, abs=0.05), f"{sym}: {mass} vs ase {ref}"


def test_flinak_ternary_counts_and_fractions():
    mix = mixture_from_mole_percent([("LiF", 46.5), ("NaF", 11.5), ("KF", 42.0)], 700)
    assert mix.n_formula_units == 700                     # counts sum exactly to N
    ec = mix.element_counts()
    # Every F comes from one of the three fluorides -> F == total formula units.
    assert ec["F"] == 700
    assert ec["Li"] + ec["Na"] + ec["K"] == 700           # one cation per unit
    fr = mix.mole_fractions()
    assert fr["LiF"] == pytest.approx(0.465, abs=0.01)
    assert fr["KF"] == pytest.approx(0.42, abs=0.01)


def test_nacl_mgcl2_chlorine_balance():
    mix = mixture_from_mole_percent([("NaCl", 58.0), ("MgCl2", 42.0)], 720)
    ec = mix.element_counts()
    # Cl = 1 per NaCl + 2 per MgCl2.
    n_nacl = dict(zip([c.name for c in mix.components], mix.counts))["NaCl"]
    n_mgcl2 = dict(zip([c.name for c in mix.components], mix.counts))["MgCl2"]
    assert ec["Cl"] == n_nacl + 2 * n_mgcl2
    assert ec["Na"] == n_nacl
    assert ec["Mg"] == n_mgcl2


def test_largest_remainder_preserves_total():
    # Awkward fractions must still sum to exactly N (no drift).
    for n in (100, 333, 720, 1001):
        mix = mixture_from_mole_percent([("LiF", 33.333), ("NaF", 33.333), ("KF", 33.334)], n)
        assert mix.n_formula_units == n


def test_density_roundtrip_general():
    mix = mixture_from_mole_percent([("NaCl", 50), ("KCl", 50)], 512)
    for rho in (1.4, 1.6, 1.8):
        L = box_length_for_density(mix, rho)
        assert density_from_box(mix, L) == pytest.approx(rho, rel=1e-9)


def test_build_box_general_elements(tmp_path):
    mix = mixture_from_mole_percent([("LiF", 46.5), ("NaF", 11.5), ("KF", 42.0)], 700)
    built = build_box(mix, initial_density_g_cm3=1.9, seed=1)
    assert len(built.elements) == mix.n_atoms
    assert set(built.elements) == {"Li", "Na", "K", "F"}
    path = str(tmp_path / "flinak.pdb")
    write_pdb(built, path)
    atoms = [l for l in open(path) if l.startswith("ATOM")]
    assert len(atoms) == mix.n_atoms
    # Two-letter element symbols (Li, Na) must sit right-justified in cols 77-78.
    assert {l[76:78].strip() for l in atoms} == {"Li", "Na", "K", "F"}
    assert all(len(l.rstrip("\n")) == 78 for l in atoms)


def test_presets_have_valid_components():
    for key, preset in presets.PRESETS.items():
        mix = mixture_from_mole_percent(preset.components, 720)
        assert mix.n_atoms > 0
        assert mix.molar_mass_g_per_mol > 0
    # Only Flibe ships with a model.
    assert presets.PRESETS["flibe"].has_bundled_model is True
    assert presets.PRESETS["flinak"].has_bundled_model is False
    with pytest.raises(KeyError):
        presets.get_preset("not-a-salt")
