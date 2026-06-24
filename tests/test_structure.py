"""Tests for structure building and PDB round-tripping."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.composition import counts_from_mole_percent, density_from_box
from saltmd.structure import build_box, write_pdb, min_image_min_distance


def test_build_box_has_correct_atoms_and_density():
    comp = counts_from_mole_percent(33.33, 720)
    built = build_box(comp, initial_density_g_cm3=1.9, seed=0)
    assert len(built.elements) == comp.n_atoms
    assert built.positions.shape == (comp.n_atoms, 3)
    assert built.elements.count("F") == comp.n_F
    assert built.elements.count("Li") == comp.n_Li
    assert built.elements.count("Be") == comp.n_Be
    # Box was sized to the requested initial density.
    assert density_from_box(comp, built.box_length) == pytest.approx(1.9, rel=1e-6)
    # Positions are wrapped inside the box.
    assert built.positions.min() >= 0.0
    assert built.positions.max() < built.box_length


def test_build_box_is_reproducible():
    comp = counts_from_mole_percent(50, 400)
    a = build_box(comp, seed=42)
    b = build_box(comp, seed=42)
    assert np.allclose(a.positions, b.positions)


def test_no_hard_overlaps():
    comp = counts_from_mole_percent(33.33, 720)
    built = build_box(comp, initial_density_g_cm3=1.9, seed=0)
    # Jittered lattice should keep ions comfortably apart for the minimizer.
    assert min_image_min_distance(built.positions, built.box_length) > 1.0


def test_pdb_format_is_openmm_compatible(tmp_path):
    comp = counts_from_mole_percent(33.33, 720)
    built = build_box(comp, seed=0)
    path = str(tmp_path / "structure.pdb")
    write_pdb(built, path)
    lines = open(path).read().splitlines()
    assert lines[0].startswith("CRYST1")
    atoms = [l for l in lines if l.startswith("ATOM")]
    assert len(atoms) == comp.n_atoms
    # Element column (77-78) must carry the real element — that's what MACE uses.
    elements = {l[76:78].strip() for l in atoms}
    assert elements == {"F", "Li", "Be"}
    # Each ATOM line is the fixed-width 78-char record OpenMM expects.
    assert all(len(l) == 78 for l in atoms)
