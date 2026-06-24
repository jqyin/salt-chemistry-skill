"""Build a randomized periodic simulation box for a molten-salt melt.

Given a composition — any object exposing ``element_counts()``, ``n_atoms`` and
``mass_g`` (a :class:`~saltmd.composition.Mixture` or the legacy ``Composition``) —
and a target initial density, this produces a cubic, fully periodic box of ions
placed on a *jittered* simple-cubic lattice. We deliberately do not try to build a
physically equilibrated liquid here — that is the job of the NPT
minimize/equilibrate phases. We only need a starting configuration that is:

  * at roughly the right density (so the barostat makes a small correction), and
  * free of hard overlaps (so the ML potential does not see two ions on top of
    each other and produce enormous forces on step 0).

A jittered lattice satisfies both: it guarantees a minimum interparticle spacing
while randomizing which lattice site each species occupies, and a short
equilibration melts the lattice into a disordered liquid.

This module uses only numpy (no OpenMM), so structure generation can run on a
login node or laptop while the GPU job waits in the queue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .composition import box_length_for_density


@dataclass
class BuiltStructure:
    """Result of :func:`build_box`: coordinates plus metadata for provenance."""

    elements: list[str]          # per-atom element symbols, length n_atoms
    positions: np.ndarray        # (n_atoms, 3) in Angstrom, wrapped into the box
    box_length: float            # cubic edge length in Angstrom
    composition: object          # the Mixture/Composition that was built
    initial_density_g_cm3: float
    seed: int


def _atom_name(element: str, index: int) -> str:
    """A cosmetic 4-char PDB atom name. The element column (77-78) is authoritative,
    so collisions here are harmless; we just keep names readable and in-spec."""
    return f"{element}{index}"[:4]


def build_box(
    comp,
    initial_density_g_cm3: float = 1.9,
    seed: int = 0,
    jitter_fraction: float = 0.2,
) -> BuiltStructure:
    """Place all ions of ``comp`` on a jittered cubic lattice at a target density.

    Parameters
    ----------
    comp : Mixture or Composition
        Anything exposing ``element_counts()``, ``n_atoms``, and ``mass_g``.
    initial_density_g_cm3 : float
        Density used to size the cubic box. A slightly *low* value (loose pack)
        is safer than too high, because NPT can compress quickly but a too-dense
        start can create overlaps. ~1.9 g/cm^3 is a good Flibe starting point;
        chloride salts are lighter (~1.5-1.7), so adjust per chemistry.
    seed : int
        RNG seed; makes structure generation fully reproducible.
    jitter_fraction : float
        Random displacement of each ion off its lattice site, as a fraction of
        the lattice spacing (0 = perfect lattice, 0.5 = sites can nearly touch).
        0.2 keeps a healthy minimum separation while randomizing the structure.
    """
    rng = np.random.default_rng(seed)

    element_counts = comp.element_counts()
    n_atoms = comp.n_atoms
    box_length = box_length_for_density(comp, initial_density_g_cm3)

    # Smallest simple-cubic grid that holds every atom.
    n_per_side = int(np.ceil(n_atoms ** (1.0 / 3.0)))
    spacing = box_length / n_per_side

    # Generate all grid sites, shuffle, and take the first n_atoms.
    grid = np.arange(n_per_side)
    sites = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    rng.shuffle(sites)
    sites = sites[:n_atoms]

    # Center each ion in its cell, then jitter.
    positions = (sites + 0.5) * spacing
    max_jitter = jitter_fraction * spacing
    positions += rng.uniform(-max_jitter, max_jitter, size=positions.shape)

    # Wrap back into [0, box_length) so nothing sits outside the periodic box.
    positions = np.mod(positions, box_length)

    # Assign species to the (already shuffled) sites in a deterministic element
    # order. Order is irrelevant to OpenMM/MACE (the element column drives the
    # potential) but a fixed order keeps generated PDBs diff-friendly.
    elements: list[str] = []
    for element in sorted(element_counts):
        elements.extend([element] * element_counts[element])
    assert len(elements) == n_atoms

    return BuiltStructure(
        elements=elements,
        positions=positions,
        box_length=box_length,
        composition=comp,
        initial_density_g_cm3=initial_density_g_cm3,
        seed=seed,
    )


def min_image_min_distance(positions: np.ndarray, box_length: float) -> float:
    """Smallest pairwise distance under the minimum-image convention (Angstrom).

    Used as a sanity check that the generated structure has no hard overlaps
    before it is handed to the (expensive) GPU job. O(N^2); fine for the few
    thousand atoms in these boxes.
    """
    n = len(positions)
    if n < 2:
        return float("inf")
    best = float("inf")
    for i in range(n - 1):
        d = positions[i + 1:] - positions[i]
        d -= box_length * np.round(d / box_length)  # minimum image
        dist = np.sqrt((d * d).sum(axis=1)).min()
        best = min(best, float(dist))
    return best


def write_pdb(built: BuiltStructure, path: str) -> None:
    """Write a single-residue PDB with a CRYST1 box record.

    The column layout reproduces exactly what OpenMM's ``PDBFile`` reader
    expects (record/serial/name/resName/coords/element columns), with the
    element symbol in columns 77-78 — that element column is what the MACE
    ``MLPotential`` uses to build the system, so it must be correct.
    """
    L = built.box_length
    lines = [
        # CRYST1: cubic box, P 1 space group, Z=1. Widths match the PDB spec.
        f"CRYST1{L:9.3f}{L:9.3f}{L:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1           1"
    ]

    serial = 0
    # Per-element running index for cosmetic atom names (F1, F2, ... / Na1, ...).
    per_elem_idx: dict[str, int] = {}
    for element, (x, y, z) in zip(built.elements, built.positions):
        serial += 1
        per_elem_idx[element] = per_elem_idx.get(element, 0) + 1
        name = _atom_name(element, per_elem_idx[element])
        lines.append(
            f"ATOM  {serial:>5d} {name:<4s} MOL A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"          {element:>2s}"
        )
    lines.append("END")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
