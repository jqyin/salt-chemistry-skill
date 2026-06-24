"""Composition <-> geometry math for molten-salt mixtures.

A molten salt here is a **mixture of ionic compounds** ("components"), each given
by a chemical formula and a mole fraction — for example:

  * Flibe        : LiF 66.67% + BeF2 33.33%   (the 2LiF.BeF2 eutectic)
  * FLiNaK       : LiF 46.5% + NaF 11.5% + KF 42%
  * LiCl-KCl     : LiCl 59% + KCl 41%

The general types are :class:`Component` and :class:`Mixture`. From a mixture and a
target density this module sizes the cubic box (and vice versa) — everything the
structure builder and analysis need, for any chemistry.

It is pure Python (standard library only) so it imports anywhere — login nodes,
laptops, notebooks — without OpenMM/PyTorch/MACE. Atomic masses come from a curated
table of common salt-forming elements; anything outside it falls back to ``ase``
if installed. (The table is checked against ``ase`` in the test suite.)

Backward compatibility: the original Flibe-only API — :class:`Composition` and
:func:`counts_from_mole_percent` — is preserved as a thin wrapper over the general
machinery, so existing scripts and saved workflows keep working.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Avogadro constant (1/mol), exact under the 2019 SI redefinition.
AVOGADRO = 6.02214076e23
# 1 Angstrom^3 = 1e-24 cm^3.
CM3_PER_ANGSTROM3 = 1.0e-24

# Curated standard atomic weights (g/mol) for elements common in molten salts
# (alkali/alkaline-earth halides, oxides, and MSR-relevant actinides/metals).
# Validated against ase.data in tests/test_composition.py. Extend as needed;
# unknown symbols fall back to ase if it is importable.
ATOMIC_MASS = {
    "H": 1.008, "Li": 6.941, "Be": 9.0122, "B": 10.811, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998403, "Na": 22.98977, "Mg": 24.305,
    "Al": 26.981539, "Si": 28.085, "P": 30.973762, "S": 32.06, "Cl": 35.45,
    "K": 39.0983, "Ca": 40.078, "Ti": 47.867, "Cr": 51.9961, "Mn": 54.938044,
    "Fe": 55.845, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Rb": 85.4678,
    "Sr": 87.62, "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95,
    "Br": 79.904, "I": 126.90447, "Cs": 132.90545, "Ba": 137.327,
    "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242,
    "Sm": 150.36, "Gd": 157.25, "Th": 232.0377, "U": 238.02891, "Pu": 244.0642,
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def atomic_mass(symbol: str) -> float:
    """Standard atomic weight (g/mol) for an element symbol.

    Uses the curated :data:`ATOMIC_MASS` table, falling back to ``ase.data`` for
    elements outside it. Raises ``KeyError`` with a clear message if neither knows
    the element.
    """
    if symbol in ATOMIC_MASS:
        return ATOMIC_MASS[symbol]
    try:
        from ase.data import atomic_masses, atomic_numbers
        return float(atomic_masses[atomic_numbers[symbol]])
    except Exception as exc:  # noqa: BLE001 - want a single clear error
        raise KeyError(
            f"No atomic mass for element '{symbol}'. Add it to "
            f"saltmd.composition.ATOMIC_MASS, or install ase."
        ) from exc


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple chemical formula into {element: count}.

    Handles the element+count formulas of common ionic salts, e.g. ``"BeF2"`` ->
    ``{"Be": 1, "F": 2}``, ``"Al2O3"`` -> ``{"Al": 2, "O": 3}``. Nested groups /
    parentheses (hydrates, complexes) are not supported.
    """
    formula = formula.strip()
    if not formula:
        raise ValueError("Empty chemical formula")
    counts: dict[str, int] = {}
    pos = 0
    for m in _FORMULA_TOKEN.finditer(formula):
        if m.start() != pos:
            raise ValueError(f"Could not parse formula '{formula}' near index {pos}")
        element, num = m.group(1), m.group(2)
        if not element:
            break
        counts[element] = counts.get(element, 0) + (int(num) if num else 1)
        pos = m.end()
    if pos != len(formula) or not counts:
        raise ValueError(f"Could not fully parse chemical formula '{formula}'")
    return counts


@dataclass(frozen=True)
class Component:
    """One ionic compound in a salt mixture (a formula + its element counts)."""

    name: str
    stoichiometry: dict[str, int]

    @classmethod
    def from_formula(cls, formula: str) -> "Component":
        return cls(name=formula, stoichiometry=parse_formula(formula))

    @property
    def molar_mass(self) -> float:
        return sum(atomic_mass(e) * n for e, n in self.stoichiometry.items())


@dataclass
class Mixture:
    """A salt mixture: components with integer formula-unit counts in the box.

    Construct via :func:`mixture_from_mole_percent` so mole fractions are converted
    to integer counts consistently (and the achieved fractions reported).
    """

    components: list[Component]
    counts: list[int]

    def __post_init__(self):
        if len(self.components) != len(self.counts):
            raise ValueError("components and counts must have the same length")

    @property
    def n_formula_units(self) -> int:
        return sum(self.counts)

    def element_counts(self) -> dict[str, int]:
        """Total atom count per element across all components."""
        out: dict[str, int] = {}
        for comp, n in zip(self.components, self.counts):
            for el, k in comp.stoichiometry.items():
                out[el] = out.get(el, 0) + k * n
        return out

    @property
    def n_atoms(self) -> int:
        return sum(self.element_counts().values())

    @property
    def molar_mass_g_per_mol(self) -> float:
        return sum(c.molar_mass * n for c, n in zip(self.components, self.counts))

    @property
    def mass_g(self) -> float:
        """Total mass of all atoms in the box, in grams."""
        return self.molar_mass_g_per_mol / AVOGADRO

    def mole_fractions(self) -> dict[str, float]:
        """Component name -> achieved mole fraction (sums to 1)."""
        n = self.n_formula_units
        if n == 0:
            return {c.name: 0.0 for c in self.components}
        return {c.name: cnt / n for c, cnt in zip(self.components, self.counts)}

    def mole_percent(self, component_name: str) -> float:
        return 100.0 * self.mole_fractions().get(component_name, 0.0)

    def as_dict(self) -> dict:
        """Serializable summary for provenance / results.json."""
        fracs = self.mole_fractions()
        return {
            "components": [
                {
                    "name": c.name,
                    "count": cnt,
                    "mole_percent": round(100.0 * fracs[c.name], 4),
                }
                for c, cnt in zip(self.components, self.counts)
            ],
            "n_formula_units": self.n_formula_units,
            "element_counts": self.element_counts(),
            "n_atoms": self.n_atoms,
            "molar_mass_g_per_mol": round(self.molar_mass_g_per_mol, 6),
        }


def _largest_remainder(fractions: list[float], total: int) -> list[int]:
    """Round real-valued shares to integers summing exactly to ``total``.

    Floor each ideal share, then hand the leftover units to the largest fractional
    remainders. This keeps the integer counts as close as possible to the requested
    mole fractions without the total drifting off ``n_formula_units``.
    """
    ideals = [f * total for f in fractions]
    floors = [int(x) for x in ideals]
    remainder = total - sum(floors)
    order = sorted(range(len(fractions)), key=lambda i: ideals[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def mixture_from_mole_percent(
    specs: list[tuple[str, float]],
    n_formula_units: int,
) -> Mixture:
    """Build a :class:`Mixture` from (formula, mol%) pairs and a total size.

    Parameters
    ----------
    specs : list of (formula, mole_percent)
        e.g. ``[("LiF", 66.67), ("BeF2", 33.33)]`` or a ternary like
        ``[("LiF", 46.5), ("NaF", 11.5), ("KF", 42)]``. Percentages should sum to
        ~100; they are normalized to exactly 1 before rounding to integer counts.
    n_formula_units : int
        Total number of formula units in the box (>= number of components).
    """
    if not specs:
        raise ValueError("Provide at least one (formula, mole_percent) component")
    if n_formula_units < len(specs):
        raise ValueError(
            f"n_formula_units ({n_formula_units}) must be >= number of components ({len(specs)})"
        )
    percents = [p for _, p in specs]
    if any(p < 0 for p in percents):
        raise ValueError("mole percentages must be non-negative")
    total_pct = sum(percents)
    if total_pct <= 0:
        raise ValueError("mole percentages must sum to a positive value")

    fractions = [p / total_pct for p in percents]
    counts = _largest_remainder(fractions, n_formula_units)
    components = [Component.from_formula(formula) for formula, _ in specs]
    return Mixture(components=components, counts=counts)


def box_length_for_density(mixture, density_g_cm3: float) -> float:
    """Edge length (Angstrom) of the cubic box giving ``density_g_cm3``.

    Accepts any object exposing ``mass_g`` (a :class:`Mixture` or the legacy
    :class:`Composition`). Initializing near the expected liquid density keeps the
    barostat's first adjustment small and avoids overlap-induced force blowups.
    """
    if density_g_cm3 <= 0.0:
        raise ValueError(f"density must be > 0, got {density_g_cm3}")
    volume_cm3 = mixture.mass_g / density_g_cm3
    volume_angstrom3 = volume_cm3 / CM3_PER_ANGSTROM3
    return volume_angstrom3 ** (1.0 / 3.0)


def density_from_box(mixture, box_length_angstrom: float) -> float:
    """Mass density (g/cm^3) of ``mixture`` in a cubic box of the given edge length."""
    if box_length_angstrom <= 0.0:
        raise ValueError(f"box length must be > 0, got {box_length_angstrom}")
    volume_cm3 = (box_length_angstrom ** 3) * CM3_PER_ANGSTROM3
    return mixture.mass_g / volume_cm3


# ---------------------------------------------------------------------------
# Backward-compatible Flibe (LiF-BeF2) API.
#
# These wrap the general Mixture machinery so existing callers keep working and
# the convenient "mol% BeF2" axis is preserved for the canonical coolant salt.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Composition:
    """LiF-BeF2 composition (legacy Flibe-specific convenience over Mixture)."""

    n_LiF: int
    n_BeF2: int

    def _mixture(self) -> Mixture:
        return Mixture(
            components=[Component("LiF", {"Li": 1, "F": 1}),
                        Component("BeF2", {"Be": 1, "F": 2})],
            counts=[self.n_LiF, self.n_BeF2],
        )

    @property
    def n_formula_units(self) -> int:
        return self.n_LiF + self.n_BeF2

    @property
    def n_Li(self) -> int:
        return self.n_LiF

    @property
    def n_Be(self) -> int:
        return self.n_BeF2

    @property
    def n_F(self) -> int:
        return self.n_LiF + 2 * self.n_BeF2

    @property
    def n_atoms(self) -> int:
        return self.n_Li + self.n_Be + self.n_F

    def element_counts(self) -> dict[str, int]:
        return {"F": self.n_F, "Li": self.n_Li, "Be": self.n_Be}

    @property
    def mole_percent_BeF2(self) -> float:
        if self.n_formula_units == 0:
            return 0.0
        return 100.0 * self.n_BeF2 / self.n_formula_units

    @property
    def molar_mass_g_per_mol(self) -> float:
        return self._mixture().molar_mass_g_per_mol

    @property
    def mass_g(self) -> float:
        return self._mixture().mass_g

    def as_dict(self) -> dict:
        return {
            "n_LiF": self.n_LiF,
            "n_BeF2": self.n_BeF2,
            "n_formula_units": self.n_formula_units,
            "mole_percent_BeF2": round(self.mole_percent_BeF2, 4),
            "element_counts": self.element_counts(),
            "n_atoms": self.n_atoms,
            "molar_mass_g_per_mol": round(self.molar_mass_g_per_mol, 6),
        }


def counts_from_mole_percent(mole_percent_BeF2: float, n_formula_units: int) -> Composition:
    """Flibe convenience: build a :class:`Composition` from mol% BeF2 and box size."""
    if not 0.0 <= mole_percent_BeF2 <= 100.0:
        raise ValueError(f"mole_percent_BeF2 must be in [0, 100], got {mole_percent_BeF2}")
    if n_formula_units < 1:
        raise ValueError(f"n_formula_units must be >= 1, got {n_formula_units}")
    n_BeF2 = round(mole_percent_BeF2 / 100.0 * n_formula_units)
    n_LiF = n_formula_units - n_BeF2
    return Composition(n_LiF=n_LiF, n_BeF2=n_BeF2)
