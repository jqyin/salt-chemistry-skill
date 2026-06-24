"""Named molten-salt presets (component formulas + approximate eutectic mol%).

These are convenience starting points so an agent can say "FLiNaK" instead of
spelling out components. The compositions are *approximate eutectics* from the
literature and are meant to be overridden when a specific composition is wanted —
the whole point of this skill is to scan composition.

IMPORTANT: a preset only describes the *chemistry to build a box for*. Actually
*simulating* it requires a machine-learning potential trained on that chemistry.
Only Flibe (LiF-BeF2) ships with a potential here (``assets/mace_flibe.model``);
the others are listed so structures and campaigns can be prepared, but you must
supply a matching MACE/ML model via ``--model`` to run them. ``has_bundled_model``
records this.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SaltPreset:
    key: str
    description: str
    components: list[tuple[str, float]]  # (formula, mole_percent)
    has_bundled_model: bool = False


PRESETS: dict[str, SaltPreset] = {
    "flibe": SaltPreset(
        "flibe", "Flibe, the 2LiF.BeF2 fusion coolant/breeder eutectic",
        [("LiF", 66.67), ("BeF2", 33.33)], has_bundled_model=True,
    ),
    "flinak": SaltPreset(
        "flinak", "FLiNaK, the LiF-NaF-KF eutectic (MSR coolant)",
        [("LiF", 46.5), ("NaF", 11.5), ("KF", 42.0)],
    ),
    "licl-kcl": SaltPreset(
        "licl-kcl", "LiCl-KCl eutectic (pyroprocessing / thermal storage)",
        [("LiCl", 59.0), ("KCl", 41.0)],
    ),
    "nacl-kcl": SaltPreset(
        "nacl-kcl", "NaCl-KCl (~equimolar)",
        [("NaCl", 50.0), ("KCl", 50.0)],
    ),
    "nacl-mgcl2": SaltPreset(
        "nacl-mgcl2", "NaCl-MgCl2 eutectic (next-gen CSP / fast-reactor coolant)",
        [("NaCl", 58.0), ("MgCl2", 42.0)],
    ),
    "kcl-mgcl2": SaltPreset(
        "kcl-mgcl2", "KCl-MgCl2 eutectic",
        [("KCl", 68.0), ("MgCl2", 32.0)],
    ),
}


def get_preset(key: str) -> SaltPreset:
    k = key.strip().lower()
    if k not in PRESETS:
        raise KeyError(
            f"Unknown salt preset '{key}'. Known: {', '.join(sorted(PRESETS))}. "
            f"Or pass components explicitly, e.g. --components 'LiF:66.67,BeF2:33.33'."
        )
    return PRESETS[k]
