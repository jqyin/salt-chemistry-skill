#!/usr/bin/env python3
"""Build a molten-salt simulation box at a target composition and density.

Turns a requested composition into a periodic PDB ready for NPT MD. Composition
can be given three ways (pick one):

  * ``--salt <preset>``               e.g. flibe, flinak, licl-kcl (see --list-salts)
  * ``--components "LiF:66.67,BeF2:33.33"``   explicit formula:mol% pairs
  * ``--mol-percent-bef2 33.33``      Flibe shortcut (LiF + BeF2)

Prints a JSON summary (achieved component mol%, element counts, box size,
closest-pair distance) so an agent can verify the box before launching a GPU job.

NOTE: building a box does not require a potential, but *simulating* it does. Only
Flibe ships with a model (assets/mace_flibe.model); other chemistries need a
matching MACE/ML model passed to run_npt.py via --model.

Examples
--------
    # Flibe eutectic (33.33 mol% BeF2)
    python scripts/build_structure.py --mol-percent-bef2 33.33 -o structure.pdb

    # FLiNaK eutectic from a preset
    python scripts/build_structure.py --salt flinak --n-formula-units 700 -o flinak.pdb

    # An explicit ternary, lighter chloride start density
    python scripts/build_structure.py --components "NaCl:58,MgCl2:42" \
        --density 1.6 -o naclmgcl2.pdb
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.composition import mixture_from_mole_percent, density_from_box
from saltmd.structure import build_box, write_pdb, min_image_min_distance
from saltmd import presets


def _parse_components(spec: str) -> list[tuple[str, float]]:
    """Parse 'LiF:66.67,BeF2:33.33' -> [('LiF',66.67),('BeF2',33.33)]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Component '{part}' must be FORMULA:MOLEPERCENT")
        formula, pct = part.rsplit(":", 1)
        out.append((formula.strip(), float(pct)))
    if not out:
        raise ValueError("No components parsed")
    return out


def _resolve_specs(args) -> tuple[list[tuple[str, float]], str | None]:
    """Return (component specs, preset key or None) from the chosen input mode."""
    if args.components:
        return _parse_components(args.components), None
    if args.salt:
        p = presets.get_preset(args.salt)
        return list(p.components), p.key
    if args.mol_percent_bef2 is not None:
        x = args.mol_percent_bef2
        return [("LiF", 100.0 - x), ("BeF2", x)], "flibe"
    raise SystemExit(
        "Specify a composition: --salt <preset>, --components 'LiF:66.67,BeF2:33.33', "
        "or --mol-percent-bef2 <x>. (Use --list-salts to see presets.)"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build a molten-salt periodic box at a target composition and density.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--salt", help="Named preset (flibe, flinak, licl-kcl, ...). See --list-salts.")
    p.add_argument("--components", help="Explicit 'FORMULA:MOLEPERCENT,...' e.g. 'LiF:66.67,BeF2:33.33'.")
    p.add_argument("--mol-percent-bef2", type=float, default=None,
                   help="Flibe shortcut: mol%% BeF2 in [0,100] (33.33 = eutectic).")
    p.add_argument("--n-formula-units", type=int, default=720,
                   help="Total formula units across all components (sets system size/cost).")
    p.add_argument("--density", type=float, default=1.9,
                   help="Initial density (g/cm^3) used to size the box; NPT relaxes it. "
                        "~1.9 for fluorides; ~1.5-1.7 for chlorides.")
    p.add_argument("--jitter-fraction", type=float, default=0.2,
                   help="Lattice jitter as a fraction of spacing (randomizes the start).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible structures).")
    p.add_argument("-o", "--output", default="structure.pdb", help="Output PDB path.")
    p.add_argument("--min-distance", type=float, default=1.2,
                   help="Warn if the closest ion pair is below this (Angstrom).")
    p.add_argument("--list-salts", action="store_true", help="List known salt presets and exit.")
    args = p.parse_args(argv)

    if args.list_salts:
        print(json.dumps({
            k: {"description": v.description,
                "components": [{"formula": f, "mole_percent": m} for f, m in v.components],
                "has_bundled_model": v.has_bundled_model}
            for k, v in presets.PRESETS.items()
        }, indent=2))
        return 0

    specs, preset_key = _resolve_specs(args)
    mix = mixture_from_mole_percent(specs, args.n_formula_units)
    built = build_box(
        mix,
        initial_density_g_cm3=args.density,
        seed=args.seed,
        jitter_fraction=args.jitter_fraction,
    )
    write_pdb(built, args.output)

    dmin = min_image_min_distance(built.positions, built.box_length)
    achieved = mix.as_dict()["components"]
    # Suggest a 1-D composition axis: the first component, by convention.
    axis = {"label": f"{achieved[0]['name']} (mol%)", "value": achieved[0]["mole_percent"]}

    has_model = bool(preset_key and presets.PRESETS.get(preset_key, None)
                     and presets.PRESETS[preset_key].has_bundled_model)
    summary = {
        "output": args.output,
        "salt": preset_key,
        "components": achieved,
        "element_counts": mix.element_counts(),
        "n_atoms": mix.n_atoms,
        "box_length_angstrom": round(built.box_length, 4),
        "initial_density_g_cm3": round(density_from_box(mix, built.box_length), 5),
        "min_interatomic_distance_angstrom": round(dmin, 3),
        "seed": args.seed,
        "composition_axis": axis,
        "bundled_model_available": has_model,
    }
    print(json.dumps(summary, indent=2))

    if dmin < args.min_distance:
        print(
            f"WARNING: closest pair {dmin:.2f} A < {args.min_distance} A; "
            "consider a lower --density or larger --jitter-fraction to avoid "
            "exploding forces at step 0.",
            file=sys.stderr,
        )
    if not has_model:
        print(
            "NOTE: no bundled ML potential for this chemistry. To simulate it, pass a "
            "matching MACE model to run_npt.py via --model. (Flibe is the only bundled model.)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
