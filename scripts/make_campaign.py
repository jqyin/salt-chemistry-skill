#!/usr/bin/env python3
"""Lay out a campaign of NPT runs over a temperature x composition grid.

For each (temperature, composition) state point this creates a run directory with
a freshly built ``structure.pdb`` and a ``config.yaml`` for ``run_npt.py``, plus
``run_dirs.txt`` and ``campaign_manifest.json`` that the SLURM job array
(``templates/submit_campaign.sbatch``) consumes — one GPU per state point.

Base composition (pick one):
  * ``--salt flinak``                       a preset
  * ``--components "LiF:66.67,BeF2:33.33"``  explicit
  * ``--mol-percent-bef2 33.33``            Flibe shortcut

Composition sweep (optional):
  * ``--mol-percent-bef2 25,33.33,50``      sweep BeF2 directly (Flibe), or
  * ``--vary-component KF --vary-percent 30,42,50``  vary one component of any
    base mixture, rebalancing the others in proportion.

Examples
--------
    # Flibe density vs temperature
    python scripts/make_campaign.py --mol-percent-bef2 33.33 \
        --temperatures 700,800,900,1000 --base-dir campaign_T

    # Flibe 2-D screen: composition x temperature
    python scripts/make_campaign.py --mol-percent-bef2 25,33.33,50 \
        --temperatures 800,900,1000 --base-dir campaign_grid

    # FLiNaK temperature sweep (needs a FLiNaK model passed via --model)
    python scripts/make_campaign.py --salt flinak --temperatures 800,900,1000 \
        --model /path/to/mace_flinak.model --base-dir campaign_flinak
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.composition import mixture_from_mole_percent, density_from_box
from saltmd.structure import build_box, write_pdb
from saltmd import presets


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _base_specs(args) -> list[tuple[str, float]]:
    if args.components:
        out = []
        for part in args.components.split(","):
            formula, pct = part.rsplit(":", 1)
            out.append((formula.strip(), float(pct)))
        return out
    if args.salt:
        return list(presets.get_preset(args.salt).components)
    if args.mol_percent_bef2:
        # The first value is the base; the sweep (if any) is handled by the caller.
        return [("LiF", 100.0 - args.mol_percent_bef2[0]), ("BeF2", args.mol_percent_bef2[0])]
    raise SystemExit("Provide a base composition: --salt, --components, or --mol-percent-bef2.")


def _rebalance(base: list[tuple[str, float]], component: str, value: float) -> list[tuple[str, float]]:
    """Set ``component`` to ``value`` mol%, distributing the rest in base proportion."""
    names = [n for n, _ in base]
    if component not in names:
        raise SystemExit(f"--vary-component '{component}' not in base components {names}")
    others = [(n, p) for n, p in base if n != component]
    other_total = sum(p for _, p in others) or 1.0
    rest = max(0.0, 100.0 - value)
    out = []
    for n, p in base:
        out.append((n, value if n == component else rest * p / other_total))
    return out


def _composition_points(args):
    """Yield (specs, axis_dict) for each composition in the sweep."""
    base = _base_specs(args)
    if args.mol_percent_bef2 and len(args.mol_percent_bef2) > 1:
        for x in args.mol_percent_bef2:
            yield [("LiF", 100.0 - x), ("BeF2", x)], {"label": "BeF2 (mol%)", "value": x}
        return
    if args.vary_component and args.vary_percent:
        for v in args.vary_percent:
            yield _rebalance(base, args.vary_component, v), {"label": f"{args.vary_component} (mol%)", "value": v}
        return
    # Single composition: axis is the first component's mol% (constant).
    first_name, first_pct = base[0]
    yield base, {"label": f"{first_name} (mol%)", "value": first_pct}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build a temperature x composition campaign of NPT runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--temperatures", required=True, type=_floats,
                   help="Comma-separated temperatures in K, e.g. 700,800,900.")
    p.add_argument("--salt", help="Base composition preset (flibe, flinak, ...).")
    p.add_argument("--components", help="Explicit base 'FORMULA:MOLEPERCENT,...'.")
    p.add_argument("--mol-percent-bef2", type=_floats, default=None,
                   help="Flibe: one value (fixed) or a list to sweep BeF2.")
    p.add_argument("--vary-component", help="Name of the component to sweep (any base mixture).")
    p.add_argument("--vary-percent", type=_floats, help="mol%% values for --vary-component.")
    p.add_argument("--base-dir", default="campaign", help="Directory to hold all run dirs.")
    p.add_argument("--model", default="assets/mace_flibe.model",
                   help="ML model path (stored absolute in each run's config). Must match the chemistry!")
    p.add_argument("--n-formula-units", type=int, default=720, help="System size per run.")
    p.add_argument("--initial-density", type=float, default=1.9, help="Initial density (g/cm^3).")
    p.add_argument("--equilibration-steps", type=int, default=50000)
    p.add_argument("--production-steps", type=int, default=250000)
    p.add_argument("--pressure", type=float, default=1.0, help="Pressure in bar.")
    p.add_argument("--seed", type=int, default=1, help="Base seed; each run gets seed+index.")
    args = p.parse_args(argv)

    import yaml

    os.makedirs(args.base_dir, exist_ok=True)
    model_abs = os.path.abspath(args.model)
    manifest, run_dirs = [], []

    idx = 0
    for specs, axis in _composition_points(args):
        for T in args.temperatures:
            mix = mixture_from_mole_percent(specs, args.n_formula_units)
            tag = f"c{axis['value']:06.2f}_T{T:06.1f}".replace(".", "p")
            run_dir = os.path.join(args.base_dir, tag)
            os.makedirs(run_dir, exist_ok=True)

            seed = args.seed + idx
            built = build_box(mix, initial_density_g_cm3=args.initial_density, seed=seed)
            write_pdb(built, os.path.join(run_dir, "structure.pdb"))

            components = [{"name": c["name"], "mole_percent": c["mole_percent"]}
                         for c in mix.as_dict()["components"]]
            cfg = {
                "structure_pdb": "structure.pdb",
                "model_file": model_abs,
                "output_dir": ".",
                "temperature_K": T,
                "pressure_bar": args.pressure,
                "equilibration_steps": args.equilibration_steps,
                "production_steps": args.production_steps,
                "seed": seed,
                "components": [[c["name"], c["mole_percent"]] for c in components],
                "composition_axis": axis,
            }
            with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)

            manifest.append({
                "run_dir": run_dir, "temperature_K": T,
                "composition_axis": axis, "components": components,
                "n_atoms": mix.n_atoms, "seed": seed,
                "initial_density_g_cm3": round(density_from_box(mix, built.box_length), 5),
            })
            run_dirs.append(run_dir)
            idx += 1

    with open(os.path.join(args.base_dir, "campaign_manifest.json"), "w") as fh:
        json.dump({"n_runs": len(manifest), "runs": manifest}, fh, indent=2)
    with open(os.path.join(args.base_dir, "run_dirs.txt"), "w") as fh:
        fh.write("\n".join(run_dirs) + "\n")

    print(json.dumps({
        "base_dir": args.base_dir,
        "n_runs": len(manifest),
        "temperatures_K": args.temperatures,
        "model": model_abs,
        "submit": f"sbatch --array=0-{len(manifest) - 1} templates/submit_campaign.sbatch {args.base_dir}/run_dirs.txt",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
