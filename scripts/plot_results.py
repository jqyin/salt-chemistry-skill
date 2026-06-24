#!/usr/bin/env python3
"""Plot and tabulate density results from a campaign of NPT runs.

Reads one or more ``results.json`` files (written by ``run_npt.py``) and produces
a density figure with error bars plus a tidy CSV summary. The x-axis can be
temperature or composition; the *other* variable becomes separate series, so a
2-D sweep over (temperature, composition) renders as a family of curves.

Composition is read from each result's ``state_point.composition_axis``
({label, value}); for legacy Flibe results it falls back to
``state_point.mole_percent_BeF2``. This makes the plotter chemistry-agnostic.

Examples
--------
    # Density vs temperature (auto-detects when composition is fixed)
    python scripts/plot_results.py runs/*/results.json -o density_vs_T.png

    # Density vs composition at fixed temperature
    python scripts/plot_results.py runs/*/results.json --x composition \
        -o density_vs_comp.png --csv summary.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: no display needed on login nodes
import matplotlib.pyplot as plt


def _composition(sp: dict) -> tuple[float | None, str]:
    """Return (composition value, axis label) from a state_point dict."""
    axis = sp.get("composition_axis")
    if isinstance(axis, dict) and axis.get("value") is not None:
        return axis["value"], axis.get("label", "Composition (mol%)")
    if sp.get("mole_percent_BeF2") is not None:  # legacy Flibe results
        return sp["mole_percent_BeF2"], "BeF$_2$ (mol%)"
    return None, "Composition (mol%)"


def _load(paths):
    rows = []
    comp_label = "Composition (mol%)"
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if not os.path.exists(path):
                print(f"WARNING: {path} not found, skipping", file=sys.stderr)
                continue
            with open(path) as fh:
                r = json.load(fh)
            sp = r.get("state_point", {})
            d = r.get("density", {})
            comp_val, comp_label = _composition(sp)
            rows.append({
                "path": path,
                "temperature_K": sp.get("temperature_K"),
                "composition": comp_val,
                "density_g_cm3": d.get("density_g_cm3"),
                "stderr_g_cm3": d.get("density_stderr_g_cm3"),
            })
    return rows, comp_label


def _choose_x(rows, requested):
    temps = {round(r["temperature_K"], 3) for r in rows if r["temperature_K"] is not None}
    comps = {round(r["composition"], 3) for r in rows if r["composition"] is not None}
    if requested != "auto":
        return requested
    # Vary along whichever axis actually changes; ties go to temperature.
    return "temperature" if len(temps) >= len(comps) else "composition"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Plot density vs temperature or composition from results.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("results", nargs="+", help="results.json paths or globs.")
    p.add_argument("--x", choices=["auto", "temperature", "composition"], default="auto",
                   help="X-axis variable; 'auto' picks the one that varies most.")
    p.add_argument("-o", "--output", default="density.png", help="Output figure path.")
    p.add_argument("--csv", default=None, help="Also write a CSV summary table here.")
    p.add_argument("--title", default=None, help="Figure title.")
    args = p.parse_args(argv)

    rows, comp_label = _load(args.results)
    rows = [r for r in rows if r["density_g_cm3"] is not None]
    if not rows:
        print("No usable results found.", file=sys.stderr)
        return 1

    x_kind = _choose_x(rows, args.x)
    if x_kind == "temperature":
        x_key, x_label, series_key, series_unit = "temperature_K", "Temperature (K)", "composition", "mol%"
    else:
        x_key, x_label, series_key, series_unit = "composition", comp_label, "temperature_K", "K"

    # Group into series by the non-x variable.
    series: dict = {}
    for r in rows:
        skey = r[series_key]
        series.setdefault(round(skey, 3) if skey is not None else None, []).append(r)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for skey in sorted(series, key=lambda v: (v is None, v)):
        pts = sorted([r for r in series[skey] if r[x_key] is not None], key=lambda r: r[x_key])
        if not pts:
            continue
        xs = [r[x_key] for r in pts]
        ys = [r["density_g_cm3"] for r in pts]
        es = [r["stderr_g_cm3"] or 0.0 for r in pts]
        label = f"{skey:g} {series_unit}" if (len(series) > 1 and skey is not None) else None
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Density (g cm$^{-3}$)")
    ax.set_title(args.title or f"Molten-salt density vs {x_kind}")
    if len(series) > 1:
        ax.legend(title=("Composition" if series_key == "composition" else "Temperature"),
                  fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote figure: {args.output}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["temperature_K", "composition_mol_percent", "density_g_cm3", "stderr_g_cm3", "source"])
            for r in sorted(rows, key=lambda r: (r["temperature_K"] or 0, r["composition"] or 0)):
                w.writerow([r["temperature_K"], r["composition"],
                            r["density_g_cm3"], r["stderr_g_cm3"], r["path"]])
        print(f"Wrote summary: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
