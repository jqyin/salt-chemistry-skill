#!/usr/bin/env python3
"""Compute the equilibrium density (with error bars) from an OpenMM state log.

Reads a StateDataReporter CSV — either a production-only log written by
``run_npt.py`` (use ``--equilibration-fraction 0``) or a combined log that still
contains the equilibration ramp (use a fraction like 0.5, the default). Prints a
JSON summary and optionally writes it to a file.

Examples
--------
    # Re-analyze a production log discarding nothing (equilibration already split off)
    python scripts/analyze_density.py runs/flibe_783K/production.csv \
        --equilibration-fraction 0

    # Analyze a legacy combined log, dropping the first half as equilibration
    python scripts/analyze_density.py output_npt.out --equilibration-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.analysis import analyze_density


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compute equilibrium density (mean + error bars) from a state log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("log", help="Path to a StateDataReporter CSV log.")
    p.add_argument("--equilibration-fraction", type=float, default=0.5,
                   help="Leading fraction of frames to discard as equilibration "
                        "(use 0 for a production-only log from run_npt.py).")
    p.add_argument("--equilibration-time-ps", type=float, default=None,
                   help="Discard frames before this time (ps); overrides fraction.")
    p.add_argument("--n-blocks", type=int, default=5,
                   help="Number of blocks for block-averaged error bars.")
    p.add_argument("-o", "--output", default=None, help="Write JSON summary to this path.")
    args = p.parse_args(argv)

    result = analyze_density(
        args.log,
        equilibration_fraction=args.equilibration_fraction,
        equilibration_time_ps=args.equilibration_time_ps,
        n_blocks=args.n_blocks,
    )
    payload = result.as_dict()
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
