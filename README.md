# molten-salt-md

OpenMM + MACE NPT molecular dynamics for predicting **chemistry properties of
molten salts** — primarily **mass density** as a function of composition and
temperature. Designed to be driven by an AI agent end-to-end on HPC GPUs (built for
OLCF Frontier).

The flagship system is **Flibe** (LiF–BeF₂, the fusion breeder/coolant salt), which
ships with a fitted MACE potential. The composition/structure/analysis/HPC machinery
is **chemistry-agnostic** (FLiNaK, chloride salts, arbitrary mixtures via `--salt`
or `--components`), but each salt family needs its own ML potential — only Flibe is
bundled. See [`saltmd/presets.py`](saltmd/presets.py) for known salts.

This repository is both a **Python package** (`saltmd`) and a **Claude skill**
(`SKILL.md` + `scripts/` + `reference/`). The skill front-matter tells an agent
when and how to use it; humans can use the same scripts directly.

## What it does

Given a candidate salt (composition in mol% of each component) and conditions
(temperature), it:

1. **Builds** a periodic simulation box at that composition (`build_structure.py`).
2. **Runs** NPT MD with a fine-tuned MACE machine-learning potential, in three
   phases — minimize → equilibrate → produce (`run_npt.py`).
3. **Analyzes** the production trajectory into an equilibrium density with proper
   error bars (`analyze_density.py`, also done automatically by `run_npt.py`).
4. **Plots / tabulates** density vs temperature or vs composition across a campaign
   (`plot_results.py`).

A campaign helper (`make_campaign.py`) fans a temperature × composition grid out
into a SLURM job array — one GPU per state point.

## Quick start

```bash
# 1) Build a structure (runs anywhere — no GPU)
python scripts/build_structure.py --mol-percent-bef2 33.33 -o structure.pdb

# 2) Run NPT on a GPU node (OLCF Frontier shown)
sbatch templates/submit_frontier.sbatch runs/flibe_783K \
    --structure structure.pdb --model assets/mace_flibe.model --temperature 783.15

# 3) Read runs/flibe_783K/results.json  -> density.density_g_cm3 ± density_stderr_g_cm3

# 4) Plot a campaign
python scripts/plot_results.py runs/*/results.json -o density.png --csv summary.csv
```

See `SKILL.md` for the agent workflow and `reference/` for the science
(`methodology.md`), HPC details (`hpc_frontier.md`), and the output schema
(`outputs.md`).

## Layout

```
SKILL.md                  Skill entry point (agent workflow + triggers)
saltmd/                   Python package (pure planning/analysis + GPU simulation)
  composition.py            general mixture <-> atom counts <-> box size/density (pure)
  presets.py                named salts (flibe, flinak, licl-kcl, ...)
  structure.py              build a randomized periodic box (any elements); write PDB
  simulation.py             NPT minimize/equilibrate/produce (imports OpenMM lazily)
  analysis.py               parse logs; equilibrium density + error bars
scripts/                  CLI tools the agent calls (argparse, JSON I/O)
templates/                SLURM sbatch, env setup, example config
reference/                methodology, HPC guide, output schema
assets/                   mace_flibe.model (potential) + reference structure
examples/                 sample state log for analysis/plotting demos
tests/                    unit tests for the pure layer (run with pytest)
environment.frontier.yml  conda env for the OpenMM+MACE ROCm stack
_legacy/                  the original input_npt.py et al., preserved for provenance
```

## Installation

The planning/analysis layer needs only numpy/pandas/matplotlib/pyyaml:

```bash
pip install -e .            # or: pip install numpy pandas matplotlib pyyaml
pytest                      # run the unit tests
```

The GPU stack (OpenMM with HIP/CUDA, openmm-ml, torch, mace-torch) is provided by
the conda environment — see `reference/hpc_frontier.md`.

## Scope

This skill computes **density** (and the equilibrated box geometry) for molten-salt
melts. The **tritium breeding ratio** is a neutronics quantity computed by other
tools; this provides the density those analyses consume.

The engine is chemistry-agnostic, but **each salt family needs its own ML
potential** — the bundled `mace_flibe.model` is valid only for Li/Be/F. To study
another salt, build/plan with `--salt`/`--components` and supply a matching model via
`--model`; never run a different chemistry through the Flibe potential.
