---
name: molten-salt-md
description: >-
  Run OpenMM NPT molecular-dynamics simulations of molten salts driven by a MACE
  machine-learning potential to predict chemistry properties — primarily mass
  density — as a function of composition and temperature, on HPC GPUs (OLCF
  Frontier). The flagship system is Flibe (LiF-BeF2), the fusion breeder/coolant
  salt, which ships with a fitted potential; the engine is chemistry-agnostic and
  also handles FLiNaK, chloride salts, and arbitrary mixtures given a matching
  potential. Use this skill whenever the user wants to compute or predict the
  density (or other equilibrium liquid properties) of a molten salt, Flibe, FLiNaK,
  a fluoride/chloride melt, or a fusion breeder-blanket / tritium-breeding salt;
  build a salt simulation box from a composition (mol% of each component); run NPT
  MD with a MACE or machine-learning interatomic potential; drive OpenMM on a GPU
  cluster or via SLURM; or screen candidate salt compositions and temperatures and
  plot density vs temperature or vs composition — even if they don't explicitly
  say "OpenMM" or "MACE".
---

# Molten-Salt MD: density from composition and temperature

This skill predicts equilibrium properties (primarily **mass density**) of molten
salts using **NPT molecular dynamics** in **OpenMM**, driven by a **MACE**
machine-learning interatomic potential trained on DFT data. It is built to be
driven end-to-end by an agent: take a candidate salt (composition) and conditions
(temperature), run on HPC GPUs, and return density with honest error bars and a
plot.

The flagship system is **Flibe** (LiF–BeF₂), the fusion breeder/coolant salt, which
ships with a fitted potential (`assets/mace_flibe.model`). The composition,
structure-building, analysis, and HPC machinery are **chemistry-agnostic** — FLiNaK,
chloride salts, and arbitrary mixtures work too — but **each salt family needs its
own ML potential**: the bundled model is valid only for Li/Be/F. See
[Other salts](#other-salts-and-the-potential-requirement).

## The 5-step workflow

Most tasks follow this pipeline. Each step is one script; outputs of one feed the
next. Run scripts from the skill root. (Commands below use `python`; on a bare
login node or laptop where only `python3` is on PATH, substitute `python3` — the
scripts also carry a `#!/usr/bin/env python3` shebang, so `./scripts/<tool>.py`
works too.)

```
composition + T  →  [build]  →  structure.pdb
structure.pdb    →  [run]    →  results.json  (density + provenance)
results.json...  →  [plot]   →  density.png + summary.csv
```

1. **Define the state point(s).** Composition is the **mol% of each component**.
   For Flibe that is mol% BeF₂ (33.33 = the 2LiF·BeF₂ eutectic). Temperature in
   kelvin (a typical Flibe coolant condition is ~783 K = 510 °C). One salt at one
   temperature is one state point; a screen is a grid of them.

2. **Build the structure** — generate a periodic box at the requested composition.
   Three equivalent ways to specify it:
   ```bash
   # Flibe shortcut (mol% BeF2)
   python scripts/build_structure.py --mol-percent-bef2 33.33 \
       --n-formula-units 720 --seed 1 -o structure.pdb

   # A named preset (run --list-salts to see them: flibe, flinak, licl-kcl, ...)
   python scripts/build_structure.py --salt flinak -o flinak.pdb

   # Explicit components (any mixture)
   python scripts/build_structure.py --components "NaCl:58,MgCl2:42" --density 1.6 -o nacl.pdb
   ```
   This prints a JSON summary (achieved component mol%, element counts, box size,
   closest-pair distance). Heed a closest-pair warning — too-close ions blow up the
   ML forces at step 0. Note chlorides start looser (~1.5–1.7 g/cm³) than fluorides.

3. **Run the NPT simulation on a GPU** (this is the only GPU/HPC step). The driver
   runs **minimize → equilibrate → produce** and writes `results.json`. **First
   edit `#SBATCH -A <YOUR_PROJECT_ID>`** in the sbatch template to your real OLCF
   allocation, or SLURM will reject the job:
   ```bash
   # On OLCF Frontier (recommended): submit a batch job
   sbatch templates/submit_frontier.sbatch runs/flibe_783K \
       --structure structure.pdb --model assets/mace_flibe.model --temperature 783.15

   # On an interactive GPU node:
   srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
       ./scripts/run_openmm_onegpu.sh --output-dir runs/flibe_783K --temperature 783.15
   ```
   See `reference/hpc_frontier.md` for modules, queues, and walltime guidance.

4. **Read the result.** `run_npt.py` already computes the equilibrium density and
   writes it into `results.json` (`density.density_g_cm3` ±
   `density.density_stderr_g_cm3`). To re-analyze with a different equilibration
   cutoff, or to analyze a legacy/combined log, use `analyze_density.py` (see
   `reference/outputs.md` for the schema).

5. **Interpret / plot.** Combine results across state points:
   ```bash
   python scripts/plot_results.py runs/*/results.json -o density.png --csv summary.csv
   ```
   The x-axis auto-detects (temperature if T varies, composition if mol% varies);
   a 2-D screen renders as a family of curves. Always report the value **with its
   error bar** and the temperature/composition it belongs to.

## Screening many salts at once (the common HPC case)

To screen candidate compositions and/or temperatures, generate the whole grid and
submit it as one SLURM job array (one GPU per state point):

```bash
# Build run dirs (each with its own structure.pdb + config.yaml).
# Flibe 2-D screen (composition x temperature):
python scripts/make_campaign.py --temperatures 700,800,900,1000 \
    --mol-percent-bef2 25,33.33,50 --base-dir campaign --production-steps 250000

# General mixture: sweep one component of any base, rebalancing the rest, e.g.
#   --salt flinak --vary-component KF --vary-percent 30,42,50 --model <flinak_model>

# Submit the array (the make_campaign output prints this exact line)
sbatch --array=0-$(($(wc -l < campaign/run_dirs.txt)-1)) \
    templates/submit_campaign.sbatch campaign/run_dirs.txt

# When done, collect and plot
python scripts/plot_results.py campaign/*/results.json -o density_grid.png --csv grid.csv
```

## Decision hints for the agent

- **"What's the density of <salt> at <T>?"** → one state point: build → run → read
  `results.json`. State the density ± error and the conditions.
- **"How does density vary with temperature / composition?"** → a sweep: use
  `make_campaign.py`, submit the array, then `plot_results.py`.
- **"Screen these candidates for tritium breeding."** → a 2-D campaign over
  (mol% BeF₂, T). Tritium-breeding ratio itself is a neutronics quantity computed
  elsewhere; this skill supplies the **density** (and box geometry) those analyses
  need. Don't claim to compute the breeding ratio.
- **No GPU available (laptop/login node)?** You can still build structures, analyze
  existing logs, and plot — only step 3 needs a GPU. For a wiring smoke test use
  `--platform CPU` with tiny step counts (slow; not for real numbers).
- **A non-Flibe salt (FLiNaK, a chloride, …)?** Build and plan exactly the same way,
  but you **must** supply a potential trained on that chemistry via `--model` — see
  below.

## Other salts and the potential requirement

The code is chemistry-agnostic: `build_structure.py`, `make_campaign.py`,
`analyze_density.py`, `plot_results.py`, and the HPC templates work for any mixture
(use `--salt <preset>` or `--components "FORMULA:mol%,..."`). **But running MD needs
an ML potential valid for that chemistry**, and only **Flibe (Li/Be/F)** ships with
one (`assets/mace_flibe.model`). For another salt:

1. Obtain/train a MACE (or other OpenMM-ML-compatible) model for its elements.
2. Pass it with `--model /path/to/model` to `run_npt.py` / `make_campaign.py`.

If asked to simulate a chemistry with no available model, say so plainly — you can
still build the structure and lay out the campaign, but the run is blocked on the
potential. **Never run a salt through the Flibe model**; results would be physically
meaningless.

## Inputs you need present

- An ML potential matching the chemistry. `assets/mace_flibe.model` (LiF–BeF₂) is
  bundled; pass any model with `--model`.
- A periodic-box PDB — build one with `build_structure.py`, or supply your own
  (must contain a `CRYST1` box record).

## Reference material (read when you need depth)

- `reference/methodology.md` — the science: NPT density protocol, why equilibration
  is split out, error bars, choosing run length and system size, validating against
  experiment. **Read this before judging whether a result is converged or trustworthy.**
- `reference/hpc_frontier.md` — OLCF Frontier specifics: modules, conda env, SLURM,
  `srun` flags, environment variables, and troubleshooting (GPU not found, OOM, NaNs).
- `reference/outputs.md` — exact `results.json` schema and the meaning of every
  output file, so you can parse results programmatically.

## Guardrails

- Report density **with uncertainty**; a single number with no error bar is
  incomplete. If `equilibration` looks too short (density still drifting in
  `equilibration.csv`), say so and rerun longer rather than reporting a biased mean.
- Composition is given in **mol%**; integer atom counts round it slightly, so record
  the **achieved** mol% from the output, not the requested one.
- **Match the potential to the chemistry.** The bundled model is Flibe-only; running
  any other salt requires its own model. Treat all predictions as potential-limited
  and, where possible, sanity-check against experiment — e.g. eutectic Flibe is
  ~1.94 g/cm³ near 783 K (see `reference/methodology.md`).
