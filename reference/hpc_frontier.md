# Running on OLCF Frontier (and other GPU clusters)

The simulation step is the only part that needs a GPU. This covers the software
stack, job submission, and troubleshooting on Frontier (AMD MI250X / ROCm). Other
clusters work too — see [Other clusters](#other-clusters-cuda--cpu).

## Contents
- [Software stack](#software-stack)
- [First-time environment setup](#first-time-environment-setup)
- [Submitting jobs](#submitting-jobs)
- [Environment variables](#environment-variables)
- [Walltime, queues, and sizing](#walltime-queues-and-sizing)
- [Troubleshooting](#troubleshooting)
- [Other clusters (CUDA / CPU)](#other-clusters-cuda--cpu)

## Software stack

Frontier nodes have 4× AMD MI250X GPUs (8 GCDs). The stack:

- `PrgEnv-gnu`, `cpe`, `miniforge3`, `rocm`, `craype-accel-amd-gfx90a` modules
- A conda env (`openmm-torch-frontier`) with PyTorch (ROCm wheels), OpenMM built
  with the **HIP** platform, `openmm-ml`, and `mace-torch`.

Exact module versions live in `templates/frontier_env.sh`; the env spec is
`environment.frontier.yml`. Both are overridable via variables if OLCF updates the
stack.

## First-time environment setup

`templates/frontier_env.sh` creates the conda env automatically on first use. To do
it explicitly (e.g. on a login node before queueing work):

```bash
cd <skill-root>
source templates/frontier_env.sh        # loads modules, creates+activates the env
```

The repo also ships `install_frontier_rocm_stack.sh`, a from-scratch installer that
pins the PyTorch ROCm wheels and pip packages step by step — use it if the
one-shot conda env create fails and you need to debug the install.

## Submitting jobs

**Single state point** (edit `-A <YOUR_PROJECT_ID>` first):
```bash
sbatch templates/submit_frontier.sbatch runs/flibe_783K \
    --structure structure.pdb --model assets/mace_flibe.model --temperature 783.15
```

**Interactive** (grab a node, then run under `srun`):
```bash
salloc -A <PROJECT> -N1 -t 1:00:00 -p batch
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    ./scripts/run_openmm_onegpu.sh --output-dir runs/flibe_783K --temperature 783.15
```

**Campaign (job array, one GPU per state point):**
```bash
python scripts/make_campaign.py --temperatures 700,800,900,1000 \
    --mol-percent-bef2 33.33 --base-dir campaign
sbatch --array=0-$(($(wc -l < campaign/run_dirs.txt)-1)) \
    templates/submit_campaign.sbatch campaign/run_dirs.txt
```

The `-c7 --gpus-per-task=1 --gpu-bind=closest` flags bind one rank to one GCD with
its nearest CPU cores (a Frontier L3 group). One MD run uses one GCD; scale by
running independent state points as separate ranks/array tasks, not by splitting
one run across GPUs.

## Environment variables

The launcher sets these; you can override them. `run_npt.py` also reads them
(CLI flags win over env, which wins over the YAML config):

| Variable | Meaning | Default |
|----------|---------|---------|
| `OPENMM_PLATFORM` | OpenMM backend | `HIP` |
| `OPENMM_PRECISION` | `single` / `mixed` / `double` | `mixed` |
| `OPENMM_DEVICE_INDEX` | GPU index string | `0` |
| `PRODUCTION_STEPS` / `STEPS` | production steps | from config |
| `EQUILIBRATION_STEPS` | equilibration steps | from config |
| `TEMPERATURE_K` | temperature | from config |
| `SEED` | master RNG seed | from config |

A quick wiring test (tiny, real numbers meaningless):
```bash
PRODUCTION_STEPS=1000 EQUILIBRATION_STEPS=0 \
  srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
  ./scripts/run_openmm_onegpu.sh --output-dir runs/smoke
```

## Walltime, queues, and sizing

- **Throughput**: with ~1680 atoms, a MACE NPT run reaches a few hundred ns/day on
  one MI250X GCD (see the `Speed (ns/day)` column in the state log). 250 ps of
  production is well under an hour; request 1–2 h to be safe.
- Use the **job array** for screens — it is far more efficient than looping `sbatch`
  and packs naturally onto Frontier's many GPUs.
- Keep one state point per GCD; these runs are GPU-bound, not communication-bound.

## Troubleshooting

- **"Torch reports no GPU"** — the job isn't on a GPU node, or the ROCm PyTorch
  build isn't loaded. Confirm `source templates/frontier_env.sh` ran and that
  `torch.cuda.is_available()` is `True` in the launcher's diagnostic block.
- **Requested platform unavailable** — `run_npt.py` falls back through
  HIP→CUDA→OpenCL→CPU and logs a warning. If it lands on CPU on a GPU node, the
  OpenMM HIP plugin didn't load (check the `rocm` module and `openmm[hip7]`).
- **NaNs / energy blow-up on step 0** — almost always overlapping atoms in the
  initial box. Rebuild with a lower `--density` or larger `--jitter-fraction`, and
  watch the closest-pair warning from `build_structure.py`. Ensure minimization is
  on (`minimize_iterations > 0`).
- **Out of memory** — reduce `--n-formula-units`, or use `Precision=single`.
- **Density not converged / drifting** — see `reference/methodology.md`; lengthen
  equilibration and/or production.

## Other clusters (CUDA / CPU)

- **NVIDIA**: set `--platform CUDA` (or `OPENMM_PLATFORM=CUDA`) and install a CUDA
  build of OpenMM + PyTorch. Everything else is identical; adapt the module loads
  in a copy of `frontier_env.sh`.
- **CPU**: `--platform CPU` works for wiring tests and tiny systems but is far too
  slow for production densities. Use it only to validate the pipeline off-GPU.
