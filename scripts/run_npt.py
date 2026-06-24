#!/usr/bin/env python3
"""Run an NPT density calculation for a molten salt (modernized input_npt.py).

This is the GPU-bound entry point, normally launched on a compute node by
``templates/submit_frontier.sbatch`` or ``scripts/run_openmm_onegpu.sh``. It runs
minimize -> equilibrate -> produce, then writes ``results.json`` containing the
equilibrium density with error bars and full provenance.

Configuration precedence (lowest to highest):
    built-in defaults  <  --config YAML  <  environment variables  <  CLI flags

Environment variables (set by the HPC launcher) mirror the original script and
add the phase split:
    STEPS / PRODUCTION_STEPS, EQUILIBRATION_STEPS, TEMPERATURE_K, SEED,
    OPENMM_PLATFORM, OPENMM_PRECISION, OPENMM_DEVICE_INDEX

Examples
--------
    # Single state point, explicit flags
    python scripts/run_npt.py --structure structure.pdb --model assets/mace_flibe.model \
        --temperature 783.15 --production-steps 250000 --output-dir runs/flibe_783K

    # Quick CPU smoke test (no GPU needed; tiny and slow, just checks the wiring)
    python scripts/run_npt.py --platform CPU --minimize-iterations 10 \
        --equilibration-steps 0 --production-steps 50 --output-dir /tmp/smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.simulation import SimulationConfig, run_npt

# CLI flag -> (config field, type).  Defaults are None so we can tell which flags
# the user actually set and let them override env/YAML.
_FLAGS = {
    "structure": ("structure_pdb", str),
    "model": ("model_file", str),
    "output_dir": ("output_dir", str),
    "temperature": ("temperature_K", float),
    "pressure": ("pressure_bar", float),
    "timestep": ("timestep_fs", float),
    "friction": ("friction_per_ps", float),
    "barostat_interval": ("barostat_interval", int),
    "minimize_iterations": ("minimize_iterations", int),
    "equilibration_steps": ("equilibration_steps", int),
    "production_steps": ("production_steps", int),
    "report_interval": ("report_interval", int),
    "checkpoint_interval": ("checkpoint_interval", int),
    "seed": ("seed", int),
    "platform": ("platform", str),
    "precision": ("precision", str),
    "device_index": ("device_index", str),
}

# Environment variable -> (config field, type).
_ENV = {
    "STEPS": ("production_steps", int),
    "PRODUCTION_STEPS": ("production_steps", int),
    "EQUILIBRATION_STEPS": ("equilibration_steps", int),
    "TEMPERATURE_K": ("temperature_K", float),
    "SEED": ("seed", int),
    "OPENMM_PLATFORM": ("platform", str),
    "OPENMM_PRECISION": ("precision", str),
    "OPENMM_DEVICE_INDEX": ("device_index", str),
}


def _build_config(argv=None) -> SimulationConfig:
    p = argparse.ArgumentParser(
        description="Run an NPT density calculation for a molten salt. The --model "
                    "must match the structure's chemistry (the bundled mace_flibe.model "
                    "is Li/Be/F only; the run aborts if the model can't cover the structure).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="YAML file of config overrides (optional).")
    p.add_argument("--structure", help="Input PDB with a periodic box (CRYST1).")
    p.add_argument("--model", help="MACE model file for the MLPotential.")
    p.add_argument("--output-dir", help="Directory for outputs (results.json, logs, dcd).")
    p.add_argument("--temperature", type=float, help="Temperature in K.")
    p.add_argument("--pressure", type=float, help="Pressure in bar.")
    p.add_argument("--timestep", type=float, help="Integration timestep in fs.")
    p.add_argument("--friction", type=float, help="Langevin friction in 1/ps.")
    p.add_argument("--barostat-interval", type=int, help="Barostat attempt interval (steps).")
    p.add_argument("--minimize-iterations", type=int, help="Max minimizer iterations (0=off).")
    p.add_argument("--equilibration-steps", type=int, help="Equilibration steps (not averaged).")
    p.add_argument("--production-steps", type=int, help="Production steps (averaged for density).")
    p.add_argument("--report-interval", type=int, help="State/trajectory report interval (steps).")
    p.add_argument("--checkpoint-interval", type=int, help="Checkpoint interval (steps; 0=off).")
    p.add_argument("--seed", type=int, help="Master RNG seed (integrator/barostat/velocities).")
    p.add_argument("--platform", help="OpenMM platform: HIP, CUDA, OpenCL, or CPU.")
    p.add_argument("--precision", help="GPU precision: single, mixed, or double.")
    p.add_argument("--device-index", help="GPU device index (string, e.g. '0').")
    p.add_argument("--no-trajectory", action="store_true", help="Skip writing the DCD trajectory.")
    p.add_argument("--restart", action="store_true", help="Resume from checkpoint.chk in output-dir.")
    args = p.parse_args(argv)

    # 1) defaults
    cfg = SimulationConfig()

    # 2) YAML config file
    if args.config:
        import yaml
        with open(args.config) as fh:
            for key, value in (yaml.safe_load(fh) or {}).items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                else:
                    logging.warning("Ignoring unknown config key '%s'", key)

    # 3) environment variables
    for env_name, (field, typ) in _ENV.items():
        if os.getenv(env_name) is not None:
            setattr(cfg, field, typ(os.environ[env_name]))

    # 4) explicit CLI flags (highest priority)
    for flag, (field, typ) in _FLAGS.items():
        value = getattr(args, flag)
        if value is not None:
            setattr(cfg, field, value)
    if args.no_trajectory:
        cfg.write_trajectory = False
    if args.restart:
        cfg.restart = True

    return cfg


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    cfg = _build_config(argv)
    logging.getLogger("saltmd").info("Configuration: %s", cfg.as_dict())
    try:
        torch.cuda.set_device(int(cfg.device_index))
        run_npt(cfg)
    except FileNotFoundError as e:
        logging.error("Missing input file: %s", e)
        return 2
    except Exception as e:  # surface the failure with a non-zero exit for the scheduler
        logging.exception("Simulation failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
