"""Configure and run an NPT molecular-dynamics simulation with a MACE potential.

This is the modernized successor to the original ``input_npt.py``. The physics is
unchanged — Langevin-middle integration with a Monte Carlo barostat, driven by a
fine-tuned MACE machine-learning potential through OpenMM-ML — but the workflow is
restructured around scientific-software best practices:

  * **Explicit phases**: energy minimization -> equilibration -> production. Only
    the production phase is averaged for density, so the reported value is not
    biased by the box relaxing from its initial size.
  * **Reproducibility**: every stochastic element (integrator, barostat, initial
    velocities) is seeded from a single ``seed``.
  * **Provenance**: every run writes a ``results.json`` capturing inputs, software
    versions, hardware, and the computed density with error bars — so a result can
    always be traced back to exactly how it was produced.
  * **Configuration over editing**: all parameters live in a :class:`SimulationConfig`
    populated from CLI/YAML/env, instead of constants edited in the source.

OpenMM, PyTorch, and MACE are imported lazily inside :func:`run_npt` so the rest of
the package (composition, structure, analysis) stays importable without a GPU stack.
"""

from __future__ import annotations

import json
import logging
import os
import platform as _platform
import socket
import time
from dataclasses import dataclass, asdict, field

logger = logging.getLogger("saltmd.simulation")

# Platforms tried, in order, when the requested one is unavailable. HIP is the
# Frontier (AMD MI250X) backend; CUDA for NVIDIA; OpenCL/CPU as last resorts.
_PLATFORM_FALLBACK = ("HIP", "CUDA", "OpenCL", "CPU")
# Platforms that accept GPU-style properties (Precision / DeviceIndex).
_GPU_PLATFORMS = {"HIP", "CUDA", "OpenCL"}


@dataclass
class SimulationConfig:
    """All parameters for one NPT density calculation at a single state point."""

    # --- inputs ---
    structure_pdb: str = "structure.pdb"
    model_file: str = "assets/mace_flibe.model"
    output_dir: str = "."

    # --- thermodynamic state point ---
    temperature_K: float = 783.15
    pressure_bar: float = 1.0

    # --- composition labelling (optional; for provenance + plotting) ---
    # The actual atoms come from the structure PDB; these only describe the
    # *intended* mixture so results.json and plots are self-labelling. Set by the
    # structure builder / campaign generator. ``components`` is a list of
    # [formula, mole_percent]; ``composition_axis`` is {"label","value"} naming the
    # scalar to use as the composition x-axis (e.g. the component being varied).
    components: list | None = None
    composition_axis: dict | None = None

    # --- integration ---
    timestep_fs: float = 1.0
    friction_per_ps: float = 10.0
    barostat_interval: int = 50

    # --- run length (steps) ---
    minimize_iterations: int = 200          # 0 disables minimization
    equilibration_steps: int = 50_000
    production_steps: int = 250_000

    # --- reporting (steps) ---
    report_interval: int = 100
    checkpoint_interval: int = 50_000
    write_trajectory: bool = True

    # --- reproducibility ---
    seed: int = 1

    # --- compute backend ---
    platform: str = "HIP"
    precision: str = "mixed"
    device_index: str = "0"

    # --- restart ---
    restart: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _software_versions() -> dict:
    """Capture versions of the key libraries for provenance."""
    versions = {"python": _platform.python_version()}
    try:
        import openmm
        versions["openmm"] = openmm.version.version
    except Exception:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["torch_hip"] = getattr(torch.version, "hip", None)
        versions["torch_cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    try:
        import mace
        versions["mace"] = getattr(mace, "__version__", "unknown")
    except Exception:
        pass
    return versions


def _select_platform(mm, requested: str):
    """Return an OpenMM Platform, falling back through known backends.

    Raising early with a clear message beats a cryptic failure deep in setup when
    a job lands on a node without the requested accelerator.
    """
    order = [requested] + [p for p in _PLATFORM_FALLBACK if p != requested]
    tried = []
    for name in order:
        try:
            plat = mm.Platform.getPlatformByName(name)
            if name != requested:
                logger.warning("Requested platform %s unavailable; using %s", requested, name)
            return plat, name
        except Exception:
            tried.append(name)
    raise RuntimeError(f"No usable OpenMM platform found. Tried: {tried}")


# Elements the bundled Flibe potential is trained on. Used as a name-based safety
# net so the shipped model is never silently applied to another chemistry.
_FLIBE_MODEL_BASENAME = "mace_flibe.model"
_FLIBE_ELEMENTS = {"Li", "Be", "F"}


def model_supported_elements(model_file: str):
    """Best-effort set of element symbols a MACE model supports, or None if unknown.

    MACE models register an ``atomic_numbers`` buffer; reading it lets us verify
    the potential actually covers the structure's chemistry before spending GPU
    time. Returns None if the model can't be introspected (then the caller falls
    back to the name-based guard).
    """
    try:
        import torch
        from ase.data import chemical_symbols
        obj = torch.load(model_file, map_location="cpu", weights_only=False)
        zs = getattr(obj, "atomic_numbers", None)
        if zs is None:
            return None
        return {chemical_symbols[int(z)] for z in zs.tolist()}
    except Exception:
        return None


def validate_model_for_elements(model_file: str, structure_elements, supported_elements=None) -> None:
    """Raise a clear error if ``model_file`` cannot cover the structure's elements.

    Running a salt through a potential not trained on its elements gives physically
    meaningless numbers, so we fail fast. When the model's true element coverage is
    known (``supported_elements``) we enforce it directly; otherwise we at least
    refuse to apply the bundled Flibe model to a non-Li/Be/F structure.
    """
    present = set(structure_elements)
    if supported_elements is not None:
        missing = present - set(supported_elements)
        if missing:
            raise RuntimeError(
                f"Model '{model_file}' does not cover element(s) {sorted(missing)} present "
                f"in the structure (model supports {sorted(supported_elements)}). Supply a "
                f"potential trained on this chemistry via --model."
            )
        return
    if os.path.basename(str(model_file)) == _FLIBE_MODEL_BASENAME and not present <= _FLIBE_ELEMENTS:
        raise RuntimeError(
            f"The bundled Flibe model ('{_FLIBE_MODEL_BASENAME}') is trained only on Li/Be/F, "
            f"but the structure contains {sorted(present - _FLIBE_ELEMENTS)}. Running it would "
            f"produce meaningless results. Supply a model trained on this chemistry via --model."
        )


def run_npt(config: SimulationConfig) -> dict:
    """Run the full minimize -> equilibrate -> produce workflow and return results.

    Writes (into ``config.output_dir``): ``production.csv`` (state log used for the
    density average), ``equilibration.csv`` (diagnostics), ``trajectory.dcd``,
    ``checkpoint.chk``, and ``results.json`` (provenance + computed density). The
    returned dict is the same content as ``results.json``.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    import torch
    from openmmml import MLPotential

    from . import composition as comp_mod
    from .analysis import analyze_density

    os.makedirs(config.output_dir, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(config.output_dir, name)

    is_gpu = config.platform in _GPU_PLATFORMS
    # On ROCm builds PyTorch reports HIP devices through the CUDA API, so this
    # check covers both NVIDIA and AMD. Skip it for CPU-only runs.
    if is_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "Torch reports no GPU. Ensure the job is on a GPU node and the "
            "ROCm/CUDA PyTorch build is loaded (or set platform=CPU for a smoke test)."
        )

    logger.info("Loading structure from %s", config.structure_pdb)
    pdb = app.PDBFile(config.structure_pdb)
    topology = pdb.topology
    positions = pdb.positions

    box = topology.getPeriodicBoxVectors()
    if box is None:
        raise RuntimeError(
            f"No periodic box (CRYST1) in {config.structure_pdb}; NPT requires a box."
        )

    # Guard against running a salt through a potential not trained on its chemistry
    # (e.g. the bundled Flibe model on a chloride). Fail fast, before the expensive
    # system build, with a clear message.
    elements = [a.element.symbol for a in topology.atoms()]
    structure_elements = sorted(set(elements))
    validate_model_for_elements(
        config.model_file, structure_elements,
        supported_elements=model_supported_elements(config.model_file),
    )

    logger.info("Building MACE system from %s", config.model_file)
    ml = MLPotential("mace", modelPath=config.model_file)
    system = ml.createSystem(topology)
    system.setDefaultPeriodicBoxVectors(*box)

    temperature = config.temperature_K * unit.kelvin
    pressure = config.pressure_bar * unit.bar

    barostat = mm.MonteCarloBarostat(pressure, temperature, config.barostat_interval)
    barostat.setRandomNumberSeed(config.seed)
    system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(
        temperature,
        config.friction_per_ps / unit.picosecond,
        config.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(config.seed)

    plat, plat_name = _select_platform(mm, config.platform)
    properties = {}
    if plat_name in _GPU_PLATFORMS:
        properties = {
            "Precision": config.precision,
            "DeviceIndex": config.device_index,
            "DeterministicForces": "false",
        }

    simulation = app.Simulation(topology, system, integrator, plat, properties)

    started_from_checkpoint = False
    chk_path = out("checkpoint.chk")
    if config.restart and os.path.exists(chk_path):
        logger.info("Restarting from checkpoint %s", chk_path)
        simulation.loadCheckpoint(chk_path)
        started_from_checkpoint = True
    else:
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature, config.seed)

    wall_start = time.time()

    # --- Phase 1: energy minimization -------------------------------------
    if not started_from_checkpoint and config.minimize_iterations > 0:
        logger.info("Minimizing energy (max %d iterations)", config.minimize_iterations)
        simulation.minimizeEnergy(maxIterations=config.minimize_iterations)

    def make_state_reporter(path: str, interval: int):
        return app.StateDataReporter(
            path, interval, step=True, time=True,
            potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
            temperature=True, volume=True, density=True, speed=True,
        )

    # --- Phase 2: equilibration -------------------------------------------
    if not started_from_checkpoint and config.equilibration_steps > 0:
        logger.info("Equilibrating for %d steps", config.equilibration_steps)
        simulation.reporters.append(make_state_reporter(out("equilibration.csv"), config.report_interval))
        simulation.step(config.equilibration_steps)
        simulation.reporters.clear()

    # --- Phase 3: production ----------------------------------------------
    logger.info("Production for %d steps", config.production_steps)
    simulation.reporters.append(make_state_reporter(out("production.csv"), config.report_interval))
    if config.write_trajectory:
        simulation.reporters.append(app.DCDReporter(out("trajectory.dcd"), config.report_interval))
    if config.checkpoint_interval > 0:
        simulation.reporters.append(app.CheckpointReporter(chk_path, config.checkpoint_interval))
    simulation.step(config.production_steps)
    simulation.saveCheckpoint(chk_path)
    simulation.reporters.clear()

    wall_seconds = time.time() - wall_start

    # --- Density analysis on the production log ---------------------------
    # The production log contains production frames only, so no further
    # equilibration trimming is needed here (equilibration_fraction=0).
    density = analyze_density(out("production.csv"), equilibration_fraction=0.0)

    # Element counts are inferred directly from the topology (computed above) so
    # results.json is self-describing for ANY chemistry, even when the structure was
    # supplied by the user rather than built by us. The density itself comes from
    # OpenMM's mass/volume and is independent of this labelling.
    element_counts = {el: elements.count(el) for el in structure_elements}
    molar_mass = sum(comp_mod.atomic_mass(el) * n for el, n in element_counts.items())
    composition_block = {
        "element_counts": element_counts,
        "n_atoms": len(elements),
        "molar_mass_g_per_mol": round(molar_mass, 6),
    }
    if config.components:
        composition_block["components"] = config.components

    state_point = {
        "temperature_K": config.temperature_K,
        "pressure_bar": config.pressure_bar,
    }
    if config.composition_axis:
        state_point["composition_axis"] = config.composition_axis
    # Convenience: if this is recognizably a LiF-BeF2 (Flibe) system, also record
    # mol% BeF2 so the canonical composition axis is always available.
    if set(element_counts) <= {"Li", "Be", "F"} and element_counts.get("Be"):
        n_BeF2 = element_counts.get("Be", 0)
        n_LiF = element_counts.get("Li", 0)
        if n_LiF + n_BeF2 > 0:
            state_point["mole_percent_BeF2"] = round(100.0 * n_BeF2 / (n_LiF + n_BeF2), 4)

    results = {
        "schema": "saltmd.results/2",
        "state_point": state_point,
        "composition": composition_block,
        "density": density.as_dict(),
        "config": config.as_dict(),
        "run": {
            "platform": plat_name,
            "gpu": (torch.cuda.get_device_name(0) if (is_gpu and torch.cuda.is_available()) else None),
            "hostname": socket.gethostname(),
            "wall_seconds": round(wall_seconds, 1),
            "restarted": started_from_checkpoint,
            "n_atoms": len(elements),
        },
        "software": _software_versions(),
        "outputs": {
            "production_log": out("production.csv"),
            "trajectory": out("trajectory.dcd") if config.write_trajectory else None,
            "checkpoint": chk_path,
        },
    }

    with open(out("results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    formula = " ".join(f"{el}{n}" for el, n in element_counts.items())
    logger.info(
        "Done. Density = %.4f +/- %.4f g/cm^3 at %.1f K  [%s]",
        density.density_g_cm3, density.density_stderr_g_cm3,
        config.temperature_K, formula,
    )
    return results
