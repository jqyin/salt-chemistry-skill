# Output files and the `results.json` schema

Every NPT run writes its outputs into its `--output-dir`. This document is the
contract for parsing them programmatically.

## Files written per run

| File | Written by | Contents |
|------|-----------|----------|
| `results.json` | `run_npt.py` | The headline result + full provenance (below). **Parse this.** |
| `production.csv` | StateDataReporter | Production-phase state log (the frames averaged for density). |
| `equilibration.csv` | StateDataReporter | Equilibration-phase state log (diagnostics only; check for a plateau). |
| `trajectory.dcd` | DCDReporter | Production trajectory (atomic coordinates over time). |
| `checkpoint.chk` | CheckpointReporter | Binary restart point; resume with `--restart`. |

The CSV logs are standard OpenMM `StateDataReporter` output, with a header line
beginning `#"Step",...` followed by numeric rows. Columns: Step, Time (ps),
Potential/Kinetic/Total Energy (kJ/mole), Temperature (K), Box Volume (nm³),
Density (g/mL), Speed (ns/day). `saltmd.analysis.parse_state_log` reads them and
tolerates a leading banner line.

## `results.json` schema

The schema is `saltmd.results/2`. It is **chemistry-agnostic**: `composition` always
carries generic per-element counts inferred from the topology (correct for any
salt), and `state_point` carries a generic `composition_axis` plus, for Flibe, the
convenience `mole_percent_BeF2`.

```jsonc
{
  "schema": "saltmd.results/2",
  "state_point": {
    "temperature_K": 783.15,
    "pressure_bar": 1.0,
    "composition_axis": { "label": "BeF2 (mol%)", "value": 33.33 }, // generic 1-D axis for plots
    "mole_percent_BeF2": 33.3333      // only present for LiF-BeF2 systems (convenience)
  },
  "composition": {
    "element_counts": { "F": 960, "Li": 480, "Be": 240 },  // always present, any chemistry
    "n_atoms": 1680,
    "molar_mass_g_per_mol": 23733.07,
    "components": [["LiF", 66.67], ["BeF2", 33.33]]          // present if the builder recorded it
  },
  "density": {
    "density_g_cm3": 1.987,                  // <-- the answer
    "density_stderr_g_cm3": 0.004,           // <-- headline error bar (block average)
    "density_stderr_autocorr_g_cm3": 0.005,  // cross-check (autocorrelation-corrected)
    "density_std_g_cm3": 0.03,               // raw spread of production frames
    "mean_temperature_K": 783.0,             // should match the setpoint
    "mean_volume_nm3": 19.6,
    "n_samples_total": 2500,
    "n_samples_production": 2500,
    "n_samples_discarded": 0,
    "equilibration_fraction": 0.0,           // 0 here: equilibration was a separate phase
    "statistical_inefficiency": 3.2,         // g = 1 + 2*tau
    "effective_sample_size": 781.0           // N_eff = N/g
  },
  "config": { /* the full SimulationConfig used (every parameter) */ },
  "run": {
    "platform": "HIP",
    "gpu": "AMD Instinct MI250X",
    "hostname": "frontier01234",
    "wall_seconds": 540.2,
    "restarted": false,
    "n_atoms": 1680
  },
  "software": {
    "python": "3.12.x", "openmm": "8.x", "torch": "2.10.0",
    "torch_hip": "...", "mace": "0.3.16"
  },
  "outputs": {
    "production_log": "runs/flibe_783K/production.csv",
    "trajectory": "runs/flibe_783K/trajectory.dcd",
    "checkpoint": "runs/flibe_783K/checkpoint.chk"
  }
}
```

### How to report a single result

Read `density.density_g_cm3` and `density.density_stderr_g_cm3` and present them
together with the conditions, e.g.:

> Density of 33.33 mol% BeF₂ Flibe at 783 K: **1.987 ± 0.004 g/cm³**
> (250 ps production, MACE potential).

### Reading many results (a campaign)

Each state point has its own `results.json`. `plot_results.py` ingests a glob of
them and emits a figure plus a CSV with columns
`temperature_K, composition_mol_percent, density_g_cm3, stderr_g_cm3, source`. It
reads the composition from `state_point.composition_axis.value` (falling back to
`mole_percent_BeF2` for legacy Flibe results), so it plots any chemistry. To do
custom analysis, load the JSONs directly and pull `state_point` + `density`.

## Re-analysis

To recompute density from a log with a different equilibration cutoff, or to
analyze a **combined/legacy** log (equilibration + production in one file, like the
shipped `examples/output_npt.out`), use `analyze_density.py`:

```bash
# production-only log -> discard nothing
python scripts/analyze_density.py runs/flibe_783K/production.csv --equilibration-fraction 0

# combined log -> drop the first half as equilibration
python scripts/analyze_density.py examples/output_npt.out --equilibration-fraction 0.5
```
