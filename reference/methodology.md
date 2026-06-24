# Methodology: NPT density of Flibe from MACE-driven MD

This document explains the science the scripts implement, so you can judge whether
a result is trustworthy and adjust the protocol when it is not.

## Contents
- [The system: Flibe (LiF–BeF₂)](#the-system-flibe-lifbef2)
- [Why NPT for density](#why-npt-for-density)
- [The minimize → equilibrate → produce protocol](#the-minimize--equilibrate--produce-protocol)
- [Computing density with honest error bars](#computing-density-with-honest-error-bars)
- [Choosing run length and system size](#choosing-run-length-and-system-size)
- [Validation and sanity checks](#validation-and-sanity-checks)
- [The MACE potential and its limits](#the-mace-potential-and-its-limits)

## The system: Flibe (LiF–BeF₂)

Flibe is a binary molten salt of lithium fluoride (LiF) and beryllium fluoride
(BeF₂). Composition is conventionally given as the **mole percent of BeF₂**. Key
points on the axis:

| mol% BeF₂ | meaning |
|-----------|---------|
| 0         | pure LiF |
| 33.33     | 2LiF·BeF₂ — the eutectic fusion coolant/breeder salt |
| 50        | LiF·BeF₂ |
| 100       | pure BeF₂ |

Atom counts for `n` formula units at mol% BeF₂ = `x`:
`n_BeF2 = round(x/100 · n)`, `n_LiF = n − n_BeF2`, giving `Li = n_LiF`,
`Be = n_BeF2`, `F = n_LiF + 2·n_BeF2`. (`saltmd.composition` does this and reports
the *achieved* mol% after rounding — record that, not the requested value.)

Flibe is of interest for fusion because the ⁶Li and Be act as a tritium breeder
and neutron multiplier. The **tritium breeding ratio is a neutronics quantity**
computed by transport codes, not by this skill — but those calculations need the
salt's **density** at operating temperature, which is exactly what this skill
provides.

**Other salts.** Nothing in the NPT density protocol below is specific to Flibe.
The composition math (`saltmd.composition`) represents any salt as a mixture of
components (FLiNaK = LiF–NaF–KF, LiCl–KCl, NaCl–MgCl₂, …), and the same
build → run → analyze → plot pipeline applies. The one chemistry-specific
ingredient is the **ML potential**: each salt family needs a model trained on its
elements. Only Flibe ships with one here; for chlorides note their lower liquid
density (~1.5–1.8 g/cm³) when choosing an initial box density.

## Why NPT for density

Density is a property of the equilibrium liquid at a given temperature and
pressure, so we sample the **isothermal–isobaric (NPT) ensemble**: the box volume
fluctuates under a barostat until the average pressure matches the setpoint, and
the equilibrium density is `mass / ⟨volume⟩`.

Implementation (`saltmd.simulation`):
- **Thermostat/integrator**: `LangevinMiddleIntegrator` at the target temperature
  with a friction of 10 ps⁻¹ — strong enough to thermalize an ionic melt quickly,
  weak enough not to over-damp the dynamics.
- **Barostat**: `MonteCarloBarostat` at 1 bar, attempting volume moves every 50
  steps. (Pressure has a negligible effect on a dense liquid's density, so 1 bar
  is fine for any realistic operating pressure.)
- **Timestep**: 1 fs. The light Li/Be/F masses and stiff ML forces make longer
  steps risky; 1 fs is a safe default.

## The minimize → equilibrate → produce protocol

The driver runs three explicit phases. Separating them is what makes the reported
density unbiased:

1. **Minimize** (`minimize_iterations`, default 200). Relaxes any close contacts
   from the randomized initial box so the first dynamics steps don't see enormous
   forces. Cheap insurance.
2. **Equilibrate** (`equilibration_steps`, default 50 000 = 50 ps). The box
   contracts/expands from its initial guess toward the equilibrium density and the
   system loses memory of its lattice-like start. **These frames are written to
   `equilibration.csv` and are *not* averaged** — including them would bias the
   density toward the initial guess.
3. **Produce** (`production_steps`, default 250 000 = 250 ps). The equilibrated
   liquid is sampled; **only these frames** (`production.csv`) enter the density
   average.

To check equilibration was long enough, look at `equilibration.csv`: density and
volume should plateau (no systematic drift) well before the phase ends. If they
are still trending, increase `equilibration_steps` and rerun.

## Computing density with honest error bars

A density without an uncertainty is incomplete. `saltmd.analysis` reports:

- **Mean density** over production frames (`density_g_cm3`).
- **Block-averaged standard error** (`density_stderr_g_cm3`, the headline error
  bar). Production frames are split into `n_blocks` (default 5) contiguous blocks;
  the standard error of the block means is robust to the short-range correlation
  between consecutive MD frames, provided each block is much longer than the
  correlation time.
- **Autocorrelation-corrected standard error** (`density_stderr_autocorr_g_cm3`, a
  cross-check) using the statistical inefficiency `g = 1 + 2τ` and effective sample
  size `N_eff = N/g`. The two error estimates should agree to within a factor of
  ~2; if they diverge badly, the run is too short or blocks too few.

Rule of thumb: aim for a relative error (stderr / density) below ~0.5% before
treating a density as "converged." If it is larger, extend production.

## Choosing run length and system size

- **System size** (`--n-formula-units`): 720 (≈1680 atoms) is a good default —
  large enough to suppress finite-size artifacts in a simple liquid, small enough
  to be cheap. Double it if you need tighter error bars or want to check
  size-dependence.
- **Production length**: 250 ps is a reasonable default for density. Density
  equilibrates and decorrelates quickly in a hot ionic liquid, so this usually
  gives sub-0.5% error. Transport properties (viscosity, diffusion) would need far
  longer and are out of scope here.
- **Temperature scan**: reuse the same composition; build one structure per
  temperature (or let `make_campaign.py` do it). Starting each near the expected
  density keeps equilibration short.

## Validation and sanity checks

- **Mean temperature** in `results.json` should match the setpoint within a few K;
  a large mismatch signals a thermostat or setup problem.
- **Experimental anchor**: for the 33.33 mol% eutectic, measured density is about
  **1.94 g/cm³ near 783 K** with a small negative temperature slope
  (~−2–3 × 10⁻⁴ g/cm³/K). A MACE prediction within ~1–3% and with the right slope
  sign is reasonable; a wild departure points to a model or setup issue.
- **Initial vs final density**: the short sample run shipped in `examples/`
  relaxes from ~2.03 g/cm³ (initial pack) toward ~2.0 g/cm³ over 1 ps — confirming
  the box is contracting toward equilibrium, not exploding.

## The MACE potential and its limits

`mace_flibe.model` is a MACE model fine-tuned on DFT data for **LiF–BeF₂**. It sets
the accuracy ceiling: results are only as good as the potential.

- Use it **only for Li/Be/F** systems. Other elements are out of distribution.
- Predicted densities are **potential-limited**; quote them as such and validate
  against experiment where data exist.
- Extrapolating far outside the fitted composition/temperature range (e.g. very
  high BeF₂, very high T) is riskier — flag such requests.
