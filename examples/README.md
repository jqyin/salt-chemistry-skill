# Examples

## `output_npt.out`

A **short demonstration** state log (10 frames, 1 ps of dynamics) from a 33.33 mol%
BeF₂ Flibe NPT run on Frontier. It exists so the analysis/plotting tools can be
exercised without a GPU — it is **not a converged production result**.

Treat it as a wiring example, not a physical answer:

- It is only 1 ps; density is still relaxing (it drifts ~2.02 → 2.00 g/cm³), so the
  tiny error bar from `analyze_density.py` reflects the spread of a handful of
  correlated frames, **not** a true equilibrium uncertainty.
- A real run uses the defaults in `templates/config.example.yaml` (50 ps
  equilibration + 250 ps production) and reports density from the production phase
  only. See `reference/methodology.md` for what "converged" means.

Demo command:
```bash
python scripts/analyze_density.py examples/output_npt.out --equilibration-fraction 0.5
```
