"""Parse OpenMM state logs and compute equilibrium density with error bars.

The physical quantity of interest is the **mean mass density of the equilibrated
liquid**, with an honest statistical uncertainty. Two things matter for that to
be meaningful:

1. **Discard equilibration.** The early part of an NPT trajectory is the box
   relaxing from its initial size toward the equilibrium density; averaging over
   it biases the result. We drop a leading fraction (or a user-specified time).

2. **Account for autocorrelation.** Consecutive MD frames are correlated, so the
   naive standard error (std / sqrt(N)) underestimates the true uncertainty. We
   report a block-averaged standard error and an autocorrelation-corrected one
   (via the statistical inefficiency g), which agree when blocks are long enough.

Only numpy/pandas are needed here, so analysis runs anywhere — typically on a
login node after the GPU job finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Column name OpenMM's StateDataReporter uses for density, and our canonical key.
_DENSITY_COL = "Density (g/mL)"
_STEP_COL = "Step"
_TIME_COL = "Time (ps)"
_TEMP_COL = "Temperature (K)"
_VOLUME_COL = "Box Volume (nm^3)"


def parse_state_log(path: str) -> pd.DataFrame:
    """Read a StateDataReporter CSV, tolerating leading non-CSV header lines.

    The reporter writes a header row beginning with ``#"Step",...`` followed by
    numeric rows. Our run driver may also write a human-readable banner line
    first; this finds the real header by looking for the ``Step`` column and
    parses from there.
    """
    with open(path) as fh:
        raw = fh.read().splitlines()

    header_idx = None
    for i, line in enumerate(raw):
        stripped = line.lstrip("#").strip()
        if stripped.startswith('"Step"') or stripped.startswith("Step"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"Could not find a StateDataReporter header (a line with 'Step') in {path}"
        )

    columns = [c.strip().strip('"') for c in raw[header_idx].lstrip("#").split(",")]
    data_rows = []
    for line in raw[header_idx + 1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != len(columns):
            continue  # skip any trailing/banner lines
        try:
            data_rows.append([float(p) for p in parts])
        except ValueError:
            continue

    if not data_rows:
        raise ValueError(f"No numeric data rows found in {path}")

    return pd.DataFrame(data_rows, columns=columns)


def block_average(values: np.ndarray, n_blocks: int = 5) -> tuple[float, float]:
    """Mean and block-averaged standard error of the mean.

    Splitting the series into ``n_blocks`` contiguous blocks and taking the
    standard error of the block means accounts for short-range autocorrelation:
    as long as each block is much longer than the correlation time, the block
    means are effectively independent.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    n_blocks = max(1, min(n_blocks, n))
    if n_blocks == 1:
        return mean, float("nan")
    block_size = n // n_blocks
    trimmed = values[: block_size * n_blocks]
    block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    # Standard error of the mean across blocks (ddof=1 for sample std).
    stderr = float(block_means.std(ddof=1) / np.sqrt(n_blocks))
    return mean, stderr


def statistical_inefficiency(values: np.ndarray) -> float:
    """Statistical inefficiency g = 1 + 2*tau (integrated autocorrelation).

    g is roughly the number of correlated samples per independent sample, so the
    effective sample size is N/g. Computed by summing the normalized
    autocorrelation function until it first goes non-positive.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return 1.0
    x = values - values.mean()
    var = float((x * x).mean())
    if var == 0.0:
        return 1.0
    g = 1.0
    for t in range(1, n):
        c_t = float((x[: n - t] * x[t:]).mean()) / var
        if c_t <= 0.0:
            break
        g += 2.0 * (1.0 - t / n) * c_t
    return max(1.0, g)


@dataclass
class DensityResult:
    """Equilibrium density estimate plus the diagnostics behind it."""

    density_g_cm3: float
    density_stderr_g_cm3: float           # block-averaged SEM (headline error bar)
    density_stderr_autocorr_g_cm3: float  # autocorrelation-corrected SEM (cross-check)
    density_std_g_cm3: float              # raw spread of production samples
    mean_temperature_K: float
    mean_volume_nm3: float
    n_samples_total: int
    n_samples_production: int
    n_samples_discarded: int
    equilibration_fraction: float
    statistical_inefficiency: float
    effective_sample_size: float

    def as_dict(self) -> dict:
        return asdict(self)


def analyze_density(
    log_path: str,
    equilibration_fraction: float = 0.5,
    equilibration_time_ps: float | None = None,
    n_blocks: int = 5,
) -> DensityResult:
    """Compute the equilibrium density from a state log.

    Equilibration is removed either by ``equilibration_time_ps`` (drop all frames
    with Time < that value) if given, otherwise by dropping the leading
    ``equilibration_fraction`` of frames (default: the first half — a safe,
    conservative choice for the short, fast-equilibrating density runs here).
    """
    df = parse_state_log(log_path)
    if _DENSITY_COL not in df.columns:
        raise ValueError(
            f"'{_DENSITY_COL}' column not found in {log_path}; "
            f"available columns: {list(df.columns)}"
        )

    n_total = len(df)

    if equilibration_time_ps is not None and _TIME_COL in df.columns:
        prod = df[df[_TIME_COL] >= equilibration_time_ps]
        n_discard = n_total - len(prod)
        equil_frac_effective = n_discard / n_total if n_total else 0.0
    else:
        if not 0.0 <= equilibration_fraction < 1.0:
            raise ValueError(
                f"equilibration_fraction must be in [0, 1), got {equilibration_fraction}"
            )
        n_discard = int(round(equilibration_fraction * n_total))
        prod = df.iloc[n_discard:]
        equil_frac_effective = equilibration_fraction

    if len(prod) == 0:
        raise ValueError(
            "No production frames left after discarding equilibration; "
            "lower equilibration_fraction or run longer."
        )

    density = prod[_DENSITY_COL].to_numpy()
    mean, stderr_block = block_average(density, n_blocks=n_blocks)
    g = statistical_inefficiency(density)
    n_eff = len(density) / g
    stderr_autocorr = float(density.std(ddof=1) / np.sqrt(n_eff)) if len(density) > 1 else float("nan")

    mean_temp = float(prod[_TEMP_COL].mean()) if _TEMP_COL in prod else float("nan")
    mean_vol = float(prod[_VOLUME_COL].mean()) if _VOLUME_COL in prod else float("nan")

    return DensityResult(
        density_g_cm3=mean,
        density_stderr_g_cm3=stderr_block,
        density_stderr_autocorr_g_cm3=stderr_autocorr,
        density_std_g_cm3=float(density.std(ddof=1)) if len(density) > 1 else float("nan"),
        mean_temperature_K=mean_temp,
        mean_volume_nm3=mean_vol,
        n_samples_total=n_total,
        n_samples_production=len(density),
        n_samples_discarded=n_discard,
        equilibration_fraction=round(equil_frac_effective, 4),
        statistical_inefficiency=round(g, 3),
        effective_sample_size=round(n_eff, 2),
    )
