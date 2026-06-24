"""Tests for density analysis (parsing, equilibration, error bars)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.analysis import (
    parse_state_log,
    block_average,
    statistical_inefficiency,
    analyze_density,
)

EXAMPLE_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "output_npt.out",
)


def _write_synthetic_log(path, n=2000, mean=1.95, noise=0.01, seed=0):
    """A StateDataReporter-style log with a short equilibration ramp."""
    rng = np.random.default_rng(seed)
    header = ('#"Step","Time (ps)","Potential Energy (kJ/mole)","Kinetic Energy (kJ/mole)",'
              '"Total Energy (kJ/mole)","Temperature (K)","Box Volume (nm^3)",'
              '"Density (g/mL)","Speed (ns/day)"')
    rows = [header]
    for i in range(n):
        t = (i + 1) * 0.1
        # First 10% ramps from a low density up to the plateau.
        ramp = max(0.0, 1.0 - i / (0.1 * n))
        dens = mean - 0.2 * ramp + rng.normal(0, noise)
        vol = 19.6 + rng.normal(0, 0.05)
        rows.append(f"{(i+1)*100},{t},-1.0e6,1.6e4,-1.0e6,783.0,{vol},{dens},200.0")
    open(path, "w").write("\n".join(rows) + "\n")


def test_block_average_basic():
    x = np.ones(100) * 2.0
    mean, stderr = block_average(x, n_blocks=5)
    assert mean == pytest.approx(2.0)
    assert stderr == pytest.approx(0.0, abs=1e-12)


def test_statistical_inefficiency_iid_is_near_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    g = statistical_inefficiency(x)
    assert 0.7 < g < 2.0  # uncorrelated data -> g ~ 1


def test_analyze_synthetic_recovers_plateau(tmp_path):
    path = str(tmp_path / "production.csv")
    _write_synthetic_log(path, n=2000, mean=1.95, noise=0.01)
    # Discard the first 10% ramp; the rest should recover the plateau density.
    res = analyze_density(path, equilibration_fraction=0.1)
    assert res.density_g_cm3 == pytest.approx(1.95, abs=0.01)
    assert res.density_stderr_g_cm3 > 0
    assert res.n_samples_production == 1800


def test_parse_legacy_example_log():
    if not os.path.exists(EXAMPLE_LOG):
        pytest.skip("example log not present")
    df = parse_state_log(EXAMPLE_LOG)
    assert "Density (g/mL)" in df.columns
    assert len(df) >= 5
