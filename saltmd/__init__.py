"""saltmd — molten-salt molecular dynamics for chemistry-property prediction.

This package drives OpenMM NPT molecular dynamics of molten-salt mixtures using a
machine-learning interatomic potential (MACE), to predict equilibrium properties
such as mass density as a function of composition and temperature.

The composition/structure/analysis machinery is chemistry-agnostic — it works for
any salt described as a mixture of components (LiF-BeF2 "Flibe", FLiNaK, chlorides,
...). The one chemistry-specific ingredient is the ML potential: each salt family
needs a model trained on it. Only Flibe ships with a potential here
(``assets/mace_flibe.model``); other chemistries require you to supply a matching
model. See :mod:`saltmd.presets` for known salt compositions.

Modules
-------
composition : pure composition <-> geometry math for arbitrary mixtures (no heavy deps)
presets     : named salt compositions (Flibe, FLiNaK, LiCl-KCl, ...)
structure   : build a randomized periodic box at a target composition/density
simulation  : configure and run the NPT simulation (imports OpenMM lazily)
analysis    : parse simulation logs and compute equilibrium density with error bars

The package is intentionally split so that planning/analysis code runs anywhere,
while the GPU-bound `simulation` module only imports OpenMM/MACE/PyTorch when a GPU
stack is available (i.e. on a compute node).
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import composition  # noqa: F401  (pure, always importable)
from . import presets      # noqa: F401

__all__ = ["composition", "presets", "__version__"]
