"""Tests for the model/chemistry compatibility guard (no torch/openmm needed)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saltmd.simulation import validate_model_for_elements


def test_known_coverage_accepts_subset():
    # Model covers Li/Be/F; a Flibe structure is fine.
    validate_model_for_elements("any.model", ["F", "Li", "Be"],
                                supported_elements={"Li", "Be", "F"})


def test_known_coverage_rejects_missing_element():
    # Model covers only Li/Be/F; a chloride structure must be refused.
    with pytest.raises(RuntimeError) as e:
        validate_model_for_elements("any.model", ["Na", "Cl"],
                                    supported_elements={"Li", "Be", "F"})
    assert "Cl" in str(e.value)


def test_flibe_name_guard_blocks_other_chemistry():
    # Coverage unknown, but the bundled Flibe model must not touch non-Li/Be/F.
    with pytest.raises(RuntimeError) as e:
        validate_model_for_elements("assets/mace_flibe.model", ["Li", "Na", "K", "F"],
                                    supported_elements=None)
    assert "Flibe" in str(e.value)


def test_flibe_name_guard_allows_flibe():
    validate_model_for_elements("assets/mace_flibe.model", ["F", "Li", "Be"],
                                supported_elements=None)


def test_unknown_model_unknown_coverage_is_permissive():
    # A user-supplied model we can't introspect and isn't the bundled Flibe one:
    # we don't block (we can't know), we trust the user's --model choice.
    validate_model_for_elements("/models/mace_flinak.model", ["Li", "Na", "K", "F"],
                                supported_elements=None)
