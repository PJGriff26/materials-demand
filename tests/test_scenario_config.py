"""Guards for the canonical representative-scenario config (src/scenario_config.py).

Added 2026-07-28 after the CO2e scenario mix-up: the manuscript's prose
percentages had been computed from the Mid_Case_CO2e_* ensemble members while
every figure and table used the non-CO2e runs. These tests pin the 2026-07-27
decision (representative scenarios = the non-CO2e runs) so the choice cannot
drift silently again.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scenario_config import (  # noqa: E402
    REFERENCE_SCENARIO,
    REPRESENTATIVE_SCENARIO_KEYS,
    SCENARIO_COLORS,
    SCENARIO_LABELS,
)


def test_reference_is_representative():
    assert REFERENCE_SCENARIO in REPRESENTATIVE_SCENARIO_KEYS


def test_representative_set_pinned():
    # Decision 2026-07-28 (supersedes 2026-07-27): net-zero highlights are the
    # CO2e-constrained runs; baselines are unconstrained and family-neutral.
    assert set(REPRESENTATIVE_SCENARIO_KEYS) == {
        "Mid_Case", "Mid_Case_No_IRA",
        "Mid_Case_CO2e_100by2035", "Mid_Case_CO2e_95by2050"}


def test_labels_and_colors_cover_all_keys():
    assert set(SCENARIO_LABELS) == set(REPRESENTATIVE_SCENARIO_KEYS)
    assert set(SCENARIO_COLORS) == set(REPRESENTATIVE_SCENARIO_KEYS)
    assert len(set(SCENARIO_LABELS.values())) == 4, "labels must be distinct"


def test_keys_exist_in_demand_csv():
    csv = ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
    scen = set(pd.read_csv(csv, usecols=["scenario"])["scenario"].unique())
    assert set(REPRESENTATIVE_SCENARIO_KEYS) <= scen
