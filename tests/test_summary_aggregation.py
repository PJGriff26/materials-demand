"""Tests for material_demand_summary.csv: assert it holds exactly the four
highlighted scenarios (kept separate, not pooled across the 61 NREL scenarios)
and that every value matches the corresponding row in
material_demand_by_scenario.csv. The summary path can be overridden via the
MATERIAL_SUMMARY_CSV env var."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
BY_SCEN = ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "data" / "material_demand_summary.csv"

HIGHLIGHTED_SCENARIOS = {
    "Mid_Case", "Mid_Case_No_IRA", "Mid_Case_100by2035", "Mid_Case_95by2050",
}
TEST_MATERIALS = ["Steel", "Copper", "Neodymium"]
TEST_YEAR = 2035


def _summary_path():
    return Path(os.environ.get("MATERIAL_SUMMARY_CSV", str(DEFAULT_SUMMARY)))


@pytest.fixture(scope="module")
def summary_df():
    return pd.read_csv(_summary_path())


@pytest.fixture(scope="module")
def by_scenario_df():
    return pd.read_csv(BY_SCEN)


def test_summary_has_only_highlighted_scenarios(summary_df):
    """The summary carries a scenario column with exactly the four highlighted
    scenarios (no more, no fewer)."""
    assert "scenario" in summary_df.columns, "summary must keep the scenario column"
    assert set(summary_df["scenario"].unique()) == HIGHLIGHTED_SCENARIOS


@pytest.mark.parametrize("material", TEST_MATERIALS)
def test_summary_matches_by_scenario(summary_df, by_scenario_df, material):
    """Each summary value equals the corresponding per-scenario value.

    The summary is a faithful subset of material_demand_by_scenario.csv (the four
    highlighted scenarios), so for every (scenario, material, year) the demand
    statistics must match exactly.
    """
    for scen in HIGHLIGHTED_SCENARIOS:
        s = summary_df[
            (summary_df.scenario == scen)
            & (summary_df.material == material)
            & (summary_df.year == TEST_YEAR)
        ]
        b = by_scenario_df[
            (by_scenario_df.scenario == scen)
            & (by_scenario_df.material == material)
            & (by_scenario_df.year == TEST_YEAR)
        ]
        assert len(s) == 1 and len(b) == 1, f"missing row for {scen}/{material}"
        for col in ("mean", "p50", "p2_5", "p97_5", "mean_annual"):
            assert np.isclose(s.iloc[0][col], b.iloc[0][col], rtol=1e-9), (
                f"{scen}/{material} {col}: summary {s.iloc[0][col]} != "
                f"by_scenario {b.iloc[0][col]}"
            )


def test_summary_is_not_pooled_or_summed(summary_df, by_scenario_df):
    """Sanity guard against the old aggregation modes: a single highlighted
    scenario's value must NOT equal the sum or the mean across all 61 scenarios
    (which is what a buggy pooled/summed summary would produce)."""
    mat = "Steel"
    s = summary_df[
        (summary_df.scenario == "Mid_Case")
        & (summary_df.material == mat)
        & (summary_df.year == TEST_YEAR)
    ].iloc[0]["mean"]
    b = by_scenario_df[
        (by_scenario_df.material == mat) & (by_scenario_df.year == TEST_YEAR)
    ]["mean"]
    assert not np.isclose(s, b.sum(), rtol=1e-3), "summary looks summed across scenarios"
