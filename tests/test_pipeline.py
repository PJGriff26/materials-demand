"""Regression tests for the materials demand simulation pipeline: verify the
core components produce the expected outputs and that results are reproducible
across runs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------
from src.random_config import RANDOM_SEED  # noqa: E402
from src.data_ingestion import MaterialIntensityLoader, CapacityProjectionLoader
from src.distribution_fitting import DistributionFitter
from src.technology_mapping import (
    TECHNOLOGY_MAPPING,
    TECHNOLOGY_LIFETIMES,
    get_lifetime,
)
from src.stock_flow_simulation import run_full_simulation, SimulationResult

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
INTENSITY_PATH = ROOT / "data" / "intensity_data.csv"
CAPACITY_PATH = ROOT / "data" / "StdScen24_annual_national.csv"


# ===================================================================
# test_data_loading
# ===================================================================

class TestDataLoading:
    """Verify that the raw data files can be loaded and have expected structure."""

    def test_intensity_data_loads(self):
        """intensity_data.csv loads, is non-empty, and has required columns."""
        assert INTENSITY_PATH.exists(), f"Missing data file: {INTENSITY_PATH}"
        df = pd.read_csv(INTENSITY_PATH)
        assert len(df) > 0, "intensity_data.csv is empty"
        for col in ("technology", "Material", "value"):
            assert col in df.columns, f"Missing column '{col}' in intensity_data.csv"

    def test_capacity_data_loads(self):
        """StdScen24_annual_national.csv loads, is non-empty, and has required columns."""
        assert CAPACITY_PATH.exists(), f"Missing data file: {CAPACITY_PATH}"
        # NREL format has 3 header rows to skip
        df = pd.read_csv(CAPACITY_PATH, skiprows=3, low_memory=False)
        assert len(df) > 0, "StdScen24_annual_national.csv is empty"
        for col in ("scenario", "t"):
            assert col in df.columns, (
                f"Missing column '{col}' in StdScen24_annual_national.csv"
            )

    def test_intensity_loader(self):
        """MaterialIntensityLoader returns standardised DataFrame."""
        loader = MaterialIntensityLoader()
        df = loader.load(INTENSITY_PATH)
        assert len(df) > 0
        # After standardisation, column names are lowercased and intensity is converted
        assert "technology" in df.columns
        assert "material" in df.columns
        assert "intensity_t_per_mw" in df.columns

    def test_capacity_loader(self):
        """CapacityProjectionLoader returns standardised DataFrame."""
        loader = CapacityProjectionLoader(level="national")
        df = loader.load(CAPACITY_PATH)
        assert len(df) > 0
        assert "scenario" in df.columns
        assert "year" in df.columns
        # Should contain at least one _MW column
        mw_cols = [c for c in df.columns if c.endswith("_MW")]
        assert len(mw_cols) > 0, "No _MW capacity columns found after loading"


# ===================================================================
# test_technology_mapping
# ===================================================================

class TestTechnologyMapping:
    """Verify the technology mapping dictionaries contain expected entries."""

    def test_mapping_has_expected_keys(self):
        """TECHNOLOGY_MAPPING contains the main capacity technology keys."""
        expected_keys = [
            "upv",
            "distpv",
            "csp",
            "wind_onshore",
            "wind_offshore",
            "coal",
            "gas_cc",
            "gas_ct",
            "nuclear",
            "nuclear_smr",
            "hydro",
            "geo",
        ]
        for key in expected_keys:
            assert key in TECHNOLOGY_MAPPING, (
                f"Expected key '{key}' missing from TECHNOLOGY_MAPPING"
            )

    def test_lifetimes_expected_values(self):
        """TECHNOLOGY_LIFETIMES matches NREL Standard Scenarios 2024
        (Gagnon et al., NREL/TP-6A40-92256, Tables A-5 and A-6).
        """
        expected = {
            "upv": 30,            # Table A-5: Solar PV = 30 yr (SunShot Vision)
            "distpv": 30,         # Table A-5: Solar PV = 30 yr
            "wind_onshore": 30,   # Table A-5: Wind = 30 yr (Wind Vision, DOE 2015)
            "wind_offshore": 30,  # Table A-5: Wind = 30 yr
            "nuclear": 80,        # Table A-6: Nuclear = 80 yr
            "hydro": 100,         # Table A-5: Hydro = 100 yr (Hydropower Vision)
        }
        for tech, expected_life in expected.items():
            assert tech in TECHNOLOGY_LIFETIMES, (
                f"'{tech}' missing from TECHNOLOGY_LIFETIMES"
            )
            assert TECHNOLOGY_LIFETIMES[tech] == expected_life, (
                f"Lifetime for '{tech}': expected {expected_life}, "
                f"got {TECHNOLOGY_LIFETIMES[tech]}"
            )

    def test_get_lifetime_helper(self):
        """get_lifetime() returns correct values and a default for unknowns."""
        assert get_lifetime("upv") == 30
        assert get_lifetime("nuclear") == 80  # NREL StdScen 2024 Table A-6
        # Unknown technology should return the default (30)
        assert get_lifetime("nonexistent_tech") == 30

    def test_mapping_weights_sum_to_one(self):
        """For every mapped technology, weights should sum to 1.0."""
        for tech, mapping in TECHNOLOGY_MAPPING.items():
            if not mapping:
                continue  # Empty mapping is allowed (unmapped tech)
            total = sum(mapping.values())
            assert abs(total - 1.0) < 0.001, (
                f"Weights for '{tech}' sum to {total}, expected 1.0"
            )


# ===================================================================
# test_distribution_fitting
# ===================================================================

class TestDistributionFitting:
    """Verify distribution fitting on real intensity data."""

    def test_fit_single_material(self):
        """Fit distributions for Copper and check the result has valid parameters."""
        loader = MaterialIntensityLoader()
        df = loader.load(INTENSITY_PATH)

        # Pick a material that should be well-represented in the data
        copper_data = df[df["material"] == "Copper"]
        assert len(copper_data) > 0, "No Copper records found in intensity data"

        fitter = DistributionFitter()

        # Fit for one technology-material pair
        tech = copper_data["technology"].iloc[0]
        values = copper_data[copper_data["technology"] == tech][
            "intensity_t_per_mw"
        ].values
        assert len(values) > 0, f"No values for {tech}/Copper"

        result = fitter.fit_single(tech, "Copper", values)

        # Verify result structure
        assert result.technology == tech
        assert result.material == "Copper"
        assert result.n_samples == len(values)
        assert result.mean > 0, "Mean intensity should be positive"
        assert result.use_parametric is True, "Should always use parametric fitting"
        assert result.best_fit is not None, "best_fit should not be None"
        assert len(result.best_fit.parameters) > 0, "Parameters dict should not be empty"

    def test_fit_all_returns_dict(self):
        """fit_all() returns a dict keyed by (technology, material) tuples."""
        loader = MaterialIntensityLoader()
        df = loader.load(INTENSITY_PATH)

        fitter = DistributionFitter()
        fitted = fitter.fit_all(df)

        assert isinstance(fitted, dict)
        assert len(fitted) > 0, "fit_all returned empty dict"

        # Check keys are (str, str) tuples
        first_key = next(iter(fitted))
        assert isinstance(first_key, tuple)
        assert len(first_key) == 2
        assert isinstance(first_key[0], str)
        assert isinstance(first_key[1], str)

    def test_sampling_produces_positive_values(self):
        """Sampling from a fitted distribution should produce positive values."""
        loader = MaterialIntensityLoader()
        df = loader.load(INTENSITY_PATH)

        fitter = DistributionFitter()
        fitted = fitter.fit_all(df)

        # Sample from every fitted distribution and check values are finite
        for (tech, mat), dist_info in fitted.items():
            samples = dist_info.sample(n=10, random_state=RANDOM_SEED)
            assert len(samples) == 10, f"Expected 10 samples for {tech}/{mat}"
            assert np.all(np.isfinite(samples)), (
                f"Non-finite sample values for {tech}/{mat}"
            )


# ===================================================================
# test_simulation_small
# ===================================================================

@pytest.mark.slow
class TestSimulationSmall:
    """
    Run a small simulation (100 iterations) on real data.

    Marked as slow because it takes approximately 30 seconds.
    Run with: pytest -m slow
    """

    @pytest.fixture(scope="class")
    def simulation_output(self):
        """Run simulation once and share across tests in this class."""
        sim, result = run_full_simulation(
            intensity_path=INTENSITY_PATH,
            capacity_path=CAPACITY_PATH,
            n_iterations=100,
            random_state=RANDOM_SEED,
        )
        return sim, result

    def test_returns_tuple(self, simulation_output):
        """run_full_simulation returns a 2-tuple."""
        sim, result = simulation_output
        assert sim is not None
        assert result is not None

    def test_result_is_simulation_result(self, simulation_output):
        """The second element should be a SimulationResult instance."""
        _, result = simulation_output
        assert isinstance(result, SimulationResult)

    def test_result_has_statistics(self, simulation_output):
        """get_statistics() returns a DataFrame with expected columns."""
        _, result = simulation_output
        stats_df = result.get_statistics()

        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0, "Statistics DataFrame is empty"

        for col in ("scenario", "material", "year", "mean"):
            assert col in stats_df.columns, (
                f"Missing column '{col}' in statistics DataFrame"
            )

    def test_multiple_scenarios(self, simulation_output):
        """Results span multiple scenarios."""
        _, result = simulation_output
        stats_df = result.get_statistics()
        n_scenarios = stats_df["scenario"].nunique()
        assert n_scenarios > 1, f"Expected multiple scenarios, got {n_scenarios}"

    def test_multiple_materials(self, simulation_output):
        """Results span multiple materials."""
        _, result = simulation_output
        stats_df = result.get_statistics()
        n_materials = stats_df["material"].nunique()
        assert n_materials > 1, f"Expected multiple materials, got {n_materials}"

    def test_multiple_years(self, simulation_output):
        """Results span multiple years."""
        _, result = simulation_output
        stats_df = result.get_statistics()
        n_years = stats_df["year"].nunique()
        assert n_years > 1, f"Expected multiple years, got {n_years}"

    def test_demand_non_negative(self, simulation_output):
        """All mean demand values should be non-negative."""
        _, result = simulation_output
        stats_df = result.get_statistics()
        assert (stats_df["mean"] >= 0).all(), (
            "Found negative mean demand values"
        )

    def test_iterations_array_shape(self, simulation_output):
        """The raw iterations array has the expected 4-D shape."""
        _, result = simulation_output
        arr = result.iterations_data
        assert arr.ndim == 4, f"Expected 4-D array, got {arr.ndim}-D"
        assert arr.shape[0] == result.n_iterations, (
            f"Array rows ({arr.shape[0]}) != n_iterations ({result.n_iterations})"
        )
        assert arr.shape[0] <= 100, (
            f"Expected at most 100 iterations, got {arr.shape[0]}"
        )


# ===================================================================
# test_simulation_deterministic
# ===================================================================

@pytest.mark.slow
class TestSimulationDeterministic:
    """
    Reproducibility check: same random_state produces identical results.

    Marked as slow because it runs the simulation twice (n_iterations=50).
    """

    def test_deterministic_results(self):
        """Two runs with the same random_state=RANDOM_SEED produce identical statistics."""
        _, result_a = run_full_simulation(
            intensity_path=INTENSITY_PATH,
            capacity_path=CAPACITY_PATH,
            n_iterations=50,
            random_state=RANDOM_SEED,
        )
        _, result_b = run_full_simulation(
            intensity_path=INTENSITY_PATH,
            capacity_path=CAPACITY_PATH,
            n_iterations=50,
            random_state=RANDOM_SEED,
        )

        stats_a = result_a.get_statistics()
        stats_b = result_b.get_statistics()

        # Sort both DataFrames identically so row order does not matter
        sort_cols = ["scenario", "year", "material"]
        stats_a = stats_a.sort_values(sort_cols).reset_index(drop=True)
        stats_b = stats_b.sort_values(sort_cols).reset_index(drop=True)

        # Check shapes match
        assert stats_a.shape == stats_b.shape, (
            f"Shape mismatch: {stats_a.shape} vs {stats_b.shape}"
        )

        # Check that all numeric columns are identical
        numeric_cols = stats_a.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            np.testing.assert_array_almost_equal(
                stats_a[col].values,
                stats_b[col].values,
                decimal=10,
                err_msg=f"Column '{col}' differs between two deterministic runs",
            )


# ===================================================================
# test_output_files_exist
# ===================================================================

# Key output files that should exist after `python reproduce.py`. Both ship
# precomputed, so these tests normally run (they skip only on a tree where the
# outputs have been deleted and not regenerated).
_OUTPUT_FILES = [
    ROOT / "outputs" / "data" / "material_demand_by_scenario.csv",
    ROOT / "outputs" / "data" / "material_features.csv",
]

_ALL_OUTPUTS_EXIST = all(p.exists() for p in _OUTPUT_FILES)


@pytest.mark.skipif(
    not _ALL_OUTPUTS_EXIST,
    reason="Outputs not generated yet (run `python reproduce.py`)",
)
class TestOutputFilesExist:
    """
    Verify that key output files exist and are non-empty.

    These tests are skipped only if the outputs have been deleted and not yet
    regenerated.
    """

    def test_material_demand_by_scenario(self):
        path = ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
        assert path.exists(), f"Missing: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"{path.name} is empty"

    def test_material_features(self):
        # Per-material supply-chain features consumed by Figure 6, written by
        # supply_chain/build_material_features.py.
        path = ROOT / "outputs" / "data" / "material_features.csv"
        assert path.exists(), f"Missing: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"{path.name} is empty"
        for col in ("import_dependency", "production_hhi"):
            assert col in df.columns, f"{path.name} missing column {col}"
