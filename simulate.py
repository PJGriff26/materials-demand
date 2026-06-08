"""Monte Carlo simulation entry point: runs the complete stock-flow simulation
that produces the canonical demand CSVs at outputs/data/material_demand_*.csv,
which every figure and table in this release consumes. The run is deterministic
(fixed seed), so it reproduces the shipped results exactly."""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path for imports. simulate.py lives at the repository
# root, so the root is simply this file's parent.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Application-level logging config. Library modules under src/ only
# acquire loggers via logging.getLogger(__name__); the entry point is
# responsible for output formatting.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Import our modules
from src.stock_flow_simulation import (
    run_full_simulation,
    MaterialsStockFlowSimulation,
)
from src.random_config import RANDOM_SEED
from src.annualization import add_annualized_columns

# Stat columns produced by SimulationResult.get_statistics. Each gets a
# companion `*_annual` column written to the output CSVs (see
# src/annualization.py for rationale).
STAT_COLS = ['mean', 'std', 'p2', 'p5', 'p25', 'p50', 'p75', 'p95', 'p97']


# ============================================================================
# CONFIGURATION - Edit these paths for your system
# ============================================================================

# Input data directory
DATA_DIR = ROOT / 'data'

# Output directory
OUTPUT_DIR = ROOT / 'outputs'
DATA_OUT_DIR = OUTPUT_DIR / 'data'
DATA_OUT_DIR.mkdir(exist_ok=True, parents=True)

# Input files
INTENSITY_FILE = DATA_DIR / 'intensity_data.csv'
CAPACITY_FILE = DATA_DIR / 'StdScen24_annual_national.csv'

# Output files — all MC outputs live under outputs/data/ so downstream
# visualizations (which all read from outputs/data/) see the latest run.
DETAILED_RESULTS_FILE = DATA_OUT_DIR / 'material_demand_by_scenario.csv'
SUMMARY_RESULTS_FILE = DATA_OUT_DIR / 'material_demand_summary.csv'
REPORT_FILE = OUTPUT_DIR / 'simulation_report.txt'

# Simulation parameters
N_ITERATIONS = 10000
# RANDOM_SEED imported from src.random_config (single source of truth; override
# with MATERIALS_DEMAND_SEED env var). Canonical value: 24601.

# The four scenarios highlighted throughout the manuscript. material_demand_
# summary.csv is written for exactly these (per scenario); the full 61-scenario
# detail lives in material_demand_by_scenario.csv.
HIGHLIGHTED_SCENARIOS = [
    "Mid_Case",            # reference / central estimate
    "Mid_Case_No_IRA",     # Mid Case without the Inflation Reduction Act
    "Mid_Case_100by2035",  # 100% clean electricity by 2035
    "Mid_Case_95by2050",   # 95% clean electricity by 2050
]

# Focus scenarios (None = all scenarios)
FOCUS_SCENARIOS = None  # or ['Mid_Case', 'Mid_Case_No_IRA', 'Mid_Case_100by2035']

# ============================================================================


def _parse_args():
    """Parse CLI args. Default behavior is unchanged from pre-April 2026."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--interpolation",
        choices=["none", "linear", "mean-preserving", "pchip"],
        default="pchip",
        help=(
            "Interpolate NREL 3-yr capacity stocks to a 1-year grid before "
            "running stock-flow. "
            "'pchip' (NEW DEFAULT, 2026-04-28) = monotone cubic Hermite "
            "(Fritsch & Carlson 1980), shape-preserving, no overshoot. "
            "Output goes to canonical outputs/data/ (annual cadence). "
            "'none' = legacy 3-yr-bucket production behavior; output goes "
            "to outputs/data/legacy_3yr_bucket/ so existing scripts that "
            "still read 3-yr buckets keep working. "
            "'linear' = piecewise-linear stock interpolation (Watari 2019 "
            "convention); writes to outputs/data/interpolated/linear/. "
            "'mean-preserving' = cubic-spline of bucket-average rates + "
            "per-bucket rescaling (Rymes & Myers 2001; Lai & Kaplan 2022); "
            "writes to outputs/data/interpolated/mean_preserving/."
        ),
    )
    p.add_argument(
        "--iterations", type=int, default=N_ITERATIONS,
        help=f"MC iterations (default {N_ITERATIONS})",
    )
    return p.parse_args()


def _resolve_output_paths(interpolation: str):
    """Pick output directory and CSV paths based on interpolation method.

    Routing flipped 2026-04-28 (Operation Bulletproof DECISIONS):
      pchip (DEFAULT)  → canonical outputs/data/  — annual cadence
      none             → outputs/data/legacy_3yr_bucket/  — legacy 3-yr bucket
      linear / mean-preserving → outputs/data/interpolated/{slug}/

    The canonical-vs-legacy flip means consumers of outputs/data/material_demand_*.csv
    automatically get annual data once they re-run the pipeline. Pre-flip
    consumers that rely on 3-yr-bucket cadence must update to read from
    outputs/data/legacy_3yr_bucket/ until they migrate.
    """
    if interpolation == "pchip":
        # New canonical default: PCHIP-interpolated annual data writes to
        # the production paths so downstream consumers get annual data
        # without code changes.
        return (
            DATA_OUT_DIR,
            DETAILED_RESULTS_FILE,
            SUMMARY_RESULTS_FILE,
            REPORT_FILE,
        )
    if interpolation == "none":
        # Legacy 3-yr-bucket path (was the default pre-2026-04-28).
        out_dir = DATA_OUT_DIR / "legacy_3yr_bucket"
        out_dir.mkdir(parents=True, exist_ok=True)
        return (
            out_dir,
            out_dir / "material_demand_by_scenario.csv",
            out_dir / "material_demand_summary.csv",
            out_dir / "simulation_report.txt",
        )
    # Other interpolation methods (linear, mean-preserving) write to a
    # method-specific subdir so all sensitivity runs are preserved.
    slug = interpolation.replace("-", "_")
    out_dir = DATA_OUT_DIR / "interpolated" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir,
        out_dir / "material_demand_by_scenario.csv",
        out_dir / "material_demand_summary.csv",
        out_dir / "simulation_report.txt",
    )


def _run_with_interpolation(method: str, n_iterations: int):
    """Full simulation with capacity interpolation before stock-flow.

    Mirrors run_full_simulation() but splices the interpolation step
    between load_all_data() and MaterialsStockFlowSimulation construction.
    """
    from src.data_ingestion import load_all_data
    from src.distribution_fitting import DistributionFitter
    from src.cv_borrowing import apply_cv_borrowing, create_cv_borrowing_report
    from src.technology_mapping import validate_mapping
    from src.interpolation import INTERPOLATION_METHODS

    print("\nStep 1: Loading data...")
    data = load_all_data(
        intensity_path=INTENSITY_FILE,
        national_capacity_path=CAPACITY_FILE,
    )

    print("\nStep 2: Fitting distributions...")
    fitter = DistributionFitter()
    fitted_dists = fitter.fit_all(data["intensity"])

    print("\nStep 2b: Applying CV borrowing for low-sample-size pairs...")
    fitted_dists = apply_cv_borrowing(
        fitted_distributions=fitted_dists, threshold_n=5, reference_n=5,
    )

    # Export fit data to the interpolated-run's own directory.
    out_dir, *_ = _resolve_output_paths(method)
    fitter.export_to_csv(out_dir / "fitted_distributions.csv")
    fitter.export_fit_summary(out_dir / "fit_summary.csv")
    create_cv_borrowing_report(
        fitted_distributions=fitted_dists,
        output_path=str(out_dir / "cv_borrowing_report.csv"),
        threshold_n=5, reference_n=5,
    )

    print("\nStep 3: Validating technology mapping...")
    validate_mapping()

    print(f"\nStep 3b: Applying capacity interpolation ({method})...")
    interp_fn = INTERPOLATION_METHODS[method]
    n_before = len(data["capacity_national"])
    data["capacity_national"] = interp_fn(data["capacity_national"])
    n_after = len(data["capacity_national"])
    print(f"  Capacity rows: {n_before} (3-yr) → {n_after} (1-yr)")

    print("\nStep 4: Initializing simulation...")
    simulation = MaterialsStockFlowSimulation(
        capacity_data=data["capacity_national"],
        intensity_data=data["intensity"],
        fitted_distributions=fitted_dists,
        random_state=RANDOM_SEED,
    )

    print("\nStep 5: Running Monte Carlo simulation...")
    results = simulation.run_monte_carlo(
        n_iterations=n_iterations,
        scenarios_to_run=FOCUS_SCENARIOS,
        convergence_rtol=0.01,
        convergence_check_every=500,
        convergence_min_iterations=1000,
    )

    return simulation, results


def main():
    """Run the complete simulation workflow"""
    args = _parse_args()
    interpolation = args.interpolation
    n_iterations = args.iterations
    out_dir, detailed_path, summary_path, report_path = _resolve_output_paths(
        interpolation
    )

    print("="*80)
    print("MATERIALS DEMAND STOCK-FLOW MONTE CARLO SIMULATION")
    print("="*80)
    print()

    print(f"Configuration:")
    print(f"  Intensity data: {INTENSITY_FILE}")
    print(f"  Capacity data: {CAPACITY_FILE}")
    print(f"  Output directory: {out_dir}")
    print(f"  Interpolation: {interpolation}")
    print(f"  Iterations: {n_iterations:,}")
    print(f"  Random seed: {RANDOM_SEED}")
    if FOCUS_SCENARIOS:
        print(f"  Focus scenarios: {FOCUS_SCENARIOS}")
    print()

    # Verify files exist
    if not INTENSITY_FILE.exists():
        print(f"ERROR: Cannot find {INTENSITY_FILE}")
        sys.exit(1)
    if not CAPACITY_FILE.exists():
        print(f"ERROR: Cannot find {CAPACITY_FILE}")
        sys.exit(1)

    # ========================================================================
    # STEP 1: RUN SIMULATION
    # ========================================================================

    print("\n" + "="*80)
    print("RUNNING SIMULATION")
    print("="*80)
    print()
    
    if interpolation == "none":
        simulation, results = run_full_simulation(
            intensity_path=INTENSITY_FILE,
            capacity_path=CAPACITY_FILE,
            n_iterations=n_iterations,
            scenarios_to_run=FOCUS_SCENARIOS,
            random_state=RANDOM_SEED,
            output_dir=out_dir,
        )
    else:
        simulation, results = _run_with_interpolation(
            method=interpolation, n_iterations=n_iterations,
        )
    
    # ========================================================================
    # STEP 2: EXTRACT STATISTICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXTRACTING STATISTICS")
    print("="*80)
    print()
    
    # Detailed results (by scenario) - NOW INCLUDES 2.5 and 97.5 percentiles
    print("Calculating detailed statistics (by scenario)...")
    detailed_stats = results.get_statistics(percentiles=[2.5, 5, 25, 50, 75, 95, 97.5])
    
    # Debug: Check what columns were actually created
    print(f"Available columns: {detailed_stats.columns.tolist()}")
    
    # Summary results: the four highlighted scenarios, taken per-scenario from
    # the detailed per-scenario statistics. This is a compact view of the
    # headline scenarios (each kept separate, since the NREL scenarios are
    # mutually-exclusive alternative futures); the full 61-scenario detail
    # stays in material_demand_by_scenario.csv.
    print("Selecting the four highlighted scenarios for the summary...")
    _missing = [s for s in HIGHLIGHTED_SCENARIOS
                if s not in detailed_stats["scenario"].unique()]
    if _missing:
        print(f"WARNING: highlighted scenarios missing from results: {_missing}")
    summary_stats = detailed_stats[
        detailed_stats["scenario"].isin(HIGHLIGHTED_SCENARIOS)
    ].reset_index(drop=True).copy()

    print(f"\n✓ Statistics calculated")
    print(f"  Detailed results: {len(detailed_stats):,} rows (all 61 scenarios)")
    print(f"  Summary results: {len(summary_stats):,} rows "
          f"({len(HIGHLIGHTED_SCENARIOS)} highlighted scenarios)")
    print(f"  Percentiles requested: 2.5, 5, 25, 50, 75, 95, 97.5")
    
    # Determine the actual column names for p2.5 and p97.5
    # They might be 'p2.5', 'p2_5', '2.5', etc.
    p2_5_col = None
    p97_5_col = None
    
    for col in detailed_stats.columns:
        col_str = str(col).lower()
        if '2.5' in col_str or '2_5' in col_str:
            p2_5_col = col
        elif '97.5' in col_str or '97_5' in col_str:
            p97_5_col = col
    
    if not p2_5_col or not p97_5_col:
        print(f"\nWARNING: Could not find p2.5 or p97.5 columns.")
        print(f"Available columns: {[c for c in detailed_stats.columns if c.startswith('p') or str(c).replace('.','').replace('_','').isdigit()]}")
        print(f"Will use p5 and p95 instead for confidence intervals.")
        p2_5_col = 'p5'
        p97_5_col = 'p95'
    else:
        print(f"  95% CI columns found: {p2_5_col} to {p97_5_col}")
    
    # ========================================================================
    # STEP 3: PREVIEW RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("PREVIEW: KEY RESULTS")
    print("="*80)
    print()
    
    # Show top materials for 2035 in the Mid_Case (central estimate)
    print("Top 10 materials by median demand (2035, Mid_Case):")
    print("-" * 80)

    summary_2035 = summary_stats[
        (summary_stats['year'] == 2035) &
        (summary_stats['scenario'] == 'Mid_Case')
    ].copy()
    summary_2035_sorted = summary_2035.sort_values('p50', ascending=False).head(10)
    
    for idx, row in summary_2035_sorted.iterrows():
        ci_text = f"(95% CI: [{row[p2_5_col]:>10,.0f}, {row[p97_5_col]:>10,.0f}])"
        print(f"{row['material']:20s}: {row['p50']:>12,.0f} mt {ci_text}")
    
    # Show example scenario comparison for Copper
    print("\n" + "-" * 80)
    print("Copper demand by scenario (2035, median with 95% CI):")
    print("-" * 80)
    
    copper_2035 = detailed_stats[
        (detailed_stats['material'] == 'Copper') &
        (detailed_stats['year'] == 2035)
    ].copy()
    
    if len(copper_2035) > 0:
        copper_sorted = copper_2035.sort_values('p50', ascending=False).head(10)
        for idx, row in copper_sorted.iterrows():
            ci_text = f"(95% CI: [{row[p2_5_col]:>10,.0f}, {row[p97_5_col]:>10,.0f}])"
            print(f"{row['scenario']:30s}: {row['p50']:>12,.0f} mt {ci_text}")
    else:
        print("No copper data available")
    
    # ========================================================================
    # STEP 4: EXPORT RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    print()
    
    # Append annualized columns (mean_annual, p50_annual, ...). Each raw
    # column represents the reporting-period total; `*_annual` divides by the
    # bucket width inferred from the year sequence. Baseline-year rows get 0
    # (no prior reporting year). See docs/METHODS.md and src/annualization.py.
    detailed_stats = add_annualized_columns(detailed_stats, STAT_COLS)
    summary_stats = add_annualized_columns(summary_stats, STAT_COLS)

    # Export detailed results
    detailed_stats.to_csv(detailed_path, index=False)
    print(f"✓ Detailed results saved: {detailed_path}")
    print(f"  95% CI columns: '{p2_5_col}' to '{p97_5_col}'")
    print(f"  Annualized columns added (suffix '_annual'); bucket_years col present")

    # Export summary results
    summary_stats.to_csv(summary_path, index=False)
    print(f"✓ Summary results saved: {summary_path}")
    print(f"  95% CI columns: '{p2_5_col}' to '{p97_5_col}'")
    print(f"  Annualized columns added (suffix '_annual'); bucket_years col present")

    # Create detailed report
    create_report(
        simulation=simulation,
        results=results,
        detailed_stats=detailed_stats,
        summary_stats=summary_stats,
        output_path=report_path,
        p2_5_col=p2_5_col,
        p97_5_col=p97_5_col
    )
    print(f"✓ Detailed report saved: {report_path}")

    # ========================================================================
    # STEP 5: SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print()
    print(f"Interpolation method: {interpolation}")
    print("Key Outputs:")
    print(f"  1. {detailed_path}")
    print(f"     - Material demand by scenario, year, material")
    print(f"     - Includes percentiles: p2.5, p5, p25, p50, p75, p95, p97.5")
    print(f"     - 95% confidence intervals available (p2.5 to p97.5)")
    print(f"     - {len(detailed_stats):,} rows")
    print()
    print(f"  2. {summary_path}")
    print(f"     - Material demand for the four highlighted scenarios "
          f"(per scenario)")
    print(f"     - {', '.join(HIGHLIGHTED_SCENARIOS)}")
    print(f"     - {len(summary_stats):,} rows")
    print()
    print(f"  3. {report_path}")
    print(f"     - Comprehensive text report with analysis")
    print()
    print("Next Steps:")
    print("  - Review the report for quality assessment")
    print("  - Analyze scenario comparisons in detailed results")
    print("  - Visualize uncertainty bands using 95% CI (p2.5-p97.5)")
    print("  - Compare IRA vs No-IRA scenarios")
    print()


def create_report(
    simulation: MaterialsStockFlowSimulation,
    results,
    detailed_stats: pd.DataFrame,
    summary_stats: pd.DataFrame,
    output_path: Path,
    p2_5_col: str,
    p97_5_col: str
):
    """Create comprehensive text report"""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MATERIALS DEMAND STOCK-FLOW MONTE CARLO SIMULATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Simulation parameters
        f.write("SIMULATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Monte Carlo iterations: {results.n_iterations:,}\n")
        f.write(f"Scenarios analyzed: {len(results.scenarios)}\n")
        f.write(f"Years: {min(results.years)} - {max(results.years)}\n")
        f.write(f"Materials tracked: {len(results.materials)}\n")
        f.write(f"Random seed: {simulation.random_state}\n")
        f.write(f"Percentiles reported: 2.5, 5, 25, 50 (median), 75, 95, 97.5\n")
        f.write(f"  95% CI: p2.5 to p97.5\n")
        f.write(f"  90% CI: p5 to p95\n")
        f.write("\n")
        
        # Model description
        f.write("MODEL DESCRIPTION\n")
        f.write("-" * 80 + "\n")
        f.write("Stock-Flow Model:\n")
        f.write("  - Tracks installed capacity stock over time\n")
        f.write("  - Calculates annual capacity additions\n")
        f.write("  - Models retirements based on technology-specific lifetimes\n")
        f.write("  - Material demand = capacity additions × material intensity\n")
        f.write("\n")
        f.write("Uncertainty Quantification:\n")
        f.write("  - Monte Carlo sampling from fitted distributions\n")
        f.write("  - Material intensity uncertainty propagated through model\n")
        f.write("  - Results reported as percentiles (2.5, 5, 25, 50, 75, 95, 97.5)\n")
        f.write("  - 95% confidence intervals: [p2.5, p97.5]\n")
        f.write("  - 90% confidence intervals: [p5, p95]\n")
        f.write("\n")
        
        # Technology mapping summary
        f.write("TECHNOLOGY MAPPING\n")
        f.write("-" * 80 + "\n")
        mapped_count = sum(1 for v in simulation.technology_mapping.values() if v)
        f.write(f"Capacity technologies: {len(simulation.capacity_tech_cols)}\n")
        f.write(f"  Mapped to intensity data: {mapped_count}\n")
        f.write(f"  Unmapped (skipped): {len(simulation.capacity_tech_cols) - mapped_count}\n")
        f.write("\n")
        
        # Key results
        f.write("KEY RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        # Top 10 materials in 2035 (Mid_Case central estimate)
        f.write("Top 10 Materials by Median Demand (2035, Mid_Case)\n")
        f.write("-" * 80 + "\n")
        summary_2035 = summary_stats[
            (summary_stats['year'] == 2035) &
            (summary_stats['scenario'] == 'Mid_Case')
        ].copy()
        summary_2035_sorted = summary_2035.sort_values('p50', ascending=False).head(10)
        
        f.write(f"{'Material':<20} {'Median (mt)':>15} {'95% CI Lower':>15} {'95% CI Upper':>15}\n")
        f.write("-" * 80 + "\n")
        for _, row in summary_2035_sorted.iterrows():
            f.write(f"{row['material']:<20} {row['p50']:>15,.0f} "
                   f"{row[p2_5_col]:>15,.0f} {row[p97_5_col]:>15,.0f}\n")
        f.write("\n")
        
        # Material demand over time (key materials, Mid_Case)
        key_materials = ['Copper', 'Aluminum', 'Steel', 'Lithium', 'Silicon']
        f.write("\nMaterial Demand Over Time (Median, Mid_Case)\n")
        f.write("-" * 80 + "\n")

        for material in key_materials:
            mat_data = summary_stats[
                (summary_stats['material'] == material) &
                (summary_stats['scenario'] == 'Mid_Case')
            ].copy()
            if len(mat_data) == 0:
                continue
            
            f.write(f"\n{material}:\n")
            mat_data_sorted = mat_data.sort_values('year')
            f.write(f"  {'Year':<6} {'Median (mt)':>15} {'95% CI':>30}\n")
            for _, row in mat_data_sorted.iterrows():
                ci = f"[{row[p2_5_col]:,.0f}, {row[p97_5_col]:,.0f}]"
                f.write(f"  {row['year']:<6} {row['p50']:>15,.0f} {ci:>30}\n")
        
        f.write("\n")
        
        # Scenario comparison for key year
        f.write("\n" + "="*80 + "\n")
        f.write("SCENARIO COMPARISON (2035, Copper)\n")
        f.write("="*80 + "\n\n")
        
        copper_2035 = detailed_stats[
            (detailed_stats['material'] == 'Copper') &
            (detailed_stats['year'] == 2035)
        ].copy()
        
        if len(copper_2035) > 0:
            copper_sorted = copper_2035.sort_values('p50', ascending=False)
            f.write(f"{'Scenario':<35} {'Median (mt)':>15} {'Std Dev':>15} {'95% CI':>30}\n")
            f.write("-" * 80 + "\n")
            for _, row in copper_sorted.iterrows():
                ci = f"[{row[p2_5_col]:,.0f}, {row[p97_5_col]:,.0f}]"
                f.write(f"{row['scenario']:<35} {row['p50']:>15,.0f} "
                       f"{row['std']:>15,.0f} {ci:>30}\n")
        else:
            f.write("No copper data available\n")
        
        f.write("\n")
        
        # Data quality notes
        f.write("\n" + "="*80 + "\n")
        f.write("DATA QUALITY & LIMITATIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Material Intensity Data:\n")
        f.write("  - Source: compiled material-intensity literature\n")
        f.write("  - Uncertainty captured via fitted (lognormal) distributions\n")
        f.write("  - Small samples (n<5) use CV borrowing, NOT bootstrap; the\n")
        f.write("    fit is always parametric (see src/cv_borrowing.py)\n")
        f.write("\n")

        f.write("Capacity Projections:\n")
        f.write("  - Source: NREL Standard Scenarios 2024\n")
        f.write("  - Reported at 3-year intervals (2026, 2029, 2032, ...)\n")
        f.write("  - Interpolated to an annual grid; default is shape-preserving\n")
        f.write("    PCHIP (see src/interpolation.py and --interpolation)\n")
        f.write("\n")
        
        f.write("Model Limitations:\n")
        f.write("  - Technology mapping based on expert judgment\n")
        f.write("  - Some technologies lack intensity data (batteries, DAC, etc.)\n")
        f.write("  - No material recovery/recycling modeled\n")
        f.write("  - Assumes constant material intensity over time\n")
        f.write("\n")
        
        # Statistics
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICAL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total data points: {len(detailed_stats):,}\n")
        f.write(f"  Scenarios: {len(results.scenarios)}\n")
        f.write(f"  Years: {len(results.years)}\n")
        f.write(f"  Materials: {len(results.materials)}\n")
        f.write("\n")
        
        f.write(f"Monte Carlo iterations: {results.n_iterations:,}\n")
        f.write(f"Percentiles reported: 2.5, 5, 25, 50 (median), 75, 95, 97.5\n")
        f.write(f"Confidence intervals:\n")
        f.write(f"  - 95% CI: [p2.5, p97.5] (recommended for publication)\n")
        f.write(f"  - 90% CI: [p5, p95] (alternative)\n")
        f.write(f"  - IQR: [p25, p75] (interquartile range)\n")
        f.write("\n")
        
        f.write("End of Report\n")
        f.write("="*80 + "\n")


if __name__ == '__main__':
    main()