"""Interpolates NREL's 3-year capacity stock cadence onto a 1-year grid so the
stock-flow Monte Carlo can report demand at an annual rate.

Provides linear, mean-preserving, and PCHIP interpolants (PCHIP is the
production default); INTERPOLATION_METHODS maps each name to its function. Run
as a script to reproduce the interpolation-method comparison into
outputs/data/experimental/ without touching the canonical outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_ingestion import load_all_data
from src.distribution_fitting import DistributionFitter
from src.cv_borrowing import apply_cv_borrowing
from src.stock_flow_simulation import MaterialsStockFlowSimulation
from src.technology_mapping import validate_mapping
from src.random_config import RANDOM_SEED


DATA_DIR = ROOT / "data"
# Comparison outputs (only written when this module is run as a script via
# main()). The directory is created lazily there, not at import time, so that
# simply importing the interpolation functions has no filesystem side effects.
OUT_DIR = ROOT / "outputs" / "data" / "experimental"

INTENSITY_FILE = DATA_DIR / "intensity_data.csv"
CAPACITY_FILE = DATA_DIR / "StdScen24_annual_national.csv"

# Lower iteration count than production (10k) for faster experimentation;
# convergence checking makes this conservative. Override via CLI.
N_ITERATIONS = 2000
FOCUS_SCENARIOS: list[str] | None = None  # None = all 61 scenarios


def interpolate_capacity_to_annual(cap_df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate capacity stock to 1-year grid per scenario.

    NREL reports stock at {2026, 2029, 2032, ..., 2050}. Between these,
    we assume capacity is built at a constant rate — the simplest
    assumption NREL itself does not make, but standard in the stock-flow
    literature when moving between reporting granularities (e.g.,
    Habib & Wenzel 2014).

    Input columns: scenario, year, *_MW. Output: same columns, with one
    row per (scenario, year) for every integer year in the span.
    """
    capacity_cols = [c for c in cap_df.columns if c.endswith("_MW")]
    if "year" not in cap_df.columns or "scenario" not in cap_df.columns:
        raise ValueError("cap_df must have 'scenario' and 'year' columns")

    y_min = int(cap_df["year"].min())
    y_max = int(cap_df["year"].max())
    target_years = np.arange(y_min, y_max + 1)

    frames = []
    for scen, sub in cap_df.groupby("scenario"):
        sub = sub.sort_values("year")
        reporting_years = sub["year"].to_numpy(dtype=float)
        interp = {"scenario": scen, "year": target_years}
        for col in capacity_cols:
            interp[col] = np.interp(
                target_years, reporting_years, sub[col].to_numpy(dtype=float)
            )
        frames.append(pd.DataFrame(interp))

    return pd.concat(frames, ignore_index=True).sort_values(
        ["scenario", "year"]
    ).reset_index(drop=True)


def interpolate_capacity_mean_preserving(cap_df: pd.DataFrame) -> pd.DataFrame:
    """Mean-preserving spline interpolation of bucket-average addition rates.

    An adaptation of the mean-preserving interpolation family (Rymes &
    Myers 2001, Solar Energy, doi:10.1016/S0038-092X(01)00052-4; Harzallah
    1995; Lai & Kaplan 2022 "Fast mean-preserving spline",
    doi:10.1175/JTECH-D-21-0154.1; Ruiz-Arias 2022, Solar Energy,
    doi:10.1016/j.solener.2022.10.038) to NREL Standard Scenarios'
    3-year-cadence capacity stocks. The shape-preserving piece leans on
    Fritsch & Carlson 1980 (doi:10.1137/0717021). To our knowledge this
    is the first application of mean-preserving interpolation to
    coarse-cadence capacity scenarios in dynamic MFA.

    Instead of assuming constant deployment within each NREL bucket (the
    linear-stock method above — cf. Watari et al. 2019
    doi:10.1016/j.resconrec.2019.05.015 for the linear-default precedent
    in dMFA), treat the **bucket-average addition rate** as the
    fundamental interval-mean quantity, smoothly interpolate it across
    the full span, then integrate to reconstruct stock.

    Algorithm:
      1. r[i] = (stock[i] - stock[i-1]) / bucket_width  — bucket-avg rate
      2. Anchor each r[i] at the bucket center c[i]
      3. Natural cubic spline on (c[i], r[i]) → smooth rate curve
      4. Sample at every annual target year → raw_rates(t)
      5. Per-bucket rescaling: scale raw rates within each bucket so
         sum(rates_in_bucket) == reported Δstock exactly (mass conservation)
      6. Cumulatively sum rescaled rates from the reported 2026 stock

    Properties:
      - Passes through every NREL reporting-year stock value exactly.
      - Addition rate is smooth *within* each bucket (no step plateau).
      - Small discontinuities in rate at bucket boundaries from the
        rescaling step (typically a few percent of local rate magnitude).
      - Declining-stock buckets produce negative rates naturally; the
        stock-flow model's addition<0 clamp (stock_flow_simulation.py:119)
        handles these.
      - Natural cubic can overshoot on sharp inflections; rescaling
        re-anchors each bucket so any overshoot is confined within.
    """
    capacity_cols = [c for c in cap_df.columns if c.endswith("_MW")]
    if "year" not in cap_df.columns or "scenario" not in cap_df.columns:
        raise ValueError("cap_df must have 'scenario' and 'year' columns")

    y_min = int(cap_df["year"].min())
    y_max = int(cap_df["year"].max())
    target_years = np.arange(y_min, y_max + 1)

    frames = []
    for scen, sub in cap_df.groupby("scenario"):
        sub = sub.sort_values("year")
        reporting_years = sub["year"].to_numpy(dtype=float)
        out = {"scenario": scen, "year": target_years}

        centers = (reporting_years[:-1] + reporting_years[1:]) / 2
        widths = np.diff(reporting_years)

        for col in capacity_cols:
            stock_rep = sub[col].to_numpy(dtype=float)
            if stock_rep.max() == 0:
                out[col] = np.zeros_like(target_years, dtype=float)
                continue

            Δstock = np.diff(stock_rep)
            rates = Δstock / widths

            # Endpoint anchors: flat-extrapolate first/last bucket rates.
            # Without these, the cubic spline behaves badly near t_min / t_max.
            anchor_x = np.concatenate(
                [[reporting_years[0]], centers, [reporting_years[-1]]]
            )
            anchor_y = np.concatenate([[rates[0]], rates, [rates[-1]]])

            spline = CubicSpline(anchor_x, anchor_y, bc_type="natural")
            raw_rates = spline(target_years.astype(float))

            # Per-bucket rescaling for exact mass conservation.
            # The rate at year y contributes to Δstock from year y-1 to y,
            # so bucket [t_lo, t_hi] is defined by years y where
            # t_lo < y <= t_hi.
            rescaled = raw_rates.copy()
            for i in range(len(reporting_years) - 1):
                t_lo, t_hi = reporting_years[i], reporting_years[i + 1]
                mask = (target_years > t_lo) & (target_years <= t_hi)
                if not mask.any():
                    continue
                target = stock_rep[i + 1] - stock_rep[i]
                raw_sum = float(rescaled[mask].sum())
                if abs(raw_sum) > 1e-9:
                    rescaled[mask] = rescaled[mask] * (target / raw_sum)
                else:
                    n = int(mask.sum())
                    rescaled[mask] = target / n

            # Integrate: stock[year_0] = NREL value; each subsequent year
            # adds that year's rescaled addition rate.
            stock_annual = np.empty_like(target_years, dtype=float)
            stock_annual[0] = stock_rep[0]
            for j in range(1, len(target_years)):
                stock_annual[j] = stock_annual[j - 1] + rescaled[j]

            out[col] = stock_annual

        frames.append(pd.DataFrame(out))

    return pd.concat(frames, ignore_index=True).sort_values(
        ["scenario", "year"]
    ).reset_index(drop=True)


def interpolate_capacity_pchip(cap_df: pd.DataFrame) -> pd.DataFrame:
    """PCHIP (monotone cubic Hermite) interpolation of capacity stock.

    Fritsch & Carlson 1980 (doi:10.1137/0717021). Shape-preserving cubic
    Hermite interpolant that does NOT overshoot, specifically addressing
    the 2045 overshoot artifact observed in the natural-cubic
    mean-preserving method for thin-film-PV materials (Tellurium,
    Cadmium, Silver, Silicon, Tin, Glass, Indium under Mid_Case).

    Algorithm:
      - Apply PchipInterpolator to NREL's reporting-year stock values.
      - Sample at every integer year 2026…2050.
      - Hand the resulting annual-stock DataFrame to the stock-flow MC
        runner unchanged — Δstock at each annual step becomes the
        addition rate for that year.

    Properties:
      - Passes through every NREL reporting-year stock value exactly
        (mass conservation automatic — no per-bucket rescaling needed).
      - Monotonicity-preserving: if three consecutive reporting stocks
        are monotone, the interpolant is monotone between them.
      - C¹-smooth across bucket boundaries (continuous addition rate).
      - Does not oscillate inside a bucket — overshoot cannot occur.
      - Handles declining stocks (coal) and zero-valued starts (emerging
        tech) gracefully.
      - Slight tradeoff: if real deployment is truly cubic within a
        bucket, PCHIP's monotone derivative rule can under-emphasize
        the peak rate slightly. See Fritsch-Carlson for discussion.

    PCHIP was added as the shape-preserving remedy after an overshoot
    artifact surfaced in the natural-cubic mean-preserving method.
    """
    capacity_cols = [c for c in cap_df.columns if c.endswith("_MW")]
    if "year" not in cap_df.columns or "scenario" not in cap_df.columns:
        raise ValueError("cap_df must have 'scenario' and 'year' columns")

    y_min = int(cap_df["year"].min())
    y_max = int(cap_df["year"].max())
    target_years = np.arange(y_min, y_max + 1)
    target_years_f = target_years.astype(float)

    frames = []
    for scen, sub in cap_df.groupby("scenario"):
        sub = sub.sort_values("year")
        reporting_years = sub["year"].to_numpy(dtype=float)
        out = {"scenario": scen, "year": target_years}
        for col in capacity_cols:
            stock = sub[col].to_numpy(dtype=float)
            if len(reporting_years) < 2 or stock.max() == 0:
                out[col] = np.zeros_like(target_years, dtype=float)
                continue
            # extrapolate=False guards against values outside the NREL
            # reporting range; all target years are inside [y_min, y_max]
            # by construction so this is belt-and-suspenders.
            pchip = PchipInterpolator(
                reporting_years, stock, extrapolate=False,
            )
            out[col] = pchip(target_years_f)
        frames.append(pd.DataFrame(out))

    return pd.concat(frames, ignore_index=True).sort_values(
        ["scenario", "year"]
    ).reset_index(drop=True)


INTERPOLATION_METHODS = {
    "linear": interpolate_capacity_to_annual,
    "mean-preserving": interpolate_capacity_mean_preserving,
    "pchip": interpolate_capacity_pchip,
}


def main(n_iterations: int = N_ITERATIONS,
         scenarios: list[str] | None = FOCUS_SCENARIOS,
         method: str = "linear") -> None:
    """Re-run the stock-flow MC on the chosen annual-grid interpolation.

    Convenience entry point for reproducing the interpolation-method
    comparison. Loads the same inputs as the main simulation, applies the
    requested interpolation (`method` keys `INTERPOLATION_METHODS`) to the
    capacity stock, runs the Monte Carlo, and writes a parallel result set
    to outputs/data/experimental/ (suffixed by method). It never touches
    the canonical outputs and is not part of the figure pipeline.
    """
    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown method: {method!r}. "
            f"Choose from {list(INTERPOLATION_METHODS)}"
        )

    print("=" * 80)
    print("EXPERIMENTAL: annual-grid interpolation stock-flow MC")
    print("=" * 80)
    print(f"  Method: {method}")
    print(f"  Iterations: {n_iterations:,}")
    print(f"  Scenarios: {'all 61' if scenarios is None else scenarios}")

    data = load_all_data(
        intensity_path=INTENSITY_FILE,
        national_capacity_path=CAPACITY_FILE,
    )

    fitter = DistributionFitter()
    fitted = fitter.fit_all(data["intensity"])
    fitted = apply_cv_borrowing(
        fitted_distributions=fitted, threshold_n=5, reference_n=5
    )

    validate_mapping()

    interp_fn = INTERPOLATION_METHODS[method]
    cap_annual = interp_fn(data["capacity_national"])
    print(f"  Capacity rows: {len(data['capacity_national'])} (3-yr) "
          f"→ {len(cap_annual)} (1-yr)")
    print(f"  Years covered: {cap_annual['year'].min()}"
          f"–{cap_annual['year'].max()} "
          f"({cap_annual['year'].nunique()} annual points)")

    simulation = MaterialsStockFlowSimulation(
        capacity_data=cap_annual,
        intensity_data=data["intensity"],
        fitted_distributions=fitted,
        random_state=RANDOM_SEED,
    )

    results = simulation.run_monte_carlo(
        n_iterations=n_iterations,
        scenarios_to_run=scenarios,
        convergence_rtol=0.01,
        convergence_check_every=500,
        convergence_min_iterations=1000,
    )

    detailed = results.get_statistics(
        percentiles=[2.5, 5, 25, 50, 75, 95, 97.5]
    )
    summary = results.get_summary_by_material(
        percentiles=[2.5, 5, 25, 50, 75, 95, 97.5]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if method == "linear" else f"_{method.replace('-', '_')}"
    detailed_path = OUT_DIR / f"material_demand_by_scenario_annual_interp{suffix}.csv"
    summary_path = OUT_DIR / f"material_demand_summary_annual_interp{suffix}.csv"
    capacity_path = OUT_DIR / f"capacity_annual_interpolated{suffix}.csv"

    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    cap_annual.to_csv(capacity_path, index=False)

    print()
    print(f"  Wrote {detailed_path.relative_to(ROOT)}")
    print(f"  Wrote {summary_path.relative_to(ROOT)}")
    print(f"  Wrote {capacity_path.relative_to(ROOT)}")
    print()
    print("Compare to outputs/data/material_demand_by_scenario.csv "
          "mean_annual column:")
    print("  annual-grid values should approximate the 3-yr-bucket ÷ 3 "
          "values except where non-linear deployment or retirement "
          "timing diverges.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS,
                        help=f"MC iterations (default {N_ITERATIONS})")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Subset of scenarios (default: all 61)")
    parser.add_argument(
        "--method",
        choices=list(INTERPOLATION_METHODS),
        default="linear",
        help="Capacity-to-annual interpolation: 'linear' (piecewise "
             "constant additions within bucket, the default, matches "
             "Watari et al. 2019) or 'mean-preserving' (cubic-spline of "
             "bucket-average rates + per-bucket rescaling for exact "
             "mass conservation, after Rymes & Myers 2001 / Lai & "
             "Kaplan 2022 / Ruiz-Arias 2022).",
    )
    args = parser.parse_args()
    main(n_iterations=args.iterations, scenarios=args.scenarios,
         method=args.method)
