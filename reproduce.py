#!/usr/bin/env python3
"""One-command reproduction of every reported result: the six main-text
figures (1-6), the three supplementary figures (S1-S3), and the supplementary
tables (S1-S9). Runs the Monte Carlo simulation (or uses the shipped
results), builds the supply-chain feature table, then renders the figures and
tables; each step runs as its own subprocess from the repo root.

Figure numbering follows the manuscript (Environmental Research: Energy,
7-23 revision):
  Fig 1  annual capacity additions by technology (15-panel facet)
  Fig 2  annual material demand projections
  Fig 3  cumulative demand by material class
  Fig 4  US supply tiers          Fig 5  global supply tiers
  Fig 6  supply-risk scatter (net import reliance vs HHI)
  Fig S1 demand across electricity-supply technologies
  Fig S2 per-material sensitivity (scenario range vs intensity CI)
  Fig S3 demand-uncertainty tornado (by material class)

Two diagnostics figures (MC convergence, interpolation robustness) render to
outputs/figures/diagnostics/; they back Methods claims but are not in the SI.
Print-size Word variants are produced by each figure script's --docx flag and
live in outputs/figures/manuscript/docx/.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMAND_CSV = ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
MP_CSV = (ROOT / "outputs" / "data" / "interpolated" / "mean_preserving"
          / "material_demand_by_scenario.csv")
FIGURES_DIR = ROOT / "outputs" / "figures" / "manuscript"
TABLES_DIR = ROOT / "outputs" / "data" / "tables"

# Figure / table steps. Each entry: (human label, [script-relative-path, *args]).
# Figures are written full-size at 300 DPI into outputs/figures/manuscript/;
# tables are written as CSVs into outputs/data/ and outputs/data/tables/.
BUILD_STEPS = [
    ("Supply-chain feature table (for Figure 6 and Tables S3/S9)",
     ["supply_chain/build_material_features.py"]),

    ("Figure 1: annual capacity additions by technology (facet)",
     ["visualizations/fig1_capacity_additions_facet.py",
      "--output-dir", "outputs/figures/manuscript"]),

    ("Figure 2: annual material demand projections",
     ["visualizations/fig2_demand_projections.py",
      "--envelope", "cross-scenario",
      "--output-dir", "outputs/figures/manuscript"]),

    ("Figure 3: cumulative demand by material class",
     ["visualizations/fig3_cumulative_by_family.py"]),

    ("Figures 4 & 5: US and global supply tiers",
     ["visualizations/fig4_fig5_supply_tiers.py",
      "--re-mode", "per-element",
      "--output-dir", "outputs/figures/manuscript"]),

    ("Figure 6: supply-risk scatter (NIR vs HHI)",
     ["visualizations/fig6_nir_vs_hhi_scatter.py", "--scale", "linear"]),

    ("Figure S1: material demand across electricity-supply technologies",
     ["visualizations/figS1_demand_by_technology.py",
      "--output-dir", "outputs/figures/manuscript"]),

    ("Figure S2 + diagnostics (convergence, interpolation robustness)",
     ["visualizations/si_figures.py",
      "--output-dir", "outputs/figures/manuscript"]),

    ("Figure S3: demand-uncertainty sensitivity tornado",
     ["visualizations/figS3_sensitivity_tornado.py",
      "--compact", "--scenario-bounds", "min-max", "--baseline", "mid_case",
      "--output", "outputs/figures/manuscript/figS3_sensitivity_tornado.png"]),

    ("Peak-demand table (backs Tables S7/S8)",
     ["visualizations/table1_peak_demand.py"]),

    ("Tables S1-S9: supplementary tables (CSV, docx numbering)",
     ["visualizations/gen_si_tables.py"]),
]


def run(argv, label):
    """Run one script as a subprocess; abort the whole run on failure."""
    print(f"\n{'-' * 70}\n  {label}\n  $ python {' '.join(argv)}\n{'-' * 70}")
    t0 = time.time()
    env = {**os.environ, "MPLBACKEND": "Agg"}  # headless rendering
    result = subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"\n  x FAILED (exit {result.returncode}) after {dt:.1f}s")
        sys.exit(result.returncode)
    print(f"  done ({dt:.1f}s)")


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce the manuscript figures and tables.")
    ap.add_argument("--simulate", action="store_true",
                    help="Re-run the Monte Carlo simulation before building "
                         "figures (overwrites the shipped demand CSVs; "
                         "deterministic at seed 24601, so it reproduces the same "
                         "values). Also regenerates the mean-preserving run "
                         "needed for the interpolation-robustness diagnostic.")
    ap.add_argument("--iterations", type=int, default=None,
                    help="Monte Carlo iterations when --simulate is set "
                         "(default: 10000, the published value).")
    args = ap.parse_args()

    print("=" * 70)
    print("  MATERIALS DEMAND MODEL: reproduce manuscript results")
    print("=" * 70)

    # Step 1: simulation (optional / on demand).
    # Canonical run = PCHIP annual (writes outputs/data/material_demand_*.csv).
    # Mean-preserving run = the robustness comparison behind the diagnostic.
    if args.simulate or not DEMAND_CSV.exists():
        sim_argv = ["simulate.py"]
        if args.iterations is not None:
            sim_argv += ["--iterations", str(args.iterations)]
        run(sim_argv, "Monte Carlo simulation (canonical, PCHIP)")

        mp_argv = ["simulate.py", "--interpolation", "mean-preserving"]
        if args.iterations is not None:
            mp_argv += ["--iterations", str(args.iterations)]
        run(mp_argv, "Monte Carlo simulation (mean-preserving, for diagnostics)")
    else:
        print(f"\n  Using shipped Monte Carlo results: "
              f"{DEMAND_CSV.relative_to(ROOT)}")
        print("  (pass --simulate to regenerate them from scratch)")
        if not MP_CSV.exists():
            print("  NOTE: mean-preserving run absent; the interpolation "
                  "diagnostic will fall back to its committed cache.")

    # Steps 2-N: features, figures, tables.
    for label, argv in BUILD_STEPS:
        run(argv, label)

    print(f"\n{'=' * 70}\n  DONE")
    print(f"  Figures (1-6, S1-S3): {FIGURES_DIR.relative_to(ROOT)}/")
    print(f"  Diagnostics: outputs/figures/diagnostics/")
    print(f"  Tables S1-S9: {TABLES_DIR.relative_to(ROOT)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
