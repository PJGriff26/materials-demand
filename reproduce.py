#!/usr/bin/env python3
"""One-command reproduction of every main-text result: the seven manuscript
figures and Table 1. Runs the Monte Carlo simulation (or uses the shipped
results), builds the supply-chain feature table, then renders the figures and
table; each step runs as its own subprocess from the repository root."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMAND_CSV = ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
FIGURES_DIR = ROOT / "outputs" / "figures" / "manuscript"

# Figure / table steps. Each entry: (human label, [script-relative-path, *args]).
# The figures are written full-size at 300 DPI into outputs/figures/manuscript/.
# (Every generator also accepts --docx to render a print-sized variant for
# direct insertion into the Word manuscript; the chart content is identical.)
BUILD_STEPS = [
    ("Supply-chain feature table (for Figure 7)",
     ["supply_chain/build_material_features.py"]),

    ("Figure 1: capacity additions & retirements",
     ["visualizations/fig1_capacity_additions_ensemble.py",
      "--disaggregate",
      "--output-dir", "outputs/figures/manuscript"]),

    ("Figure 2: annual material demand projections",
     ["visualizations/fig1_demand_projections.py",
      "--envelope", "cross-scenario",
      "--output-dir", "outputs/figures/manuscript"]),

    ("Figure 3: cumulative demand by material family",
     ["visualizations/scenario_cum_by_family_tonnage.py"]),

    ("Figure 4: demand-uncertainty sensitivity tornado",
     ["visualizations/fig4_sensitivity_tornado.py",
      "--compact", "--scenario-bounds", "min-max", "--baseline", "mid_case",
      "--output", "outputs/figures/manuscript/fig4_sensitivity_tornado.png"]),

    ("Figures 5 & 6: US and global supply tiers",
     ["visualizations/fig5_supply_chain_4panel.py",
      "--re-mode", "per-element",
      "--output-dir", "outputs/figures/manuscript"]),

    ("Figure 7: supply-risk scatter (NIR vs HHI)",
     ["visualizations/nir_vs_hhi_scatter.py", "--scale", "linear"]),

    ("Table 1: peak annual demand by scenario",
     ["visualizations/table1_peak_demand.py"]),
]


def run(argv, label):
    """Run one script as a subprocess; abort the whole run on failure."""
    rel = argv[0]
    print(f"\n{'─' * 70}\n  {label}\n  $ python {' '.join(argv)}\n{'─' * 70}")
    t0 = time.time()
    # MPLBACKEND=Agg => headless rendering (no display needed).
    import os
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit {result.returncode}) after {dt:.1f}s")
        sys.exit(result.returncode)
    print(f"  ✓ done ({dt:.1f}s)")


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce the manuscript figures and Table 1.")
    ap.add_argument("--simulate", action="store_true",
                    help="Re-run the Monte Carlo simulation before building "
                         "figures (overwrites the shipped demand CSV; "
                         "deterministic, so it reproduces the same values).")
    ap.add_argument("--iterations", type=int, default=None,
                    help="Monte Carlo iterations when --simulate is set "
                         "(default: 10000, the published value).")
    args = ap.parse_args()

    print("=" * 70)
    print("  MATERIALS DEMAND MODEL: reproduce manuscript results")
    print("=" * 70)

    # Step 1: simulation (optional / on demand).
    if args.simulate or not DEMAND_CSV.exists():
        sim_argv = ["simulate.py"]
        if args.iterations is not None:
            sim_argv += ["--iterations", str(args.iterations)]
        run(sim_argv, "Monte Carlo simulation")
    else:
        print(f"\n  Using shipped Monte Carlo results: "
              f"{DEMAND_CSV.relative_to(ROOT)}")
        print("  (pass --simulate to regenerate them from scratch)")

    # Steps 2-4: features, figures, table.
    for label, argv in BUILD_STEPS:
        run(argv, label)

    print(f"\n{'=' * 70}\n  DONE")
    print(f"  Figures: {FIGURES_DIR.relative_to(ROOT)}/")
    print(f"  Table 1: outputs/data/table1_peak_demand.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
