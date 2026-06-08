"""Manuscript Figure 2: annual material demand projections.

Per-material panels of annual demand across 2026-2050 for four
representative NREL scenarios over a shaded uncertainty envelope (selected
via --envelope: cross-scenario min/max, or full-uncertainty pooled 95% CI).
Reads outputs/data/material_demand_by_scenario.csv. The file/function names
keep a legacy "fig1_*" prefix; this is Figure 2.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE_DIR = Path(__file__).resolve().parent.parent
# DEMAND_CSV / DEFAULT_OUT_DIR are set by main() after parsing --interpolation
# so the script can target either the production CSV or an interpolated
# variant at outputs/data/interpolated/{method}/. Defaults match the
# pre-April-2026 behavior exactly.
from _interpolation_io import (
    demand_csv_path, figures_output_dir, add_interpolation_arg,
)
DEMAND_CSV = demand_csv_path("none")
DEFAULT_OUT_DIR = figures_output_dir("none", subdir="manuscript")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Four representative scenarios (NREL Standard Scenarios 2024)
SCENARIOS = {
    "Mid_Case_100by2035":  "Net-Zero by 2035",
    "Mid_Case_95by2050":   "Net-Zero by 2050",
    "Mid_Case":            "Mid Case (with IRA)",
    "Mid_Case_No_IRA":     "Mid Case (no IRA)",
}

# Colorblind-friendly palette
SCENARIO_COLORS = {
    "Mid_Case_100by2035":  "#d62728",  # red (most aggressive)
    "Mid_Case_95by2050":   "#9467bd",  # purple (NZ-2050)
    "Mid_Case":            "#1f77b4",  # blue (baseline)
    "Mid_Case_No_IRA":     "#ff7f0e",  # orange (rollback)
}

# Materials plotted, ordered roughly by demand magnitude: bulk commodities,
# then intermediate-volume metals, then tech-specific specialty metals and
# rare-earth elements. Specialty entries include Silver (PV), Vanadium (offshore
# wind redox-flow), Indium (thin-film PV), and Manganese (wind steel alloys,
# battery cathodes).
MATERIALS = [
    # All 27 demand-bearing materials (31 tracked minus the 4 zeroed under the PV mix).
    "Steel", "Cement", "Aluminum", "Glass", "Fiberglass",
    "Copper", "Lead", "Zinc", "Silicon", "Manganese",
    "Tin", "Nickel", "Vanadium", "Chromium", "Molybdenum",
    "Magnesium", "Niobium", "Silver", "Boron",
    "Tellurium", "Cadmium", "Indium",
    "Neodymium", "Dysprosium", "Praseodymium", "Terbium", "Yttrium",
]

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def kt_formatter(x, pos):
    """Format y-axis values: 1500 -> '2k', 1500000 -> '1.5M'."""
    if x >= 1e6: return f"{x/1e6:.1f}M"
    if x >= 1e3: return f"{x/1e3:.0f}k"
    return f"{x:.0f}"


def compute_envelope(mat_data, mode):
    """Compute lower/upper envelope per year for a single material.

    Parameters
    ----------
    mat_data : DataFrame
        Subset of demand data for one material, all 61 scenarios.
    mode : str
        'cross-scenario' or 'full-uncertainty'.

    Returns
    -------
    (years, lo, hi, label) : tuple
    """
    # Y-axis is annual demand \u2192 use `_annual` columns so magnitudes are
    # consistent across PCHIP (annual) and legacy 3-yr-bucket runs.
    mean_col = "mean_annual" if "mean_annual" in mat_data.columns else "mean"
    p2_col   = "p2_annual"   if "p2_annual"   in mat_data.columns else "p2"
    p97_col  = "p97_annual"  if "p97_annual"  in mat_data.columns else "p97"
    by_year = mat_data.groupby("year")
    if mode == "cross-scenario":
        # Outer envelope of scenario means
        lo = by_year[mean_col].min()
        hi = by_year[mean_col].max()
        label = "min/max across 61 scenario means"
    elif mode == "full-uncertainty":
        # Pooled outer 95% CI across scenarios x MC draws
        lo = by_year[p2_col].min()
        hi = by_year[p97_col].max()
        label = "outer 95% CI (pooled scenarios \u00d7 MC)"
    else:
        raise ValueError(f"Unknown envelope mode: {mode}")
    return lo.index.values, lo.values, hi.values, label


def plot_fig1(demand, mode, output_path, docx=False):
    """Render manuscript Figure 2 for one envelope mode and save it to output_path.

    Lays out one panel per material in a fixed 4-column grid (rows derived from
    the material count), drawing the four representative scenario lines over the
    shaded uncertainty band produced by compute_envelope(). `mode` selects the
    band ('cross-scenario' or 'full-uncertainty'); `docx` scales text up for the
    Word manuscript without changing the default output.
    """
    # Fixed 4-column grid; row count is ceil(len(MATERIALS) / n_cols). Shared
    # axis labels suppress per-panel "Demand" y-title and "Year" x-title clutter.
    n_cols = 4
    n_rows = (len(MATERIALS) + n_cols - 1) // n_cols
    _orig_w = 18
    _orig_h = 3.2 * n_rows + 1.5
    # Keep the original large canvas for BOTH default and --docx. Shrinking the
    # figsize to print width spaced the 27 panels out unattractively; instead
    # the --docx path (below) keeps this canvas and only scales the text up a
    # little. The figure is inserted full-page (~6.5 in wide) in Word.
    _figsize = (_orig_w, _orig_h)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_figsize,
                              sharex=True)
    axes = axes.flatten()

    for idx, (ax, mat) in enumerate(zip(axes, MATERIALS)):
        mat_data = demand[demand["material"] == mat]

        # Envelope
        years, lo, hi, env_label = compute_envelope(mat_data, mode)
        ax.fill_between(years, lo, hi, color="gray", alpha=0.20,
                        linewidth=0, label=env_label, zorder=1)

        # Four representative scenario lines with MC intensity-uncertainty bands.
        # Band = p2.5/p97.5 (95% CI) from 10,000 MC draws within each scenario,
        # so the viewer can separate scenario choice (line) from intensity
        # uncertainty (band); MC intensity uncertainty must appear somewhere in
        # the visualization set.
        # Same `_annual`-aware column resolution as the envelope above.
        mean_col = "mean_annual" if "mean_annual" in mat_data.columns else "mean"
        p2_col   = "p2_annual"   if "p2_annual"   in mat_data.columns else "p2"
        p97_col  = "p97_annual"  if "p97_annual"  in mat_data.columns else "p97"
        for sc, label in SCENARIOS.items():
            sc_data = mat_data[mat_data["scenario"] == sc].sort_values("year")
            yrs = sc_data["year"].values
            mean = sc_data[mean_col].values
            p2  = sc_data[p2_col].values  if p2_col  in sc_data.columns else mean
            p97 = sc_data[p97_col].values if p97_col in sc_data.columns else mean
            color = SCENARIO_COLORS[sc]
            ax.plot(yrs, mean, color=color, lw=1.8, label=label, zorder=3)

        ax.set_title(mat, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(2026, 2050)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(kt_formatter))
        # Per-panel axis titles suppressed; shared figure-level labels below
        # are sufficient.

    # Hide any unused subplot axes
    for extra_ax in axes[len(MATERIALS):]:
        extra_ax.set_visible(False)

    # Shared figure-level axis labels (replaces per-panel titles)
    fig.supxlabel("", fontsize=1)
    fig.supylabel("Annual material demand (tonnes)", fontsize=11,
                  fontweight="bold", x=0.005)

    # Legend at bottom (avoids overlap with suptitle)
    handles, labels = axes[0].get_legend_handles_labels()
    env_idx = [i for i, l in enumerate(labels) if "CI" in l or "min/max" in l]
    sc_idx = [i for i in range(len(labels)) if i not in env_idx]
    ordered_h = [handles[i] for i in sc_idx] + [handles[i] for i in env_idx]
    ordered_l = [labels[i] for i in sc_idx] + [labels[i] for i in env_idx]
    # LAYOUT FIX (docx): the single-row 5-column legend is too wide to be
    # legible at the 6.5 in print width. Reflow to 3 columns (2 rows) so the
    # five labels fit at 8.5 pt without clipping. Default (no --docx) keeps
    # the original single-row 5-column arrangement byte-for-byte.
    _legend_ncol = 3 if docx else 5
    fig.legend(ordered_h, ordered_l,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=_legend_ncol, fontsize=10, framealpha=0.95, frameon=True)

    fig.suptitle("Annual material demand under four NREL scenarios, 2026\u20132050",
                 fontsize=12, fontweight="bold", y=0.995)

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    # --docx: keep the original large canvas (above) and scale ALL text up by a
    # modest factor so it reads better when the figure is inserted full-page in
    # Word, WITHOUT respacing the panels. Typography only; no data, series,
    # colors, or label content touched. Default (no --docx) output is
    # byte-for-byte unchanged. The bottom legend is already reflowed to 3
    # columns (above) so it still fits the width at the larger size.
    if docx:
        F = 1.5   # slight text scale-up on the original-size canvas

        def _bump(t):
            if t is not None and t.get_fontsize():
                t.set_fontsize(t.get_fontsize() * F)

        _bump(getattr(fig, "_suptitle", None))
        _bump(getattr(fig, "_supylabel", None))
        for _ax in fig.get_axes():
            _bump(_ax.title)
            _bump(_ax.xaxis.label)
            _bump(_ax.yaxis.label)
            for _t in _ax.get_xticklabels() + _ax.get_yticklabels():
                _bump(_t)
            for _t in _ax.texts:
                _bump(_t)
            _lg = _ax.get_legend()
            if _lg:
                for _x in _lg.get_texts():
                    _bump(_x)
        for _lg in fig.legends:               # bottom figure-level legend
            for _x in _lg.get_texts():
                _bump(_x)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envelope", choices=["cross-scenario", "full-uncertainty", "both"],
                         default="both", help="Envelope mode (default: both)")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Output directory (default: auto-resolved from "
                              "--interpolation)")
    parser.add_argument("--docx", action="store_true", default=False,
                         help="Render at 6.5 in print width with absolute "
                              "per-role font sizes for insertion into the Word "
                              "manuscript at the saved width. Default output "
                              "(no flag) is byte-for-byte unchanged.")
    add_interpolation_arg(parser)
    args = parser.parse_args()

    # Resolve interpolation-aware paths. When --interpolation is 'none' the
    # defaults are identical to the pre-April-2026 script behavior.
    global DEMAND_CSV
    DEMAND_CSV = demand_csv_path(args.interpolation)
    if args.output_dir is None:
        args.output_dir = figures_output_dir(args.interpolation, subdir="manuscript")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not DEMAND_CSV.exists():
        print(f"ERROR: demand CSV not found at {DEMAND_CSV}", file=sys.stderr)
        if args.interpolation != "none":
            print(f"  Run `python simulate.py --interpolation "
                  f"{args.interpolation}` first (or `python reproduce.py`).",
                  file=sys.stderr)
        sys.exit(1)

    print(f"Loading demand data from {DEMAND_CSV} (interpolation={args.interpolation})...")
    demand = pd.read_csv(DEMAND_CSV)
    print(f"  {demand['scenario'].nunique()} scenarios, {demand['material'].nunique()} materials, "
          f"years {demand['year'].min()}-{demand['year'].max()}")

    # Validate scenario list
    missing = [s for s in SCENARIOS if s not in demand["scenario"].unique()]
    if missing:
        print(f"ERROR: scenarios missing from data: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.envelope in ("cross-scenario", "both"):
        plot_fig1(demand, "cross-scenario", args.output_dir / "fig2_demand_projections.png",
                  docx=args.docx)
    if args.envelope in ("full-uncertainty", "both"):
        plot_fig1(demand, "full-uncertainty", args.output_dir / "fig2_demand_projections_full_uncertainty.png",
                  docx=args.docx)

    print("Done.")


if __name__ == "__main__":
    main()
