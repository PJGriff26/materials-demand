"""Manuscript Figure 2: annual material demand projections.

Per-material panels of annual demand across 2026-2050 for four
representative NREL scenarios over a shaded uncertainty envelope (selected
via --envelope: cross-scenario min/max, or full-uncertainty pooled 95% CI).
Reads outputs/data/material_demand_by_scenario.csv. The file/function names
keep a legacy "fig1_*" prefix; this is Figure 2.

INVENTORY:
  name: fig2_demand_projections
  output: outputs/figures/manuscript/fig2_demand_projections{,_full_uncertainty}.png
  category: Manuscript main text — Fig 2
  axes:
    x: year (2026-2050)
    y: annual material demand (tonnes; one panel per material, 27 panels)
    color: 4 representative NREL scenarios (rest shown as envelope band)
  data_sources:
    - outputs/data/material_demand_by_scenario.csv (or interpolated variant
      via --interpolation), `_annual` columns.
  description: >
    Small-multiples line chart (5-column grid, alphabetical) of annual
    demand for all 27 demand-bearing materials under four highlighted NREL
    scenarios over a shaded uncertainty envelope (--envelope selects
    cross-scenario min/max or pooled 95% CI). Legend hosted in an empty
    bottom-row grid cell; --docx scales all text x1.5 on the same canvas
    for the Word manuscript. 2026-08-07 (PI request): y-axis text enlarged
    (supylabel 13->16, tick labels 11->13) and legend 13->16; both carry
    through the --docx x1.5 scaling (24 / 19.5 / 24 pt).
END_INVENTORY
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.scenario_config import SCENARIO_LABELS, SCENARIO_COLORS  # noqa: E402
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
# Display order (legend top-to-bottom): most aggressive first. Keys, labels,
# and the cross-figure palette come from src/scenario_config.py (single
# source of truth for the representative-scenario set).
_DISPLAY_ORDER = ["Mid_Case_CO2e_100by2035", "Mid_Case_CO2e_95by2050",
                  "Mid_Case", "Mid_Case_No_IRA"]
SCENARIOS = {k: SCENARIO_LABELS[k] for k in _DISPLAY_ORDER}

# Materials plotted, in alphabetical facet order (2026-07-13 figure
# revision; was demand-magnitude order). sorted() keeps the ordering
# self-enforcing however the underlying list is edited.
MATERIALS = sorted([
    # All 27 demand-bearing materials (31 tracked minus the 4 zeroed under the PV mix).
    "Steel", "Cement", "Aluminum", "Glass", "Fiberglass",
    "Copper", "Lead", "Zinc", "Silicon", "Manganese",
    "Tin", "Nickel", "Vanadium", "Chromium", "Molybdenum",
    "Magnesium", "Niobium", "Silver", "Boron",
    "Tellurium", "Cadmium", "Indium",
    "Neodymium", "Dysprosium", "Praseodymium", "Terbium", "Yttrium",
])

# Rare-earth facet titles are rendered in the same bold navy used for the REE
# material labels in Figs 4/5 (supply_tiers_variants.REE_HIGHLIGHT_COLOR), so the
# REE set reads consistently across figures (2026-07-13 figure revision).
REE_SET = {"Neodymium", "Dysprosium", "Praseodymium", "Terbium", "Yttrium"}
REE_TITLE_COLOR = "#0d3b66"

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def kt_formatter(x, pos):
    """Format y-axis tick labels with at most two significant digits.

    e.g. 1_500_000 -> '1.5M', 12_000_000 -> '12M', 350_000 -> '350k'. One
    decimal is kept only for single-digit mantissas, so labels never run past
    two digits (review item C15).
    """
    if x == 0:
        return "0"
    for thresh, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= thresh:
            v = x / thresh
            return f"{v:.1f}{suffix}" if abs(v) < 10 else f"{v:.0f}{suffix}"
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
        label = "Scenario ensemble range"
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

    Lays out one panel per material in a fixed 5-column grid (rows derived from
    the material count), drawing the four representative scenario lines over the
    shaded uncertainty band produced by compute_envelope(). `mode` selects the
    band ('cross-scenario' or 'full-uncertainty'); `docx` scales text up for the
    Word manuscript without changing the default output.
    """
    # Fixed 5-column grid (2026-07-13 figure revision; was 4); row count is
    # ceil(len(MATERIALS) / n_cols). Shared axis labels suppress per-panel
    # "Demand" y-title and "Year" x-title clutter.
    n_cols = 5
    n_rows = (len(MATERIALS) + n_cols - 1) // n_cols
    _orig_w = 19
    _orig_h = 3.0 * n_rows + 1.5
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

        # REE facet titles carry the Figs 4/5 REE label formatting (bold +
        # navy) while the other titles stay regular weight, so the REE set
        # stands out the same way it does there (2026-07-13 figure revision).
        is_ree = mat in REE_SET
        ax.set_title(mat, fontsize=13,
                     fontweight="bold" if is_ree else "normal",
                     color=REE_TITLE_COLOR if is_ree else "black")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(2026, 2050)
        # 10-year tick interval (2026-07-13 figure revision), matching the
        # Fig 1 capacity-additions facet.
        ax.set_xticks([2030, 2040, 2050])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(kt_formatter))
        # Review item C15: at most six y-axis increments with two-digit labels,
        # larger tick text, and no y-axis tick marks.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        # PI request 2026-08-07: larger y-axis text (x kept consistent).
        ax.tick_params(axis="both", labelsize=13)
        ax.tick_params(axis="y", length=0)
        # Per-panel axis titles suppressed; shared figure-level labels below
        # are sufficient.

    # Review item C15: host the legend in an empty bottom-row grid cell rather
    # than a bottom-center figure legend. With 27 materials on a 5-column grid
    # the bottom row has three empty cells; the middle one hosts the legend
    # (visually centered in the empty span) and the rest are hidden. The
    # legend cell is its own axes, so it can never overlap a data panel
    # (2026-07-13 figure revision).
    empty_axes = list(axes[len(MATERIALS):])
    legend_ax = None
    if empty_axes:
        legend_ax = empty_axes[len(empty_axes) // 2]
        for extra_ax in empty_axes:
            if extra_ax is legend_ax:
                extra_ax.axis("off")
            else:
                extra_ax.set_visible(False)

    # Under sharex, year labels render only on the grid's bottom row; but the
    # bottom row also holds the legend and hidden cells, so re-enable the x
    # tick labels on the bottom-most occupied panel of every column (same
    # treatment as fig1_capacity_additions_facet.py).
    for col in range(n_cols):
        col_panels = [col + r * n_cols for r in range(n_rows)
                      if col + r * n_cols < len(MATERIALS)]
        if col_panels:
            axes[max(col_panels)].tick_params(axis="x", labelbottom=True)

    # Shared figure-level axis labels (replaces per-panel titles). x is pulled
    # slightly negative so the rotated title clears the first-column tick
    # labels (2026-07-13 figure revision); bbox_inches='tight' at save time
    # grows the canvas to include it.
    fig.supxlabel("", fontsize=1)
    fig.supylabel("Annual material demand (tonnes)", fontsize=16,
                  fontweight="normal", x=-0.004)

    # Legend in the middle empty bottom-row cell, borderless
    # (2026-07-13 figure revision: no frame around the legend).
    handles, labels = axes[0].get_legend_handles_labels()
    env_idx = [i for i, l in enumerate(labels)
               if "across scenarios" in l or "CI" in l]
    sc_idx = [i for i in range(len(labels)) if i not in env_idx]
    ordered_h = [handles[i] for i in sc_idx] + [handles[i] for i in env_idx]
    ordered_l = [labels[i] for i in sc_idx] + [labels[i] for i in env_idx]
    legend_host = legend_ax if legend_ax is not None else fig
    legend_host.legend(ordered_h, ordered_l, loc="center", ncol=1,
                       fontsize=16, frameon=False)

    # No figure title (the caption carries it).
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    # --docx: keep the original large canvas (above) and scale ALL text up by a
    # modest factor so it reads better when the figure is inserted full-page in
    # Word, WITHOUT respacing the panels. Typography only; no data, series,
    # colors, or label content touched. Default (no --docx) output is
    # unaffected by this block. The 2026-08-07 base-size bump (supylabel/legend
    # 13->16, ticks 11->13) carries through here automatically because _bump
    # multiplies whatever base size each element already has.
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

    # A little extra padding around the tight bbox so the bold figure-level
    # y-axis label (anchored near the very left edge) gets some breathing room
    # instead of sitting flush against the canvas edge. docx only, so the
    # default output stays byte-for-byte unchanged.
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                pad_inches=0.5 if docx else 0.1)
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
