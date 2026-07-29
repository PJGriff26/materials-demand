"""
Fig 1: Annual capacity ADDITIONS by electricity-supply type (facet)
====================================================================

Laid out to parallel Fig 2 (material demand):
same small-multiples layout, same four representative NREL scenarios,
same colour scheme, same grey min/max envelope across the 61 scenarios,
and the same styling conventions locked on Fig 2 (no figure title,
<=6 y-increments with two-digit labels, no y-axis tick marks, larger
text, legend in the empty bottom-right panel cell).

Supersedes the single stacked bar chart (fig1_capacity_additions_ensemble.py)
for the manuscript. Differences:
  - one panel PER technology instead of one stacked bar chart. Panels are
    the demand-modeled NREL technologies (2026-07-23 review: all 18
    individually, alphabetical; was 12 aggregated supply types), MINUS the
    three with zero gross additions in every one of the 61 scenarios
    (Biomass, Coal, Nuclear; omitted per the 2026-07-27 decision and
    disclosed in the caption) = 15 panels. The 5 NREL technologies excluded
    from demand modeling (battery 4-hr, battery 8-hr, o-g-s, DAC,
    electrolyzer) are not shown.
  - the metric is GROSS capacity additions (per-technology positive
    year-over-year change), NOT net change. Retirements are not netted
    out, so the figure shows new-build that drives material demand and is
    consistent with the Methods statement that end-of-life retirements are
    excluded over the horizon.
  - annual cadence via monotone PCHIP (Fritsch & Carlson 1980), matching
    Figs 2/3/5/6.

Manuscript sizing: pass --docx to render at print width (6.5 in) with
absolute point sizes by role, so the text is legible when the figure is
inserted into the Word manuscript at full text width. (Per
reference_docx_figure_sizing: rendering at the target size with absolute
fonts is reliable; post-hoc font-scaling under bbox_inches='tight' is not,
because the canvas grows with the fonts and on-page size stays invariant.)

INVENTORY:
  name: fig1_capacity_additions_facet
  output: outputs/figures/manuscript/fig1_capacity_additions_facet.{png,pdf}
  category: Manuscript main text — Fig 1
  axes:
    x: year (2026-2050, annual cadence after PCHIP interpolation)
    y: annual gross capacity additions (GW/yr; one panel per technology)
    color: 4 representative NREL scenarios (rest shown as scenario-range band)
  data_sources:
    - NREL Standard Scenarios 2024 (StdScen24_annual_national.csv), PCHIP-
      interpolated to an annual grid; gross additions = per-technology
      positive year-over-year capacity change.
  description: >
    Small-multiples line chart, one panel per demand-modeled NREL technology
    (15 panels, alphabetical; the 5 technologies without material-intensity
    data are omitted, as are Biomass/Coal/Nuclear which have zero additions
    in every scenario - disclosed in the caption), of annual gross capacity
    additions under four
    highlighted NREL scenarios (Net-Zero by 2035, Net-Zero by 2050,
    Baseline with IRA, Baseline without IRA) against a grey band spanning
    the scenario ensemble range across all 61 scenarios. Parallels Fig 2 in
    layout, colour, and styling. Supersedes fig1_capacity_additions_ensemble.py.
END_INVENTORY

Usage:
    python visualizations/fig1_capacity_additions_facet.py            # full-size
    python visualizations/fig1_capacity_additions_facet.py --docx     # manuscript size
    python visualizations/fig1_capacity_additions_facet.py --no-pchip # 3-yr cadence
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "supply_chain"))

from config import FIGURES_MANUSCRIPT_DIR, FIGURE_DPI, NREL_SCENARIOS_FILE
from src.scenario_config import SCENARIO_LABELS, SCENARIO_COLORS  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — mirrors Fig 2 (fig2_demand_projections.py) so the two
# figures read as a matched pair. Keep these in sync with that script.
# ═══════════════════════════════════════════════════════════════════════════

# Display order (legend top-to-bottom): most aggressive first. Keys, labels,
# and the cross-figure palette come from src/scenario_config.py (single
# source of truth for the representative-scenario set).
_DISPLAY_ORDER = ["Mid_Case_CO2e_100by2035", "Mid_Case_CO2e_95by2050",
                  "Mid_Case", "Mid_Case_No_IRA"]
SCENARIOS = {k: SCENARIO_LABELS[k] for k in _DISPLAY_ORDER}

# One panel per demand-modeled NREL technology (2026-07-23 review, Fig 1
# comment). Exactly the 18 technologies with a non-empty TECHNOLOGY_MAPPING
# entry — i.e. every technology that contributes to projected material
# demand. The 5 excluded NREL technologies (battery_4, battery_8, o-g-s,
# dac, electrolyzer) have no material-intensity data and are not shown.
# Display names double as the abbreviations defined in SI Table S2 — keep
# the two in sync.
SUPPLY_TYPES = {
    "Biomass":         ["bio_MW"],
    "Biomass CCS":     ["bio-ccs_MW"],
    "Coal":            ["coal_MW"],
    "Coal CCS":        ["coal_ccs_MW"],
    "CSP":             ["csp_MW"],
    "Distributed PV":  ["distpv_MW"],
    "Gas CC":          ["gas_cc_MW"],
    "Gas CC CCS":      ["gas_cc_ccs_MW"],
    "Gas CT":          ["gas_ct_MW"],
    "Geothermal":      ["geo_MW"],
    "H2 CT":           ["h2-ct_MW"],
    "Hydropower":      ["hydro_MW"],
    "Nuclear":         ["nuclear_MW"],
    "Nuclear SMR":     ["nuclear_smr_MW"],
    "Offshore wind":   ["wind_offshore_MW"],
    "Onshore wind":    ["wind_onshore_MW"],
    "Pumped hydro":    ["pumped-hydro_MW"],
    "Utility PV":      ["upv_MW"],
}

# Technologies with ZERO gross additions in every one of the 61 scenarios
# (Biomass, Coal, Nuclear: stocks only decline over 2026-2050) are omitted
# from the figure and disclosed in the caption instead (2026-07-27
# decision). Verified: total positive year-over-year capacity change
# across all scenarios = 0.0 MW for bio_MW, coal_MW, nuclear_MW. All other
# demand-modeled technologies are always shown regardless of magnitude.
ACTIVE_THRESHOLD_GW = 0.0
OMIT_IF_ZERO = True

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def gw_formatter(x, pos):
    """GW tick labels at a consistent 2 significant figures, plain decimal
    (no scientific notation). Every label on every panel carries exactly two
    significant figures, so e.g. 0.8 renders as "0.80" and 120 as "120"."""
    if x == 0:
        return "0"
    decimals = 1 - math.floor(math.log10(abs(x)))  # places for 2 sig figs
    if decimals <= 0:
        return f"{x:.0f}"
    return f"{x:.{decimals}f}"


def load_nrel():
    df = pd.read_csv(NREL_SCENARIOS_FILE, skiprows=3)
    return df.rename(columns={"t": "year"})


def compute_gross_additions(nrel):
    """Annual GROSS capacity additions (GW/yr) per scenario x supply type x year.

    Gross = per-technology positive year-over-year capacity change, summed
    within the supply type. Retirements (negative changes) are floored at
    zero per technology so a within-group retirement never cancels a
    co-located new build. The 2026 baseline has no prior year, so its
    addition is zero by construction (matches Fig 2's zero-at-2026 start).
    """
    recs = []
    for scen, g in nrel.sort_values(["scenario", "year"]).groupby("scenario"):
        g = g.sort_values("year").reset_index(drop=True)
        years = g["year"].values
        for name, cols in SUPPLY_TYPES.items():
            present = [c for c in cols if c in g.columns]
            if not present:
                continue
            add_mw = np.zeros(len(g))
            for c in present:
                add_mw += g[c].diff().fillna(0).clip(lower=0).values
            for i, yr in enumerate(years):
                recs.append({"scenario": scen, "year": int(yr),
                             "supply_type": name, "add_gw": add_mw[i] / 1000.0})
    return pd.DataFrame(recs)


def active_supply_types(adds):
    """Panel technologies in alphabetical order. Technologies whose gross
    additions are zero in EVERY scenario-year (ensemble max = 0) are omitted
    when OMIT_IF_ZERO is set (2026-07-27 decision: remove all-zero
    panels, disclose in the caption). Near-zero-but-nonzero technologies are
    always shown. Returns (kept, dropped)."""
    order, dropped = [], []
    for name in SUPPLY_TYPES:
        sub = adds[adds["supply_type"] == name]
        if sub.empty:
            dropped.append((name, 0.0)); continue
        ens_max = sub["add_gw"].max()
        if OMIT_IF_ZERO and ens_max <= 1e-9:
            dropped.append((name, 0.0)); continue
        peak = sub.groupby("year")["add_gw"].mean().max()
        total = sub.groupby("scenario")["add_gw"].sum().max()
        (order if peak >= ACTIVE_THRESHOLD_GW else dropped).append(
            (name, total if peak >= ACTIVE_THRESHOLD_GW else peak))
    kept = sorted((n for n, _ in order), key=str.lower)
    return kept, dropped


# ═══════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig1(adds, panels, output_path, docx=False):
    n = len(panels)

    # Layout: 5 columns x 3 rows for the 15-panel facet (parallels Fig 2's
    # 5-column grid; no spare cells). For the manuscript (docx) the
    # grid is sized to full text width (~6.5 in) so each panel is ~1.3 in; for
    # on-screen review use the same 5 columns on a larger canvas. The legend
    # sits in a strip below the grid, so no panel cell is wasted and each
    # panel keeps its full size. Absolute point sizes are set for the print
    # dimensions (per reference_docx_figure_sizing).
    if docx:
        n_cols = 5
        FS = dict(title=8.0, tick=6.5, ylabel=10.0, legend=8.0)
        lw = 1.1
        xticks = [2030, 2040, 2050]
        panel_w, panel_h, legend_pad = 1.3, 1.15, 0.6
    else:
        n_cols = 5
        FS = dict(title=14.0, tick=12.0, ylabel=15.0, legend=12.0)
        lw = 1.9
        xticks = [2030, 2040, 2050]
        panel_w, panel_h, legend_pad = 3.2, 2.5, 1.0

    n_rows = math.ceil(n / n_cols)
    fig_w = panel_w * n_cols
    fig_h = panel_h * n_rows + legend_pad

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    band_label = "Scenario ensemble range"
    for ax, stype in zip(axes, panels):
        sub = adds[adds["supply_type"] == stype]
        by_year = sub.groupby("year")["add_gw"]
        years = np.array(sorted(sub["year"].unique()))
        lo = by_year.min().reindex(years).values
        hi = by_year.max().reindex(years).values
        ax.fill_between(years, lo, hi, color="gray", alpha=0.20,
                        linewidth=0, label=band_label, zorder=1)

        for sc, label in SCENARIOS.items():
            s = sub[sub["scenario"] == sc].sort_values("year")
            ax.plot(s["year"].values, s["add_gw"].values,
                    color=SCENARIO_COLORS[sc], lw=lw, label=label, zorder=3)

        ax.set_title(stype, fontsize=FS["title"], fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(2026, 2050)
        ax.set_ylim(bottom=0)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.yaxis.set_major_formatter(FuncFormatter(gw_formatter))
        # <=6 y-increments, two-digit labels,
        # larger tick text, no y-axis tick marks.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=FS["tick"])
        ax.tick_params(axis="y", which="both", length=0)

    # Under sharex, year labels render only on the grid's bottom row — but that
    # row holds the legend and blank cells, so re-enable the x tick labels on
    # the bottom-most occupied panel of every column.
    for col in range(n_cols):
        col_panels = [col + r * n_cols for r in range(n_rows)
                      if col + r * n_cols < n]
        if col_panels:
            axes[max(col_panels)].tick_params(axis="x", labelbottom=True)

    # Shared y-axis label (replaces per-panel titles), parallel to Fig 2.
    # Unbold per the 2026-07-23 review (Fig 1 comment).
    fig.supylabel("Annual capacity additions (GW)", fontsize=FS["ylabel"],
                  fontweight="normal", x=0.005)

    # Legend handles in a fixed order (scenario lines, then the band patch).
    handles = [Line2D([0], [0], color=SCENARIO_COLORS[sc], lw=2.4,
                      label=SCENARIOS[sc]) for sc in SCENARIOS]
    handles.append(Patch(facecolor="gray", alpha=0.20,
                         label="Scenario ensemble range"))
    labels = [h.get_label() for h in handles]

    # Hide any spare cells (when n is not a multiple of n_cols).
    for extra_ax in axes[n:]:
        extra_ax.set_visible(False)

    # no figure title (the caption carries it). The legend sits in
    # a centered strip just below the grid.
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, fontsize=FS["legend"], frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name} (+ .pdf)  [{fig_w:.1f}x{fig_h:.1f} in]")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_MANUSCRIPT_DIR)
    parser.add_argument("--no-pchip", action="store_true", default=False,
                        help="Use the raw 3-yr NREL reporting cadence instead of "
                             "the PCHIP annual grid (default uses PCHIP to match Fig 2).")
    parser.add_argument("--docx", action="store_true", default=False,
                        help="Render at manuscript print width (6.5 in) with absolute "
                             "point sizes so the text is legible when inserted into the "
                             "Word manuscript at full text width.")
    args = parser.parse_args()

    nrel = load_nrel()
    print(f"Loaded NREL: {nrel.shape[0]:,} rows "
          f"({nrel['scenario'].nunique()} scenarios, "
          f"{nrel['year'].min()}-{nrel['year'].max()})")

    if not args.no_pchip:
        from src.interpolation import interpolate_capacity_pchip
        nrel = interpolate_capacity_pchip(nrel)
        print(f"  PCHIP-interpolated to annual grid: "
              f"{nrel['year'].min()}-{nrel['year'].max()} ({nrel['year'].nunique()} years)")

    missing = [s for s in SCENARIOS if s not in nrel["scenario"].unique()]
    if missing:
        sys.exit(f"ERROR: scenarios missing from data: {missing}")

    adds = compute_gross_additions(nrel)
    panels, dropped = active_supply_types(adds)
    print(f"  Panels ({len(panels)}): {', '.join(panels)}")
    if dropped:
        print("  Dropped (negligible gross additions, peak < "
              f"{ACTIVE_THRESHOLD_GW} GW/yr): "
              f"{', '.join(f'{n} ({v:.2g})' for n, v in dropped)}")

    out = args.output_dir / "fig1_capacity_additions_facet.png"
    plot_fig1(adds, panels, out, docx=args.docx)
    print("Done.")


if __name__ == "__main__":
    main()
