"""
Figure 4 - material demand across electricity-supply technologies.

INVENTORY:
  name: figS1_demand_by_technology
  output: outputs/figures/manuscript/figS1_demand_by_technology.{png,pdf}
          outputs/data/demand_by_tech_shares_{cumulative,peak2035}.csv
  category: Manuscript
  x_axis: share of cumulative 2026-2050 demand (%, 100%-stacked)
  y_axis: material (row), sorted by total demand descending
  color: electricity-supply technology (variable renewables -> firm -> fossil)
  data_sources:
    - StdScen24_annual_national.csv (NREL Standard Scenarios 2024, capacity)
    - fitted_distributions.csv (mean intensity, t/MW)
    - material_demand_by_scenario.csv (published MC mean)
    - src/technology_mapping.py (TECHNOLOGY_MAPPING weights)
  description: >
    Manuscript Figure S1. One horizontal 100%-stacked bar per demand-bearing
    material; each segment is a technology's share of Baseline-with-IRA
    (Mid_Case) cumulative demand, from a deterministic mean-intensity
    decomposition rescaled per material to the published Monte Carlo mean
    (so the figure is a decomposition of the MC mean, not itself probabilistic).
    Covers all 27 demand-bearing materials (Glass and Fiberglass included).
    Backs claims F4a-F4c in Scientific_Draft/claims_registry.yaml via the
    demand_by_tech_shares_*.csv traceability exports.
END_INVENTORY

Usage:
    python visualizations/figS1_demand_by_technology.py
    python visualizations/figS1_demand_by_technology.py --output-dir outputs/figures/manuscript
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import demand_by_technology as L

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TEXT_DARK = "#2B2B2B"
TEXT_MUTED = "#595959"

# Rare-earth material labels carry the bold navy used across the figure set
# (Figs 2, 4, and 5), so the REE set reads consistently between figures
# (2026-07-13 figure revision).
REE_SET = {"Neodymium", "Dysprosium", "Praseodymium", "Terbium", "Yttrium"}
REE_LABEL_COLOR = "#0d3b66"


def _kept_techs(scenario: str) -> list[str]:
    """Technologies contributing to any material across BOTH modes, so the
    legend is identical between the cumulative and peak decompositions."""
    max_share = {t: 0.0 for t in L.TECH_ORDER}
    for mode in ("cumulative", "peak2035"):
        df = L.compute_split(scenario, mode, verbose=False)
        materials = L.ordered_materials(df)
        p = L.pivot(df, materials)
        mat_tot = p.sum(axis=1).replace(0, np.nan)
        for t in L.TECH_ORDER:
            share = (p[t] / mat_tot).max() * 100.0
            max_share[t] = max(max_share[t], 0.0 if np.isnan(share) else share)
    return [t for t in L.TECH_ORDER if max_share[t] > 0.0]


def _stack_matrix(scenario: str, mode: str, keep: list[str]):
    df = L.compute_split(scenario, mode, verbose=False)
    materials = L.ordered_materials(df)
    p = L.pivot(df, materials)
    totals = p.sum(axis=1)
    shares = p.div(totals, axis=0).fillna(0.0) * 100.0
    return materials, shares[keep].copy(), totals


def render(mode: str, scenario: str, keep: list[str], out_dir: Path):
    """Sorted-by-total-demand 100%-stacked composition bars (manuscript Fig 4)."""
    materials, mat, totals = _stack_matrix(scenario, mode, keep)
    order = sorted(materials, key=lambda m: totals[m], reverse=True)
    materials = order
    mat = mat.reindex(order)
    cols = list(mat.columns)

    n = len(materials)
    y = np.arange(n)[::-1]
    y_of = dict(zip(materials, y))

    fig, ax = plt.subplots(figsize=(10.6, 9.4))
    left = np.zeros(n)
    for t in cols:
        vals = mat[t].values
        ax.barh(y, vals, left=left, height=0.74, color=L.TECH_COLORS[t],
                edgecolor="white", linewidth=0.6, zorder=3)
        left += vals
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.9, n - 0.1)
    ax.set_yticks(y)
    ax.set_yticklabels(materials, fontsize=8.8, color=TEXT_DARK)
    for lab, m in zip(ax.get_yticklabels(), materials):
        if m in REE_SET:
            lab.set_color(REE_LABEL_COLOR)
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", length=0, labelsize=9, colors=TEXT_MUTED)
    ax.set_xlabel(f"Share of {L.MODE_LABEL[mode]} demand (%)",
                  fontsize=10.5, color=TEXT_DARK, labelpad=6)
    for x in (20, 40, 60, 80):
        ax.axvline(x, color="white", lw=0.6, zorder=2)
    for m in materials:
        ax.text(101.5, y_of[m], L.fmt_tonnes(totals[m]), va="center",
                ha="left", fontsize=8.3, color=TEXT_MUTED,
                family="DejaVu Sans Mono", zorder=4)
    ax.text(101.5, n - 0.35, "Total demand", va="bottom", ha="left",
            fontsize=8.3, style="italic", color=TEXT_MUTED)

    handles = [Patch(facecolor=L.TECH_COLORS[t], edgecolor="white", label=t)
               for t in cols]
    leg = ax.legend(handles=handles, ncol=5, fontsize=8.6,
                    loc="upper center", bbox_to_anchor=(0.5, -0.085),
                    frameon=False, handlelength=1.1, handleheight=1.1,
                    columnspacing=1.3, handletextpad=0.5)
    leg.set_title("Electricity-supply technology  (variable renewables "
                  "→ firm → fossil)", prop={"size": 8.8})
    leg._legend_box.align = "left"

    fig.subplots_adjust(left=0.13, right=0.86, top=0.965, bottom=0.17)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "figS1_demand_by_technology.png"
    pdf = out_dir / "figS1_demand_by_technology.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


def main():
    ap = argparse.ArgumentParser(description="Render manuscript Figure 4 "
                                 "(demand by technology) and export share CSVs.")
    ap.add_argument("--scenario", default=L.REFERENCE_SCENARIO)
    ap.add_argument("--mode", choices=["cumulative", "peak2035"],
                    default="cumulative",
                    help="manuscript Figure 4 is the cumulative decomposition")
    ap.add_argument("--output-dir", default="outputs/figures/manuscript")
    args = ap.parse_args()

    # Traceability CSVs for both modes (back claims F4a-F4c).
    L.export_shares("cumulative")
    L.export_shares("peak2035")

    keep = _kept_techs(args.scenario)
    print(f"kept techs ({len(keep)}): {keep}")
    render(args.mode, args.scenario, keep, Path(args.output_dir))


if __name__ == "__main__":
    main()
