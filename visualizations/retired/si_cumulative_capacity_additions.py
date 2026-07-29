"""
SI Fig: Cumulative 2026-2050 capacity additions by technology and scenario
===========================================================================

Supplementary figure complementing Fig 1 (capacity additions):
"In SI, include a bar chart that is cumulative capacity additions from
2026-2050; there should be a bar for each combination of technology and
scenario (including four main scenarios + min + max); this also will help
to explain some of the material dynamics."

Each technology (electricity-supply type) gets six bars:
  - the four highlighted scenarios (Net-Zero by 2035, Net-Zero by 2050,
    Mid Case with IRA, Mid Case no IRA), coloured to match Figs 1 and 2;
  - Minimum and Maximum: the lowest and highest cumulative additions across
    all 61 NREL scenarios for that technology (grey), bracketing the four.
    (This mirrors the grey min/max band in the Fig 1 facet; the Minimum and
    Maximum bars may correspond to different scenarios for different
    technologies.)

Cumulative additions = sum over 2026-2050 of annual GROSS additions
(per-technology positive year-over-year capacity change), the same metric
plotted in the Fig 1 facet. Shared data logic is imported from
fig1_capacity_additions_facet so the two figures cannot drift.

INVENTORY:
  name: si_cumulative_capacity_additions
  output: outputs/figures/manuscript/si_cumulative_capacity_additions.{png,pdf}
  category: Supplementary information
  axes:
    x: cumulative 2026-2050 gross capacity additions (GW)
    y: electricity-supply type (sorted by maximum cumulative additions)
    color: scenario (4 highlighted + Minimum/Maximum across 61 scenarios)
  data_sources:
    - NREL Standard Scenarios 2024 (StdScen24_annual_national.csv), PCHIP-
      interpolated to an annual grid; cumulative gross additions per
      technology x scenario.
  description: >
    Grouped horizontal bar chart of cumulative 2026-2050 gross capacity
    additions, one bar per technology x scenario for the four highlighted
    scenarios plus the per-technology minimum and maximum across all 61
    scenarios. Explains the buildout magnitudes behind the material-demand
    dynamics (solar, wind, and storage dominate new capacity).

Usage:
    python visualizations/si_cumulative_capacity_additions.py
    python visualizations/si_cumulative_capacity_additions.py --docx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "supply_chain"))
sys.path.insert(0, str(BASE_DIR / "visualizations"))

from config import FIGURES_MANUSCRIPT_DIR, FIGURE_DPI
from fig1_capacity_additions_facet import (  # shared logic — keeps figures in sync
    SCENARIOS, SCENARIO_COLORS, load_nrel, compute_gross_additions,
    active_supply_types,
)

MIN_COLOR = "#cfcfcf"   # light grey — per-tech minimum across 61 scenarios
MAX_COLOR = "#6f6f6f"   # dark grey  — per-tech maximum across 61 scenarios


def cumulative_by_scenario(adds):
    """Cumulative 2026-2050 gross additions (GW) per scenario x supply type."""
    return (adds.groupby(["scenario", "supply_type"])["add_gw"].sum()
            .reset_index(name="cum_gw"))


def plot_si(cum, panels, output_path, docx=False):
    # Series order = legend / bar order within each technology group.
    series = list(SCENARIOS.keys()) + ["__min__", "__max__"]
    series_label = {**SCENARIOS, "__min__": "Minimum (across scenarios)",
                    "__max__": "Maximum (across scenarios)"}
    series_color = {**SCENARIO_COLORS, "__min__": MIN_COLOR, "__max__": MAX_COLOR}

    # Biggest builder at the top of the chart.
    panels = list(panels)
    cum_by = {(r.scenario, r.supply_type): r.cum_gw for r in cum.itertuples()}
    by_type_max = (cum[cum.supply_type.isin(panels)]
                   .groupby("supply_type")["cum_gw"].max())
    panels = sorted(panels, key=lambda t: by_type_max.get(t, 0.0))  # ascending -> top is largest in barh

    # Assemble the 6 values per technology.
    def value(stype, s):
        if s == "__min__":
            return cum[cum.supply_type == stype]["cum_gw"].min()
        if s == "__max__":
            return cum[cum.supply_type == stype]["cum_gw"].max()
        return cum_by.get((s, stype), 0.0)

    if docx:
        fig_w, fig_h = 6.5, 0.62 * len(panels) + 1.0
        FS = dict(tick=6.5, ylabel=8.0, xlabel=8.0, legend=6.8)
    else:
        fig_w, fig_h = 11.0, 0.95 * len(panels) + 1.5
        FS = dict(tick=10.0, ylabel=12.0, xlabel=12.0, legend=10.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y = np.arange(len(panels))
    n_series = len(series)
    bar_h = 0.84 / n_series
    for j, s in enumerate(series):
        offset = (j - (n_series - 1) / 2) * bar_h
        vals = [value(t, s) for t in panels]
        ax.barh(y + offset, vals, height=bar_h, color=series_color[s],
                edgecolor="black", linewidth=0.3, label=series_label[s], zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(panels, fontsize=FS["tick"])
    ax.set_xlabel("Cumulative 2026–2050 capacity additions (GW)",
                  fontsize=FS["xlabel"], fontweight="bold")
    ax.tick_params(axis="x", labelsize=FS["tick"])
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.5, len(panels) - 0.5)

    # Legend in reading order (4 scenarios, then Min/Max), top-right.
    ax.legend(loc="lower right", fontsize=FS["legend"], framealpha=0.95,
              frameon=True, ncol=1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name} (+ .pdf)  [{fig_w:.1f}x{fig_h:.1f} in]")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_MANUSCRIPT_DIR)
    parser.add_argument("--no-pchip", action="store_true", default=False,
                        help="Use the raw 3-yr NREL cadence instead of the PCHIP "
                             "annual grid (cumulative totals are nearly identical).")
    parser.add_argument("--docx", action="store_true", default=False,
                        help="Render at manuscript/SI print width (6.5 in) with "
                             "absolute point sizes for legible insertion.")
    args = parser.parse_args()

    nrel = load_nrel()
    print(f"Loaded NREL: {nrel.shape[0]:,} rows ({nrel['scenario'].nunique()} scenarios)")
    if not args.no_pchip:
        from src.interpolation import interpolate_capacity_pchip
        nrel = interpolate_capacity_pchip(nrel)

    adds = compute_gross_additions(nrel)
    panels, _ = active_supply_types(adds)
    cum = cumulative_by_scenario(adds)

    # Report the actual min/max scenario behind each technology's grey bars,
    # so the caption can name them and nothing is silently hidden.
    print("  Cumulative additions (GW) — Maximum / Minimum scenario per technology:")
    for t in sorted(panels, key=lambda x: -cum[cum.supply_type == x]['cum_gw'].max()):
        sub = cum[cum.supply_type == t]
        hi = sub.loc[sub["cum_gw"].idxmax()]
        lo = sub.loc[sub["cum_gw"].idxmin()]
        print(f"    {t:18s} max {hi.cum_gw:7.1f} ({hi.scenario}); "
              f"min {lo.cum_gw:6.1f} ({lo.scenario})")

    out = args.output_dir / "si_cumulative_capacity_additions.png"
    plot_si(cum, panels, out, docx=args.docx)
    print("Done.")


if __name__ == "__main__":
    main()
